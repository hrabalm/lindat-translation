import asyncio
from email.utils import parsedate_to_datetime
import json
import logging
from datetime import datetime, timezone
import time
import uuid

import httpx
import iso639
from flask import g, has_request_context
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
)
from tenacity.wait import wait_base, wait_random_exponential

import app.models as models
from app.models.async_llm_runner import get_async_llm_runner
from app.models.llm_errors import (
    LLMBackendError,
    LLMBackendRejected,
    LLMBackendTimeout,
    LLMBackendUnavailable,
)


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
RETRYABLE_TRANSPORT_ERRORS = (
    httpx.TimeoutException,
    httpx.NetworkError,
    httpx.ProxyError,
    httpx.RemoteProtocolError,
)


class RetryableHTTPStatus(RuntimeError):
    def __init__(self, response):
        super().__init__(f"LLM backend returned HTTP {response.status_code}")
        self.status_code = response.status_code
        self.retry_after = response.headers.get("Retry-After")


class WaitRetryAfterOrExponential(wait_base):
    def __init__(self, minimum=1, maximum=60):
        self.fallback = wait_random_exponential(min=minimum, max=maximum)
        self.maximum = maximum

    def __call__(self, retry_state):
        exception = retry_state.outcome.exception() if retry_state.outcome else None
        if isinstance(exception, RetryableHTTPStatus) and exception.retry_after:
            delay = self._retry_after_seconds(exception.retry_after)
            if delay is not None:
                return min(delay, self.maximum)
        return self.fallback(retry_state)

    @staticmethod
    def _retry_after_seconds(value):
        try:
            return max(0, float(value))
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(value)
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=timezone.utc)
                return max(0, (retry_at - datetime.now(timezone.utc)).total_seconds())
            except (TypeError, ValueError, OverflowError):
                return None


class OaiLLMModel(models.Model):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.provider = cfg['provider']
        default_prompt = "Translate the following text from {src} to {tgt}, including correctly transferring the markup from the source sentence into the translation. Do not add any explanations, make sure to only output the translated text, including markup like HTML tags, transferred from the source and nothing else. Source text: {sentence} "
        default_prompt_terms = "Translate the following text from {src} to {tgt}, including correctly transferring the markup from the source sentence into the translation. Do not add any explanations, make sure to only output one single line with the translated sentence and nothing else.  Use the following terminology database to translate specific terms: Source term -> Target term \n {terms} \n Source sentence: {sentence} "
        self.token = cfg.get('token')
        self.prompt = cfg.get('prompt', default_prompt)
        self.prompt_terms = cfg.get('prompt_terms', default_prompt_terms)
        self.allow_custom_prompt = cfg.get('allow_custom_prompt', True)
        self.max_completion_tokens = cfg.get('max_completion_tokens')
        self.temperature = cfg.get('temperature', 0.05)
        self.retry_attempts = cfg.get('retry_attempts', 3)
        self.retry_wait_min = cfg.get('retry_wait_min', 1)
        self.retry_wait_max = cfg.get('retry_wait_max', 60)
        self.connect_timeout = cfg.get('connect_timeout', 60)
        self.read_timeout = cfg.get('read_timeout', 900)
        self.write_timeout = cfg.get('write_timeout', 900)
        self.pool_timeout = cfg.get('pool_timeout', 900)
        self.overall_timeout = cfg.get('overall_timeout', 840)
        self._runtime_key = object()

    @property
    def batch_size(self):
        # The LLM backend receives one request per segment, so batch_size is
        # currently the per-worker concurrency limit rather than a true batch.
        return self._batch_size if hasattr(self, '_batch_size') else 256

    def _create_client(self, server):
        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return httpx.AsyncClient(
            base_url=f"{server}/v1",
            headers=headers,
            timeout=httpx.Timeout(
                connect=self.connect_timeout,
                read=self.read_timeout,
                write=self.write_timeout,
                pool=self.pool_timeout,
            ),
            limits=httpx.Limits(
                max_connections=self.batch_size,
                max_keepalive_connections=self.batch_size,
            ),
        )

    def send_sentences_to_backend(self, sentences, src=None, tgt=None,
                                  custom_prompt=None, terms=None):
        batch_id = str(uuid.uuid4())
        api_request_id = (
            getattr(g, 'request_id', 'unknown') if has_request_context()
            else 'none'
        )
        prompts = [
            self._format_prompt(sentence, index, src, tgt, custom_prompt, terms)
            for index, sentence in enumerate(sentences)
        ]
        runner = get_async_llm_runner()
        server = self.server
        started = time.monotonic()
        log.info(
            "LLM batch started api_request_id=%s batch_id=%s model=%s src=%s tgt=%s segments=%s concurrency=%s",
            api_request_id,
            batch_id,
            self.model,
            src,
            tgt,
            len(prompts),
            self.batch_size,
        )
        try:
            result = runner.submit(
                self._translate_prompts(runner, prompts, server, batch_id)
            )
        except LLMBackendError as error:
            log.error(
                "LLM batch failed api_request_id=%s batch_id=%s model=%s segments=%s duration_ms=%.1f error=%s detail=%s",
                api_request_id,
                batch_id,
                self.model,
                len(prompts),
                (time.monotonic() - started) * 1000,
                type(error).__name__,
                error,
            )
            raise
        except BaseException:
            log.exception(
                "LLM batch failed api_request_id=%s batch_id=%s model=%s segments=%s duration_ms=%.1f",
                api_request_id,
                batch_id,
                self.model,
                len(prompts),
                (time.monotonic() - started) * 1000,
            )
            raise
        log.info(
            "LLM batch completed api_request_id=%s batch_id=%s model=%s segments=%s duration_ms=%.1f",
            api_request_id,
            batch_id,
            self.model,
            len(prompts),
            (time.monotonic() - started) * 1000,
        )
        return result

    def _format_prompt(self, sentence, index, src, tgt, custom_prompt, terms):
        sent_terms = None
        if terms:
            try:
                sent_terms = "".join(
                    f"{source} -> {target}\n" for source, target in terms[index]
                )
            except (IndexError, TypeError, ValueError) as error:
                log.warning("Unable to format terminology for segment %s: %s", index, error)

        values = {
            "src": iso639.to_name(src),
            "tgt": iso639.to_name(tgt),
            "sentence": sentence,
            "terms": sent_terms,
        }
        if custom_prompt and self.allow_custom_prompt:
            return custom_prompt.format(**values)
        if sent_terms:
            return self.prompt_terms.format(**values)
        return self.prompt.format(**values)

    async def _translate_prompts(self, runner, prompts, server, batch_id):
        try:
            return await asyncio.wait_for(
                self._translate_prompts_without_deadline(
                    runner, prompts, server, batch_id
                ),
                timeout=self.overall_timeout,
            )
        except asyncio.TimeoutError as error:
            raise LLMBackendTimeout(
                "The LLM translation batch exceeded its overall deadline"
            ) from error

    async def _translate_prompts_without_deadline(self, runner, prompts, server,
                                                   batch_id):
        client, semaphore = await runner.model_state(
            self._runtime_key,
            self.batch_size,
            lambda: self._create_client(server),
        )
        results = [None] * len(prompts)
        queue = asyncio.Queue()
        for item in enumerate(prompts):
            queue.put_nowait(item)

        async def worker():
            while True:
                try:
                    index, prompt = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                results[index] = await self._translate_prompt(
                    client, semaphore, prompt, index, batch_id
                )

        tasks = [asyncio.create_task(worker()) for _ in range(min(
            self.batch_size, len(prompts)
        ))]
        try:
            await asyncio.gather(*tasks)
            return results
        except BaseException:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    async def _translate_prompt(self, client, semaphore, prompt, index,
                                batch_id):
        request_id = str(uuid.uuid4())
        retrying = AsyncRetrying(
            retry=retry_if_exception_type(
                RETRYABLE_TRANSPORT_ERRORS + (RetryableHTTPStatus,)
            ),
            stop=stop_after_attempt(self.retry_attempts),
            wait=WaitRetryAfterOrExponential(
                minimum=self.retry_wait_min,
                maximum=self.retry_wait_max,
            ),
            before_sleep=lambda state: self._log_retry(
                state, batch_id, request_id, index
            ),
            reraise=True,
        )
        try:
            async for attempt in retrying:
                with attempt:
                    async with semaphore:
                        attempt_number = attempt.retry_state.attempt_number
                        started = time.monotonic()
                        log.info(
                            "LLM request started batch_id=%s request_id=%s model=%s segment=%s attempt=%s",
                            batch_id,
                            request_id,
                            self.model,
                            index,
                            attempt_number,
                        )
                        response = await client.post(
                            "/chat/completions",
                            json=self._request_body(prompt),
                            headers={"X-Request-ID": request_id},
                        )
                        log.info(
                            "LLM request completed batch_id=%s request_id=%s model=%s segment=%s attempt=%s status=%s duration_ms=%.1f",
                            batch_id,
                            request_id,
                            self.model,
                            index,
                            attempt_number,
                            response.status_code,
                            (time.monotonic() - started) * 1000,
                        )
                        if response.status_code in RETRYABLE_STATUS_CODES:
                            raise RetryableHTTPStatus(response)
                        self._raise_for_permanent_status(response)
                        return self._completion_text(response, index)
        except httpx.TimeoutException as error:
            raise LLMBackendTimeout(
                f"LLM request for segment {index} timed out"
            ) from error
        except (httpx.NetworkError, httpx.ProxyError, httpx.RemoteProtocolError,
                RetryableHTTPStatus) as error:
            raise LLMBackendUnavailable(
                f"LLM request for segment {index} failed after retries"
            ) from error
        except httpx.DecodingError as error:
            raise LLMBackendError(
                f"LLM backend returned invalid content for segment {index}"
            ) from error

    def _log_retry(self, retry_state, batch_id, request_id, index):
        exception = retry_state.outcome.exception() if retry_state.outcome else None
        delay = retry_state.next_action.sleep if retry_state.next_action else 0
        log.warning(
            "LLM request retry batch_id=%s request_id=%s model=%s segment=%s attempt=%s delay_seconds=%.1f error=%s",
            batch_id,
            request_id,
            self.model,
            index,
            retry_state.attempt_number,
            delay,
            type(exception).__name__ if exception else "unknown",
        )

    def _request_body(self, prompt):
        body = {
            "model": self.model if self.provider == "openai"
            else f"{self.provider}/{self.model}",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "stop": ["###"],
        }
        if self.max_completion_tokens is not None:
            body["max_tokens"] = self.max_completion_tokens
        return body

    @staticmethod
    def _raise_for_permanent_status(response):
        if response.is_success:
            return
        detail = f"LLM backend returned HTTP {response.status_code}"
        if response.status_code in {400, 413, 422}:
            raise LLMBackendRejected(detail)
        raise LLMBackendError(detail)

    @staticmethod
    def _completion_text(response, index):
        try:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise ValueError("completion content is empty")
        except (json.JSONDecodeError, UnicodeDecodeError, KeyError, IndexError,
                TypeError, ValueError) as error:
            raise LLMBackendError(
                f"Invalid LLM response for segment {index}"
            ) from error
        return content.replace('\n', ' ')

    def split_to_sent_array(self, text, lang):
        return text.split("\n")
