import asyncio
from email.utils import parsedate_to_datetime
import json
import logging
from datetime import datetime, timezone
from dataclasses import dataclass
import re
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
    LLMCompletionTruncated,
)
from app.models.llm_request_state import (
    LLMRequestState,
    LLMSegmentRecord,
    get_request_llm_state,
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
TOKEN_RE = re.compile(
    r"<[^>]+>|[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]|[^\W_]+|[^\w\s]",
    re.UNICODE,
)
SENTENCE_ENDINGS = {'.', '!', '?', '。', '！', '？'}
HTML_VOID_TAGS = {
    'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link',
    'meta', 'param', 'source', 'track', 'wbr',
}


@dataclass
class LLMChunkFormatting:
    prefixes: list
    suffixes: list


@dataclass
class LLMTranslationSegment:
    source: str
    index: int
    label: str
    depth: int = 0


@dataclass
class LLMSegmentResult:
    output: str
    records: list
    resplit_count: int = 0


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
        self.max_completion_tokens = cfg.get('max_completion_tokens', 512)
        self.temperature = cfg.get('temperature', 0.05)
        self.retry_attempts = cfg.get('retry_attempts', 3)
        self.retry_wait_min = cfg.get('retry_wait_min', 1)
        self.retry_wait_max = cfg.get('retry_wait_max', 60)
        self.connect_timeout = cfg.get('connect_timeout', 60)
        self.read_timeout = cfg.get('read_timeout', 900)
        self.write_timeout = cfg.get('write_timeout', 900)
        self.pool_timeout = cfg.get('pool_timeout', 900)
        self.overall_timeout = cfg.get('overall_timeout', 840)
        self.max_input_tokens = cfg.get('max_input_tokens', 256)
        self.max_resplit_depth = cfg.get('max_resplit_depth', 3)
        if self.max_resplit_depth < 0:
            raise ValueError("max_resplit_depth must not be negative")
        if (self.max_completion_tokens is not None
                and self.max_input_tokens >= self.max_completion_tokens):
            raise ValueError(
                "max_input_tokens must be smaller than max_completion_tokens"
            )
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
        segments = [
            LLMTranslationSegment(sentence, index, str(index))
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
            len(segments),
            self.batch_size,
        )
        try:
            result = runner.submit(
                self._translate_segments(
                    runner, segments, server, batch_id, src, tgt,
                    custom_prompt, terms,
                )
            )
        except LLMBackendError as error:
            log.error(
                "LLM batch failed api_request_id=%s batch_id=%s model=%s segments=%s duration_ms=%.1f error=%s detail=%s",
                api_request_id,
                batch_id,
                self.model,
                len(segments),
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
                len(segments),
                (time.monotonic() - started) * 1000,
            )
            raise
        records = [record for segment in result for record in segment.records]
        resplit_count = sum(segment.resplit_count for segment in result)
        state = get_request_llm_state()
        standalone_error = None
        if state is not None:
            state.add(records, resplit_count)
        elif records and not any(record.translated for record in records):
            standalone_state = LLMRequestState(records, resplit_count)
            standalone_error = standalone_state.representative_error()
        fallback_count = sum(not record.translated for record in records)
        log.info(
            "LLM batch completed api_request_id=%s batch_id=%s model=%s segments=%s leaf_segments=%s fallbacks=%s resplits=%s duration_ms=%.1f",
            api_request_id,
            batch_id,
            self.model,
            len(segments),
            len(records),
            fallback_count,
            resplit_count,
            (time.monotonic() - started) * 1000,
        )
        if standalone_error is not None:
            log.error(
                "LLM batch has no successful segments api_request_id=%s batch_id=%s model=%s segments=%s error=%s",
                api_request_id,
                batch_id,
                self.model,
                len(segments),
                type(standalone_error).__name__,
            )
            raise standalone_error
        return [segment.output for segment in result]

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

    async def _translate_segments(self, runner, segments, server, batch_id,
                                  src, tgt, custom_prompt, terms):
        results = [None] * len(segments)
        operation = asyncio.create_task(
            self._translate_segments_without_deadline(
                runner, segments, server, batch_id, src, tgt,
                custom_prompt, terms, results,
            )
        )
        try:
            await asyncio.wait_for(
                asyncio.shield(operation),
                timeout=self.overall_timeout,
            )
        except asyncio.TimeoutError:
            operation.cancel()
            await asyncio.gather(operation, return_exceptions=True)
            error = LLMBackendTimeout(
                "The LLM translation batch exceeded its overall deadline"
            )
            for index, result in enumerate(results):
                if result is None:
                    results[index] = self._fallback_result(
                        segments[index], error, batch_id
                    )
        except asyncio.CancelledError:
            operation.cancel()
            await asyncio.gather(operation, return_exceptions=True)
            raise
        return results

    async def _translate_segments_without_deadline(
            self, runner, segments, server, batch_id, src, tgt,
            custom_prompt, terms, results):
        client, semaphore = await runner.model_state(
            self._runtime_key,
            self.batch_size,
            lambda: self._create_client(server),
        )
        queue = asyncio.Queue()
        for item in enumerate(segments):
            queue.put_nowait(item)

        async def worker():
            while True:
                try:
                    index, segment = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                results[index] = await self._translate_segment(
                    client, semaphore, segment, batch_id, src, tgt,
                    custom_prompt, terms,
                )

        tasks = [asyncio.create_task(worker()) for _ in range(min(
            self.batch_size, len(segments)
        ))]
        try:
            await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    async def _translate_segment(self, client, semaphore, segment, batch_id,
                                 src, tgt, custom_prompt, terms):
        prompt = self._format_prompt(
            segment.source, segment.index, src, tgt, custom_prompt, terms
        )
        try:
            output = await self._translate_prompt(
                client, semaphore, prompt, segment.label, batch_id
            )
            return LLMSegmentResult(output, [LLMSegmentRecord(
                segment=segment.label,
                estimated_tokens=self._estimated_text_tokens(segment.source),
                translated=True,
                depth=segment.depth,
                batch_id=batch_id,
            )])
        except LLMCompletionTruncated as error:
            split = self._resplit_segment(segment)
            if split is None:
                return self._fallback_result(segment, error, batch_id)
            children, formatting = split
            tasks = [
                asyncio.create_task(self._translate_segment(
                    client, semaphore, child, batch_id, src, tgt,
                    custom_prompt, terms,
                ))
                for child in children
            ]
            try:
                child_results = await asyncio.gather(*tasks)
            except BaseException:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
            output = "".join(self.reconstruct_formatting(
                [child.output for child in child_results], formatting
            ))
            return LLMSegmentResult(
                output,
                [
                    record
                    for child in child_results
                    for record in child.records
                ],
                1 + sum(child.resplit_count for child in child_results),
            )
        except LLMBackendError as error:
            return self._fallback_result(segment, error, batch_id)

    def _fallback_result(self, segment, error, batch_id):
        log.warning(
            "LLM segment source fallback batch_id=%s model=%s segment=%s depth=%s error=%s",
            batch_id,
            self.model,
            segment.label,
            segment.depth,
            type(error).__name__,
        )
        return LLMSegmentResult(segment.source, [LLMSegmentRecord(
            segment=segment.label,
            estimated_tokens=self._estimated_text_tokens(segment.source),
            translated=False,
            error=error,
            depth=segment.depth,
            batch_id=batch_id,
        )])

    def _resplit_segment(self, segment):
        if segment.depth >= self.max_resplit_depth:
            return None
        token_count = self._estimated_text_tokens(segment.source)
        if token_count <= 1:
            return None
        spans = self._chunk_spans(
            segment.source, max_input_tokens=max(1, token_count // 2)
        )
        if len(spans) < 2:
            return None
        chunks, formatting = self._chunks_and_formatting(segment.source, spans)
        children = [
            LLMTranslationSegment(
                source=source,
                index=segment.index,
                label=f"{segment.label}.{index}",
                depth=segment.depth + 1,
            )
            for index, source in enumerate(chunks)
        ]
        return children, formatting

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
            choice = data["choices"][0]
            if choice.get("finish_reason") == "length":
                raise LLMCompletionTruncated(
                    f"LLM completion for segment {index} reached its token limit"
                )
            content = choice["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise ValueError("completion content is empty")
        except LLMCompletionTruncated:
            raise
        except (json.JSONDecodeError, UnicodeDecodeError, KeyError, IndexError,
                TypeError, ValueError) as error:
            raise LLMBackendError(
                f"Invalid LLM response for segment {index}"
            ) from error
        return content.replace('\n', ' ')

    def extract_sentences(self, text, text_lang, split=True):
        spans = self._chunk_spans(text) if split else self._single_span(text)
        return self._chunks_and_formatting(text, spans)

    @staticmethod
    def _chunks_and_formatting(text, spans):
        if not spans:
            return [], LLMChunkFormatting([], [])
        chunks = [text[start:end] for start, end in spans]
        prefixes = [''] * len(spans)
        suffixes = [''] * len(spans)
        prefixes[0] = text[:spans[0][0]]
        for index in range(len(spans) - 1):
            suffixes[index] = text[spans[index][1]:spans[index + 1][0]]
        suffixes[-1] = text[spans[-1][1]:]
        return chunks, LLMChunkFormatting(prefixes, suffixes)

    @staticmethod
    def reconstruct_formatting(outputs, formatting):
        return [
            prefix + output + suffix
            for prefix, output, suffix in zip(
                formatting.prefixes, outputs, formatting.suffixes
            )
        ]

    @staticmethod
    def _single_span(text):
        start = len(text) - len(text.lstrip())
        end = len(text.rstrip())
        return [(start, end)] if start < end else []

    def _chunk_spans(self, text, max_input_tokens=None):
        max_input_tokens = max_input_tokens or self.max_input_tokens
        spans = []
        offset = 0
        for line in text.splitlines(keepends=True):
            content = line.rstrip('\r\n')
            spans.extend(
                (offset + start, offset + end)
                for start, end in self._line_chunk_spans(
                    content, max_input_tokens
                )
            )
            offset += len(line)
        if offset < len(text):
            spans.extend(
                (offset + start, offset + end)
                for start, end in self._line_chunk_spans(
                    text[offset:], max_input_tokens
                )
            )
        return spans

    def _line_chunk_spans(self, text, max_input_tokens):
        tokens = []
        depth = 0
        sentence_pending = False
        for match in TOKEN_RE.finditer(text):
            value = match.group(0)
            if value.startswith('</'):
                depth = max(0, depth - 1)
            elif (value.startswith('<') and not value.endswith('/>')
                  and not value.startswith(('<!--', '<!', '<?'))
                  and self._tag_name(value) not in HTML_VOID_TAGS):
                depth += 1
            if value in SENTENCE_ENDINGS:
                sentence_pending = True
            sentence_boundary = sentence_pending and depth == 0
            if sentence_boundary:
                sentence_pending = False
            tokens.append({
                'start': match.start(),
                'end': match.end(),
                'count': self._estimated_tokens(value),
                'safe': depth == 0,
                'sentence': sentence_boundary,
            })

        chunks = []
        start = 0
        while start < len(tokens):
            count = 0
            safe_end = None
            sentence_end = None
            chosen = None
            for index in range(start, len(tokens)):
                next_count = count + tokens[index]['count']
                if next_count > max_input_tokens and safe_end is not None:
                    chosen = sentence_end if (
                        sentence_end is not None
                        and sentence_end - start >= (safe_end - start) // 2
                    ) else safe_end
                    break
                count = next_count
                if tokens[index]['safe']:
                    safe_end = index
                    if tokens[index]['sentence']:
                        sentence_end = index
                if count > max_input_tokens and safe_end == index:
                    chosen = index
                    break
            if chosen is None:
                chosen = len(tokens) - 1
            chunks.append((tokens[start]['start'], tokens[chosen]['end']))
            start = chosen + 1
        return chunks

    @staticmethod
    def _tag_name(value):
        match = re.match(r'<\s*([^\s/>]+)', value)
        return match.group(1).lower() if match else ''

    @staticmethod
    def _estimated_tokens(value):
        if len(value) == 1 and (not value.isalnum() or ord(value) >= 0x3400):
            return 1
        return max(1, (len(value) + 3) // 4)

    def _estimated_text_tokens(self, text):
        return max(1, sum(
            self._estimated_tokens(match.group(0))
            for match in TOKEN_RE.finditer(text)
        ))

    def split_to_sent_array(self, text, lang):
        return [text]
