import asyncio
import atexit
import logging
import os
import threading


log = logging.getLogger(__name__)


class AsyncLLMRunner:
    def __init__(self):
        self.pid = os.getpid()
        self.loop = asyncio.new_event_loop()
        self._clients = {}
        self._semaphores = {}
        self._ready = threading.Event()
        self._closing = threading.Event()
        self._state_lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._run,
            name="llm-async-runner",
            daemon=True,
        )
        self._thread.start()
        self._ready.wait()

    def _run(self):
        asyncio.set_event_loop(self.loop)
        self._ready.set()
        self.loop.run_forever()

    def submit(self, coroutine):
        with self._state_lock:
            if self._closing.is_set():
                coroutine.close()
                raise RuntimeError("The asynchronous LLM runner is shutting down")
            future = asyncio.run_coroutine_threadsafe(coroutine, self.loop)
        try:
            return future.result()
        except BaseException:
            future.cancel()
            raise

    async def model_state(self, key, concurrency, client_factory):
        if key not in self._clients:
            self._clients[key] = client_factory()
            self._semaphores[key] = asyncio.Semaphore(concurrency)
        return self._clients[key], self._semaphores[key]

    async def _shutdown(self):
        current = asyncio.current_task()
        tasks = [task for task in asyncio.all_tasks() if task is not current]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await asyncio.gather(
            *(client.aclose() for client in self._clients.values()),
            return_exceptions=True,
        )
        self._clients.clear()
        self._semaphores.clear()

    def close(self):
        with self._state_lock:
            if (os.getpid() != self.pid or self._closing.is_set()
                    or not self._thread.is_alive()):
                return
            self._closing.set()
            future = asyncio.run_coroutine_threadsafe(self._shutdown(), self.loop)
        try:
            future.result(timeout=10)
        except Exception as error:
            future.cancel()
            log.warning("Failed to cleanly stop the asynchronous LLM runner: %s", error)
        finally:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self._thread.join(timeout=10)
            if not self._thread.is_alive():
                self.loop.close()


_runner = None
_runner_lock = threading.Lock()


def get_async_llm_runner():
    global _runner
    pid = os.getpid()
    with _runner_lock:
        if _runner is None or _runner.pid != pid:
            _runner = AsyncLLMRunner()
        return _runner


def close_async_llm_runner():
    global _runner
    with _runner_lock:
        if _runner is not None and _runner.pid == os.getpid():
            _runner.close()
            _runner = None


def _reset_async_llm_runner_after_fork():
    global _runner, _runner_lock
    _runner = None
    _runner_lock = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_async_llm_runner_after_fork)


atexit.register(close_async_llm_runner)
