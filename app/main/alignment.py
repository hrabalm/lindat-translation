"""Alignment client configuration and host-wide admission control."""

from contextlib import contextmanager
import fcntl
import hashlib
import math
import os
from pathlib import Path
import time

from flask import current_app
from document_translation.lindat_services.align import LindatAligner
from document_translation.lindat_services.errors import AlignmentRejected, AlignmentTimeout

from app import settings


def alignment_setting(name):
    return current_app.config.get(name, getattr(settings, name))


class AlignmentSlots:
    """Slots shared by all workers on this host using the same directory/URL.

    A slot is held only during an HTTP attempt, not during retry backoff. Kernel
    locks release on process exit, including worker termination.
    """

    def __init__(self, directory, endpoint, limit, wait_timeout):
        if type(limit) is not int or limit < 0:
            raise ValueError('ALIGNER_MAX_IN_FLIGHT must be a nonnegative integer')
        if not math.isfinite(wait_timeout) or wait_timeout <= 0:
            raise ValueError('ALIGNER_QUEUE_TIMEOUT must be finite and positive')
        self.directory = Path(directory) / hashlib.sha256(endpoint.rstrip('/').encode()).hexdigest()
        self.limit = limit
        self.wait_timeout = wait_timeout

    @contextmanager
    def __call__(self, deadline):
        if self.limit == 0:
            yield
            return
        deadline = min(deadline, time.monotonic() + self.wait_timeout)
        acquired = None
        try:
            self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            while acquired is None:
                if time.monotonic() >= deadline:
                    raise AlignmentTimeout('Timed out waiting for an alignment request slot')
                for slot in range(self.limit):
                    fd = os.open(self.directory / f'{slot}.lock', os.O_CREAT | os.O_RDWR, 0o600)
                    try:
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        os.close(fd)
                    except BaseException:
                        os.close(fd)
                        raise
                    else:
                        acquired = fd
                        break
                if acquired is None:
                    time.sleep(max(0, min(0.05, deadline - time.monotonic())))
        except OSError as error:
            raise AlignmentRejected('Cannot access alignment admission-control directory') from error
        try:
            yield
        finally:
            if acquired is not None:
                os.close(acquired)


def create_aligner(src, tgt):
    endpoint = alignment_setting('ALIGNER_BASE_URL')
    return LindatAligner(
        src, tgt, show_progress=False, base_url=endpoint,
        connect_timeout=alignment_setting('ALIGNER_CONNECT_TIMEOUT'),
        read_timeout=alignment_setting('ALIGNER_READ_TIMEOUT'),
        max_attempts=alignment_setting('ALIGNER_MAX_ATTEMPTS'),
        retry_deadline=alignment_setting('ALIGNER_RETRY_DEADLINE'),
        request_slot=AlignmentSlots(
            alignment_setting('ALIGNER_SLOT_DIR'), endpoint,
            alignment_setting('ALIGNER_MAX_IN_FLIGHT'), alignment_setting('ALIGNER_QUEUE_TIMEOUT')),
    )
