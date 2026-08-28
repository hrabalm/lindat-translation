from dataclasses import dataclass, field

from flask import g, has_request_context

from app.models.llm_errors import (
    LLMBackendError,
    LLMBackendRejected,
    LLMBackendTimeout,
    LLMBackendUnavailable,
)


@dataclass
class LLMSegmentRecord:
    segment: str
    estimated_tokens: int
    translated: bool
    error: LLMBackendError = None
    depth: int = 0
    batch_id: str = None

    def diagnostic(self):
        result = {
            "segment": self.segment,
            "estimated_tokens": self.estimated_tokens,
            "strategy": "translated" if self.translated else "original_source",
            "resplit_depth": self.depth,
        }
        if self.error is not None:
            result.update({
                "error": type(self.error).__name__,
                "status": self.error.status_code,
            })
        return result


@dataclass
class LLMRequestState:
    records: list = field(default_factory=list)
    resplit_count: int = 0

    def checkpoint(self):
        return len(self.records), self.resplit_count

    def rollback(self, checkpoint):
        record_count, self.resplit_count = checkpoint
        del self.records[record_count:]

    def add(self, records, resplit_count=0):
        self.records.extend(records)
        self.resplit_count += resplit_count

    @property
    def total_segments(self):
        return len(self.records)

    @property
    def successful_segments(self):
        return sum(record.translated for record in self.records)

    @property
    def fallback_segments(self):
        return self.total_segments - self.successful_segments

    @property
    def total_tokens(self):
        return sum(record.estimated_tokens for record in self.records)

    @property
    def fallback_tokens(self):
        return sum(
            record.estimated_tokens for record in self.records
            if not record.translated
        )

    @property
    def fallback_token_ratio(self):
        return self.fallback_tokens / self.total_tokens if self.total_tokens else 0

    @property
    def partial(self):
        return self.successful_segments > 0 and self.fallback_segments > 0

    def all_failed_since(self, checkpoint):
        records = self.records[checkpoint[0]:]
        return bool(records) and not any(record.translated for record in records)

    def representative_error(self):
        errors = [record.error for record in self.records if record.error]
        if not errors:
            return LLMBackendError("Every LLM segment failed")
        rank = {
            LLMBackendRejected: 1,
            LLMBackendError: 2,
            LLMBackendUnavailable: 3,
            LLMBackendTimeout: 4,
        }
        return max(errors, key=lambda error: rank.get(type(error), 2))

    def fallback_diagnostics(self):
        return [
            record.diagnostic() for record in self.records
            if not record.translated
        ]


def initialize_request_llm_state():
    if has_request_context():
        g.llm_request_state = LLMRequestState()


def get_request_llm_state():
    if not has_request_context():
        return None
    state = getattr(g, "llm_request_state", None)
    if state is None:
        state = LLMRequestState()
        g.llm_request_state = state
    return state


def llm_state_checkpoint():
    state = get_request_llm_state()
    return state.checkpoint() if state is not None else None


def rollback_llm_state(checkpoint):
    state = get_request_llm_state()
    if state is not None and checkpoint is not None:
        state.rollback(checkpoint)
