class LLMBackendError(RuntimeError):
    status_code = 502
    public_message = "The translation backend returned an invalid response."


class LLMBackendUnavailable(LLMBackendError):
    status_code = 503
    public_message = "The translation backend is temporarily unavailable."


class LLMBackendTimeout(LLMBackendError):
    status_code = 504
    public_message = "The translation backend timed out."


class LLMBackendRejected(LLMBackendError):
    status_code = 422
    public_message = "The translation backend rejected the input."


class LLMCompletionTruncated(LLMBackendError):
    pass
