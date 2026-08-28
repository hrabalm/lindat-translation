class Translatable:
    def __init__(self):
        self._input_file_name = None
        self._input_word_count = None
        self._output_word_count = None
        self._input_nfc_len = None

    def translate_from_to(self, src, tgt, custom_prompt=None, terms=None, split=True):
        raise NotImplementedError()
    def translate_with_model(self, model, src, tgt, custom_prompt=None, terms=None, split=True):
        raise NotImplementedError()
    def get_text(self):
        raise NotImplementedError()
    def get_translation(self):
        raise NotImplementedError()
    def create_response(self, extra_headers):
        raise NotImplementedError()

    def prep_billing_headers(self):
        return {
            'X-Billing-Filename': self._input_file_name,
            'X-Billing-Input-Word-Count': self._input_word_count,
            'X-Billing-Output-Word-Count': self._output_word_count,
            'X-Billing-Input-NFC-Len': self._input_nfc_len,
        }

    @staticmethod
    def finalize_llm_translation():
        from app.models.llm_request_state import get_request_llm_state

        state = get_request_llm_state()
        if (state is not None and state.total_segments
                and not state.successful_segments):
            raise state.representative_error()
