#from app.logging_utils import logged
from app.model_settings import models
from app.text_utils import extract_text as _extract_text
from app.models.llm_request_state import (
    get_request_llm_state,
    llm_state_checkpoint,
)

import logging
log = logging.getLogger(__name__)


def translate_with_model(model, text, src=None, tgt=None, return_source_sentences=False, custom_prompt=None, terms=None, split=True):
    if not text or not text.strip():
        return []
    if return_source_sentences:
        src_sents = model.reconstruct_formatting(*model.extract_sentences(text, src, split=split))
        tgt_sents = model.translate(text, src, tgt, custom_prompt=custom_prompt, terms=terms, split=split)
        return src_sents, tgt_sents
    else:
        return model.translate(text, src, tgt, custom_prompt=custom_prompt, terms=terms, split=split)


def translate_from_to(source, target, text, return_source_sentences=False, custom_prompt=None, terms=None, split=True):
    models_on_path = models.get_model_list(source, target)
    if not models_on_path:
        raise ValueError('No models found for the given pair')

    original_text = text
    first_model = models_on_path[0]['model']
    if return_source_sentences:
        src_sents = first_model.reconstruct_formatting(
            *first_model.extract_sentences(text, source, split=split)
        )

    translation = []
    path_checkpoint = llm_state_checkpoint()
    for obj in models_on_path:
        checkpoint = llm_state_checkpoint()
        translation = translate_with_model(obj['model'], text, obj['src'], obj['tgt'], custom_prompt=custom_prompt, terms=terms, split=split)
        text = _extract_text(translation)
        state = get_request_llm_state()
        if (state is not None and checkpoint is not None
                and state.all_failed_since(checkpoint)):
            failed_records = list(state.records[checkpoint[0]:])
            failed_resplits = state.resplit_count - checkpoint[1]
            state.rollback(path_checkpoint)
            state.add(failed_records, failed_resplits)
            translation = first_model.reconstruct_formatting(
                *first_model.extract_sentences(
                    original_text, source, split=split
                )
            )
            break

    if return_source_sentences:
        return src_sents, translation
    else:
        return translation
