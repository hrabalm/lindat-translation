import sentencepiece as spm
from flask import current_app
from websocket import create_connection

import app.models as models
from app.text_utils import split_text_into_sentences
from openai import OpenAI
import sys
import iso639


class OaiLLMModel(models.Model):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.provider=cfg['provider']
        default_prompt = "### Instruction\nTranslate Input from {src} to {tgt}\n### Input\n{text}\n### Response\n"
        self.token=cfg.get('token', None)
        self.prompt=cfg.get('prompt', default_prompt)
        self.allow_custom_prompt=cfg.get('allow_custom_prompt', True)
        self.max_completion_tokens=cfg.get('max_completion_tokens', None)
    @property
    def batch_size(self):
        """
        This method needs a valid app context, current_app is not available at init time.
        """
        if hasattr(self, '_batch_size'):
            return self._batch_size
        else:
            return 1

    def send_sentences_to_backend(self, sentences, src=None, tgt=None, custom_prompt=None, replace_newlines=False):

        self.client = OpenAI(
            base_url=f"{self.server}/v1",
            api_key=self.token,
        )
        print("Connecting to '{}'".format(self.server), flush=True, file=sys.stderr)
        print("Sentences: ", sentences, flush=True, file=sys.stderr)
        res=[]

        if self.provider=="openai":
            model=self.model
        else:
            model=f"{self.provider}/{self.model}"
        src_context = '\n'.join(sentences)
        tgt_context=[]
        for sentence in sentences:
            #prompt = f"Translate the following text from {iso639.to_name(src)} to {iso639.to_name(tgt)}, including correctly transferring the markup from the source sentence into the translation. Do not add any explanations, make sure to only output one single line with the translated sentence and nothing else. Source sentence: " + sentence
            if custom_prompt and self.allow_custom_prompt:
                prompt = custom_prompt.format(src=iso639.to_name(src), tgt=iso639.to_name(tgt), text=sentence, src_context=src_context, tgt_context='\n'.join(tgt_context))
            else:
                prompt = self.prompt.format(src=iso639.to_name(src), tgt=iso639.to_name(tgt), text=sentence, src_context=src_context, tgt_context='\n'.join(tgt_context))

            print("Prompt: ", prompt, flush=True, file=sys.stderr)

            completion = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_completion_tokens=self.max_completion_tokens,
        )
            out=completion.choices[0].message.content
            if replace_newlines:
                out=out.replace('\n', ' ')
            tgt_context.append(out)
            chars=0
            #only keep max the last 4000 characters
            for tgt_context_line in tgt_context:
                chars+=len(tgt_context_line)
            if len(out)>4000:
                tgt_context=[out[-4000:]]
            else:
                while chars+len(tgt_context_line)>4000:
                    tgt_context=tgt_context[1:]
                    chars=len(''.join(tgt_context))
                    print("Removing first line from tgt_context", flush=True, file=sys.stderr)

            res.append(out)

        print("Result: ", '\n'.join(res), flush=True, file=sys.stderr)
        return res

    def split_to_sent_array(self, text, lang):
        sent_array = text.split("\n")
        return sent_array
