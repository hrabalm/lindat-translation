#!/usr/bin/env python3
import sys
import argparse
import requests
import uuid
import html
def translate_text(src_lang, tgt_lang, text, model, tags, terms, prompt, base_url):
    """
    Translate text using the specified translation API.

    Parameters:
        src_lang (str): Source language code.
        tgt_lang (str): Target language code.
        text (str): The text to translate.
        model (str): The model identifier.
        tags (bool): Whether to wrap the text with XML tags.
        prompt (str or None): Optional prompt for the translation.
        base_url (str): Base URL for the translation API.

    Returns:
        str: The translated text if successful, or an error message.
    """
    url = f"{base_url}/{model}"

    if tags:
        #text=text.replace('\n','<newLineTag/>')
        files = {
            'input_text': (
                f'line{uuid.uuid4().hex}.inxml',
                f"<mytestdoc>{text}</mytestdoc>",
                'text/xml'
            )
        }
        data = {"src": src_lang, "tgt": tgt_lang}
        if prompt:
            data["prompt"] = prompt
        if terms:
            data["terms"]="["+terms+"]"

        response = requests.post(url, files=files, data=data)
    else:
        data = {"input_text": text, "src": src_lang, "tgt": tgt_lang}
        if prompt:
            data["prompt"] = prompt
        if terms:
            data["terms"]="["+terms+"]"
        print(data)
        response = requests.post(url, data=data)

    response.encoding = "utf-8"
    if response.status_code == 200:
        # Remove XML wrapper if it was added
        return response.text.strip().replace("<mytestdoc>", "").replace("</mytestdoc>", "")#.replace("<newLineTag/>","\n")
    else:
        return f"Error: {response.status_code} - {response.text}"

def main():
    parser = argparse.ArgumentParser(
        description="Translate a block of text using a translation API."
    )
    parser.add_argument("--src", default="en",
                        help="Source language (default: en)")
    parser.add_argument("--tgt", default="fr",
                        help="Target language (default: fr)")
    parser.add_argument("--model", default="TowerInstruct-7B-v0.2",
                        help="Model identifier (default: TowerInstruct-7B-v0.2)")
    parser.add_argument("--tags", action="store_true",
                        help="Wrap the input text with XML tags.")
    parser.add_argument("--prompt", default=None,
                        help="Optional prompt for translation.")
    parser.add_argument("--terms", default=None,
                        help="Optional terms for translation.")

    parser.add_argument("--base-url", default="http://localhost:5001/api/v2/models",
                        help="Base URL for the translation API (default: http://localhost:5001/api/v2/models)")
    parser.add_argument("input_file", nargs="?", type=argparse.FileType("r"),
                        default=sys.stdin,
                        help="Input file to read from (default: standard input)")

    args = parser.parse_args()

    # Read the entire content from the input (file or stdin)
    text = args.input_file.read()

    try:
        translated_text = translate_text(
            src_lang=args.src,
            tgt_lang=args.tgt,
            text=text,
            model=args.model,
            tags=args.tags,
            terms=args.terms,
            prompt=args.prompt,
            base_url=args.base_url
        )
        # Replace HTML entity for apostrophe if present.
        print(html.unescape(translated_text))
    except Exception as e:
        print(f"Exception during translation: {e}", file=sys.stderr)
        print("ERROR", file=sys.stderr)

if __name__ == "__main__":
    main()

