# MBartTranslator :
# Author : ParisNeo
# Description : This script translates Stable diffusion prompt from one of the 50 languages supported by MBART
#    It uses MBartTranslator class that provides a simple interface for translating text using the MBart language model.

import os
import re

import gradio as gr
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration

import modules.scripts as scripts

# The directory to store the models
cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')


class MBartTranslator:
    """MBartTranslator class provides a simple interface for translating text using the MBart language model.

    The class can translate between 50 languages and is based on the "facebook/mbart-large-50-many-to-many-mmt"
    pre-trained MBart model. However, it is possible to use a different MBart model by specifying its name.

    Attributes:
        model (MBartForConditionalGeneration): The MBart language model.
        tokenizer (MBart50TokenizerFast): The MBart tokenizer.
    """

    def __init__(self, model_name="facebook/mbart-large-50-many-to-many-mmt", src_lang=None, tgt_lang=None):

        print("Building translator")
        print("Loading generator (this may take few minutes the first time as I need to download the model)")
        self.model = MBartForConditionalGeneration.from_pretrained(model_name, cache_dir=cache_dir)
        print("Loading tokenizer")
        self.tokenizer = MBart50TokenizerFast.from_pretrained(model_name, src_lang=src_lang, tgt_lang=tgt_lang,
                                                              cache_dir=cache_dir)
        print("Translator is ready")

    def translate(self, text: str, input_language: str, output_language: str) -> str:
        """Translate the given text from the input language to the output language.

        Args:
            text (str): The text to translate.
            input_language (str): The input language code (e.g. "hi_IN" for Hindi).
            output_language (str): The output language code (e.g. "en_US" for English).

        Returns:
            str: The translated text.
        """
        if input_language not in self.tokenizer.lang_code_to_id:
            raise ValueError(
                f"Input language not supported. Supported languages: {self.tokenizer.lang_code_to_id}")
        if output_language not in self.tokenizer.lang_code_to_id:
            raise ValueError(
                f"Output language not supported. Supported languages: {self.tokenizer.lang_code_to_id}")

        self.tokenizer.src_lang = input_language
        encoded_input = self.tokenizer(text, return_tensors="pt")
        generated_tokens = self.model.generate(
            **encoded_input, forced_bos_token_id=self.tokenizer.lang_code_to_id[output_language]
        )
        translated_text = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        return translated_text[0]


# This is a list of LanguageOption objects that represent the various language options available.
# Each LanguageOption object contains a label that represents the display name of the language and 
# a language code that represents the code for the language that will be used by the translation model.
# The language codes follow a format of "xx_YY" where "xx" represents the language code and "YY" represents the 
# country or region code. If the language code is not specific to a country or region, then "XX" is used instead.
# For example, "en_XX" represents English language and "fr_FR" represents French language specific to France.
# These LanguageOption objects will be used to display the language options to the user and to retrieve the 
# corresponding language code when the user selects a language.
language_options = {
    'Afrikaans': 'af_ZA',
    'Arabic': 'ar_AR',
    'Azerbaijani (Azerbaijan)': 'az_AZ',
    'Bengali (India)': 'bn_IN',
    'Burmese (Myanmar)': 'my_MM',
    'Chinese': 'zh_CN',
    'Croatian': 'hr_HR',
    'Czech': 'cs_CZ',
    'Deutsch': 'de_DE',
    "Dutch": "nl_XX",
    'English': 'en_XX',
    'Estonian': 'et_EE',
    'Finnish': 'fi_FI',
    'French': 'fr_XX',
    'Galician (Spain)': 'gl_ES',
    'Georgian': 'ka_GE',
    'Gujarati': 'gu_IN',
    'Hebrew': 'he_IL',
    'Hindi': 'hi_IN',
    'Indonesian': 'id_ID',
    'Italian': 'it_IT',
    'Japanese': 'ja_XX',
    'Kazakh (Kazakhstan)': 'kk_KZ',
    'Khmer': 'km_KH',
    'Korean': 'ko_KR',
    'Latvian': 'lv_LV',
    'Lithuanian': 'lt_LT',
    'Macedonian': 'mk_MK',
    'Malayalam': 'ml_IN',
    'Marathi': 'mr_IN',
    'Mongolian (Mongolia)': 'mn_MN',
    'Nepali': 'ne_NP',
    'Pashto (Afghanistan)': 'ps_AF',
    'Persian': 'fa_IR',
    'Polish (Poland)': 'pl_PL',
    'Portuguese': 'pt_XX',
    'Romanian': 'ro_RO',
    'Russian': 'ru_RU',
    'Sinhala': 'si_LK',
    'Slovenian': 'sl_SI',
    'Spanish': 'es_XX',
    'Swahili (Kenya)': 'sw_KE',
    'Swedish (Sweden)': 'sv_SE',
    'Tamil': 'ta_IN',
    'Telugu': 'te_IN',
    "Tagalog": "tl_XX",
    'Thai': 'th_TH',
    'Turkish': 'tr_TR',
    'Ukrainian': 'uk_UA',
    'Urdu': 'ur_PK',
    'Vietnamese': 'vi_VN',
    'Xhosa (South Africa)': 'xh_ZA'
}


def remove_unnecessary_spaces(text):
    """Removes unnecessary spaces between characters."""
    pattern = r"\)\s*\+\+|\)\+\+\s*"
    replacement = r")++"
    return re.sub(pattern, replacement, text)


def correct_translation_format(original_text, translated_text):
    original_parts = original_text.split('++')
    translated_parts = translated_text.split('++')

    corrected_parts = []
    for i, original_part in enumerate(original_parts):
        translated_part = translated_parts[i]

        original_plus_count = original_part.count('+')
        translated_plus_count = translated_part.count('+')
        plus_difference = translated_plus_count - original_plus_count

        if plus_difference > 0:
            translated_part = translated_part.replace('+' * plus_difference, '', 1)
        elif plus_difference < 0:
            translated_part += '+' * abs(plus_difference)

        corrected_parts.append(translated_part)

    corrected_text = '++'.join(corrected_parts)
    return corrected_text


def extract_plus_positions(text):
    """
    Given a string of text, extracts the positions of all sequences of one or more '+' characters.
    
    Args:
    - text (str): the input text
    
    Returns:
    - positions (list of lists): a list of [start, end, count] for each match, where start is the index of the
      first '+' character, end is the index of the last '+' character + 1, and count is the number of '+' characters
      in the match.
    """
    # Match any sequence of one or more '+' characters
    pattern = re.compile(r'\++')

    # Find all matches of the pattern in the text
    matches = pattern.finditer(text)

    # Loop through the matches and add their positions to the output list
    positions = []
    last_match_end = None
    for match in matches:
        if last_match_end is not None and match.start() != last_match_end:
            # If there is a gap between the current match and the previous one, add a new position
            j = last_match_end - 1
            while text[j] == "+":
                j -= 1
            j += 1
            positions.append([j, last_match_end, last_match_end - j])

        last_match_end = match.end()

    # If the final match extends to the end of the string, add its position to the output list
    if last_match_end is not None and last_match_end == len(text):
        j = last_match_end - 1
        while text[j] == "+":
            j -= 1
        j += 1
        positions.append([j, last_match_end, last_match_end - j])

    return positions


def match_pluses(original_text, translated_text):
    """
    Given two strings of text, replaces sequences of '+' characters in the second string with the corresponding
    sequences of '+' characters in the first string.
    
    Args:
    - original_text (str): the original text
    - translated_text (str): the translated text with '+' characters
    
    Returns:
    - output (str): the translated text with '+' characters replaced by those in the original text
    """
    in_positions = extract_plus_positions(original_text)
    out_positions = extract_plus_positions(translated_text)

    out_vals = []
    out_current_pos = 0

    if len(in_positions) == len(out_positions):
        # Iterate through the positions and replace the sequences of '+' characters in the translated text
        # with those in the original text
        for in_, out_ in zip(in_positions, out_positions):
            out_vals.append(translated_text[out_current_pos:out_[0]])
            out_vals.append(original_text[in_[0]:in_[1]])
            out_current_pos = out_[1]

            # Check that the number of '+' characters in the original and translated sequences is the same
            if in_[2] != out_[2]:
                print("detected different + count")

    # Add any remaining text from the translated string to the output
    out_vals.append(translated_text[out_current_pos:])

    # Join the output values into a single string
    output = "".join(out_vals)
    return output


def post_process_prompt(original, translated):
    """Applies post-processing to the translated prompt such as removing unnecessary spaces and extra plus signs."""
    clean_prompt = remove_unnecessary_spaces(translated)
    clean_prompt = match_pluses(original, clean_prompt)
    # clean_prompt = remove_extra_plus(clean_prompt)
    return clean_prompt


class Script(scripts.Script):
    def __init__(self) -> None:
        """Initializes the Script class and sets the default value for enable_translation attribute."""
        super().__init__()
        self.enable_translation = False
        self.is_negative_translate_active = False

    def title(self):
        """Returns the title of the script."""
        return "Translate prompt to english"

    def show(self, is_img2img):
        """Returns the visibility status of the script in the interface."""
        return scripts.AlwaysVisible

    def set_active(self, active):
        """Sets the is_active attribute and initializes the translator object if not already created. 
        Also, sets the visibility of the language dropdown to True."""
        self.is_active = active
        if not hasattr(self, "translator"):
            self.translator = MBartTranslator()
        if self.is_active:
            return "# I am ready", self.options.update(visible=True)
        else:
            return "After enabling translation, please Wait until I am ready", self.options.update(visible=False)

    def set_negative_translate_active(self, negative_translate_active):
        """Sets the is_active attribute and initializes the translator object if not already created. 
        Also, sets the visibility of the language dropdown to True."""
        self.is_negative_translate_active = negative_translate_active

    def ui(self, is_img2img):
        """Sets up the user interface of the script."""
        self.is_active = False

        with gr.Row():
            with gr.Column(scale=19):
                with gr.Accordion("Prompt Translator", open=False):
                    with gr.Accordion("Help", open=False):
                        gr.Markdown("""
                        # Description
                        This script translates your prompt from another language to english before generating the image allowing you to write prompts in your native language.
                        # How to use
                        Select Enable translation and wait until you the label shows ready.
                        Once the label has Ready on it, select the prompt language, write the prompt in the prompt field then press generate. The script will translate the prompt and generate the text.
                        # Note
                        First time you enable the script, it may take a long time (around a minute), but once loaded, it will be faster.
                        """)
                    with gr.Column():
                        self.enable_translation = gr.Checkbox(label="Enable translation")
                        with gr.Column() as options:
                            self.options = options
                            self.translate_negative_prompt = gr.Checkbox(label="Translate negative prompt")
                            self.enable_translation.value = False
                            self.language = gr.Dropdown(
                                label="Source language",
                                choices=sorted(language_options.keys()),
                                value="Chinese",
                                type="value",
                                elem_id=self.elem_id("x_type")
                            )
                        self.output = gr.Markdown(value="After enabling translation, please Wait until I am ready",
                                                  visible=True)
                        self.enable_translation.change(
                            self.set_active,
                            [self.enable_translation],
                            [self.output, self.options],
                            show_progress=True
                        )
                        self.translate_negative_prompt.change(
                            self.set_negative_translate_active,
                            [self.translate_negative_prompt],
                        )

        self.options.visible = False
        return [self.language]

    def get_prompts(self, p):
        """Returns the original prompts and negative prompts associated with a Prompt object."""
        original_prompts = p.all_prompts if len(p.all_prompts) > 0 else [p.prompt]
        original_negative_prompts = (
            p.all_negative_prompts
            if len(p.all_negative_prompts) > 0
            else [p.negative_prompt]
        )

        return original_prompts, original_negative_prompts

    def process(self, p, language, **kwargs):
        """Translates the prompts from a non-English language to English using the MBartTranslator object."""

        if hasattr(self, "translator") and self.is_active:
            original_prompts, original_negative_prompts = self.get_prompts(p)
            translated_prompts = []
            previous_prompt = ""
            previous_translated_prompt = ""

            for original_prompt in original_prompts:
                if previous_prompt != original_prompt:
                    print(f"Translating prompt to English from {language}")
                    print(f"Initial prompt:{original_prompt}")
                    ln_code = language_options[language]
                    translated_prompt = self.translator.translate(original_prompt, ln_code, "en_XX")
                    translated_prompt = post_process_prompt(original_prompt, translated_prompt)
                    print(f"Translated prompt:{translated_prompt}")
                    translated_prompts.append(translated_prompt)

                    previous_prompt = original_prompt
                    previous_translated_prompt = translated_prompt
                else:
                    translated_prompts.append(previous_translated_prompt)

            if p.negative_prompt != '' and self.is_negative_translate_active:
                previous_negative_prompt = ""
                previous_translated_negative_prompt = ""
                translated_negative_prompts = []
                for negative_prompt in original_negative_prompts:
                    if previous_negative_prompt != negative_prompt:
                        print(f"Translating negative prompt to English from {language}")
                        print(f"Initial negative prompt:{negative_prompt}")
                        ln_code = language_options[language]
                        translated_negative_prompt = self.translator.translate(negative_prompt, ln_code, "en_XX")
                        translated_negative_prompt = post_process_prompt(negative_prompt, translated_negative_prompt)
                        print(f"Translated negative prompt:{translated_negative_prompt}")
                        translated_negative_prompts.append(translated_negative_prompt)

                        previous_negative_prompt = negative_prompt
                        previous_translated_negative_prompt = translated_negative_prompt
                    else:
                        translated_negative_prompts.append(previous_translated_negative_prompt)

                p.negative_prompt = translated_negative_prompts[0]
                p.all_negative_prompts = translated_negative_prompts
            p.prompt = translated_prompts[0]
            p.prompt_for_display = translated_prompts[0]
            p.all_prompts = translated_prompts
