# Prompt Translator for Stable Diffusion
[![Follow me on Twitter](https://img.shields.io/twitter/follow/SpaceNerduino?style=social)](https://twitter.com/SpaceNerduino)
[![Follow Me on YouTube](https://img.shields.io/badge/Follow%20Me%20on-YouTube-red?style=flat&logo=youtube)](https://www.youtube.com/user/Parisneo)

Stable diffusion is a text to image generation model. The `prompt_translator` extension allows people to translate their prompt from any language to English before executing the generation, making it accessible to a wider audience.

## Utility

Here is an example of how useful prompt_translator can be:

Let's say that you are a French speaker and want to use Stable Diffusion to generate an image of a beautiful mountain. However, the model requires the prompt to be in English. Without prompt_translator, you would need to either write your prompt in English (if you're comfortable with the language) or use an external translation tool to manually translate your prompt to English.

With prompt_translator, you can simply select "French" as your source language and write your prompt in French. The extension will automatically translate your prompt to English before sending it to the model. This way, you can use Stable Diffusion without having to worry about the language barrier.

Using prompt_translator makes Stable Diffusion more accessible to people who are not comfortable with English or have limited proficiency in the language. It opens up new possibilities for text-to-image generation by allowing users to use their native language to write prompts.

We hope that prompt_translator will make Stable Diffusion more accessible and easier to use for people around the world. Please feel free to contribute to the project or report any issues on our GitHub page.

## Installation

To install `prompt_translator`, clone the repository or extract the zip file to the extensions folder of the sd_webui mother application.

## Usage

After installing `prompt_translator`, a new entry will be added to the Gradio UI. To use the automatic translation tool, click the "Load Translation Model" button to load the translation model. The translation model used in this tool is the `mbart-large-50-many-to-one-mmt` model developed by Meta (formerly Facebook). You can find more information about the model on its [Hugging Face model card](https://huggingface.co/facebook/mbart-large-50-many-to-one-mmt).

Once the model is loaded, a dropdown UI will be displayed, where the user can select the source language of their prompt. The user can then write their prompt in the desired language and press the "Generate" button to generate the image. The prompt will automatically be translated to English, and the resulting image will look as described in the text.

Here are some screenshots of the extension in work:

![image](https://user-images.githubusercontent.com/827993/228090321-2554472d-6fd0-4449-a6d4-190a62ddcce9.png)
![image](https://user-images.githubusercontent.com/827993/228090380-9f2f8928-4698-403e-8ed5-94043ed25480.png)

Using the X/Y/Z script, we can test changing words in another language. Here is French:
![image](https://user-images.githubusercontent.com/827993/229276434-6e024886-13d8-4aa5-b143-6622e544f192.png)

## Supported Languages

The list of supported languages is: Arabic (ar_AR), Czech (cs_CZ), German (de_DE), English (en_XX), Spanish (es_XX), Estonian (et_EE), Finnish (fi_FI), French (fr_XX), Gujarati (gu_IN), Hindi (hi_IN), Italian (it_IT), Japanese (ja_XX), Kazakh (kk_KZ), Korean (ko_KR), Lithuanian (lt_LT), Latvian (lv_LV), Burmese (my_MM), Nepali (ne_NP), Dutch (nl_XX), Romanian (ro_RO), Russian (ru_RU), Sinhala (si_LK), Turkish (tr_TR), Vietnamese (vi_VN), Chinese (zh_CN), Afrikaans (af_ZA), Azerbaijani (az_AZ), Bengali (bn_IN), Persian (fa_IR), Hebrew (he_IL), Croatian (hr_HR), Indonesian (id_ID), Georgian (ka_GE), Khmer (km_KH), Macedonian (mk_MK), Malayalam (ml_IN), Mongolian (mn_MN), Marathi (mr_IN), Polish (pl_PL), Pashto (ps_AF), Portuguese (pt_XX), Swedish (sv_SE), Swahili (sw_KE), Tamil (ta_IN), Telugu (te_IN), Thai (th_TH), Tagalog (tl_XX), Ukrainian (uk_UA), Urdu (ur_PK), Xhosa (xh_ZA), Galician (gl_ES), Slovene (sl_SI).

## License

This project is licensed under the MIT license.

## Contributing
Contributions to prompt_translator are welcome! If you find a bug or have an idea for a new feature, please create an issue on the project's GitHub page. If you'd like to contribute code, please fork the repository, make your changes, and submit a pull request.
