# prompt_translator
Prompt_translator is an extension for Stable Diffusion Web UI (sd_webui), which adds an automatic translation tool to the Gradio UI. This tool allows users to generate images based on prompts written in 50 different languages.

## Installation
To install prompt_translator, clone the repository or extract the zip file to the extensions folder of the sd_webui mother application.

## Usage
After installing prompt_translator, a new entry will be added to the Gradio UI. To use the automatic translation tool, click the "Load Translation Model" button to load the translation model. The translation model used in this tool is the mbart-large-50-one-to-many-mmt model developed by Meta (formerly Facebook). You can find more information about the model on its Hugging Face model card. Once the model is loaded, a dropdown UI will be displayed, where the user can select the source language of their prompt.

The user can then write their prompt in the desired language and press the "Generate" button to generate the image. The prompt will automatically be translated to English, and the resulting image will look as described in the text.

![image](https://user-images.githubusercontent.com/827993/228090321-2554472d-6fd0-4449-a6d4-190a62ddcce9.png)
![image](https://user-images.githubusercontent.com/827993/228090380-9f2f8928-4698-403e-8ed5-94043ed25480.png)


## License
This project is licensed under the MIT license.

## Contributing
Contributions to prompt_translator are welcome! If you find a bug or have an idea for a new feature, please create an issue on the project's GitHub page. If you'd like to contribute code, please fork the repository, make your changes, and submit a pull request.
