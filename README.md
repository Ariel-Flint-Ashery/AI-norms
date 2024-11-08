# AI-norms
Spontaenous emergence of conventions in populations of LLMs.

Note 1: the theoretical minimal name game data required to plot figure S3 (theoretical_word_use.pdf) is too large to upload to github. This data can be provided on request, or you can generate it yourself using the "run_NG.py" and "NG_module.py" scripts, playing with the bias parameters to your liking. 

Note 2: The 'api' mode in the config runs the LLM agents in the simulations using the Huggingface serverless API. The script "run_API.py" can modified to your liking to support any other API. We run LLMs locally using Huggingface's Transformers library, making use of its quantization features.
