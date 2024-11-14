# AI-norms
Spontaenous emergence of conventions in populations of LLMs.

Note 1: the theoretical minimal name game data required to plot figure S3 (theoretical_word_use.pdf) is too large to upload to github. This data can be provided on request, or you can generate it yourself using the "run_NG.py" and "NG_module.py" scripts, playing with the bias parameters to your liking. 

Note 2: The 'api' mode in the config runs the LLM agents in the simulations using the Huggingface serverless API. The script "run_API.py" can modified to support any other API. We run LLMs locally using Huggingface's Transformers library, making use of its quantization features.

Note 3: The prompt structure in 'prompting.py' uses the role assignment tokenization method of Llama 3/3.1 . For Llama 2, perform the following substitutions:
1. "<|begin_of_text|><|start_header_id|>system<|end_header_id|>" replaced by "[INST] <<SYS>>"
2. "<|eot_id|><|start_header_id|>user<|end_header_id|>" replaced by "<<SYS>>"
3. "<|eot_id|><|start_header_id|>assistant<|end_header_id|>" replaced by "[/INST]"
 
Note 4: Several files have been saved under different 'shorthands'.
1. Llama-3.1-70B-Instruct: uses "llama31_70b" for convergence, stability, and CM simulations. Uses "llama31" for individual bias tests and meta testing.
2. Meta-Llama-3-70B-Instruct: uses "llama3_70b" for convergence, stability, and CM simulations. Uses "llama3" for individual bias tests and meta testing.
3. Llama-2-70b-chat-hf: uses "llama2_70b" for convergence simulations. All other tests use "llama2".

Note 5: "run_real_player_metaprompting.py" does a meta test on real player data from a real simulation. "meta_prompting_runner".py uses a new script to randomly generate a player and test it.
