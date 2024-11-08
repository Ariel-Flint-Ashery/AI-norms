#%%
import yaml
from munch import munchify
with open("config.yaml", "r") as f:
    doc = yaml.safe_load(f)
config = munchify(doc)
import huggingface_hub
huggingface_hub.login(config.model.API_TOKEN)
print('Start', flush=True)
import sys
import torch
print(f'torch available: {torch.cuda.is_available()}', flush=True)
print(torch.version.cuda)
for i in range(torch.cuda.device_count()):
   print(torch.cuda.get_device_properties(i))
from torch import cuda, bfloat16
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
import transformers
print(f'torch available: {torch.cuda.is_available()}', flush=True)
import bitsandbytes
import accelerate
print(f'python: {sys.version}', flush=True)
print(f'torch: {torch.__version__}', flush=True)
print(f'transformers: {transformers.__version__}', flush=True)
print(f'bitsandbytes: {bitsandbytes.__version__}', flush=True)
print(f'accelerate: {accelerate.__version__}', flush=True)
model_name = config.model.model_name
print(f'model: {model_name}')

# %%
quantized = config.model.quantized
# loading tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, token=config.model.API_TOKEN)
if not quantized:
    # full model
    print('Loading full model', flush=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, resume_download = True,token=config.model.API_TOKEN)#, local_files_only = True)
    model = model.to('cuda')
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    #pipeline = transformers.pipeline('text-generation', model=model, tokenizer=tokenizer, torch_dtype=torch.float16, device_map="auto")
else:
    # quantized version
    print('Loading quantized model', flush=True)
    bnb_config = transformers.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=bfloat16)
    model = AutoModelForCausalLM.from_pretrained(model_name, resume_download = True, device_map='cuda:0', quantization_config=bnb_config)#, local_files_only = True)
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    #pipeline = transformers.pipeline('text-generation', model=model, tokenizer=tokenizer, device_map="auto")

def query(text, temperature = config.params.temperature, max_new_tokens = 15):
    inputs = tokenizer.encode(text, return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        outputs = model.generate(inputs, max_new_tokens = max_new_tokens, temperature =temperature, return_dict_in_generate=True, output_hidden_states=True)
        generated_tokens = outputs.sequences[0]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return {'generated_text': generated_text, 'generated_tokens': generated_tokens, 'outputs': outputs}

def get_response(chat, options):
    """Generate a response from the model."""

    overloaded = 1
    while overloaded == 1:
        response = query(text = chat)
        #print(response)
                
        if any(option in response['generated_text'].split("'") for option in options):
            overloaded=0
    response_split = response['generated_text'].split("'")
    for opt in options:
        try:
            index = response_split.index(opt)
        except:
            continue
    print(response_split[index])
    return response_split[index]

def get_meta_response(chat):
    """Generate a response from the model."""

    overloaded = 1
    while overloaded == 1:
        response = query(text = chat)

        if 'value' in response['generated_text']:
            overloaded=0
    
            response_split = response['generated_text'].split(";")
            response_split = response_split[0].split(": ")
            if len(response_split)<2:
                overloaded = 1
    print(response_split[1])
    return response_split[1]

