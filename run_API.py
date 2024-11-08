#%%
import requests
import time
import yaml
from munch import munchify
#%%
with open("config.yaml", "r") as f:
    doc = yaml.safe_load(f)
config = munchify(doc)
# set temperature to 0 for deterministic outcomes
temperature = config.params.temperature
if temperature == 0:
    llm_params = {"do_sample": False,
            "max_new_tokens": 12,
            "return_full_text": False, 
            }
else:
    llm_params = {"do_sample": True,
            "temperature": temperature,
            "top_k": 10,
            "max_new_tokens": 15,
            "return_full_text": False, 
            }  
#%%
API_TOKEN = config.model.API_TOKEN   
headers = {"Authorization": f"Bearer {API_TOKEN}"}
API_URL = "https://api-inference.huggingface.co/models/"+config.model.model_name
#%%
def query(payload):
    "Query the Hugging Face API"
    try:
        response = requests.post(API_URL, headers=headers, json=payload).json()
    except:
        return None
    return response

def get_response(chat, options):
    """Generate a response from the model."""

    overloaded = 1
    while overloaded == 1:
        response = query({"inputs": chat, "parameters": llm_params, "options": {"use_cache": False}})
        #print(response)
        if response == None:
            print('CAUGHT JSON ERROR')
            continue

        if type(response)==dict:
            print("AN EXCEPTION: ", response)
            time.sleep(2.5)
            if "Inference Endpoints" in response['error']:
              print("HOURLY RATE LIMIT REACHED")
              time.sleep(450)
                
        elif any(option in response[0]['generated_text'].split("'") for option in options):
            overloaded=0
    response_split = response[0]['generated_text'].split("'")
    for opt in options:
        try:
            index = response_split.index(opt)
        except:
            continue
    print(response_split[index])
    return response_split[index]

def get_meta_response(chat):
    """Generate a response from the Llama model."""

    overloaded = 1
    while overloaded == 1:
        response = query({"inputs": chat, "parameters": llm_params, "options": {"use_cache": False}})
        #print(response)
        if response == None:
            print('CAUGHT JSON ERROR')
            continue

        if type(response)==dict:
            print("AN EXCEPTION")
            time.sleep(2.5)
            if "Inference Endpoints" in response['error']:
              print("HOURLY RATE LIMIT REACHED")
              time.sleep(900)
                
        elif 'value' in response[0]['generated_text']:
            overloaded=0
    
            response_split = response[0]['generated_text'].split(";")
            response_split = response_split[0].split(": ")
            if len(response_split)<2:
                overloaded = 1
    print(response_split[1])
    return response_split[1]