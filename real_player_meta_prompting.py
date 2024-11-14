#%%
import pickle
import numpy as np
import requests
from transformers import AutoTokenizer
import random
import time
import yaml
from munch import munchify
#%%
with open("config.yaml", "r") as f:
    doc = yaml.safe_load(f)
config = munchify(doc)
#%%
temperature = config.params.temperature
#%%
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
#%% LOAD LANGUAGE MODEL
#tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-70B-Instruct")
API_TOKEN = '<YOUR_TOKEN_HERE>'   
headers = {"Authorization": f"Bearer {API_TOKEN}"}
API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3.1-70B-Instruct"
#%% DEFINE SIMULATION

def query(payload):
    "Query the Hugginface API"
    try:
        response = requests.post(API_URL, headers=headers, json=payload).json()
    except:
        return None
    return response

def get_llama_response(chat):
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
    # for opt in options:
    print(response_split[1])
    return response_split[1]

def get_rules(rewards, options):
  incorrect, correct = rewards
  rule_set = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
  Context: Player 1 is playing a multi-round partnership game with Player 2 for 100 rounds.
  At each round, Player 1 and Player 2 simultaneously pick an action from the following values: {options}.
  The payoff that both players get is determined by the following rule:
  1. If Players play the SAME action as each other, they will both be REWARDED with payoff +{correct} points.
  2. If Players play DIFFERENT actions to each other, they will both be PUNISHED with payoff {incorrect} points.
  The objective of each Player is to maximize their own accumulated point tally, conditional on the behavior of the other player.
  """ 
  return rule_set

def get_outcome(my_answer, partner_answer, rewards, options):
    if my_answer == partner_answer:
        #if my_answer == options[1]:
        #    return 2*rewards[1]
        return rewards[1]
    #if my_answer != partner_answer:
    return rewards[0]

def get_question(q, i, options):
    if q == 'min':
        question = "Answer saying what is the lowest payoff Player 1 can get in a single round."

    if q == 'max':
        question = "Answer saying what is the highest payoff Player 1 can get in a single round."
    if q == 'actions':
        question = "Answer saying all the action values Player 1 can pick."

    if q == 'payoff':
        x, y = random.choices(options, k=2)
        question = f"Answer saying what is Player 1's payoff in a single round if Player 1 plays {x} and Player 2 plays {y}."
   
    if q == 'round':
        question = "Answer saying what is the current round of the game."
   
    
    if q == 'action_i':
        x = random.choice([1, 2])
        question = f"Answer saying which action Player {x} played in round {i}."

    if q == 'points_i':
        question = f"Answer saying how many points Player 1 collected in round {i}."

    if q == 'no_actions':
        x = random.choice(options)
        y = random.choice([1, 2])
        question = f"Answer saying how many times Player {y} played action {x} overall." #maybe use chose instead of played?
    
    if q == 'no_points':
        question = f"Answer saying what is Player 1's current total payoff."
    
    return question


def get_prompt(player, rules, question):
    # add initial round
    #current_score = 0 #local score tracking --ignores global scoring.
    new_query = f"It is now round 1." + " The current score of Player 1 is 0. You are an observer who answers questions about the game using a single value. Please think step by step before making a decision. Remember, examining history explicitly is important. You write your response using the following format: {'value': <YOUR_ANSWER>; 'reason': <YOUR_REASON>}. <|eot_id|><|start_header_id|>user<|end_header_id|>" + f" {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    l = len(player['my_history'])
    if l == 0:
        return """\n """.join([rules, new_query])
    
    current_score = 0
    history_intro = "This is the history of choices in past rounds:"
    histories = []
    for idx in range(l):
        my_answer = player['my_history'][idx] 
        partner_answer = player['partner_history'][idx] 
        outcome = player['outcome'][idx] #get_outcome(my_answer, partner_answer)
        current_score+=outcome
        histories.append({'round':idx+1, 'Player 1':my_answer, 'Player 2':partner_answer, 'payoff':outcome})
  
    new_query = f"It is now round {idx+2}. The current score of Player 1 is {current_score}." + " You are an observer who answers questions about the game using a single value. Please think step by step before making a decision. Remember, examining history explicitly is important. You write your response using the following format: {'value': <YOUR_ANSWER>; 'reason': <YOUR_REASON>}. <|eot_id|><|start_header_id|>user<|end_header_id|>" + f" {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    histories = "\n ".join([f"{hist}" for hist in histories])
    prompt = """\n """.join([rules, history_intro, histories, new_query])
    return prompt

def get_meta_prompts(some_player, memory_size, rules, options):
    q_list = ['min', 'max', 'actions', 'payoff', 'round', 'action_i', 'points_i', 'no_actions', 'no_points']
    if len(some_player['my_history']) == 0:
        q_list = ['min', 'max', 'actions', 'payoff', 'round']
        i=1
    
    else:
        i = random.choice(range(len(some_player['my_history']))) + 1
    #q_list = ['min', 'max']
    #print(q_list)
    prompts = []
    questions = []
    for q in q_list:
        question = get_question(q, i, options)
        questions.append(question)
        prompts.append(get_prompt(some_player, rules, question = question))

    return i, questions, q_list, prompts

def gold_sim(q, question, running_player, i, options):
    if q == 'min':
        return '-50'
    
    if q == 'max':
        return '100'
    
    if q == 'actions':
        return f"{options} or {list(reversed(options))}"

    if q == 'payoff':
        for x in options:
            if question == f"Answer saying what is Player 1's payoff in a single round if Player 1 plays {x} and Player 2 plays {x}.":
                return '100'
        return '-50'
    
    if q == 'round':
        return f"{len(running_player['my_history'])+1}"

    if q == 'action_i':
        if question == f"Answer saying which action Player 1 played in round {i}.":
            return f"'{running_player['my_history'][i-1]}'"
        if question == f"Answer saying which action Player 2 played in round {i}.":
            return f"'{running_player['partner_history'][i-1]}'"
    
    if q == 'points_i':
        return str(running_player['outcome'][i-1])
    
    if q == 'no_points':
        return str(sum(running_player['outcome']))
    
    if q == 'no_actions':
        for x in options:
            if question == f"Answer saying how many times Player 1 played action {x} overall.":
                return str(running_player['my_history'].count(x))
            
            if question == f"Answer saying how many times Player 2 played action {x} overall.":
                return str(running_player['partner_history'].count(x))

def run(dataframe, tracker, run, p, memory_size, rewards, options, fname):
    rules = get_rules(rewards, options = options)
    dataframe['rules'] = rules
    # choose random player
    player = dataframe[run]['simulation'][p]
    running_player = {'my_history': [], 'partner_history': [], 'outcome': []}
    question_list = ['min', 'max', 'actions', 'payoff', 'round', 'action_i', 'points_i', 'no_actions', 'no_points']
    temp_tracker = {q: [] for q in question_list}
    new_options = options.copy()
    # load their current history up to given round.
    for t in range(len(player['my_history'])):
        random.shuffle(new_options)
        rules = get_rules(rewards, options = new_options)

        if t < memory_size:
            running_player['my_history'] = player['my_history'][:t]
            running_player['partner_history'] = player['partner_history'][:t]
            running_player['outcome'] = player['outcome'][:t]
        else:
            running_player['my_history'] = player['my_history'][t-memory_size:t]
            running_player['partner_history'] = player['partner_history'][t-memory_size:t]
            running_player['outcome'] = player['outcome'][t-memory_size:t]
        
        # get questions
        i, questions, q_list, prompts = get_meta_prompts(running_player, memory_size, rules, options)

        # get answers
        responses = []
        gold_responses = []
        for prompt, question, q in zip(prompts, questions, q_list):
            #print(question)
            #print(prompt)
            response = get_llama_response(prompt)
            gold_response = gold_sim(q, question, running_player, i, options)
            
            if q == 'actions':
                if all(option in response for option in options):
                    temp_tracker[q].append(1)
                    print('Success')
                else:
                    temp_tracker[q].append(0)
            else:
                print("GOLD: ", gold_response) 
                if gold_response in response:
                    temp_tracker[q].append(1)
                    print('SUCCESS')
                else:
                    temp_tracker[q].append(0)
            #time.sleep(2)
        print(f"PLAYER {p} -- INTERACTION {t}")
        if t % 5 == 0:
            tracker[p] = temp_tracker
            f = open(fname, 'wb')
            pickle.dump(tracker, f)
            f.close()
    return temp_tracker
# %%
