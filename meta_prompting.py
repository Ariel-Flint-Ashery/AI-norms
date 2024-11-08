#%%
import random
import prompting as pr
from utils import get_random_prepared_player
#%%
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
        question = f"Answer saying how many times Player {y} played action {x} overall."
    
    if q == 'no_points':
        question = f"Answer saying what is Player 1's current total payoff."
    
    return question

def get_meta_prompt_list(some_player, rules, options):
    q_list = ['min', 'max', 'actions', 'payoff', 'round', 'action_i', 'points_i', 'no_actions', 'no_points']
    if len(some_player['my_history']) == 0:
        q_list = ['min', 'max', 'actions', 'payoff', 'round']
        i=1
    
    else:
        i = random.choice(range(len(some_player['my_history']))) + 1
    prompts = []
    questions = []
    for q in q_list:
        question = get_question(q, i, options)
        questions.append(question)
        prompts.append(pr.get_meta_prompt(some_player, rules, question = question))

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
            
def running_player(options, memory_size, rewards):
    t = random.choice(list(range(memory_size))) + 1
    history = [random.choices(options, k=2) for step in range(t)]
    player = get_random_prepared_player(history = history, rewards = rewards)
    return player
