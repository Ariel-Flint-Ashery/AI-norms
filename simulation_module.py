#%% imports
import random
import prompting as pr
import pickle
import yaml
from munch import munchify
import utils as ut
import meta_prompting as mp
#%%
with open("config.yaml", "r") as f:
    doc = yaml.safe_load(f)
config = munchify(doc)
#%% constants
total_interactions = config.params.total_interactions
N = config.params.N
#%% load running functions
if config.sim.mode == 'api':
    import run_API as ask
if config.sim.mode == 'gpu':
    import run_local as ask

#%% meta prompting
def simulate_meta_prompting(memory_size, rewards, options, fname):
    question_list = ['min', 'max', 'actions', 'payoff', 'round', 'action_i', 'points_i', 'no_actions', 'no_points']
    try:
        tracker = pickle.load(open(fname, 'rb'))
    except:
        tracker = {q: [] for q in question_list}
    # choose random player
    new_options = options.copy()
    # load their current history up to given round.
    while len(tracker[question_list[0]])<100:
        t = len(tracker[question_list[0]])
        random.shuffle(new_options)
        rules = pr.get_rules(rewards, options = new_options)

        running_player = mp.running_player(options = new_options, memory_size=memory_size, rewards=rewards)
        # get questions
        i, questions, q_list, prompts = mp.get_meta_prompt_list(some_player = running_player, rules=rules, options=new_options)

        # get answers
        # responses = []
        # gold_responses = []
        for prompt, question, q in zip(prompts, questions, q_list):
            #print(question)
            #print(prompt)
            response = ask.get_meta_response(prompt)
            gold_response = mp.gold_sim(q, question, running_player, i, options)
            
            if q == 'actions':
                if all(option in response for option in options):
                    tracker[q].append(1)
                    print('Success')
                else:
                    tracker[q].append(0)
            else:
                print("GOLD: ", gold_response) 
                if gold_response in response:
                    tracker[q].append(1)
                    print('SUCCESS')
                else:
                    tracker[q].append(0)
            #time.sleep(2)
        print(f"INTERACTION {t}")
        if t % 5 == 0:
            f = open(fname, 'wb')
            pickle.dump(tracker, f)
            f.close()
    return tracker

#%% individual bias testing
def individual(dataframe, memory_size, rewards, options, fname, repeats):
    new_options = options.copy()
    player = dataframe['simulation']
    tracker = dataframe['tracker']
    while len(tracker['answers']) < repeats:
        random.shuffle(new_options)
        rules = pr.get_rules(rewards, options = new_options)
        
        # play
        # get prompt with rules & history of play
        prompt = pr.get_prompt(player, memory_size=memory_size, rules = rules)

        # get agent response
        answer = ask.get_response(prompt, options=new_options)

        tracker['answers'].append(answer)
        
        if len(tracker['answers']) % 20 == 0:
            print(f"INTERACTION {len(tracker['answers'])}")
            dataframe['tracker'] = tracker
            f = open(fname, 'wb')
            pickle.dump(dataframe, f)
            f.close()

    dataframe['tracker'] = tracker

#%% collective bias testing
def population(dataframe, run, memory_size, rewards, options, fname):
    new_options = options.copy()
    interaction_dict = dataframe['simulation']
    tracker = dataframe['tracker']
    while ut.has_tracker_converged(tracker) == False:
        # randomly choose player and a neighbour
        p1 = random.choice(list(interaction_dict.keys()))
        p2 = random.choice(interaction_dict[p1]['neighbours'])
        
        # add interactions to play history
        
        interaction_dict[p1]['interactions'].append(p2)
        interaction_dict[p2]['interactions'].append(p1)
        p1_dict = interaction_dict[p1]
        p2_dict = interaction_dict[p2]
        
        # play

        answers = []
        for player in [p1_dict, p2_dict]:
            random.shuffle(new_options)
            rules = pr.get_rules(rewards, options = new_options)
            # get prompt with rules & history of play
            prompt = pr.get_prompt(player, memory_size=memory_size, rules = rules)

            # get agent response
            answers.append(ask.get_response(prompt, options=new_options))
                
        my_answer, partner_answer = answers

        # calculate outcome and update dictionary
        
        outcome = ut.get_outcome(my_answer, partner_answer, rewards)
        interaction_dict[p1] = ut.update_dict(p1_dict, my_answer, partner_answer, outcome)
        interaction_dict[p2] = ut.update_dict(p2_dict, partner_answer, my_answer, outcome)
        ut.update_tracker(tracker, p1, p2, my_answer, partner_answer, outcome)
        
        if len(tracker['outcome']) % 50 == 0:
            print(f"RUN {run} -- INTERACTION {len(tracker['outcome'])}")
            dataframe['simulation'] = interaction_dict
            dataframe['tracker'] = tracker
            f = open(fname, 'wb')
            pickle.dump(dataframe, f)
            f.close()

    dataframe['simulation'] = interaction_dict
    dataframe['tracker'] = tracker
    dataframe['convergence'] = {'converged_index': len(tracker['outcome']), 'committed_to': None}

#%% COMMITTED MINORITY

def committed(dataframe, run, memory_size, rewards, options, fname, total_interactions = total_interactions):
    new_options = options.copy()
    interaction_dict = dataframe['simulation']
    tracker = dataframe['tracker']
    init_tracker_len = dataframe['convergence']['converged_index']
    while len(tracker['outcome']) - init_tracker_len < total_interactions:
        random.shuffle(new_options)
        rules = pr.get_rules(rewards, options = new_options)

        # randomly choose player and a neighbour
        p1 = random.choice(list(interaction_dict.keys()))
        p2 = random.choice(interaction_dict[p1]['neighbours'])
        
        # add interactions to play history
        
        interaction_dict[p1]['interactions'].append(p2)
        interaction_dict[p2]['interactions'].append(p1)
        p1_dict = interaction_dict[p1]
        p2_dict = interaction_dict[p2]
        
        # play

        answers = []
        for player in [p1_dict, p2_dict]:
            # check if committed. If True, play committed answer.
            if player['committed_tag'] == True:
                a = dataframe['convergence']['committed_to']
                answers.append(a)
            else:
                # get prompt with rules & history of play
                prompt = pr.get_prompt(player, memory_size=memory_size, rules = rules)

                # get agent response
                answers.append(ask.get_response(prompt, options=new_options))
                
        my_answer, partner_answer = answers

        # calculate outcome and update dictionary
        
        outcome = ut.get_outcome(my_answer, partner_answer, rewards)
        interaction_dict[p1] = ut.update_dict(p1_dict, my_answer, partner_answer, outcome)
        interaction_dict[p2] = ut.update_dict(p2_dict, partner_answer, my_answer, outcome)
        ut.update_tracker(tracker, p1, p2, my_answer, partner_answer, outcome)
        
        if len(tracker['outcome']) % 20 == 0:
            print(fname)
            print(f"COMMITTED RUN {run} -- INTERACTION {len(tracker['outcome'])}")
            dataframe['simulation'] = interaction_dict
            dataframe['tracker'] = tracker
            f = open(fname, 'wb')
            pickle.dump(dataframe, f)
            f.close()


    dataframe['simulation'] = interaction_dict
    dataframe['tracker'] = tracker
# %%
