import random
import networkx as nx
import pickle
import yaml
from munch import munchify
#%%
with open("config.yaml", "r") as f:
    doc = yaml.safe_load(f)
config = munchify(doc)
#%%#
network_type = config.network.network_type
degree = config.network.degree
alpha = config.network.alpha
beta = config.network.beta
erdos_p = config.network.erdos_p
N = config.params.N
initial = config.params.initial
#%%
  
def get_interaction_network(network_type, minority_size, network_dict = None, degree=degree, alpha = alpha, beta = beta, erdos_p = erdos_p):
  if network_dict == None:
    network_dict = {n+1: {'my_history': [], 'partner_history': [], 'interactions': [], 'score': 0, 'score_history': [], 'outcome': [], 'committed_tag': False} for n in range(N+minority_size)}
  
  # graph structure

  if network_type == 'random_regular':
    graph = nx.random_regular_graph(d=degree, n=len(network_dict.keys()))
    for n in network_dict.keys():
      network_dict[n]['neighbours'] = [i+1 for i in set(graph[n-1])]

  if network_type == 'complete':
    for n in network_dict.keys():
      nodes = list(network_dict.keys())
      nodes.remove(n)
      network_dict[n]['neighbours'] = nodes

  if network_type == 'scale_free':
    graph = nx.scale_free_graph(n=len(network_dict.keys()), alpha=alpha, beta=beta)
    for n in network_dict.keys():
      network_dict[n]['neighbours'] = [i+1 for i in set(graph[n-1])]

  if network_type == 'ER':
    graph = nx.erdos_renyi_graph(n=len(network_dict.keys()), p = erdos_p, directed=False)
    for n in network_dict.keys():
      network_dict[n]['neighbours'] = [i+1 for i in set(graph[n-1])]

  # commitment
  if minority_size > 0:  
    committed_ids = random.sample(list(network_dict.keys()), k = minority_size)
    for id in committed_ids:
      network_dict[id]['committed_tag'] = True
  return network_dict

def load_mainframe(fname):
    try:
        mainframe = pickle.load(open(fname, 'rb'))
    except:
        mainframe = dict()
    
    return mainframe

def get_player():
    return {'my_history': [], 'partner_history': [], 'interactions': [], 'score': 0, 'score_history': [], 'outcome': []}

def get_outcome(my_answer, partner_answer, rewards):
    if my_answer == partner_answer:
        return rewards[1]
    return rewards[0]

def update_dict(player, my_answer, partner_answer, outcome):
  player['score'] += outcome
  player['my_history'].append(my_answer)
  player['partner_history'].append(partner_answer)
  player['score_history'].append(player['score'])
  player['outcome'].append(outcome)

  return player

def get_random_prepared_player(history, rewards):
    #print("CREATING RANDOM PREPARED PLAYER")
    dataframe = get_player()
    for h in history:
        my_answer, partner_answer = h       
        update_dict(dataframe, my_answer, partner_answer, get_outcome(my_answer,partner_answer, rewards))
    # print("---------- CREATING NEW INITIALISED DATAFRAME ----------")
    # print(dataframe['simulation'])
    return dataframe

def has_tracker_converged(tracker, threshold = config.params.convergence_threshold):
    if sum(tracker['outcome'][-config.params.convergence_time:]) < threshold*config.params.convergence_time:
        return False
    return True

def update_tracker(tracker, p1, p2, p1_answer, p2_answer, outcome):
  tracker['players'].append([p1, p2])
  tracker['answers'].append([p1_answer, p2_answer])
  if outcome > 5:
    tracker['outcome'].append(1)
  else:
    tracker['outcome'].append(0)

def get_empty_population(fname):
  try:
    dataframe = pickle.load(open(fname, 'rb'))
  except:
    dataframe = {'simulation': get_interaction_network(network_type = network_type, minority_size=0), 'tracker': {'players': [], 'answers': [], 'outcome': []}}
  print("My history: ", dataframe['simulation'][1]['my_history'])
  return dataframe

def set_initial_state(network_dict, rewards, options, memory_size):
  if initial == 'None':
    pass
  if initial == 'random':
    for m in range(memory_size):
      for p in network_dict.keys():  
        try:
          a = network_dict[p]['committed_tag']
        except:
          a = False

        if a == False:
          my_choice = options[random.choice(range(len([0,1])))]
          partner_choice = options[random.choice(range(len([0,1])))]
          update_dict(network_dict[p], my_choice, partner_choice, get_outcome(my_answer=my_choice, partner_answer=partner_choice, rewards = rewards))


  if type(initial) == int:
    for m in range(memory_size):
      for p in network_dict.keys():  
        try:
          a = network_dict[p]['committed_tag']
        except:
          a = False

        if a == False:         
          update_dict(network_dict[p], options[initial], options[initial], rewards[1])

def test_if_initialisation_worked(dataframe, memory_size, options):
    counter = 0
    for player in dataframe['simulation'].keys():
        my_ans = dataframe['simulation'][player]['my_history'][:memory_size]
        # partner_ans = dataframe['simulation'][player]['partner_history'][:memory_size]
        # score = dataframe['simulation'][player]['outcome'][:memory_size]
        if my_ans.count(options[initial]) == memory_size:
            counter +=1
    #print(counter)
    if counter == N:
        return True
    else:
        raise ValueError("prepared initialisation failed")
    
def get_prepared_population(fname, rewards, options, minority_size, memory_size):
    try:
        return pickle.load(open(fname, 'rb'))
    except:
        dataframe = {'simulation': get_interaction_network(network_type = network_type, minority_size=minority_size), 'tracker': {'players': [], 'answers': [], 'outcome': []}}
    print("---------- CREATING NEW INITIALISED DATAFRAME ----------")
    set_initial_state(dataframe['simulation'], rewards, options, memory_size)
    dataframe['convergence'] = {'converged_index': 0, 'committed_to': options[1]}
    
    #print(dataframe[0]['simulation'][1]['my_history'])
    test_if_initialisation_worked(dataframe, memory_size, options)

    return dataframe

def swap_committed(df, minority_size):
    if minority_size > 0:  
        committed_ids = random.sample(list(df['simulation'].keys()), k = minority_size)
        for id in committed_ids:
            df['simulation'][id]['committed_tag'] = True
    return df
    
def add_committed(df, minority_size):
    new_keys = [n+1 for n in range(N+minority_size)]
    for n in new_keys:
        nodes = [n+1 for n in range(N+minority_size)]
        nodes.remove(n)
        if n>N:
            df['simulation'][n] = {'my_history': [], 'partner_history': [], 'interactions': [], 'score': 0, 'score_history': [], 'outcome': [], 'committed_tag': True, 'neighbours': []}
        df['simulation'][n]['neighbours'] = nodes
    return df