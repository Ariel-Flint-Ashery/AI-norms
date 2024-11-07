#%%
import random
import networkx as nx
import time
import numpy as np
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
# %%
