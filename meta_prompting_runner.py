#%% imports
import pickle
import simulation_module as sm
import yaml
from munch import munchify

#%%
with open("config.yaml", "r") as f:
    doc = yaml.safe_load(f)
config = munchify(doc)
#%%
rewards_set = config.params.rewards_set
memory_size_set = config.params.memory_size_set
shorthand = config.model.shorthand
options_set = config.params.options_set
minority_size_set = config.minority.minority_size_set

mp_fname = f"data/{shorthand}_meta_test.pkl"
tracker = sm.simulate_meta_prompting(memory_size=memory_size_set[0], rewards=rewards_set[0], options=options_set[0], fname = mp_fname)
f = open(mp_fname, 'wb')
pickle.dump(tracker, f)
f.close()
# %%
