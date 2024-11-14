#%%
import pickle
import real_player_meta_prompting as mp
#%%
fname = "llama31_70b_converged_baseline_Q_M_-50_100_5mem_complete_24ps_0.5tmp.pkl"
try:
    dataframe = pickle.load(open(fname, 'rb'))
except:
    raise ValueError('NO DATAFILE FOUND')
# %%
rewards = [-50, +100]
options = ['Q', 'M']
fname = "llama31_meta_test_final.pkl"
#%%
tracker = {p+1: {} for p in range(8)}
for key in tracker.keys():
    print(f"STARTING PLAYER {key} META PROMPTING")
    tracker[key]=mp.run(dataframe, tracker, 0, key, 5, rewards, options, fname)
    f = open(fname, 'wb')
    pickle.dump(tracker, f)
    f.close()