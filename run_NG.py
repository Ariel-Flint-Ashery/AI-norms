#%%
import numpy as np
import pickle
import multiprocessing
from tqdm import tqdm
from NG_module import NamingGame
#%% PARAMETERS
bias_list = [0]
cm_list = [0]
runs = 10000
N = 24
interactions = 1000
print(multiprocessing.cpu_count())
#%%
def do_run(params):
    NG = NamingGame(params)
    return NG.simulate()
#%%
for b in tqdm(bias_list):
    for cm in cm_list:
        mainfname = "NG_10.pkl"#f"theoretical_data_files/10_NG_{N}ps_{cm}cm_{b}bias.pkl"
        try:
            mainframe = pickle.load(open(mainfname, 'rb'))
        except:
            
            mainframe = dict()

        params = []
        for run in range(runs):
            if len(mainframe.keys()) > run:
                continue

            params.append({'interactions': interactions,
                            'N': N,
                            'bias': b,
                            'cm': cm,
                            })
            
        # create dataframe
        if __name__ == '__main__':
            # Create a pool of worker processes
            pool = multiprocessing.Pool(5)

            # Map the function to the data using the pool
            dataframes = pool.map(do_run, params)

            # Close the pool and wait for the tasks to finish
            pool.close()
            pool.join()

        for i, r in enumerate(range(len(mainframe.keys()), runs)):
            mainframe[r] = dataframes[i]
        # f = open(mainfname, 'wb')
        # pickle.dump(mainframe, f)
        # f.close()
# %%
params = {'interactions': interactions,
                            'N': N,
                            'bias': 0.5,
                            'cm': 0,
                            }
dataframes = []
for r in tqdm(range(runs)):
    dataframes.append(do_run(params))
#%%
import matplotlib.pyplot as plt
import pandas as pd

theory_dataframe = dataframes#mainframe

average_bins = [theory_dataframe[k]['tracker']['outcome'] for k in range(len(theory_dataframe))]

#%%
#longest_trajectory = max([len(trajectory) for trajectory in average_bins])
#average_bins = [trajectory+[1]*(longest_trajectory-len(trajectory)) for trajectory in average_bins]
theory_avg_outcome = pd.DataFrame(average_bins).mean(axis = 0)#np.mean(average_bins, axis = 0)
theory_steps = np.array(range(len(theory_avg_outcome)))

plt.plot(theory_steps/24, theory_avg_outcome, 'k', label='Model', marker = '.', ls = '')

# %%
f = open("NG_theoretical_10.pkl", 'wb')
pickle.dump(dataframes, f)
f.close()
# %%
