#%%
import random
#%%
def roulette_wheel(normedprobs):
    r=random.random() #generate a random number between 0 and 1
    accumulator = normedprobs[0]
    for i in range(len(normedprobs)):
        if r < accumulator:
            return i
        accumulator = accumulator + normedprobs[i + 1]

class NamingGame():

    binary_choice = list(range(10))#[0,1,2,3,4,5,6,7,8,9]

    def __init__(self, params):
        self.N = params['N']
        self.interactions = params['interactions']
        self.cm = params['cm']
        #self.contagion = params['contagion']
        self.bias = params['bias']
        self.convergence_time = 3*self.N

    def get_interaction_network(self, minority_size):
        network_dict = {n+1: {'inventory': [], 'interactions': [], 'committed_tag': False} for n in range(self.N+minority_size)}
        for n in network_dict.keys():
            nodes = list(network_dict.keys())
            nodes.remove(n)
            network_dict[n]['neighbours'] = nodes

        if minority_size > 0:  
            committed_ids = random.sample(list(network_dict.keys()), k = minority_size)
            for id in committed_ids:
                network_dict[id]['committed_tag'] = True
        
        return network_dict

    def get_empty_dataframe(self):
        dataframe = {'simulation': self.get_interaction_network(minority_size=self.cm),
                      'tracker': {'answers': [], 'outcome': []}}

        return dataframe

    def get_dataframe(self):
        df = self.get_empty_dataframe()

        if self.cm > 0:
            print('committed')
            for p in df['simulation'].keys():
                # we prepare on option 0
                if df['simulation'][p]['committed_tag'] == False:
                    df['simulation'][p]['inventory'] = [self.binary_choice[0]]
                
                # we commit on option 1
                else:
                    df['simulation'][p]['inventory'] = [self.binary_choice[1]]
        return df

    def update_tracker(self, tracker, output, outcome):
        tracker['answers'].append(output)
        tracker['outcome'].append(outcome)

    def get_signal(self, player):
        if len(player['inventory']) == 0:
            return random.choice(self.binary_choice)
        else:
            return random.choice(player['inventory']) #roulette_wheel([1-self.bias, self.bias])

    
    def play(self, speaker, listener, tracker):
        output = self.get_signal(speaker)
        
        # check if listener has output in inventory
        
        if output in listener['inventory']:
            speaker['inventory'] = [output]
            listener['inventory'] = [output]
            self.update_tracker(tracker, output, 1)
        
        # if listener does not have output and is not committed, add new word to inventory
        elif listener['committed_tag'] == False:
            listener['inventory'].append(output)
            self.update_tracker(tracker, output, 0)
        
        # if listener is committed, only record failure
        else:
            self.update_tracker(tracker, output, 0)


    def has_tracker_converged(self, tracker):
        if self.cm == 0:
            if sum(tracker['outcome'][-self.convergence_time:]) < self.convergence_time:
                return False
            return True
        else:
            if len(tracker['outcome']) < self.interactions:
                return False
            return True
    
    def simulate(self):
        #print("SIMULATING CONVERGENCE")
        self.df = self.get_dataframe()
        interaction_dict = self.df['simulation']
        tracker = self.df['tracker']
        #print(self.interactions)
        #while self.has_tracker_converged(tracker) == False:
        while len(tracker['outcome']) < self.interactions:
            p1 = random.choice(list(interaction_dict.keys()))
            p2 = random.choice(interaction_dict[p1]['neighbours'])

            # record interactions   
            interaction_dict[p1]['interactions'].append(p2)
            interaction_dict[p2]['interactions'].append(p1)
            speaker = interaction_dict[p1]
            listener = interaction_dict[p2]

            # play
            self.play(speaker, listener, tracker)
            
            interaction_dict[p1] = speaker
            interaction_dict[p2] = listener

        self.df = {'simulation': interaction_dict,
                     'tracker': tracker,
                     'convergence': {'converged_index': len(tracker['outcome']),
                                      'committed_to': None}
                    }
        return self.df
# %%
