#%%
import pickle
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
from scipy.stats import chi2
#%%
np.random.seed(42)
def load_dataframe(fname):
    try:
        mainframe = pickle.load(open(fname, 'rb'))
    except:
        raise ValueError('NO DATAFILE FOUND')
    
    return mainframe

def analyze_dice_rolls(rolls, n):
    # Count the occurrences of each number
    observed = np.bincount(rolls)[1:]  # Remove the 0 count since dice are 1-10
    
    # Expected count for each number (assuming unbiased die)
    expected = len(rolls) / n
    
    # Perform chi-square test
    chi2, p_value = stats.chisquare(observed, f_exp=[expected]*n)
    
    print(f"Observed frequencies: {observed}")
    print(f"Expected frequency: {expected}")
    print(f"Chi-square statistic: {chi2:.4f}")
    print(f"p-value: {p_value:.4f}")
    
    # Interpret the results
    alpha = 0.05  # significance level
    if p_value < alpha:
        print(f"The die is likely biased (p < {alpha}).")
    else:
        print(f"There's not enough evidence to conclude the die is biased (p >= {alpha}).")


def chi_squared_test(observed, expected):
    """
    Perform chi-squared test on categorical data.
    
    Args:
    observed (list or np.array): List of observed frequencies
    expected (list or np.array): List of expected frequencies
    
    Returns:
    tuple: (chi_squared_statistic, p_value)
    """
    observed = np.array(observed)
    expected = np.array(expected)
    
    # Calculate chi-squared statistic
    chi_squared = np.sum((observed - expected)**2 / expected)
    
    # Calculate degrees of freedom
    df = len(observed) - 1
    
    # Calculate p-value
    p_value = 1 - chi2.cdf(chi_squared, df)
    
    return chi_squared, p_value


#%% NEW BOOTSTRAP TEST
# Function to calculate the observed mean based on counts of 0's and 1's
def calculate_observed_mean(count):
    total_count = sum(count)
    return count[1] / total_count

# Bootstrap function to generate bootstrap samples and calculate their means
def bootstrap(count, num_bootstrap_samples=10000):
    total_count = sum(count)
    boot_means = []
    data = np.array([0] * count[0] + [1] * count[1])  # Recreate the data from counts
    for _ in range(num_bootstrap_samples):
        bootstrap_sample = np.random.choice(data, size=int(0.7*total_count), replace=True)
        boot_means.append(np.mean(bootstrap_sample))
    return np.array(boot_means)

# Function to perform a one-tailed hypothesis test
def one_tailed_test(count, null_mean=0.5, num_bootstrap_samples=10000, alpha=0.05):
    observed_mean = calculate_observed_mean(count)
    boot_means = bootstrap(count, num_bootstrap_samples)
    
    # Determine whether it's left-tailed or right-tailed
    if observed_mean < null_mean:
        # Left-tailed test (bias towards 0)
        p_value = np.sum(boot_means <= observed_mean) / len(boot_means)
    else:
        # Right-tailed test (bias towards 1)
        p_value = np.sum(boot_means >= observed_mean) / len(boot_means)
    
    # Print the result
    print(f"Observed Mean: {observed_mean}")
    print(f"One-Tailed P-value: {p_value}")
    if p_value < alpha:
        print("Reject the measured probability. More likely produce less extreme values")
    else:
        print("Fail to reject the measured probability. More likely to produce more extreme values.")
    
    # Plot the bootstrap distribution with observed mean
    plot_bootstrap_distribution(boot_means, observed_mean)
    
    return p_value


def exact_binomial_test(counts, null_mean = 0.5):
    observed_mean = calculate_observed_mean(counts)
    if observed_mean > null_mean:
        test_direction = 'greater'
    else: 
        test_direction = 'less'
    p_value = stats.binomtest(k= counts[1], n = sum(counts), p = 0.5, alternative=test_direction)
    print(f"Observed Mean: {observed_mean}")
    print(f"Bias P-value: {p_value}")

# Function to plot the bootstrap distribution
def plot_bootstrap_distribution(boot_means, observed_mean):
    plt.hist(boot_means, bins=30, density=True, edgecolor='black')
    plt.axvline(observed_mean, color='red', linestyle='--', label='Observed Mean')
    plt.title("Bootstrap Distribution of Means")
    plt.xlabel("Mean")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Step 1: Define the counts of 0's and 1's
    count_0 = 10  # Number of 0's
    count_1 = 90  # Number of 1's (for example, you may have more 1's than 0's)
    count = [count_0, count_1]
    # One-tailed test
    print("One-tailed hypothesis test:")
    one_tailed_test(count, null_mean=0.5, num_bootstrap_samples=10000, alpha=0.05)
    

#%% INDIVIDUAL BIAS TESTS
options = ['Q', 'M']
P1 = [['Q', 'M'], ['M', 'Q'], ['Q', 'M'], ['M', 'Q'], ['Q', 'Q'], ['M', 'M'], ['M', 'M'], ['Q', 'Q']]
P2 = [['M', 'Q'], ['Q', 'M'], ['M', 'M'], ['Q', 'Q'], ['Q', 'M'], ['M', 'Q'], ['M', 'M'], ['Q', 'Q']]
model = "data/llama31_data/llama31"
for my_history, partner_history in zip(P1, P2):
    mainfname = f"{model}_bias_test_{''.join([str(m) for m in my_history])}_{''.join([str(m) for m in partner_history])}.pkl"
    dataframe = load_dataframe(fname = mainfname)
    #new_fname = mainfname+".pkl"
    # print(new_fname)
    # f = open(new_fname, 'wb')
    # pickle.dump(dataframe, f)
    # f.close()
    counts = [dataframe['tracker']['answers'].count(option) for option in options]
    counts = np.array(counts)
    measured_prob = counts[1]/sum(counts)
    print(counts)
    print(f"{''.join([str(m) for m in my_history])}_{''.join([str(m) for m in partner_history])}")
    #data = dataframe['tracker']['answers']

    # test for any bias
    print("probability of more extreme values than measuredif p = 0.5")
    exact_binomial_test(counts, null_mean=0.5)
    # one tailed test
    print("One-tailed hypothesis test:")
    one_tailed_test(count=counts, null_mean=0.5, num_bootstrap_samples=10000, alpha=0.05)

#%%
options = ['Q', 'M']
P1 = [['Q'], ['Q'], ['M'], ['M']]
P2 = [['Q'], ['M'], ['M'], ['Q']]
model = "data/llama31_data/llama31"
for my_history, partner_history in zip(P1, P2):
    mainfname = f"{model}_bias_test_{''.join([str(m) for m in my_history])}_{''.join([str(m) for m in partner_history])}.pkl"
    dataframe = load_dataframe(fname = mainfname)
    counts = [dataframe['tracker']['answers'].count(option) for option in options]
    print(counts)
    print(f"{''.join([str(m) for m in my_history])}_{''.join([str(m) for m in partner_history])}")
    # test for any bias
    exact_binomial_test(counts, null_mean=0.5)
    # one tailed test
    print("One-tailed hypothesis test:")
    one_tailed_test(count=counts, null_mean=0.5, num_bootstrap_samples=10000, alpha=0.05)


# %% NULL PLOTTER
############
#########################
##############################
####################################
options = ['Q', 'M', 'X', 'Y', 'F', 'J', 'P', 'R', 'C', 'D']

fname = f"data/llama31_data/llama31_no_memory_bias_test_{''.join([str(m) for m in options])}_0mem.pkl"
dataframe = load_dataframe(fname = fname)

counts = [dataframe['tracker']['answers'].count(option) for option in options]
answers = sum(counts)
print(answers)
counts = np.array(counts)
expected = [answers/len(counts)]*len(counts)
print(expected)
chi_squared, p_value = chi_squared_test(observed = counts, expected = expected)

print(f"Chi-squared statistic: {chi_squared:.4f}")
print(f"p-value: {p_value:.4f}")

print(counts)
fig, ax = plt.subplots(layout='constrained', figsize = (10,6))
CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
rects = ax.bar(range(len(options)), counts, color = ['tab:red', 'tab:blue'])
ax.bar_label(rects, padding=3, fmt="{:#.4g}", fontsize = 18)
ax.set_xticks(range(len(options)), options, fontsize = 28)
ax.set_ylabel('Production probability', fontsize = 28)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.tick_params(axis='y', which='major', labelsize=16)
ax.tick_params(axis='both', which='minor', labelsize=14) 
plt.tight_layout()
plt.savefig(f"figures/{''.join([str(m) for m in options])}_llama3_initial.png", dpi = 300, bbox_inches = 'tight', pad_inches = 0.15)
plt.show()

#%%
options = ['Q', 'M']
fname = f"data/llama31_data/llama31_no_memory_bias_test_{''.join([str(m) for m in options])}_0mem.pkl"
dataframe = load_dataframe(fname = fname)
counts = [dataframe['tracker']['answers'].count(option) for option in options]
counts = np.array(counts)
measured_prob = counts[1]/sum(counts)
print(counts)
print(f"{''.join([str(m) for m in my_history])}_{''.join([str(m) for m in partner_history])}")
#data = dataframe['tracker']['answers']

# test for any bias
print("probability of more extreme values than measuredif p = 0.5")
exact_binomial_test(counts, null_mean=0.5)
fig, ax = plt.subplots(layout='constrained', figsize = (10,6))
CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
rects = ax.bar(range(len(options)), counts, color = ['tab:red', 'tab:blue'])
ax.bar_label(rects, padding=3, fmt="{:#.4g}", fontsize = 18)
ax.set_xticks(range(len(options)), options, fontsize = 28)
ax.set_ylabel('Production probability', fontsize = 28)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.tick_params(axis='y', which='major', labelsize=16)
ax.tick_params(axis='both', which='minor', labelsize=14) 
plt.tight_layout()
plt.savefig(f"figures/{''.join([str(m) for m in options])}_llama3_initial.png", dpi = 300, bbox_inches = 'tight', pad_inches = 0.15)