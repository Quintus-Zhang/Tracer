import numpy as np
import pandas as pd
import time

from num_to_idx import num_to_idx
from exp_val import exp_val

start_time = time.time()
######################### Setup #########################

# utility function declaration
def utility(values, gamma):
    return values**(1-gamma) / (1-gamma)

# set constants
START_AGE = 22
END_AGE = 65
N_W = 801
N_C = 3001
GAMMA = 10
R = 0.02
DELTA = 0.99

# Quadrature
WEIGHT = np.array([[1/6], [2/3], [1/6]])
GRID = np.array([[-1.73205080756887], [0.0], [1.73205080756887]])


######################### Read Data #########################
# labor income process
income = pd.read_excel('labor_income_process.xlsx', sheetname='Income profiles')
income.rename(columns={'e(Labor~)': 'f'}, inplace=True)

# decomposed variance
std = pd.read_excel('labor_income_process.xlsx', sheetname='Variance', skiprows=2)
std.drop(std.columns[0], axis=1, inplace=True)
std.drop([1, 3], inplace=True)
std.index = pd.CategoricalIndex(['sigma_permanent', 'sigma_transitory'])

# conditional survival probabilities
# TODO: survprob
surviv_prob = pd.read_excel('MORTALITY-QUANCHENG.xlsx')
surviv_prob = surviv_prob[surviv_prob['AGE'] >= 22][surviv_prob['AGE'] < 66]
prob = surviv_prob[surviv_prob.columns[1]].values

# TODO  for test, choose sigma of college students
income = income.loc[income['AltDeg'] == 2]
sigma_perm_shock = std.loc['sigma_permanent',  'High School Graduates']
sigma_tran_shock = std.loc['sigma_transitory', 'High School Graduates']

inc_shk_perm = GRID * sigma_perm_shock
inc_shk_tran = GRID * sigma_tran_shock

exp_inc_shk_perm = np.exp(inc_shk_perm)

income_with_tran = np.exp(inc_shk_tran + income['f'].values)  # 3-by-44

# construct grids
grid_w = np.arange(N_W) + 4.0  # TODO:
grid_c = np.arange(N_C) * 0.25

# initialize arrays for value function and consumption
v = np.zeros((2, N_W))
c = np.zeros((2, N_W))

# terminal period
ut = utility(grid_w, GAMMA)
u  = utility(grid_c, GAMMA)
v[0, :] = ut
c[0, :] = grid_w

# collect data
col_names = [str(age) for age in range(END_AGE-START_AGE, -1, -1)]
c_collection = pd.DataFrame(index=pd.Int64Index(range(N_W)), columns=col_names)
v_collection = pd.DataFrame(index=pd.Int64Index(range(N_W)), columns=col_names)

######################### Dynamic Programming #########################
for t in range(END_AGE-START_AGE, -1, -1):                                     # t start from 43
    print('############ Age: ', t, ' ', '#############')
    for i in range(N_W):                                                       # for each possible W_t
        low_c  = c[0, i] - 10.0   # change of consumption ?
        high_c = c[0, i] + 10.0    # TODO: uncollateralized borrowing?

        low_c_idx = num_to_idx(low_c, grid_c)
        high_c_idx = num_to_idx(high_c, grid_c)
        num_c_r = high_c_idx - low_c_idx + 1

        #print('i = ', i)
        grid_c_r = grid_c[low_c_idx:high_c_idx+1]
        savings = grid_w[i] - grid_c_r

        u_r = u[low_c_idx:high_c_idx+1]
        u_r[savings < 0] = -1e+10
        u_r = u_r[None].T

        savings[savings < 0] = 0.0
        savings_incr = savings * (1 + R)
        savings_incr = savings_incr[None].T   # transpose

        expected_value = exp_val(income_with_tran[:, t], exp_inc_shk_perm, savings_incr, grid_w, v[0, :], WEIGHT)

        v_array = u_r + DELTA * prob[t] * expected_value    # v_array has size num_c_r-by-1 / TODO: add survprob[t]
        v[1, i] = np.max(v_array)
        pos = np.argmax(v_array)
        c[1, i] = grid_c[pos+low_c_idx-1]  # !!!

    # dump consumption array and value function array
    c_collection[str(t)] = c[1, :]
    v_collection[str(t)] = v[1, :]

    # change v & c for calculation next stage
    v[0, :] = v[1, :]
    c[0, :] = c[1, :]

c_collection.to_excel('consumption.xlsx')
v_collection.to_excel('v.xlsx')


print("--- %s seconds ---" % (time.time() - start_time))
