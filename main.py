import os
import time
import numpy as np
import pandas as pd
from functions import read_input_data, cal_income, adj_income_process
from dp import dp_solver
from cal_ce import cal_certainty_equi, generate_consumption_process
from constants import *
import matplotlib.pyplot as plt
import cProfile

start_time = time.time()

# TODO: test if std of simulated utility decrease as the number of simulations increase
###########################################################################
#                      Setup - file path & raw data                       #
###########################################################################
# set file path
income_fn = 'age_coefficients_and_var.xlsx'
surviv_fn = 'Conditional Survival Prob Feb 16.xlsx'
base_path = os.path.dirname(__file__)
income_fp = os.path.join(base_path, 'data', income_fn)
mortal_fp = os.path.join(base_path, 'data', surviv_fn)

# read raw data
age_coeff, std, surv_prob = read_input_data(income_fp, mortal_fp)


###########################################################################
#              Setup - income process & std & survival prob               #
###########################################################################
income_bf_ret = cal_income(age_coeff)
# income_ret = income_bf_ret[-1]

# get std
sigma_perm = std.loc['sigma_permanent', 'Labor Income Only'][education_level[AltDeg]]
sigma_tran = std.loc['sigma_transitory', 'Labor Income Only'][education_level[AltDeg]]


adj_income = adj_income_process(income_bf_ret, sigma_perm, sigma_tran)


# get conditional survival probabilities
cond_prob = surv_prob.loc[START_AGE:END_AGE - 1, 'CSP']  # 22:99
cond_prob = cond_prob.values


###########################################################################
#                  DP - generate consumption functions                    #
###########################################################################
# prof = cProfile.Profile()
# prof.enable()

if run_dp:
    c_func_fp = os.path.join(base_path, 'results', 'c function_' + education_level[AltDeg] + '.xlsx')
    v_func_fp = os.path.join(base_path, 'results', 'v function_' + education_level[AltDeg] + '.xlsx')
    c_func_df, v = dp_solver(adj_income, cond_prob)
    c_func_df.to_excel(c_func_fp)
    # v.to_excel(v_func_fp)
else:
    c_func_fp = os.path.join(base_path, 'results', 'c function_' + education_level[AltDeg] + '.xlsx')
    c_func_df = pd.read_excel(c_func_fp)

# prof.disable()
# prof.dump_stats(os.path.join(base_path, 'results', f'prof.stats'))


###########################################################################
#        CE - calculate consumption process & certainty equivalent        #
###########################################################################
c_proc_fp = os.path.join(base_path, 'results', 'c process_' + education_level[AltDeg] + '.xlsx')
inc_proc_fp = os.path.join(base_path, 'results', 'inc process_' + education_level[AltDeg] + '.xlsx')

c_ce_arr = np.zeros(N)
for i in range(N):
    c_proc, inc = generate_consumption_process(income_bf_ret, sigma_perm, sigma_tran, c_func_df, flag)

    # c_proc = pd.DataFrame(c_proc)
    # c_proc.to_excel(c_proc_fp, index=False)

    # inc = pd.DataFrame(inc)
    # inc.to_excel(inc_proc_fp, index=False)

    # plt.figure()
    # plt.plot(c_proc[:100, :])
    # plt.show()
    # f = (c_proc == 3000000)
    # g = f.sum(axis=1) > 0
    # print(g.sum())

    cond_prob = surv_prob.loc[START_AGE:END_AGE, 'CSP']
    prob = cond_prob.cumprod().values

    c_ce, _ = cal_certainty_equi(prob, c_proc)
    c_ce_arr[i] = c_ce
    print(c_ce)

# Params check
print('STD:', c_ce_arr.std())
print('Mean:', c_ce_arr.mean())
print("--- %s seconds ---" % (time.time() - start_time))
print('AltDeg: ', AltDeg)
print('permanent shock: ', sigma_perm)
print('transitory shock: ', sigma_tran)
print('lambda: ', ret_frac[AltDeg])
print('theta: ',  unemp_frac[AltDeg])
print('pi: ', unempl_rate[AltDeg])
print('flag: ', flag)
print('W0: ', INIT_WEALTH)
print('Gamma: ', GAMMA)
print('rho: ', rho)
print('term: ', TERM)

