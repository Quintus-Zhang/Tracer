import os
import time
import numpy as np
import pandas as pd
from functions import *
from dp import dp_solver
from cal_ce import cal_certainty_equi, generate_consumption_process
from constants import *
import multiprocessing as mp
import itertools


def run_model(TERM, rho, gamma):
    start = time.time()
    print(f'########## Term: {TERM} | Rho: {rho:.2f} | Gamma: {gamma} ##########')

    # get conditional survival probabilities
    cond_prob = surv_prob.loc[START_AGE:END_AGE - 1, 'CSP']  # 22:99
    cond_prob = cond_prob.values

    ###########################################################################
    #                  DP - generate consumption functions                    #
    ###########################################################################
    c_func_fp = os.path.join(base_path, 'results', f'c function_{TERM}_{rho:.2f}_{gamma}.xlsx')
    c_func_df, _ = dp_solver(income_bf_ret, income_ret, sigma_perm, sigma_tran, cond_prob, TERM, rho, gamma)
    c_func_df.to_excel(c_func_fp)

    ###########################################################################
    #        CE - calculate consumption process & certainty equivalent        #
    ###########################################################################
    c_ce_arr = np.zeros(N)
    for i in range(N):
        c_proc, _ = generate_consumption_process(income_bf_ret, sigma_perm, sigma_tran, c_func_df, TERM, rho)

        prob = surv_prob.loc[START_AGE:END_AGE, 'CSP'].cumprod().values

        c_ce, _ = cal_certainty_equi(prob, c_proc, gamma)
        c_ce_arr[i] = c_ce

    print(f"------ {time.time() - start} seconds ------")
    print(c_ce_arr.mean())
    return c_ce_arr.mean()


start_time = time.time()

###########################################################################
#                      Setup - file path & raw data                       #
###########################################################################
# set file path
income_fn = 'age_coefficients_and_var.xlsx'
surviv_fn = 'Conditional Survival Prob Feb 16.xlsx'
isa_fn = 'Loop on rho.xlsx'
base_path = os.path.dirname(__file__)
income_fp = os.path.join(base_path, 'data', income_fn)
mortal_fp = os.path.join(base_path, 'data', surviv_fn)
isa_fp = os.path.join(base_path, 'data', isa_fn)
ce_fp = os.path.join(base_path, 'results', 'ce.xlsx')

# read raw data
age_coeff, std, surv_prob = read_input_data(income_fp, mortal_fp)


###########################################################################
#              Setup - income process & std & survival prob               #
###########################################################################
income_bf_ret = cal_income(age_coeff)
income_ret = income_bf_ret[-1]

# get std
sigma_perm = std.loc['sigma_permanent', 'Labor Income Only'][education_level[AltDeg]]
sigma_tran = std.loc['sigma_transitory', 'Labor Income Only'][education_level[AltDeg]]



# read isa params
isa_params = pd.read_excel(isa_fp)
ce = isa_params[["TERM FOR ISA", "1- rho"]].copy()

gamma_arr = np.arange(0.1, 8.1, 0.25)
ce = pd.concat([ce]*gamma_arr.size, ignore_index=True)
ce['gamma'] = np.repeat(gamma_arr, isa_params.shape[0])

search_args = list(itertools.product(ce.values[:, 0], ce.values[:, 1], ce.values[:, 2]))

with mp.Pool(processes=mp.cpu_count()) as p:
    c_ce = p.starmap(run_model, search_args)

ce['Consumption CE'] = c_ce

ce.to_excel(ce_fp)


# Params check
# print('STD:', c_ce_arr.std())
# print('Mean:', c_ce_arr.mean())
print("--- %s seconds ---" % (time.time() - start_time))
print('AltDeg: ', AltDeg)
print('permanent shock: ', sigma_perm)
print('transitory shock: ', sigma_tran)
print('lambda: ', ret_frac[AltDeg])
print('theta: ',  unemp_frac[AltDeg])
print('pi: ', unempl_rate[AltDeg])
print('W0: ', INIT_WEALTH)


