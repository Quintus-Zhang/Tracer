import os
import time
import pandas as pd
import numpy as np
from functions import read_input_data, cal_income, utility
from dp import dp_solver
from cal_ce import cal_certainty_equi
from constants import START_AGE, END_AGE, N_W, UPPER_BOUND_W, N_C, GAMMA, R, DELTA, education_level, RETIRE_AGE

######## block - print the income data
# start_time = time.time()
#
# # set file path
# income_fn = 'age_coefficients_and_var.xlsx'
# surviv_fn = 'Conditional Survival Prob Feb 16.xlsx'
# base_path = os.path.dirname(__file__)
# income_fp = os.path.join(base_path, 'data', income_fn)
# mortal_fp = os.path.join(base_path, 'data', surviv_fn)
# ce_fp = os.path.join(base_path, 'results', 'ce.xlsx')
# # c_func_dir = 'c_v_2m_income_test'
#
# # read data
# age_coeff, std, surv_prob = read_input_data(income_fp, mortal_fp)
#
# coeff_this_group = age_coeff.loc[education_level[4]]
# ages = np.arange(START_AGE, RETIRE_AGE + 1)  # 22 to 65
# income = cal_income(coeff_this_group, ages)  # 0:43, 22:65
# print(income)
#
# inc_df = pd.DataFrame(income)
# inc_df.to_excel('inc.xlsx', index=False)

unempl_rate = 0.0738

c_simu = pd.read_excel('/Users/Quintus/Google Drive/Dynamic Programming/code/results/c process_College Graduates_Labor Income Only.xlsx', header=0)
c = c_simu.as_matrix()

# c_simu_repl = pd.read_excel('c_simu_replace.xlsx', header=0)
# spls_c_simu = c_simu.sample(frac=(1-unempl_rate)).as_matrix()
# spls_c_simu_repl = c_simu_repl.sample(frac=unempl_rate).as_matrix()
#
# c = np.append(spls_c_simu, spls_c_simu_repl, axis=0)

# TODO: temp - calculate the prob
# set file path
income_fn = 'age_coefficients_and_var.xlsx'
surviv_fn = 'Conditional Survival Prob Feb 16.xlsx'
base_path = os.path.dirname(__file__)
income_fp = os.path.join(base_path, 'data', income_fn)
mortal_fp = os.path.join(base_path, 'data', surviv_fn)
ce_fp = os.path.join(base_path, 'results', 'ce.xlsx')
# read data
_, _, surv_prob = read_input_data(income_fp, mortal_fp)
# conditional survival probabilities
cond_prob = surv_prob.loc[START_AGE:END_AGE, 'CSP']
prob = cond_prob.cumprod().values

# discount factor
YEARS = END_AGE - START_AGE + 1
delta = np.ones((YEARS, 1)) * DELTA
delta[0] = 1
delta = np.cumprod(delta)


util_c = np.apply_along_axis(utility, 1, c, GAMMA)
simu_util = np.sum(np.multiply(util_c[:, 1:45], (delta * prob)[:44]), axis=1)

c_ce = -np.sum(delta * prob) / np.mean(simu_util)   # TODO: this formula should change with GAMMA
total_w_ce = prob[:44].sum() * c_ce   # 42.7

print(c_ce)
