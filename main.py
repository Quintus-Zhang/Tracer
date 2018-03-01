import os
import time
import pandas as pd
from functions import read_input_data, cal_income
from dp import dp_solver
from cal_ce import cal_certainty_equi, generate_consumption_process
from constants import START_AGE, END_AGE, N_W, UPPER_BOUND_W, N_C, GAMMA, R, DELTA, education_level, ret_frac, unemp_frac, unempl_rate

start_time = time.time()

# TODO: 1. bernoulli
# TODO: 2. take out the variance of labor income and unemployment income

# set file path
income_fn = 'age_coefficients_and_var.xlsx'
surviv_fn = 'Conditional Survival Prob Feb 16.xlsx'
base_path = os.path.dirname(__file__)
income_fp = os.path.join(base_path, 'data', income_fn)
mortal_fp = os.path.join(base_path, 'data', surviv_fn)
ce_fp = os.path.join(base_path, 'results', 'ce.xlsx')

# read data
age_coeff, std, surv_prob = read_input_data(income_fp, mortal_fp)

AltDeg = 4

# calculate income process
income = cal_income(age_coeff, AltDeg, True)              # labor income only
# income_unempl = cal_income(age_coeff, AltDeg, False)      # with unemployment income
# income_bf_ret = (1 - unempl_rate[AltDeg]) * income + unempl_rate[AltDeg] * income_unempl
income_bf_ret = income
income_ret = ret_frac[AltDeg] * income_bf_ret[-1]

# get std
sigma_perm = 0  # std.loc['sigma_permanent', 'Labor Income Only'][education_level[AltDeg]]
sigma_tran = 0  # std.loc['sigma_transitory', 'Labor Income Only'][education_level[AltDeg]]

# get conditional survival probabilities
cond_prob = surv_prob.loc[START_AGE:END_AGE - 1, 'CSP']  # 22:99
cond_prob = cond_prob.values

# set output files path & generate consumption functions
c_func_fp = os.path.join(base_path, 'results', 'c function_' + education_level[AltDeg] + '.xlsx')
if False:
    dp_solver(income_bf_ret, income_ret, sigma_perm, sigma_tran, cond_prob, c_func_fp)

# calculate the consumption process, assumed initial wealth 0
c_proc_fp = os.path.join(base_path, 'results', 'c process_' + education_level[AltDeg] + '.xlsx')
c_func_df = pd.read_excel(c_func_fp)
c_proc = generate_consumption_process(income_bf_ret, income_ret, sigma_perm, sigma_tran, c_func_df, c_proc_fp)


# c_proc_fp = os.path.join(base_path, 'results', 'c process_College Graduates_Labor Income Only.xlsx')
# c_proc = pd.read_excel(c_proc_fp, header=0)

cond_prob = surv_prob.loc[START_AGE:END_AGE, 'CSP']
prob = cond_prob.cumprod().values

c_ce, _ = cal_certainty_equi(prob, c_proc)
print(c_ce)



# col_names = ['Consumption CE', 'Total Wealth CE']
# idx_names = education_level.values()
# ce = pd.DataFrame(index=idx_names, columns=col_names)
# for AltDeg in [4]:
#     ce.loc[education_level[AltDeg]] = cal_certainty_equi(age_coeff, std, surv_prob, AltDeg)
#
# print(ce)
# ce.to_excel(ce_fp)


# os.system('say "your program has finished"')
print("--- %s seconds ---" % (time.time() - start_time))
print('AltDeg: ', AltDeg)
print('permanent shock: ', sigma_perm)
print('transitory shock: ', sigma_tran)
print('lambda: ', ret_frac[AltDeg])
print('theta: ',  unemp_frac[AltDeg])
print('pi: ', unempl_rate[AltDeg])
