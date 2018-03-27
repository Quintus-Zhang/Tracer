import os
import time
import pandas as pd
from functions import read_input_data, cal_income
from dp import dp_solver
from cal_ce import cal_certainty_equi, generate_consumption_process
from constants import START_AGE, END_AGE, education_level, ret_frac, unemp_frac, unempl_rate, AltDeg, flag, run_dp

start_time = time.time()

###########################################################################
#                      Setup - file path & raw data                       #
###########################################################################
# set file path
income_fn = 'age_coefficients_and_var.xlsx'
surviv_fn = 'Conditional Survival Prob Feb 16.xlsx'
base_path = os.path.dirname(__file__)
income_fp = os.path.join(base_path, 'data', income_fn)
mortal_fp = os.path.join(base_path, 'data', surviv_fn)
ce_fp = os.path.join(base_path, 'results', 'ce.xlsx')

# read raw data
age_coeff, std, surv_prob = read_input_data(income_fp, mortal_fp)


###########################################################################
#              Setup - income process & std & survival prob               #
###########################################################################
income = cal_income(age_coeff, AltDeg)              # labor income only
income_bf_ret = income
income_ret = income_bf_ret[-1]

# get std
sigma_perm = std.loc['sigma_permanent', 'Labor Income Only'][education_level[AltDeg]]
sigma_tran = std.loc['sigma_transitory', 'Labor Income Only'][education_level[AltDeg]]
# sigma_perm = 0
# sigma_perm = 0

# get conditional survival probabilities
cond_prob = surv_prob.loc[START_AGE:END_AGE - 1, 'CSP']  # 22:99
cond_prob = cond_prob.values

###########################################################################
#                  DP - generate consumption functions                    #
###########################################################################
# c_func_fp = os.path.join(base_path, 'results', 'c function_' + education_level[AltDeg] + '.xlsx')
v_func_fp = os.path.join(base_path, 'results', 'v function_' + education_level[AltDeg] + '.xlsx')
c_func_fp = os.path.join(base_path, 'results', 'c_orig_2_grid_cw.xlsx')
if run_dp:
    dp_solver(income_bf_ret, income_ret, sigma_perm, sigma_tran, cond_prob, unemp_frac[AltDeg], unempl_rate[AltDeg], flag, c_func_fp, v_func_fp)


###########################################################################
#        CE - calculate consumption process & certainty equivalent        #
###########################################################################
# c_proc_fp = os.path.join(base_path, 'results', 'c process_' + education_level[AltDeg] + '.xlsx')
c_func_df = pd.read_excel(c_func_fp)

c_ce_list = []
for i in range(1):
    c_proc = generate_consumption_process(income_bf_ret, sigma_perm, sigma_tran, c_func_df, AltDeg, flag)

    # c_proc_fp = os.path.join(base_path, 'results', 'c process_College Graduates_Labor Income Only.xlsx')
    # c_proc = pd.read_excel(c_proc_fp, header=0)

    cond_prob = surv_prob.loc[START_AGE:END_AGE, 'CSP']
    prob = cond_prob.cumprod().values

    c_ce, _ = cal_certainty_equi(prob, c_proc)
    c_ce_list.append(c_ce)
    print(c_ce)
print('Mean:', sum(c_ce_list) / 1)


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
