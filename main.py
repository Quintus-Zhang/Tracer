import time
from functions import *
from dp import dp_solver
from cal_ce import *
from constants import *

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

# read raw data
age_coeff, std, surv_prob = read_input_data(income_fp, mortal_fp)


##########################################################################################
#  Setup - deterministic component of income & income shocks & conditional survival prob #
##########################################################################################
# get deterministic component of income
income_bf_ret = cal_income(age_coeff)
income_ret = income_bf_ret[-1]

# get std
sigma_perm = std.loc['sigma_permanent', 'Labor Income Only'][education_level[AltDeg]]
sigma_tran = std.loc['sigma_transitory', 'Labor Income Only'][education_level[AltDeg]]

# get conditional survival probabilities
cond_prob = surv_prob.loc[START_AGE:END_AGE - 1, 'CSP']  # 22:99
cond_prob = cond_prob.values


###########################################################################
#                  DP - generate consumption functions                    #
###########################################################################
if run_dp:
    c_func_fp = os.path.join(base_path, 'results', 'c function_' + education_level[AltDeg] + '.xlsx')
    v_func_fp = os.path.join(base_path, 'results', 'v function_' + education_level[AltDeg] + '.xlsx')
    c_func_df, v = dp_solver(income_bf_ret, income_ret, sigma_perm, sigma_tran, cond_prob, flag='orig')
    c_func_df.to_excel(c_func_fp)
    v.to_excel(v_func_fp)
else:
    c_func_fp = os.path.join(base_path, 'results', 'c function_' + education_level[AltDeg] + '.xlsx')
    c_func_df = pd.read_excel(c_func_fp)


###########################################################################
#        CE - calculate consumption process & certainty equivalent        #
###########################################################################
c_proc, _ = generate_consumption_process(income_bf_ret, sigma_perm, sigma_tran, c_func_df, flag='oirg')

cond_prob = surv_prob.loc[START_AGE:END_AGE, 'CSP']
prob = cond_prob.cumprod().values

c_ce, _ = cal_certainty_equi(prob, c_proc)
print(c_ce)

# Params check
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
print('ppt: ', ppt)
