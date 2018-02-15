import os
import time
import pandas as pd
from functions import utility, read_input_data
from dp import dp_solver
from cal_ce import cal_certainty_equi
from constants import START_AGE, END_AGE, N_W, UPPER_BOUND_W, N_C, GAMMA, R, DELTA, education_level

start_time = time.time()

# set file path
income_fn = 'labor_income_process_test.xls'
surviv_fn = 'Conditional Survival Prob Feb 16.xlsx'
base_path = os.path.dirname(__file__)
income_fp = os.path.join(base_path, 'data', income_fn)
mortal_fp = os.path.join(base_path, 'data', surviv_fn)
ce_fp = os.path.join(base_path, 'results', 'ce.xlsx')
c_func_dir = 'c_v_2m_income_test'

# read data
income, std, surviv_prob = read_input_data(income_fp, mortal_fp)

# generate comsumption functions
if True:
    for AltDeg in [1, 2, 4]:
        print('#'*30, ' AltDeg: ', AltDeg, ' ', '#'*30)
        dp_solver(income, std, surviv_prob, AltDeg)

# calculate ce
col_names = ['Consumption CE', 'Total Wealth CE']
idx_names = education_level.values()
ce = pd.DataFrame(index=idx_names, columns=col_names)
for AltDeg in [1, 2, 4]:
    ce.loc[education_level[AltDeg]] = cal_certainty_equi(income, std, surviv_prob, AltDeg, c_func_dir)

print(ce)
ce.to_excel(ce_fp)

# os.system('say "your program has finished"')
print("--- %s seconds ---" % (time.time() - start_time))
