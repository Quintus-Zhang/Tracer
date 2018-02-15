import os
import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss
from functions import utility, exp_val
from constants import START_AGE, END_AGE, N_W, UPPER_BOUND_W, N_C, GAMMA, R, DELTA, education_level


def dp_solver(income, std, surviv_prob, AltDeg):
    ###########################################################################
    #                                Setup                                    #
    ###########################################################################
    # set output file path
    base_path = os.path.dirname(__file__)
    consmp_fp = os.path.join(base_path, 'results', 'consumption_' + education_level[AltDeg] +'.xlsx')
    v_fp = os.path.join(base_path, 'results', 'v_' + education_level[AltDeg] + '.xlsx')

    # Gauss-Hermite Quadrature
    [sample_points, weights] = hermgauss(3)
    sample_points = sample_points[None].T
    weights = weights[None].T

    ###########################################################################
    #                                                                         #
    ###########################################################################
    prob = surviv_prob.loc[START_AGE:END_AGE-1, 'CSP']  # 22:99
    prob = prob.values

    income = income.loc[income['AltDeg'] == AltDeg]  # 20 to 100
    income.set_index('Age', inplace=True)
    income = income.loc[START_AGE:END_AGE]

    sigma_perm_shock = std.loc['sigma_permanent', education_level[AltDeg]]
    sigma_tran_shock = std.loc['sigma_transitory', education_level[AltDeg]]
    # sigma_perm_shock = 0
    # sigma_tran_shock = 0
    # inc_shk_perm = lambda t: np.sqrt(2) * np.sqrt(t) * sample_points * sigma_perm_shock
    inc_shk_perm = np.sqrt(2) * sample_points * sigma_perm_shock
    inc_shk_tran = np.sqrt(2) * sample_points * sigma_tran_shock
    income_with_tran = np.exp(inc_shk_tran) * income['f'].values

    # construct grids
    grid_w = np.linspace(1, UPPER_BOUND_W, N_W)

    # initialize arrays for value function and consumption
    v = np.zeros((2, N_W))
    c = np.zeros((2, N_W))

    # terminal period: consume all the wealth
    ut = utility(grid_w, GAMMA)
    v[0, :] = ut
    c[0, :] = grid_w

    # collect results
    col_names = [str(age + START_AGE) for age in range(END_AGE-START_AGE, -1, -1)]     # 100 to 22
    c_collection = pd.DataFrame(index=pd.Int64Index(range(N_W)), columns=col_names)
    v_collection = pd.DataFrame(index=pd.Int64Index(range(N_W)), columns=col_names)
    c_collection[str(END_AGE)] = c[0, :]
    v_collection[str(END_AGE)] = v[0, :]

    ###########################################################################
    #                         Dynamic Programming                             #
    ###########################################################################
    for t in range(END_AGE-START_AGE-1, -1, -1):       # t: 77 to 0 / t+22: 99 to 22
        print('############ Age: ', t+START_AGE, '#############')
        for i in range(N_W):

            # Grid Search: for each W in the grid_w, we search for the C which maximizes the V
            consmp = np.linspace(0, grid_w[i], N_C)
            u_r = utility(consmp, GAMMA)
            u_r = u_r[None].T

            savings = grid_w[i] - np.linspace(0, grid_w[i], N_C)
            savings_incr = savings * (1 + R)
            savings_incr = savings_incr[None].T

            expected_value = exp_val(income_with_tran[:, t+1], np.exp(inc_shk_perm),
                                     savings_incr, grid_w, v[0, :], weights)

            v_array = u_r + DELTA * prob[t] * expected_value    # v_array has size N_C-by-1
            v[1, i] = np.max(v_array)
            pos = np.argmax(v_array)
            c[1, i] = consmp[pos]

        # dump consumption array and value function array
        c_collection[str(t + START_AGE)] = c[1, :]
        v_collection[str(t + START_AGE)] = v[1, :]

        # change v & c for calculation next stage
        v[0, :] = v[1, :]
        c[0, :] = c[1, :]  # useless here

    c_collection.to_excel(consmp_fp)
    v_collection.to_excel(v_fp)

    return

