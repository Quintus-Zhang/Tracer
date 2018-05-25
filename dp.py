from numpy.polynomial.hermite import hermgauss
from functions import *
from constants import *


def dp_solver(income, income_ret, sigma_perm_shock, sigma_tran_shock, prob, *, flag='orig'):
    """ Solve for the policy rules (a.k.a consumption functions) using the backward induction method from the
    perspective of Dynamic Programming

    :param income          : array with size of (44,), deterministic component income over the working period
    :param income_ret      : float64, income at the last working period
    :param sigma_perm_shock: float64, standard deviation of the permanent shock in the log term
    :param sigma_tran_shock: float64, standard deviation of the transitory shock in the log term
    :param prob            : array with size of (78,), conditional survival probability
    :param flag            : string, flag var to choose adding details of ISA or Loan or nothing(just the original model)
    :return:
           c_collection    : DataFrame, consumption functions from age 100 to 22
           v_collection    : DataFrame, value functions from age 100 to 22
    """
    ###########################################################################
    #                                Setup                                    #
    ###########################################################################
    # Gauss-Hermite Quadrature
    [sample_points, weights] = hermgauss(3)
    sample_points = sample_points[None].T   # transpose
    weights = weights[None].T

    # Income Shocks
    inc_shk_perm = lambda t: np.sqrt(2) * np.sqrt(t) * sample_points * sigma_perm_shock  # std of the random walk
    inc_shk_tran = np.sqrt(2) * sample_points * sigma_tran_shock
    income_with_tran = np.exp(inc_shk_tran) * income   # income with transitory shocks

    # Construct the grid of cash-on-hand
    even_grid = np.linspace(0, 1, N_COH)
    grid_coh = LOWER_BOUND_COH + (UPPER_BOUND_COH - LOWER_BOUND_COH) * even_grid**EXPAND_FAC

    # initialize arrays for value function and consumption
    v = np.zeros((2, N_COH))
    c = np.zeros((2, N_COH))

    # Terminal Period: consume all the cash-on-hand
    ut = utility(grid_coh, GAMMA)
    v[0, :] = ut
    c[0, :] = grid_coh

    # collect results - v and c
    col_names = [str(age + START_AGE) for age in range(END_AGE-START_AGE, -1, -1)]     # 100 to 22
    c_collection = pd.DataFrame(index=pd.Int64Index(range(N_COH)), columns=col_names)
    v_collection = pd.DataFrame(index=pd.Int64Index(range(N_COH)), columns=col_names)
    c_collection[str(END_AGE)] = c[0, :]
    v_collection[str(END_AGE)] = v[0, :]

    ###########################################################################
    #                         Dynamic Programming                             #
    ###########################################################################
    for t in range(END_AGE-START_AGE-1, -1, -1):       # t: 77 to 0 / t+22: 99 to 22
        print('############ Age: ', t+START_AGE, '#############')
        for i in range(N_COH):

            # Grid Search: for each coh in the grid_coh, we search for the consumption which maximizes the v
            consmp = np.linspace(0, grid_coh[i], N_C)
            u_r = utility(consmp, GAMMA)
            u_r = u_r[None].T

            savings = grid_coh[i] - consmp
            savings_incr = savings * (1 + R)
            savings_incr = savings_incr[None].T

            if t + START_AGE >= RETIRE_AGE:
                expected_value = exp_val_r(income_ret, np.exp(inc_shk_perm(RETIRE_AGE-START_AGE+1)), savings_incr, grid_coh, v[0, :], weights)
            else:
                expected_value = exp_val(income_with_tran[:, t+1], np.exp(inc_shk_perm(t+1)),
                                         savings_incr, grid_coh, v[0, :], weights, t+START_AGE, flag)   # using Y_t+1 !

            v_array = u_r + DELTA * prob[t] * expected_value    # v_array has size N_C-by-1
            v[1, i] = np.max(v_array)
            pos = np.argmax(v_array)
            c[1, i] = consmp[pos]

        # dump consumption array and value function array
        c_collection[str(t + START_AGE)] = c[1, :]
        v_collection[str(t + START_AGE)] = v[1, :]

        # update v and c for calculation next stage
        v[0, :] = v[1, :]
        c[0, :] = c[1, :]  # useless here

    return c_collection, v_collection