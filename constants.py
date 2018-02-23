###########################################################################
#                            Set Constants                                #
###########################################################################
START_AGE = 22            #
END_AGE = 100             #
RETIRE_AGE = 65           # retirement age
N_W = 1001
UPPER_BOUND_W = 2000000   # upper bound of wealth (2001 * 4001: 142min)
N_C = 1001
GAMMA = 2                 # risk preference parameter
R = 0.02                  # risk-free rate
DELTA = 0.99              # discount factor
MU = 0                    # expectation of income shocks
N_SIM = 100000            # number of draws

education_level = {
    1: 'No High School',
    2: 'High School Graduates',
    4: 'College Graduates'
}

# replacement rate of retirement earnings
ret_frac = {
    1: 0.68212,
    2: 0.68212,
    4: 0.68212,
}
