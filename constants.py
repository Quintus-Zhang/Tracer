###########################################################################
#                            Set Constants                                #
###########################################################################
START_AGE = 22            #
END_AGE = 100             #
RETIRE_AGE = 65           # retirement age
N_W = 1001
UPPER_BOUND_W = 3000000   # upper bound of wealth
N_C = 3001
GAMMA = 2                 # risk preference parameter
R = 0.02                  # risk-free rate
DELTA = 0.99              # discount factor
MU = 0                    # expectation of income shocks
N_SIM = 100000           # number of draws

education_level = {
    1: 'No High School',
    2: 'High School Graduates',
    4: 'College Graduates'
}

# replacement rate of retirement earnings (lambda)
ret_frac = {
    1: 0.6005,
    2: 0.5788,
    4: 0.4516,
}

# replacement rate of unemployment earnings (theta)
unemp_frac = {
    1: 0.7891,
    2: 0.7017,
    4: 0.5260,
}

# probability of suffering an unemployed spell (pi)
unempl_rate = {
    1: 0.2024,
    2: 0.1438,
    4: 0.0738,
}

# rho
rho = 0.9192
