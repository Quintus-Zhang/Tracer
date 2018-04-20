### The Model
We are trying to solve a stochastic dynamic programming problem with discrete and finite horizon, and continuous state space. The Bellman equation for this problem is given by:

[comment]: <> ($$V_t(W_t) = Max_{C_t \ge 0} [U(C_t) + \beta p_tE_t(V_{t+1}(W_{t+1}))],\ \ for\ t\lt T$$)

![Bellman Equation](http://www.sciweavers.org/tex2img.php?eq=%24%24V_t%28W_t%29%20%3D%20Max_%7BC_t%20%5Cge%200%7D%20%5BU%28C_t%29%20%2B%20%5Cbeta%20p_tE_t%28V_%7Bt%2B1%7D%28W_%7Bt%2B1%7D%29%29%5D%2C%5C%20%5C%20for%20t%20%3C%20T%24%24%0A&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)

where $C_t$ is the level of consumption at time $t$, $\beta \lt 1$ is the discount factor, $p_t$ is the probability that an agent is alive at time $t+1$ conditional on being alive at time $t$

$W_{t+1}$ is given by the agent's wealth accumulation equation:

$$W_{t+1} = R(W_t-C_t) + Y_{t+1}$$

$Y_t$, agent's labor income at time $t$, is given by:

$$log(Y_t) = f_t + v_t + \varepsilon_t$$

$U(C_t)$ is the utility function:

$$U(C_t) = {(C_t)^{1-\gamma} \over 1-\gamma}$$

The problem is to solve for the policy rules as a function of the state variables, that is, $C_t(W_t)$.


### Backward Induction Method
The idea of backward induction is to solve the problem from the end and working backwards towards the initial period. Steps of the method are as follows:
1. Determine the value function for $V_T(W_T)$ for all $W_T$
2. Start from $t=T-1$, we discretize the state space for the $W_t$ and $C_t$, solve a one-step optimization problem using grid search, that is, for every $W_{t}$ and $C_{t}$, evaluate the value function, and then choose for every $W_{t}$
$$C_{t}^\*(W_t) = argmax_{C_t\ge 0} [U(C_t) + \beta p_tE_t(V_{t+1}(R(W_{t}-C_{t})+Y_{t+1}))]$$
3. If $t \gt 0$, $t = t - 1$, and go back to step 2, otherwise stop.


### Cubic Spline
In a continuous state model, the value function is infinite-dimensional, and we must devise a grid in the state space. When $W_t$ does not lie in the chosen grid, we can approximate value function using interpolation method such as cubic spline.


### Gauss-Hermite Quadrature
The Gauss-Hermite Quadrature can approximate the value of integrals of the following kind:

$$\int_{+\infty}^{-\infty} e^{-x^2}f(x)dx \approx \sum_{i=1}^n w_if(x_i) $$

Integral such as the left-hand side can be derived from the expectation, because the random variable $Y_{t+1}$ follows the normal distribution.

$$E_t[V_{t+1}(R(W_{t}-C_{t})+Y_{t+1})]$$


[ref1](https://en.wikipedia.org/wiki/Gauss–Hermite_quadrature)
[ref2](https://stats.stackexchange.com/questions/159650/why-does-the-variance-of-the-random-walk-increase)
[ref3](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.polynomial.hermite.hermgauss.html)
