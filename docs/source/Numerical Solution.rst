
******************
Numerical Solution
******************


Backward Induction Method
=========================
The idea of backward induction is to solve the problem from the end and working
backwards towards the initial period. Steps of the method are as follows:

    1. Determine the value function for :math:`V_T(COH_T)` for all :math:`COH_T`

    2. Start from :math:`t=T-1`, we discretize the state space for the :math:`COH_t` and
       :math:`C_t`, solve a one-step optimization problem using grid search, that is, for
       every :math:`COH_{t}` and :math:`C_{t}`, evaluate the value function, and then choose
       for every :math:`COH_{t}`

       .. math::

         C_{t}^{*}(COH_t) = argmax_{C_t\ge 0} [U(C_t) + \beta p_tE_t(V_{t+1}(R(COH_{t}-C_{t})+Y_{t+1}))]

    3. If :math:`t \gt 0`, :math:`t = t - 1`, and go back to step 2, otherwise stop.


Cubic Spline
============
In a continuous state model, the value function is infinite-dimensional, and we must
devise a grid in the state space. When :math:`COH_t` does not lie in the chosen grid, we can
approximate value function using interpolation method such as cubic spline.


Gauss-Hermite Quadrature
========================
The Gauss-Hermite Quadrature approximates the value of integrals of the following kind:

.. math::
  \int_{+\infty}^{-\infty} e^{-x^2}f(x)dx \approx \sum_{i=1}^n w_if(x_i)

Integral such as the left-hand side can be derived from the expectation, because the random
variable :math:`Y_{t+1}` follows the normal distribution.
