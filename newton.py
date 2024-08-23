import warnings
import numpy as np # tk-violin

def first_derivative(f, x, eps = 0.01):
    """ Find first derivative of given function evaluated at a given point
    with a fixed distance for the numerical approximation to the derivative.

    Args:
        f: function of which to take derivative. Must take x as input
        x: input to f
        eps (optional): distance used for numerical approximation

    Returns:
        f'(x): the derivative of f evaluated at x
    """
    f_1 = (f(x+eps) - f(x))/eps
    return f_1

def second_derivative(f,x, eps = 0.01):
    """ Find second derivative of given function evaluated at a given point
    with a fixed distance for the numerical approximation to the derivative.

    Args:
        f: function of which to take derivative. Must take x as input
        x: input to f
        eps (optional): distance used for numerical approximation
    Returns:
        f''(x): the second derivative of f evaluated at x
    """
    f_2 = (first_derivative(f,x+eps,eps) - first_derivative(f,x,eps))/eps
    return f_2

def optimize(f,x_0, delta = 1e-6, eps = 1e-6, verbose = False):
    """
    Runs 1-dimensional newton's method to find local minima of f starting
    from input x_0.

    Args:
        f: objective function that we are trying to minimize
        x_0: starting point for Newton's method
        eps (optional): distance used for numerical approximation of derivatives
        delta (optional): threshhold in difference of input to signal the end of
        the iterative method

    Returns:
        (x, f'(x), f''(x), counter)
        x : the local argmin of f found
        f(x): the local  minimum of the function found
        f'(x): the first derivative at the local argmin
        f''(x): the second derivative at the local argmin
        counter: number of iterations taken in method to reach x
    """
    x_t = x_0
    counter = 0
    if not callable(f):
        raise TypeError('Function provided is not a function')
    if not isinstance(x_0, (float, int)):
        raise TypeError('Starting point must be numeric')
    while counter < 1000:
        counter+=1
        import pdb
        pdb.set_trace()
        if abs(second_derivative(f,x_t,eps)) < 1e-5:
            warnings.warn("Second Derivative is close to zero")
        if second_derivative(f,x_t,eps) == 0:
            raise ZeroDivisionError('Second Derivative is identically zero')
        x_tplus1 = x_t - first_derivative(f,x_t, eps)/second_derivative(f,x_t, eps)
        if counter%1 == 0 and verbose:
            print("Counter : ", counter)
            print("x_t : ", x_t, "x_t+1", x_tplus1)
            print("f(x_t): ", f(x_t), "f(x+1): ", f(x_tplus1)) 
            print("Diff: ", f(x_tplus1) - f(x_t)) 
        if abs(x_tplus1- x_t) < delta :
            break
        x_t = x_tplus1
    if counter == 1000:
        raise Exception("Newton's method did not converge within \
            the set max numberof iteration")
    return x_t, f(x_t), first_derivative(f,x_t, eps), second_derivative(f,x_t, eps), counter
