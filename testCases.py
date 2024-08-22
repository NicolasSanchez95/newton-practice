import pytest
import numpy as np
import math

import newton
import warnings

def test_basic_function():
    assert np.isclose(newton.optimize(np.cos,2.95)[0], math.pi)

def test_bad_input():
    with pytest.raises(TypeError, match = 'Starting point must be numeric'):
        newton.optimize(np.cos, np.cos)[0]
    with pytest.raises(TypeError, match = 'Function provided is not a function'):
        newton.optimize(2.95, 2.95)[0]

def test_small_second_deriv():
    with warnings.catch_warnings(record=True) as w:
        newton.optimize(lambda x: 1e-16*x**2 ,0)[-1]
    assert w == "Second Derivative is close to zero"

def test_zero_second_deriv():
    with pytest.raises(ZeroDivisionError, match = 'Second Derivative is identically zero'):
        newton.optimize(lambda x: 1e-16 ,2.)


    