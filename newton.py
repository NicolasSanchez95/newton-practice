

def first_derivative(f, x, eps = 0.01):
    f_1 = (f(x+eps) - f (x))/eps
    return f_1

def second_derivative(f,x, eps = 0.01):
    f_2 = (first_derivative(f,x+eps,eps) - first_derivative(f,x,eps))/eps
    return f_2

def optimize(f,x_0, delta = 0.0001, eps = 0.0001, verbose = False):
    x_t = x_0
    counter = 0
    while counter < 1000:
        counter+=1
        x_tplus1 = x_t - first_derivative(f,x_t, eps)/second_derivative(f,x_t, eps)
        if counter%1 == 0 and verbose:
            print("Counter : ", counter)
            print("x_t : ", x_t, "x_t+1", x_tplus1)
            print("f(x_t): ", f(x_t), "f(x+1): ", f(x_tplus1)) 
            print("Diff: ", f(x_tplus1) - f(x_t)) 
        if abs(f(x_tplus1) - f(x_t)) < delta :
            break
        x_t = x_tplus1
    return x_t, first_derivative(f,x_t, eps), second_derivative(f,x_t, eps), counter


f = lambda x : 2 + (x-1)**2+(x-1)**4
optimize(f,100, verbose = True)