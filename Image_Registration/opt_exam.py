import scipy.optimize as spo
def f(x):
    y=(x**2)-12*x+20
    return y
x_test=2
result=spo.minimize(f,x_test,options={"disp":True})
