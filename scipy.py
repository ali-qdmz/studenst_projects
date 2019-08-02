import numpy as np
import scipy
import scipy.optimize as scopt
import matplotlib.pyplot as plt
y_array = np.array([3.5,4,4.75,6.5])
x_array = np.array([3.5,3.75,5.75,6.25])
y_array = y_array
x_array = x_array
pi = np.array([0,0.015,0.025,0.035])


def linear(x, a, b, c):
    return a*x + (np.exp(b*x))*(np.exp(c*x))

popt_linear, pcov_linear = scopt.curve_fit(linear, x_array, y_array)

#perr_linear = np.sqrt(np.diag(pcov_linear))
print (popt_linear[0])
print (popt_linear[1])
print (popt_linear[2])
def myfunc(x):
    result = np.ones((len(x)))
    for i in range(len(x)):
        result[i]= linear(x_array[i],popt_linear[0],popt_linear[1],popt_linear[2])
    return result

def my(x):
    res = popt_linear[0]*x + (np.exp(popt_linear[1]*x)*np.exp(popt_linear[2]*x))
    return res
plt.plot(pi,x_array)
plt.plot(pi,y_array)
#plt.show()
z = np.array(myfunc(x_array))
print(z)
print(z - x_array)
z = abs(z - x_array)
print(z.sum())
