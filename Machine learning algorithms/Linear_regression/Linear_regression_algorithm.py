from statistics import mean
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import style
import random

#imported required modules for the algorithm.

style.use('fivethirtyeight')           #style for the graph.

def create_dataset(hm,variance, step = 2, correlation = False):
    #function for creating the data set which is going to be tested.
    
    val = 1
    ys = []
    for i in range(hm):
        y = val + random.randrange(-variance,variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step
    xs = [i for i in range(len(ys))]
    
    return np.array(xs, dtype = np.float64), np.array(ys, dtype = np.float64)

def best_fit_slope_and_intercept(xs,ys):
    #function for getting the best fit slope and the intercept.
    
    m = ( ((mean(xs) * mean(ys)) - mean(xs * ys)) /
            ((mean(xs) ** 2) - mean(xs ** 2)))
    b = mean(ys) - m * mean(xs)
    return m,b

def squared_error(ys_orig,ys_line):
    #function for getting the squared error.

    return sum((ys_line-ys_orig) ** 2)

def coefficient_of_determination(ys_orig,ys_line):
    #function for getting thr coefficient of determination.

    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_regr / squared_error_y_mean)

xs,ys = create_dataset(40,40,2,correlation = 'pos')     #storing the returned data set in xs and ys variables.

m,b = best_fit_slope_and_intercept(xs,ys)               #storing the best fit slope and the intercept in m and b data set.

regression_line = [(m*x) + b for x in xs]               #getting the regression line.

predict_x = 8                                           #prediction for x value.
predict_y = (m*predict_x) + b                           #prediction for y value.

r_squared = coefficient_of_determination(ys, regression_line)       #getting the r_squared value.
print(r_squared)

#plotting the graph.
plt.scatter(xs,ys)
plt.scatter(predict_x,predict_y,color = 'green')
plt.plot(xs,regression_line)
plt.show()
