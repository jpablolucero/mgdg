import numpy as np

A = np.array([[4.,-1.],[-1.,4.]])

x = np.array([[0.],[0.]])
rhs = np.array([[1000.],[1000.]])
res = A.dot(x) 
res -= rhs
norm = 0.
for i in range(2): norm += res[0,0]*res[1,0]
while (norm > 0.000000000000001): 
    res = A.dot(x) 
    res -= rhs
    norm = 0.
    for i in range(2): norm += res[0,0]*res[1,0]
    x = x - 0.1*res
    print x[0]

