import math
import numpy as np
import sympy as sy
import matplotlib.pyplot as mp
np.set_printoptions(precision=3,threshold=np.nan,linewidth=np.inf,suppress=True)
import lobatto as lo

x,h,d,e,w,w0,a,b,c,f,g,p = sy.symbols('x h d e w w0 a b c f g p')

Dom = 1.
n = 4
p = 1

[xtab,weights] = lo.lobatto_compute(p+1)

Fx = sy.ones(1,p + 1)
for i in range(p + 1):
    for j in range(p + 1):
        if (i != j):
            Fx[i] *= (x - xtab[j] * h) / (xtab[i] * h - xtab[j] * h)

CM = sy.Matrix(sy.zeros((p + 1),(p + 1)))
CM0 = sy.Matrix(sy.zeros((p + 1),(p + 1)))
CM1 = sy.Matrix(sy.zeros((p + 1),(p + 1)))
FM = sy.Matrix(sy.zeros((p + 1),(p + 1)))
for i in range((p + 1)):
    for j in range((p + 1)):
        fa=Fx[i]
        fb=Fx[j]
        CM[i,j] = sy.integrate(fa.diff(x)*fb.diff(x),(x,0,h)) \
                  + 1/e * sy.integrate(fa*fb,(x,0,h)) \
                  - (0                    - fa.subs(x,0)        )     * (0                    + fb.diff(x).subs(x,0)) / 2 \
                  - (0                    + fa.diff(x).subs(x,0)) / 2 * (0                    - fb.subs(x,0)        )     \
                  - (fa.subs(x,h)         - 0                   )     * (fb.diff(x).subs(x,h) + 0                   ) / 2 \
                  - (fa.diff(x).subs(x,h) + 0                   ) / 2 * (fb.subs(x,h)         - 0                   )     \
                  + d / h * (0            - fa.subs(x,0)) * (0            - fb.subs(x,0))                                 \
                  + d / h * (fa.subs(x,h) - 0           ) * (fb.subs(x,h) - 0           )
        CM0[i,j] = CM[i,j]
        CM1[i,j] = CM[i,j]
        if ((i==0)or(j==0)):
            CM0[i,j] += - (0                    - fa.subs(x,0)        )     * (0                    + fb.diff(x).subs(x,0)) / 2 \
                        - (0                    + fa.diff(x).subs(x,0)) / 2 * (0                    - fb.subs(x,0)        )     \
                        + d / h * (0            - fa.subs(x,0)) * (0            - fb.subs(x,0))                                 
        if ((i==p)or(j==p)):
            CM1[i,j] += - (fa.subs(x,h)         - 0                   )     * (fb.diff(x).subs(x,h) + 0                   ) / 2 \
                        - (fa.diff(x).subs(x,h) + 0                   ) / 2 * (fb.subs(x,h)         - 0                   )     \
                        + d / h * (fa.subs(x,h) - 0           ) * (fb.subs(x,h) - 0           )

        FM[i,j] = - (fa.subs(x,h)         - 0)     * (0 + fb.diff(x).subs(x,0)) / 2 \
                  - (fa.diff(x).subs(x,h) + 0) / 2 * (0 - fb.subs(x,0)        )     \
                  + d / h * (fa.subs(x,h) - 0) * (0 - fb.subs(x,0))

A = sy.Matrix(sy.zeros((p + 1)*n,(p + 1)*n))
for b in range(n):
    for i in range((p + 1)):
        for j in range((p + 1)):
            if (b==0): A[(p + 1)*b+i,(p + 1)*b+j] = CM0[i,j]
            elif (b==n-1): A[(p + 1)*b+i,(p + 1)*b+j] = CM1[i,j]
            else: A[(p + 1)*b+i,(p + 1)*b+j] = CM[i,j]
            if (b != n - 1): A[(p + 1)*b+i,(p + 1)*b+j+(p + 1)] = FM[i,j]
            if (b != 0): A[(p + 1)*b+i,(p + 1)*b+j-(p + 1)] = FM.transpose()[i,j]

for i in range(n*(p + 1)):
    for j in range(n*(p + 1)):
        A[i,j] = sy.simplify(sy.limit(A[i,j],e,sy.oo))

        
Lp = A.subs({h:Dom/n,d:p*(p+1)})
Lp = np.matrix(Lp).astype(float)

g = sy.zeros(1,(p + 1)*n).transpose()
f = sy.zeros(1,(p + 1)*n).transpose()

for b in range(n):
    for i in range(p + 1):
        g[(p + 1)*b + i] = Dom / n * (b + i / p) 
        f[(p + 1)*b + i] = sy.integrate(Fx[i],(x,0,h)).subs({h:Dom/n})
        # f[(p + 1)*b + i] = sy.integrate(Fx[i]*(x+b*h),(x,0,h)).subs({h:Dom/n})

mp.grid(which='major', axis='x', linewidth=0.75, linestyle='-', color='0.75')
mp.grid(which='minor', axis='x', linewidth=0.25, linestyle='-', color='0.75')
mp.grid(which='major', axis='y', linewidth=0.75, linestyle='-', color='0.75')
mp.grid(which='minor', axis='y', linewidth=0.25, linestyle='-', color='0.75')
mp.xlabel("x")
mp.ylabel("u(x)")
x = np.linalg.pinv(Lp) * f
mp.plot(g,(np.array(x)).transpose()[0])
mp.show()

D = sy.Matrix(sy.zeros((p + 1)*n,(p + 1)*n))
for b in range(n):
    for i in range((p + 1)):
        for j in range((p + 1)):
            D[(p + 1)*b+i,(p + 1)*b+j] = A[(p + 1)*b+i,(p + 1)*b+j]

# s = int((p + 1)/2)
# D = sy.Matrix(sy.zeros((p + 1)*n,(p + 1)*n))
# for i in range(s):
#     for j in range(s):
#         D[i,j] = Lp[i,j]
#         D[(p + 1)*n-1-i,(p + 1)*n-1-j] = Lp[(p + 1)*n-1-i,(p + 1)*n-1-j]
# for b in range(1,n):
#     for i in range((p + 1)):
#         for j in range((p + 1)):
#             D[(p + 1)*b+i-s,(p + 1)*b+j-s] = Lp[(p + 1)*b+i-s,(p + 1)*b+j-s]

RT = sy.Matrix(sy.zeros((p + 1)*n,((p + 1)*n)/2))
for k in range(0,int(((p + 1)*n)/2),p + 1):
    avg = False 
    j = k
    for i in range(2*k,2*k+(p + 1),1):
        if avg:
            RT[i,j] = sy.S("1/2")
            RT[i,j+1] = sy.S("1/2")
            j += 1
            avg = False
        else:
            RT[i,j] = sy.S("1")
            avg = True
    if avg:
        avg = False
    else:
        avg = True
        j -= 1
    for i in range(2*k+(p + 1),2*k+2*(p + 1),1):
        if avg:
            RT[i,j] = sy.S("1/2")
            RT[i,j+1] = sy.S("1/2")
            j += 1
            avg = False
        else:
            RT[i,j] = sy.S("1")
            avg = True
R = RT.transpose()

dd = float(p*(p + 1))
hh = 1./float(n)
A = A.subs({d:dd,h:hh})
D = D.subs({d:dd,h:hh})

x = sy.zeros((p + 1)*n,1)
it = 0

mesh = np.array(sy.zeros(1,(p + 1)*n))[0]
g = sy.zeros((p + 1)*n,1)

norm0 = np.linalg.norm(f-A*x)
while (True):
    norm = np.linalg.norm(f-A*x)
    print("\rN:"+str(n)+"\t"+"p:"+str(p)+"\t"+ "Iteration: "+str(it)+"\t"+
          str('%.3e' % norm0)+" -> "+str('%.3E' % norm)+" ",end='')
    if (norm/norm0 < 1.E-8):
        print('')
        break
    ++it
    g = f-A*x

    x0 = sy.zeros((p + 1)*n,1)
    q0 = sy.zeros((p + 1)*n,1)
    x1 = x0 + D**(-1) * (g - A*x0)
    
    print(int(n),end='')
    print(" \u2198 ",end='')
    q1 = q0 + RT * ( R * A * RT )**(-1) * R * (g - A * x1)
    print(" "+str(int(n/2))+" ",end='')
    y1 = x1 + q1
    print(" \u2197 ",end='')
    y2 = y1 + D**(-1) * (g - A*y1)
    print(int(n),end='')

    it+=1
    x += y2
