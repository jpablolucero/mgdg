import math
import numpy as np
import sympy as sy
import matplotlib.pyplot as mp
np.set_printoptions(precision=3,threshold=np.nan,linewidth=np.inf,suppress=True)

x,h,d,e,w,w0,a,b,c,f,g,p = sy.symbols('x h d e w w0 a b c f g p')

psi = sy.Function('psi')
phi = sy.Function('phi')

psi = x / h 
phi = 1 - x / h

CM = sy.Matrix(sy.zeros(2,2))

#(phi,phi)
CM[0,0]= sy.integrate(phi.diff(x)*phi.diff(x),(x,0,h)) \
+ 1/e * sy.integrate(phi*phi,(x,0,h)) \
- (0 - phi.subs(x,0)) * (phi.diff(x).subs(x,h) + 0) / 2 \
- (0 - phi.subs(x,0)) * (phi.diff(x).subs(x,h) + 0) / 2 \
+ d/h * (0 - psi.subs(x,h)) * (0 - psi.subs(x,h))

#(psi,phi)
CM[0,1] = sy.integrate(phi.diff(x)*psi.diff(x),(x,0,h)) \
+ 1/e * sy.integrate(phi*psi,(x,0,h)) \
- (0 - phi.subs(x,0)) * (0 + psi.diff(x).subs(x,0)) / 2 \
- (psi.subs(x,h) - 0) * (phi.diff(x).subs(x,h) + 0) / 2 \
+ d/h * (0 - phi.subs(x,0)) * (0 - psi.subs(x,0))

#(phi,psi)
CM[1,0]= CM[0,1] 

#(psi,psi)
CM[1,1]= sy.integrate(psi.diff(x)*psi.diff(x),(x,0,h)) \
+ 1/e * sy.integrate(psi*psi,(x,0,h)) \
- (psi.subs(x,h) - 0) * (psi.diff(x).subs(x,h) + 0) / 2 \
- (psi.subs(x,h) - 0) * (psi.diff(x).subs(x,h) + 0) / 2 \
+ d/h * (psi.subs(x,h) - 0) * (psi.subs(x,h) - 0)

FM = sy.Matrix(sy.zeros(2,2))

#(phi,phi)
FM[0,0] = - (0 - phi.subs(x,0)) * (phi.diff(x).subs(x,0) + 0) / 2 

#(psi,phi)
FM[0,1] = 0

#(phi,psi)
FM[1,0] = - (0 - phi.subs(x,0)) * (psi.diff(x).subs(x,h) + 0) / 2 \
- (psi.subs(x,h) - 0) * (0 + phi.diff(x).subs(x,h)) / 2 \
+ d/h * (0 - phi.subs(x,0)) * (psi.subs(x,h) - 0)

#(psi,psi)
FM[1,1] = - (psi.subs(x,h) - 0) * (psi.diff(x).subs(x,0) + 0) / 2

Dom = 1.
n = 4

A = sy.Matrix(sy.zeros(2*n,2*n))
for b in range(n):
    for i in range(2):
        for j in range(2):
            A[2*b+i,2*b+j] = CM[i,j]
            if (b != n-1): A[2*b+i,2*b+j+2] = FM[i,j]
            if (b != 0): A[2*b+i,2*b+j-2] = FM.transpose()[i,j]

A = A.subs({e:np.inf})

CM = np.matrix(CM.subs({h:Dom/n,e:np.inf,d:1.})).astype(float)
FM = np.matrix(FM.subs({h:Dom/n,e:np.inf,d:1.})).astype(float)
Lp = np.matrix(sy.zeros(2*n,2*n)).astype(float)
g = np.array(sy.zeros(1,2*n))[0]
for b in range(n):
    for i in range(2):
        for j in range(2):
            Lp[2*b+i,2*b+j] = CM[i,j]
            if (b != n-1): Lp[2*b+i,2*b+j+2] = FM[i,j]
            if (b != 0): Lp[2*b+i,2*b+j-2] = FM.transpose()[i,j]
    g[2*b] = Dom / n * b
    g[2*b+1] = Dom / n * (b + 1)
f = sy.ones(1,2*n).transpose() / n / 2.
mp.grid(which='major', axis='x', linewidth=0.75, linestyle='-', color='0.75')
mp.grid(which='minor', axis='x', linewidth=0.25, linestyle='-', color='0.75')
mp.grid(which='major', axis='y', linewidth=0.75, linestyle='-', color='0.75')
mp.grid(which='minor', axis='y', linewidth=0.25, linestyle='-', color='0.75')
mp.xlabel("x")
mp.ylabel("u(x)")
mp.plot(g,(np.array(np.linalg.inv(Lp) * f)).transpose()[0])
mp.show()

D = sy.Matrix(sy.zeros(2*n,2*n))
for b in range(n):
    for i in range(2):
        for j in range(2):
            D[2*b+i,2*b+j] = A[i,j]

D0 = sy.Matrix(sy.zeros(2*n,2*n))
D0[0,0] = A[0,0]
for b in range(n-1):
    for i in range(2):
        for j in range(2):
            D0[2*b+i+1,2*b+j+1] = A[i+1,j+1]
D0[2*n-1,2*n-1] = A[2*n-1,2*n-1]

RT = sy.Matrix(sy.zeros(2*n,n))
RT[0,0]=1
RT[1,0]=sy.S("1/2")
RT[1,1]=sy.S("1/2")
RT[2,0]=sy.S("1/2")
RT[2,1]=sy.S("1/2")
i = 1
for j in range(3,2*n-3,4):
    RT[j,i]=1
    RT[j+1,i+1]=1
    if (j<2*n+3):
        RT[j+2,i+1]=sy.S("1/2")
        RT[j+2,i+2]=sy.S("1/2")
        RT[j+3,i+1]=sy.S("1/2")
        RT[j+3,i+2]=sy.S("1/2")
    i+=2
    RT[2*n-1,n-1]=1
R = RT.transpose();

A0 = sy.Matrix(sy.zeros(n,n))
for b in range(int(n/2)):
    for i in range(2):
        for j in range(2):
            A0[2*b+i,2*b+j] = CM[i,j]
            if (b != n/2-1): A0[2*b+i,2*b+j+2] = FM[i,j]
            if (b != 0): A0[2*b+i,2*b+j-2] = FM.transpose()[i,j]
A0 = A0.subs({e:np.inf}).subs({h:2*h})

dd = 1.2
hh = 1./float(n)
nn = 20
A = A.subs({d:dd,h:hh})
A0 = A0.subs({d:dd,h:hh})
D = D.subs({d:dd,h:hh})

f = sy.ones(2*n,1)/(2*n)
x = sy.zeros(2*n,1)
it = 0

mesh = np.array(sy.zeros(1,2*n))[0]
g = sy.zeros(2*n,1)

norm0 = np.linalg.norm(f-A*x)
while (True):
    norm = np.linalg.norm(f-A*x)
    print("\rN:"+str(n)+"\t"+"Iteration: "+str(it)+"\t"+
          str('%.3e' % norm0)+" -> "+str('%.3E' % norm)+" ",end='')
    if (norm/norm0 < 1.E-8):
        print('')
        break
    ++it
    g = f-A*x
    x0 = sy.zeros(2*n,1)
    q0 = sy.zeros(2*n,1)
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

C = A * RT * ( R * A * RT )**(-1) * R

L = sy.Matrix(sy.zeros(2*n,2*n))
U = sy.Matrix(sy.zeros(2*n,2*n))
for b in range(n-1):
    for i in range(2):
        for j in range(2):
            L[2*(b+1)+i,2*b+j] = A[2+i,j]
            U[2*b+i,2*(b+1)+j] = A[i,2+j]

def swap(mat,i,j):
    v = mat[:,i]
    mat[:,i] = mat[:,j]
    mat[:,j] = v
    v = mat[i,:]
    mat[i,:] = mat[j,:]
    mat[j,:] = v
    
for i in range(n):
    for j in range(n-i):
        swap(A,2*j+i,2*j+i+1)
        swap(L,2*j+i,2*j+i+1)
        swap(D,2*j+i,2*j+i+1)
        swap(U,2*j+i,2*j+i+1)
        swap(C,2*j+i,2*j+i+1)

      
import matplotlib.mlab as mlab
def numerical_range(A,resolution=0.01):
    # Function implements algorithm for calculation of numerical range
    # http://www.math.iupui.edu/~ccowen/Downloads/33NumRange.html
    A=np.asmatrix(A)
    th=np.arange(0,2*np.pi+resolution,resolution)
    k=0
    w=[]
    for j in th:
        Ath=np.exp(1j*-j)*A
        Hth=(Ath+Ath.H)/2.
        Hth=np.array(Hth).astype(complex)
        e,r=np.linalg.eigh(Hth)
        r=np.matrix(r)
        e=np.real(e)
        m=e.max()
        s=mlab.find(e==m)
        if np.size(s)==1:
            w.append(np.matrix.item(r[:,s].H*A*r[:,s]))
        else:
            Kth=1j*(Hth-Ath)
            pKp=r[:,s].H*Kth*r[:,s]
            ee,rr=np.linalg.eigh(pKp)
            rr=np.matrix(rr)
            ee=np.real(ee)
            mm=min(ee)
            sm=mlab.find(ee==mm)
            temp=rr[:,sm[0]].H*r[:,s].H*A*r[:,s]*rr[:,sm[0]]
            w.append(temp[0,0])
            k+=1
            mM=ee.max()
            sM=mlab.find(ee==mM)
            temp=rr[:,sM[0]].H*r[:,s].H*A*r[:,s]*rr[:,sM[0]]
            w.append(temp[0,0])
        k+=1
    return w

Ac = sy.Matrix(sy.zeros(2,2))
Cc = sy.Matrix(sy.zeros(2,2))
Lc = sy.Matrix(sy.zeros(2,2))
Dc = sy.Matrix(sy.zeros(2,2))
Uc = sy.Matrix(sy.zeros(2,2))

u = sy.Matrix([[sy.exp(-sy.I*w*h)]   ,              [1], [sy.exp(sy.I*w*h)], [sy.exp(2*sy.I*w*h)]])
v = sy.Matrix([[sy.exp(-2*sy.I*w*h)] , [sy.exp(-sy.I*w*h)],             [1],   [sy.exp(sy.I*w*h)]])

u = u.subs({h:hh})
v = v.subs({h:hh})

Ac[0,0] = (A[1,0:4]*u)[0,0]
Ac[0,1] = (A[1,4:8]*v)[0,0]
Ac[1,0] = (A[6,0:4]*u)[0,0]
Ac[1,1] = (A[6,4:8]*v)[0,0]

Cc[0,0] = (C[1,0:4]*u)[0,0]
Cc[0,1] = (C[1,4:8]*v)[0,0]
Cc[1,0] = (C[6,0:4]*u)[0,0]
Cc[1,1] = (C[6,4:8]*v)[0,0]

Lc[0,0] = (L[1,0:4]*u)[0,0]
Lc[0,1] = (L[1,4:8]*v)[0,0]
Lc[1,0] = (L[6,0:4]*u)[0,0]
Lc[1,1] = (L[6,4:8]*v)[0,0]

Dc[0,0] = (D[1,0:4]*u)[0,0]
Dc[0,1] = (D[1,4:8]*v)[0,0]
Dc[1,0] = (D[6,0:4]*u)[0,0]
Dc[1,1] = (D[6,4:8]*v)[0,0]

Uc[0,0] = (U[1,0:4]*u)[0,0]
Uc[0,1] = (U[1,4:8]*v)[0,0]
Uc[1,0] = (U[6,0:4]*u)[0,0]
Uc[1,1] = (U[6,4:8]*v)[0,0]

Itm = (sy.eye(2) - Ac * Dc**(-1)) * (sy.eye(2)-Cc)

pl = mp.figure()
l1 = list(Itm.eigenvals().keys())[0]
if (len(list(Itm.eigenvals().keys())) > 1):
    l2 = list(Itm.eigenvals().keys())[1]
else:
    l2 = list(Itm.eigenvals().keys())[0]
gr = np.array(sy.zeros(1,nn+1))[0]
eig1 = np.array(sy.zeros(1,nn+1))[0]
eig2 = np.array(sy.zeros(1,nn+1))[0]
for i in range(nn+1):
    gr[i] = i * math.pi/nn
    eig1[i] = max(np.real(complex(l1.subs({w:math.pi*i*(Dom/hh)/nn}))),
                  np.real(complex(l2.subs({w:math.pi*i*(Dom/hh)/nn}))))
    eig2[i] = min(np.real(complex(l1.subs({w:math.pi*i*(Dom/hh)/nn}))),
                  np.real(complex(l2.subs({w:math.pi*i*(Dom/hh)/nn}))))
    mp.plot(gr,eig1,'b.',alpha=1.)
    mp.plot(gr,eig2,'b.',alpha=1.)
mp.xlim(0,math.pi)
mp.ylim(-1,1)
mp.grid(which='major', axis='x', linewidth=0.75, linestyle='-', color='0.75')
mp.grid(which='minor', axis='x', linewidth=0.25, linestyle='-', color='0.75')
mp.grid(which='major', axis='y', linewidth=0.75, linestyle='-', color='0.75')
mp.grid(which='minor', axis='y', linewidth=0.25, linestyle='-', color='0.75')
mp.xlabel(r'$\omega$')
mp.ylabel(r'$\lambda$')
mp.xticks([0,math.pi/4,math.pi/2,3*math.pi/4,math.pi])
mp.title("")    
mp.show()

pl = mp.figure()
from mpl_toolkits.mplot3d import Axes3D
ax = pl.gca(projection='3d')
gr = np.array(sy.zeros(1,nn+1))[0]
eig0 = np.array(sy.zeros(1,nn+1))[0]
eigim0 = np.array(sy.zeros(1,nn+1))[0]
eig1 = np.array(sy.zeros(1,nn+1))[0]
eigim1 = np.array(sy.zeros(1,nn+1))[0]
for i in range(nn+1):
    gr[i] = i * 1./nn * math.pi
    Itm2 = np.matrix(Itm.subs({w:i*(Dom/hh)/nn * np.pi}))
    Itm2 = np.array(Itm2).astype(complex)
    NR = numerical_range(Itm2,0.1)
    ax.plot(np.ones(np.array(np.real(NR)).shape)* i * 1./nn * np.pi,
            np.real(NR),
            np.imag(NR),'r')
    eigenvalues = np.linalg.eig(Itm2)[0]
    eig0[i] = np.real(complex(eigenvalues[0]))
    eigim0[i] = np.imag(complex(eigenvalues[0]))
    eig1[i] = np.real(complex(eigenvalues[1]))
    eigim1[i] = np.imag(complex(eigenvalues[1]))
ax.plot(gr,eig0,eigim0,'b.')
ax.plot(gr,eig1,eigim1,'b.')
ax.view_init(elev=90., azim=-90.)
ax.set_xlabel(r'$\omega$')
ax.set_ylabel(r'$\Re(W)$')
ax.set_zlabel(r'$\Im(W)$')
ax.set_xlim(0,math.pi)
ax.set_ylim(-1.,1.)
ax.set_zlim(-1.,1.)
ax.set_xticks([0,math.pi/4,math.pi/2,3*math.pi/4,math.pi])
mp.title("")    
mp.show()

