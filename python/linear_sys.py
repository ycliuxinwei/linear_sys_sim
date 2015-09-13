# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 11:38:00 2015

@author: Edmund Liu
License: BSD license

reference:

'numpy'
http://wiki.scipy.org/Tentative_NumPy_Tutorial

'scipy - Linear Algebra'
http://docs.scipy.org/doc/scipy/reference/tutorial/linalg.html

'scipy - Signal processing'
http://docs.scipy.org/doc/scipy-dev/reference/signal.html

'matplotlib'
http://matplotlib.org/1.4.3/users/index.html

'PyLMI-SDP'
https://github.com/cdsousa/PyLMI-SDP
"""

import numpy as np
from scipy import linalg as lin
from scipy import signal as sgn
import matplotlib.pyplot as plt

""" necessory functions
obsv(A, C) - Observability of pair (A, C)
ctrb(A, B) - Controllabilty of pair (A, B)
"""
# Observability of pair (A, C)
def obsv(A, C):
    amat = np.mat(A)
    cmat = np.mat(C)
    n = np.shape(amat)[0]
    # Construct the controllability matrix
    obsv = cmat
    for i in range(1, n):
        obsv = np.vstack((obsv, cmat*amat**i))
    observability = np.linalg.matrix_rank(obsv.getA())
    return observability

# Controllabilty of pair (A, B)
def ctrb(A, B):
    amat = np.mat(A)
    bmat = np.mat(B)
    n = np.shape(amat)[0]
    # Construct the controllability matrix
    ctrb = bmat
    for i in range(1, n):
        ctrb = np.hstack((ctrb, amat**i*bmat))
    controllability = np.linalg.matrix_rank(ctrb.getA())
    return controllability

"""
self-functions
"""
# system function
def Sys(x, u, other, A, B):
    x = A.dot(x) + B.dot(u) + other
    return x

# Runge-Kutta method
def RK(x, u, other, A, B, h):
    K1 = Sys(x, u, other, A, B)
    K2 = Sys(x + h*K1/2, u, other, A, B)
    K3 = Sys(x + h*K2/2, u, other, A, B)
    K4 = Sys(x + h*K3, u, other, A, B)
    x1 = x + (K1 + 2*K2 + 2*K3 + K4)*h/6
    return x1

#  parameters
Ml = 0.5; ml = 0.2
bl = 0.1; Il = 0.006
gl = 9.8; ll = 0.3
pl = Il*(Ml+ml)+Ml*ml*ll**2

#  Inverted Pendulum System
#  x' = Ax+Bu
#  y  = Cx+Du
A = np.array([[0, 1.0, 0, 0],
               [0, -(Il+ml*ll**2)*bl/pl, (ml**2*gl*ll**2)/pl, 0],
               [0, 0, 0, 1.0],
               [0, -(ml*ll*bl)/pl, ml*gl*ll*(Ml+ml)/pl, 0]])
B = np.array([[0], [(Il+ml*ll**2)/pl], [0], [ml*ll/pl]])
C = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])

# initial state
x0 = np.array([[0.98], [0], [0.2], [0]])

# size of A and B
n1 = np.size(A,0)
n2 = np.size(B,1)

# Solve the continuous time lqr controller
# to get controller gain 'K'
# J = x'Qx + u'Ru
# A'X + XA - XBR^(-1)B'X + Q = 0
# K = inv(R)B'X
Q = C.T.dot(C)
R = np.eye(n2)
X = np.matrix(lin.solve_continuous_are(A, B, Q, R))
K = -np.matrix(lin.inv(R)*(B.T*X)).getA()

# observability
ob = obsv(A, C)
print "observability =", ob

# controllability
cb = ctrb(A, B)
print "controllability =", cb

# get the observer gain, using pole placement
p = np.array([-13, -12, -11, -10])
fsf1 = sgn.place_poles(A.T, C.T, p)
L = fsf1.gain_matrix.T

x = x0; xhat = 0*x0
h = 0.01; ts = 1000;
X = np.array([[],[],[],[]])
Y = np.array([[],[]])
Xhat = X; t = []

for k in range(ts):
    t.append(k*h)
    u = K.dot(xhat)
    y = C.dot(x)
    yhat = C.dot(xhat)
    X = np.hstack((X, x))
    Xhat = np.hstack((Xhat, xhat))
    Y = np.hstack((Y, y))
    x = RK(x, u, np.zeros([n1,1]), A, B, h)
    xhat = RK(xhat, u, L.dot((y-yhat)), A, B, h)

# x_1
fig1 = plt.figure('state x_1')
ax1 = fig1.add_subplot(111)
ax1.plot(t, X[0], linewidth = 2)
ax1.plot(t, Xhat[0], '--', linewidth = 2)
ax1.legend(('$x_1$', '$\hat{x}_1$'))
ax1.grid(True)
ax1.set_xlabel('Time(s)')
ax1.set_ylabel('States')

# x_2
fig2 = plt.figure('state x_2')
ax2 = fig2.add_subplot(111)
ax2.plot(t, X[1], linewidth = 2)
ax2.plot(t, Xhat[1], '--', linewidth = 2)
ax2.legend(('$x_2$', '$\hat{x}_2$'))
ax2.grid(True)
ax2.set_xlabel('Time(s)')
ax2.set_ylabel('States')

# x_3
fig3 = plt.figure('state x_3')
ax3 = fig3.add_subplot(111)
ax3.plot(t, X[2], linewidth = 2)
ax3.plot(t, Xhat[2], '--', linewidth = 2)
ax3.legend(('$x_3$', '$\hat{x}_3$'))
ax3.grid(True)
ax3.set_xlabel('Time(s)')
ax3.set_ylabel('States')

# x_4
fig4 = plt.figure('state x_4')
ax4 = fig4.add_subplot(111)
ax4.plot(t, X[3], linewidth = 2)
ax4.plot(t, Xhat[3], '--', linewidth = 2)
ax4.legend(('$x_4$', '$\hat{x}_4$'))
ax4.grid(True)
ax4.set_xlabel('Time(s)')
ax4.set_ylabel('States')

# y
fig5 = plt.figure('outputs y')
ax5 = fig5.add_subplot(111)
ax5.plot(t, Y[0], linewidth = 2)
ax5.plot(t, Y[1], linewidth = 2)
ax5.legend(('$y_0$', '$y_1$'))
ax5.grid(True)
ax5.set_xlabel('Time(s)')
ax5.set_ylabel('Outputs')

plt.show()