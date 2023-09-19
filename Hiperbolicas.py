#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 15:09:40 2021

@author: chetevidaljustamante
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.linalg import solve, lstsq
from scipy.sparse.linalg import spsolve
from scipy import sparse


# =============================================================================
# FDTD (EXPLÍCITO)
# =============================================================================

'''
Nt = 4000
Nx = 100
L = 100.
T = 40.
rho= 0.01
c = np.sqrt(T/rho)
dx = L/Nx
dt = 5e-3
tfin=dt*Nt
a = (c**2*dt**2/dx**2)

x = np.linspace(0,L,Nx+1)
P = np.zeros((Nt+1,Nx+1))

#condiciones iniciales
for i in range(0,Nx):
    if x[i]<=0.8*L:
        P[0,i] = 1.25*x[i]/L
    else:
        P[0,i] = 5.-5.*x[i]/L
P[1,:]=P[0,:]


if c>dx/dt: print('No se cumple la condición de Estabilidad')

#el método
def FDTD(u):
    for k in range(1,Nt):
        temp = u[k-1,1:-1]-2*u[k,1:-1]
        esp = u[k,2:]+u[k,:-2]-2*u[k,1:-1]
        u[k+1,1:-1] = a*esp-temp
    return u

U = FDTD(P)


#graficacion de la animacion
fig = plt.figure()

ax = plt.axes(xlim=(0,L), ylim=(-np.max(U),np.max(U)))
line, = ax.plot([], [], lw=2)

def animate(k):
    line.set_data(x,U[k,:])
    return line,

ani = FuncAnimation(fig,animate, blit=False,
                              interval=1)
plt.show()
'''

# =============================================================================
# MÉTODO IMPLÍCITO
# =============================================================================
'''
#mismo ejemplo que FDTD
Nt = 4000
N = 100
L = 100.
T = 40.
rho= 0.01
c = np.sqrt(T/rho)
dx = L/N
dt = 5e-3
tfin=dt*Nt
a = (c**2*dt**2/dx**2)
x = np.linspace(0,L,N+1)
P = np.zeros((Nt+1,N+1))

#condiciones iniciales
for i in range(0,N):
    if x[i]<=0.8*L:
        P[0,i] = 1.25*x[i]/L
    else:
        P[0,i] = 5.-5.*x[i]/L
P[1,:]=P[0,:]


#definimos las matrices de coeficientes
a = (dt**2*c**2)/(2*dx**2)
A = (1+2*a)*np.eye(N,N)-a*np.eye(N,N,k=1)-a*np.eye(N,N,k=-1)
A[0,0], A[-1,-1] = 1+a, 1+a
B = -(1+2*a)*np.eye(N,N)+a*np.eye(N,N,k=1)+a*np.eye(N,N,k=-1)
B[0,0], B[-1,-1] = -1-a, -1-a
C = 2*np.eye(N,N)
C[0,0], C[-1,-1] = 1, 1

#el metodo que hace las operaciones matriciales
def crank(A,B,C,u):
    for k in range(1,Nt):
        sol = spsolve(A,np.dot(B,u[k-1,:])+np.dot(C,u[k,:]))
        u[k+1,0] = sol
        u[k+1,0],T[k+1,int(N*dx),]=0,0
    return u

U = crank(A,B,C,P)
'''

# =============================================================================
# ONDAS AMORTIGUADAS
# =============================================================================

'''
Nx=100
Nt=4000
L = 1
T = 40
kap = 0.001
rho= 0.01
c = np.sqrt(T/rho)
dx = L/Nx
dt = 5e-3
tfin=dt*Nt

x = np.linspace(0,L,Nx+1)
P = np.zeros((Nt+1,Nx+1))

for i in range(0,Nx):
    if x[i]<=0.8*L:
        P[0,i] = 1.25*x[i]/L
    else:
        P[0,i] = 5.-5.*x[i]/L
P[1,:]=P[0,:]

def FDTD_amort(u):
    for k in range(1,Nt):
        nu = (1/dt**2 + (2*kap)/(dt*rho))
        temp = -u[k-1,1:-1]+2*u[k,1:-1]
        esp = u[k,2:]+u[k,:-2]-2*u[k,1:-1]
        raro = u[k,1:-1]*2*kap/(dt*rho)
        u[k+1,1:-1]=(1/nu)*(esp/dx**2-raro+temp/dt**2)
    return u

U = FDTD_amort(P)

fig = plt.figure()

ax = plt.axes(xlim=(0,L), ylim=(-np.max(U),np.max(U)))
line, = ax.plot([], [], lw=2)

def animate(k):
    line.set_data(x,U[k,:])
    return line,

ani = FuncAnimation(fig,animate, blit=False,
                              interval=1)
plt.show()

'''

# =============================================================================
# ECUACIÓN DEL TELÉGRAFO
# =============================================================================

'''
L = 1.
Nx = 200
Nt = 4000  
dx = L/Nx
dt = 5e-3
tini = 0.    
tfin = dt*Nt
x = np.linspace(0,L,Nx+1)


# inicializo la onda con condiciones iniciales
u = np.zeros((Nt+1,Nx+1))
u[0,:] = np.sin(4*np.pi*x)
u[0,2:] = u[0,1:-1]
u[1,:] = u[0,:]


#metodo FDTD para la ecuación del telégrafo
def teleg(u):
    nu = (1/dt**2+1/dt)
    for k in range(1,Nt-1):
        term1 = u[k,2:]+u[k,:-2]-2*u[k,1:-1]
        term2 = u[k,1:-1]*(1/dt-2)
        term3 = -u[k-1,1:-1]+2*u[k,1:-1]
        u[k+1,1:-1] = 1/nu*(term1/dx**2+term2+term3/dt**2)
    return u

u = teleg(u)

#graficamos la animación
fig = plt.figure()
ax = plt.axes(xlim=(0,L), ylim=(-1,1))
line, = ax.plot([], [], lw=2)

def animate(k):
    line.set_data(x,u[k,:])
    return line,

ani = FuncAnimation(fig,animate, blit=False,
                              interval=1)

plt.show()

'''




