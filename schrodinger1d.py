#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 12:16:39 2021

@author: chetevidaljustamante
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numba import jit

@jit
def crankie(A,Ab,CC,u,Nt):
    '''
    
    cranck nicholson para (A)U = (Ab)(u) 
    Donde u representa U en el paso anterior

    Parameters
    ----------
    A : matriz de coeficientes a la izquierda
    Ab : matriz de coeficientes a la derecha
    CC : condiciones de contorno
    u : función a diferenciar
    Nt : número de pasos de tiempo

    Returns
    -------
    Un vector de matrices u, con u en cada paso

    '''
    for i in range(1,Nt):
        u[i]=np.linalg.solve(A , np.dot(Ab,u[i-1])+CC)
    return u




# =============================================================================
# SCHRODINGER EN 1D POR CRANK NICHOLSON
# =============================================================================


dx = 0.02
x = np.arange(-4,4,dx)
Nx = len(x)

Nt = 1000
dt = dx**2/2
W = np.zeros([Nt,Nx],dtype=complex)
k = 15*np.pi
sigma = 0.5

for i in range(0,Nx):
    W[0,i] = np.exp((-1/2)*(x[i]/sigma)**2)*np.exp(1j*k*x[i])
    #W[0,i] = np.exp((-1/2)*(x[i]/sigma)**2)

a = 1j
r = a*dt/(2*dx**2)

A = (1+2*r)*np.eye(Nx,Nx,dtype=complex) - r*np.eye(Nx,Nx,k=-1,dtype=complex) - r*np.eye(Nx,Nx,k=1,dtype=complex)
Ab = (1-2*r)*np.eye(Nx,Nx,dtype=complex) + r*np.eye(Nx,Nx,k=-1,dtype=complex) + r*np.eye(Nx,Nx,k=1,dtype=complex)


CC = 0
W = crankie(A,Ab,CC,W,Nt)


#ploteo la animación==========================================================


t = np.linspace(0,Nt,Nt+1)

fig = plt.figure()
ax = plt.axes(xlim=(-4,4),ylim=(-2,2))
line, = ax.plot([],[])  #la coma es para que plot no devuelva una tupla

def animate(k):
    line.set_data(x,W[k,:])

ani = FuncAnimation(fig,animate,frames=Nt+1, interval=1e-3)
plt.show()

















