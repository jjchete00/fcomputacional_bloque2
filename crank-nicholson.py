#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 12:37:57 2021

@author: chetevidaljustamante
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from matplotlib.animation import FuncAnimation




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



'''CRANCK-NICHOLSON'''

N = 40 #puntos de la barra
Nt = 1000
Namp = N+2
T = np.zeros([Nt,N])

ti = 0
tf = 1000. #segundos
dt = tf/Nt

L = 1
dx = L/N
dx = 0.01

a = 210./(900.*2700.)

#condiciones iniciales
T[0,1:-1]=373



#construyo matriz de coeficientes=============================================

r = a*dt/(2*dx**2)
A = (1+2*r)*np.eye(N,N) - r*np.eye(N,N,k=-1) - r*np.eye(N,N,k=1)
Ab = (1-2*r)*np.eye(N,N) + r*np.eye(N,N,k=-1) + r*np.eye(N,N,k=1)

CC = 0
T = crankie(A,Ab,CC,T,Nt)


#ploteo la animación==========================================================
x = np.linspace(0,1,N)
t = np.linspace(0,tf,Nt+1)


fig = plt.figure()
ax = plt.axes(xlim=(0,1),ylim=(0,400))
line, = ax.plot([],[])  #la coma es para que plot no devuelva una tupla

def animate(k):
    line.set_data(x,T[k,:])
    
ani = FuncAnimation(fig,animate,frames=Nt+1, interval=0.000000001)
plt.show()




