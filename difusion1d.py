#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 15:03:25 2021

@author: chetevidaljustamante
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import animation
#Parametros
Tf = 0.  #temperatura fria
Tc = 50. #temperatura caliente
Tm = 20.
D = 4.25e-6 #difusividad del acero
L = 0.01 #longitud de la barra

Nx = 100 
dx = L/Nx

tini=0
tfin=4
Nt = 4000 #40000 segun el ejemplo pero son muchas
dt = (tfin-tini)/Nt

c = D*dt/dx**2

x = np.linspace(0,L,Nx+1)
t = np.linspace(tini,tfin,Nt+1)
T = np.zeros((Nx+1,Nt+1)) #temperatura


#condiciones iniciales y de contorno

T[:,0]=Tm
T[0,:]=Tc
T[-1:]=Tf

def FTCS1D(Nx,Nt,T):
    for k in range(1,Nt):
        for i in range(1,Nx):
            T[i,k] = T[i,k-1] + c*(T[i+1,k-1] + T[i-1,k-1] - 2*T[i,k-1])
    return T

T = FTCS1D(Nx,Nt,T)



# la animación 
fig = plt.figure()
ax = plt.axes(xlim=(0,L),ylim=(Tf,Tc))
line, = ax.plot([],[])  #la coma es para que plot no devuelva una tupla

def animate(k):
    line.set_data(x,T[:,k])
    
ani = FuncAnimation(fig,animate,frames=Nt+1, interval=0.000000001)
plt.show()





















