#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 17:07:07 2021

@author: chetevidaljustamante
"""
import numpy as np
import matplotlib.pyplot as plt
from numba import jit


def dist(a,b):
    return np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)

@jit
def FTCS(Nx,Ny,T,D,dt,dx,dy):
    
    for k in range(0,Nt):
        for i in range(1,Nx-1):
            for j in range(1,Nx-1):
                dT = D*dt*((T[i+1,j]+T[i-1,j]-2*T[i,j])/dx**2+(T[i,j+1]+T[i,j-1]-2*T[i,j])/dy**2)
                T[i,j] = T[i,j]+ dT
    return T


#parametros
T0 = 300.   #K
T1 = 700.   #K
L = 10.     #mm

dt = 0.001       # el tiempo que tarda es Nt*dt (segundos)
Nt = 600

Nx = 100       # 100 puntos en cada eje
dx = L/Nx      # mm

Ny=Nx
dy = dx

D = 4.0




#construyo T=======================================================
T = np.zeros((Nx,Ny))

for i in range(0,Nx):
    for j in range(0,Nx):
        if np.round(dist([i,j],[Nx/2,Nx/2]),0) <= 20:
            T[i,j]=T1
        else:
            T[i,j]=T0
#construyo T=======================================================

T = FTCS(Nx,Ny,T,D,dt,dx,dy)



#graficamos
plt.figure()

x = np.linspace(0,L,Nx)
y = x
X,Y = np.meshgrid(x, y)

plt.contourf(X, Y, T, 4, alpha=.75, cmap=plt.cm.coolwarm)
C = plt.contour(X, Y, T, 4 , colors='black', linewidth=.5)
plt.clabel(C, inline=1, fontsize=10)
plt.xticks(())
plt.yticks(())
plt.show()













