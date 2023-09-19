#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 16:10:32 2021

@author: chetevidaljustamante
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# =============================================================================
# RESOLUCIÓN DE LA ECUACIÓN DE ADVECCIÓN POR LAX-WENDROFF
# =============================================================================

'''
xi = 0
xf = 10
Nx = 101
dx = (xf-xi)/Nx
Nt = 1000

c = 1
beta=0.8
dt = beta*dx/c
#dt = 0.1
beta = (c**2*dt**2)/(2*dx**2)
alpha= (c*dt)/(2*dx)


x = np.linspace(0,10,Nx)

u = np.zeros((Nt,Nx))
u[0] = [np.exp(-10*(i-1)**2) for i in x]


for k in range(0,Nt-1):
    u[k+1,1:-1] = u[k,1:-1]-alpha*(u[k,2:]-u[k,:-2])+beta*(u[k,2:]-2*u[k,1:-1]+u[k,:-2])


# la animación 

fig = plt.figure()
ax = plt.axes(xlim=(0,xf),ylim=(-1.5,1.5))
line, = ax.plot([],[])  #la coma es para que plot no devuelva una tupla

x = np.linspace(0,10,Nx)
def animate(k):
    line.set_data(x,u[k,:])
    
ani = FuncAnimation(fig,animate,frames=Nt+1, interval=0.000000001)
plt.show()
'''



# =============================================================================
# ECUACIÓN DE BURGERS
# =============================================================================
'''
L = 25
Nx = 100
Nt = 100 
c = 1
x = np.linspace(0,25,Nx)
beta = 0.1

u = np.zeros((Nt,Nx))
u[0] = [3*np.sin((2*np.pi*i)/L) for i in x]

for k in range(0,Nt-1):
    a1 = u[k,1:-1]-beta/4*(u[k,2:]**2-u[k,:-2]**2)
    a2 = beta**2/8*((u[k,2:]+u[k,1:-1])*(u[k,2:]**2-u[k,1:-1]**2)-(u[k,1:-1]+u[k,:-2])*(u[k,1:-1]**2-u[k,:-2]**2))
    u[k+1,1:-1] = a1+a2
    
# la animación 
fig = plt.figure()
ax = plt.axes(xlim=(0,L),ylim=(-3,3))
line, = ax.plot([],[])  #la coma es para que plot no devuelva una tupla

def animate(k):
    line.set_data(x,u[k,:])
    
ani = FuncAnimation(fig,animate,frames=Nt+1, interval=0.000000001)
plt.show()

'''

