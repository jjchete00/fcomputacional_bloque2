#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 15:20:36 2021

@author: chetevidaljustamante
"""

import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from scipy import sparse

# =============================================================================
# JACOBI Y GAUSS-SEIDEL
# =============================================================================

Nx = 100
Ny = 100
N = Nx*Ny # puntos totales de la malla
V = 100
w = 0.8
Delta = 0.01
Phi = np.zeros([Nx,Ny])
x,y = Delta*np.arange(Nx), Delta*np.arange(Ny)
#aplicamos condiciones de contorno a Phi

#Phi[0,:] = V #condiciones 1

''' EXTRA 1 (CONDICIONES 2)
for i in range(0,Nx):
    for j in range(0,Ny):
        if 0.6<=x[i]<=0.8 and 0.6<=y[j]<=0.8:
            Phi[i,j]=V
        if 0.2<=x[i]<=0.4 and 0.2<=y[j]<=0.4:
            Phi[i,j]=V
'''
# para que se vea el avance en estas condiciones 
# debemos cambiar el error a err=1

def Jacobi(Phi):
    Phip=np.copy(Phi)
    w = 0.8
    err = 10**-6
    sigue= True
    while sigue:
        sigue=False
        for i in range(1,Nx-1):
            for j in range(1,Ny-1):
                Phi_a = Phi[i,j]
                Phip[i,j] = (1+w)/4*(Phi[i+1,j]+Phi[i-1,j]+Phi[i,j+1]+Phi[i,j-1])-w*Phi[i,j]
                Phi=Phip
                dPhi = np.abs(Phi[i,j]-Phi_a)
                if dPhi>err:
                    sigue=True
    return Phi

    
def GaussSeidel(Phi):
    w = 0.8
    err = 10**-6
    sigue= True
    while sigue:
        sigue=False
        for i in range(1,Nx-1):
            for j in range(1,Ny-1):
                Phi_a = Phi[i,j]
                Phi[i,j] = (1+w)/4*(Phi[i+1,j]+Phi[i-1,j]+Phi[i,j+1]+Phi[i,j-1])-w*Phi[i,j]
                dPhi = np.abs(Phi[i,j]-Phi_a)
                if dPhi>err:
                    sigue=True
    return Phi


#Phi = Jacobi(Phi)
#Phi = GaussSeidel(Phi)
#X,Y = np.meshgrid(x,y)
#plt.contourf(X,Y,Phi,alpha=.75,cmap=plt.cm.coolwarm)
#plt.colorbar(label='Voltaje(V)')
#plt.xlabel('m')
#plt.ylabel('m')
#plt.show()


# =============================================================================
# RESOLUCIÓN DIRECTA
# =============================================================================


#construyo la matriz C========================================================

#dimensiones de la malla
N=40
M=40

A = -4*np.eye(N,M)+np.eye(N,M,k=1)+np.eye(N,M,k=-1)
I = np.eye(N,M) #matriz identidad
B = np.zeros((N,M))+np.eye(N,M,k=1)+np.eye(N,M,k=-1)

#Hago los productos tensoriales para obtener la matriz final
C= sparse.kron(A,I).toarray()+sparse.kron(I,B).toarray() #ojo con el orden

#aplico condiciones iniciales=================================================
R1 = np.zeros((N,M))
R1[N-1,M-2] = 1
R1 = sparse.kron(R1,I).toarray()

R2 = np.zeros((N,M))
R2[N-1,M-1]=1
R2b = -5*np.eye(N,M)+np.eye(N,M,k=1)+np.eye(N,M,k=-1)
R2 = sparse.kron(R2,R2b).toarray()

CI=-(R1+R2)
Af = C+CI

Phi = np.zeros(N*M)
Phi[N*(M-1):N*M]=1


#resuelvo el sistema==========================================================

sol1 = np.linalg.solve(Af,Phi)
sol = sol1.reshape((M,N))


#grafico
#x,y = Delta*np.arange(N), Delta*np.arange(M)
#X,Y = np.meshgrid(x,y)
#plt.contourf(X,Y,sol,alpha=.75,cmap=plt.cm.coolwarm)
#plt.colorbar(label='Voltaje(V)')
#plt.xlabel('m')
#plt.ylabel('m')
#plt.show()