#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Numerical methods module """
# Colection of classic numerical methods in Python(3.5)
# Author: Martin Noblía

#**********************************************************************
#imports

import numpy as np
#**********************************************************************
#Function forward
#**********************************************************************


def forward_subs(L, b):

    """
    Function for computing the forward substitution

    Inputs:
    -------
    L: lower triangular matrix
    b: input vector

    Output:
    -------
    x: Solution of the system

    """
    n = np.size(L, 1)
    x = np.zeros_like(b, float)

    x[0] = b[0] / L[0, 0]

    for i in range(1, n):

        x[i] = (b[i] - (L[i, 0:i] @ x[0:i])) / L[i, i]  #the dot product operator

    return x

#**********************************************************************
# Function backward
#**********************************************************************


def backward_subs(U, b):
    """
    Function for computing the backward substitution

    Inputs:
    -------
    U: upper triangular matrix
    b: input vector

    Output:
    -------
    x: Solution of the system
    """
    n = np.size(U, 1)
    x = np.zeros_like(b, float)


    for k in range(n-1, -1, -1):

        x[k] = (b[k] - (U[k, k+1:n] @ x[k+1:n])) / U[k, k] # the dot product @

    return x

#**********************************************************************
# LU Factorization
#**********************************************************************


def lu_fact(A):
    """
    Function for computing the LU factorization
    Inputs:
    -------
    A: Matrix of the system

    Output:
    -------
    L: Lower triangular matrix
    U: Upper triangular matrix

    """
    n = np.size(A, 1)
    L = np.zeros_like(A, float)
    U = np.zeros_like(A, float)

    for k in range(0, n-1):
        for i in range(k+1, n):
            if A[i, k] != 0.0:
                lam = A[i, k] / A[k, k]
                A[i, k+1:n] = A[i, k+1:n] - lam*A[k, k+1:n]
                A[i, k] = lam

    L = np.tril(A, -1) + np.identity(n)
    U = np.triu(A)
    return L, U

#**********************************************************************
# Cholesky factorization
#**********************************************************************

def cholesky_fact(a):

    n = len(a)
    for k in range(n):
        try:
            a[k, k] = sqrt(a[k, k] - dot(a[k, 0:k], a[k, 0:k]))
        except ValueError:
            error.err('La matriz no es definida positiva')
        for i in range(k+1, n):
            a[i, k] = (a[i, k] - dot(a[i, 0: k], a[k, 0:k]))/a[k, k]
    for k in range(1, n):
        a[0:k, k] = 0.0
    return a
#**********************************************************************
# Iterative methods
#**********************************************************************
#Gauss-Seidel


def gauss_seidel(A, b, x_0, omega, tol, n_max):
    k = 0
    #p=1
    n = np.size(A, 1)
    x = np.zeros_like(x_0)

    r = b-np.dot(A, x_0)
    err = np.norm(r)
    r_0 = np.norm(r)
    x_anterior = x_0
    s = lu_fact(a, b)

    while(err > tol)and(k < n_max):

        k = k+1
        for i in xrange(0, n):

            s = 0
            for j in xrange(0, i-1):
                s = s+A[i, j] * x[j]
            for j in xrange(i+1, n):

                s = s+A[i, j]*x_anterior[j]

            x[i] = omega * (b[i]-s)/A[i, i]+(1+omega) * x_anterior[i]

        r = b-np.dot(A, x)
        err = np.norm(r)/r_0
        x_anterior = x

    return x
