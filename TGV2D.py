import numpy as np
from scipy.interpolate import lagrange
import matplotlib.pyplot as plt


def beta(M):
    """
    Generates beta coefficients for Adam-Bashforth integrating scheme
    These coefficients are stored in reversed compared to conventional
    Adam-Bashforth implementations (the first element of beta corresponds to
    earlier point in time).
    input:
    M: the order of Adam-Bashforth scheme
    """
    if M == 2:
        return np.array([-1./2, 3./2])
    elif M == 3:
        return np.array([5./12, -16./12, 23./12])
    elif M == 4:
        return np.array([-9./24, 37./24, -59./24, 55./24])
    elif M == 5:
        return np.array([251./720, -1274./720, 2616./720, -2774./720, 1901./720])
    elif M == 6:
        return np.array([-475./720, 2877./720, -7298./720, 9982./720, -7923./720, 4277./720])


def TaylorVortex2D(T, IC, IC1, N, L, M):

    """
    RIDC-AB solver for 2D Incompressible Navier-Stokes

    Inputs:
    T:  integration interval [0,T]
    IC: initial conditions for vorticity
    IC1: initial conditions for stream function
    N:  number of time nodes
    L: number of spatial nodes in x and y directions
    M: the number of points in calculating quadrature integral
    (and also the number of steps used in Adam-Bashforth predictor)
    or number of correction loops PLUS the prediction loop

    Output:
    yy: vorticity
    stream: stream function
    """

    # time step and spacial step
    h = float(T) / N
    dx = 2 * np.pi / L  # Length of sides of domain are 2pi
    visc = 1. / 2000  # Viscosity, 1/Re

    # Define the semi-discrete system's RHS for the vorticity equation using centred differences, I assume there's a
    # much better way of doing this (maybe define outside of function and use as input?)
    def func(w, psi):
        f = w.copy()
        zeta = w[:, :, 0]

        for i in range(1, L):
            for j in range(1, L):

                # updating inner points
                f[i, j, 0] = -((psi[i, j + 1] - psi[i, j - 1]) * (zeta[i + 1, j] - zeta[i - 1, j])) / (4 * dx ** 2) + \
                             ((psi[i + 1, j] - psi[i - 1, j]) * (zeta[i, j + 1] - zeta[i, j - 1])) / (4 * dx ** 2) + \
                             visc * (zeta[i - 1, j] + zeta[i + 1, j] + zeta[i, j - 1] + zeta[i, j + 1] - 4 * zeta[
                              i, j]) / dx ** 2

            # updating BC
            f[0, i, 0] = -((psi[0, i + 1] - psi[0, i - 1]) * (zeta[1, i] - zeta[-2, i])) / (4 * dx ** 2) + \
                        ((psi[1, i] - psi[-2, i]) * (zeta[0, i + 1] - zeta[0, i - 1])) / (4 * dx ** 2) + \
                         visc * (zeta[1, i] + zeta[-2, i] + zeta[0, i + 1] + zeta[0, i - 1] - 4 * zeta[0, i]) / dx ** 2

            f[i, 0, 0] = -((psi[i, 1] - psi[i, -2]) * (zeta[i + 1, 0] - zeta[i - 1, 0])) / (4 * dx ** 2) + \
                         ((psi[i + 1, 0] - psi[i - 1, 0]) * (zeta[i, 1] - zeta[i, -2])) / (4 * dx ** 2) + \
                         visc * (zeta[i + 1, 0] + zeta[i - 1, 0] + zeta[i, 1] + zeta[i, -2] - 4 * zeta[i, 0]) / dx ** 2
            f[L, i, 0] = f[0, i, 0]
            f[i, L, 0] = f[i, 0, 0]

        # updating corners
        f[0, 0, 0] = -((psi[0, 1] - psi[0, -2]) * (zeta[1, 0] - zeta[-2, 0])) / (4 * dx ** 2) + \
                     ((psi[1, 0] - psi[-2, 0]) * (zeta[0, 1] - zeta[0, -2])) / (4 * dx ** 2) + \
                     visc * (zeta[-2, 0] + zeta[1, 0] + zeta[0, -2] + zeta[0, 1] - 4 * zeta[0, 0]) / dx ** 2
        f[0, L, 0] = f[L, 0, 0] = f[L, L, 0] = f[0, 0, 0]

        return f

    # Solving the poisson equation for the stream function (but not in a good way, replace with spectral method or
    # conjugate gradient)
    def poisson(zeta, psi):
        f = psi.copy()

        for i in range(1, L):
            for j in range(1, L):

                # solving inner points
                f[i, j] = (zeta[i, j] * dx ** 2) / 4 + 1 / 4 * (
                            psi[i - 1, j] + psi[i + 1, j] + psi[i, j - 1] + psi[i, j + 1])

            # updating BC
            f[0, i] = (zeta[0, i] * dx ** 2) / 4 + 1 / 4 * (psi[-2, i] + psi[1, i] + psi[0, i - 1] + psi[0, i + 1])
            f[i, 0] = (zeta[i, 0] * dx ** 2) / 4 + 1 / 4 * (psi[i - 1, 0] + psi[i + 1, 0] + psi[i, -2] + psi[i, 1])
            f[L, i] = f[0, i]
            f[i, L] = f[i, 0]

        # updating corners
        f[0, 0] = (zeta[0, 0] * dx ** 2) / 4 + 1 / 4 * (psi[-2, 0] + psi[1, 0] + psi[0, -2] + psi[0, 1])
        f[0, L] = f[L, 0] = f[L, L] = f[0, 0]
        return f

    # M: the number of points in calculating quadrature integral
    # (and also the number of steps used in Adam-Bashforth predictor)
    # Note Mm is the number of correctors
    Mm = M - 1
    # Forming the quadrature matrix S[m,i]
    S = np.zeros([Mm, Mm + 1])
    for m in range(Mm):  # Calculate quadrature weights
        for i in range(Mm + 1):
            x = np.arange(Mm + 1)  # Construct a polynomial
            y = np.zeros(Mm + 1)  # which equals to 1 at i, 0 at other points
            y[i] = 1
            p = lagrange(x, y)
            para = np.array(p)  # Compute its integral
            P = np.zeros(Mm + 2)
            for k in range(Mm + 1):
                P[k] = para[k] / (Mm + 1 - k)
            P = np.poly1d(P)
            S[m, i] = P(m + 1) - P(m)
    Svec = S[Mm - 1, :]

    # These are unused but may be eventually
    # the time vector
    t = np.arange(0, T + h, h)
    # extended time vector (temporary: cuz I didn't write code for end part)
    t_ext = np.arange(0, T + h + M * h, h)

    #  Inputting initial conditions
    x = np.arange(0, 2 * np.pi + dx, dx)
    X, Y = np.meshgrid(x, x)

    # the final answer for vorticity will be stored in yy
    yy = np.zeros([N + 1, L + 1, L + 1, 1])  # time x space x space
    # putting the initial condition in for vorticity
    yy[0, :, :, 0] = IC(X, Y)

    # The solution for the stream function will be stored in stream
    stream = np.zeros([N + 1, L + 1, L + 1])
    # putting the initial condition in for stream function
    stream[0, :, :] = IC1(X, Y)

    # Value of RHS at initial time
    F0 = func(yy[0, :, :, :], stream[0, :, :])

    # F vector and matrix:
    # the RHS of ODE is evaluated and stored in this vector and matrix:
    # F1 [M x M x L x L]: first index is the order (0=prediction, 1=first correction)
    # second index is the time (iTime), the final index is redundant I just never took it out
    # Note F1 could have been [M-1 x M] as the first two rows are equal to each
    # other BUT we designed it as a place holder for future parallelization

    F1 = np.zeros([Mm, M, L + 1, L + 1, 1])
    F1[:, 0, :, :, :] = F0
    F2 = F0
    # Y2 [M] new set of points derived in each level (prediction and corrections)
    Y2 = np.zeros((M, L + 1, L + 1, 1))
    Y2[:, :, :, :] = yy[0, :, :, :]

    #  Note the extra index on the end of yy, Y2 and F1 are hangovers from solving ODEs and are not needed

    # ===============================================================================

    # ================== INITIAL PART (1) ==================
    # for this part the predictor and correctors step up to M points in time
    # ** predictor ** uses Runge-Kutta 8 (10 stages)
    for iTime in range(0, M - 1):
        k_1 = F1[0, iTime, :, :, :]
        k_2 = func(Y2[0, :, :, :] + (h * 4 / 27) * k_1, stream[iTime, :, :])
        k_3 = func(Y2[0, :, :, :] + (h / 18) * (k_1 + 3 * k_2), stream[iTime, :, :])
        k_4 = func(Y2[0, :, :, :] + (h / 12) * (k_1 + 3 * k_3), stream[iTime, :, :])
        k_5 = func(Y2[0, :, :, :] + (h / 8) * (k_1 + 3 * k_4), stream[iTime, :, :])
        k_6 = func(Y2[0, :, :, :] + (h / 54) * (13 * k_1 - 27 * k_3 + 42 * k_4 + 8 * k_5), stream[iTime, :, :])
        k_7 = func(Y2[0, :, :, :] + (h / 4320) * (389 * k_1 - 54 * k_3 + 966 * k_4 - 824 * k_5 + 243 * k_6),
                   stream[iTime, :, :])
        k_8 = func(Y2[0, :, :, :] + (h / 20) * (-234 * k_1 + 81 * k_3 - 1164 * k_4 + 656 * k_5 - 122 * k_6 + 800 * k_7),
                   stream[iTime, :, :])
        k_9 = func(Y2[0, :, :, :] + (h / 288) * (
                    -127 * k_1 + 18 * k_3 - 678 * k_4 + 456 * k_5 - 9 * k_6 + 576 * k_7 + 4 * k_8), stream[iTime, :, :])
        k_10 = func(Y2[0, :, :, :] + (h / 820) * (
                    1481 * k_1 - 81 * k_3 + 7104 * k_4 - 3376 * k_5 + 72 * k_6 - 5040 * k_7 - 60 * k_8 + 720 * k_9),
                    stream[iTime, :, :])
        Y2[0, :, :, :] = Y2[0, :, :, :] + h / 840 * (
                    41 * k_1 + 27 * k_4 + 272 * k_5 + 27 * k_6 + 216 * k_7 + 216 * k_9 + 41 * k_10)

        F1[0, iTime + 1, :, :, :] = func(Y2[0, :, :, :], stream[iTime, :, :])
        stream[iTime + 1, :, :] = poisson(Y2[0, :, :, :], stream[iTime, :, :])

    # ** correctors ** use Integral Deferred Correction
    for iCor in range(1, M - 1):
        ll = iCor - 1
        for iTime in range(0, M - 1):
            Y2[iCor, :, :, :] = Y2[iCor, :, :, :] + h * (F1[iCor, iTime, :, :, :] - F1[ll, iTime, :, :, :]) + \
                                h * np.tensordot(S[iTime, :], F1[ll, :, :, :, :], axes=1)

            F1[iCor, iTime + 1, :, :, :] = func(Y2[iCor, :, :, :], stream[iTime, :, :])
            stream[iTime + 1, :, :] = poisson(Y2[iCor, :, :, :], stream[iTime, :, :])

    # treat the last correction loop a little different
    for iTime in range(0, M - 1):
        Y2[M - 1, :, :, :] = Y2[M - 1, :, :, :] + h * (F2[:, :, :] - F1[M - 2, iTime, :, :, :]) + \
                             h * np.tensordot(S[iTime, :], F1[ll, :, :, :, :], axes=1)

        F2 = func(Y2[M - 1, :, :, :], stream[iTime, :, :])
        stream[iTime + 1, :, :] = poisson(Y2[M - 1, :, :, :], stream[iTime, :, :])
        yy[iTime + 1, :, :, :] = Y2[M - 1, :, :, :]

    # ================== INITIAL PART (2) ==================

    beta_vec = beta(4)
    beta_vec2 = beta(3)
    for iTime in range(M - 1, 2 * M - 2):
        iStep = iTime - (M - 1)

        # prediction loop
        Y2[0, :, :, :] = Y2[0, :, :, :] + h * np.tensordot(beta_vec, F1[0, -4:, :, :, :], axes=1)

        # correction loops
        for ll in range(iStep):
            iCor = ll + 1

            Y2[iCor, :, :, :] = Y2[iCor, :, :, :] + h * (F1[iCor, -1, :, :, :] - F1[ll, -2, :, :, :]) + \
                                h * np.tensordot(Svec, F1[ll, :, :, :, :], axes=1)

        stream[iTime + 1, :, :] = poisson(Y2[M - 1, :, :, :], stream[iTime, :, :])  # updating stream function array

        F1[0, 0:M - 1, :, :, :] = F1[0, 1:M, :, :, :]  # updating prediction stencil
        F1[0, M - 1, :, :, :] = func(Y2[iCor, :, :, :], stream[iTime, :, :])

        for ll in range(iStep):  # updating correction stencils
            iCor = ll + 1
            F1[iCor, 0:M - 1, :, :, :] = F1[iCor, 1:M, :, :, :]
            F1[iCor, M - 1, :, :, :] = func(Y2[iCor, :, :, :], stream[iTime, :, :])

    # ================== MAIN LOOP FOR TIME ==================

    for iTime in range(2 * M - 2, N + M - 1):
        # prediction loop
        Y2[0, :, :, :] = Y2[0, :, :, :] + h * np.tensordot(beta_vec, F1[0, -4:, :, :, :], axes=1)

        # correction loops up to the second last one
        for ll in range(M - 2):
            iCor = ll + 1
            Fvec = np.array([F1[iCor, -3, :, :, :] - F1[ll, -4, :, :, :], F1[iCor, -2, :, :, :] -
                             F1[ll, -3, :, :, :], F1[iCor, -1, :, :, :] - F1[ll, -2, :, :, :]])

            Y2[iCor, :, :, :] = Y2[iCor, :, :, :] + h * np.tensordot(beta_vec2, Fvec, axes=1) + \
                                h * np.tensordot(Svec, F1[ll, :, :, :, :], axes=1)

            # last correction loop
        F2m = func(yy[iTime + 1 - (M - 1) - 2, :, :, :], stream[iTime + 1 - (M - 1) - 2, :, :])
        F2mm = func(yy[iTime + 1 - (M - 1) - 3, :, :, :], stream[iTime + 1 - (M - 1) - 3, :, :])
        Fvec = np.array([F2mm[:, :, :] - F1[M - 2, -4, :, :, :], F2m[:, :, :] - F1[M - 2, -3, :, :, :],
                         F2[:, :, :] - F1[M - 2, -2, :, :, :]])

        Y2[M - 1, :, :, :] = Y2[M - 1, :, :, :] + h * np.tensordot(beta_vec2, Fvec, axes=1) + \
                             h * np.tensordot(Svec, F1[ll, :, :, :, :], axes=1)

        # ~~~~~~~~~~~ Updating Stencil ~~~~~~~~~~~
        # ---> updating correctors stencil
        for ll in range(1, M - 1):
            F1[ll, 0:M - 1, :, :, :] = F1[ll, 1:M, :, :, :]
            F1[ll, M - 1, :, :, :] = func(Y2[ll, :, :, :], stream[iTime - (M - 1), :, :])

        # storing the final answer
        yy[iTime + 1 - (M - 1), :, :, :] = Y2[M - 1, :, :, :]
        F2 = func(Y2[M - 1, :, :, :], stream[iTime - (M - 1), :, :])
        stream[iTime + 1 - (M - 1), :, :] = poisson(Y2[M - 1, :, :, :], stream[iTime - (M - 1), :, :])

        # ---> updating predictor stencil
        F1[0, 0:M - 1, :, :, :] = F1[0, 1:M, :, :, :]
        F1[0, M - 1, :, :, :] = func(Y2[0, :, :, :], stream[iTime - (M - 1), :, :])

    return yy[:, :, :, 0], stream


#  Vorticity
def ICfunc1(x, y):
    r = 2*np.sin(x)*np.sin(y)
    return r


#  Stream function
def ICfunc2(x, y):
    r = np.sin(x)*np.sin(y)
    return r


zeta, psi = TaylorVortex2D(1, ICfunc1, ICfunc2, 1000, 127, 4)

dx = 2*np.pi/127
x = np.linspace(0,2*np.pi,128)
X, Y = np.meshgrid(x,x)
fig = plt.figure(figsize=(11, 7), dpi=100)
plt.contour(X, Y, psi[1000,:,:],15)
plt.show()

