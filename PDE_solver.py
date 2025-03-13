from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import pandas as pd
from scipy.interpolate import interp1d
plt.rcParams['font.family'] = 'Times New Roman'

def calculate_relative_l2_error(P, A):
    D = P - A

    l2_norm_D = np.linalg.norm(D)
    l2_norm_A = np.linalg.norm(A)

    relative_l2_error = l2_norm_D / l2_norm_A

    return relative_l2_error

def calculate_RMSE(P, A):
    return (np.sum((P - A) ** 2) / P.size) ** 0.5

def softplus(x):
    return np.log1p(np.exp(x))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def MSE(f, g):
    return np.mean((f - g) ** 2, axis=1)


def calculate_r2(y_true, y_pred):
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2


def solve_1group_ND_2D_case_1(x_min, x_interface, x_max, y_min, y_interface, y_max, Nx, Ny,
                                               D_core_1, sigma_core_r_1, X_core_p_1, D_cladding_1, sigma_cladding_r_1, X_cladding_p_1, keff_previous, phi1_previous):
    """Solve 1-group nuclear neutron diffusion equation
        - div(D1 * div phi1) + sigmaR1 * phi1 - Xp1 * phi1 = 0
        with editable boundary condition.
    """
    x = np.linspace(x_min, x_max, Nx)  # x direction discretization
    y = np.linspace(y_min, y_max, Ny)  # y direction discretization
    dx = x[1] - x[0]  # delta x
    dy = y[1] - y[0]  # delta y
    dx2 = dx ** 2  # delta x square
    dy2 = dy ** 2  # delta y square
    x_interface_pos = int(x_interface / dx)  # position of x interface
    y_interface_pos = int(y_interface / dy)  # position of y interface

    # solving linear equations Au = c, where A is coefficient matrix, c = a + b
    a = np.zeros((Nx, Ny))
    A = np.eye(Nx * Ny)


    # Group 1
    for i in range(Nx * Ny):
        y_pos = int(i % Ny)
        x_pos = int(i / Ny)

        if x_pos == 0:
            # x=0 symmetry boundary
            A[i, i] = - 3 / (2 * dx)  # u(x1,y1)
            A[i, i + Ny] = 2 / dx  # u(x2,y1)
            A[i, i + 2 * Ny] = - 1 / (2 * dx)  # u(x3,y1)
        if 0 < x_pos < x_interface_pos:
            if y_pos == 0:
                # y=0 symmetry boundary
                A[i, i] = 3 / (2 * dy)  # u(x2,y1)
                A[i, i + 1] = - 2 / dy   # u(x2,y2)
                A[i, i + 2] = 1 / (2 * dy)  # u(x2,y3)
            if 0 < y_pos < y_interface_pos:
                # If you want to get u(x2,y2)
                A[i, i] = 2 * D_core_1 / dx2 + 2 * D_core_1 / dy2 + sigma_core_r_1  # u(x2,y2)
                A[i, i - Ny] = - D_core_1 / dx2  # u(x1,y2)
                A[i, i - 1] = - D_core_1 / dy2  # u(x2,y1)
                A[i, i + Ny] = - D_core_1 / dx2  # u(x3,y2)
                A[i, i + 1] = - D_core_1 / dy2  # u(x2,y3)
                a[x_pos, y_pos] = X_core_p_1 * phi1_previous[x_pos, y_pos] / keff_previous
            if y_pos == y_interface_pos:
                # y=y_interface_pos interface boundary
                A[i, i] = 3 * D_core_1 / (2 * dy) + 3 * D_cladding_1 / (2 * dy)  # u(x2,y50)
                A[i, i - 1] = - 2 * D_core_1 / dy  # u(x2,y49)
                A[i, i - 2] = D_core_1 / (2 * dy)  # u(x2,y48)
                A[i, i + 1] = -2 * D_cladding_1 / dy  # u(x2,y51)
                A[i, i + 2] = D_cladding_1 / (2 * dy)  # u(x2,y52)
            if y_interface_pos < y_pos < (Ny - 1):
                # If you want to get u(x2,y52)
                A[i, i] = 2 * D_cladding_1 / dx2 + 2 * D_cladding_1 / dy2 + sigma_cladding_r_1  # u(x2,y52)
                A[i, i - Ny] = - D_cladding_1 / dx2  # u(x1,y52)
                A[i, i - 1] = - D_cladding_1 / dy2  # u(x2,y51)
                A[i, i + Ny] = - D_cladding_1 / dx2  # u(x3,y52)
                A[i, i + 1] = - D_cladding_1 / dy2  # u(x2,y53)
                a[x_pos, y_pos] = X_cladding_p_1 * phi1_previous[x_pos, y_pos] / keff_previous
            if y_pos == (Ny - 1):
                # y = Ny-1 vacuum boundary
                A[i, i] = 3 * D_cladding_1 / (2 * dy) + 0.5  # u(x2,y100)
                A[i, i - 1] = - 2 * D_cladding_1 / dy  # u(x2,y99)
                A[i, i - 2] = D_cladding_1 / (2 * dy)  # u(x2,y98)
        if x_pos == x_interface_pos:
            if 0 <= y_pos <= y_interface_pos:
                # x=x_interface_pos interface boundary
                A[i, i] = 3 * D_core_1 / (2 * dx) + 3 * D_cladding_1 / (2 * dx)  # u(x50,y2)
                A[i, i - Ny] = - 2 * D_core_1 / dx  # u(x49,y2)
                A[i, i - 2 * Ny] = D_core_1 / (2 * dx)  # u(x48,y2)
                A[i, i + Ny] = - 2 * D_cladding_1 / dx  # u(x51,y2)
                A[i, i + 2 * Ny] = D_cladding_1 / (2 * dx)  # u(x52,y2)
            if y_interface_pos < y_pos < (Ny - 1):
                # If you want to get u(x50,y52)
                A[i, i] = 2 * D_cladding_1 / dx2 + 2 * D_cladding_1 / dy2 + sigma_cladding_r_1  # u(x50,y52)
                A[i, i - Ny] = - D_cladding_1 / dx2  # u(x49,y52)
                A[i, i - 1] = - D_cladding_1 / dy2  # u(x50,y51)
                A[i, i + Ny] = - D_cladding_1 / dx2  # u(x51,y52)
                A[i, i + 1] = - D_cladding_1 / dy2  # u(x50,y53)
                a[x_pos, y_pos] = X_cladding_p_1 * phi1_previous[x_pos, y_pos] / keff_previous
            if y_pos == (Ny - 1):
                # y = Ny-1 vacuum boundary
                A[i, i] = 3 * D_cladding_1 / (2 * dy) + 0.5  # u(x50,y100)
                A[i, i - 1] = - 2 * D_cladding_1 / dy  # u(x50,y99)
                A[i, i - 2] = D_cladding_1 / (2 * dy)  # u(x50,y98)
        if x_interface_pos < x_pos < Nx - 1:
            if y_pos == 0:
                # y=0 symmetry boundary
                A[i, i] = 3 / (2 * dy)  # u(x52,y1)
                A[i, i + 1] = - 2 / dy   # u(x52,y2)
                A[i, i + 2] = 1 / (2 * dy)  # u(x52,y3)
            if 0 < y_pos < (Ny - 1):
                # If you want to get u(x52,y52)
                A[i, i] = 2 * D_cladding_1 / dx2 + 2 * D_cladding_1 / dy2 + sigma_cladding_r_1  # u(x52,y52)
                A[i, i - Ny] = - D_cladding_1 / dx2  # u(x51,y52)
                A[i, i - 1] = - D_cladding_1 / dy2  # u(x52,y51)
                A[i, i + Ny] = - D_cladding_1 / dx2  # u(x53,y52)
                A[i, i + 1] = - D_cladding_1 / dy2  # u(x52,y53)
                a[x_pos, y_pos] = X_cladding_p_1 * phi1_previous[x_pos, y_pos] / keff_previous
            if y_pos == (Ny - 1):
                # y = Ny-1 vacuum boundary
                A[i, i] = 3 * D_cladding_1 / (2 * dy) + 0.5  # u(x2,y100)
                A[i, i - 1] = - 2 * D_cladding_1 / dy  # u(x2,y99)
                A[i, i - 2] = D_cladding_1 / (2 * dy)  # u(x2,y98)
        if x_pos == Nx - 1:
            # x = Nx - 1 vacuum boundary
            A[i, i] = 3 * D_cladding_1 / (2 * dx) + 0.5  # u(x100,y1)
            A[i, i - Ny] = - 2 * D_cladding_1 / dx  # u(x99,y1)
            A[i, i - 2 * Ny] = D_cladding_1 / (2 * dx)  # u(x98,y1)

    a_vec = a.reshape(Nx * Ny)  # from ab to c
    u = np.linalg.solve(A, a_vec).reshape(Nx, Ny)  # Au = c

    # The first 51 lines are phi 1, the last 51 lines are phi 2
    phi1_now = u
    F_previous = np.zeros(Nx * Ny)
    F_now = np.zeros(Nx * Ny)
    # calculate keff
    for i in range(Nx * Ny):
        x_pos = int(i / Ny)
        y_pos = int(i % Ny)
        if 0 < x_pos < x_interface_pos and 0 < y_pos < y_interface_pos:
            F_previous[i] = phi1_previous[x_pos, y_pos]
            F_now[i] = phi1_now[x_pos, y_pos]
        else:
            F_previous[i] = 0.0 * phi1_previous[x_pos, y_pos]
            F_now[i] = 0.0 * phi1_now[x_pos, y_pos]
    keff_now = keff_previous * np.sum(F_now) / np.sum(F_previous)
    regulation = 1.0
    phi = u / regulation

    return phi, keff_now



def set_ND_1group_para_2D_case_1():
    x = np.linspace(0, 1.0, 101)
    y = np.linspace(0, 1.0, 101)

    x_min, x_interface, x_max = 0, 0.5, 1.0
    y_min, y_interface, y_max = 0, 0.5, 1.0
    Nx, Ny = 101, 101

    # The meanings of each parameter are in solve_ND, using linear interpolation
    D_core_1 = 0.05
    sigma_core_r_1 = 0.15
    X_core_p_1 = 0.3
    D_cladding_1 = 0.10
    sigma_cladding_r_1 = 0.01
    X_cladding_p_1 = 0.0

    keff_previous = 1.0  # initial keff values of the iteration
    phi1 = (lambda x, y: x * y * 0 + 1)  # initial phi 1 values of the iteration
    phi1_previous = phi1(x[:, None], y)

    # Source iteration process
    while True:
        u1, keff_now = solve_1group_ND_2D_case_1(x_min, x_interface, x_max, y_min, y_interface, y_max, Nx, Ny,
                                               D_core_1, sigma_core_r_1, X_core_p_1, D_cladding_1, sigma_cladding_r_1, X_cladding_p_1, keff_previous, phi1_previous)
        # print(u1, u1.shape, u1.max(), u1.min())
        phi1_now = u1
        if np.sum(np.abs(phi1_now - phi1_previous)) > 0.0001:
            print("Diff is: ", np.sum(np.abs(phi1_now - phi1_previous)))
            phi1_previous = phi1_now
            keff_previous = keff_now
        else:
            print("Diff is: ", np.sum(np.abs(phi1_now - phi1_previous)))
            break

    # print(phi_now, phi_now.shape)
    # print("keff=", keff_now)
    # phi1_now = phi1_now / phi1_now[0, 45] * 1.75
    return phi1_now, keff_now


# def solve_1group_ND_2D_case_2(x_min, x_max, y_min, y_max, fuel_1_x, fuel_1_y, fuel_1_r, fuel_2_x,
#                                                  fuel_2_y, fuel_2_r, Nx, Ny, D_fuel_1, sigma_fuel_1_r, X_fuel_1_p,
#                                                  D_fuel_2, sigma_fuel_2_r, X_fuel_2_p, D_fluent, sigma_fluent_r,
#                                                  X_fluent_p, keff_previous, phi1_previous):
#     """Solve 1-group nuclear neutron diffusion equation
#         - div(D1 * div phi1) + sigmaR1 * phi1 - Xp1 * phi1 = 0
#         with editable boundary condition.
#     """
#     x = np.linspace(x_min, x_max, Nx)  # x direction discretization
#     y = np.linspace(y_min, y_max, Ny)  # y direction discretization
#     dx = x[1] - x[0]  # delta x
#     dy = y[1] - y[0]  # delta y
#     dx2 = dx ** 2  # delta x square
#     dy2 = dy ** 2  # delta y square
#     X, Y = np.meshgrid(x, y)  # 创建网格点
#
#     # 2 Circles
#     circle_1 = np.abs((X - fuel_1_x) ** 2 + (Y - fuel_1_y) ** 2 - fuel_1_r ** 2) < 0.0015
#     circle_1_inner = np.abs((X - fuel_1_x) ** 2 + (Y -fuel_1_y) ** 2) < fuel_1_r ** 2 - 0.0015
#     circle_2 = np.abs((X - fuel_2_x) ** 2 + (Y - fuel_2_y) ** 2 - fuel_2_r ** 2) < 0.0015
#     circle_2_inner = np.abs((X - fuel_2_x) ** 2 + (Y - fuel_2_y) ** 2) < fuel_2_r ** 2 - 0.0015
#
#     # solving linear equations Au = c, where A is coefficient matrix, c = a + b
#     a = np.zeros((Nx, Ny))
#     A = np.eye(Nx * Ny)
#
#
#     # Group 1
#     for i in range(Nx * Ny):
#         y_pos = int(i % Ny)
#         x_pos = int(i // Ny)
#
#         if x_pos == 0:
#             # x=0 symmetry boundary
#             A[i, i] = - 3 / (2 * dx)  # u(x1,y1)
#             A[i, i + Ny] = 2 / dx  # u(x2,y1)
#             A[i, i + 2 * Ny] = - 1 / (2 * dx)  # u(x3,y1)
#         if 0 < x_pos < Nx - 1:
#             if y_pos == 0:
#                 # y=0 symmetry boundary
#                 A[i, i] = 3 / (2 * dy)  # u(x2,y1)
#                 A[i, i + 1] = - 2 / dy   # u(x2,y2)
#                 A[i, i + 2] = 1 / (2 * dy)  # u(x2,y3)
#             if 0 < y_pos < (Ny - 1):
#                 # 检查点是否位于circle_1边缘上
#                 if circle_1[y_pos, x_pos]:
#                     # y=circle_1 interface boundary
#                     pos_r = ((fuel_1_x - x[x_pos]) ** 2 + (fuel_1_y - y[y_pos]) ** 2) ** 0.5
#                     pos_cos = (x[x_pos] - fuel_1_x) / pos_r
#                     pos_sin = (y[y_pos] - fuel_1_y) / pos_r
#                     if pos_cos >= 0 and pos_sin >= 0:  # 右上角
#                         A[i, i] = - 3 * D_fluent * pos_cos / (2 * dx) - 3 * D_fluent * pos_sin / (2 * dy) - 3 * D_fuel_1 * pos_cos / (
#                                     2 * dx) - 3 * D_fuel_1 * pos_sin / (2 * dy)  # u(x2,y50)
#                         A[i, i - 1] = 2 * D_fuel_1 * pos_sin / dy  # u(x2,y49)
#                         A[i, i - 2] = - D_fuel_1 * pos_sin / (2 * dy)  # u(x2,y48)
#                         A[i, i + 1] = 2 * D_fluent * pos_sin / dy  # u(x2,y51)
#                         A[i, i + 2] = - D_fluent * pos_sin / (2 * dy)  # u(x2,y52)
#                         A[i, i - Ny] = 2 * D_fuel_1 * pos_cos / dy  # u(x1,y50)
#                         A[i, i - 2 * Ny] = - D_fuel_1 / (2 * dy)  # u(x0,y50)
#                         A[i, i + Ny] = 2 * D_fluent * pos_cos / dy  # u(x3,y50)
#                         A[i, i + 2 * Ny] = - D_fluent * pos_cos / (2 * dy)  # u(x4,y50)
#                     elif pos_cos < 0 and pos_sin > 0:  # 左上角
#                         A[i, i] = 3 * D_fluent * pos_cos / (2 * dx) - 3 * D_fluent * pos_sin / (2 * dy) + 3 * D_fuel_1 * pos_cos / (
#                                     2 * dx) - 3 * D_fuel_1 * pos_sin / (2 * dy)  # u(x2,y50)
#                         A[i, i - 1] = 2 * D_fuel_1 * pos_sin / dy  # u(x2,y49)
#                         A[i, i - 2] = - D_fuel_1 * pos_sin / (2 * dy)  # u(x2,y48)
#                         A[i, i + 1] = 2 * D_fluent * pos_sin / dy  # u(x2,y51)
#                         A[i, i + 2] = - D_fluent * pos_sin / (2 * dy)  # u(x2,y52)
#                         A[i, i - Ny] = - 2 * D_fluent * pos_cos / dy  # u(x1,y50)
#                         A[i, i - 2 * Ny] = D_fluent / (2 * dy)  # u(x0,y50)
#                         A[i, i + Ny] = - 2 * D_fuel_1 * pos_cos / dy  # u(x3,y50)
#                         A[i, i + 2 * Ny] = D_fuel_1 * pos_cos / (2 * dy)  # u(x4,y50)
#                     elif pos_cos <= 0 and pos_sin <= 0:  # 左下角
#                         A[i, i] = 3 * D_fluent * pos_cos / (2 * dx) + 3 * D_fluent * pos_sin / (2 * dy) + 3 * D_fuel_1 * pos_cos / (
#                                     2 * dx) + 3 * D_fuel_1 * pos_sin / (2 * dy)  # u(x2,y50)
#                         A[i, i - 1] = - 2 * D_fluent * pos_sin / dy  # u(x2,y49)
#                         A[i, i - 2] = D_fluent * pos_sin / (2 * dy)  # u(x2,y48)
#                         A[i, i + 1] = - 2 * D_fuel_1 * pos_sin / dy  # u(x2,y51)
#                         A[i, i + 2] = D_fuel_1 * pos_sin / (2 * dy)  # u(x2,y52)
#                         A[i, i - Ny] = - 2 * D_fluent * pos_cos / dy  # u(x1,y50)
#                         A[i, i - 2 * Ny] = D_fluent / (2 * dy)  # u(x0,y50)
#                         A[i, i + Ny] = - 2 * D_fuel_1 * pos_cos / dy  # u(x3,y50)
#                         A[i, i + 2 * Ny] = D_fuel_1 * pos_cos / (2 * dy)  # u(x4,y50)
#                     elif pos_cos > 0 and pos_sin < 0:  # 右下角
#                         A[i, i] = - 3 * D_fluent * pos_cos / (2 * dx) + 3 * D_fluent * pos_sin / (2 * dy) - 3 * D_fuel_1 * pos_cos / (
#                                     2 * dx) + 3 * D_fuel_1 * pos_sin / (2 * dy)  # u(x2,y50)
#                         A[i, i - 1] = - 2 * D_fluent * pos_sin / dy  # u(x2,y49)
#                         A[i, i - 2] = D_fluent * pos_sin / (2 * dy)  # u(x2,y48)
#                         A[i, i + 1] = - 2 * D_fuel_1 * pos_sin / dy  # u(x2,y51)
#                         A[i, i + 2] = D_fuel_1 * pos_sin / (2 * dy)  # u(x2,y52)
#                         A[i, i - Ny] = 2 * D_fuel_1 * pos_cos / dy  # u(x1,y50)
#                         A[i, i - 2 * Ny] = - D_fuel_1 / (2 * dy)  # u(x0,y50)
#                         A[i, i + Ny] = 2 * D_fluent * pos_cos / dy  # u(x3,y50)
#                         A[i, i + 2 * Ny] = - D_fluent * pos_cos / (2 * dy)  # u(x4,y50)
#                 # 检查点是否位于circle_2边缘上
#                 if circle_2[y_pos, x_pos]:
#                     # y=circle_2 interface boundary
#                     pos_r = ((fuel_2_x - x[x_pos]) ** 2 + (fuel_2_y - y[y_pos]) ** 2) ** 0.5
#                     pos_cos = (x[x_pos] - fuel_2_x) / pos_r
#                     pos_sin = (y[y_pos] - fuel_2_y) / pos_r
#                     if pos_cos >= 0 and pos_sin >= 0:  # 右上角
#                         A[i, i] = - 3 * D_fluent * pos_cos / (2 * dx) - 3 * D_fluent * pos_sin / (2 * dy) - 3 * D_fuel_1 * pos_cos / (
#                                     2 * dx) - 3 * D_fuel_1 * pos_sin / (2 * dy)  # u(x2,y50)
#                         A[i, i - 1] = 2 * D_fuel_1 * pos_sin / dy  # u(x2,y49)
#                         A[i, i - 2] = - D_fuel_1 * pos_sin / (2 * dy)  # u(x2,y48)
#                         A[i, i + 1] = 2 * D_fluent * pos_sin / dy  # u(x2,y51)
#                         A[i, i + 2] = - D_fluent * pos_sin / (2 * dy)  # u(x2,y52)
#                         A[i, i - Ny] = 2 * D_fuel_1 * pos_cos / dy  # u(x1,y50)
#                         A[i, i - 2 * Ny] = - D_fuel_1 / (2 * dy)  # u(x0,y50)
#                         A[i, i + Ny] = 2 * D_fluent * pos_cos / dy  # u(x3,y50)
#                         A[i, i + 2 * Ny] = - D_fluent * pos_cos / (2 * dy)  # u(x4,y50)
#                     elif pos_cos < 0 and pos_sin > 0:  # 左上角
#                         A[i, i] = 3 * D_fluent * pos_cos / (2 * dx) - 3 * D_fluent * pos_sin / (2 * dy) + 3 * D_fuel_1 * pos_cos / (
#                                     2 * dx) - 3 * D_fuel_1 * pos_sin / (2 * dy)  # u(x2,y50)
#                         A[i, i - 1] = 2 * D_fuel_1 * pos_sin / dy  # u(x2,y49)
#                         A[i, i - 2] = - D_fuel_1 * pos_sin / (2 * dy)  # u(x2,y48)
#                         A[i, i + 1] = 2 * D_fluent * pos_sin / dy  # u(x2,y51)
#                         A[i, i + 2] = - D_fluent * pos_sin / (2 * dy)  # u(x2,y52)
#                         A[i, i - Ny] = - 2 * D_fluent * pos_cos / dy  # u(x1,y50)
#                         A[i, i - 2 * Ny] = D_fluent / (2 * dy)  # u(x0,y50)
#                         A[i, i + Ny] = - 2 * D_fuel_1 * pos_cos / dy  # u(x3,y50)
#                         A[i, i + 2 * Ny] = D_fuel_1 * pos_cos / (2 * dy)  # u(x4,y50)
#                     elif pos_cos <= 0 and pos_sin <= 0:  # 左下角
#                         A[i, i] = 3 * D_fluent * pos_cos / (2 * dx) + 3 * D_fluent * pos_sin / (2 * dy) + 3 * D_fuel_1 * pos_cos / (
#                                     2 * dx) + 3 * D_fuel_1 * pos_sin / (2 * dy)  # u(x2,y50)
#                         A[i, i - 1] = - 2 * D_fluent * pos_sin / dy  # u(x2,y49)
#                         A[i, i - 2] = D_fluent * pos_sin / (2 * dy)  # u(x2,y48)
#                         A[i, i + 1] = - 2 * D_fuel_1 * pos_sin / dy  # u(x2,y51)
#                         A[i, i + 2] = D_fuel_1 * pos_sin / (2 * dy)  # u(x2,y52)
#                         A[i, i - Ny] = - 2 * D_fluent * pos_cos / dy  # u(x1,y50)
#                         A[i, i - 2 * Ny] = D_fluent / (2 * dy)  # u(x0,y50)
#                         A[i, i + Ny] = - 2 * D_fuel_1 * pos_cos / dy  # u(x3,y50)
#                         A[i, i + 2 * Ny] = D_fuel_1 * pos_cos / (2 * dy)  # u(x4,y50)
#                     elif pos_cos > 0 and pos_sin < 0:  # 右下角
#                         A[i, i] = - 3 * D_fluent * pos_cos / (2 * dx) + 3 * D_fluent * pos_sin / (2 * dy) - 3 * D_fuel_1 * pos_cos / (
#                                     2 * dx) + 3 * D_fuel_1 * pos_sin / (2 * dy)  # u(x2,y50)
#                         A[i, i - 1] = - 2 * D_fluent * pos_sin / dy  # u(x2,y49)
#                         A[i, i - 2] = D_fluent * pos_sin / (2 * dy)  # u(x2,y48)
#                         A[i, i + 1] = - 2 * D_fuel_1 * pos_sin / dy  # u(x2,y51)
#                         A[i, i + 2] = D_fuel_1 * pos_sin / (2 * dy)  # u(x2,y52)
#                         A[i, i - Ny] = 2 * D_fuel_1 * pos_cos / dy  # u(x1,y50)
#                         A[i, i - 2 * Ny] = - D_fuel_1 / (2 * dy)  # u(x0,y50)
#                         A[i, i + Ny] = 2 * D_fluent * pos_cos / dy  # u(x3,y50)
#                         A[i, i + 2 * Ny] = - D_fluent * pos_cos / (2 * dy)  # u(x4,y50)
#                 # 检查点是否位于circle_1内部
#                 if circle_1_inner[y_pos, x_pos] and not circle_1[y_pos, x_pos]:
#                     # If you want to get u(x2,y2)
#                     A[i, i] = 2 * D_fuel_1 / dx2 + 2 * D_fuel_1 / dy2 + sigma_fuel_1_r  # u(x2,y2)
#                     A[i, i - Ny] = - D_fuel_1 / dx2  # u(x1,y2)
#                     A[i, i - 1] = - D_fuel_1 / dy2  # u(x2,y1)
#                     A[i, i + Ny] = - D_fuel_1 / dx2  # u(x3,y2)
#                     A[i, i + 1] = - D_fuel_1 / dy2  # u(x2,y3)
#                     a[x_pos, y_pos] = X_fuel_1_p * phi1_previous[x_pos, y_pos] / keff_previous
#                 # 检查点是否位于circle_2内部
#                 if circle_2_inner[y_pos, x_pos] and not circle_2[y_pos, x_pos]:
#                     # If you want to get u(x2,y2)
#                     A[i, i] = 2 * D_fuel_2 / dx2 + 2 * D_fuel_2 / dy2 + sigma_fuel_2_r  # u(x2,y2)
#                     A[i, i - Ny] = - D_fuel_2 / dx2  # u(x1,y2)
#                     A[i, i - 1] = - D_fuel_2 / dy2  # u(x2,y1)
#                     A[i, i + Ny] = - D_fuel_2 / dx2  # u(x3,y2)
#                     A[i, i + 1] = - D_fuel_2 / dy2  # u(x2,y3)
#                     a[x_pos, y_pos] = X_fuel_2_p * phi1_previous[x_pos, y_pos] / keff_previous
#                 # 检查点是否既不在两个圆上也不在两个圆内
#                 if not (circle_1[y_pos, x_pos] or circle_2[y_pos, x_pos] or circle_1_inner[y_pos, x_pos] or
#                         circle_2_inner[y_pos, x_pos]):
#                     # If you want to get u(x2,y2)
#                     A[i, i] = 2 * D_fluent / dx2 + 2 * D_fluent / dy2 + sigma_fluent_r  # u(x2,y2)
#                     A[i, i - Ny] = - D_fluent / dx2  # u(x1,y2)
#                     A[i, i - 1] = - D_fluent / dy2  # u(x2,y1)
#                     A[i, i + Ny] = - D_fluent / dx2  # u(x3,y2)
#                     A[i, i + 1] = - D_fluent / dy2  # u(x2,y3)
#                     a[x_pos, y_pos] = X_fluent_p * phi1_previous[x_pos, y_pos] / keff_previous
#             if y_pos == (Ny - 1):
#                 # y = Ny-1 vacuum boundary
#                 A[i, i] = 3 * D_fluent / (2 * dy) + 0.5  # u(x2,y100)
#                 A[i, i - 1] = - 2 * D_fluent / dy  # u(x2,y99)
#                 A[i, i - 2] = D_fluent / (2 * dy)  # u(x2,y98)
#         if x_pos == Nx - 1:
#             # x = Nx - 1 vacuum boundary
#             A[i, i] = 3 * D_fluent / (2 * dx) + 0.5  # u(x100,y1)
#             A[i, i - Ny] = - 2 * D_fluent / dx  # u(x99,y1)
#             A[i, i - 2 * Ny] = D_fluent / (2 * dx)  # u(x98,y1)
#
#     a_vec = a.reshape(Nx * Ny)  # from ab to c
#     u = np.linalg.solve(A, a_vec).reshape(Nx, Ny)  # Au = c
#
#     # The first 51 lines are phi 1, the last 51 lines are phi 2
#     phi1_now = u
#     F_previous = np.zeros(Nx * Ny)
#     F_now = np.zeros(Nx * Ny)
#     # calculate keff
#     for i in range(Nx * Ny):
#         x_pos = int(i // Ny)
#         y_pos = int(i % Ny)
#         # 检查点是否位于circle_1内部
#         if circle_1_inner[y_pos, x_pos] and not circle_1[y_pos, x_pos]:
#             F_previous[i] = phi1_previous[x_pos, y_pos]
#             F_now[i] = phi1_now[x_pos, y_pos]
#         elif circle_2_inner[y_pos, x_pos] and not circle_2[y_pos, x_pos]:
#             F_previous[i] = phi1_previous[x_pos, y_pos]
#             F_now[i] = phi1_now[x_pos, y_pos]
#         else:
#             F_previous[i] = 0.0 * phi1_previous[x_pos, y_pos]
#             F_now[i] = 0.0 * phi1_now[x_pos, y_pos]
#     keff_now = keff_previous * np.sum(F_now) / np.sum(F_previous)
#     regulation = 1.0
#     phi = u / regulation
#
#     return phi, keff_now

def solve_1group_ND_2D_case_2(x_min, x_max, y_min, y_max, fuel_1_x, fuel_1_y, fuel_1_r, fuel_2_x,
                                                 fuel_2_y, fuel_2_r, Nx, Ny, D_fuel_1, sigma_fuel_1_r, X_fuel_1_p,
                                                 D_fuel_2, sigma_fuel_2_r, X_fuel_2_p, D_fluent, sigma_fluent_r,
                                                 X_fluent_p, keff_previous, phi1_previous):
    """Solve 1-group nuclear neutron diffusion equation
        - div(D1 * div phi1) + sigmaR1 * phi1 - Xp1 * phi1 = 0
        with editable boundary condition.
    """
    x = np.linspace(x_min, x_max, Nx)  # x direction discretization
    y = np.linspace(y_min, y_max, Ny)  # y direction discretization
    dx = x[1] - x[0]  # delta x
    dy = y[1] - y[0]  # delta y
    dx2 = dx ** 2  # delta x square
    dy2 = dy ** 2  # delta y square
    X, Y = np.meshgrid(x, y)  # 创建网格点

    # 2 Circles
    circle_1 = np.abs((X - fuel_1_x) ** 2 + (Y - fuel_1_y) ** 2 - fuel_1_r ** 2) < 0.0006172
    circle_1_inner = np.abs((X - fuel_1_x) ** 2 + (Y - fuel_1_y) ** 2) < fuel_1_r ** 2 - 0.0006172
    circle_2 = np.abs((X - fuel_2_x) ** 2 + (Y - fuel_2_y) ** 2 - fuel_2_r ** 2) < 0.0006172
    circle_2_inner = np.abs((X - fuel_2_x) ** 2 + (Y - fuel_2_y) ** 2) < fuel_2_r ** 2 - 0.0006172
    # 0.0006172->0.0006174
    # solving linear equations Au = c, where A is coefficient matrix, c = a + b
    a = np.zeros((Nx, Ny))
    A = np.eye(Nx * Ny)


    # Group 1
    for i in range(Nx * Ny):
        y_pos = int(i % Ny)
        x_pos = int(i // Ny)

        if x_pos == 0:
            # x=0 symmetry boundary
            A[i, i] = - 3 / (2 * dx)  # u(x1,y1)
            A[i, i + Ny] = 2 / dx  # u(x2,y1)
            A[i, i + 2 * Ny] = - 1 / (2 * dx)  # u(x3,y1)
        if 0 < x_pos < Nx - 1:
            if y_pos == 0:
                # y=0 symmetry boundary
                A[i, i] = 3 / (2 * dy)  # u(x2,y1)
                A[i, i + 1] = - 2 / dy   # u(x2,y2)
                A[i, i + 2] = 1 / (2 * dy)  # u(x2,y3)
            if 0 < y_pos < (Ny - 1):
                # 检查点是否位于circle_1边缘上
                if circle_1[y_pos, x_pos]:
                    # y=circle_1 interface boundary
                    pos_r = ((fuel_1_x - x[x_pos]) ** 2 + (fuel_1_y - y[y_pos]) ** 2) ** 0.5
                    pos_cos = (x[x_pos] - fuel_1_x) / pos_r
                    pos_sin = (y[y_pos] - fuel_1_y) / pos_r
                    if pos_cos >= 0 and pos_sin >= 0:  # 右上角
                        A[i, i] = - D_fluent * pos_cos / dx - D_fluent * pos_sin / dy - D_fuel_1 * pos_cos / dx - D_fuel_1 * pos_sin / dy  # u(x2,y50)
                        A[i, i - 1] = D_fuel_1 * pos_sin / dy  # u(x2,y49)
                        A[i, i + 1] = D_fluent * pos_sin / dy  # u(x2,y51)
                        A[i, i - Ny] = D_fuel_1 * pos_cos / dy  # u(x1,y50)
                        A[i, i + Ny] = D_fluent * pos_cos / dy  # u(x3,y50)
                    elif pos_cos < 0 and pos_sin > 0:  # 左上角
                        A[i, i] = D_fluent * pos_cos / dx - D_fluent * pos_sin / dy + D_fuel_1 * pos_cos / dx - D_fuel_1 * pos_sin / dy  # u(x2,y50)
                        A[i, i - 1] = D_fuel_1 * pos_sin / dy  # u(x2,y49)
                        A[i, i + 1] = D_fluent * pos_sin / dy  # u(x2,y51)
                        A[i, i - Ny] = - D_fluent * pos_cos / dy  # u(x1,y50)
                        A[i, i + Ny] = - D_fuel_1 * pos_cos / dy  # u(x3,y50)
                    elif pos_cos <= 0 and pos_sin <= 0:  # 左下角
                        A[i, i] = D_fluent * pos_cos / dx + D_fluent * pos_sin / dy + D_fuel_1 * pos_cos / dx + D_fuel_1 * pos_sin / dy  # u(x2,y50)
                        A[i, i - 1] = - D_fluent * pos_sin / dy  # u(x2,y49)
                        A[i, i + 1] = - D_fuel_1 * pos_sin / dy  # u(x2,y51)
                        A[i, i - Ny] = - D_fluent * pos_cos / dy  # u(x1,y50)
                        A[i, i + Ny] = - D_fuel_1 * pos_cos / dy  # u(x3,y50)
                    elif pos_cos > 0 and pos_sin < 0:  # 右下角
                        A[i, i] = - D_fluent * pos_cos / dx + D_fluent * pos_sin / dy - D_fuel_1 * pos_cos / dx + D_fuel_1 * pos_sin / dy  # u(x2,y50)
                        A[i, i - 1] = - D_fluent * pos_sin / dy  # u(x2,y49)
                        A[i, i + 1] = - D_fuel_1 * pos_sin / dy  # u(x2,y51)
                        A[i, i - Ny] = D_fuel_1 * pos_cos / dy  # u(x1,y50)
                        A[i, i + Ny] = D_fluent * pos_cos / dy  # u(x3,y50)
                # 检查点是否位于circle_2边缘上
                if circle_2[y_pos, x_pos]:
                    # y=circle_2 interface boundary
                    pos_r = ((fuel_2_x - x[x_pos]) ** 2 + (fuel_2_y - y[y_pos]) ** 2) ** 0.5
                    pos_cos = (x[x_pos] - fuel_2_x) / pos_r
                    pos_sin = (y[y_pos] - fuel_2_y) / pos_r
                    if pos_cos >= 0 and pos_sin >= 0:  # 右上角
                        A[i, i] = - D_fluent * pos_cos / dx - D_fluent * pos_sin / dy - D_fuel_2 * pos_cos / dx - D_fuel_2 * pos_sin / dy  # u(x2,y50)
                        A[i, i - 1] = D_fuel_2 * pos_sin / dy  # u(x2,y49)
                        A[i, i + 1] = D_fluent * pos_sin / dy  # u(x2,y51)
                        A[i, i - Ny] = D_fuel_2 * pos_cos / dy  # u(x1,y50)
                        A[i, i + Ny] = D_fluent * pos_cos / dy  # u(x3,y50)
                    elif pos_cos < 0 and pos_sin > 0:  # 左上角
                        A[i, i] = D_fluent * pos_cos / dx - D_fluent * pos_sin / dy + D_fuel_2 * pos_cos / dx - D_fuel_2 * pos_sin / dy  # u(x2,y50)
                        A[i, i - 1] = D_fuel_2 * pos_sin / dy  # u(x2,y49)
                        A[i, i + 1] = D_fluent * pos_sin / dy  # u(x2,y51)
                        A[i, i - Ny] = - D_fluent * pos_cos / dy  # u(x1,y50)
                        A[i, i + Ny] = - D_fuel_2 * pos_cos / dy  # u(x3,y50)
                    elif pos_cos <= 0 and pos_sin <= 0:  # 左下角
                        A[i, i] = D_fluent * pos_cos / dx + D_fluent * pos_sin / dy + D_fuel_2 * pos_cos / dx + D_fuel_2 * pos_sin / dy  # u(x2,y50)
                        A[i, i - 1] = - D_fluent * pos_sin / dy  # u(x2,y49)
                        A[i, i + 1] = - D_fuel_2 * pos_sin / dy  # u(x2,y51)
                        A[i, i - Ny] = - D_fluent * pos_cos / dy  # u(x1,y50)
                        A[i, i + Ny] = - D_fuel_2 * pos_cos / dy  # u(x3,y50)
                    elif pos_cos > 0 and pos_sin < 0:  # 右下角
                        A[i, i] = - D_fluent * pos_cos / dx + D_fluent * pos_sin / dy - D_fuel_2 * pos_cos / dx + D_fuel_2 * pos_sin / dy  # u(x2,y50)
                        A[i, i - 1] = - D_fluent * pos_sin / dy  # u(x2,y49)
                        A[i, i + 1] = - D_fuel_2 * pos_sin / dy  # u(x2,y51)
                        A[i, i - Ny] = D_fuel_2 * pos_cos / dy  # u(x1,y50)
                        A[i, i + Ny] = D_fluent * pos_cos / dy  # u(x3,y50)
                # 检查点是否位于circle_1内部
                if circle_1_inner[y_pos, x_pos] and not circle_1[y_pos, x_pos]:
                    # If you want to get u(x2,y2)
                    A[i, i] = 2 * D_fuel_1 / dx2 + 2 * D_fuel_1 / dy2 + sigma_fuel_1_r  # u(x2,y2)
                    A[i, i - Ny] = - D_fuel_1 / dx2  # u(x1,y2)
                    A[i, i - 1] = - D_fuel_1 / dy2  # u(x2,y1)
                    A[i, i + Ny] = - D_fuel_1 / dx2  # u(x3,y2)
                    A[i, i + 1] = - D_fuel_1 / dy2  # u(x2,y3)
                    a[x_pos, y_pos] = X_fuel_1_p * phi1_previous[x_pos, y_pos] / keff_previous
                # 检查点是否位于circle_2内部
                if circle_2_inner[y_pos, x_pos] and not circle_2[y_pos, x_pos]:
                    # If you want to get u(x2,y2)
                    A[i, i] = 2 * D_fuel_2 / dx2 + 2 * D_fuel_2 / dy2 + sigma_fuel_2_r  # u(x2,y2)
                    A[i, i - Ny] = - D_fuel_2 / dx2  # u(x1,y2)
                    A[i, i - 1] = - D_fuel_2 / dy2  # u(x2,y1)
                    A[i, i + Ny] = - D_fuel_2 / dx2  # u(x3,y2)
                    A[i, i + 1] = - D_fuel_2 / dy2  # u(x2,y3)
                    a[x_pos, y_pos] = X_fuel_2_p * phi1_previous[x_pos, y_pos] / keff_previous
                # 检查点是否既不在两个圆上也不在两个圆内
                if not (circle_1[y_pos, x_pos] or circle_2[y_pos, x_pos] or circle_1_inner[y_pos, x_pos] or
                        circle_2_inner[y_pos, x_pos]):
                    # If you want to get u(x2,y2)
                    A[i, i] = 2 * D_fluent / dx2 + 2 * D_fluent / dy2 + sigma_fluent_r  # u(x2,y2)
                    A[i, i - Ny] = - D_fluent / dx2  # u(x1,y2)
                    A[i, i - 1] = - D_fluent / dy2  # u(x2,y1)
                    A[i, i + Ny] = - D_fluent / dx2  # u(x3,y2)
                    A[i, i + 1] = - D_fluent / dy2  # u(x2,y3)
                    a[x_pos, y_pos] = X_fluent_p * phi1_previous[x_pos, y_pos] / keff_previous
            if y_pos == (Ny - 1):
                # y = Ny-1 vacuum boundary
                A[i, i] = 3 * D_fluent / (2 * dy) + 0.5  # u(x2,y100)
                A[i, i - 1] = - 2 * D_fluent / dy  # u(x2,y99)
                A[i, i - 2] = D_fluent / (2 * dy)  # u(x2,y98)
        if x_pos == Nx - 1:
            # x = Nx - 1 vacuum boundary
            A[i, i] = 3 * D_fluent / (2 * dx) + 0.5  # u(x100,y1)
            A[i, i - Ny] = - 2 * D_fluent / dx  # u(x99,y1)
            A[i, i - 2 * Ny] = D_fluent / (2 * dx)  # u(x98,y1)

    a_vec = a.reshape(Nx * Ny)  # from ab to c
    u = np.linalg.solve(A, a_vec).reshape(Nx, Ny)  # Au = c

    # The first 51 lines are phi 1, the last 51 lines are phi 2
    phi1_now = u
    F_previous = np.zeros(Nx * Ny)
    F_now = np.zeros(Nx * Ny)
    # calculate keff
    for i in range(Nx * Ny):
        x_pos = int(i // Ny)
        y_pos = int(i % Ny)
        if circle_1_inner[y_pos, x_pos] or circle_1[y_pos, x_pos] or circle_2_inner[y_pos, x_pos] or circle_2[y_pos, x_pos]:
            F_previous[i] = phi1_previous[x_pos, y_pos]
            F_now[i] = phi1_now[x_pos, y_pos]
        else:
            F_previous[i] = 0.0 * phi1_previous[x_pos, y_pos]
            F_now[i] = 0.0 * phi1_now[x_pos, y_pos]
    keff_now = keff_previous * np.sum(F_now) / np.sum(F_previous)
    regulation = 1.0
    phi = u / regulation

    return phi, keff_now

def set_ND_1group_para_2D_case_2():
    Nx, Ny = 181, 181
    x = np.linspace(0, 1.0, Nx)
    y = np.linspace(0, 1.0, Ny)

    x_min, x_max = 0, 1.0
    y_min, y_max = 0, 1.0
    fuel_1_x, fuel_1_y, fuel_1_r = 0.2, 0.2, 0.15
    fuel_2_x, fuel_2_y, fuel_2_r = 0.6, 0.2, 0.15

    # The meanings of each parameter are in solve_ND, using linear interpolation
    D_fuel_1 = 0.05
    sigma_fuel_1_r = 0.15
    X_fuel_1_p = 0.3
    D_fuel_2 = 0.05
    sigma_fuel_2_r = 0.15
    X_fuel_2_p = 0.3
    D_fluent = 0.10
    sigma_fluent_r = 0.01
    X_fluent_p = 0.0

    keff_previous = 0.2  # initial keff values of the iteration
    phi1 = (lambda x, y: x * y * 0 + 0.5)  # initial phi 1 values of the iteration
    phi1_previous = phi1(x[:, None], y)

    # Source iteration process
    while True:
        u1, keff_now = solve_1group_ND_2D_case_2(x_min, x_max, y_min, y_max, fuel_1_x, fuel_1_y, fuel_1_r, fuel_2_x,
                                                 fuel_2_y, fuel_2_r, Nx, Ny, D_fuel_1, sigma_fuel_1_r, X_fuel_1_p,
                                                 D_fuel_2, sigma_fuel_2_r, X_fuel_2_p, D_fluent, sigma_fluent_r,
                                                 X_fluent_p, keff_previous, phi1_previous)
        # print(u1, u1.shape, u1.max(), u1.min())
        phi1_now = u1
        if np.sum(np.abs(phi1_now - phi1_previous)) > 0.0001:
            print("Diff is: ", np.sum(np.abs(phi1_now - phi1_previous)))
            phi1_previous = phi1_now
            keff_previous = keff_now
        else:
            print("Diff is: ", np.sum(np.abs(phi1_now - phi1_previous)))
            break

    # print(phi_now, phi_now.shape)
    # print("keff=", keff_now)
    # phi1_now = phi1_now / phi1_now[0, 45] * 1.75
    return phi1_now, keff_now





def solve_1group_ND_2D_case_3(x_min, x_interface_1, x_interface_2, x_max, y_min, y_interface_1, y_interface_2, y_max, Nx, Ny,
                                               D_1, sigma_1, X_1, D_2, sigma_2, X_2, keff_previous, phi1_previous):
    """Solve 1-group nuclear neutron diffusion equation
        - div(D1 * div phi1) + sigmaR1 * phi1 - Xp1 * phi1 = 0
        with editable boundary condition.
    """
    x = np.linspace(x_min, x_max, Nx)  # x direction discretization
    y = np.linspace(y_min, y_max, Ny)  # y direction discretization
    dx = x[1] - x[0]  # delta x
    dy = y[1] - y[0]  # delta y
    dx2 = dx ** 2  # delta x square
    dy2 = dy ** 2  # delta y square
    x_interface_1_pos = int(x_interface_1 / dx)  # position of x interface 1
    x_interface_2_pos = int(x_interface_2 / dx)  # position of x interface 2
    y_interface_1_pos = int(y_interface_1 / dy)  # position of y interface 1
    y_interface_2_pos = int(y_interface_2 / dy)  # position of y interface 2

    # solving linear equations Au = c, where A is coefficient matrix, c = a + b
    a = np.zeros((Nx, Ny))
    A = np.eye(Nx * Ny)


    # Group 1
    for i in range(Nx * Ny):
        y_pos = int(i % Ny)
        x_pos = int(i / Ny)

        if x_pos == 0:
            # x=0 vacuum boundary
            A[i, i] = 3 * D_1 / (2 * dx) + 0.5  # u(x1,y1)
            A[i, i + Ny] = - 2 * D_1 / dx  # u(x2,y1)
            A[i, i + 2 * Ny] = D_1 / (2 * dx)  # u(x3,y1)
        if 0 < x_pos < x_interface_1_pos:
            if y_pos == 0:
                # y=0 vacuum boundary
                A[i, i] = 3 * D_1 / (2 * dy) + 0.5  # u(x2,y1)
                A[i, i + 1] = - 2 * D_1 / dy   # u(x2,y2)
                A[i, i + 2] = D_1 / (2 * dy)  # u(x2,y3)
            if 0 < y_pos < (Ny - 1):
                # If you want to get u(x2,y2)
                A[i, i] = 2 * D_1 / dx2 + 2 * D_1 / dy2 + sigma_1  # u(x2,y2)
                A[i, i - Ny] = - D_1 / dx2  # u(x1,y2)
                A[i, i - 1] = - D_1 / dy2  # u(x2,y1)
                A[i, i + Ny] = - D_1 / dx2  # u(x3,y2)
                A[i, i + 1] = - D_1 / dy2  # u(x2,y3)
                a[x_pos, y_pos] = X_1 * phi1_previous[x_pos, y_pos] / keff_previous
            if y_pos == (Ny - 1):
                # y = Ny-1 vacuum boundary
                A[i, i] = 3 * D_1 / (2 * dy) + 0.5  # u(x2,y100)
                A[i, i - 1] = - 2 * D_1 / dy  # u(x2,y99)
                A[i, i - 2] = D_1 / (2 * dy)  # u(x2,y98)
        if x_pos == x_interface_1_pos:
            if y_pos == 0:
                # y=0 vacuum boundary
                A[i, i] = 3 * D_1 / (2 * dy) + 0.5  # u(x2,y1)
                A[i, i + 1] = - 2 * D_1 / dy   # u(x2,y2)
                A[i, i + 2] = D_1 / (2 * dy)  # u(x2,y3)
            if 0 < y_pos <= y_interface_1_pos:
                # If you want to get u(x50,y52)
                A[i, i] = 2 * D_1 / dx2 + 2 * D_1 / dy2 + sigma_1  # u(x50,y52)
                A[i, i - Ny] = - D_1 / dx2  # u(x49,y52)
                A[i, i - 1] = - D_1 / dy2  # u(x50,y51)
                A[i, i + Ny] = - D_1 / dx2  # u(x51,y52)
                A[i, i + 1] = - D_1 / dy2  # u(x50,y53)
                a[x_pos, y_pos] = X_1 * phi1_previous[x_pos, y_pos] / keff_previous
            if y_interface_1_pos < y_pos < y_interface_2_pos:
                # x=x_interface_pos interface boundary
                A[i, i] = 3 * D_1 / (2 * dx) + 3 * D_2 / (2 * dx)  # u(x50,y2)
                A[i, i - Ny] = - 2 * D_1 / dx  # u(x49,y2)
                A[i, i - 2 * Ny] = D_1 / (2 * dx)  # u(x48,y2)
                A[i, i + Ny] = - 2 * D_2 / dx  # u(x51,y2)
                A[i, i + 2 * Ny] = D_2 / (2 * dx)  # u(x52,y2)
            if y_interface_2_pos <= y_pos < (Ny - 1):
                # If you want to get u(x50,y52)
                A[i, i] = 2 * D_1 / dx2 + 2 * D_1 / dy2 + sigma_1  # u(x50,y52)
                A[i, i - Ny] = - D_1 / dx2  # u(x49,y52)
                A[i, i - 1] = - D_1 / dy2  # u(x50,y51)
                A[i, i + Ny] = - D_1 / dx2  # u(x51,y52)
                A[i, i + 1] = - D_1 / dy2  # u(x50,y53)
                a[x_pos, y_pos] = X_1 * phi1_previous[x_pos, y_pos] / keff_previous
            if y_pos == (Ny - 1):
                # y = Ny-1 vacuum boundary
                A[i, i] = 3 * D_1 / (2 * dy) + 0.5  # u(x2,y100)
                A[i, i - 1] = - 2 * D_1 / dy  # u(x2,y99)
                A[i, i - 2] = D_1 / (2 * dy)  # u(x2,y98)
        if x_interface_1_pos < x_pos < x_interface_2_pos:
            if y_pos == 0:
                # y=0 vacuum boundary
                A[i, i] = 3 * D_1 / (2 * dy) + 0.5  # u(x2,y1)
                A[i, i + 1] = - 2 * D_1 / dy   # u(x2,y2)
                A[i, i + 2] = D_1 / (2 * dy)  # u(x2,y3)
            if 0 < y_pos < y_interface_1_pos:
                # If you want to get u(x2,y2)
                A[i, i] = 2 * D_1 / dx2 + 2 * D_1 / dy2 + sigma_1  # u(x2,y2)
                A[i, i - Ny] = - D_1 / dx2  # u(x1,y2)
                A[i, i - 1] = - D_1 / dy2  # u(x2,y1)
                A[i, i + Ny] = - D_1 / dx2  # u(x3,y2)
                A[i, i + 1] = - D_1 / dy2  # u(x2,y3)
                a[x_pos, y_pos] = X_1 * phi1_previous[x_pos, y_pos] / keff_previous
            if y_pos == y_interface_1_pos:
                # y=y_interface_pos interface boundary
                A[i, i] = 3 * D_1 / (2 * dy) + 3 * D_2 / (2 * dy)  # u(x2,y50)
                A[i, i - 1] = - 2 * D_1 / dy  # u(x2,y49)
                A[i, i - 2] = D_1 / (2 * dy)  # u(x2,y48)
                A[i, i + 1] = - 2 * D_2 / dy  # u(x2,y51)
                A[i, i + 2] = D_2 / (2 * dy)  # u(x2,y52)
            if y_interface_1_pos < y_pos < y_interface_2_pos:
                # If you want to get u(x2,y52)
                A[i, i] = 2 * D_2 / dx2 + 2 * D_2 / dy2 + sigma_2  # u(x2,y52)
                A[i, i - Ny] = - D_2 / dx2  # u(x1,y52)
                A[i, i - 1] = - D_2 / dy2  # u(x2,y51)
                A[i, i + Ny] = - D_2 / dx2  # u(x3,y52)
                A[i, i + 1] = - D_2 / dy2  # u(x2,y53)
                a[x_pos, y_pos] = X_2 * phi1_previous[x_pos, y_pos] / keff_previous
            if y_pos == y_interface_2_pos:
                # y=y_interface_pos interface boundary
                A[i, i] = 3 * D_2 / (2 * dy) + 3 * D_1 / (2 * dy)  # u(x2,y50)
                A[i, i - 1] = - 2 * D_2 / dy  # u(x2,y49)
                A[i, i - 2] = D_2 / (2 * dy)  # u(x2,y48)
                A[i, i + 1] = - 2 * D_1 / dy  # u(x2,y51)
                A[i, i + 2] = D_1 / (2 * dy)  # u(x2,y52)
            if y_interface_2_pos < y_pos < (Ny - 1):
                # If you want to get u(x2,y52)
                A[i, i] = 2 * D_1 / dx2 + 2 * D_1 / dy2 + sigma_1  # u(x2,y52)
                A[i, i - Ny] = - D_1 / dx2  # u(x1,y52)
                A[i, i - 1] = - D_1 / dy2  # u(x2,y51)
                A[i, i + Ny] = - D_1 / dx2  # u(x3,y52)
                A[i, i + 1] = - D_1 / dy2  # u(x2,y53)
                a[x_pos, y_pos] = X_1 * phi1_previous[x_pos, y_pos] / keff_previous
            if y_pos == (Ny - 1):
                # y = Ny-1 vacuum boundary
                A[i, i] = 3 * D_1 / (2 * dy) + 0.5  # u(x2,y100)
                A[i, i - 1] = - 2 * D_1 / dy  # u(x2,y99)
                A[i, i - 2] = D_1 / (2 * dy)  # u(x2,y98)
        if x_pos == x_interface_2_pos:
            if y_pos == 0:
                # y=0 vacuum boundary
                A[i, i] = 3 * D_1 / (2 * dy) + 0.5  # u(x2,y1)
                A[i, i + 1] = - 2 * D_1 / dy   # u(x2,y2)
                A[i, i + 2] = D_1 / (2 * dy)  # u(x2,y3)
            if 0 < y_pos <= y_interface_1_pos:
                # If you want to get u(x50,y52)
                A[i, i] = 2 * D_1 / dx2 + 2 * D_1 / dy2 + sigma_1  # u(x50,y52)
                A[i, i - Ny] = - D_1 / dx2  # u(x49,y52)
                A[i, i - 1] = - D_1 / dy2  # u(x50,y51)
                A[i, i + Ny] = - D_1 / dx2  # u(x51,y52)
                A[i, i + 1] = - D_1 / dy2  # u(x50,y53)
                a[x_pos, y_pos] = X_1 * phi1_previous[x_pos, y_pos] / keff_previous
            if y_interface_1_pos < y_pos < y_interface_2_pos:
                # x=x_interface_pos interface boundary
                A[i, i] = 3 * D_2 / (2 * dx) + 3 * D_1 / (2 * dx)  # u(x50,y2)
                A[i, i - Ny] = - 2 * D_2 / dx  # u(x49,y2)
                A[i, i - 2 * Ny] = D_2 / (2 * dx)  # u(x48,y2)
                A[i, i + Ny] = - 2 * D_1 / dx  # u(x51,y2)
                A[i, i + 2 * Ny] = D_1 / (2 * dx)  # u(x52,y2)
            if y_interface_2_pos <= y_pos < (Ny - 1):
                # If you want to get u(x50,y52)
                A[i, i] = 2 * D_1 / dx2 + 2 * D_1 / dy2 + sigma_1  # u(x50,y52)
                A[i, i - Ny] = - D_1 / dx2  # u(x49,y52)
                A[i, i - 1] = - D_1 / dy2  # u(x50,y51)
                A[i, i + Ny] = - D_1 / dx2  # u(x51,y52)
                A[i, i + 1] = - D_1 / dy2  # u(x50,y53)
                a[x_pos, y_pos] = X_1 * phi1_previous[x_pos, y_pos] / keff_previous
            if y_pos == (Ny - 1):
                # y = Ny-1 vacuum boundary
                A[i, i] = 3 * D_1 / (2 * dy) + 0.5  # u(x2,y100)
                A[i, i - 1] = - 2 * D_1 / dy  # u(x2,y99)
                A[i, i - 2] = D_1 / (2 * dy)  # u(x2,y98)
        if x_interface_2_pos < x_pos < Nx - 1:
            if y_pos == 0:
                # y=0 vacuum boundary
                A[i, i] = 3 * D_1 / (2 * dy) + 0.5  # u(x2,y1)
                A[i, i + 1] = - 2 * D_1 / dy   # u(x2,y2)
                A[i, i + 2] = D_1 / (2 * dy)  # u(x2,y3)
            if 0 < y_pos < (Ny - 1):
                # If you want to get u(x2,y2)
                A[i, i] = 2 * D_1 / dx2 + 2 * D_1 / dy2 + sigma_1  # u(x2,y2)
                A[i, i - Ny] = - D_1 / dx2  # u(x1,y2)
                A[i, i - 1] = - D_1 / dy2  # u(x2,y1)
                A[i, i + Ny] = - D_1 / dx2  # u(x3,y2)
                A[i, i + 1] = - D_1 / dy2  # u(x2,y3)
                a[x_pos, y_pos] = X_1 * phi1_previous[x_pos, y_pos] / keff_previous
            if y_pos == (Ny - 1):
                # y = Ny-1 vacuum boundary
                A[i, i] = 3 * D_1 / (2 * dy) + 0.5  # u(x2,y100)
                A[i, i - 1] = - 2 * D_1 / dy  # u(x2,y99)
                A[i, i - 2] = D_1 / (2 * dy)  # u(x2,y98)
        if x_pos == Nx - 1:
            # x = Nx - 1 vacuum boundary
            A[i, i] = 3 * D_1 / (2 * dx) + 0.5  # u(x100,y1)
            A[i, i - Ny] = - 2 * D_1 / dx  # u(x99,y1)
            A[i, i - 2 * Ny] = D_1 / (2 * dx)  # u(x98,y1)

    a_vec = a.reshape(Nx * Ny)  # from ab to c
    u = np.linalg.solve(A, a_vec).reshape(Nx, Ny)  # Au = c
    # The first 51 lines are phi 1, the last 51 lines are phi 2
    phi1_now = u
    F_previous = np.zeros(Nx * Ny)
    F_now = np.zeros(Nx * Ny)
    # calculate keff
    for i in range(Nx * Ny):
        x_pos = int(i / Ny)
        y_pos = int(i % Ny)
        if (x_interface_1_pos <= x_pos <= x_interface_2_pos and y_interface_1_pos <= y_pos <= y_interface_2_pos):
            F_previous[i] = phi1_previous[x_pos, y_pos]
            F_now[i] = phi1_now[x_pos, y_pos]
        else:
            F_previous[i] = 0.0 * phi1_previous[x_pos, y_pos]
            F_now[i] = 0.0 * phi1_now[x_pos, y_pos]
    keff_now = keff_previous * np.sum(F_now) / np.sum(F_previous)
    regulation = 1.0
    phi = u / regulation
    print("keff_now=", keff_now)
    return phi, keff_now



def set_ND_1group_para_2D_case_3():
    x_min, x_interface_1, x_interface_2, x_max = 0, 40.0, 60.0, 100.0
    y_min, y_interface_1, y_interface_2, y_max = 0, 40.0, 60.0, 100.0
    Nx, Ny = 101, 101

    x = np.linspace(x_min, x_max, Nx)
    y = np.linspace(y_min, y_max, Ny)

    # The meanings of each parameter are in solve_ND, using linear interpolation
    D_1 = 2.0950
    sigma_1 = 0.064256
    X_1 = 0.0
    D_2 = 2.2008
    sigma_2 = 0.062158
    X_2 = 0.107622


    keff_previous = 1.0  # initial keff values of the iteration
    phi1 = (lambda x, y: x * y * 0 + 1)  # initial phi 1 values of the iteration
    phi1_previous = phi1(x[:, None], y)

    # Source iteration process
    while True:
        u1, keff_now = solve_1group_ND_2D_case_3(x_min, x_interface_1, x_interface_2, x_max, y_min, y_interface_1, y_interface_2, y_max, Nx, Ny,
                                               D_1, sigma_1, X_1, D_2, sigma_2, X_2, keff_previous, phi1_previous)
        # print(u1, u1.shape, u1.max(), u1.min())
        phi1_now = u1
        if np.sum(np.abs(phi1_now - phi1_previous)) > 0.0001:
            print("Diff is: ", np.sum(np.abs(phi1_now - phi1_previous)))
            phi1_previous = phi1_now
            keff_previous = keff_now
        else:
            print("Diff is: ", np.sum(np.abs(phi1_now - phi1_previous)))
            break

    # print(phi_now, phi_now.shape)
    # print("keff=", keff_now)
    # phi1_now = phi1_now / phi1_now[0, 45] * 1.75
    return phi1_now, keff_now


def solve_1group_ND_2D_case_4(x_min, x_interface_1, x_interface_2, x_max, y_min, y_interface_1, y_interface_2, y_max, Nx, Ny,
                                               D_1, sigma_1, X_1, D_2, sigma_2, X_2, keff_previous, phi1_previous):
    """Solve 1-group nuclear neutron diffusion equation
        - div(D1 * div phi1) + sigmaR1 * phi1 - Xp1 * phi1 = 0
        with editable boundary condition.
    """
    x = np.linspace(x_min, x_max, Nx)  # x direction discretization
    y = np.linspace(y_min, y_max, Ny)  # y direction discretization
    dx = x[1] - x[0]  # delta x
    dy = y[1] - y[0]  # delta y
    dx2 = dx ** 2  # delta x square
    dy2 = dy ** 2  # delta y square
    x_interface_1_pos = int(x_interface_1 / dx)  # position of x interface 1
    x_interface_2_pos = int(x_interface_2 / dx)  # position of x interface 2
    y_interface_1_pos = int(y_interface_1 / dy)  # position of y interface 1
    y_interface_2_pos = int(y_interface_2 / dy)  # position of y interface 2

    # solving linear equations Au = c, where A is coefficient matrix, c = a + b
    a = np.zeros((Nx, Ny))
    A = np.eye(Nx * Ny)


    # Group 1
    for i in range(Nx * Ny):
        y_pos = int(i % Ny)
        x_pos = int(i / Ny)

        if x_pos == 0:
            # x=0 symmetry boundary
            A[i, i] = - 3 / (2 * dx)  # u(x1,y1)
            A[i, i + Ny] = 2 / dx  # u(x2,y1)
            A[i, i + 2 * Ny] = - 1 / (2 * dx)  # u(x3,y1)
        if 0 < x_pos < x_interface_1_pos:
            if y_pos == 0:
                # y=0 symmetry boundary
                A[i, i] = - 3 / (2 * dy)  # u(x2,y1)
                A[i, i + 1] = 2 / dy   # u(x2,y2)
                A[i, i + 2] = - 1 / (2 * dy)  # u(x2,y3)
            if 0 < y_pos < y_interface_1_pos:
                # If you want to get u(x2,y2)
                A[i, i] = 2 * D_2 / dx2 + 2 * D_2 / dy2 + sigma_2  # u(x2,y2)
                A[i, i - Ny] = - D_2 / dx2  # u(x1,y2)
                A[i, i - 1] = - D_2 / dy2  # u(x2,y1)
                A[i, i + Ny] = - D_2 / dx2  # u(x3,y2)
                A[i, i + 1] = - D_2 / dy2  # u(x2,y3)
                a[x_pos, y_pos] = X_2 * phi1_previous[x_pos, y_pos] / keff_previous
            if y_pos == y_interface_1_pos:
                # y=y_interface_pos interface boundary
                A[i, i] = 3 * D_2 / (2 * dy) + 3 * D_1 / (2 * dy)  # u(x2,y50)
                A[i, i - 1] = - 2 * D_2 / dy  # u(x2,y49)
                A[i, i - 2] = D_2 / (2 * dy)  # u(x2,y48)
                A[i, i + 1] = -2 * D_1 / dy  # u(x2,y51)
                A[i, i + 2] = D_1 / (2 * dy)  # u(x2,y52)
            if y_interface_1_pos < y_pos < y_interface_2_pos:
                # If you want to get u(x2,y2)
                A[i, i] = 2 * D_1 / dx2 + 2 * D_1 / dy2 + sigma_1  # u(x2,y2)
                A[i, i - Ny] = - D_1 / dx2  # u(x1,y2)
                A[i, i - 1] = - D_1 / dy2  # u(x2,y1)
                A[i, i + Ny] = - D_1 / dx2  # u(x3,y2)
                A[i, i + 1] = - D_1 / dy2  # u(x2,y3)
                a[x_pos, y_pos] = X_1 * phi1_previous[x_pos, y_pos] / keff_previous
            if y_pos == y_interface_2_pos:
                # y=y_interface_pos interface boundary
                A[i, i] = 3 * D_1 / (2 * dy) + 3 * D_2 / (2 * dy)  # u(x2,y50)
                A[i, i - 1] = - 2 * D_1 / dy  # u(x2,y49)
                A[i, i - 2] = D_1 / (2 * dy)  # u(x2,y48)
                A[i, i + 1] = -2 * D_2 / dy  # u(x2,y51)
                A[i, i + 2] = D_2 / (2 * dy)  # u(x2,y52)
            if y_interface_2_pos < y_pos < (Ny - 1):
                # If you want to get u(x2,y52)
                A[i, i] = 2 * D_2 / dx2 + 2 * D_2 / dy2 + sigma_2  # u(x2,y52)
                A[i, i - Ny] = - D_2 / dx2  # u(x1,y52)
                A[i, i - 1] = - D_2 / dy2  # u(x2,y51)
                A[i, i + Ny] = - D_2 / dx2  # u(x3,y52)
                A[i, i + 1] = - D_2 / dy2  # u(x2,y53)
                a[x_pos, y_pos] = X_2 * phi1_previous[x_pos, y_pos] / keff_previous
            if y_pos == (Ny - 1):
                # y = Ny-1 symmetry boundary
                A[i, i] = 3 / (2 * dy)   # u(x2,y100)
                A[i, i - 1] = - 2 / dy  # u(x2,y99)
                A[i, i - 2] = 1 / (2 * dy)  # u(x2,y98)
            # if y_pos == (Ny - 1):
            #     # y = Ny-1 vacuum boundary
            #     A[i, i] = 3 * D_2 / (2 * dy) + 0.5  # u(x2,y100)
            #     A[i, i - 1] = - 2 * D_2 / dy  # u(x2,y99)
            #     A[i, i - 2] = D_2 / (2 * dy)  # u(x2,y98)
        if x_pos == x_interface_1_pos:
            if 0 <= y_pos <= y_interface_1_pos:
                # x=x_interface_pos interface boundary
                A[i, i] = 3 * D_2 / (2 * dx) + 3 * D_1 / (2 * dx)  # u(x50,y2)
                A[i, i - Ny] = - 2 * D_2 / dx  # u(x49,y2)
                A[i, i - 2 * Ny] = D_2 / (2 * dx)  # u(x48,y2)
                A[i, i + Ny] = - 2 * D_1 / dx  # u(x51,y2)
                A[i, i + 2 * Ny] = D_1 / (2 * dx)  # u(x52,y2)
            if y_interface_1_pos < y_pos <= y_interface_2_pos:
                # If you want to get u(x50,y52)
                A[i, i] = 2 * D_1 / dx2 + 2 * D_1 / dy2 + sigma_1  # u(x50,y52)
                A[i, i - Ny] = - D_1 / dx2  # u(x49,y52)
                A[i, i - 1] = - D_1 / dy2  # u(x50,y51)
                A[i, i + Ny] = - D_1 / dx2  # u(x51,y52)
                A[i, i + 1] = - D_1 / dy2  # u(x50,y53)
                a[x_pos, y_pos] = X_1 * phi1_previous[x_pos, y_pos] / keff_previous
            if y_interface_2_pos < y_pos < (Ny - 1):
                # If you want to get u(x50,y52)
                A[i, i] = 2 * D_2 / dx2 + 2 * D_2 / dy2 + sigma_2  # u(x50,y52)
                A[i, i - Ny] = - D_2 / dx2  # u(x49,y52)
                A[i, i - 1] = - D_2 / dy2  # u(x50,y51)
                A[i, i + Ny] = - D_2 / dx2  # u(x51,y52)
                A[i, i + 1] = - D_2 / dy2  # u(x50,y53)
                a[x_pos, y_pos] = X_2 * phi1_previous[x_pos, y_pos] / keff_previous
            if y_pos == (Ny - 1):
                # y = Ny-1 symmetry boundary
                A[i, i] = 3 / (2 * dy)  # u(x50,y100)
                A[i, i - 1] = - 2 / dy  # u(x50,y99)
                A[i, i - 2] = 1 / (2 * dy)  # u(x50,y98)
            # if y_pos == (Ny - 1):
            #     # y = Ny-1 vacuum boundary
            #     A[i, i] = 3 * D_2 / (2 * dy) + 0.5  # u(x2,y100)
            #     A[i, i - 1] = - 2 * D_2 / dy  # u(x2,y99)
            #     A[i, i - 2] = D_2 / (2 * dy)  # u(x2,y98)
        if x_interface_1_pos < x_pos < x_interface_2_pos:
            if y_pos == 0:
                # y=0 symmetry boundary
                A[i, i] = - 3 / (2 * dy)  # u(x2,y1)
                A[i, i + 1] = 2 / dy   # u(x2,y2)
                A[i, i + 2] = - 1 / (2 * dy)  # u(x2,y3)
            if 0 < y_pos < y_interface_2_pos:
                # If you want to get u(x2,y2)
                A[i, i] = 2 * D_1 / dx2 + 2 * D_1 / dy2 + sigma_1  # u(x2,y2)
                A[i, i - Ny] = - D_1 / dx2  # u(x1,y2)
                A[i, i - 1] = - D_1 / dy2  # u(x2,y1)
                A[i, i + Ny] = - D_1 / dx2  # u(x3,y2)
                A[i, i + 1] = - D_1 / dy2  # u(x2,y3)
                a[x_pos, y_pos] = X_1 * phi1_previous[x_pos, y_pos] / keff_previous
            if y_pos == y_interface_2_pos:
                # y=y_interface_pos interface boundary
                A[i, i] = 3 * D_1 / (2 * dy) + 3 * D_2 / (2 * dy)  # u(x2,y50)
                A[i, i - 1] = - 2 * D_1 / dy  # u(x2,y49)
                A[i, i - 2] = D_1 / (2 * dy)  # u(x2,y48)
                A[i, i + 1] = -2 * D_2 / dy  # u(x2,y51)
                A[i, i + 2] = D_2 / (2 * dy)  # u(x2,y52)
            if y_interface_2_pos < y_pos < (Ny - 1):
                # If you want to get u(x2,y52)
                A[i, i] = 2 * D_2 / dx2 + 2 * D_2 / dy2 + sigma_2  # u(x2,y52)
                A[i, i - Ny] = - D_2 / dx2  # u(x1,y52)
                A[i, i - 1] = - D_2 / dy2  # u(x2,y51)
                A[i, i + Ny] = - D_2 / dx2  # u(x3,y52)
                A[i, i + 1] = - D_2 / dy2  # u(x2,y53)
                a[x_pos, y_pos] = X_2 * phi1_previous[x_pos, y_pos] / keff_previous
            if y_pos == (Ny - 1):
                # y = Ny-1 symmetry boundary
                A[i, i] = 3 / (2 * dy)   # u(x2,y100)
                A[i, i - 1] = - 2 / dy  # u(x2,y99)
                A[i, i - 2] = 1 / (2 * dy)  # u(x2,y98)
            # if y_pos == (Ny - 1):
            #     # y = Ny-1 vacuum boundary
            #     A[i, i] = 3 * D_2 / (2 * dy) + 0.5  # u(x2,y100)
            #     A[i, i - 1] = - 2 * D_2 / dy  # u(x2,y99)
            #     A[i, i - 2] = D_2 / (2 * dy)  # u(x2,y98)
        if x_pos == x_interface_2_pos:
            if 0 <= y_pos <= y_interface_2_pos:
                # x=x_interface_pos interface boundary
                A[i, i] = 3 * D_1 / (2 * dx) + 3 * D_2 / (2 * dx)  # u(x50,y2)
                A[i, i - Ny] = - 2 * D_1 / dx  # u(x49,y2)
                A[i, i - 2 * Ny] = D_1 / (2 * dx)  # u(x48,y2)
                A[i, i + Ny] = - 2 * D_2 / dx  # u(x51,y2)
                A[i, i + 2 * Ny] = D_2 / (2 * dx)  # u(x52,y2)
            if y_interface_2_pos < y_pos < (Ny - 1):
                # If you want to get u(x50,y52)
                A[i, i] = 2 * D_2 / dx2 + 2 * D_2 / dy2 + sigma_2  # u(x50,y52)
                A[i, i - Ny] = - D_2 / dx2  # u(x49,y52)
                A[i, i - 1] = - D_2 / dy2  # u(x50,y51)
                A[i, i + Ny] = - D_2 / dx2  # u(x51,y52)
                A[i, i + 1] = - D_2 / dy2  # u(x50,y53)
                a[x_pos, y_pos] = X_2 * phi1_previous[x_pos, y_pos] / keff_previous
            if y_pos == (Ny - 1):
                # y = Ny-1 symmetry boundary
                A[i, i] = 3 / (2 * dy)  # u(x50,y100)
                A[i, i - 1] = - 2 / dy  # u(x50,y99)
                A[i, i - 2] = 1 / (2 * dy)  # u(x50,y98)
            # if y_pos == (Ny - 1):
            #     # y = Ny-1 vacuum boundary
            #     A[i, i] = 3 * D_2 / (2 * dy) + 0.5  # u(x50,y100)
            #     A[i, i - 1] = - 2 * D_2 / dy  # u(x50,y99)
            #     A[i, i - 2] = D_2 / (2 * dy)  # u(x50,y98)
        if x_interface_2_pos < x_pos < Nx - 1:
            if y_pos == 0:
                # y=0 symmetry boundary
                A[i, i] = 3 / (2 * dy)  # u(x52,y1)
                A[i, i + 1] = - 2 / dy   # u(x52,y2)
                A[i, i + 2] = 1 / (2 * dy)  # u(x52,y3)
            if 0 < y_pos < (Ny - 1):
                # If you want to get u(x52,y52)
                A[i, i] = 2 * D_2 / dx2 + 2 * D_2 / dy2 + sigma_2  # u(x52,y52)
                A[i, i - Ny] = - D_2 / dx2  # u(x51,y52)
                A[i, i - 1] = - D_2 / dy2  # u(x52,y51)
                A[i, i + Ny] = - D_2 / dx2  # u(x53,y52)
                A[i, i + 1] = - D_2 / dy2  # u(x52,y53)
                a[x_pos, y_pos] = X_2 * phi1_previous[x_pos, y_pos] / keff_previous
            if y_pos == (Ny - 1):
                # y = Ny-1 symmetry boundary
                A[i, i] = 3 / (2 * dy)  # u(x2,y100)
                A[i, i - 1] = - 2 / dy  # u(x2,y99)
                A[i, i - 2] = 1 / (2 * dy)  # u(x2,y98)
            # if y_pos == (Ny - 1):
            #     # y = Ny-1 vacuum boundary
            #     A[i, i] = 3 * D_2 / (2 * dy) + 0.5  # u(x2,y100)
            #     A[i, i - 1] = - 2 * D_2 / dy  # u(x2,y99)
            #     A[i, i - 2] = D_2 / (2 * dy)  # u(x2,y98)
        if x_pos == Nx - 1:
            # x = Nx - 1 symmetry boundary
            A[i, i] = 3 / (2 * dx)  # u(x100,y1)
            A[i, i - Ny] = - 2 / dx  # u(x99,y1)
            A[i, i - 2 * Ny] = 1 / (2 * dx)  # u(x98,y1)
        # if x_pos == Nx - 1:
        #     # x = Nx - 1 vacuum boundary
        #     A[i, i] = 3 * D_2 / (2 * dx) + 0.5  # u(x100,y1)
        #     A[i, i - Ny] = - 2 * D_2 / dx  # u(x99,y1)
        #     A[i, i - 2 * Ny] = D_2 / (2 * dx)  # u(x98,y1)

    a_vec = a.reshape(Nx * Ny)  # from ab to c
    u = np.linalg.solve(A, a_vec).reshape(Nx, Ny)  # Au = c
    # The first 51 lines are phi 1, the last 51 lines are phi 2
    phi1_now = u
    F_previous = np.zeros(Nx * Ny)
    F_now = np.zeros(Nx * Ny)
    # calculate keff
    for i in range(Nx * Ny):
        x_pos = int(i / Ny)
        y_pos = int(i % Ny)
        if (0 < x_pos < x_interface_2_pos and 0 < y_pos < y_interface_2_pos) and ~(0 < x_pos <= x_interface_1_pos and 0 < y_pos <= y_interface_1_pos):
            F_previous[i] = phi1_previous[x_pos, y_pos]
            F_now[i] = phi1_now[x_pos, y_pos]
        else:
            F_previous[i] = 0.0 * phi1_previous[x_pos, y_pos]
            F_now[i] = 0.0 * phi1_now[x_pos, y_pos]
    keff_now = keff_previous * np.sum(F_now) / np.sum(F_previous)
    regulation = 1.0
    phi = u / regulation
    print("keff_now=", keff_now)
    return phi, keff_now

def set_ND_1group_para_2D_case_4():
    x_min, x_interface_1, x_interface_2, x_max = 0, 24.0, 56.0, 80.0
    y_min, y_interface_1, y_interface_2, y_max = 0, 24.0, 56.0, 80.0
    Nx, Ny = 81, 81

    x = np.linspace(x_min, x_max, Nx)
    y = np.linspace(y_min, y_max, Ny)

    # The meanings of each parameter are in solve_ND, using linear interpolation
    D_1 = 0.4
    sigma_1 = 0.15
    X_1 = 0.2
    D_2 = 0.5
    sigma_2 = 0.05
    X_2 = 0.06


    keff_previous = 1.0  # initial keff values of the iteration
    phi1 = (lambda x, y: x * y * 0 + 1)  # initial phi 1 values of the iteration
    phi1_previous = phi1(x[:, None], y)

    # Source iteration process
    while True:
        u1, keff_now = solve_1group_ND_2D_case_4(x_min, x_interface_1, x_interface_2, x_max, y_min, y_interface_1, y_interface_2, y_max, Nx, Ny,
                                               D_1, sigma_1, X_1, D_2, sigma_2, X_2, keff_previous, phi1_previous)
        # print(u1, u1.shape, u1.max(), u1.min())
        phi1_now = u1
        if np.sum(np.abs(phi1_now - phi1_previous)) > 0.0001:
            print("Diff is: ", np.sum(np.abs(phi1_now - phi1_previous)))
            phi1_previous = phi1_now
            keff_previous = keff_now
        else:
            print("Diff is: ", np.sum(np.abs(phi1_now - phi1_previous)))
            break

    # print(phi_now, phi_now.shape)
    # print("keff=", keff_now)
    # phi1_now = phi1_now / phi1_now[0, 45] * 1.75
    return phi1_now, keff_now


def main():
    Nx, Ny = 181, 181


    # T = (lambda r, z: r * z * 0 + 700)
    # T_previous = T(r[:, None], z)
    # phi1 = (lambda r, z: r * z * 0 + 0.24)
    # phi1_start = phi1(r[:, None], z)
    # phi2 = (lambda r, z: r * z * 0 + 0.011)
    # phi2_start = phi2(r[:, None], z)



    phi1_final, keff_final = set_ND_1group_para_2D_case_2()
    print("Keff = ", keff_final)
    # T_tmp = set_HC_1group_para(phi1_start)
    #
    # print("T_tmp=", T_tmp, T_tmp.shape, T_tmp.min(), T_tmp.max())

    phi1_true = (phi1_final * 1.0 / phi1_final[0, 0]).reshape(Ny, Nx)

    fig, ax = plt.subplots(figsize=(8, 6))  # 设置图形尺寸
    im = ax.imshow(np.flipud(phi1_true.T), cmap='plasma', vmin=phi1_true.min(), vmax=phi1_true.max(),
                    interpolation='nearest', aspect='auto', extent=[0, 100, 0, 100])
    cb = plt.colorbar(im)  # 添加颜色条
    cb.set_label('$\phi$', fontsize=12)  # 在colorbar旁边添加希腊字符phi
    plt.xlabel('x', fontsize=12)  # 设置x轴标签
    plt.ylabel('y', fontsize=12)  # 设置y轴标签
    plt.title('Case 4 PINN + Numerical Result', fontsize=14)  # 设置标题

    # 在(0,0)点画一个大黑点
    ax.plot(0, 0, 'ko', markersize=12, color='black')  # 'ko' 表示黑色圆点
    # 在(0,0)点旁边添加文本"Fixed Point: (0,0,1.0)"
    ax.text(0.02, 0.01, 'Fixed Point: (0, 0, 1)', color='black', va='bottom', ha='left', transform=ax.transAxes,
            fontsize=14)

    plt.xticks(fontsize=10)  # 设置x轴刻度字体大小
    plt.yticks(fontsize=10)  # 设置y轴刻度字体大小
    # plt.grid()  # 添加网格线
    plt.axis('on')  # 移除坐标轴边框

    plt.show()



if __name__ == "__main__":
    main()
