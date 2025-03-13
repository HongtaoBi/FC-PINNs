import torch
import deepxde as dde
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as ticker
import numpy as np
from sklearn.metrics import r2_score
# from deepxde.nn.pytorch import deeponet
# from test_solver import set_HC_para
# from plotting import calculate_relative_l2_error
from scipy.interpolate import griddata, interp1d
from scipy.optimize import fsolve
from PDE_solver import set_ND_1group_para_2D_case_2, calculate_RMSE
import os
import pandas as pd

os.environ["DDE_BACKEND"] = "pytorch"
plt.rcParams['font.family'] = 'Times New Roman'

# # GPU CPU usage
# gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
# cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
# print(gpus, cpus)
# tf.config.experimental.set_visible_devices(devices=gpus[0:4], device_type='GPU')
print("Cuda available is", torch.cuda.is_available())
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
dde.config.set_default_float("float64")

left = 0.0
right = 1.0
down = 0.0
up = 1.0
fuel_1_x, fuel_1_y, fuel_1_r = 0.2, 0.2, 0.15
fuel_2_x, fuel_2_y, fuel_2_r = 0.6, 0.2, 0.15
D_fuel_1 = 0.05
sigma_fuel_1_r = 0.15
X_fuel_1_p = 0.3
D_fuel_2 = 0.05
sigma_fuel_2_r = 0.15
X_fuel_2_p = 0.3
D_fluent = 0.10
sigma_fluent_r = 0.01
X_fluent_p = 0.0
extrapolated_ratio = 2.0
keff = dde.Variable(0.5)

def pde_nd(x, y):
    # x = x[:, 0:1]
    # y = x[:, 1:2]
    phi1_fuel_1 = y[:, 0:1]
    phi1_fuel_2 = y[:, 1:2]
    phi1_fluent = y[:, 2:3]

    dphi1_fuel_1_dxx = dde.grad.hessian(phi1_fuel_1, x, i=0, j=0)
    dphi1_fuel_1_dyy = dde.grad.hessian(phi1_fuel_1, x, i=1, j=1)

    dphi1_fuel_2_dxx = dde.grad.hessian(phi1_fuel_2, x, i=0, j=0)
    dphi1_fuel_2_dyy = dde.grad.hessian(phi1_fuel_2, x, i=1, j=1)

    dphi1_fluent_dxx = dde.grad.hessian(phi1_fluent, x, i=0, j=0)
    dphi1_fluent_dyy = dde.grad.hessian(phi1_fluent, x, i=1, j=1)

    nd_fuel_1_pde = - (D_fuel_1 * (dphi1_fuel_1_dxx + dphi1_fuel_1_dyy)) + sigma_fuel_1_r * phi1_fuel_1 - X_fuel_1_p / keff * phi1_fuel_1
    nd_fuel_2_pde = - (D_fuel_2 * (dphi1_fuel_2_dxx + dphi1_fuel_2_dyy)) + sigma_fuel_2_r * phi1_fuel_2 - X_fuel_2_p / keff * phi1_fuel_2
    nd_fluent_pde = - (D_fluent * (dphi1_fluent_dxx + dphi1_fluent_dyy)) + sigma_fluent_r * phi1_fluent - X_fluent_p / keff * phi1_fluent

    distance_to_circle_1 = torch.sqrt((x[:, 0:1] - fuel_1_x) ** 2 + (x[:, 1:2] - fuel_1_y) ** 2)
    distance_to_circle_2 = torch.sqrt((x[:, 0:1] - fuel_2_x) ** 2 + (x[:, 1:2] - fuel_2_y) ** 2)

    nd_fuel_1 = torch.where(distance_to_circle_1 < fuel_1_r, nd_fuel_1_pde, 0.0)
    nd_fuel_2 = torch.where(distance_to_circle_2 < fuel_2_r, nd_fuel_2_pde, 0.0)
    nd_fluent = torch.where((distance_to_circle_1 <= fuel_1_r) | (distance_to_circle_2 <= fuel_2_r), 0.0, nd_fluent_pde)

    return [nd_fuel_1, nd_fuel_2, nd_fluent]

    # return torch.where(x < interface, nd1_left_pde, nd1_right_pde)

def boundary_x_left_fluent_func(x, y, X):
    phi1_fluent = y[:, 2:3]

    dphi1_fluent_dx = dde.grad.jacobian(phi1_fluent, x, i=0, j=0)

    return dphi1_fluent_dx - 0.0


def boundary_x_right_fluent_func(x, y, X):
    phi1_fluent = y[:, 2:3]

    dphi1_fluent_dx = dde.grad.jacobian(phi1_fluent, x, i=0, j=0)

    return dphi1_fluent_dx + phi1_fluent / (2 * D_fluent)

def boundary_y_down_fluent_func(x, y, X):
    phi1_fluent = y[:, 2:3]

    dphi1_fluent_dy = dde.grad.jacobian(phi1_fluent, x, i=0, j=1)

    return dphi1_fluent_dy - 0.0


def boundary_y_up_fluent_func(x, y, X):
    phi1_fluent = y[:, 2:3]

    dphi1_fluent_dy = dde.grad.jacobian(phi1_fluent, x, i=0, j=1)

    return dphi1_fluent_dy + phi1_fluent / (2 * D_fluent)


def boundary_x_extrapolated_right_fluent_func(x, y, X):
    phi1_fluent = y[:, 2:3]

    return phi1_fluent - 0.0

def boundary_y_extrapolated_up_fluent_func(x, y, X):
    phi1_fluent = y[:, 2:3]

    return phi1_fluent - 0.0


def continuity_flux_circle_1_interface_func(x, y, X):
    phi1_fuel_1 = y[:, 0:1]
    phi1_fluent = y[:, 2:3]

    pos_r = torch.sqrt((x[:, 0:1] - fuel_1_x) ** 2 + (x[:, 1:2] - fuel_1_y) ** 2)
    pos_cos = (x[:, 0:1] - fuel_1_x) / pos_r
    pos_sin = (x[:, 1:2] - fuel_1_y) / pos_r

    dphi1_fuel_1_dx = dde.grad.jacobian(phi1_fuel_1, x, i=0, j=0)
    dphi1_fuel_1_dy = dde.grad.jacobian(phi1_fuel_1, x, i=0, j=1)
    dphi1_fuel_1_dr = dphi1_fuel_1_dx * pos_cos + dphi1_fuel_1_dy * pos_sin

    dphi1_fluent_dx = dde.grad.jacobian(phi1_fluent, x, i=0, j=0)
    dphi1_fluent_dy = dde.grad.jacobian(phi1_fluent, x, i=0, j=1)
    dphi1_fluent_dr = dphi1_fluent_dx * pos_cos + dphi1_fluent_dy * pos_sin

    return D_fuel_1 * dphi1_fuel_1_dr - D_fluent * dphi1_fluent_dr

def continuity_neutron_circle_1_interface_func(x, y, X):
    phi1_fuel_1 = y[:, 0:1]
    phi1_fluent = y[:, 2:3]

    return phi1_fuel_1 - phi1_fluent

def continuity_flux_circle_2_interface_func(x, y, X):
    phi1_fuel_2 = y[:, 1:2]
    phi1_fluent = y[:, 2:3]

    pos_r = torch.sqrt((x[:, 0:1] - fuel_2_x) ** 2 + (x[:, 1:2] - fuel_2_y) ** 2)
    pos_cos = (x[:, 0:1] - fuel_2_x) / pos_r
    pos_sin = (x[:, 1:2] - fuel_2_y) / pos_r

    dphi1_fuel_2_dx = dde.grad.jacobian(phi1_fuel_2, x, i=0, j=0)
    dphi1_fuel_2_dy = dde.grad.jacobian(phi1_fuel_2, x, i=0, j=1)
    dphi1_fuel_2_dr = dphi1_fuel_2_dx * pos_cos + dphi1_fuel_2_dy * pos_sin

    dphi1_fluent_dx = dde.grad.jacobian(phi1_fluent, x, i=0, j=0)
    dphi1_fluent_dy = dde.grad.jacobian(phi1_fluent, x, i=0, j=1)
    dphi1_fluent_dr = dphi1_fluent_dx * pos_cos + dphi1_fluent_dy * pos_sin

    return D_fuel_2 * dphi1_fuel_2_dr - D_fluent * dphi1_fluent_dr

def continuity_neutron_circle_2_interface_func(x, y, X):
    phi1_fuel_2 = y[:, 1:2]
    phi1_fluent = y[:, 2:3]

    return phi1_fuel_2 - phi1_fluent

def fixed_point_func(x, y, X):
    phi1_fluent = y[:, 2:3]

    return phi1_fluent - 1.0

def output_transform(x, y):
    phi1_fuel_1 = y[:, 0:1]
    phi1_fuel_2 = y[:, 1:2]
    phi1_fluent = y[:, 2:3]


    phi1_fuel_1_transformed = phi1_fuel_1
    phi1_fuel_2_transformed = phi1_fuel_2
    phi1_fluent_transformed = phi1_fluent * ((0.0 - x[:, 0:1]) ** 2 + (0.0 - x[:, 1:2]) ** 2) + 1.0
    y_new = torch.cat((phi1_fuel_1_transformed, phi1_fuel_2_transformed, phi1_fluent_transformed), dim=1)
    return y_new


def set_pinn_nd():
    geom = dde.geometry.Rectangle([left, down], [right, up])
    theta = np.linspace(0, 2 * np.pi, 100)
    circle_1_interface_points = np.vstack((fuel_1_r * np.cos(theta) + fuel_1_x, fuel_1_r * np.sin(theta) + fuel_1_y)).T    # [100, 2]
    circle_2_interface_points = np.vstack((fuel_2_r * np.cos(theta) + fuel_2_x, fuel_2_r * np.sin(theta) + fuel_2_y)).T    # [100, 2]
    phi1_left_BC_fluent_points = np.vstack((np.full((100), left), np.linspace(down, up, num=100))).T    # [100, 2]
    phi1_right_BC_fluent_points = np.vstack((np.full((100), right), np.linspace(down, up, num=100))).T    # [100, 2]
    phi1_down_BC_fluent_points = np.vstack((np.linspace(left, right, num=100), np.full((100), down))).T    # [100, 2]
    phi1_up_BC_fluent_points = np.vstack((np.linspace(left, right, num=100), np.full((100), up))).T    # [100, 2]
    phi1_right_extrapolated_BC_points = np.vstack((np.full((100), right + extrapolated_ratio * D_fluent), np.linspace(down, up, num=100))).T    # [100, 2]
    phi1_up_extrapolated_BC_points = np.vstack((np.linspace(left, right, num=100), np.full((100), up + extrapolated_ratio * D_fluent))).T    # [100, 2]
    fixed_point = np.vstack((np.full((1), 0.0), np.full((1), 0.0))).T    # [100, 2]

    circle_1_interface_bc_continuity_flux = dde.icbc.PointSetOperatorBC(circle_1_interface_points, 0.0, continuity_flux_circle_1_interface_func)
    circle_1_interface_bc_continuity_neutron = dde.icbc.PointSetOperatorBC(circle_1_interface_points, 0.0, continuity_neutron_circle_1_interface_func)
    circle_2_interface_bc_continuity_flux = dde.icbc.PointSetOperatorBC(circle_2_interface_points, 0.0, continuity_flux_circle_2_interface_func)
    circle_2_interface_bc_continuity_neutron = dde.icbc.PointSetOperatorBC(circle_2_interface_points, 0.0, continuity_neutron_circle_2_interface_func)

    fluent_bc_x_left = dde.icbc.PointSetOperatorBC(phi1_left_BC_fluent_points, 0.0, boundary_x_left_fluent_func)
    fluent_bc_x_right = dde.icbc.PointSetOperatorBC(phi1_right_BC_fluent_points, 0.0, boundary_x_right_fluent_func)
    fluent_extrapolated_bc_x_right = dde.icbc.PointSetOperatorBC(phi1_right_extrapolated_BC_points, 0.0, boundary_x_extrapolated_right_fluent_func)
    fluent_bc_y_down = dde.icbc.PointSetOperatorBC(phi1_down_BC_fluent_points, 0.0, boundary_y_down_fluent_func)
    fluent_bc_y_up = dde.icbc.PointSetOperatorBC(phi1_up_BC_fluent_points, 0.0, boundary_y_up_fluent_func)
    fluent_extrapolated_bc_y_up = dde.icbc.PointSetOperatorBC(phi1_up_extrapolated_BC_points, 0.0, boundary_y_extrapolated_up_fluent_func)

    fixed_point_constraint = dde.icbc.PointSetOperatorBC(fixed_point, 0.0, fixed_point_func)

    BC = [circle_1_interface_bc_continuity_flux, circle_1_interface_bc_continuity_neutron,
          circle_2_interface_bc_continuity_flux, circle_2_interface_bc_continuity_neutron,
          fluent_bc_x_left, fluent_bc_x_right, fluent_extrapolated_bc_x_right,
          fluent_bc_y_down, fluent_bc_y_up, fluent_extrapolated_bc_y_up, fixed_point_constraint]
    # loss_weights = [10000, 10000, 10, 1000, 10, 1000, 10, 1000, 10, 100, 1000, 10, 100]
    # loss_weights_hard = [1000, 1000, 10, 100, 1, 100, 1, 100, 1, 10, 100, 1, 10]
    # loss_weights = [1000, 1000, 1, 100, 100, 100, 100, 100, 1, 1, 100, 1, 1]
    loss_weights_soft = [1000, 1000, 10, 100, 1, 100, 1, 100, 1, 10, 100, 1, 10, 1]

    data = dde.data.PDE(
        geom,
        pde_nd,
        BC,
        num_domain=5000,
        num_test=5000,
    )

    net = dde.nn.pytorch.FNN([2] + 3 * [100] + [3], "tanh", "Glorot normal")

    # Use apply_output_transform if using hard-constraint, otherwise it's a soft-constraint
    # net.apply_output_transform(output_transform)

    model = dde.Model(data, net)
    variable = dde.callbacks.VariableValue(
        [keff], period=50, filename="keff.dat"
    )

    model.compile("adam", lr=0.0001, loss_weights=loss_weights_soft, external_trainable_variables=[keff])
    model.train(iterations=20000, display_every=1000, model_save_path="PINN_2D_example_models/PINN_2D_example_model_case_2.ckpt", callbacks=[variable])
    model.compile("L-BFGS", loss_weights=loss_weights_soft, external_trainable_variables=[keff])
    dde.optimizers.config.set_LBFGS_options(maxiter=180000)
    loss_history, train_state = model.train(callbacks=[variable])
    # model.compile("adam", lr=0.0001, loss_weights=loss_weights)
    # model.train(iterations=5000, display_every=1000, model_save_path="PINN_HC_model/HC_model.ckpt")
    dde.saveplot(loss_history, train_state, issave=True, isplot=True)

    # model.compile("adam", lr=0.0001, loss_weights=loss_weights, external_trainable_variables=[keff])
    # model.restore("./PINN_2D_example_models/PINN_2D_example_model_case_1.ckpt-50000.pt")
    # model.train(iterations=2000, display_every=1000, callbacks=[variable])
    # model.compile("L-BFGS", loss_weights=loss_weights, external_trainable_variables=[keff])
    # dde.optimizers.config.set_LBFGS_options(maxiter=8000)
    # loss_history, train_state = model.train(callbacks=[variable])
    # dde.saveplot(loss_history, train_state, issave=True, isplot=True)


    # model.compile("adam", lr=0.0001, loss_weights=loss_weights, external_trainable_variables=[keff])
    # model.restore("./PINN_HC_model/HC_model.ckpt-25000.pt")
    # loss_history, train_state = model.train(iterations=7000, display_every=1000, callbacks=[variable])
    # # model.compile("L-BFGS", loss_weights=loss_weights, external_trainable_variables=[keff])
    # # dde.optimizers.config.set_LBFGS_options(maxiter=28000)
    # # loss_history, train_state = model.train(callbacks=[variable])
    # dde.saveplot(loss_history, train_state, issave=False, isplot=False)


    x_cor = np.linspace(left, right, num=101)
    y_cor = np.linspace(down, up, num=101)
    x_grid, y_grid = np.meshgrid(x_cor, y_cor)
    circle_1_condition = np.sqrt((x_grid - fuel_1_x) ** 2 + (y_grid - fuel_1_y) ** 2) < fuel_1_r
    circle_2_condition = np.sqrt((x_grid - fuel_2_x) ** 2 + (y_grid - fuel_2_y) ** 2) < fuel_2_r
    X = np.column_stack((x_grid.flatten(), y_grid.flatten()))
    y_pred = model.predict(X)
    phi1_fuel_1 = y_pred[:, 0:1].reshape(101, 101)
    phi1_fuel_2 = y_pred[:, 1:2].reshape(101, 101)
    phi1_fluent = y_pred[:, 2:3].reshape(101, 101)
    phi1_pred = np.zeros((101, 101))
    phi1_pred[circle_1_condition] = phi1_fuel_1[circle_1_condition]
    phi1_pred[circle_2_condition] = phi1_fuel_2[circle_2_condition]
    phi1_pred[~(circle_1_condition | circle_2_condition)] = phi1_fluent[~(circle_1_condition | circle_2_condition)]
    print("keff_pred=", keff)
    return phi1_pred, keff

def result_surface_plot(phi1_pred, phi1_true):
    fig = plt.figure(figsize=(8, 6))  # 设置图形尺寸
    ax = fig.add_subplot(111, projection='3d')  # 创建3D轴

    # 将phi1_final的值转换为X, Y网格
    X, Y = np.meshgrid(np.arange(phi1_pred.shape[1]), np.arange(phi1_pred.shape[0]))

    # 绘制曲面图
    surf = ax.plot_surface(X, Y, phi1_pred, cmap='afmhot_r', vmin=np.min(phi1_pred), vmax=np.max(phi1_pred),
                           edgecolor='none')

    # 添加颜色条
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    # 设置标签和标题
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_zlabel('phi1_final', fontsize=12)
    ax.set_title('ND PINN Solution', fontsize=14)

    # 绘制虚线：x=0.5和y=0.5
    ax.axvline(x=0.5, color='black', linestyle='--', linewidth=1)
    ax.axhline(y=0.5, color='black', linestyle='--', linewidth=1)

    # 设置坐标轴刻度字体大小
    ax.tick_params(axis='both', which='major', labelsize=10)





    fig = plt.figure(figsize=(8, 6))  # 设置图形尺寸
    ax = fig.add_subplot(111, projection='3d')  # 创建3D轴

    # 将phi1_final的值转换为X, Y网格
    X, Y = np.meshgrid(np.arange(phi1_true.shape[1]), np.arange(phi1_true.shape[0]))

    # 绘制曲面图
    surf = ax.plot_surface(X, Y, phi1_true, cmap='afmhot_r', vmin=np.min(phi1_true), vmax=np.max(phi1_true),
                           edgecolor='none')

    # 添加颜色条
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    # 设置标签和标题
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_zlabel('phi1_final', fontsize=12)
    ax.set_title('ND Numerical Solution', fontsize=14)

    # 绘制虚线：x=0.5和y=0.5
    ax.axvline(x=0.5, color='black', linestyle='--', linewidth=1)
    ax.axhline(y=0.5, color='black', linestyle='--', linewidth=1)

    # 设置坐标轴刻度字体大小
    ax.tick_params(axis='both', which='major', labelsize=10)


    # 显示图像
    plt.show()



def result_plot(phi1_pred_hard, phi1_pred_soft, phi1_true):
    # fig, ax = plt.subplots(figsize=(8, 6))  # 设置图形尺寸
    # im = ax.imshow(np.flipud(phi1_true), cmap='turbo', vmin=0.0, vmax=1.05,
    #                 interpolation='nearest', aspect='auto', extent=[0, 1, 0, 1])
    # cb = plt.colorbar(im)  # 添加颜色条
    # cb.ax.tick_params(labelsize=20, length=8)
    # # cb.set_label('$\phi$', fontsize=14, fontweight='bold')  # 在colorbar旁边添加希腊字符phi
    # # plt.xlabel('x', fontsize=14, fontweight='bold')  # 设置x轴标签
    # # plt.ylabel('y', fontsize=14, fontweight='bold')  # 设置y轴标签
    # # plt.title('Case 4 Numerical Solution', fontsize=18, fontweight='bold')  # 设置标题
    # # # 绘制第一个圆：(0.2, 0.2)，半径0.15
    # circle1 = plt.Circle((0.2, 0.2), 0.15, color='black', fill=False, linestyle='--', linewidth=2)
    # ax.add_artist(circle1)
    # # ax.text(0.2, 0.35, 'Interface', color='black', va='bottom', ha='center', transform=ax.transAxes, fontsize=18)
    # #
    # # # 绘制第二个圆：(0.6, 0.2)，半径0.15
    # circle2 = plt.Circle((0.6, 0.2), 0.15, color='black', fill=False, linestyle='--', linewidth=2)
    # ax.add_artist(circle2)
    # # ax.text(0.6, 0.35, 'Interface', color='black', va='bottom', ha='center', transform=ax.transAxes, fontsize=18)
    #
    # # # 设置x轴的主要刻度位置
    # ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # ax.tick_params(axis='x', length=8)  # 设置次要刻度线长度和隐藏刻度数值
    #
    # # # 添加次要刻度位置，刻度线短且不显示刻度数值
    # ax.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9], minor=True)
    # ax.tick_params(axis='x', which='minor', length=4, labelsize=0)  # 设置次要刻度线长度和隐藏刻度数值
    #
    # # # 设置y轴的主要刻度位置
    # ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # ax.tick_params(axis='y', length=8)  # 设置次要刻度线长度和隐藏刻度数值
    #
    # # # 添加次要刻度位置，刻度线短且不显示刻度数值
    # ax.set_yticks([0.1, 0.3, 0.5, 0.7, 0.9], minor=True)
    # ax.tick_params(axis='y', which='minor', length=4, labelsize=0)  # 设置次要刻度线长度和隐藏刻度数值
    #
    # # # 在(0,0)点画一个大黑点
    # ax.plot(0, 0, 'ko', markersize=12, color='black')  # 'ko' 表示黑色圆点
    # # # 在(0,0)点旁边添加文本"Fixed Point: (0,0,1.0)"
    # # ax.text(0.02, 0.01, 'Fixed Point: (0, 0, 1)', color='black', va='bottom', ha='left', transform=ax.transAxes, fontsize=18)
    #
    # plt.xticks(fontsize=20)  # 设置x轴刻度字体大小
    # plt.yticks(fontsize=20)  # 设置y轴刻度字体大小
    # # plt.grid()  # 添加网格线
    # plt.axis('on')  # 移除坐标轴边框

    # fig, ax = plt.subplots(figsize=(8, 6))  # 设置图形尺寸
    # im = ax.imshow(np.flipud(phi1_pred_hard), cmap='turbo', vmin=0.0, vmax=1.05,
    #                 interpolation='nearest', aspect='auto', extent=[0, 1, 0, 1])
    # cb = plt.colorbar(im)  # 添加颜色条
    # cb.ax.tick_params(labelsize=20, length=8)
    # # cb.set_label('$\phi$', fontsize=14, fontweight='bold')  # 在colorbar旁边添加希腊字符phi
    # # plt.xlabel('x', fontsize=14, fontweight='bold')  # 设置x轴标签
    # # plt.ylabel('y', fontsize=14, fontweight='bold')  # 设置y轴标签
    # # plt.title('Case 4 FC-PINNs(Hard-constraint) Result', fontsize=18, fontweight='bold')  # 设置标题
    # # 绘制第一个圆：(0.2, 0.2)，半径0.15
    # circle1 = plt.Circle((0.2, 0.2), 0.15, color='black', fill=False, linestyle='--', linewidth=2)
    # ax.add_artist(circle1)
    # # ax.text(0.2, 0.35, 'Interface', color='black', va='bottom', ha='center', transform=ax.transAxes, fontsize=18)
    #
    # # 绘制第二个圆：(0.6, 0.2)，半径0.15
    # circle2 = plt.Circle((0.6, 0.2), 0.15, color='black', fill=False, linestyle='--', linewidth=2)
    # ax.add_artist(circle2)
    # # ax.text(0.6, 0.35, 'Interface', color='black', va='bottom', ha='center', transform=ax.transAxes, fontsize=18)
    #
    # # 在(0,0)点画一个大黑点
    # ax.plot(0, 0, 'ko', markersize=12, color='black')  # 'ko' 表示黑色圆点
    # # 在(0,0)点旁边添加文本"Fixed Point: (0,0,1.0)"
    # # ax.text(0.02, 0.01, 'Fixed Point: (0, 0, 1)', color='black', va='bottom', ha='left', transform=ax.transAxes,
    # #         fontsize=18)
    #
    # # # 设置x轴的主要刻度位置
    # ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # ax.tick_params(axis='x', length=8)  # 设置次要刻度线长度和隐藏刻度数值
    #
    # # # 添加次要刻度位置，刻度线短且不显示刻度数值
    # ax.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9], minor=True)
    # ax.tick_params(axis='x', which='minor', length=4, labelsize=0)  # 设置次要刻度线长度和隐藏刻度数值
    #
    # # # 设置y轴的主要刻度位置
    # ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # ax.tick_params(axis='y', length=8)  # 设置次要刻度线长度和隐藏刻度数值
    #
    # # # 添加次要刻度位置，刻度线短且不显示刻度数值
    # ax.set_yticks([0.1, 0.3, 0.5, 0.7, 0.9], minor=True)
    # ax.tick_params(axis='y', which='minor', length=4, labelsize=0)  # 设置次要刻度线长度和隐藏刻度数值
    #
    # plt.xticks(fontsize=20)  # 设置x轴刻度字体大小
    # plt.yticks(fontsize=20)  # 设置y轴刻度字体大小
    # # plt.grid()  # 添加网格线
    # plt.axis('on')  # 移除坐标轴边框

    fig, ax = plt.subplots(figsize=(8, 6))  # 设置图形尺寸
    im = ax.imshow(np.flipud(phi1_pred_soft), cmap='turbo', vmin=0.0, vmax=1.05,
                    interpolation='nearest', aspect='auto', extent=[0, 1, 0, 1])
    cb = plt.colorbar(im)  # 添加颜色条
    cb.ax.tick_params(labelsize=20, length=8)
    # cb.set_label('$\phi$', fontsize=14, fontweight='bold')  # 在colorbar旁边添加希腊字符phi
    # plt.xlabel('x', fontsize=14, fontweight='bold')  # 设置x轴标签
    # plt.ylabel('y', fontsize=14, fontweight='bold')  # 设置y轴标签
    # plt.title('Case 4 FC-PINNs(Soft-constraint) Result', fontsize=18, fontweight='bold')  # 设置标题
    # 绘制第一个圆：(0.2, 0.2)，半径0.15
    circle1 = plt.Circle((0.2, 0.2), 0.15, color='black', fill=False, linestyle='--', linewidth=2)
    ax.add_artist(circle1)
    # ax.text(0.2, 0.35, 'Interface', color='black', va='bottom', ha='center', transform=ax.transAxes, fontsize=18)

    # 绘制第二个圆：(0.6, 0.2)，半径0.15
    circle2 = plt.Circle((0.6, 0.2), 0.15, color='black', fill=False, linestyle='--', linewidth=2)
    ax.add_artist(circle2)
    # ax.text(0.6, 0.35, 'Interface', color='black', va='bottom', ha='center', transform=ax.transAxes, fontsize=18)

    # 在(0,0)点画一个大黑点
    ax.plot(0, 0, 'ko', markersize=12, color='black')  # 'ko' 表示黑色圆点
    # 在(0,0)点旁边添加文本"Fixed Point: (0,0,1.0)"
    # ax.text(0.02, 0.01, 'Fixed Point: (0, 0, 1)', color='black', va='bottom', ha='left', transform=ax.transAxes,
    #         fontsize=18)

    # # 设置x轴的主要刻度位置
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis='x', length=8)  # 设置次要刻度线长度和隐藏刻度数值

    # # 添加次要刻度位置，刻度线短且不显示刻度数值
    ax.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9], minor=True)
    ax.tick_params(axis='x', which='minor', length=4, labelsize=0)  # 设置次要刻度线长度和隐藏刻度数值

    # # 设置y轴的主要刻度位置
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis='y', length=8)  # 设置次要刻度线长度和隐藏刻度数值

    # # 添加次要刻度位置，刻度线短且不显示刻度数值
    ax.set_yticks([0.1, 0.3, 0.5, 0.7, 0.9], minor=True)
    ax.tick_params(axis='y', which='minor', length=4, labelsize=0)  # 设置次要刻度线长度和隐藏刻度数值

    plt.xticks(fontsize=20)  # 设置x轴刻度字体大小
    plt.yticks(fontsize=20)  # 设置y轴刻度字体大小
    # plt.grid()  # 添加网格线
    plt.axis('on')  # 移除坐标轴边框

    plt.show()


def error_plot(phi1_pred_hard, phi1_pred_soft, phi1_true):
    colors = ["white", "orange", "red"]  # 定义颜色列表
    cmap_name = "white_to_red"  # 颜色映射的名称
    n_bins = 100  # 颜色的细分数量
    cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

    fig, ax = plt.subplots(figsize=(8, 6))  # 设置图形尺寸
    im = ax.imshow(np.abs(np.flipud(phi1_true - phi1_pred_hard)) * 100, cmap=cmap, vmin=0, vmax=1.0,
                   interpolation='nearest', aspect='auto', extent=[0, 1, 0, 1])
    cb = plt.colorbar(im)  # 添加颜色条
    cb.ax.tick_params(labelsize=20, length=8)
    # cb.set_label('$\phi$ True - $\phi$ Pred', fontsize=14, fontweight='bold')  # 在colorbar旁边添加希腊字符phi
    # plt.xlabel('x', fontsize=14, fontweight='bold')  # 设置x轴标签
    # plt.ylabel('y', fontsize=14, fontweight='bold')  # 设置y轴标签
    # plt.title('Case 4 AE Distribution(Hard-constraint)', fontsize=18, fontweight='bold')  # 设置标题

    # 绘制第一个圆：(0.2, 0.2)，半径0.15
    circle1 = plt.Circle((0.2, 0.2), 0.15, color='black', fill=False, linestyle='--', linewidth=2)
    ax.add_artist(circle1)
    # ax.text(0.2, 0.35, 'Interface', color='black', va='bottom', ha='center', transform=ax.transAxes, fontsize=18)

    # 绘制第二个圆：(0.6, 0.2)，半径0.15
    circle2 = plt.Circle((0.6, 0.2), 0.15, color='black', fill=False, linestyle='--', linewidth=2)
    ax.add_artist(circle2)
    # ax.text(0.6, 0.35, 'Interface', color='black', va='bottom', ha='center', transform=ax.transAxes, fontsize=18)

    # 在(0,0)点画一个大黑点
    # ax.plot(0, 0, 'ko', markersize=12)  # 'ko' 表示黑色圆点
    # 在(0,0)点旁边添加文本"Fixed Point: (0,0,1.0)"
    # ax.text(0.02, 0.01, 'Fixed Point: (0, 0, 1)', color='black', va='bottom', ha='left', transform=ax.transAxes, fontsize=18)

    # # 设置x轴的主要刻度位置
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis='x', length=8)  # 设置次要刻度线长度和隐藏刻度数值

    # # 添加次要刻度位置，刻度线短且不显示刻度数值
    ax.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9], minor=True)
    ax.tick_params(axis='x', which='minor', length=4, labelsize=0)  # 设置次要刻度线长度和隐藏刻度数值

    # # 设置y轴的主要刻度位置
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis='y', length=8)  # 设置次要刻度线长度和隐藏刻度数值

    # # 添加次要刻度位置，刻度线短且不显示刻度数值
    ax.set_yticks([0.1, 0.3, 0.5, 0.7, 0.9], minor=True)
    ax.tick_params(axis='y', which='minor', length=4, labelsize=0)  # 设置次要刻度线长度和隐藏刻度数值

    plt.xticks(fontsize=20)  # 设置x轴刻度字体大小
    plt.yticks(fontsize=20)  # 设置y轴刻度字体大小
    plt.axis('on')  # 移除坐标轴边框

    # fig, ax = plt.subplots(figsize=(8, 6))  # 设置图形尺寸
    # im = ax.imshow(np.abs(np.flipud(phi1_true - phi1_pred_soft)) * 100, cmap=cmap, vmin=0, vmax=1,
    #                interpolation='nearest', aspect='auto', extent=[0, 1, 0, 1])
    # cb = plt.colorbar(im)  # 添加颜色条
    # cb.ax.tick_params(labelsize=20, length=8)
    # # cb.set_label('$\phi$ True - $\phi$ Pred', fontsize=14, fontweight='bold')  # 在colorbar旁边添加希腊字符phi
    # # plt.xlabel('x', fontsize=14)  # 设置x轴标签
    # # plt.ylabel('y', fontsize=14)  # 设置y轴标签
    # # plt.title('Case 4 AE Distribution(Soft-constraint)', fontsize=18, fontweight='bold')  # 设置标题
    #
    # circle1 = plt.Circle((0.2, 0.2), 0.15, color='black', fill=False, linestyle='--', linewidth=2)
    # ax.add_artist(circle1)
    # # ax.text(0.2, 0.35, 'Interface', color='black', va='bottom', ha='center', transform=ax.transAxes, fontsize=18)
    #
    # # 绘制第二个圆：(0.6, 0.2)，半径0.15
    # circle2 = plt.Circle((0.6, 0.2), 0.15, color='black', fill=False, linestyle='--', linewidth=2)
    # ax.add_artist(circle2)
    # # ax.text(0.6, 0.35, 'Interface', color='black', va='bottom', ha='center', transform=ax.transAxes, fontsize=18)
    #
    # # # 设置x轴的主要刻度位置
    # ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # ax.tick_params(axis='x', length=8)  # 设置次要刻度线长度和隐藏刻度数值
    #
    # # # 添加次要刻度位置，刻度线短且不显示刻度数值
    # ax.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9], minor=True)
    # ax.tick_params(axis='x', which='minor', length=4, labelsize=0)  # 设置次要刻度线长度和隐藏刻度数值
    #
    # # # 设置y轴的主要刻度位置
    # ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # ax.tick_params(axis='y', length=8)  # 设置次要刻度线长度和隐藏刻度数值
    #
    # # # 添加次要刻度位置，刻度线短且不显示刻度数值
    # ax.set_yticks([0.1, 0.3, 0.5, 0.7, 0.9], minor=True)
    # ax.tick_params(axis='y', which='minor', length=4, labelsize=0)  # 设置次要刻度线长度和隐藏刻度数值
    #
    # # 在(0,0)点画一个大黑点
    # # ax.plot(0, 0, 'ko', markersize=12)  # 'ko' 表示黑色圆点
    # # 在(0,0)点旁边添加文本"Fixed Point: (0,0,1.0)"
    # # ax.text(0.02, 0.01, 'Fixed Point: (0, 0, 1)', color='black', va='bottom', ha='left', transform=ax.transAxes, fontsize=14)
    #
    # plt.xticks(fontsize=20)  # 设置x轴刻度字体大小
    # plt.yticks(fontsize=20)  # 设置y轴刻度字体大小
    # plt.axis('on')  # 移除坐标轴边框


    plt.show()


def loss_plot():
    case_4_loss_keff = pd.read_excel('case_4_loss_keff.xlsx', engine='openpyxl')

    loss_iteration = case_4_loss_keff.iloc[:, 0].values
    PDE_fuel_1_loss = case_4_loss_keff.iloc[:, 1].values
    PDE_fuel_2_loss = case_4_loss_keff.iloc[:, 2].values
    PDE_fluent_loss = case_4_loss_keff.iloc[:, 3].values
    circle_1_interface_bc_continuity_flux_loss = case_4_loss_keff.iloc[:, 4].values
    circle_1_interface_bc_continuity_neutron_loss = case_4_loss_keff.iloc[:, 5].values
    circle_2_interface_bc_continuity_flux_loss = case_4_loss_keff.iloc[:, 6].values
    circle_2_interface_bc_continuity_neutron_loss = case_4_loss_keff.iloc[:, 7].values
    fluent_bc_x_left_loss = case_4_loss_keff.iloc[:, 8].values
    fluent_bc_x_right_loss = case_4_loss_keff.iloc[:, 9].values
    fluent_extrapolated_bc_x_right_loss = case_4_loss_keff.iloc[:, 10].values
    fluent_bc_y_down_loss = case_4_loss_keff.iloc[:, 11].values
    fluent_bc_y_up_loss = case_4_loss_keff.iloc[:, 12].values
    fluent_extrapolated_bc_y_up_loss = case_4_loss_keff.iloc[:, 13].values

    loss_iteration_soft = case_4_loss_keff.iloc[:, 16].values
    PDE_fuel_1_loss_soft = case_4_loss_keff.iloc[:, 17].values
    PDE_fuel_2_loss_soft = case_4_loss_keff.iloc[:, 18].values
    PDE_fluent_loss_soft = case_4_loss_keff.iloc[:, 19].values
    circle_1_interface_bc_continuity_flux_loss_soft = case_4_loss_keff.iloc[:, 20].values
    circle_1_interface_bc_continuity_neutron_loss_soft = case_4_loss_keff.iloc[:, 21].values
    circle_2_interface_bc_continuity_flux_loss_soft = case_4_loss_keff.iloc[:, 22].values
    circle_2_interface_bc_continuity_neutron_loss_soft = case_4_loss_keff.iloc[:, 23].values
    fluent_bc_x_left_loss_soft = case_4_loss_keff.iloc[:, 24].values
    fluent_bc_x_right_loss_soft = case_4_loss_keff.iloc[:, 25].values
    fluent_extrapolated_bc_x_right_loss_soft = case_4_loss_keff.iloc[:, 26].values
    fluent_bc_y_down_loss_soft = case_4_loss_keff.iloc[:, 27].values
    fluent_bc_y_up_loss_soft = case_4_loss_keff.iloc[:, 28].values
    fluent_extrapolated_bc_y_up_loss_soft = case_4_loss_keff.iloc[:, 29].values
    fixed_point_loss_soft = case_4_loss_keff.iloc[:, 30].values


    fig, ax = plt.subplots(figsize=(8, 5.5))

    # 使用不同的标记和颜色来区分不同的数据集
    ax.plot(loss_iteration, circle_1_interface_bc_continuity_neutron_loss, label='Hard-constraint Interface', linewidth=1.5, linestyle='-', color='#2486b9')
    # ax.plot(loss_iteration, fluent_extrapolated_bc_x_right_loss, label='Hard-constraint Extrapolated Boundary', linewidth=1.5, linestyle='-', color='#93b5cf')
    ax.plot(loss_iteration_soft, circle_1_interface_bc_continuity_neutron_loss_soft, label='Soft-constraint Interface', linewidth=1.5, linestyle='--', color='#fb9968')
    # ax.plot(loss_iteration_soft, fluent_extrapolated_bc_x_right_loss_soft, label='Soft-constraint Extrapolated Boundary', linewidth=1.5, linestyle='--', color='#ed5126')
    ax.plot(loss_iteration_soft, fixed_point_loss_soft, label='Soft-constraint Fixed Point', linewidth=1.5, linestyle='--', color='#e60012')

    # 设置图例，确保图例不遮挡数据点
    ax.legend(loc='best', fontsize=18, framealpha=0.9)

    # 设置标题和轴标签，确保字体大小和样式一致
    ax.set_title('Case 4 Loss', fontsize=18, fontweight='bold')
    ax.set_xlabel('Iteration', fontsize=14, fontweight='bold')
    ax.set_ylabel('Loss Value', fontsize=14, fontweight='bold')  # 使用LaTeX语法设置希腊字符φ

    # 设置刻度参数，确保刻度标签清晰
    ax.tick_params(axis='both', which='major', labelsize=15, direction='in', right=True, top=True)

    # 设置x轴的范围为0到1
    ax.set_yscale('log')
    # 自定义y轴刻度
    # majorLocator = ticker.LogLocator(base=10.0, numticks=10)  # 设置主要刻度数量
    # minorLocator = ticker.LogLocator(base=10.0, subs=np.arange(0.1, 1.0, 0.1), numticks=15)  # 设置次要刻度
    #
    # ax.yaxis.set_major_locator(majorLocator)
    # ax.yaxis.set_minor_locator(minorLocator)
    # ax.set_ylim([0, 0.01])
    ax.set_xlim([0, 201000])

    ax.axvline(x=20000, color='black', linestyle='--', linewidth=1)
    ax.text(20000 - 1000, 4e-10, 'adam', color='black', va='center', ha='right', fontsize=18)
    ax.text(20000 + 1000, 4e-10, 'L-BFGS', color='black', va='center', ha='left', fontsize=18)


    # 调整布局，确保图表紧凑
    plt.tight_layout()

    # 显示图表
    plt.show()


def keff_plot():
    case_4_loss_keff = pd.read_excel('case_4_loss_keff.xlsx', engine='openpyxl')
    keff_iteration = case_4_loss_keff.iloc[:, 14].values
    keff_value = case_4_loss_keff.iloc[:, 15].values
    keff_iteration_soft = case_4_loss_keff.iloc[:, 31].values
    keff_value_soft = case_4_loss_keff.iloc[:, 32].values


    fig, ax = plt.subplots(figsize=(8, 4))

    # 使用不同的标记和颜色来区分不同的数据集
    ax.plot(keff_iteration[::2] / 10000, keff_value[::2], label='FC-PINNs(Hard-constraint)', linewidth=2, color='red')
    ax.plot(keff_iteration_soft[::2] / 10000, keff_value_soft[::2], label='FC-PINNs(Soft-constraint)', linewidth=2, color='orange')


    # 设置图例，确保图例不遮挡数据点
    # ax.legend(loc='best', fontsize=18, framealpha=0.9)

    # 设置标题和轴标签，确保字体大小和样式一致
    # ax.set_title('Case 4 Keff', fontsize=18, fontweight='bold')
    # ax.set_xlabel('Iteration', fontsize=14, fontweight='bold')
    # ax.set_ylabel('Keff Value', fontsize=14, fontweight='bold')  # 使用LaTeX语法设置希腊字符φ

    # 设置刻度参数，确保刻度标签清晰
    ax.tick_params(axis='both', which='major', labelsize=20, direction='out')

    # 隐藏上边和右边的坐标轴线
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # # 设置x轴的主要刻度位置
    ax.set_xticks([0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5])
    ax.tick_params(axis='x', length=8)  # 设置次要刻度线长度和隐藏刻度数值

    # # 添加次要刻度位置，刻度线短且不显示刻度数值
    ax.set_xticks([1.25, 3.75, 6.25, 8.75, 11.25, 13.75, 16.25], minor=True)
    ax.tick_params(axis='x', which='minor', length=4, labelsize=0)  # 设置次要刻度线长度和隐藏刻度数值
    #
    # # 设置y轴的主要刻度位置
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis='y', length=8)  # 设置次要刻度线长度和隐藏刻度数值

    # # 添加次要刻度位置，刻度线短且不显示刻度数值
    ax.set_yticks([0.3, 0.5, 0.7, 0.9], minor=True)
    ax.tick_params(axis='y', which='minor', length=4, labelsize=0)  # 设置次要刻度线长度和隐藏刻度数值

    # 设置x轴的范围为0到1
    ax.set_ylim([0.19, 1.05])
    ax.set_xlim([-0.36, 18.36])

    # ax.axvline(x=20000, color='black', linestyle='--', linewidth=1)
    # ax.text(20000 - 1000, 0.7, 'adam', color='black', va='center', ha='right', fontsize=18)
    # ax.text(20000 + 1000, 0.7, 'L-BFGS', color='black', va='center', ha='left', fontsize=18)

    # 在y=0.9处画一条横线
    ax.axhline(0.24662768412482594, color='grey', linestyle='--', linewidth=1)

    # # 在横线旁边添加带有箭头的文字“keff True”
    # ax.annotate('Keff True', xy=(150000, 0.24662768412482594), xytext=(150000, 0.34662768412482594),
    #             arrowprops=dict(facecolor='black', arrowstyle='->', connectionstyle='arc3'),
    #             ha='center', va='center', fontsize=18)

    # 调整布局，确保图表紧凑
    plt.tight_layout()

    # 显示图表
    plt.show()



def main():
    
    # Numerical results for 2D Case 4
    # phi1_true, keff_true = set_ND_1group_para_2D_case_2()
    
    # np.save('ndpinn_2D_example_2_true.npy', phi1_true)
    # print("keff_true =", keff_true)
    phi1_true = np.load('ndpinn_2D_example_2_true.npy').T
    # 生成原始的网格点
    x_true = np.linspace(0, 1, 171)
    y_true = np.linspace(0, 1, 171)
    X_true, Y_true = np.meshgrid(x_true, y_true)
    x_new = np.linspace(0, 1, 101)
    y_new = np.linspace(0, 1, 101)
    X_new, Y_new = np.meshgrid(x_new, y_new)

    phi1_true_new = griddata((X_true.flatten(), Y_true.flatten()), phi1_true.flatten(), (X_new, Y_new), method='cubic')

    # print("phi1_true_new=", phi1_true_new.shape)

    # Run FC-PINNs for 2D Case 2 (Case 4 in total)
    # phi1_pred_soft, keff_pred = set_pinn_nd()
    
    # # keff_true = 0.24662768412482594
    # # keff_pred_hard = 0.2487
    # # keff_pred_soft = 0.2487
    # np.save('ndpinn_2D_example_2_pred_soft.npy', phi1_pred_soft)


    # phi1_wrong = np.load('ndpinn_2D_example_1_wrong.npy')
    phi1_pred_soft = np.load('ndpinn_2D_example_2_pred_soft.npy')
    phi1_pred_hard = np.load('ndpinn_2D_example_2_pred.npy')
    # phi1_wrong = (phi1_wrong * 1.0 / phi1_wrong[0, 0]).reshape(101, 101)
    phi1_true = (phi1_true_new * 1.0 / phi1_true_new[0, 0]).reshape(101, 101)

    # print("RMSE_hard =", calculate_RMSE(phi1_true, phi1_pred_hard))
    # print("R_square_hard =", r2_score(phi1_true, phi1_pred_hard))
    # print("RMSE_soft =", calculate_RMSE(phi1_true, phi1_pred_soft))
    # print("R_square_soft =", r2_score(phi1_true, phi1_pred_soft))
    # RMSE_hard = 0.001247135619933948
    # R_square_hard = 0.9999393743710558
    # RMSE_soft = 0.0011008154498475102
    # R_square_soft = 0.9999513026246907
    # print("L2_relative_error1=", calculate_relative_l2_error(phi1_pred, phi1_true))


    # result_surface_plot(phi1_pred_soft, phi1_true)
    # result_plot(phi1_pred_hard, phi1_pred_soft, phi1_true)
    # error_plot(phi1_pred_hard, phi1_pred_soft, phi1_true)
    # loss_plot()
    keff_plot()







if __name__ == "__main__":
    main()
