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
from PDE_solver import set_ND_1group_para_2D_case_1, calculate_RMSE
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
x_interface = 0.5
y_interface = 0.5
extrapolated_ratio = 2.0
keff = dde.Variable(1.0)

def pde_nd(x, y):
    # x = x[:, 0:1]
    # y = x[:, 1:2]
    phi1_core = y[:, 0:1]
    phi1_cladding = y[:, 1:2]

    dphi1_core_dxx = dde.grad.hessian(phi1_core, x, i=0, j=0)
    dphi1_core_dyy = dde.grad.hessian(phi1_core, x, i=1, j=1)

    dphi1_cladding_dxx = dde.grad.hessian(phi1_cladding, x, i=0, j=0)
    dphi1_cladding_dyy = dde.grad.hessian(phi1_cladding, x, i=1, j=1)

    D_core_1 = 0.05
    sigma_core_r_1 = 0.15
    X_core_p_1 = 0.3
    D_cladding_1 = 0.10
    sigma_cladding_r_1 = 0.01
    X_cladding_p_1 = 0.0

    nd1_core_pde = - (D_core_1 * (dphi1_core_dxx + dphi1_core_dyy)) + sigma_core_r_1 * phi1_core - X_core_p_1 / keff * phi1_core
    nd1_cladding_pde = - (D_cladding_1 * (dphi1_cladding_dxx + dphi1_cladding_dyy)) + sigma_cladding_r_1 * phi1_cladding - X_cladding_p_1 / keff * phi1_cladding

    nd1_core = torch.where((x[:, 1:2] < y_interface) & (x[:, 0:1] < x_interface), nd1_core_pde, 0.0)
    nd1_cladding = torch.where((x[:, 1:2] < y_interface) & (x[:, 0:1] < x_interface), 0.0, nd1_cladding_pde)

    return [nd1_core, nd1_cladding]

    # return torch.where(x < interface, nd1_left_pde, nd1_right_pde)

def boundary_x_left_core_func(x, y, X):
    phi1_core = y[:, 0:1]

    dphi1_core_dx = dde.grad.jacobian(phi1_core, x, i=0, j=0)

    return dphi1_core_dx - 0.0

def boundary_x_left_cladding_func(x, y, X):
    phi1_cladding = y[:, 1:2]

    dphi1_cladding_dx = dde.grad.jacobian(phi1_cladding, x, i=0, j=0)

    return dphi1_cladding_dx - 0.0


def boundary_x_right_cladding_func(x, y, X):
    phi1_cladding = y[:, 1:2]

    D_cladding_1 = 0.10

    dphi1_cladding_dx = dde.grad.jacobian(phi1_cladding, x, i=0, j=0)

    return dphi1_cladding_dx + phi1_cladding / (2 * D_cladding_1)

def boundary_y_down_core_func(x, y, X):
    phi1_core = y[:, 0:1]

    dphi1_cladding_dy = dde.grad.jacobian(phi1_core, x, i=0, j=1)

    return dphi1_cladding_dy - 0.0

def boundary_y_down_cladding_func(x, y, X):
    phi1_cladding = y[:, 1:2]

    dphi1_cladding_dy = dde.grad.jacobian(phi1_cladding, x, i=0, j=1)

    return dphi1_cladding_dy - 0.0

def boundary_y_up_cladding_func(x, y, X):
    phi1_cladding = y[:, 1:2]

    D_cladding_1 = 0.10

    dphi1_cladding_dy = dde.grad.jacobian(phi1_cladding, x, i=0, j=1)

    return dphi1_cladding_dy + phi1_cladding / (2 * D_cladding_1)


def boundary_x_extrapolated_right_cladding_func(x, y, X):
    phi1_cladding = y[:, 1:2]

    return phi1_cladding - 0.0

def boundary_y_extrapolated_up_cladding_func(x, y, X):
    phi1_cladding = y[:, 1:2]

    return phi1_cladding - 0.0


def continuity_flux_x_interface_func(x, y, X):
    phi1_core = y[:, 0:1]
    phi1_cladding = y[:, 1:2]

    dphi1_core_dx = dde.grad.jacobian(phi1_core, x, i=0, j=0)
    dphi1_cladding_dx = dde.grad.jacobian(phi1_cladding, x, i=0, j=0)

    D_core_1 = 0.05
    D_cladding_1 = 0.10

    return D_core_1 * dphi1_core_dx - D_cladding_1 * dphi1_cladding_dx

def continuity_neutron_x_interface_func(x, y, X):
    phi1_core = y[:, 0:1]
    phi1_cladding = y[:, 1:2]

    return phi1_core - phi1_cladding

def continuity_flux_y_interface_func(x, y, X):
    phi1_core = y[:, 0:1]
    phi1_cladding = y[:, 1:2]

    dphi1_core_dy = dde.grad.jacobian(phi1_core, x, i=0, j=1)
    dphi1_cladding_dy = dde.grad.jacobian(phi1_cladding, x, i=0, j=1)

    D_core_1 = 0.05
    D_cladding_1 = 0.10

    return D_core_1 * dphi1_core_dy - D_cladding_1 * dphi1_cladding_dy

def continuity_neutron_y_interface_func(x, y, X):
    phi1_core = y[:, 0:1]
    phi1_cladding = y[:, 1:2]

    return phi1_core - phi1_cladding

def fixed_point_func(x, y, X):
    phi1_core = y[:, 0:1]

    return phi1_core - 1.0


def output_transform(x, y):
    phi1_core = y[:, 0:1]
    phi1_cladding = y[:, 1:2]

    phi1_core_transformed = phi1_core * ((0.0 - x[:, 0:1]) ** 2 + (0.0 - x[:, 1:2]) ** 2) + 1.0
    phi1_cladding_transformed = phi1_cladding
    y_new = torch.cat((phi1_core_transformed, phi1_cladding_transformed), dim=1)
    return y_new


def set_pinn_nd():
    geom = dde.geometry.Rectangle([left, down], [right, up])
    D_cladding_1 = 0.10
    x_interface_points = np.vstack((np.full((50), x_interface), np.linspace(down, y_interface, num=50))).T    # [100, 2]
    y_interface_points = np.vstack((np.linspace(left, x_interface, num=50), np.full((50), y_interface))).T    # [100, 2]
    phi1_left_BC_core_points = np.vstack((np.full((50), left), np.linspace(down, y_interface, num=50))).T    # [100, 2]
    phi1_left_BC_cladding_points = np.vstack((np.full((50), left), np.linspace(y_interface, up, num=50))).T    # [100, 2]
    phi1_right_BC_cladding_points = np.vstack((np.full((100), right), np.linspace(down, up, num=100))).T    # [100, 2]
    phi1_down_BC_core_points = np.vstack((np.linspace(left, x_interface, num=50), np.full((50), down))).T    # [100, 2]
    phi1_down_BC_cladding_points = np.vstack((np.linspace(x_interface, right, num=50), np.full((50), down))).T    # [100, 2]
    phi1_up_BC_cladding_points = np.vstack((np.linspace(left, right, num=100), np.full((100), up))).T    # [100, 2]
    phi1_right_extrapolated_BC_points = np.vstack((np.full((100), right + extrapolated_ratio * D_cladding_1), np.linspace(down, up, num=100))).T    # [100, 2]
    phi1_up_extrapolated_BC_points = np.vstack((np.linspace(left, right, num=100), np.full((100), up + extrapolated_ratio * D_cladding_1))).T    # [100, 2]
    fixed_point = np.vstack((np.full((1), 0.0), np.full((1), 0.0))).T    # [100, 2]

    x_interface_bc_continuity_flux = dde.icbc.PointSetOperatorBC(x_interface_points, 0.0, continuity_flux_x_interface_func)
    x_interface_bc_continuity_neutron = dde.icbc.PointSetOperatorBC(x_interface_points, 0.0, continuity_neutron_x_interface_func)
    y_interface_bc_continuity_flux = dde.icbc.PointSetOperatorBC(y_interface_points, 0.0, continuity_flux_y_interface_func)
    y_interface_bc_continuity_neutron = dde.icbc.PointSetOperatorBC(y_interface_points, 0.0, continuity_neutron_y_interface_func)

    core_bc_x_left = dde.icbc.PointSetOperatorBC(phi1_left_BC_core_points, 0.0, boundary_x_left_core_func)
    cladding_bc_x_left = dde.icbc.PointSetOperatorBC(phi1_left_BC_cladding_points, 0.0, boundary_x_left_cladding_func)
    cladding_bc_x_right = dde.icbc.PointSetOperatorBC(phi1_right_BC_cladding_points, 0.0, boundary_x_right_cladding_func)
    cladding_extrapolated_bc_x_right = dde.icbc.PointSetOperatorBC(phi1_right_extrapolated_BC_points, 0.0, boundary_x_extrapolated_right_cladding_func)
    core_bc_y_down = dde.icbc.PointSetOperatorBC(phi1_down_BC_core_points, 0.0, boundary_y_down_core_func)
    cladding_bc_y_down = dde.icbc.PointSetOperatorBC(phi1_down_BC_cladding_points, 0.0, boundary_y_down_cladding_func)
    cladding_bc_y_up = dde.icbc.PointSetOperatorBC(phi1_up_BC_cladding_points, 0.0, boundary_y_up_cladding_func)
    cladding_extrapolated_bc_y_up = dde.icbc.PointSetOperatorBC(phi1_up_extrapolated_BC_points, 0.0, boundary_y_extrapolated_up_cladding_func)

    fixed_point_constraint = dde.icbc.PointSetOperatorBC(fixed_point, 0.0, fixed_point_func)

    BC = [x_interface_bc_continuity_flux, x_interface_bc_continuity_neutron, y_interface_bc_continuity_flux, y_interface_bc_continuity_neutron, core_bc_x_left,
          cladding_bc_x_left, cladding_bc_x_right, cladding_extrapolated_bc_x_right, core_bc_y_down, cladding_bc_y_down, cladding_bc_y_up, cladding_extrapolated_bc_y_up, fixed_point_constraint]
    # loss_weights = [100000, 1, 100, 100, 100, 10000]
    # loss_weights_hard = [1000, 1, 100, 100, 100, 100, 100, 100, 1, 1, 100, 100, 1, 1]
    loss_weights_soft = [1000, 1, 100, 100, 100, 100, 100, 100, 1, 1, 100, 100, 1, 1, 1]

    data = dde.data.PDE(
        geom,
        pde_nd,
        BC,
        num_domain=5000,
        num_test=5000,
    )

    net = dde.nn.pytorch.FNN([2] + 2 * [50] + [2], "tanh", "Glorot normal")
    # net.apply_output_transform(output_transform)

    model = dde.Model(data, net)
    variable = dde.callbacks.VariableValue(
        [keff], period=50, filename="keff.dat"
    )

    model.compile("adam", lr=0.0001, loss_weights=loss_weights_soft, external_trainable_variables=[keff])
    model.train(iterations=50000, display_every=1000, model_save_path="PINN_2D_example_models/PINN_2D_example_model_case_1.ckpt", callbacks=[variable])
    model.compile("L-BFGS", loss_weights=loss_weights_soft, external_trainable_variables=[keff])
    dde.optimizers.config.set_LBFGS_options(maxiter=20000)
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
    condition = (x_grid >= 0) & (x_grid < 0.5) & (y_grid >= 0) & (y_grid < 0.5)
    X = np.column_stack((x_grid.flatten(), y_grid.flatten()))
    y_pred = model.predict(X)
    phi1_core = y_pred[:, 0:1].reshape(101, 101)
    phi1_cladding = y_pred[:, 1:2].reshape(101, 101)
    phi1_pred = np.zeros((101, 101))
    phi1_pred[condition] = phi1_core[condition]
    phi1_pred[~condition] = phi1_cladding[~condition]
    print("keff_pred=", keff)
    return phi1_pred, phi1_core, phi1_cladding, keff

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
    # im = ax.imshow(np.flipud(phi1_true), cmap='turbo', vmin=0.0, vmax=1.0,
    #                 interpolation='nearest', aspect='auto', extent=[0, 1, 0, 1])
    # cb = plt.colorbar(im)  # 添加颜色条
    # cb.ax.tick_params(labelsize=20, length=8)
    # # cb.set_label('$\phi$', fontsize=14, fontweight='bold')  # 在colorbar旁边添加希腊字符phi
    # # plt.xlabel('x', fontsize=14, fontweight='bold')  # 设置x轴标签
    # # plt.ylabel('y', fontsize=14, fontweight='bold')  # 设置y轴标签
    # # plt.title('Case 3 Numerical Solution', fontsize=18, fontweight='bold')  # 设置标题
    # # # 绘制第一条虚线：x=0.5，从y=0到y=0.5的垂直虚线
    # ax.axvline(x=0.5, color='black', linestyle='--', linewidth=2.0, ymin=0.0, ymax=0.5)
    # # # 在x=0.5处添加文本"Interface"
    # # ax.text(0.56, 0.32, 'Interface', rotation=90, color='black', va='top', ha='right', transform=ax.transAxes, fontsize=18)
    # #
    # # # 绘制第二条虚线：y=0.5，从x=0到x=0.5的水平虚线
    # ax.axhline(y=0.5, color='black', linestyle='--', linewidth=2.0, xmin=0.0, xmax=0.5)
    # # # 在y=0.5处添加文本"Interface"
    # # ax.text(0.32, 0.52, 'Interface', color='black', va='center', ha='right', transform=ax.transAxes, fontsize=18)
    #
    # # # 在(0,0)点画一个大黑点
    # ax.plot(0, 0, 'ko', markersize=12, color='black')  # 'ko' 表示黑色圆点
    # # # 在(0,0)点旁边添加文本"Fixed Point: (0,0,1.0)"
    # # ax.text(0.02, 0.02, 'Fixed Point: (0, 0, 1)', color='black', va='bottom', ha='left', transform=ax.transAxes, fontsize=18)
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

    # fig, ax = plt.subplots(figsize=(8, 6))  # 设置图形尺寸
    # im = ax.imshow(np.flipud(phi1_pred_hard), cmap='turbo', vmin=0.0, vmax=1.0,
    #                 interpolation='nearest', aspect='auto', extent=[0, 1, 0, 1])
    # cb = plt.colorbar(im)  # 添加颜色条
    # cb.ax.tick_params(labelsize=20, length=8)
    # # cb.set_label('$\phi$', fontsize=14, fontweight='bold')  # 在colorbar旁边添加希腊字符phi
    # # plt.xlabel('x', fontsize=14, fontweight='bold')  # 设置x轴标签
    # # plt.ylabel('y', fontsize=14, fontweight='bold')  # 设置y轴标签
    # # plt.title('Case 3 FC-PINNs(Hard-constraint) Result', fontsize=18, fontweight='bold')  # 设置标题
    # # 绘制第一条虚线：x=0.5，从y=0到y=0.5的垂直虚线
    # ax.axvline(x=0.5, color='black', linestyle='--', linewidth=2.0, ymin=0.0, ymax=0.5)
    # # 在x=0.5处添加文本"Interface"
    # # ax.text(0.56, 0.32, 'Interface', rotation=90, color='black', va='top', ha='right', transform=ax.transAxes, fontsize=18)
    #
    # # 绘制第二条虚线：y=0.5，从x=0到x=0.5的水平虚线
    # ax.axhline(y=0.5, color='black', linestyle='--', linewidth=2.0, xmin=0.0, xmax=0.5)
    # # 在y=0.5处添加文本"Interface"
    # # ax.text(0.32, 0.52, 'Interface', color='black', va='center', ha='right', transform=ax.transAxes, fontsize=18)
    #
    # # 在(0,0)点画一个大黑点
    # ax.plot(0, 0, 'ko', markersize=12, color='black')  # 'ko' 表示黑色圆点
    # # 在(0,0)点旁边添加文本"Fixed Point: (0,0,1.0)"
    # # ax.text(0.02, 0.02, 'Fixed Point: (0, 0, 1)', color='black', va='bottom', ha='left', transform=ax.transAxes,
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
    im = ax.imshow(np.flipud(phi1_pred_soft), cmap='turbo', vmin=0.0, vmax=1.0,
                    interpolation='nearest', aspect='auto', extent=[0, 1, 0, 1])
    cb = plt.colorbar(im)  # 添加颜色条
    cb.ax.tick_params(labelsize=20, length=8)
    # cb.set_label('$\phi$', fontsize=14, fontweight='bold')  # 在colorbar旁边添加希腊字符phi
    # plt.xlabel('x', fontsize=14, fontweight='bold')  # 设置x轴标签
    # plt.ylabel('y', fontsize=14, fontweight='bold')  # 设置y轴标签
    # plt.title('Case 3 FC-PINNs(Soft-constraint) Result', fontsize=18, fontweight='bold')  # 设置标题
    # 绘制第一条虚线：x=0.5，从y=0到y=0.5的垂直虚线
    ax.axvline(x=0.5, color='black', linestyle='--', linewidth=2.0, ymin=0.0, ymax=0.5)
    # 在x=0.5处添加文本"Interface"
    # ax.text(0.56, 0.32, 'Interface', rotation=90, color='black', va='top', ha='right', transform=ax.transAxes, fontsize=18)

    # 绘制第二条虚线：y=0.5，从x=0到x=0.5的水平虚线
    ax.axhline(y=0.5, color='black', linestyle='--', linewidth=2.0, xmin=0.0, xmax=0.5)
    # 在y=0.5处添加文本"Interface"
    # ax.text(0.32, 0.52, 'Interface', color='black', va='center', ha='right', transform=ax.transAxes, fontsize=18)

    # 在(0,0)点画一个大黑点
    ax.plot(0, 0, 'ko', markersize=12, color='black')  # 'ko' 表示黑色圆点
    # 在(0,0)点旁边添加文本"Fixed Point: (0,0,1.0)"
    # ax.text(0.02, 0.02, 'Fixed Point: (0, 0, 1)', color='black', va='bottom', ha='left', transform=ax.transAxes,
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
    # 创建一个自定义的颜色映射
    colors = ["white", "orange", "red"]  # 定义颜色列表
    cmap_name = "white_to_red"  # 颜色映射的名称
    n_bins = 100  # 颜色的细分数量
    cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

    # fig, ax = plt.subplots(figsize=(8, 6))  # 设置图形尺寸
    # # im = ax.imshow(np.abs(np.flipud(phi1_true - phi1_pred_hard)), cmap='seismic', vmin=-0.01, vmax=0.01,
    # #                interpolation='nearest', aspect='auto', extent=[0, 1, 0, 1])
    # im = ax.imshow(np.abs(np.flipud(phi1_true - phi1_pred_hard) * 100), cmap=cmap, vmin=0.0, vmax=0.6,
    #                interpolation='nearest', aspect='auto', extent=[0, 1, 0, 1])
    # cb = plt.colorbar(im)  # 添加颜色条
    # cb.ax.tick_params(labelsize=20, length=8)
    # # cb.set_label('$\phi$ True - $\phi$ Pred', fontsize=14, fontweight='bold')  # 在colorbar旁边添加希腊字符phi
    # # plt.xlabel('x', fontsize=14, fontweight='bold')  # 设置x轴标签
    # # plt.ylabel('y', fontsize=14, fontweight='bold')  # 设置y轴标签
    # # plt.title('Case 3 AE Distribution(Hard-constraint)', fontsize=18, fontweight='bold')  # 设置标题
    #
    # # 绘制第一条虚线：x=0.5，从y=0到y=0.5的垂直虚线
    # ax.axvline(x=0.5, color='black', linestyle='--', linewidth=2, ymin=0.0, ymax=0.5)
    # # 在x=0.5处添加文本"Interface"
    # # ax.text(0.56, 0.32, 'Interface', rotation=90, color='black', va='top', ha='right', transform=ax.transAxes, fontsize=18)
    #
    # # 绘制第二条虚线：y=0.5，从x=0到x=0.5的水平虚线
    # ax.axhline(y=0.5, color='black', linestyle='--', linewidth=2, xmin=0.0, xmax=0.5)
    # # 在y=0.5处添加文本"Interface"
    # # ax.text(0.32, 0.52, 'Interface', color='black', va='center', ha='right', transform=ax.transAxes, fontsize=18)
    #
    # # # 在(0,0)点画一个大黑点
    # # ax.plot(0, 0, 'ko', markersize=12, color='black')  # 'ko' 表示黑色圆点
    # # # 在(0,0)点旁边添加文本"Fixed Point: (0,0,1.0)"
    # # # ax.text(0.02, 0.02, 'Fixed Point: (0, 0, 1)', color='black', va='bottom', ha='left', transform=ax.transAxes, fontsize=18)
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
    # plt.axis('on')  # 移除坐标轴边框

    fig, ax = plt.subplots(figsize=(8, 6))  # 设置图形尺寸
    im = ax.imshow(np.abs(np.flipud(phi1_true - phi1_pred_soft)) * 100, cmap=cmap, vmin=0, vmax=0.6,
                   interpolation='nearest', aspect='auto', extent=[0, 1, 0, 1])
    cb = plt.colorbar(im)  # 添加颜色条
    cb.ax.tick_params(labelsize=20, length=8)
    # cb.set_label('$\phi$ True - $\phi$ Pred', fontsize=14, fontweight='bold')  # 在colorbar旁边添加希腊字符phi
    # plt.xlabel('x', fontsize=14, fontweight='bold')  # 设置x轴标签
    # plt.ylabel('y', fontsize=14, fontweight='bold')  # 设置y轴标签
    # plt.title('Case 3 AE Distribution(Soft-constraint)', fontsize=18, fontweight='bold')  # 设置标题

    # 绘制第一条虚线：x=0.5，从y=0到y=0.5的垂直虚线
    ax.axvline(x=0.5, color='black', linestyle='--', linewidth=2, ymin=0.0, ymax=0.5)
    # 在x=0.5处添加文本"Interface"
    # ax.text(0.56, 0.32, 'Interface', rotation=90, color='black', va='top', ha='right', transform=ax.transAxes, fontsize=18)

    # 绘制第二条虚线：y=0.5，从x=0到x=0.5的水平虚线
    ax.axhline(y=0.5, color='black', linestyle='--', linewidth=2, xmin=0.0, xmax=0.5)
    # 在y=0.5处添加文本"Interface"
    # ax.text(0.32, 0.52, 'Interface', color='black', va='center', ha='right', transform=ax.transAxes, fontsize=18)

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

    # # 在(0,0)点画一个大黑点
    # ax.plot(0, 0, 'ko', markersize=12, color='black')  # 'ko' 表示黑色圆点
    # # 在(0,0)点旁边添加文本"Fixed Point: (0,0,1.0)"
    # ax.text(0.02, 0.02, 'Fixed Point: (0, 0, 1)', color='black', va='bottom', ha='left', transform=ax.transAxes, fontsize=18)

    plt.xticks(fontsize=20)  # 设置x轴刻度字体大小
    plt.yticks(fontsize=20)  # 设置y轴刻度字体大小
    plt.axis('on')  # 移除坐标轴边框

    plt.show()


def loss_plot():
    case_3_loss_keff = pd.read_excel('case_3_loss_keff.xlsx', engine='openpyxl')

    loss_iteration = case_3_loss_keff.iloc[:, 0].values
    PDE_core_loss = case_3_loss_keff.iloc[:, 1].values
    PDE_cladding_loss = case_3_loss_keff.iloc[:, 2].values
    x_interface_bc_continuity_flux = case_3_loss_keff.iloc[:, 3].values
    x_interface_bc_continuity_neutron = case_3_loss_keff.iloc[:, 4].values
    y_interface_bc_continuity_flux = case_3_loss_keff.iloc[:, 5].values
    y_interface_bc_continuity_neutron = case_3_loss_keff.iloc[:, 6].values
    core_bc_x_left = case_3_loss_keff.iloc[:, 7].values
    cladding_bc_x_left = case_3_loss_keff.iloc[:, 8].values
    cladding_bc_x_right = case_3_loss_keff.iloc[:, 9].values
    cladding_extrapolated_bc_x_right = case_3_loss_keff.iloc[:, 10].values
    core_bc_y_down = case_3_loss_keff.iloc[:, 11].values
    cladding_bc_y_down = case_3_loss_keff.iloc[:, 12].values
    cladding_bc_y_up = case_3_loss_keff.iloc[:, 13].values
    cladding_extrapolated_bc_y_up = case_3_loss_keff.iloc[:, 14].values

    loss_iteration_soft = case_3_loss_keff.iloc[:, 17].values
    PDE_core_loss_soft = case_3_loss_keff.iloc[:, 18].values
    PDE_cladding_loss_soft = case_3_loss_keff.iloc[:, 19].values
    x_interface_bc_continuity_flux_soft = case_3_loss_keff.iloc[:, 20].values
    x_interface_bc_continuity_neutron_soft = case_3_loss_keff.iloc[:, 21].values
    y_interface_bc_continuity_flux_soft = case_3_loss_keff.iloc[:, 22].values
    y_interface_bc_continuity_neutron_soft = case_3_loss_keff.iloc[:, 23].values
    core_bc_x_left_soft = case_3_loss_keff.iloc[:, 24].values
    cladding_bc_x_left_soft = case_3_loss_keff.iloc[:, 25].values
    cladding_bc_x_right_soft = case_3_loss_keff.iloc[:, 26].values
    cladding_extrapolated_bc_x_right_soft = case_3_loss_keff.iloc[:, 27].values
    core_bc_y_down_soft = case_3_loss_keff.iloc[:, 28].values
    cladding_bc_y_down_soft = case_3_loss_keff.iloc[:, 29].values
    cladding_bc_y_up_soft = case_3_loss_keff.iloc[:, 30].values
    cladding_extrapolated_bc_y_up_soft = case_3_loss_keff.iloc[:, 31].values
    fixed_point_loss_soft = case_3_loss_keff.iloc[:, 32].values


    fig, ax = plt.subplots(figsize=(8, 5.5))

    # 使用不同的标记和颜色来区分不同的数据集
    ax.plot(loss_iteration, x_interface_bc_continuity_neutron, label='Hard-constraint Interface', linewidth=1.5, linestyle='-', color='#2486b9')
    # ax.plot(loss_iteration, cladding_extrapolated_bc_x_right, label='Hard-constraint Extrapolated Boundary', linewidth=1.5, linestyle='-', color='#93b5cf')
    ax.plot(loss_iteration_soft, x_interface_bc_continuity_neutron_soft, label='Soft-constraint Interface', linewidth=1.5, linestyle='--', color='#fb9968')
    # ax.plot(loss_iteration_soft, cladding_extrapolated_bc_x_right_soft, label='Soft-constraint Extrapolated Boundary', linewidth=1.5, linestyle='--', color='#ed5126')
    ax.plot(loss_iteration_soft, fixed_point_loss_soft, label='Soft-constraint Fixed Point', linewidth=1.5, linestyle='--', color='#e60012')



    # 设置图例，确保图例不遮挡数据点
    ax.legend(loc='best', fontsize=18, framealpha=0.9)

    # 设置标题和轴标签，确保字体大小和样式一致
    ax.set_title('Case 3 Loss', fontsize=18, fontweight='bold')
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
    ax.set_xlim([0, 71000])

    ax.axvline(x=50000, color='black', linestyle='--', linewidth=1)
    ax.text(50000 - 500, 6e-7, 'adam', color='black', va='center', ha='right', fontsize=18)
    ax.text(50000 + 500, 6e-7, 'L-BFGS', color='black', va='center', ha='left', fontsize=18)

    # 调整布局，确保图表紧凑
    plt.tight_layout()

    # 显示图表
    plt.show()


def keff_plot():
    case_3_loss_keff = pd.read_excel('case_3_loss_keff.xlsx', engine='openpyxl')
    keff_iteration = case_3_loss_keff.iloc[:, 15].values
    keff_value = case_3_loss_keff.iloc[:, 16].values
    keff_iteration_soft = case_3_loss_keff.iloc[:, 33].values
    keff_value_soft = case_3_loss_keff.iloc[:, 34].values


    fig, ax = plt.subplots(figsize=(8, 4))

    # 使用不同的标记和颜色来区分不同的数据集
    ax.plot(keff_iteration[::2] / 10000, keff_value[::2], label='FC-PINNs(Hard-constraint)', linewidth=2, color='red')
    ax.plot(keff_iteration_soft[::2] / 10000, keff_value_soft[::2], label='FC-PINNs(Soft-constraint)', linewidth=2, color='orange')


    # # 设置图例，确保图例不遮挡数据点
    # ax.legend(loc='best', fontsize=18, framealpha=0.9)

    # 设置标题和轴标签，确保字体大小和样式一致
    # ax.set_title('Case 3 Keff', fontsize=18, fontweight='bold')
    # ax.set_xlabel('Iteration', fontsize=14, fontweight='bold')
    # ax.set_ylabel('Keff Value', fontsize=14, fontweight='bold')  # 使用LaTeX语法设置希腊字符φ

    # 设置刻度参数，确保刻度标签清晰
    ax.tick_params(axis='both', which='major', labelsize=20, direction='out')

    # 隐藏上边和右边的坐标轴线
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # # 设置x轴的主要刻度位置
    ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7])
    ax.tick_params(axis='x', length=8)  # 设置次要刻度线长度和隐藏刻度数值

    # # 添加次要刻度位置，刻度线短且不显示刻度数值
    ax.set_xticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5], minor=True)
    ax.tick_params(axis='x', which='minor', length=4, labelsize=0)  # 设置次要刻度线长度和隐藏刻度数值

    # # 设置y轴的主要刻度位置
    ax.set_yticks([0.6, 0.8, 1.0, 1.2])
    ax.tick_params(axis='y', length=8)  # 设置次要刻度线长度和隐藏刻度数值

    # # 添加次要刻度位置，刻度线短且不显示刻度数值
    ax.set_yticks([0.5, 0.7, 0.9, 1.1, 1.3], minor=True)
    ax.tick_params(axis='y', which='minor', length=4, labelsize=0)  # 设置次要刻度线长度和隐藏刻度数值

    # 设置x轴的范围为0到1
    # ax.set_ylim([0.8, 1.1])
    ax.set_xlim([-0.1, 7.1])

    # ax.axvline(x=50000, color='black', linestyle='--', linewidth=1)
    # ax.text(50000 - 500, 0.8, 'adam', color='black', va='center', ha='right', fontsize=18)
    # ax.text(50000 + 500, 0.8, 'L-BFGS', color='black', va='center', ha='left', fontsize=18)

    # 在y=0.9处画一条横线
    ax.axhline(0.49227863989246945, color='grey', linestyle='--', linewidth=1)

    # # 在横线旁边添加带有箭头的文字“keff True”
    # ax.annotate('Keff True', xy=(63500, 0.49227863989246945), xytext=(63500, 0.61227863989246945),
    #             arrowprops=dict(facecolor='black', arrowstyle='->', connectionstyle='arc3'),
    #             ha='center', va='center', fontsize=18)

    # 调整布局，确保图表紧凑
    plt.tight_layout()

    # 显示图表
    plt.show()



def main():
    x_points = 101
    y_points = 101
    x = np.linspace(left, right, x_points)
    y = np.linspace(down, up, y_points)
    # phi1_true, keff_true = set_ND_1group_para_2D_case_1()
    # np.save('ndpinn_2D_example_1_true.npy', phi1_true)
    # print("keff_true =", keff_true)
    # phi1_pred_soft, phi1_core, phi1_cladding, keff_pred = set_pinn_nd()
    # # keff_true = 0.49227863989246945
    # # keff_pred_hard = 0.4922
    # # keff_pred_soft = 0.4913
    # np.save('ndpinn_2D_example_1_pred_soft.npy', phi1_pred_soft)


    # phi1_wrong = np.load('ndpinn_2D_example_1_wrong.npy')
    phi1_pred_hard = np.load('ndpinn_2D_example_1_pred.npy')
    phi1_pred_soft = np.load('ndpinn_2D_example_1_pred_soft.npy')
    phi1_true = np.load('ndpinn_2D_example_1_true.npy').T
    # phi1_wrong = (phi1_wrong * 1.0 / phi1_wrong[0, 0]).reshape(101, 101)
    phi1_true = (phi1_true * 1.0 / phi1_true[0, 0]).reshape(101, 101)

    # print("RMSE_hard=", calculate_RMSE(phi1_true, phi1_pred_hard))
    # print("R_square_hard=", r2_score(phi1_true, phi1_pred_hard))
    # print("RMSE_soft=", calculate_RMSE(phi1_true, phi1_pred_soft))
    # print("R_square_soft=", r2_score(phi1_true, phi1_pred_soft))
    # RMSE_hard = 0.002053378124074889
    # R_square_hard = 0.9992707509614702
    # RMSE_soft = 0.0012559778007690309
    # R_square_soft = 0.9998430660299189
    # print("L2_relative_error1=", calculate_relative_l2_error(phi1_pred, phi1_true))



    result_plot(phi1_pred_hard, phi1_pred_soft, phi1_true)
    # error_plot(phi1_pred_hard, phi1_pred_soft, phi1_true)
    # loss_plot()
    # keff_plot()






if __name__ == "__main__":
    main()