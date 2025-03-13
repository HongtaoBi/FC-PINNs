import torch
import deepxde as dde
import tensorflow as tf
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from sklearn.metrics import r2_score
# from deepxde.nn.pytorch import deeponet
# from test_solver import set_HC_para
# from plotting import calculate_relative_l2_error
from scipy.interpolate import griddata, interp1d
from scipy.optimize import fsolve
from PDE_solver import calculate_RMSE
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
interface = 0.5
extrapolated_ratio = 2.0
keff = dde.Variable(1.0)

def pde_nd(x, y):
    phi1_left = y[:, 0:1]
    phi1_right = y[:, 1:2]

    # dphi1_left_dx = dde.grad.jacobian(phi1_left, x, i=0, j=0)
    dphi1_left_dxx = dde.grad.hessian(phi1_left, x, i=0, j=0)

    # dphi1_right_dx = dde.grad.jacobian(phi1_right, x, i=0, j=0)
    dphi1_right_dxx = dde.grad.hessian(phi1_right, x, i=0, j=0)

    D_left = 1 / torch.pi
    D_right = 1 / (2 * torch.pi)

    sigma_left = - torch.pi / 4 + 10 / 9
    sigma_right = torch.pi / 2 + 10 / 9

    X_left = 1.0
    X_right = 1.0

    nd1_left_pde = - (D_left * (dphi1_left_dxx)) + sigma_left * phi1_left - X_left / keff * phi1_left
    nd1_right_pde = - (D_right * (dphi1_right_dxx)) + sigma_right * phi1_right - X_right / keff * phi1_right

    nd1_left = torch.where(x < interface, nd1_left_pde, 0.0)
    nd1_right = torch.where(x < interface, 0.0, nd1_right_pde)

    return [nd1_left, nd1_right]

    # return torch.where(x < interface, nd1_left_pde, nd1_right_pde)

def boundary_x_left_func(x, y, X):
    phi1_left = y[:, 0:1]

    dphi1_left_dx = dde.grad.jacobian(phi1_left, x, i=0, j=0)

    return dphi1_left_dx - 0.0


def boundary_x_right_func(x, y, X):
    phi1_right = y[:, 1:2]

    D_right = 1 / (2 * torch.pi)

    dphi1_right_dx = dde.grad.jacobian(phi1_right, x, i=0, j=0)

    return dphi1_right_dx + phi1_right / (2 * D_right)


def boundary_x_extrapolated_right_func(x, y, X):
    phi1_right = y[:, 1:2]

    return phi1_right - 0.0


def continuity_flux(x, y, X):
    phi1_left = y[:, 0:1]
    phi1_right = y[:, 1:2]

    dphi1_left_dr = dde.grad.jacobian(phi1_left, x, i=0, j=0)
    dphi1_right_dr = dde.grad.jacobian(phi1_right, x, i=0, j=0)

    D_left = 1 / torch.pi
    D_right = 1 / (2 * torch.pi)

    return D_left * dphi1_left_dr - D_right * dphi1_right_dr

def continuity_neutron(x, y, X):
    phi1_left = y[:, 0:1]
    phi1_right = y[:, 1:2]

    return phi1_left - phi1_right

def fixed_point_func(x, y, X):
    phi1_left = y[:, 0:1]

    return phi1_left - 1.0


def output_transform(x, y):
    phi1_left = y[:, 0:1]
    phi1_right = y[:, 1:2]

    D_right = 1 / (2 * torch.pi)

    phi1_left_transformed = phi1_left * (x - 0.2) + 1.0
    phi1_right_transformed = phi1_right * (right + extrapolated_ratio * D_right - x)
    # phi1_right_transformed = phi1_right
    y_new = torch.cat((phi1_left_transformed, phi1_right_transformed), dim=1)
    return y_new


def set_pinn_nd():
    geom = dde.geometry.Interval(left, right)
    D_right = 1 / (2 * np.pi)
    phi1_left_BC_points = np.full((1, 1), left)  # [1, 1]
    # print("phi1_left_BC_points=", phi1_left_BC_points, phi1_left_BC_points.shape)
    phi1_right_BC_points = np.full((1, 1), right)  # [1, 1]
    phi1_right_extrapolated_BC_points = np.full((1, 1), right + extrapolated_ratio * D_right)  # [1, 1]
    x_interface_points = np.full((1, 1), interface)  # [1, 1]
    fixed_point = np.full((1, 1), 0.2)  # [1, 1]

    bc_continuity_flux = dde.icbc.PointSetOperatorBC(x_interface_points, 0.0, continuity_flux)
    bc_continuity_neutron = dde.icbc.PointSetOperatorBC(x_interface_points, 0.0, continuity_neutron)

    bc_x_left = dde.icbc.PointSetOperatorBC(phi1_left_BC_points, 0.0, boundary_x_left_func)
    bc_x_right = dde.icbc.PointSetOperatorBC(phi1_right_BC_points, 0.0, boundary_x_right_func)
    bc_x_extrapolated_right = dde.icbc.PointSetOperatorBC(phi1_right_extrapolated_BC_points, 0.0, boundary_x_extrapolated_right_func)
    fixed_point_constraint = dde.icbc.PointSetOperatorBC(fixed_point, 0.0, fixed_point_func)

    BC = [bc_continuity_neutron, bc_continuity_flux, bc_x_left, bc_x_right, fixed_point_constraint]
    # loss_weights = [100000, 1, 100, 100, 100, 10000]
    # loss_weights_hard = [100, 0.1, 10, 10, 10, 100]
    loss_weights_soft = [100, 0.1, 10, 10, 10, 100, 100]

    data = dde.data.PDE(
        geom,
        pde_nd,
        BC,
        num_domain=400,
        num_test=400,
    )

    net = dde.nn.pytorch.FNN([1] + 2 * [50] + [2], "tanh", "Glorot normal")
    # net.apply_output_transform(output_transform)

    model = dde.Model(data, net)
    variable = dde.callbacks.VariableValue(
        [keff], period=50, filename="keff.dat"
    )

    model.compile("adam", lr=0.0001, loss_weights=loss_weights_soft, external_trainable_variables=[keff])
    loss_history, train_state = model.train(iterations=50000, display_every=1000, model_save_path="PINN_1D_example_models/PINN_1D_example_model.ckpt", callbacks=[variable])
    # model.compile("L-BFGS", loss_weights=loss_weights, external_trainable_variables=[keff])
    # dde.optimizers.config.set_LBFGS_options(maxiter=50000)
    # loss_history, train_state = model.train(model_save_path="PINN_HC_model/HC_model.ckpt", callbacks=[variable])
    # model.compile("adam", lr=0.0001, loss_weights=loss_weights)
    # model.train(iterations=5000, display_every=1000, model_save_path="PINN_HC_model/HC_model.ckpt")
    dde.saveplot(loss_history, train_state, issave=True, isplot=True)

    # model.compile("adam", lr=0.0001, loss_weights=loss_weights, external_trainable_variables=[keff])
    # model.restore("./PINN_HC_model/HC_model.ckpt-505000.pt")
    # model.train(iterations=2000, display_every=1000, callbacks=[variable])
    # model.compile("L-BFGS", loss_weights=loss_weights, external_trainable_variables=[keff])
    # dde.optimizers.config.set_LBFGS_options(maxiter=8000)
    # loss_history, train_state = model.train(callbacks=[variable])
    # dde.saveplot(loss_history, train_state, issave=False, isplot=False)


    # model.compile("adam", lr=0.0001, loss_weights=loss_weights, external_trainable_variables=[keff])
    # model.restore("./PINN_HC_model/HC_model.ckpt-25000.pt")
    # loss_history, train_state = model.train(iterations=7000, display_every=1000, callbacks=[variable])
    # # model.compile("L-BFGS", loss_weights=loss_weights, external_trainable_variables=[keff])
    # # dde.optimizers.config.set_LBFGS_options(maxiter=28000)
    # # loss_history, train_state = model.train(callbacks=[variable])
    # dde.saveplot(loss_history, train_state, issave=False, isplot=False)


    X = np.linspace(left, right, num=101).reshape(-1, 1)    # [100, 1]
    y_pred = model.predict(X)
    phi1_core = y_pred[:, 0:1]        # [100, 1]
    phi1_cladding = y_pred[:, 1:2]        # [100, 1]
    phi1_pred = np.concatenate((phi1_core[:50], phi1_cladding[50:]), axis=0)
    print("keff_pred=", keff)
    return phi1_pred, phi1_core, phi1_cladding, keff

def result_plot(x, phi1_wrong, phi1_pred_hard, phi1_pred_soft, phi1_true):
    fig, ax = plt.subplots(figsize=(8, 4))

    # 使用不同的标记和颜色来区分不同的数据集
    ax.plot(x, phi1_true, label='True', linewidth=5, color='blue', linestyle='-')  # 增加蓝色线的线宽
    ax.plot(x[::5], phi1_pred_hard[::5], label='FC-PINNs (Hard-constraint)', linewidth=3, color='red', linestyle='--', marker='o', markersize=8)
    ax.plot(x[::5], phi1_pred_soft[::5], label='FC-PINNs (Soft-constraint)', linewidth=3, color='orange', linestyle='-.', marker='^', markersize=8)
    ax.plot(x, phi1_wrong, label='PINNs', linewidth=3, color='green', linestyle=':')
    # ax.scatter(x[::10], phi1_pred[::10], label='PINN + FixedPoint', color='red', s=20)  # 将红色线改为散点图，间隔为10

    # 设置标题和轴标签，确保字体大小和样式一致
    # ax.set_title('Case 1 Results', fontsize=18, fontweight='bold')
    # ax.set_xlabel('x', fontsize=14, fontweight='bold')
    # ax.set_ylabel('$\phi$', fontsize=14, fontweight='bold')  # 使用LaTeX语法设置希腊字符φ

    # 设置刻度参数，确保刻度标签清晰，将direction改为out
    ax.tick_params(axis='both', which='major', labelsize=20, direction='out')

    # 隐藏上边和右边的坐标轴线
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # 设置x轴的主要刻度位置
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis='x', length=8)  # 设置次要刻度线长度和隐藏刻度数值

    # 添加次要刻度位置，刻度线短且不显示刻度数值
    ax.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9], minor=True)
    ax.tick_params(axis='x', which='minor', length=4, labelsize=0)  # 设置次要刻度线长度和隐藏刻度数值

    # 设置y轴的主要刻度位置
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis='y', length=8)  # 设置次要刻度线长度和隐藏刻度数值

    # 添加次要刻度位置，刻度线短且不显示刻度数值
    ax.set_yticks([0.1, 0.3, 0.5, 0.7, 0.9], minor=True)
    ax.tick_params(axis='y', which='minor', length=4, labelsize=0)  # 设置次要刻度线长度和隐藏刻度数值

    # # 在[0.2, 1]处画一个黑色大圆点，并在其旁边标注文字:Fixed Point: [0.2, 1]
    ax.plot(0.2, 1, 'ko', markersize=12, label='Fixed point')  # 画黑色大圆点
    # ax.annotate('Fixed Point: (0.2, 1)', xy=(0.20, 0.99), xytext=(0.185, 0.8),
    #             arrowprops=dict(facecolor='black', arrowstyle='->', connectionstyle='arc3'),
    #             ha='center', va='center', fontsize=18)  # 标注文字并添加箭头
    #
    # # 为x=0.5画一条竖着的线，并在线旁边加个单词：Interface
    ax.axvline(x=0.5, color='black', linestyle='--', linewidth=2.0, label='Interface')  # 画竖直线
    # ax.annotate('Interface', xy=(0.5, 0.925), xytext=(0.65, 0.925),
    #             arrowprops=dict(facecolor='black', arrowstyle='->', connectionstyle='arc3'),
    #             ha='center', va='center', fontsize=18)  # 标注文字并添加箭头

    # 设置x轴的范围为0到1
    ax.set_xlim([-0.02, 1.02])

    # # 设置图例，确保图例不遮挡数据点
    ax.legend(loc=(0.02, 0.1), fontsize=18, framealpha=0.9)

    # 调整布局，确保图表紧凑
    plt.tight_layout()

    # 显示图表
    plt.show()

def error_plot(x, phi1_pred_hard, phi1_pred_soft, phi1_true):
    fig, ax = plt.subplots(figsize=(8, 4))

    # 使用不同的标记和颜色来区分不同的数据集
    ax.plot(x[::5], np.abs((phi1_true - phi1_pred_hard)[::5] * 100), label='FC-PINNs(Hard-constraint)', linewidth=2,
            color='red', linestyle='--', marker='o', markersize=8)
    ax.plot(x[::5], np.abs((phi1_true - phi1_pred_soft)[::5] * 100), label='FC-PINNs(Soft-constraint)', linewidth=2,
            color='orange', linestyle='-.', marker='^', markersize=6)

    # 设置图例，确保图例不遮挡数据点
    # ax.legend(loc='best', fontsize=18, framealpha=0.9)

    # 设置标题和轴标签，确保字体大小和样式一致
    # ax.set_title('Case 1 Absolute Error Distribution', fontsize=18, fontweight='bold')
    # ax.set_xlabel('x', fontsize=14, fontweight='bold')
    # ax.set_ylabel('$\phi$ True - $\phi$ Pred', fontsize=14, fontweight='bold')  # 使用LaTeX语法设置希腊字符φ

    # 设置刻度参数，确保刻度标签清晰，将direction改为out
    ax.tick_params(axis='both', which='major', labelsize=20, direction='out')

    # 隐藏上边和右边的坐标轴线
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # 设置x轴的主要刻度位置
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis='x', length=8)  # 设置次要刻度线长度和隐藏刻度数值

    # 添加次要刻度位置，刻度线短且不显示刻度数值
    ax.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9], minor=True)
    ax.tick_params(axis='x', which='minor', length=4, labelsize=0)  # 设置次要刻度线长度和隐藏刻度数值

    # 设置y轴的主要刻度位置
    ax.set_yticks([0, 0.2, 0.4, 0.6])
    ax.tick_params(axis='y', length=8)  # 设置次要刻度线长度和隐藏刻度数值

    # 添加次要刻度位置，刻度线短且不显示刻度数值
    ax.set_yticks([0.1, 0.3, 0.5], minor=True)
    ax.tick_params(axis='y', which='minor', length=4, labelsize=0)  # 设置次要刻度线长度和隐藏刻度数值

    # 为x=0.5画一条竖着的线，并在线旁边加个单词：Interface
    ax.axvline(x=0.5, color='black', linestyle='--', linewidth=2.0)  # 画竖直线
    # ax.annotate('Interface', xy=(0.5, -0.003), xytext=(0.35, -0.003),
    #             arrowprops=dict(facecolor='black', arrowstyle='->', connectionstyle='arc3'),
    #             ha='center', va='center', fontsize=18)  # 标注文字并添加箭头

    # 设置x轴的范围为0到1
    ax.set_xlim([-0.02, 1.02])

    # 调整布局，确保图表紧凑
    plt.tight_layout()

    # 显示图表
    plt.show()


def loss_plot():
    case_1_loss_keff = pd.read_excel('case_1_loss_keff.xlsx', engine='openpyxl')
    loss_iteration = case_1_loss_keff.iloc[:, 0].values
    PDE_left_loss = case_1_loss_keff.iloc[:, 1].values
    PDE_right_loss = case_1_loss_keff.iloc[:, 2].values
    bc_continuity_neutron_loss = case_1_loss_keff.iloc[:, 3].values
    bc_continuity_flux_loss = case_1_loss_keff.iloc[:, 4].values
    bc_x_left_loss = case_1_loss_keff.iloc[:, 5].values
    bc_x_right_loss = case_1_loss_keff.iloc[:, 6].values

    loss_iteration_soft = case_1_loss_keff.iloc[:, 10].values
    PDE_left_loss_soft = case_1_loss_keff.iloc[:, 11].values
    PDE_right_loss_soft = case_1_loss_keff.iloc[:, 12].values
    bc_continuity_neutron_loss_soft = case_1_loss_keff.iloc[:, 13].values
    bc_continuity_flux_loss_soft = case_1_loss_keff.iloc[:, 14].values
    bc_x_left_loss_soft = case_1_loss_keff.iloc[:, 15].values
    bc_x_right_loss_soft = case_1_loss_keff.iloc[:, 16].values
    fixed_point_loss_soft = case_1_loss_keff.iloc[:, 17].values

    fig, ax = plt.subplots(figsize=(8, 5.5))

    # 使用不同的标记和颜色来区分不同的数据集
    ax.plot(loss_iteration, bc_continuity_neutron_loss, label='Hard-constraint Interface', linewidth=1.5, linestyle='-', color='#2486b9')
    # ax.plot(loss_iteration, bc_continuity_flux_loss, label='Hard-constraint Interface Continuity Flux', linewidth=1.5, linestyle='-', color='#93b5cf#2486b9')
    ax.plot(loss_iteration_soft, bc_continuity_neutron_loss_soft, label='Soft-constraint Interface', linewidth=1.5, linestyle='--', color='#fb9968')
    # ax.plot(loss_iteration_soft, bc_continuity_flux_loss_soft, label='Soft-constraint Interface Continuity Flux', linewidth=1.5, linestyle='--', color='#ed5126')
    ax.plot(loss_iteration_soft, fixed_point_loss_soft, label='Soft-constraint Fixed Point', linewidth=1.5, linestyle='--', color='#e60012')


    # 设置图例，确保图例不遮挡数据点
    ax.legend(loc='best', fontsize=18, framealpha=0.9)

    # 设置标题和轴标签，确保字体大小和样式一致
    ax.set_title('Case 1 Loss', fontsize=18, fontweight='bold')
    ax.set_xlabel('Iteration', fontsize=14, fontweight='bold')
    ax.set_ylabel('Loss Value', fontsize=14, fontweight='bold')  # 使用LaTeX语法设置希腊字符φ

    # 设置刻度参数，确保刻度标签清晰
    ax.tick_params(axis='both', which='major', labelsize=15, direction='in', right=True, top=True)

    # 设置x轴的范围为0到1
    ax.set_yscale('log')
    # 自定义y轴刻度
    # majorLocator = ticker.LogLocator(base=10.0, numticks=14)  # 设置主要刻度数量
    # minorLocator = ticker.LogLocator(base=10.0, subs=np.arange(0.1, 1.0, 0.1), numticks=10)  # 设置次要刻度
    #
    # ax.yaxis.set_major_locator(majorLocator)
    # ax.yaxis.set_minor_locator(minorLocator)
    # ax.set_ylim([0, 0.01])
    ax.set_xlim([0, 51000])

    # 调整布局，确保图表紧凑
    plt.tight_layout()
    # plt.grid('on')

    # 显示图表
    plt.show()


def keff_plot():
    case_1_loss_keff = pd.read_excel('case_1_loss_keff.xlsx', engine='openpyxl')
    keff_iteration = case_1_loss_keff.iloc[:, 7].values
    keff_value = case_1_loss_keff.iloc[:, 8].values
    keff_iteration_soft = case_1_loss_keff.iloc[:, 18].values
    keff_value_soft = case_1_loss_keff.iloc[:, 19].values


    fig, ax = plt.subplots(figsize=(8, 4))

    # 使用不同的标记和颜色来区分不同的数据集
    ax.plot(keff_iteration[::2] / 10000, keff_value[::2], label='FC-PINNs(Hard-constraint)', linewidth=2, color='red')
    ax.plot(keff_iteration_soft[::2] / 10000, keff_value_soft[::2], label='FC-PINNs(Soft-constraint)', linewidth=2, color='orange')


    # 设置图例，确保图例不遮挡数据点
    # ax.legend(loc='best', fontsize=18, framealpha=0.9)

    # 设置标题和轴标签，确保字体大小和样式一致
    # ax.set_title('Case 1 Keff', fontsize=18, fontweight='bold')
    # ax.set_xlabel('Iteration', fontsize=14, fontweight='bold')
    # ax.set_ylabel('Keff Value', fontsize=14, fontweight='bold')  # 使用LaTeX语法设置希腊字符φ

    # 设置刻度参数，确保刻度标签清晰，将direction改为out
    ax.tick_params(axis='both', which='major', labelsize=20, direction='out')

    # 隐藏上边和右边的坐标轴线
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # 设置x轴的主要刻度位置
    ax.set_xticks([0, 1, 2, 3, 4, 5])
    ax.tick_params(axis='x', length=8)  # 设置次要刻度线长度和隐藏刻度数值

    # 添加次要刻度位置，刻度线短且不显示刻度数值
    ax.set_xticks([0.5, 1.5, 2.5, 3.5, 4.5], minor=True)
    ax.tick_params(axis='x', which='minor', length=4, labelsize=0)  # 设置次要刻度线长度和隐藏刻度数值

    # 设置y轴的主要刻度位置
    ax.set_yticks([1.0, 1.2, 1.4])
    ax.tick_params(axis='y', length=8)  # 设置次要刻度线长度和隐藏刻度数值

    # 添加次要刻度位置，刻度线短且不显示刻度数值
    ax.set_yticks([0.9, 1.1, 1.3], minor=True)
    ax.tick_params(axis='y', which='minor', length=4, labelsize=0)  # 设置次要刻度线长度和隐藏刻度数值

    # 设置x轴的范围为0到1
    # ax.set_ylim([0.8, 1.1])
    ax.set_xlim([-0.1, 5.1])

    # 在y=0.9处画一条横线
    ax.axhline(0.9, color='grey', linestyle='--', linewidth=1)

    # # 在横线旁边添加带有箭头的文字“keff True”
    # ax.annotate('Keff True', xy=(40000, 0.9), xytext=(40000, 0.98),
    #             arrowprops=dict(facecolor='black', arrowstyle='->', connectionstyle='arc3'),
    #             ha='center', va='center', fontsize=18)

    # 调整布局，确保图表紧凑
    plt.tight_layout()

    # 显示图表
    plt.show()



def main():
    x_points = 101
    x = np.linspace(left, right, x_points)
    phi1_left_true = 2 ** 0.5 * np.exp(- np.pi / 2) * np.cos(np.pi / 2 * x)
    phi1_right_true = np.exp(- np.pi * x)
    phi1_true = np.concatenate((phi1_left_true[:50], phi1_right_true[50:]), axis=0)
    # phi1_pred_soft, phi1_core, phi1_cladding, keff_pred = set_pinn_nd()
    # # keff_true = 0.9
    # # keff_pred_hard = 0.9006
    # # keff_pred_soft = 0.9003
    # # np.save('ndpinn_1D_example_1_pred.npy', phi1_pred)
    # np.save('ndpinn_1D_example_1_pred_soft.npy', phi1_pred_soft)
    # # np.save('ndpinn_1D_example_1_true.npy', phi1_true)


    phi1_wrong = np.load('ndpinn_1D_example_1_pred_all_zero.npy')
    phi1_pred_hard = np.load('ndpinn_1D_example_1_pred.npy')
    phi1_pred_soft = np.load('ndpinn_1D_example_1_pred_soft.npy')
    phi1_true = np.load('ndpinn_1D_example_1_true.npy')
    phi1_true = (phi1_true / phi1_true[20]).reshape(101, 1)

    # print("RMSE hard=", calculate_RMSE(phi1_true, phi1_pred_hard))
    # print("R square hard=", r2_score(phi1_true, phi1_pred_hard))
    # print("RMSE soft=", calculate_RMSE(phi1_true, phi1_pred_soft))
    # print("R square soft=", r2_score(phi1_true, phi1_pred_soft))
    # RMSE_hard = 0.0021593963101537317
    # R_square_hard = 0.999953797712835
    # RMSE_soft = 0.0010270740511082747
    # R_square_soft = 0.999989547930734
    # print("L2_relative_error1=", calculate_relative_l2_error(phi1_pred, phi1_true))

    result_plot(x, phi1_wrong, phi1_pred_hard, phi1_pred_soft, phi1_true)
    # error_plot(x, phi1_pred_hard, phi1_pred_soft, phi1_true)
    # loss_plot()
    # keff_plot()



if __name__ == "__main__":
    main()