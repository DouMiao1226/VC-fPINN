import os
import time

import numpy as np
import torch
from matplotlib import pyplot as plt, gridspec
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch import nn
from scipy.special import gamma
from torch.autograd import Variable
from torchaudio.functional import gain
from tqdm import trange
from GaussJacobiQuadRule_V3 import Jacobi, DJacobi, GaussLobattoJacobiWeights, GaussJacobiWeights

PDE_name = '1D_FDE_fPINN'

def set_seed(seed):
    torch.set_default_dtype(torch.float)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def is_cuda(data):
    if use_gpu:
        data = data.cuda()
    return data

def C(n):
    return (n + 1) ** (1 - alpha) - n ** (1 - alpha)

def exact_u(X):
    return X[:, [1]] ** 2 + (2 * k * X[:, [0]] ** alpha) / gamma(alpha + 1)

def data_quad(N_Quad):
    NQ_u = N_Quad  # 定义积分点的数量
    [x_quad, w_quad] = GaussLobattoJacobiWeights(NQ_u, 0, 0)  # (N_Quad,)
    # x_quad = lb[0] + (ub[0] - lb[0]) / 2 * (x_quad + 1) # 将积分区间由[-1,1]变为[0,1]
    jacobian = (ub[1] - lb[1]) / 2
    w_quad = w_quad[:, None]  # (N_Quad,1)

    return x_quad, w_quad, jacobian

def data_train(x_quad):
    t = np.linspace(lb[0], ub[0], t_N)[:, None]
    x_data = x_quad[:, None]  # (N_Quad,1)
    x_data = torch.from_numpy(x_data).float()
    return t, x_data

# def data_train():
#     t = np.linspace(lb[0], ub[0], t_N)[:, None]
#     x_data = np.linspace(lb[1], ub[1], x_N)[:, None]
#     x_data = torch.from_numpy(x_data).float()
#
#     return t, x_data

def data_test():
    t_test = np.linspace(lb[0], ub[0], t_test_N)[:, None]
    x_test = np.linspace(lb[1], ub[1], x_test_N)[:, None]
    t_star, x_star = np.meshgrid(t_test, x_test)
    t_star = t_star.flatten()[:, None]
    x_star = x_star.flatten()[:, None]
    tx_star = np.hstack((t_star, x_star))

    tx_test = is_cuda(torch.from_numpy(tx_star).float())
    tx_test_exact = exact_u(tx_test)

    return t_test, x_test, tx_test, tx_test_exact

class Net_PMNN(nn.Module):
    def __init__(self, layers):
        super(Net_PMNN, self).__init__()
        self.layers = layers
        self.iter = 0
        self.activation = nn.Tanh()
        self.loss_function = nn.MSELoss(reduction='mean')
        self.linear = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])
        self.attention1 = nn.Linear(layers[0], layers[1])
        self.attention2 = nn.Linear(layers[0], layers[1])
        for i in range(len(layers) - 1):
            nn.init.xavier_normal_(self.linear[i].weight.data, gain=1.0)
            nn.init.zeros_(self.linear[i].bias.data)
        nn.init.xavier_normal_(self.attention1.weight.data, gain=1.0)
        nn.init.zeros_(self.attention1.bias.data)
        nn.init.xavier_normal_(self.attention2.weight.data, gain=1.0)
        nn.init.zeros_(self.attention2.bias.data)

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x)
        a = self.activation(self.linear[0](x))
        encoder_1 = self.activation(self.attention1(x))
        encoder_2 = self.activation(self.attention2(x))
        a = a * encoder_1 + (1 - a) * encoder_2
        for i in range(1, len(self.layers) - 2):
            z = self.linear[i](a)
            a = self.activation(z)
            a = a * encoder_1 + (1 - a) * encoder_2
        a = self.linear[-1](a)
        return a


class Model:
    def __init__(self, net, x_data, t, lb, ub,
                 tx_test, tx_test_exact
                 ):

        self.tx = None
        self.tx_t0 = None
        self.tx_b1 = None
        self.tx_b2 = None
        self.u_t0 = None
        self.u_x_b1 = None
        self.u_x_b2 = None

        self.optimizer_u = None
        self.optimizer_LBGFS = None

        self.lambda_bc = 1.0
        self.lambda_ic = 1.0

        self.net = net

        self.x_data = x_data
        self.x_N = len(x_data)

        self.t = t
        self.t_N = len(t)
        self.dt = ((ub[0] - lb[0]) / (self.t_N - 1))
        self.lb = lb
        self.ub = ub

        self.tx_test = tx_test
        self.tx_test_exact = tx_test_exact

        self.i_loss_collect = []
        self.b_loss_collect = []
        self.f_loss_collect = []
        self.total_loss_collect = []
        self.error_collect = []
        self.pred_u_collect = []

        self.logger_lambada_b = []
        self.logger_lambada_i = []

        self.tx_test_estimate_collect = []

        self.init_data()

    def init_data(self):
        temp_t = torch.full_like(torch.zeros(self.x_N, 1), self.t[0][0])
        self.tx_t0 = is_cuda(torch.cat((temp_t, self.x_data), dim=1))
        self.tx = torch.cat((temp_t, self.x_data), dim=1)
        for i in range(self.t_N - 1):
            temp_t = torch.full_like(torch.zeros(self.x_N, 1), self.t[i + 1][0])
            temp_tx = torch.cat((temp_t, self.x_data), dim=1)
            self.tx = torch.cat((self.tx, temp_tx), dim=0)

        self.tx = is_cuda(self.tx)

        temp_t = torch.from_numpy(self.t).float()
        temp_lb = torch.full_like(torch.zeros(self.t_N, 1), self.lb[1])
        temp_ub = torch.full_like(torch.zeros(self.t_N, 1), self.ub[1])
        self.tx_b1 = is_cuda(torch.cat((temp_t, temp_lb), dim=1))
        self.tx_b2 = is_cuda(torch.cat((temp_t, temp_ub), dim=1))
        self.u_x_b1 = exact_u(self.tx_b1)
        self.u_x_b2 = exact_u(self.tx_b2)
        self.u_t0 = exact_u(self.tx_t0)

        self.lb = is_cuda(torch.from_numpy(lb).float())
        self.ub = is_cuda(torch.from_numpy(ub).float())

    def train_U(self, x):
        H = 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0
        return self.net(H)

    def predict_U(self, x):
        return self.train_U(x)

    def Lu_and_F(self):
        x = Variable(self.tx, requires_grad=True)
        u_n = self.train_U(x)
        d = torch.autograd.grad(u_n, x, grad_outputs=torch.ones_like(u_n),
                                      create_graph=True)
        u_x = d[0][:, [1]]
        dd = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                                         create_graph=True)
        u_xx = dd[0][:, [1]]

        u_n = u_n.reshape(self.t_N, -1)
        Lu = u_xx.reshape(self.t_N, -1)

        return u_n, Lu

    def PDE_loss(self):
        u_n, Lu = self.Lu_and_F()

        loss = is_cuda(torch.tensor(0.))

        for n in range(1, self.t_N):
            if n == 1:
                pre_Ui = cof * Lu[n] + (C(n - 1) / C(0)) * u_n[0]
            else:
                pre_Ui = is_cuda(cof * Lu[n] + (C(n - 1) / C(0)) * u_n[0])
                for k in range(1, n):
                    pre_Ui += ((C(n - k - 1) - C(n - k)) / C(0)) * u_n[k]
            loss += torch.mean((pre_Ui - u_n[n]) ** 2)

        return loss

    def calculate_loss(self):
        loss_i = torch.mean((self.train_U(self.tx_t0) - self.u_t0) ** 2)
        self.i_loss_collect.append([self.net.iter, loss_i.item()])
        loss_b1 = torch.mean((self.train_U(self.tx_b1) - self.u_x_b1) ** 2)
        loss_b2 = torch.mean((self.train_U(self.tx_b2) - self.u_x_b2) ** 2)
        loss_b = loss_b1 + loss_b2
        self.b_loss_collect.append([self.net.iter, loss_b.item()])

        loss_f = self.PDE_loss()
        self.f_loss_collect.append([self.net.iter, loss_f.item()])

        return 10 * loss_i, 10 * loss_b, loss_f

    # computer backward loss
    def LBGFS_loss(self):
        self.optimizer_LBGFS.zero_grad()
        loss_i, loss_b, loss_f = self.calculate_loss()
        loss = loss_i + loss_b + loss_f
        self.total_loss_collect.append([self.net.iter, loss.item()])
        loss.backward()
        self.net.iter += 1
        print('Iter:', self.net.iter, 'Loss:', loss.item())
        pred = self.train_U(tx_test).cpu().detach().numpy()
        exact = self.tx_test_exact.cpu().detach().numpy()
        error = np.linalg.norm(pred - exact, 2) / np.linalg.norm(exact, 2)
        error = torch.tensor(error)
        self.error_collect.append([self.net.iter, error.item()])


        if self.net.iter % 10 == 0:
            pred_u = self.train_U(tx_test).cpu().detach().numpy()
            pred_u = torch.from_numpy(pred_u).cpu()
            self.pred_u_collect.append([pred_u.tolist()])
            # self.pred_u_collect.append([self.net.iter, pred_u.tolist()])
            # self.pred_u_collect.append(pred_u.tolist())

        return loss

    def train(self, LBGFS_epochs=50000):

        self.optimizer_LBGFS = torch.optim.LBFGS(
            self.net.parameters(),
            lr=1,
            max_iter=LBGFS_epochs,
            max_eval=LBGFS_epochs,
            history_size=100,
            tolerance_grad=1e-12,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe"
        )

        start_time = time.time()

        self.optimizer_LBGFS.step(self.LBGFS_loss)
        print('LBGFS done!')
        pred = self.train_U(tx_test).cpu().detach().numpy()
        exact = self.tx_test_exact.cpu().detach().numpy()
        error = np.linalg.norm(pred - exact, 2) / np.linalg.norm(exact, 2)
        print('LBGFS==Test_L2error:', '{0:.2e}'.format(error))

        elapsed = time.time() - start_time
        print('LBGFS==Training time: %.2f' % elapsed)

        save_error(self.error_collect)
        save_loss_fPINN(self.i_loss_collect, self.b_loss_collect, self.f_loss_collect, self.total_loss_collect)

        return error, elapsed, self.LBGFS_loss().item()

def save_error(error_collect):
    np.savetxt(f'Error/Error_{PDE_name}.txt', error_collect)

def save_loss_fPINN(i_loss_collect, b_loss_collect, v_loss_collect, total_loss):
    np.savetxt(f'Loss/Loss_i_{PDE_name}.txt', i_loss_collect)
    np.savetxt(f'Loss/Loss_b_{PDE_name}.txt', b_loss_collect)
    np.savetxt(f'Loss/Loss_v_{PDE_name}.txt', v_loss_collect)
    np.savetxt(f'Loss/Loss_total_{PDE_name}.txt', total_loss)

def show_accuracy():
    u_exact_np = tx_test_exact.cpu().detach().numpy()
    u_test_np = model.predict_U(tx_test).cpu().detach().numpy()
    TT, XX = np.meshgrid(t_test, x_test)
    ex = np.reshape(u_exact_np, (TT.shape[0], TT.shape[1]))
    pre = np.reshape(u_test_np, (TT.shape[0], TT.shape[1]))
    err = np.reshape(abs(u_test_np - u_exact_np), (TT.shape[0], TT.shape[1]))

    # 子图1：Exact u(t,x)
    fig1 = plt.figure(figsize=(6, 4))
    ax1 = fig1.add_subplot(111, projection='3d')
    surf1 = ax1.plot_surface(TT, XX, ex, rstride=1, cstride=1, cmap='jet')
    cbar1 = plt.colorbar(surf1, ax=ax1, fraction=0.02, pad=0.1, shrink=0.8)
    cbar1.ax.tick_params(labelsize=6)
    ax1.set_xlabel('$t$', labelpad=5, fontsize=10)
    ax1.set_ylabel('$x$', labelpad=5, fontsize=10)
    ax1.set_title('Exact $u(t,x)$', fontsize=10)
    fig1.savefig(f'Figure/{PDE_name}/{PDE_name}_exact.pdf', dpi=300)
    plt.close(fig1)  # 关闭当前画布

    # 子图2：Pred u(t,x)
    fig2 = plt.figure(figsize=(6, 4))
    ax2 = fig2.add_subplot(111, projection='3d')
    surf2 = ax2.plot_surface(TT, XX, pre, rstride=1, cstride=1, cmap='jet')
    cbar2 = plt.colorbar(surf2, ax=ax2, fraction=0.02, pad=0.1, shrink=0.8)
    cbar2.ax.tick_params(labelsize=6)
    ax2.set_xlabel('$t$', labelpad=5, fontsize=10)
    ax2.set_ylabel('$x$', labelpad=5, fontsize=10)
    ax2.set_title('Pred $u(t,x)$', fontsize=10)
    fig2.savefig(f'Figure/{PDE_name}/{PDE_name}_pred.pdf', dpi=300)
    plt.close(fig2)  # 关闭当前画布

    # 子图3：Error
    fig3 = plt.figure(figsize=(6, 4))
    ax3 = fig3.add_subplot(111)
    heatmap = ax3.pcolor(TT, XX, err, cmap='jet', shading='auto')
    cbar3 = plt.colorbar(heatmap, ax=ax3, orientation='vertical', pad=0.02, fraction=0.05, shrink=0.8)
    cbar3.ax.tick_params(labelsize=6)
    cbar3.set_label('Error Magnitude', fontsize=10)
    ax3.set_xlabel('$t$', labelpad=5, fontsize=10)
    ax3.set_ylabel('$x$', labelpad=5, fontsize=10)
    ax3.set_title('Absolute Error', fontsize=10)
    fig3.savefig(f'Figure/{PDE_name}/{PDE_name}_error_2D.pdf', dpi=300)
    plt.close(fig3)  # 关闭当前画布

    # 子图4-6：Slices at different times
    for idx, t_idx in enumerate([0, int(t_test_N / 2), -1]):
        fig_slice = plt.figure(figsize=(6, 4))
        ax_slice = fig_slice.add_subplot(111)
        ax_slice.plot(x_test, u_test_np.reshape((t_test_N, x_test_N))[:, t_idx], 'b-', linewidth=2, label='Prediction')
        ax_slice.plot(x_test, u_exact_np.reshape((t_test_N, x_test_N))[:, t_idx], 'r--', linewidth=2, label='Exact')
        ax_slice.set_xlabel('$x$', labelpad=5, fontsize=10)
        ax_slice.set_ylabel('$u(x,t)$', labelpad=5, fontsize=10)
        ax_slice.set_title(f'$t = {float(t_test[t_idx].item()):.1f}$', fontsize=10)
        ax_slice.legend(fontsize=8)
        fig_slice.savefig(f'Figure/{PDE_name}/{PDE_name}_t_{float(t_test[t_idx].item()):.1f}.pdf', dpi=300)
        plt.close(fig_slice)  # 关闭当前画布

    # 总图
    fig_all = plt.figure(figsize=(12.8, 6.8), constrained_layout=True)
    gs1 = gridspec.GridSpec(2, 3, figure=fig_all)

    # 将所有子图整合到一张画布
    ax1 = fig_all.add_subplot(gs1[0, 0], projection='3d')
    fig1 = ax1.plot_surface(TT, XX, ex, rstride=1, cstride=1, cmap='jet')
    cbar1 = plt.colorbar(fig1, ax=ax1, fraction=0.02, pad=0.1, shrink=0.8)
    cbar1.ax.tick_params(labelsize=6)  # 调整颜色条数字字号
    ax1.set_xlabel('$t$', labelpad=5, fontsize=10)
    ax1.set_ylabel('$x$', labelpad=5, fontsize=10)
    ax1.set_title('Exact $u(t,x)$', fontsize=10)
    ax1.tick_params(axis='both', which='major', labelsize=6)  # 调整字号

    ax2 = plt.subplot(gs1[0, 1], projection='3d')
    fig2 = ax2.plot_surface(TT, XX, pre, rstride=1, cstride=1, cmap='jet')
    cbar2 = plt.colorbar(fig2, ax=ax2, fraction=0.02, pad=0.1, shrink=0.8)
    cbar2.ax.tick_params(labelsize=6)  # 调整颜色条数字字号
    ax2.set_xlabel('$t$', labelpad=5, fontsize=10)
    ax2.set_ylabel('$x$', labelpad=5, fontsize=10)
    ax2.set_title('Pred $u(t,x)$', fontsize=10)
    ax2.tick_params(axis='both', which='major', labelsize=6)  # 调整字号

    '''3D error'''
    # ax3 = plt.subplot(gs1[0, 2], projection='3d')
    # fig3 = ax3.plot_surface(TT, XX, err, rstride=1, cstride=1, cmap='jet')
    # plt.colorbar(fig3, fraction=0.02, pad=0.1, shrink=0.8)
    '''2D error'''
    ax3 = plt.subplot(gs1[0, 2])
    fig3 = plt.pcolor(TT, XX, err, cmap='jet', shading='auto')  # 绘制热图
    cbar3 = plt.colorbar(fig3, ax=ax3, orientation='vertical', pad=0.02, fraction=0.05, shrink=0.8)  # 调整色条
    cbar3.ax.tick_params(labelsize=6)  # 调整颜色条数字字号    ax2.set_xlabel('$t$', labelpad=10)
    # cbar3.set_label('Error Magnitude', fontsize=10)
    ax3.set_xlabel('$t$', labelpad=5, fontsize=10)
    ax3.set_ylabel('$x$', labelpad=5, fontsize=10)
    ax3.set_title('Absolute Error', fontsize=10)
    ax3.tick_params(axis='both', which='major', labelsize=6)
    ax3.set_box_aspect(0.8)  # 设置宽高比例为1:1

    for idx, t_idx in enumerate([0, int(t_test_N / 2), -1]):
        ax = fig_all.add_subplot(gs1[1, idx])
        ax.plot(x_test, u_test_np.reshape((t_test_N, x_test_N))[:, t_idx], 'b-', linewidth=2, label='Prediction')
        ax.plot(x_test, u_exact_np.reshape((t_test_N, x_test_N))[:, t_idx], 'r--', linewidth=2, label='Exact')
        ax.set_title(f'$t = {float(t_test[t_idx].item()):.1f}$', fontsize=10)
        ax.set_xlabel('$x$', labelpad=5, fontsize=10)
        ax.set_ylabel('$u(x,t)$', labelpad=5, fontsize=10)
        ax.set_box_aspect(0.8)
        ax.legend(fontsize=10)

    fig_all.savefig(f'Figure/{PDE_name}_accuracy.pdf', dpi=300)
    plt.show()

def show_loss():
    i_loss_collect = np.array(model.i_loss_collect)
    b_loss_collect = np.array(model.b_loss_collect)
    v_loss_collect = np.array(model.v_loss_collect)
    causal_w_collect = np.array(model.causal_w_collect)

    fig = plt.figure(figsize=(8.8, 2.8), constrained_layout=True)
    gs1 = gridspec.GridSpec(1, 2, figure=fig)

    ax1 = plt.subplot(gs1[0, 0])
    ax1.plot(b_loss_collect[:, 0], b_loss_collect[:, 1], 'b-', linewidth=1, label='$\mathcal{L}_b$')
    ax1.plot(v_loss_collect[:, 0], v_loss_collect[:, 1], 'r-', linewidth=1, label='$\mathcal{L}_v$')
    ax1.set_xlabel('Epoch', labelpad=5, fontsize=10)
    ax1.set_ylabel('Loss', labelpad=5, fontsize=10)
    ax1.set_yscale('log')
    ax1.set_box_aspect(0.8)
    ax1.legend()
    # plt.savefig(f'Figure/{PDE_name}/{PDE_name}_loss.pdf')

    ax2 = plt.subplot(gs1[0, 1])
    ax2.plot(t, causal_w_collect[0, :], 'b-', linewidth=1, label='Iter = 0')
    ax2.plot(t, causal_w_collect[50, :], 'g-', linewidth=1, label='Iter = 50')
    ax2.plot(t, causal_w_collect[100, :], 'orange', linewidth=1, label='Iter = 100')
    ax2.plot(t, causal_w_collect[500, :], 'purple', linewidth=1, label='Iter = 500')
    ax2.plot(t, causal_w_collect[1000, :], 'brown', linewidth=1, label='Iter = 1000')
    ax2.plot(t, causal_w_collect[-1, :], 'r-', linewidth=1, label=f'Iter = {model.net.iter}')
    ax2.set_xlabel('$t$', labelpad=5, fontsize=10)
    ax2.set_ylabel('Causal weight', labelpad=5, fontsize=10)
    ax2.set_box_aspect(0.8)
    # 调整图例位置和字体大小
    ax2.legend(
        loc='upper left',  # 图例放置在左上角
        fontsize=8,  # 调整字体大小
        bbox_to_anchor=(1.05, 1),  # 将图例放置在子图右侧外部
        borderaxespad=0.  # 图例与子图的距离
    )
    # plt.savefig(f'Figure/{PDE_name}/{PDE_name}_weight.pdf')

    # ax3 = plt.subplot(gs1[0, 2])
    # ax3.plot(b_loss_collect[:, 0], b_loss_collect[:, 1], 'b-', linewidth=2, label='$\mathcal{L}_b$')
    # ax3.plot(v_loss_collect[:, 0], v_loss_collect[:, 1], 'r-', linewidth=2, label='$\mathcal{L}_v$')
    # ax3.set_xlabel('Epoch', labelpad=5, fontsize=10)
    # ax3.set_ylabel('Loss', labelpad=5, fontsize=10)
    # ax3.set_yscale('log')
    # ax3.set_box_aspect(0.8)
    # ax3.legend()
    # plt.savefig(f'Figure/{PDE_name}/{PDE_name}_loss.pdf')

    plt.savefig(f'Figure/{PDE_name}_loss.pdf', dpi=300)
    plt.show()

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    # use_gpu = True
    use_gpu = False  # torch.cuda.is_available()
    set_seed(1234)

    layers = [2, 15, 1]
    # layers = [2, 20, 20, 20, 20, 20, 20, 20, 1]
    # layers = [2, 10, 10, 10, 10, 10, 10, 10, 1]
    # layers = [2, 20, 20, 20, 1]
    net = is_cuda(Net_PMNN(layers))

    alpha = 0.8
    sigma = 1 - alpha / 2
    k = 1

    lb = np.array([0.0, 0.0]) # low boundary
    ub = np.array([1.0, 1.0]) # up boundary

    '''train data'''
    t_N = 101
    N_Quad = 18  # 积分点的数量
    x_quad, w_quad, jacobian = data_quad(N_Quad)
    x_N = N_Quad

    x_quad = lb[1] + (ub[1] - lb[1]) / 2 * (x_quad + 1)  # 将积分区间由[-1,1]变为[0,1]
    t, x_data = data_train(x_quad)

    # t, x_data = data_train()

    '''test data'''
    t_test_N = 100
    x_test_N = 100

    t_test, x_test, tx_test, tx_test_exact = data_test()

    '''Train'''
    model = Model(
        net=net,
        x_data=x_data,
        t=t,
        lb=lb,
        ub=ub,
        tx_test=tx_test,
        tx_test_exact=tx_test_exact,
    )

    cof = (gamma(2 - alpha) / model.dt ** (-alpha)) / C(0)

    model.train(LBGFS_epochs=5000)

    '''画图'''
    # show_accuracy()
    # show_loss()

'''
0.75
LBGFS done!
LBGFS==Test_L2error: 1.08e-03
LBGFS==Training time: 80.83
Iter: 1088 Loss: 7.856919546611607e-05



0.5
LBGFS done!
LBGFS==Test_L2error: 2.16e-03
LBGFS==Training time: 116.37
Iter: 1498 Loss: 0.0003592545399442315


0.25
LBGFS done!
LBGFS==Test_L2error: 3.93e-03
LBGFS==Training time: 113.41
Iter: 1475 Loss: 0.0025826781056821346
'''