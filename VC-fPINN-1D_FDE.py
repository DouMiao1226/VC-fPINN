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
from pyDOE import lhs

PDE_name = '1D_FDE'

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
    NQ_u = N_Quad  
    [x_quad, w_quad] = GaussLobattoJacobiWeights(NQ_u, 0, 0)  # (N_Quad,)
    jacobian = (ub[1] - lb[1]) / 2
    w_quad = w_quad[:, None]  # (N_Quad,1)

    return x_quad, w_quad, jacobian

def Test_fcn(N_testfcn, x_quad, test_type='Legendre'):
    x_testfcn = []
    dx_testfcn = []
    dxx_testfcn = []
    x = x_quad
    for n in range(1, N_testfcn + 1):
        '''poly'''
        if test_type == 'Chebyshev1':  
            n = n + 1
            test_fcn_value = Jacobi(n+1, -1/2, -1/2, x) - Jacobi(n-1, -1/2, -1/2, x) 
            if n == 1:
                d1test = n * Jacobi(n-1, -1/2, -1/2, x)
                d2test = n * (n-1) * Jacobi(n-2, -1/2, -1/2, x)
            else:
                d1test = n * Jacobi(n-1, -1/2, -1/2, x)
                d2test = n * (n-1) * Jacobi(n-2, -1/2, -1/2, x)
        elif test_type == 'Chebyshev2':  
            n = n + 1
            test_fcn_value = Jacobi(n + 1, 1/2, 1/2, x) - Jacobi(n - 1, 1/2, 1/2, x) 
            if n == 1:
                d1test = (n + 1) * Jacobi(n, 1 / 2, 1 / 2, x) - n * Jacobi(n - 1, 1 / 2, 1 / 2, x)
                d2test = (n + 1) * n * Jacobi(n - 1, 2, 2, x) - n * (n - 1) * Jacobi(n - 2, 2, 2, x)
            else:
                d1test = (n + 1) * Jacobi(n, 1 / 2, 1 / 2, x) - n * Jacobi(n - 1, 1 / 2, 1 / 2, x)
                d2test = (n + 1) * n * Jacobi(n - 1, 2, 2, x) - n * (n - 1) * Jacobi(n - 2, 2, 2, x)
        else:  
            test_fcn_value = Jacobi(n+1, 0, 0, x) - Jacobi(n-1, 0, 0, x) 
            x_testfcn.append(test_fcn_value)  
            if n == 1:
                d1test = ((n + 2) / 2) * Jacobi(n, 1, 1, x)
                d2test = ((n + 2) * (n + 3) / (2 * 2)) * Jacobi(n - 1, 2, 2, x)
            elif n == 2:
                d1test = ((n + 2) / 2) * Jacobi(n, 1, 1, x) - ((n) / 2) * Jacobi(n - 2, 1, 1, x)
                d2test = ((n + 2) * (n + 3) / (2 * 2)) * Jacobi(n - 1, 2, 2, x)
            else:
                d1test = ((n + 2) / 2) * Jacobi(n, 1, 1, x) - ((n) / 2) * Jacobi(n - 2, 1, 1, x)
                d2test = ((n + 2) * (n + 3) / (2 * 2)) * Jacobi(n - 1, 2, 2, x) - ((n) * (n + 1) / (2 * 2)) * Jacobi(
                    n - 3, 2, 2, x)
        x_testfcn.append(test_fcn_value)  
        dx_testfcn.append(d1test)
        dxx_testfcn.append(d2test)
    x_testfcn = np.asarray(x_testfcn)
    dx_testfcn = np.asarray(dx_testfcn)
    dxx_testfcn = np.asarray(dxx_testfcn)

    return x_testfcn, dx_testfcn, dxx_testfcn

def data_train(x_quad):
    t = np.linspace(lb[0], ub[0], t_N)[:, None]
    x_data = x_quad[:, None]  # (N_Quad,1)
    x_data = torch.from_numpy(x_data).float()
    return t, x_data

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

class Net(nn.Module):
    def __init__(self, layers):
        super(Net, self).__init__()
        self.layers = layers
        self.iter = 0
        self.activation = nn.Tanh()
        # self.activation = nn.ELU()
        # self.activation = torch.sin
        self.loss_function = nn.MSELoss(reduction='mean')
        self.linear = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])
        for i in range(len(layers) - 1):
            nn.init.xavier_normal_(self.linear[i].weight.data, gain=1.0)
            nn.init.zeros_(self.linear[i].bias.data)

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x)
        a = self.activation(self.linear[0](x))
        for i in range(1, len(self.layers) - 2):
            z = self.linear[i](a)
            a = self.activation(z)
        a = self.linear[-1](a)
        return a

class StopTrainingException(Exception):
    pass

class Model:
    def __init__(self, net, x_quad, x_data, t, lb, ub,
                 tx_test, tx_test_exact, tol = 1.0
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
        self.tol = tol

        self.x_quad = x_quad,
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
        self.rr_loss_collect = []
        self.v_loss_collect = []
        self.W_min_collect = []
        self.total_loss_collect = []
        self.error_collect = []
        self.pred_u_collect = []
        self.t_collect = []
        self.causal_w_collect = []
        self.causal_w_collect.append(np.zeros_like(self.t).flatten())

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
        return self.net(H) * x[:, [0]] + x[:, [1]] ** 2

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
        u_x = u_x.reshape(self.t_N, -1)

        return u_n, Lu, u_x

    def u_x_b(self):
        x_b1 = Variable(self.tx_b1, requires_grad=True)
        x_b2 = Variable(self.tx_b2, requires_grad=True)
        u_b1 = self.train_U(x_b1)
        u_b2 = self.train_U(x_b2)
        d1 = torch.autograd.grad(u_b1, x_b1, grad_outputs=torch.ones_like(u_b1),
                                 create_graph=True)
        d2 = torch.autograd.grad(u_b2, x_b2, grad_outputs=torch.ones_like(u_b2),
                                      create_graph=True)
        u_x_b1 = d1[0][:, [1]]
        u_x_b2 = d2[0][:, [1]]

        u_b1 = u_b1.reshape(self.t_N, -1)
        u_x_b1 = u_x_b1.reshape(self.t_N, -1)

        u_b2 = u_b2.reshape(self.t_N, -1)
        u_x_b2 = u_x_b2.reshape(self.t_N, -1)

        return u_b1, u_x_b1, u_b2, u_x_b2

    def Var_loss(self):
        u_n, Lu, u_x = self.Lu_and_F()
        u_b1, u_x_b1, u_b2, u_x_b2 = self.u_x_b()
        Lu_Var = torch.zeros((t_N, N_testfcn))
        u_n_Var = torch.zeros((t_N, N_testfcn))
        u_b_Var = torch.zeros((t_N, N_testfcn))

        if var_form == 0: 
            # for i in range(N_testfcn):
            for j in range(t_N):
                Lu_Var[j, :] = (Lu[j, :] * u_n[j, :] * w_quad.T).sum()
                u_n_Var[j, :] = (u_n[j, :] * u_n[j, :] * w_quad.T).sum()
            Lu_Var = jacobian * Lu_Var
            u_n_Var = jacobian * u_n_Var

        if var_form == 1:
            for i in range(N_testfcn):
                for j in range(t_N):
                    Lu_Var[j, i] = (Lu[j, :] * x_testfcn[i, :] * w_quad.T).sum()
                    u_n_Var[j, i] = (u_n[j, :] * x_testfcn[i, :] * w_quad.T).sum()
                    # Lu_Var[j, i] = (Lu[j, :] ** 2 * x_testfcn[i, :] * w_quad.T).sum()
                    # u_n_Var[j, i] = (u_n[j, :] ** 2 * x_testfcn[i, :] * w_quad.T).sum()
            Lu_Var = jacobian * Lu_Var
            u_n_Var = jacobian * u_n_Var

        if var_form == 2:
            for i in range(N_testfcn):
                for j in range(t_N):
                    Lu_Var[j, i] = (u_x[j, :] * dx_testfcn[i, :] * w_quad.T).sum()
                    u_n_Var[j, i] = (u_n[j, :] * x_testfcn[i, :] * w_quad.T).sum()
                    u_b_Var[j, i] = u_x_b2[j, :] * testfcn[i, -1] - u_x_b1[j, :] * testfcn[i, 0]
            Lu_Var = -jacobian * Lu_Var
            u_n_Var = jacobian * u_n_Var
            u_b_Var = jacobian * u_b_Var

        loss_V = []
        # loss_casual = torch.zeros_like(u_n)
        loss_casual = torch.zeros_like(u_n_Var)
        # print('shape of u_n:', u_n.shape)
        # print('shape of u_n_Var:', u_n_Var.shape)

        for j in range(1, N_testfcn):
            loss_Var = []
            for n in range(1, self.t_N):
                if n == 1:
                    loss_var = cof * (Lu_Var[n, j] + u_b_Var[n, j]) + (C(n - 1) / C(0)) * u_n_Var[0, j] - u_n_Var[n, j]
                else:
                    loss_var = is_cuda(cof * (Lu_Var[n, j] + u_b_Var[n, j]) + (C(n - 1) / C(0)) * u_n_Var[0, j] - u_n_Var[n, j])
                    for k in range(1, n):
                        loss_var = loss_var + ((C(n - k - 1) - C(n - k)) / C(0)) * u_n_Var[k, j]

                loss_casual[n, j] = torch.square(loss_var)
                loss_Var.append(loss_var)

            loss_Var = torch.stack(loss_Var)
            loss_Var = torch.mean(torch.square(loss_Var))
            # print(loss_Var)
            loss_V.append(loss_Var)

        loss_V = torch.stack(loss_V)
        loss_V = torch.mean(torch.square(loss_V))
        loss_C = loss_casual

        return loss_V, loss_C

    def calculate_loss(self):
        loss_i = torch.mean((self.train_U(self.tx_t0) - self.u_t0) ** 2)
        self.i_loss_collect.append([self.net.iter, loss_i.item()])
        loss_b1 = torch.mean((self.train_U(self.tx_b1) - self.u_x_b1) ** 2)
        loss_b2 = torch.mean((self.train_U(self.tx_b2) - self.u_x_b2) ** 2)
        loss_b = loss_b1 + loss_b2
        self.b_loss_collect.append([self.net.iter, loss_b.item()])

        loss_va, loss_c = self.Var_loss()
        self.v_loss_collect.append([self.net.iter, torch.mean(loss_c).item()])

        return loss_i, loss_b, loss_va, loss_c

    def adam_loss(self):
        loss_i, loss_b, loss_va, loss_c = self.calculate_loss()

        loss_it = loss_i
        L_0 = 1e1 * loss_it
        M = np.triu(np.ones((t_N, t_N)), k=1).T
        M = torch.tensor(M, dtype=torch.float32)
        L_t = loss_c 
        W = 1 / ((1 + self.tol * (M @ L_t + L_0)) ** 3) 

        loss = torch.mean(W * L_t) + loss_b
        W_ADAM = W

        loss.backward()
        self.net.iter += 1
        return loss, W_ADAM

    def LBGFS_loss(self):
        self.optimizer_LBGFS.zero_grad()
        loss_i, loss_b, loss_va, loss_c = self.calculate_loss()

        loss_it = loss_i
        L_0 = 1e1 * loss_it
        M = np.triu(np.ones((t_N, t_N)), k=1).T
        M = torch.tensor(M, dtype=torch.float32)
        L_t = torch.mean(loss_c, dim=1) #+ tol * loss_b
        W = 1 / ((1 + self.tol * (M @ L_t + L_0)) ** 3)#.detach() # p阶多项式 # 5.01e-4

        loss_rr = torch.mean(W * L_t)
        loss = torch.mean(W * L_t) + loss_b #* N_quad / N_testfcn
        W_LBFGS = W
        W_min = W_LBFGS.min()
        self.total_loss_collect.append([self.net.iter, loss.item()])
        self.rr_loss_collect.append([self.net.iter, loss_rr.item()])
        self.causal_w_collect.append(W_LBFGS.detach().numpy())
        loss.backward()
        self.net.iter += 1

        print('Iter:', self.net.iter)
        print('Loss:', loss.item())
        pred = self.train_U(tx_test).cpu().detach().numpy()
        exact = self.tx_test_exact.cpu().detach().numpy()
        error = np.linalg.norm(pred - exact, 2) / np.linalg.norm(exact, 2)
        error = torch.tensor(error)
        self.error_collect.append([self.net.iter, error.item()])
        print('L2error:', '{0:.2e}'.format(error))
        print('W_min:', W_min)
        print('-------------------------------------------------')

        if self.net.iter % 10 == 0:
            pred_u = self.train_U(tx_test).cpu().detach().numpy()
            pred_u = torch.from_numpy(pred_u).cpu()
            self.pred_u_collect.append([pred_u.tolist()])

        return loss

    def train(self, adam_epochs=1000, LBGFS_epochs=1500):
        optimizer_adam = torch.optim.Adam(self.net.parameters())
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
        pbar = trange(adam_epochs, ncols=100)
        for i in pbar:
            optimizer_adam.zero_grad()
            loss, W_adam = self.adam_loss()
            optimizer_adam.step()
            W_min = W_adam.min()
            pbar.set_postfix({'Iter': self.net.iter,
                              'Loss': '{0:.2e}'.format(loss.item()),
                              'W_min': W_min
                              })

        print('Adam done!')

        self.optimizer_LBGFS.step(self.LBGFS_loss)
        print('LBGFS done!')
        pred = self.train_U(tx_test).cpu().detach().numpy()
        exact = self.tx_test_exact.cpu().detach().numpy()
        error = np.linalg.norm(pred - exact, 2) / np.linalg.norm(exact, 2)
        print('LBGFS==Test_L2error:', '{0:.2e}'.format(error))

        elapsed = time.time() - start_time
        print('LBGFS==Training time: %.2f' % elapsed)

        pred = self.train_U(tx_test).cpu().detach().numpy()
        exact = self.tx_test_exact.cpu().detach().numpy()
        error = np.linalg.norm(pred - exact, 2) / np.linalg.norm(exact, 2)
        print('Test_L2error:', '{0:.2e}'.format(error))

        elapsed = time.time() - start_time
        print('Training time: %.2f' % elapsed)
        return error, elapsed, self.LBGFS_loss().item()


# def save_causal_weight(causal_w_collect):
#     np.savetxt(f'Weight/W_{PDE_name}.txt', causal_w_collect)
#     np.savetxt(f'Weight/t_{PDE_name}.txt', t)

# def save_error(error_collect):
#     np.savetxt(f'Error/Error_{PDE_name}.txt', error_collect)

# def save_loss(i_loss_collect, b_loss_collect, v_loss_collect, total_loss):
#     np.savetxt(f'Loss/Loss_i_{PDE_name}.txt', i_loss_collect)
#     np.savetxt(f'Loss/Loss_b_{PDE_name}.txt', b_loss_collect)
#     np.savetxt(f'Loss/Loss_v_{PDE_name}.txt', v_loss_collect)
#     np.savetxt(f'Loss/Loss_total_{PDE_name}.txt', total_loss)

# def save_draw_data(draw_t, draw_x, draw_ex, draw_pre):
#     save_dir = "DrawData"
#     os.makedirs(save_dir, exist_ok=True)
#     np.save(f'{save_dir}/draw_t_{PDE_name}.npy', draw_t)
#     np.save(f'{save_dir}/draw_x_{PDE_name}.npy', draw_x)
#     np.save(f'{save_dir}/draw_ex_{PDE_name}.npy', draw_ex)
#     np.save(f'{save_dir}/draw_pre_{PDE_name}.npy', draw_pre)

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    # use_gpu = True
    use_gpu = False  
    set_seed(1234)

    layers = [2, 15, 1] 
    
    net = is_cuda(Net(layers))

    alpha = 0.25
    sigma = 1 - alpha / 2
    k = 1
    tol = 1e0

    lb = np.array([0.0, 0.0]) # low boundary
    ub = np.array([1.0, 1.0]) # up boundary

    '''test function'''
    var_form = 1  
    N_testfcn = 12  
    N_Quad = 18  
    x_quad, w_quad, jacobian = data_quad(N_Quad)
    w_quad = torch.from_numpy(w_quad)

    '''test_type = Chebyshev1, Chebyshev2, Legendre'''
    # Test_fcn_type = 'Chebyshev1'
    # Test_fcn_type = 'Chebyshev2'
    Test_fcn_type = 'Legendre'
    x_testfcn, dx_testfcn, dxx_testfcn = Test_fcn(N_testfcn, x_quad, test_type=Test_fcn_type)
    x_testfcn = torch.from_numpy(x_testfcn)
    dx_testfcn = torch.from_numpy(dx_testfcn)
    dxx_testfcn = torch.from_numpy(dxx_testfcn)

    '''train data'''
    t_N = 101 
    x_N = N_Quad

    x_quad = lb[1] + (ub[1] - lb[1]) / 2 * (x_quad + 1) # 将积分区间由[-1,1]变为[0,1]
    t, x_data = data_train(x_quad)

    '''test data'''
    t_test_N = 100
    x_test_N = 100

    t_test, x_test, tx_test, tx_test_exact = data_test()
    '''test_type = Chebyshev1, Chebyshev2, Legendre'''
    testfcn, d_testfcn, dd_testfcn = Test_fcn(N_testfcn, np.array([[lb[1]],[ub[1]]]),test_type=Test_fcn_type)
    testfcn = torch.from_numpy(testfcn)
    d_testfcn = torch.from_numpy(d_testfcn)

    '''Train'''
    model = Model(
        net=net,
        x_quad=x_quad,
        x_data=x_data,
        t=t,
        lb=lb,
        ub=ub,
        tx_test=tx_test,
        tx_test_exact=tx_test_exact,
    )

    cof = (gamma(2 - alpha) / model.dt ** (-alpha)) / C(0)

    model.tol = tol
    model.train(adam_epochs=0, LBGFS_epochs=5000)

    '''画图'''
    # u_exact_np = tx_test_exact.cpu().detach().numpy()
    # u_test_np = model.predict_U(tx_test).cpu().detach().numpy()
    # draw_t = t_test
    # draw_x = x_test
    # draw_ex = u_exact_np
    # draw_pre = u_test_np
    # save_draw_data(draw_t, draw_x, draw_ex, draw_pre)


