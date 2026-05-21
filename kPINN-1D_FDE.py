import os
import time

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from scipy.special import gamma
from pyDOE import lhs
from tqdm import trange
from GaussJacobiQuadRule_V3 import GaussLobattoJacobiWeights

# -----------------------------------------------------------------------------
# kPINN-type baseline for Example 1 in the VC-fPINNs manuscript
# PDE:   D_t^alpha u = u_xx,  x in [0,1], t in (0,1]
# Exact: u(x,t) = x^2 + 2 t^alpha / Gamma(1+alpha)
#
# kPINN reformulation:
#   u - 1/Gamma(alpha) * sum_i w_i (phi_i)_xx - x^2 = 0,
#   (phi_i)_t + s_i phi_i - u = 0.
# The network outputs [N_theta, phi_1, ..., phi_Np], and
#   u_hat(x,t) = x^2 + t * N_theta(t,x),
# so the initial condition u(x,0)=x^2 is imposed exactly.
# -----------------------------------------------------------------------------

PDE_name = 'kPINN-1D_FDE'


def set_seed(seed):
    torch.set_default_dtype(torch.float)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def is_cuda(data):
    if use_gpu:
        data = data.cuda()
    return data


def exact_u(X):
    """X[:,0]=t, X[:,1]=x."""
    return X[:, [1]] ** 2 + (2.0 * k * X[:, [0]] ** alpha) / gamma(alpha + 1.0)


def g_kpinn(X):
    """
    For Example 1, after applying I_t^alpha:
        u - I_t^alpha u_xx = u(x,0) = x^2.
    """
    return X[:, [1]] ** 2


def get_soe_params(alpha_value, Np=32, y_min=-12.0, y_max=12.0):
    """
    Generate a simple sum-of-exponentials approximation for t^(alpha-1):
        t^(alpha-1) ~= sum_i w_i exp(-s_i t).

    This uses the identity
        t^(alpha-1) = 1/Gamma(1-alpha) * int_0^inf exp(-s t) s^(-alpha) ds,
    and a trapezoidal rule after the transform s=exp(y).

    In the residual we use
        1/Gamma(alpha) * sum_i w_i A phi_i,
    so w_i here approximate the kernel without the factor 1/Gamma(alpha).

    This is a practical kPINN-type baseline. For a strict reproduction of Gao et
    al., replace these parameters by their optimized SoE table when available.
    """
    y = np.linspace(y_min, y_max, Np)
    h = (y_max - y_min) / (Np - 1)
    s = np.exp(y)
    w = h * np.exp((1.0 - alpha_value) * y) / gamma(1.0 - alpha_value)

    w = torch.tensor(w, dtype=torch.float32).view(1, -1)
    s = torch.tensor(s, dtype=torch.float32).view(1, -1)
    return is_cuda(w), is_cuda(s)


# Optional fixed parameters reported in the kPINN paper for alpha=0.3,0.5,0.8.
# They are kept here in case you want to compare these alpha values directly.
def get_paper_soe_params(alpha_value):
    if abs(alpha_value - 0.3) < 1e-12:
        w = [3.8524e-01, 1.8809e-01, 9.2045e-02, 5.0064e-02, 3.2360e-02,
             7.8903e-01, 1.6161e+00, 3.3099e+00, 6.7793e+00, 1.3885e+01,
             2.8439e+01, 5.8247e+01]
        s = [3.5908e-01, 1.2894e-01, 4.6279e-02, 1.5892e-02, 3.0785e-03,
             1.0000e+00, 2.7849e+00, 7.7555e+00, 2.1598e+01, 6.0148e+01,
             1.6750e+02, 4.6647e+02]
    elif abs(alpha_value - 0.5) < 1e-12:
        w = [3.4967e-01, 2.0735e-01, 1.2349e-01, 8.6633e-02, 8.7347e-02,
             5.8968e-01, 9.9443e-01, 1.6770e+00, 2.8281e+00, 4.7692e+00,
             8.0427e+00, 1.3563e+01]
        s = [3.5163e-01, 1.2364e-01, 4.3438e-02, 1.4209e-02, 1.8169e-03,
             1.0000e+00, 2.8439e+00, 8.0878e+00, 2.3001e+01, 6.5412e+01,
             1.8603e+02, 5.2904e+02]
    elif abs(alpha_value - 0.8) < 1e-12:
        w = [1.8932e-01, 1.5259e-01, 1.2404e-01, 1.3083e-01, 3.7063e-01,
             2.3488e-01, 2.9141e-01, 3.6155e-01, 4.4857e-01, 5.5653e-01,
             6.9048e-01, 8.5667e-01]
        s = [3.4017e-01, 1.1572e-01, 3.9294e-02, 1.1883e-02, 6.5088e-04,
             1.0000e+00, 2.9397e+00, 8.6418e+00, 2.5404e+01, 7.4681e+01,
             2.1954e+02, 6.4538e+02]
    else:
        raise ValueError('Paper SoE table only supports alpha=0.3, 0.5, 0.8.')

    w = torch.tensor(w, dtype=torch.float32).view(1, -1)
    s = torch.tensor(s, dtype=torch.float32).view(1, -1)
    return is_cuda(w), is_cuda(s)


def data_train_grid(t_N, x_points=None, x_N=None):
    """
    Generate fixed collocation points.

    If x_points is given, use the same spatial quadrature points as the
    VC-fPINNs code. Otherwise use a uniform grid with x_N points.
    """
    t = np.linspace(lb[0], ub[0], t_N)[:, None]
    if x_points is None:
        if x_N is None:
            raise ValueError('Either x_points or x_N must be provided.')
        x = np.linspace(lb[1], ub[1], x_N)[:, None]
    else:
        x = np.asarray(x_points).reshape(-1, 1)
    tt, xx = np.meshgrid(t, x, indexing='ij')
    tx = np.hstack((tt.reshape(-1, 1), xx.reshape(-1, 1)))
    return t, x, is_cuda(torch.from_numpy(tx).float())


def data_test():
    t_test = np.linspace(lb[0], ub[0], t_test_N)[:, None]
    x_test = np.linspace(lb[1], ub[1], x_test_N)[:, None]
    t_star, x_star = np.meshgrid(t_test, x_test, indexing='ij')
    tx_star = np.hstack((t_star.reshape(-1, 1), x_star.reshape(-1, 1)))
    tx_test = is_cuda(torch.from_numpy(tx_star).float())
    tx_test_exact = exact_u(tx_test)
    return t_test, x_test, tx_test, tx_test_exact


class Net(nn.Module):
    def __init__(self, layers):
        super(Net, self).__init__()
        self.layers = layers
        self.iter = 0
        self.activation = nn.Tanh()
        self.linear = nn.ModuleList([
            nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)
        ])
        for i in range(len(layers) - 1):
            nn.init.xavier_normal_(self.linear[i].weight.data, gain=1.0)
            nn.init.zeros_(self.linear[i].bias.data)

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).float()
        a = self.activation(self.linear[0](x))
        for i in range(1, len(self.layers) - 2):
            a = self.activation(self.linear[i](a))
        return self.linear[-1](a)


class Model:
    def __init__(self, net, tx_f, lb, ub, tx_test, tx_test_exact,
                 w_soe, s_soe, lambda_b=100.0, lambda_phi0=100.0,
                 lambda_u=1.0, lambda_phi=1.0):
        self.net = net
        self.tx_f = tx_f
        self.lb = is_cuda(torch.from_numpy(lb).float())
        self.ub = is_cuda(torch.from_numpy(ub).float())
        self.tx_test = tx_test
        self.tx_test_exact = tx_test_exact
        self.w_soe = w_soe
        self.s_soe = s_soe
        self.Np = w_soe.shape[1]

        self.lambda_b = lambda_b
        self.lambda_phi0 = lambda_phi0
        self.lambda_u = lambda_u
        self.lambda_phi = lambda_phi

        self.optimizer_LBFGS = None
        self.loss_collect = []
        self.loss_u_collect = []
        self.loss_phi_collect = []
        self.loss_b_collect = []
        self.loss_phi0_collect = []
        self.error_collect = []

        self.init_data()

    def init_data(self):
        # Boundary points: x=0 and x=1 with the same time grid as collocation grid.
        t_unique = torch.unique(self.tx_f[:, [0]].detach().cpu()).view(-1, 1)
        t_unique = is_cuda(t_unique.float())
        x0 = torch.zeros_like(t_unique)
        x1 = torch.ones_like(t_unique)
        self.tx_b1 = torch.cat((t_unique, x0), dim=1)
        self.tx_b2 = torch.cat((t_unique, x1), dim=1)
        self.u_b1 = exact_u(self.tx_b1)
        self.u_b2 = exact_u(self.tx_b2)

        # Initial points: t=0 with the same x grid as collocation grid.
        x_unique = torch.unique(self.tx_f[:, [1]].detach().cpu()).view(-1, 1)
        x_unique = is_cuda(x_unique.float())
        t0 = torch.zeros_like(x_unique)
        self.tx_t0 = torch.cat((t0, x_unique), dim=1)

    def net_u_phi(self, x):
        """
        Hard IC: u_hat(t,x)=x^2 + t*N_theta(t,x).
        Network output: [N_theta, phi_1,...,phi_Np].
        """
        H = 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0
        out = self.net(H)
        u_nn = out[:, [0]]
        phi = out[:, 1:]
        u = x[:, [1]] ** 2 + x[:, [0]] * u_nn
        return u, phi

    def predict_U(self, x):
        u, _ = self.net_u_phi(x)
        return u

    @staticmethod
    def gradients(y, x):
        return torch.autograd.grad(
            y, x,
            grad_outputs=torch.ones_like(y),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

    def residuals(self):
        x = Variable(self.tx_f, requires_grad=True)
        u, phi = self.net_u_phi(x)

        phi_t_list = []
        phi_xx_list = []
        for i in range(self.Np):
            phi_i = phi[:, [i]]
            grad_phi_i = self.gradients(phi_i, x)
            phi_i_t = grad_phi_i[:, [0]]
            phi_i_x = grad_phi_i[:, [1]]
            phi_i_xx = self.gradients(phi_i_x, x)[:, [1]]
            phi_t_list.append(phi_i_t)
            phi_xx_list.append(phi_i_xx)

        phi_t = torch.cat(phi_t_list, dim=1)
        phi_xx = torch.cat(phi_xx_list, dim=1)

        weighted_phi_xx = torch.sum(self.w_soe * phi_xx, dim=1, keepdim=True)

        # Main residual: u - 1/Gamma(alpha) sum_i w_i phi_i_xx - x^2 = 0.
        R_u = u - weighted_phi_xx / gamma(alpha) - g_kpinn(x)

        # Auxiliary residuals: phi_t + s_i phi_i - u = 0.
        R_phi = phi_t + self.s_soe * phi - u

        return R_u, R_phi

    def calculate_loss(self):
        R_u, R_phi = self.residuals()

        # Weighted auxiliary loss, following the kPINN idea.
        phi_weights = self.w_soe / torch.sum(self.w_soe)
        loss_u = torch.mean(R_u ** 2)
        loss_phi = torch.mean(phi_weights * (R_phi ** 2))

        u_b1, _ = self.net_u_phi(self.tx_b1)
        u_b2, _ = self.net_u_phi(self.tx_b2)
        loss_b = torch.mean((u_b1 - self.u_b1) ** 2) + torch.mean((u_b2 - self.u_b2) ** 2)

        # Since u(x,0)=x^2 is hard constrained, only phi_i(x,0)=0 is penalized.
        _, phi0 = self.net_u_phi(self.tx_t0)
        loss_phi0 = torch.mean(phi0 ** 2)

        loss = (self.lambda_u * loss_u
                + self.lambda_phi * loss_phi
                + self.lambda_b * loss_b
                + self.lambda_phi0 * loss_phi0)

        return loss, loss_u, loss_phi, loss_b, loss_phi0

    def current_error(self):
        pred = self.predict_U(self.tx_test).detach().cpu().numpy()
        exact = self.tx_test_exact.detach().cpu().numpy()
        return np.linalg.norm(pred - exact, 2) / np.linalg.norm(exact, 2)

    def adam_loss(self):
        loss, loss_u, loss_phi, loss_b, loss_phi0 = self.calculate_loss()
        loss.backward()
        self.net.iter += 1
        return loss, loss_u, loss_phi, loss_b, loss_phi0

    def LBFGS_loss(self):
        self.optimizer_LBFGS.zero_grad()
        loss, loss_u, loss_phi, loss_b, loss_phi0 = self.calculate_loss()
        loss.backward()
        self.net.iter += 1

        if self.net.iter % 10 == 0:
            error = self.current_error()
            self.loss_collect.append([self.net.iter, loss.item()])
            self.loss_u_collect.append([self.net.iter, loss_u.item()])
            self.loss_phi_collect.append([self.net.iter, loss_phi.item()])
            self.loss_b_collect.append([self.net.iter, loss_b.item()])
            self.loss_phi0_collect.append([self.net.iter, loss_phi0.item()])
            self.error_collect.append([self.net.iter, error])
            print('Iter:', self.net.iter)
            print('Loss:', '{0:.3e}'.format(loss.item()),
                  '| Ru:', '{0:.3e}'.format(loss_u.item()),
                  '| Rphi:', '{0:.3e}'.format(loss_phi.item()),
                  '| B:', '{0:.3e}'.format(loss_b.item()),
                  '| Phi0:', '{0:.3e}'.format(loss_phi0.item()))
            print('L2error:', '{0:.2e}'.format(error))
            print('-------------------------------------------------')

        return loss

    def train(self, adam_epochs=20000, LBFGS_epochs=5000, adam_lr=1e-3):
        start_time = time.time()

        if adam_epochs > 0:
            optimizer_adam = torch.optim.Adam(self.net.parameters(), lr=adam_lr)
            pbar = trange(adam_epochs, ncols=100)
            for _ in pbar:
                optimizer_adam.zero_grad()
                loss, loss_u, loss_phi, loss_b, loss_phi0 = self.adam_loss()
                optimizer_adam.step()

                if self.net.iter % 100 == 0:
                    pbar.set_postfix({
                        'Iter': self.net.iter,
                        'Loss': '{0:.2e}'.format(loss.item()),
                        'Ru': '{0:.2e}'.format(loss_u.item()),
                        'Rphi': '{0:.2e}'.format(loss_phi.item())
                    })
            print('Adam done!')

        if LBFGS_epochs > 0:
            self.optimizer_LBFGS = torch.optim.LBFGS(
                self.net.parameters(),
                lr=1.0,
                max_iter=LBFGS_epochs,
                max_eval=LBFGS_epochs,
                history_size=100,
                tolerance_grad=1e-12,
                tolerance_change=1.0 * np.finfo(float).eps,
                line_search_fn='strong_wolfe'
            )
            self.optimizer_LBFGS.step(self.LBFGS_loss)
            print('LBFGS done!')

        error = self.current_error()
        elapsed = time.time() - start_time
        print('Test_L2error:', '{0:.2e}'.format(error))
        print('Training time: %.2f' % elapsed)
        return error, elapsed


def save_results(model):
    os.makedirs('Error', exist_ok=True)
    os.makedirs('Loss', exist_ok=True)
    if len(model.error_collect) > 0:
        np.savetxt(f'Error/Error_{PDE_name}.txt', np.array(model.error_collect))
    if len(model.loss_collect) > 0:
        np.savetxt(f'Loss/Loss_total_{PDE_name}.txt', np.array(model.loss_collect))
        np.savetxt(f'Loss/Loss_u_{PDE_name}.txt', np.array(model.loss_u_collect))
        np.savetxt(f'Loss/Loss_phi_{PDE_name}.txt', np.array(model.loss_phi_collect))
        np.savetxt(f'Loss/Loss_b_{PDE_name}.txt', np.array(model.loss_b_collect))
        np.savetxt(f'Loss/Loss_phi0_{PDE_name}.txt', np.array(model.loss_phi0_collect))


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    use_gpu = False  # torch.cuda.is_available()
    set_seed(1234)

    # Problem parameters
    alpha = 0.3
    k = 1.0
    lb = np.array([0.0, 0.0])
    ub = np.array([1.0, 1.0])

    # Collocation and test grids
    t_N = 101
    N_Quad = 18
    t_test_N = 100
    x_test_N = 100

    # Use the same GLL spatial training points as Var-1D_FDE.py.
    x_quad_ref, _ = GaussLobattoJacobiWeights(N_Quad, 0, 0)
    x_quad_ref = lb[1] + (ub[1] - lb[1]) / 2.0 * (x_quad_ref + 1.0)
    t, x, tx_f = data_train_grid(t_N=t_N, x_points=x_quad_ref)
    t_test, x_test, tx_test, tx_test_exact = data_test()

    # Use the SoE table reported in the kPINN paper whenever available.
    # This is more faithful to FP_Cauchy_1D.py and is also much faster than
    # the crude generated 32-term approximation.
    if alpha in [0.3, 0.5, 0.8]:
        w_soe, s_soe = get_paper_soe_params(alpha)
    else:
        w_soe, s_soe = get_soe_params(alpha, Np=32, y_min=-12.0, y_max=12.0)
    Np = w_soe.shape[1]
    print('alpha =', alpha, '| Np =', Np)

    # Network outputs [N_theta, phi_1, ..., phi_Np]
    layers = [2, 32, 32, 32, 1 + Np]
    net = is_cuda(Net(layers))

    model = Model(
        net=net,
        tx_f=tx_f,
        lb=lb,
        ub=ub,
        tx_test=tx_test,
        tx_test_exact=tx_test_exact,
        w_soe=w_soe,
        s_soe=s_soe,
        lambda_b=100.0,
        lambda_phi0=100.0,
        lambda_u=1.0,
        lambda_phi=1.0
    )

    # Suggested first run: Adam 20000 + LBFGS 5000.
    # For a quick smoke test, reduce these two numbers.
    error, elapsed = model.train(adam_epochs=0, LBFGS_epochs=25000, adam_lr=1e-3)
    save_results(model)

'''
0.5
20000 adams-------------------------------------------------
Iter: 25000
Loss: 3.779e-04 | Ru: 7.994e-05 | Rphi: 2.312e-04 | B: 3.871e-07 | Phi0: 2.801e-07
L2error: 8.44e-03
-------------------------------------------------
LBFGS done!
Test_L2error: 8.44e-03
Training time: 895.65

no adams
-------------------------------------------------
Iter: 13340
Loss: 1.423e-04 | Ru: 1.593e-05 | Rphi: 1.064e-04 | B: 1.199e-07 | Phi0: 7.988e-08
L2error: 1.34e-02
-------------------------------------------------
LBFGS done!
Test_L2error: 1.34e-02
Training time: 631.61

-------------------------------------------------
Iter: 5000
Loss: 2.402e-03 | Ru: 1.274e-04 | Rphi: 1.981e-03 | B: 2.069e-06 | Phi0: 8.607e-07
L2error: 4.57e-03
-------------------------------------------------
LBFGS done!
Test_L2error: 4.58e-03
Training time: 135.82


0.3
-------------------------------------------------
Iter: 5000
Loss: 1.965e-03 | Ru: 9.077e-05 | Rphi: 1.744e-03 | B: 1.025e-06 | Phi0: 2.787e-07
L2error: 5.19e-02
-------------------------------------------------
LBFGS done!
Test_L2error: 5.19e-02
Training time: 208.45

-------------------------------------------------
Iter: 7940
Loss: 8.127e-04 | Ru: 6.426e-05 | Rphi: 6.866e-04 | B: 3.241e-07 | Phi0: 2.937e-07
L2error: 5.54e-02
-------------------------------------------------
LBFGS done!
Test_L2error: 5.54e-02
Training time: 778.94



0.8
-------------------------------------------------
Iter: 12400
Loss: 9.187e-05 | Ru: 9.399e-06 | Rphi: 5.093e-05 | B: 1.620e-07 | Phi0: 1.534e-07
L2error: 4.99e-03
-------------------------------------------------
LBFGS done!
Test_L2error: 4.99e-03
Training time: 855.47

'''