# define the networks

import torch
import os
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt
from torch import nn
from torch import autograd
from torch.autograd import Variable
from options import Opt
from sklearn.metrics import r2_score
from utils import *


class Dataset(torch.utils.data.Dataset):

    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x = self.X[index]

        return x


class Fnn(nn.Module):
    def __init__(self, layers_size, dropout=False, act_func=nn.LeakyReLU(0.01), sn=False):
        super().__init__()
        self.model = nn.Sequential()
        self.activation_func = act_func
        # 参数化网络结构
        if len(layers_size) < 4:
            raise ValueError("网络结构太浅啦")
        i_layer_size = layers_size[0]
        o_layer_size = layers_size[-1]
        h_layer_list = layers_size[1:-1]

        if not sn:
            self.input = nn.Linear(i_layer_size, h_layer_list[0])
        else:
            self.input = nn.utils.spectral_norm(nn.Linear(i_layer_size, h_layer_list[0]))
        self.model.add_module("input_layer", self.input)
        self.model.add_module("ac_fun", self.activation_func)

        for i in range(len(h_layer_list)-1):
            layer_name = "hidden_layer_" + str(i)
            if not sn:
                layer_i = nn.Linear(h_layer_list[i], h_layer_list[i + 1])
            else:
                layer_i = nn.utils.spectral_norm(nn.Linear(h_layer_list[i], h_layer_list[i + 1]))
            self.model.add_module(layer_name, layer_i)
            self.model.add_module("ac_fun"+str(i), self.activation_func)
            if dropout:
                self.model.add_module("dropout_" + str(i), nn.Dropout(p=0.1))
        if not sn:
            self.output = nn.Linear(h_layer_list[-1], o_layer_size)
        else:
            self.output = nn.utils.spectral_norm(nn.Linear(h_layer_list[-1], o_layer_size))
        self.model.add_module("output_layer", self.output)

    def forward(self, x):
        out = self.model(x)

        return out


class TrunkNet(nn.Module):
    def __init__(self, in_dim, out_dim, width=64):
        """
        :param in_dim: 空间坐标维度d
        :param out_dim: 论文中的p
        :param width: 网络宽度
        """
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, width),
            # nn.utils.spectral_norm(nn.Linear(in_dim, width)),
            nn.Tanh(),

            nn.Linear(width, width),
            # nn.utils.spectral_norm(nn.Linear(width, width)),
            nn.Tanh(),

            nn.Linear(width, width),
            # nn.utils.spectral_norm(nn.Linear(width, width)),
            nn.Tanh(),

            nn.Linear(width, width),
            # nn.utils.spectral_norm(nn.Linear(width, width)),
            nn.Tanh(),

            nn.Linear(width, out_dim),
            # nn.utils.spectral_norm(nn.Linear(width, out_dim)),
        )

        self.out = nn.Tanh()

    def forward(self, x):
        x = self.model(x)
        x = self.out(x)

        return x


class BranchNet(nn.Module):
    def __init__(self, in_dim, out_dim, width=32):
        """
        :param in_dim: 输入的场的sensor数目
        :param out_dim: 论文中的p
        :param width: 网络宽度
        """
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, width),
            # nn.utils.spectral_norm(nn.Linear(in_dim, width)),
            nn.Tanh(),

            nn.Linear(width, width),
            # nn.utils.spectral_norm(nn.Linear(width, width)),
            nn.Tanh(),

            nn.Linear(width, width),
            # nn.utils.spectral_norm(nn.Linear(width, width)),
            nn.Tanh(),

            nn.Linear(width, width),
            # nn.utils.spectral_norm(nn.Linear(width, width)),
            nn.Tanh(),

            nn.Linear(width, out_dim),
            # nn.utils.spectral_norm(nn.Linear(width, width)),
        )

    def forward(self, x):
        x = self.model(x)

        return x


class DeepONet(nn.Module):
    def __init__(self, noise_feature, coords_features, p, share=1):
        super().__init__()

        self.trunk = TrunkNet(coords_features, p)
        self.branch = BranchNet(noise_feature, p)

        self.bias_params = torch.ones(1, requires_grad=True)
        self.bias_params = torch.nn.Parameter(self.bias_params)
        self.trunk.register_parameter('params_bias', self.bias_params)

        self.share = share
        self.p = p

    def forward(self, noise, coordinates):
        out_b = self.branch(noise)  # batch_size个函数的快照 的 branch 输出
        out_t = self.trunk(coordinates)  # 所有坐标 的 trunk 输出

        # 数据对齐 （batch_size * sensor_num） * 10
        out_b = out_b.repeat_interleave(coordinates.shape[0], dim=0)  # 按个体重复 如： [1,2] to [1,1,2,2]
        out_t = torch.tile(out_t, [noise.shape[0], 1])  # 按整体重复  如：[1,2] to [1,2,1,2]
        if self.share == 1:
            g_uy_pred = (torch.sum(out_b * out_t, dim=1) + self.bias_params)
            return g_uy_pred

        elif self.share == 2:
            g_uy_pred_x = (torch.sum(out_b[:, :int(0.5*self.p)] * out_t[:, :int(0.5*self.p)], dim=1) + self.bias_params)
            g_uy_pred_y = (torch.sum(out_b[:, int(0.5*self.p):] * out_t[:, int(0.5*self.p):], dim=1) + self.bias_params)
            return g_uy_pred_x, g_uy_pred_y

        elif self.share == 3:
            g_uy_pred_e = (torch.sum(out_b[:, :int((1/3)*self.p)] * out_t[:, :int((1/3)*self.p)], dim=1) + self.bias_params)
            g_uy_pred_x = (torch.sum(out_b[:, int((1/3)*self.p):int((2/3)*self.p)] * out_t[:, int((1/3)*self.p):int((2/3)*self.p)], dim=1) + self.bias_params)
            g_uy_pred_y = (torch.sum(out_b[:, int((2/3)*self.p):] * out_t[:, int((2/3)*self.p):], dim=1) + self.bias_params)
            return g_uy_pred_e, g_uy_pred_x, g_uy_pred_y

    def user_forward(self, noise_array, coordinates_array):
        out_b = self.branch(noise_array)  # batch_size个函数的快照 的 branch 输出
        out_t = self.trunk(coordinates_array)  # 所有坐标 的 trunk 输出

        g_uy_pred = (torch.sum(out_b * out_t, dim=1) + self.bias_params)

        return g_uy_pred


class PodDeepONet(nn.Module):
    def __init__(self, noise_feature, p, trunk_means, trunk_components):
        """
        trunk_means: 预训练的PCA均值代理
        trunk_components： 预训练的PCA成分代理
        """
        super().__init__()

        self.branch = BranchNet(noise_feature, p*3)
        self.trunk_components = trunk_components
        self.trunk_means = trunk_means
        self.p = p*3

    def forward(self, noise, coordinates):
        out_b = self.branch(noise)  # batch_size个函数的快照 的 branch 输出
        out_t = self.trunk_components(coordinates)  # 所有坐标 的 trunk 输出
        out_bias = self.trunk_means(coordinates)

        # 数据对齐
        out_b = out_b.repeat_interleave(coordinates.shape[0], dim=0)  # 按个体重复 如： [1,2] to [1,1,2,2]
        out_t = torch.tile(out_t, [noise.shape[0], 1])  # 按整体重复  如：[1,2] to [1,2,1,2]
        out_bias = torch.squeeze(torch.tile(out_bias, [noise.shape[0], 1]))

        g_uy_pred_e = (torch.sum(out_b[:, :int((1/3)*self.p)]*out_t[:, :int((1/3)*self.p)], dim=1) +
                       out_bias[:, 0])
        g_uy_pred_x = (torch.sum(out_b[:, int((1/3)*self.p):int((2/3)*self.p)]*out_t[:, int((1/3)*self.p):int((2/3)*self.p)], dim=1) +
                       out_bias[:, 1])
        g_uy_pred_y = (torch.sum(out_b[:, int((2/3)*self.p):]*out_t[:, int((2/3)*self.p):], dim=1) +
                       out_bias[:, 2])

        return g_uy_pred_e, g_uy_pred_x, g_uy_pred_y


class PodDeepONet2(nn.Module):
    """ independent branch net"""
    def __init__(self, noise_feature, p, trunk_means, trunk_components):
        """
        trunk_means: 预训练的PCA均值代理
        trunk_components： 预训练的PCA成分代理
        """
        super().__init__()

        self.branch_e = BranchNet(noise_feature, p)
        self.branch_x = BranchNet(noise_feature, p)
        self.branch_y = BranchNet(noise_feature, p)
        self.trunk_components = trunk_components
        self.trunk_means = trunk_means
        self.p = p*3

    def forward(self, noise, coordinates):
        out_b_e = self.branch_e(noise)  # batch_size个函数的快照 的 branch 输出
        out_b_x = self.branch_x(noise)
        out_b_y = self.branch_y(noise)
        out_b = torch.cat([out_b_e, out_b_x, out_b_y], dim=1)
        out_t = self.trunk_components(coordinates)  # 所有坐标 的 trunk 输出
        out_bias = self.trunk_means(coordinates)

        # 数据对齐
        out_b = out_b.repeat_interleave(coordinates.shape[0], dim=0)  # 按个体重复 如： [1,2] to [1,1,2,2]
        out_t = torch.tile(out_t, [noise.shape[0], 1])  # 按整体重复  如：[1,2] to [1,2,1,2]
        out_bias = torch.squeeze(torch.tile(out_bias, [noise.shape[0], 1]))

        g_uy_pred_e = (torch.sum(out_b[:, :int((1/3)*self.p)]*out_t[:, :int((1/3)*self.p)], dim=1) +
                       out_bias[:, 0])
        g_uy_pred_x = (torch.sum(out_b[:, int((1/3)*self.p):int((2/3)*self.p)]*out_t[:, int((1/3)*self.p):int((2/3)*self.p)], dim=1) +
                       out_bias[:, 1])
        g_uy_pred_y = (torch.sum(out_b[:, int((2/3)*self.p):]*out_t[:, int((2/3)*self.p):], dim=1) +
                       out_bias[:, 2])

        return g_uy_pred_e, g_uy_pred_x, g_uy_pred_y


class WGanGP(object):
    """ the architecture used in the paper, 3 DeepONets"""
    def __init__(self, coords_T, resp_features, cuda=True):
        self.coords_T = coords_T
        self.resp_features = resp_features

        self.gen_params = DeepONet(Opt.noise_features, Opt.coords_features, Opt.p)
        self.gen_resp_x = DeepONet(Opt.noise_features, Opt.coords_features, Opt.p)
        self.gen_resp_y = DeepONet(Opt.noise_features, Opt.coords_features, Opt.p)
        self.disc = Fnn(Opt.layers_size_disc, sn=False)

        self.mse = nn.MSELoss()

        # Check if cuda is available
        self.check_cuda(cuda)

        # WGAN_values from paper
        self.batch_size = Opt.batch_size
        self.lr_g = Opt.lr_g
        self.lr_d = Opt.lr_d
        self.b1 = Opt.beta1
        self.b2 = Opt.beta2

        # WGAN_gradient penalty uses ADAM
        self.d_optimizer = torch.optim.Adam(self.disc.parameters(), lr=self.lr_d, betas=(self.b1, self.b2))
        self.g_optimizer = torch.optim.Adam([{"params": self.gen_resp_x.parameters(), "lr": self.lr_g},
                                             {"params": self.gen_resp_y.parameters(), "lr": self.lr_g},
                                             {"params": self.gen_params.parameters(), "lr": self.lr_g}],
                                            betas=(self.b1, self.b2))

        self.start_g_iter = -1
        self.generator_iters = Opt.epochs
        self.critic_iter = Opt.critic_iter
        self.lambda_term = Opt.gp_lambda

    def get_torch_variable(self, arg):
        if self.cuda:
            return Variable(arg).cuda()
        else:
            return Variable(arg)

    def check_cuda(self, cuda_flag=False):
        print(cuda_flag)
        if cuda_flag:
            self.cuda = True
            self.disc.cuda()
            self.gen_resp_x.cuda()
            self.gen_resp_y.cuda()
            self.gen_params.cuda()
            print("Cuda enabled flag: {}".format(self.cuda))
        else:
            self.cuda = False

    @staticmethod
    def get_batches(data_loader):
        while True:
            for i, data in enumerate(data_loader):
                yield data

    def train(self, train_loader):
        """ training """
        time_begin = time.time()

        self.data = self.get_batches(train_loader)
        batch_num = len(train_loader)

        one = torch.tensor(1, dtype=torch.float)
        mone = one * -1
        if self.cuda:
            one = one.cuda()
            mone = mone.cuda()

        for g_iter in range(self.start_g_iter + 1, self.generator_iters + 1):
            # Requires grad, Generator requires_grad = False
            iter_time = time.time()
            for batch_iter in range(batch_num):
                data_batch = self.data.__next__()
                # Check for batch to have full batch_size
                if data_batch.size()[0] != self.batch_size:
                    continue

                data_batch = self.get_torch_variable(data_batch)
                for p in self.disc.parameters():
                    p.requires_grad = True

                # Train discriminator forward-loss-backward-update self.critic_iter times
                # while 1 Generator forward-loss-backward-update

                for d_iter in range(self.critic_iter):
                    self.disc.zero_grad()
                    # Train discriminator
                    # WGan - Training discriminator more iterations than generator Train with real data
                    d_loss_real = self.disc(data_batch)
                    d_loss_real = d_loss_real.mean()
                    d_loss_real.backward(mone)  # -Ex~pr(D(x))

                    # Train with fake data
                    z = self.get_torch_variable(torch.randn(self.batch_size, Opt.noise_features))
                    fake_data_batch = self.generate_fake_data(z, self.coords_T)

                    d_loss_fake = self.disc(fake_data_batch)
                    d_loss_fake = d_loss_fake.mean()
                    d_loss_fake.backward(one)  # Ex~pg(D(G(x)))

                    # Train with gradient penalty
                    gradient_penalty = self.calculate_gradient_penalty(data_batch.data, fake_data_batch.data, div=False)
                    gradient_penalty.backward()
                    #
                    d_loss = d_loss_fake - d_loss_real + gradient_penalty
                    Wasserstein_d = d_loss_real - d_loss_fake
                    self.d_optimizer.step()
                    print(f'batch: {batch_iter+1}/{batch_num},  Discriminator iteration: {d_iter}/{self.critic_iter},'
                          f'd-loss: {d_loss}, loss_fake: {d_loss_fake}, loss_real: {d_loss_real}')

                # Generator update
                for p in self.disc.parameters():
                    p.requires_grad = False  # to avoid computation

                self.gen_resp_x.zero_grad()
                self.gen_resp_y.zero_grad()
                self.gen_params.zero_grad()

                # train generator
                # compute loss with fake images
                z = self.get_torch_variable(torch.randn(self.batch_size, Opt.noise_features))
                fake_data_batch = self.generate_fake_data(z, self.coords_T)

                g_loss = self.disc(fake_data_batch)
                g_loss = g_loss.mean()
                g_loss.backward(mone)  # -Ex~pg(D(G(x))),

                # g_cost = -g_loss
                self.g_optimizer.step()
                print(f'batch: {batch_iter+1}/{batch_num}, Generator iteration: {g_iter}/{self.generator_iters}, '
                      f'g_loss: {-g_loss}')

            res_s = (self.generator_iters - g_iter)*(time.time() - iter_time)
            res_h = int(res_s / 3600)
            res_min = int((res_s - res_h*3600) / 60)
            print(f" =======  Remaining training time：{res_h} h {res_min} min =======")
            # Saving sampling images every SAVE_PER_EPOCH generator iterations

            if g_iter % Opt.SAVE_PER_EPOCH == 0:
                self.save_images(fake_data_batch, g_iter)
                self.save_models(g_iter)

        time_end = time.time()
        print('Time of training-{}'.format((time_end - time_begin)))

    def generate_fake_data(self, z, coords_T):
        fake_data_params = self.gen_params(z, coords_T).reshape(-1, Opt.params_features)
        fake_data_resp_x = self.gen_resp_x(z, coords_T).reshape(-1, int(0.5*Opt.resp_features))
        fake_data_resp_y = self.gen_resp_y(z, coords_T).reshape(-1, int(0.5*Opt.resp_features))

        fake_data_batch = torch.cat([fake_data_params, fake_data_resp_x, fake_data_resp_y], dim=1)

        return fake_data_batch

    def calculate_gradient_penalty(self, real_data_batch, fake_data_batch, div=False):
        eta = torch.FloatTensor(self.batch_size, 1).uniform_(0, 1)
        eta = eta.expand(self.batch_size, real_data_batch.size(1))
        if self.cuda:
            eta = eta.cuda()
        else:
            eta = eta

        interpolated = eta * real_data_batch + ((1 - eta) * fake_data_batch)

        if self.cuda:
            interpolated = interpolated.cuda()
        else:
            interpolated = interpolated

        # define it to calculate gradient
        interpolated = Variable(interpolated, requires_grad=True)

        # calculate probability of interpolated examples
        prob_interpolated = self.disc(interpolated)

        # calculate gradients of probabilities with respect to examples
        gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                                  grad_outputs=torch.ones(
                                      prob_interpolated.size()).cuda() if self.cuda else torch.ones(
                                      prob_interpolated.size()),
                                  create_graph=True, retain_graph=True)[0]

        if not div:
            grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_term
        else:
            grad_penalty = ((gradients.norm(2, dim=1)) ** 6).mean() * self.lambda_term

        return grad_penalty

    def save_models(self, g_iter):
        if not os.path.exists('save_models/'):
            os.makedirs('save_models/')

        state = {
            "stat_dict_params": self.gen_params.state_dict(),
            "stat_dict_resp_x": self.gen_resp_x.state_dict(),
            "stat_dict_resp_y": self.gen_resp_y.state_dict(),
            "stat_dict_disc": self.disc.state_dict(),
            "optimizer_g": self.g_optimizer.state_dict(),
            "optimizer_d": self.d_optimizer.state_dict(),
            "g_iter": g_iter
        }
        path = "save_models/state_" + str(g_iter) + ".pt"
        torch.save(state, path)

    def load_models(self, path):
        checkpoint = torch.load(path)
        self.gen_params.load_state_dict(checkpoint["stat_dict_params"])
        self.gen_resp_x.load_state_dict(checkpoint["stat_dict_resp_x"])
        self.gen_resp_y.load_state_dict(checkpoint["stat_dict_resp_y"])
        self.disc.load_state_dict(checkpoint["stat_dict_disc"])
        self.g_optimizer.load_state_dict(checkpoint["optimizer_g"])
        self.d_optimizer.load_state_dict(checkpoint["optimizer_d"])
        self.start_g_iter = checkpoint["g_iter"]

    @staticmethod
    def save_images(fake_data_batch, g_iter):
        if not os.path.exists('training_result_images/'):
            os.makedirs('training_result_images/')

        xx, yy = Opt.x_mesh_sensor, Opt.y_mesh_sensor  # 中途加入的变量
        resolution = xx.shape[0]
        samples = fake_data_batch.reshape(-1, Opt.joint_features)
        samples_params = samples[:, :Opt.params_features].data.cpu().numpy()
        samples_resp = samples[:, Opt.params_features:].data.cpu().numpy()
        samples_resp_x = samples_resp[:, :int(0.5*Opt.resp_features)]
        samples_resp_y = samples_resp[:, int(0.5*Opt.resp_features):]
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=[12, 6])
        ax0 = axes[0][0].contourf(xx, yy, samples_params[0].reshape(resolution, resolution))
        axes[0][0].set_title("E")
        plt.colorbar(ax0, ax=axes[0][0])
        ax1 = axes[0][1].contourf(xx, yy, samples_resp_x[0].reshape(resolution, resolution))
        axes[0][1].set_title("disp_x")
        plt.colorbar(ax1, ax=axes[0][1])
        ax2 = axes[0][2].contourf(xx, yy, samples_resp_y[0].reshape(resolution, resolution))
        axes[0][2].set_title("disp_y")
        plt.colorbar(ax2, ax=axes[0][2])

        ax3 = axes[1][0].contourf(xx, yy, samples_params[1].reshape(resolution, resolution))
        plt.colorbar(ax3, ax=axes[1][0])
        ax4 = axes[1][1].contourf(xx, yy, samples_resp_x[1].reshape(resolution, resolution))
        plt.colorbar(ax4, ax=axes[1][1])
        ax5 = axes[1][2].contourf(xx, yy, samples_resp_y[1].reshape(resolution, resolution))
        plt.colorbar(ax5, ax=axes[1][2])
        plt.tight_layout()
        plt.savefig('training_result_images/img_generator_iter_{}.png'.format(str(g_iter)))
        plt.close()


class WGanGP2(WGanGP):
    """ do not use"""
    def __init__(self, coords_T, resp_features, cuda=True):
        super().__init__(coords_T, resp_features, cuda=cuda)

        self.gen_resp = DeepONet(Opt.noise_features, Opt.coords_features, Opt.p*3, share=3).to(Opt.device)

        self.lr_g = Opt.lr_g
        self.lr_d = Opt.lr_d
        # WGAN_gradient penalty uses ADAM
        self.d_optimizer = torch.optim.Adam(self.disc.parameters(), lr=self.lr_d, betas=(self.b1, self.b2))
        self.g_optimizer = torch.optim.Adam([{"params": self.gen_resp.parameters(), "lr": self.lr_g}],
                                             betas=(self.b1, self.b2))

    def generate_fake_data(self, z, coords_T):
        fake_data_params, fake_data_resp_x, fake_data_resp_y = self.gen_resp(z, coords_T)
        fake_data_params = fake_data_params.reshape(-1, Opt.params_features)
        fake_data_resp_x = fake_data_resp_x.reshape(-1, int(0.5*Opt.resp_features))
        fake_data_resp_y = fake_data_resp_y.reshape(-1, int(0.5*Opt.resp_features))
        fake_data_batch = torch.cat([fake_data_params, fake_data_resp_x, fake_data_resp_y], dim=1)

        return fake_data_batch

    def add_sn(self, m):
        for name, layer in m.named_children():
            m.add_module(name, self.add_sn(layer))
            if isinstance(m, nn.Linear):
                return nn.utils.spectral_norm(m)
            else:
                return m


class WGanGP3(WGanGP):
    """ Pod-based OL_GANs"""
    def __init__(self, coords_T, resp_features, means_nn, components_nn, cuda=True):
        self.coords_T = coords_T
        self.resp_features = resp_features

        self.gen = PodDeepONet2(Opt.noise_features, Opt.p, means_nn, components_nn)
        self.disc = Fnn(Opt.layers_size_disc, sn=False)

        self.mse = nn.MSELoss()

        # Check if cuda is available
        self.check_cuda(cuda)

        # WGAN_values from paper
        self.batch_size = Opt.batch_size
        self.lr_g = Opt.lr_g
        self.lr_d = Opt.lr_d
        self.b1 = Opt.beta1
        self.b2 = Opt.beta2

        # WGAN_gradient penalty uses ADAM
        self.d_optimizer = torch.optim.Adam(self.disc.parameters(), lr=self.lr_d, betas=(self.b1, self.b2))
        self.g_optimizer = torch.optim.Adam([{"params": self.gen.parameters(), "lr": self.lr_g}],
                                            betas=(self.b1, self.b2))

        self.start_g_iter = -1
        self.generator_iters = Opt.epochs
        self.critic_iter = Opt.critic_iter
        self.lambda_term = Opt.gp_lambda

    def check_cuda(self, cuda_flag=False):
        print(cuda_flag)
        if cuda_flag:
            self.cuda = True
            self.disc.cuda()
            self.gen.cuda()
            print("Cuda enabled flag: {}".format(self.cuda))
        else:
            self.cuda = False

    def train(self, train_loader):
        """ training """
        time_begin = time.time()

        self.data = self.get_batches(train_loader)
        batch_num = len(train_loader)

        one = torch.tensor(1, dtype=torch.float)
        mone = one * -1
        if self.cuda:
            one = one.cuda()
            mone = mone.cuda()

        for g_iter in range(self.start_g_iter + 1, self.generator_iters + 1):
            # Requires grad, Generator requires_grad = False
            iter_time = time.time()
            for batch_iter in range(batch_num):
                data_batch = self.data.__next__()
                # Check for batch to have full batch_size
                if data_batch.size()[0] != self.batch_size:
                    continue

                data_batch = self.get_torch_variable(data_batch)
                for p in self.disc.parameters():
                    p.requires_grad = True

                # Train discriminator forward-loss-backward-update self.critic_iter times
                # while 1 Generator forward-loss-backward-update

                for d_iter in range(self.critic_iter):
                    self.disc.zero_grad()
                    # Train discriminator
                    # WGan - Training discriminator more iterations than generator Train with real data
                    d_loss_real = self.disc(data_batch)
                    d_loss_real = d_loss_real.mean()
                    d_loss_real.backward(mone)  # -Ex~pr(D(x))

                    # Train with fake data
                    z = self.get_torch_variable(torch.randn(self.batch_size, Opt.noise_features))
                    fake_data_batch = self.generate_fake_data(z, self.coords_T)

                    d_loss_fake = self.disc(fake_data_batch)
                    d_loss_fake = d_loss_fake.mean()
                    d_loss_fake.backward(one)  # Ex~pg(D(G(x)))

                    # Train with gradient penalty
                    gradient_penalty = self.calculate_gradient_penalty(data_batch.data, fake_data_batch.data, div=False)
                    gradient_penalty.backward()
                    #
                    d_loss = d_loss_fake - d_loss_real + gradient_penalty
                    Wasserstein_d = d_loss_real - d_loss_fake
                    self.d_optimizer.step()
                    print(f'batch: {batch_iter+1}/{batch_num},  Discriminator iteration: {d_iter}/{self.critic_iter},'
                          f'd-loss: {d_loss}, loss_fake: {d_loss_fake}, loss_real: {d_loss_real}')

                # Generator update
                for p in self.disc.parameters():
                    p.requires_grad = False  # to avoid computation

                self.gen.zero_grad()

                # train generator
                # compute loss with fake images
                z = self.get_torch_variable(torch.randn(self.batch_size, Opt.noise_features))
                fake_data_batch = self.generate_fake_data(z, self.coords_T)

                g_loss = self.disc(fake_data_batch)
                g_loss = g_loss.mean()
                g_loss.backward(mone)  # -Ex~pg(D(G(x)))

                # g_cost = -g_loss
                self.g_optimizer.step()
                print(f'batch: {batch_iter+1}/{batch_num}, Generator iteration: {g_iter}/{self.generator_iters}, '
                      f'g_loss: {-g_loss}')

            res_s = (self.generator_iters - g_iter)*(time.time() - iter_time)
            res_h = int(res_s / 3600)
            res_min = int((res_s - res_h*3600) / 60)
            print(f" =======  Remaining training time：{res_h} h {res_min} min =======")
            # Saving sampling images every SAVE_PER_EPOCH generator iterations

            if g_iter % Opt.SAVE_PER_EPOCH == 0:
                self.save_images(fake_data_batch, g_iter)
                self.save_models(g_iter)
            # if g_iter > 19000 and g_iter % 100 == 0:
            #     self.save_models(g_iter)

        time_end = time.time()
        print('Time of training-{}'.format((time_end - time_begin)))

    def generate_fake_data(self, z, coords_T):
        fake_data_params, fake_data_resp_x, fake_data_resp_y = self.gen(z, coords_T)
        fake_data_params = fake_data_params.reshape(-1, Opt.params_features)
        fake_data_resp_x = fake_data_resp_x.reshape(-1, int(0.5*Opt.resp_features))
        fake_data_resp_y = fake_data_resp_y.reshape(-1, int(0.5*Opt.resp_features))
        fake_data_batch = torch.cat([fake_data_params, fake_data_resp_x, fake_data_resp_y], dim=1)

        return fake_data_batch

    def save_models(self, g_iter):
        if not os.path.exists('save_models/'):
            os.makedirs('save_models/')

        state = {
            "stat_dict_gen": self.gen.state_dict(),
            "stat_dict_disc": self.disc.state_dict(),
            "optimizer_g": self.g_optimizer.state_dict(),
            "optimizer_d": self.d_optimizer.state_dict(),
            "g_iter": g_iter
        }
        path = "save_models/state_" + str(g_iter) + ".pt"
        torch.save(state, path)

    def load_models(self, path):
        checkpoint = torch.load(path)
        self.gen.load_state_dict(checkpoint["stat_dict_gen"])
        self.disc.load_state_dict(checkpoint["stat_dict_disc"])
        self.g_optimizer.load_state_dict(checkpoint["optimizer_g"])
        self.d_optimizer.load_state_dict(checkpoint["optimizer_d"])
        self.start_g_iter = checkpoint["g_iter"]


class WGanGP4(WGanGP):
    """Compare with plain FNN. without operator learning"""
    def __init__(self, coords_T, resp_features, cuda=True):
        self.coords_T = coords_T
        self.resp_features = resp_features

        self.gen_params = Fnn([Opt.noise_features, 32, 32, 32, 32, 625], act_func=nn.Tanh())  # 53559参数
        self.gen_resp_x = Fnn([Opt.noise_features, 32, 32, 32, 32, 625], act_func=nn.Tanh())
        self.gen_resp_y = Fnn([Opt.noise_features, 32, 32, 32, 32, 625], act_func=nn.Tanh())
        self.disc = Fnn(Opt.layers_size_disc, sn=False)

        self.mse = nn.MSELoss()

        # Check if cuda is available
        self.check_cuda(cuda)

        # WGAN_values from paper
        self.batch_size = Opt.batch_size
        self.lr_g = Opt.lr_g
        self.lr_d = Opt.lr_d
        self.b1 = Opt.beta1
        self.b2 = Opt.beta2

        # WGAN_gradient penalty uses ADAM
        self.d_optimizer = torch.optim.Adam(self.disc.parameters(), lr=self.lr_d, betas=(self.b1, self.b2))
        self.g_optimizer = torch.optim.Adam([{"params": self.gen_resp_x.parameters(), "lr": self.lr_g},
                                             {"params": self.gen_resp_y.parameters(), "lr": self.lr_g},
                                             {"params": self.gen_params.parameters(), "lr": self.lr_g}],
                                            betas=(self.b1, self.b2))

        self.start_g_iter = -1
        self.generator_iters = Opt.epochs
        self.critic_iter = Opt.critic_iter
        self.lambda_term = Opt.gp_lambda

    def generate_fake_data(self, z, coords):
        fake_data_params = self.gen_params(z)
        fake_data_resp_x = self.gen_resp_x(z)
        fake_data_resp_y = self.gen_resp_y(z)

        fake_data_batch = torch.cat([fake_data_params, fake_data_resp_x, fake_data_resp_y], dim=1)

        return fake_data_batch
