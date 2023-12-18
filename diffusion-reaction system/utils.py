import numpy as np
from options import Opt
from scipy.interpolate import interp2d, Rbf
import torch
from sklearn.decomposition import PCA
import tqdm
import matplotlib.pyplot as plt
import os


def seed_torch(seed):
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def observation_maker(sensor_lin, resp):
    x_lin = np.linspace(0, 1, 100)

    x_mesh_sensor, y_mesh_sensor = np.meshgrid(sensor_lin, sensor_lin)
    coords_sensor = np.vstack((x_mesh_sensor.flatten(), y_mesh_sensor.flatten())).T
    resp_sensor = np.zeros([resp.shape[0], coords_sensor.shape[0]])
    for i in range(resp.shape[0]):
        func = interp2d(x_lin, x_lin, resp[i, :])
        resp_sensor[i, :] = func(sensor_lin, sensor_lin).T.flatten()  # 这里有一个转置需要注意

    return x_mesh_sensor, y_mesh_sensor, coords_sensor, resp_sensor


def new_observation_maker(new_coords, coords, resp):
    """ rbf interpolate"""
    coords_sensor = new_coords
    resp_sensor = np.zeros([resp.shape[0], coords_sensor.shape[0]])
    for i in range(resp.shape[0]):
        func = Rbf(coords[:, 0], coords[:, 1], resp[i, :].flatten())
        resp_sensor[i, :] = func(coords_sensor[:, 0], coords_sensor[:, 1]).T.flatten()

    return resp_sensor


def get_ev(x):
    pca = PCA()
    pca.fit(x)
    return pca.explained_variance_ratio_


def get_basic(x):
    pca = PCA()
    pca.fit(x)
    return pca.mean_, pca.components_


def plot_ev(ax, ev_real, ev_fake, str, p=10):
    ax.plot(ev_real[:p], label="reference", marker="o")
    ax.plot(ev_fake[:p], label="generated", marker="o")
    ax.set_title(str)
    ax.legend()


def plot_modes(xx, yy, components, title, res):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 3))

    def plot_contourf(ax, xx, yy, zz, str, levels=11):
        ax0 = ax.contourf(xx, yy, zz, levels=levels)
        ax.set_title(str)
        plt.colorbar(ax0, ax=ax)
    
    plot_contourf(axes[0], xx, yy, components[:, 0].reshape(res, res), "first resp mode")
    plot_contourf(axes[1], xx, yy, components[:, 1].reshape(res, res), "second resp mode")
    plot_contourf(axes[2], xx, yy, components[:, 2].reshape(res, res), "third resp mode")
    plt.title(title)
    plt.savefig(f"training_result_images/{title}.png", dpi=300)


def get_in3std(post_params, post_std, ref_params):
    lb = post_params - 3*post_std
    ub = post_params + 3*post_std
    in3std = np.zeros_like(lb)
    for i, (ref_params_i, lb_i, ub_i) in enumerate(zip(ref_params, lb, ub)):
        in3std[i] = 1 if lb_i < ref_params_i < ub_i else 0

    return in3std


def posterior_extractor(data_post, mode=0):
    """extract useful information"""
    burn = data_post["burn"]
    if not mode:
        posterior_joint = data_post["posterior_joint"]
    else:
        posterior_joint = data_post["posterior_joint_grid"]
    posterior_joint_burned = posterior_joint[burn:, :]
    posterior_params = posterior_joint_burned[:, :Opt.params_features]
    posterior_resp = posterior_joint_burned[:, Opt.params_features:]
    posterior_params_mean = np.mean(posterior_params, axis=0)
    posterior_params_std = np.std(posterior_params, axis=0)
    posterior_resp_mean = np.mean(posterior_resp, axis=0)
    upper = posterior_params_mean + 3 * posterior_params_std
    lower = posterior_params_mean - 3 * posterior_params_std

    return posterior_params_mean, posterior_resp_mean, upper, lower


def tune(scale, acceptance):
    """ Borrowed from PyMC3 """
    # Switch statement
    if acceptance < 0.001:
        # reduce by 90 percent
        scale *= 0.1
    elif acceptance < 0.05:
        # reduce by 50 percent
        scale *= 0.5
    elif acceptance < 0.2:
        # reduce by ten percent
        scale *= 0.9
    elif acceptance > 0.95:
        # increase by factor of ten
        scale *= 10.0
    elif acceptance > 0.75:
        # increase by double
        scale *= 2.0
    elif acceptance > 0.5:
        # increase by ten percent
        scale *= 1.1
    # print("############### tuning ################")
    # print("scale:", scale, "acceptance:", acceptance)

    return scale


class MHSampler(object):
    def __init__(self, init_position, logp, noise_features, step_tune=100, scale=1.0):
        self.init_position = init_position
        self.logp = logp
        self.noise_features = noise_features

        self.step_tune = step_tune
        self.scale = scale
        self.accepted = 0
        self.position_array = []
        self.ll_array = []
        self.position_array.append(np.squeeze(init_position.reshape(1, -1)))
        self.ll_array.append(logp(init_position.reshape(1, -1)))

    def sampling(self, samples_num=10000, burn=None):
        print("Start sampling ......")
        current_position = self.position_array[0].copy()
        logp_current = self.ll_array[0].copy()

        for i in tqdm.tqdm(range(samples_num - 1)):
            delta_p = np.random.randn(self.noise_features).reshape(1, -1) * self.scale
            next_position = current_position + delta_p

            logp_next = self.logp(next_position)

            alpha = np.min([logp_next - logp_current, 0])
            u = np.log(np.random.uniform())

            if alpha > u:
                self.accepted += 1
                current_position = next_position
                logp_current = logp_next
                self.position_array.append(np.squeeze(next_position))

            else:
                self.position_array.append(np.squeeze(current_position))

            self.ll_array.append(logp_current)
            self.acceptance = self.accepted / (i + 1)
            self.step_tune -= 1
            if self.step_tune == 0:
                self.scale = tune(self.scale, self.acceptance)
                self.step_tune = 100

        print(f"sampling completed! acceptance: {self.acceptance}")
        self.ll_array = np.asarray(self.ll_array)
        self.position_array = np.asarray(self.position_array)

        if burn:
            return self.position_array[burn:]
        else:
            return self.position_array
