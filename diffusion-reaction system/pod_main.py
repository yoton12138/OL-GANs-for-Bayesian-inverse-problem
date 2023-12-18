# DeepONet GANS prior for inverse problem
# resolution-independent
# CASE 3: diffusion reaction system
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import torch.nn

from utils import *
from networks import *
import tqdm
from pyDOE import lhs

matplotlib.use("Agg")
plt.rcParams['font.family'] = 'Times New Roman'

seed_torch(616)
# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    # === make data set (loader) ===
    data = np.load("data/meta_data_case3.npz")
    resp = data["resp"].reshape(-1, 10000)
    params = data["params"]
    x_mesh, t_mesh, coords = data["x_mesh"], data["t_mesh"], data["coords"]
    sigma = 0.2
    obs_resp = data["observation_02_grid1089"]  # choose one sparse observation
    ref_params = data["ref_params"]
    ref_resp = data["ref_resp"]

    # sparse resp sensors
    nx = 33
    sensor_lin = np.linspace(0.0, 1.0, nx)
    x_mesh_sensor, t_mesh_sensor, coords_sensor, resp_sensor = observation_maker(sensor_lin, resp.reshape(-1, 10000))
    _, _, _, ref_resp_sensor = observation_maker(sensor_lin, ref_resp.reshape(1, -1))

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=[8, 3])
    x_lin = np.linspace(0, 1, 100)
    ax0 = axes[0].plot(x_lin, ref_params)
    ax1 = axes[1].pcolor(x_mesh_sensor, t_mesh_sensor, ref_resp_sensor.reshape(nx, nx))
    plt.colorbar(ax1, ax=axes[1])
    plt.savefig("training_result_images/ref_params_resp.png", dpi=300)

    joint_snapshots = np.hstack((params, resp_sensor))
    Opt.resp_features = resp_sensor.shape[1]
    Opt.joint_features = joint_snapshots.shape[1]
    Opt.x_mesh_sensor, Opt.t_mesh_sensor = x_mesh_sensor, t_mesh_sensor

    # train test split
    train_snapshots, test_snapshots = joint_snapshots[:Opt.train_num, :], joint_snapshots[Opt.train_num:, :]

    # numpy to tensor
    coords_params_T = torch.tensor(np.linspace(0, 1, 100).reshape(-1, 1), dtype=torch.float32).to(Opt.device)
    coords_sensor_T = torch.tensor(coords_sensor, dtype=torch.float32).to(Opt.device)
    train_snapshots_T = torch.tensor(train_snapshots, dtype=torch.float32).to(Opt.device)
    test_snapshots_T = torch.tensor(test_snapshots, dtype=torch.float32).to(Opt.device)
    sigma_T = torch.tensor(sigma, dtype=torch.float32).to(Opt.device)
    obs_resp_T = torch.tensor(obs_resp.copy(), dtype=torch.float32).to(Opt.device)

    train_data = Dataset(train_snapshots_T)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=Opt.batch_size, drop_last=True,
                                               shuffle=False, num_workers=0)

    # === pca decomposition ===
    params_means_nn = Fnn([1, 20, 20, 20, 1], act_func=nn.Tanh()).to(Opt.device)
    params_components_nn = Fnn([1, 64, 64, 64, 10], act_func=nn.Tanh()).to(Opt.device)
    resp_means_nn = Fnn([2, 20, 20, 20, 1], act_func=nn.Tanh()).to(Opt.device)
    resp_components_nn = Fnn([2, 64, 64, 64, 10], act_func=nn.Tanh()).to(Opt.device)

    if not os.path.exists('save_models/pod_main_state/state_pod.pt'):
        params_mean, params_components = get_basic(params)
        resp_mean, resp_components = get_basic(resp_sensor)

        params_components = params_components.T[:, :10]
        resp_components = resp_components.T[:, :10]

        params_mean_T = torch.tensor(params_mean, dtype=torch.float32).to(Opt.device)
        resp_mean_T = torch.tensor(resp_mean, dtype=torch.float32).to(Opt.device)
        params_components_T = torch.tensor(params_components, dtype=torch.float32).to(Opt.device)
        resp_components_T = torch.tensor(resp_components, dtype=torch.float32).to(Opt.device)
        adam = torch.optim.Adam([{"params": params_means_nn.parameters(), "lr": 0.001},
                                 {"params": params_components_nn.parameters(), "lr": 0.001},
                                 {"params": resp_means_nn.parameters(), "lr": 0.001},
                                 {"params": resp_components_nn.parameters(), "lr": 0.001}
                                 ], betas=(0.9, 0.999))
        mse_func = nn.MSELoss()
        for i in range(int(50000) + 1):
            adam.zero_grad()
            params_mean_pred = params_means_nn(coords_params_T).squeeze()
            resp_mean_pred = resp_means_nn(coords_sensor_T).squeeze()
            params_components_pred = params_components_nn(coords_params_T)
            resp_components_pred = resp_components_nn(coords_sensor_T)
            loss_m_p = mse_func(params_mean_pred, params_mean_T)
            loss_m_r = mse_func(resp_mean_pred, resp_mean_T)
            loss_c_p = mse_func(params_components_pred, params_components_T)
            loss_c_r = mse_func(resp_components_pred, resp_components_T)
            loss = loss_m_p + loss_m_r + loss_c_p + loss_c_r
            loss.backward()
            adam.step()
            if i % 1000 == 0:
                print(f"epoch: {i}, mse_loss--- \nmean ==> params:{loss_m_p.data.cpu().numpy()}, resp:{loss_m_p.data.cpu().numpy()} \n"
                      f"components ==> params:{loss_c_p.data.cpu().numpy()}, resp:{loss_c_r.data.cpu().numpy()}")

        torch.save({"stat_dict_means_p": params_means_nn.state_dict(),
                    "stat_dict_components_p": params_components_nn.state_dict(),
                    "stat_dict_means_r": resp_means_nn.state_dict(),
                    "stat_dict_components_r": resp_components_nn.state_dict(),
                    },
                   "save_models/pod_main_state/state_pod.pt")
        for p in params_means_nn.parameters():
            p.requires_grad = False
        for p in params_components_nn.parameters():
            p.requires_grad = False
        for p in resp_means_nn.parameters():
            p.requires_grad = False
        for p in resp_components_nn.parameters():
            p.requires_grad = False
        # compare prediction and reference modes
        plot_modes(x_mesh_sensor, t_mesh_sensor, resp_components, "reference", nx)
        plot_modes(x_mesh_sensor, t_mesh_sensor, resp_components_pred.data.cpu().numpy(), "prediction", nx)

    else:
        print("Load POD files")
        checkpoint = torch.load("save_models/pod_main_state/state_pod.pt")
        params_means_nn.load_state_dict(checkpoint["stat_dict_means_p"])
        resp_means_nn.load_state_dict(checkpoint["stat_dict_means_r"])
        params_components_nn.load_state_dict(checkpoint["stat_dict_components_p"])
        resp_components_nn.load_state_dict(checkpoint["stat_dict_components_r"])
        for p in params_means_nn.parameters():
            p.requires_grad = False
        for p in params_components_nn.parameters():
            p.requires_grad = False
        for p in resp_means_nn.parameters():
            p.requires_grad = False
        for p in resp_components_nn.parameters():
            p.requires_grad = False

    # === model setting ===
    model = WGanGP3(coords_params_T, coords_sensor_T, Opt.resp_features, params_means_nn, params_components_nn, resp_means_nn, resp_components_nn)
    g_iter = 20000
    model.load_models(f"save_models/pod_main_state/state_{g_iter}.pt")
    # model.train(train_loader)

    z = torch.randn(2000, Opt.noise_features).to(Opt.device)
    fake_snapshots = model.generate_fake_data(z, coords_params_T, coords_sensor_T).data.cpu().numpy()
    fake_params = fake_snapshots[:, :Opt.params_features]
    fake_resp = fake_snapshots[:, Opt.params_features:]

    ev_real_p = get_ev(params[:Opt.train_num, :])
    ev_real_r = get_ev(resp[:Opt.train_num, :])

    ev_fake_p = get_ev(fake_params[:Opt.train_num, :])
    ev_fake_r = get_ev(fake_resp[:, :Opt.resp_features])
    #
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=[12, 3])
    plot_ev(axes[0], ev_real_p, ev_fake_p, "params")
    plot_ev(axes[1], ev_real_r, ev_fake_r, "resp")
    plt.savefig(f"training_result_images/explained_variance_ratio_{g_iter}.png", dpi=300)

    fig = plt.figure(figsize=[4, 3])
    sns.distplot(params.reshape(-1, 1), norm_hist=True, bins=100, kde=True, label="truth")
    sns.distplot(fake_params.reshape(-1, 1), norm_hist=True, bins=100, kde=True, label="generator")
    plt.legend()
    plt.savefig(f"training_result_images/params_density_sns_{g_iter}.png", dpi=600)

    # === sampling procedure ===
    def logp(z):
        """ define log prob density """
        z_T = torch.tensor(z, dtype=torch.float32, requires_grad=True).to(Opt.device)
        fake_snapshot = model.generate_fake_data(z_T.view(1, -1), coords_params_T, coords_sensor_T).data.cpu().numpy()
        fake_resp = fake_snapshot[:, Opt.params_features:]

        log_llh = np.sum(-(fake_resp - obs_resp) ** 2 / (2 * sigma ** 2)) - obs_resp.shape[0] * np.log(sigma)
        log_piror = np.sum(-(z - 0) ** 2 / (2 * 1.0 ** 2))

        return log_llh + log_piror


    print("find MAP position .........")

    # gradient MAP
    def map_loss(z):
        fake_snapshot = model.generate_fake_data(z.view(1, -1), coords_params_T, coords_sensor_T)
        fake_resp = torch.squeeze(fake_snapshot[:, Opt.params_features:])
        llh_1 = torch.sum(-(fake_resp - obs_resp_T) ** 2 / (2 * sigma_T ** 2))
        llh_2 = - obs_resp_T.shape[0] * torch.log(sigma_T)
        log_llh = llh_1 + llh_2
        log_piror = torch.sum(-(z - 0) ** 2 / (2 * 1.0 ** 2))

        return log_llh + log_piror


    z_T = torch.nn.Parameter(torch.zeros(1, Opt.noise_features).to(Opt.device))
    lr = 0.05
    opt_iter = 1000
    iter_array = np.zeros((opt_iter, Opt.noise_features + 1))
    map_optimizer = torch.optim.Adam([z_T], lr=lr, betas=(0.9, 0.999))

    for i in tqdm.tqdm(range(opt_iter)):
        map_optimizer.zero_grad()
        loss = - map_loss(z_T)
        loss.backward()
        map_optimizer.step()
        iter_array[i, :Opt.noise_features] = z_T.data.cpu().numpy()
        iter_array[i, -1] = loss.data.cpu().numpy()

    print(f"-MAP LOSS: {loss}")

    fake_map = model.generate_fake_data(z_T, coords_params_T, coords_sensor_T).data.cpu().numpy()
    z_map = z_T.data.cpu().numpy()

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=[8, 3])
    x_lin = np.linspace(0, 1, 100)
    ax0 = axes[0].plot(x_lin, fake_map[0, :Opt.params_features])
    ax1 = axes[1].pcolor(x_mesh_sensor, t_mesh_sensor, fake_map[0, Opt.params_features:].reshape(nx, nx))
    plt.colorbar(ax1, ax=axes[1])
    plt.savefig("training_result_images/map.png", dpi=300)

    #  Metropolis Hasting Sampling
    init_position = z_map
    burn = 5000
    sampler = MHSampler(init_position, logp, Opt.noise_features)
    position_array = sampler.sampling(10000)
    ll_array = sampler.ll_array
    #
    fig = plt.figure(figsize=[4, 3])
    plt.plot(np.asarray(ll_array))
    plt.tight_layout()
    plt.savefig("training_result_images/ll.png", dpi=300)
    #
    position_array = np.asarray(position_array)
    position_array_T = torch.tensor(position_array, dtype=torch.float32).to(Opt.device)

    posterior_joint = model.generate_fake_data(position_array_T, coords_params_T, coords_sensor_T).data.cpu().numpy()
    posterior_joint_burned = posterior_joint[burn:, :]
    posterior_params = posterior_joint_burned[:, :Opt.params_features]
    posterior_resp = posterior_joint_burned[:, Opt.params_features:]
    posterior_params_mean = np.mean(posterior_params, axis=0)
    posterior_params_mean_square = posterior_params_mean ** 2
    posterior_params_square_mean = np.mean(posterior_params ** 2, axis=0)
    posterior_params_std = np.std(posterior_params, axis=0)
    posterior_resp_mean = np.mean(posterior_resp, axis=0)
    in3std = get_in3std(posterior_params_mean, posterior_params_std, ref_params)
    error_E = posterior_params_mean - ref_params
    upper = posterior_params_mean + 3 * posterior_params_std
    lower = posterior_params_mean - 3 * posterior_params_std


    def plot_axes(axes, xx, yy, zz, str):
        ax0 = axes.pcolor(xx, yy, zz.reshape(xx.shape[0], xx.shape[0]))
        plt.colorbar(ax0, ax=axes)
        axes.set_title(str)


    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=[12, 3])
    axes[0].plot(x_lin, ref_params, linestyle="-", label="Reference")
    axes[0].plot(x_lin, posterior_params_mean, linestyle="--", label="Posterior_mean")
    axes[0].fill_between(x_lin, lower.flatten(), upper.flatten(), alpha=0.5, rasterized=True,
                         label="Epistemic uncertainty")
    axes[0].legend()
    plot_axes(axes[1], x_mesh_sensor, t_mesh_sensor, posterior_resp_mean.reshape(nx, nx), "Posterior_mean")
    plot_axes(axes[2], x_mesh_sensor, t_mesh_sensor, obs_resp.reshape(nx, nx), "Observation")
    plt.savefig(f"training_result_images/posterior{g_iter}.png", dpi=300)
    #

    error_l2_params = np.linalg.norm(posterior_params_mean - ref_params, 2) / np.linalg.norm(ref_params, 2)
    r2 = r2_score(ref_params, posterior_params_mean)
    error_l2_resp = np.linalg.norm(posterior_resp_mean - ref_resp_sensor.squeeze(), 2) / np.linalg.norm(
        ref_resp_sensor.squeeze(), 2)
    in3std_ration = in3std[in3std > 0].shape[0] / in3std.shape[0]
    print(f"Relative l2 error of parameters： {error_l2_params}, r2: {r2}, in 3-sigma ratio %: {round(in3std_ration,2)*100} %")
    print(f"Relative l2 error of response  ： {error_l2_resp}")

    # save data
    # file_name = "training_result_images/postprocess_data.npz"
    # np.savez(file_name,
    #          fake_params=fake_params, fake_resp=fake_resp,
    #          fake_map=fake_map, z_map=z_map,
    #          sigma=sigma, burn=burn,
    #          position_array=position_array, ll_array=ll_array,
    #          posterior_joint=posterior_joint)

