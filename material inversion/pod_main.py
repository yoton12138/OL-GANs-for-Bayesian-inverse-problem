# DeepONet GANS prior for inverse problem
# resolution-independent
# pod based trunk net
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
    data = np.load("data/meta_data_case2.npz")
    disp_x = data["disp_x"]
    disp_y = data["disp_y"]
    params = data["E"]
    x_mesh_sensor, y_mesh_sensor, coords_sensor = data["x_mesh"], data["y_mesh"], data["coords"]
    sigma = 0.01
    obs_disp = data["observation_001"]
    ref_params = data["ref_params"]
    ref_disp = data["ref_disp"]

    joint_snapshots = np.hstack((params, disp_x, disp_y))
    Opt.joint_features = joint_snapshots.shape[1]
    Opt.x_mesh_sensor, Opt.y_mesh_sensor = x_mesh_sensor, y_mesh_sensor

    # train test split
    train_snapshots, test_snapshots = joint_snapshots[:Opt.train_num, :], joint_snapshots[Opt.train_num:, :]

    # numpy to tensor
    coords_sensor_T = torch.tensor(coords_sensor, dtype=torch.float32, requires_grad=True).to(Opt.device)
    train_snapshots_T = torch.tensor(train_snapshots, dtype=torch.float32, requires_grad=True).to(Opt.device)
    test_snapshots_T = torch.tensor(test_snapshots, dtype=torch.float32, requires_grad=True).to(Opt.device)
    sigma_T = torch.tensor(sigma, dtype=torch.float32).to(Opt.device)
    obs_disp_T = torch.tensor(obs_disp.copy(), dtype=torch.float32).to(Opt.device)

    train_data = Dataset(train_snapshots_T)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=Opt.batch_size, drop_last=True,
                                               shuffle=False, num_workers=0)

    # === pca decomposition ===
    means_nn = Fnn([2, 20, 20, 20, 3], act_func=nn.Tanh()).to(Opt.device)
    components_nn = Fnn([2, 64, 64, 64, 30], act_func=nn.Tanh()).to(Opt.device)

    if not os.path.exists('save_models/state_pod.pt'):
        disp_e_mean, disp_e_components = get_basic(params)
        disp_x_mean, disp_x_components = get_basic(disp_x)
        disp_y_mean, disp_y_components = get_basic(disp_y)
        resp_mean = np.vstack([disp_e_mean, disp_x_mean, disp_y_mean]).T
        resp_components = np.hstack([disp_e_components.T[:, :10], disp_x_components.T[:, :10], disp_y_components.T[:, :10]])
        resp_mean_T = torch.tensor(resp_mean, dtype=torch.float32, requires_grad=True).to(Opt.device)
        resp_components_T = torch.tensor(resp_components, dtype=torch.float32, requires_grad=True).to(Opt.device)
        adam = torch.optim.Adam([{"params": means_nn.parameters(), "lr": 0.001},
                                 {"params": components_nn.parameters(), "lr": 0.001}], betas=(0.9, 0.999))
        mse_func = nn.MSELoss()
        for i in range(int(20000)):
            adam.zero_grad()
            mean_pred = means_nn(coords_sensor_T)
            components_pred = components_nn(coords_sensor_T)
            loss_m = mse_func(mean_pred, resp_mean_T)
            loss_c = mse_func(components_pred, resp_components_T)
            loss_m.backward()
            loss_c.backward()
            adam.step()
            if i % 1000 == 0:
                print(f"epoch: {i}, mse_loss--- mean:{loss_m.data.cpu().numpy()}, components:{loss_c.data.cpu().numpy()}")

        torch.save({"stat_dict_means": means_nn.state_dict(), "stat_dict_components": components_nn.state_dict()},
                   "save_models/state_pod.pt")
        for p in means_nn.parameters():
            p.requires_grad = False
        for p in components_nn.parameters():
            p.requires_grad = False
        # compare prediction and reference modes
        plot_modes(x_mesh_sensor, y_mesh_sensor, resp_components, "reference")
        plot_modes(x_mesh_sensor, y_mesh_sensor, components_pred.data.cpu().numpy(), "prediction")

    else:
        print("Load POD files")
        checkpoint = torch.load("save_models/state_pod.pt")
        means_nn.load_state_dict(checkpoint["stat_dict_means"])
        components_nn.load_state_dict(checkpoint["stat_dict_components"])
        for p in means_nn.parameters():
            p.requires_grad = False
        for p in components_nn.parameters():
            p.requires_grad = False

    # === model setting ===
    model = WGanGP3(coords_sensor_T, Opt.resp_features, means_nn, components_nn)
    g_iter = 20000
    model.load_models(f"save_models/pod_main_state/state_{g_iter}.pt")
    # model.train(train_loader)

    z = torch.randn(2000, Opt.noise_features).to(Opt.device)
    fake_snapshots = model.generate_fake_data(z, coords_sensor_T).data.cpu().numpy()
    fake_params = fake_snapshots[:, :Opt.params_features]
    fake_resp = fake_snapshots[:, Opt.params_features:]

    ev_real_e = get_ev(params[:Opt.train_num, :])
    ev_real_x = get_ev(disp_x[:Opt.train_num, :])
    ev_real_y = get_ev(disp_y[:Opt.train_num, :])

    ev_fake_e = get_ev(fake_params[:Opt.train_num, :])
    ev_fake_x = get_ev(fake_resp[:, :int(0.5*Opt.resp_features)])
    ev_fake_y = get_ev(fake_resp[:, int(0.5*Opt.resp_features):])

    sum_real_e = np.cumsum(ev_real_e)
    sum_real_x = np.cumsum(ev_real_x)
    sum_real_y = np.cumsum(ev_real_y)
    #
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=[12, 3])
    plot_ev(axes[0], ev_real_e, ev_fake_e, "E")
    plot_ev(axes[1], ev_real_x, ev_fake_x, "disp_x")
    plot_ev(axes[2], ev_real_y, ev_fake_y, "disp_y")
    plt.savefig(f"training_result_images/explained_variance_ratio_{g_iter}.png", dpi=300)

    fig = plt.figure(figsize=[4, 3])
    plt.xlim([0.8, 2])
    sns.distplot(params.reshape(-1, 1), norm_hist=True, bins=100, kde=True, label="truth")
    sns.distplot(fake_params.reshape(-1, 1), norm_hist=True, bins=100, kde=True, label="generator")
    plt.legend()
    plt.savefig(f"training_result_images/E_density_sns_{g_iter}.png", dpi=600)

    # === sampling procedure ===
    # define the log probability function
    def logp(z):
        """ 定义高斯对数概率函数 """
        z_T = torch.tensor(z, dtype=torch.float32, requires_grad=True).to(Opt.device)
        fake_snapshot = model.generate_fake_data(z_T.view(1, -1), coords_sensor_T)
        fake_resp = fake_snapshot[:, Opt.params_features:].data.cpu().numpy()

        log_llh = np.sum(-(fake_resp - obs_disp)**2 / (2*sigma**2)) - obs_disp.shape[0]*np.log(sigma)
        log_piror = np.sum(-(z - 0)**2/(2*1.0**2))

        return log_llh + log_piror

    print("find MAP position .........")

    # gradient MAP
    def map_loss(z):
        fake_snapshot = model.generate_fake_data(z.view(1, -1), coords_sensor_T)
        fake_resp = torch.squeeze(fake_snapshot[:, Opt.params_features:])

        llh_1 = torch.sum(-(fake_resp - obs_disp_T)**2 / (2*sigma_T**2))
        llh_2 = - obs_disp_T.shape[0]*torch.log(sigma_T)
        log_llh = llh_1 + llh_2
        log_piror = torch.sum(-(z - 0)**2/(2*1.0**2))

        return log_llh + log_piror

    z_T = torch.nn.Parameter(torch.randn(1, Opt.noise_features).to(Opt.device))
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

    fake_map = model.generate_fake_data(z_T, coords_sensor_T).data.cpu().numpy()
    z_map = z_T.data.cpu().numpy()

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=[12, 3])
    ax0 = axes[0].contourf(x_mesh_sensor, y_mesh_sensor, fake_map[0, :Opt.params_features].reshape(25, 25))
    plt.colorbar(ax0, ax=axes[0])
    ax1 = axes[1].contourf(x_mesh_sensor, y_mesh_sensor, fake_map[0, Opt.params_features:-625].reshape(25, 25))
    plt.colorbar(ax1, ax=axes[1])
    ax2 = axes[2].contourf(x_mesh_sensor, y_mesh_sensor, fake_map[0, -625:].reshape(25, 25))
    plt.colorbar(ax2, ax=axes[2])
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
    position_array_T = torch.tensor(position_array, dtype=torch.float32, requires_grad=True).to(Opt.device)

    posterior_joint = model.generate_fake_data(position_array_T, coords_sensor_T).data.cpu().numpy()
    posterior_joint_burned = posterior_joint[burn:, :]
    posterior_params = posterior_joint_burned[:, :Opt.params_features]
    posterior_resp = posterior_joint_burned[:, Opt.params_features:]
    posterior_params_mean = np.mean(posterior_params, axis=0)
    posterior_params_mean_square = posterior_params_mean**2
    posterior_params_square_mean = np.mean(posterior_params**2, axis=0)
    posterior_params_std = np.std(posterior_params, axis=0)
    posterior_resp_mean = np.mean(posterior_resp, axis=0)
    in3std = get_in3std(posterior_params_mean, posterior_params_std, ref_params)
    error_E = posterior_params_mean - ref_params

    def plot_axes(axes, xx, yy, zz, str):
        ax0 = axes.pcolor(xx, yy, zz.reshape(xx.shape[0], xx.shape[0]))
        plt.colorbar(ax0, ax=axes)
        axes.set_title(str)

    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=[16, 6])
    plot_axes(axes[0][0], x_mesh_sensor, y_mesh_sensor, posterior_params_mean, "posterior_mean_E")
    plot_axes(axes[0][1], x_mesh_sensor, y_mesh_sensor, ref_params, "reference_E")
    plot_axes(axes[0][2], x_mesh_sensor, y_mesh_sensor, error_E, "error_E")
    plot_axes(axes[0][3], x_mesh_sensor, y_mesh_sensor, posterior_params_std, "std_E")
    plot_axes(axes[1][0], x_mesh_sensor, y_mesh_sensor, posterior_resp_mean[:625], "posterior_mean_disp_x")
    plot_axes(axes[1][1], x_mesh_sensor, y_mesh_sensor, obs_disp[:625], "observation_x")
    plot_axes(axes[1][2], x_mesh_sensor, y_mesh_sensor, posterior_resp_mean[625:], "posterior_mean_disp_y")
    plot_axes(axes[1][3], x_mesh_sensor, y_mesh_sensor, obs_disp[625:], "observation_y")
    plt.savefig(f"training_result_images/posterior{g_iter}.png", dpi=300)
    #
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=[4, 3])
    ax = axes.pcolor(x_mesh_sensor, y_mesh_sensor, in3std.reshape(25, 25), cmap="coolwarm", vmax=1.0, vmin=0.0)
    plt.tight_layout()
    plt.savefig(f"training_result_images/in3std_{g_iter}.png", dpi=300)

    error_l2_e = np.linalg.norm(posterior_params_mean - ref_params, 2) / np.linalg.norm(ref_params, 2)
    error_l2_x = np.linalg.norm(posterior_resp_mean[:625] - ref_disp[:625], 2) / np.linalg.norm(ref_disp[:625], 2)
    error_l2_y = np.linalg.norm(posterior_resp_mean[625:] - ref_disp[625:], 2) / np.linalg.norm(ref_disp[625:], 2)
    in3std_ration = in3std[in3std > 0].shape[0] / in3std.shape[0]
    print(f"Relative l2 error of elastic modulus field: {error_l2_e}, in 3-sigma ratio %: {round(in3std_ration,2)*100} %")
    print(f"Relative l2 error of x-displacement field : {error_l2_x}")
    print(f"Relative l2 error of y-displacement field : {error_l2_y}")

    # save data
    # file_name = "training_result_images/postprocess_data.npz"
    # np.savez(file_name,
    #          fake_params=fake_params, fake_resp=fake_resp,
    #          fake_map=fake_map, z_map=z_map,
    #          sigma=sigma, burn=burn,
    #          position_array=position_array, ll_array=ll_array,
    #          posterior_joint=posterior_joint)


