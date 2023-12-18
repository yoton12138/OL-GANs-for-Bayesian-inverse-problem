# DeepONet GANS prior for inverse problem
# resolution-independent, using the same network with main.py
# new sensor inversion, 400 random sensors
import numpy as np
import seaborn as sns
import pandas as pd
from utils import *
from networks import *
import tqdm
from pyDOE import lhs

matplotlib.use("Agg")
plt.rcParams['font.family'] = 'Times New Roman'

seed_torch(616)
# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    data = np.load("data/meta_data_case2.npz")
    ref_params = data["ref_params"]
    ref_disp = data["ref_disp"]
    x_mesh_sensor, y_mesh_sensor, coords_sensor = data["x_mesh"], data["y_mesh"], data["coords"]

    sigma = 0.01  # ensure the sigma consistent with the obs_disp
    n_points = 400  # ensure the n_points consistent with the obs_disp

    # note 1: use the existed observation
    obs_disp = data["observation_r400_001"]  # choosing observation_r400_001, observation_r50_001
    coords_sensor_new = data["coords_r400"]  # consistent with the observation dim
    ref_params_new = new_observation_maker(coords_sensor_new, coords_sensor, ref_params.reshape(1, -1)).squeeze()
    disp_x_new = new_observation_maker(coords_sensor_new, coords_sensor, ref_disp[:625].reshape(1, -1))
    disp_y_new = new_observation_maker(coords_sensor_new, coords_sensor, ref_disp[625:].reshape(1, -1))
    ref_disp_new = np.hstack((disp_x_new, disp_y_new)).squeeze()

    # note 2: use new sensor coords and observation, replace the coords_sensor_new and obs_disp
    # ub = np.array([1.0, 1.0])
    # lb = np.array([0.0, 0.0])
    # coords_sensor_new = lb + (ub - lb) * lhs(2, samples=n_points)
    # ref_params_new = new_observation_maker(coords_sensor_new, coords_sensor, ref_params.reshape(1, -1)).squeeze()
    # disp_x_new = new_observation_maker(coords_sensor_new, coords_sensor, ref_disp[:625].reshape(1, -1))
    # disp_y_new = new_observation_maker(coords_sensor_new, coords_sensor, ref_disp[625:].reshape(1, -1))
    # ref_disp_new = np.hstack((disp_x_new, disp_y_new)).squeeze()
    # obs_disp = ref_disp_new + sigma*np.random.randn(2*n_points)

    # update the number of features
    Opt.resp_features = n_points*2
    Opt.params_features = n_points

    # numpy to tensor
    coords_sensor_T = torch.tensor(coords_sensor, dtype=torch.float32, requires_grad=True).to(Opt.device)
    coords_sensor_new_T = torch.tensor(coords_sensor_new, dtype=torch.float32, requires_grad=True).to(Opt.device)
    sigma_T = torch.tensor(sigma, dtype=torch.float32).to(Opt.device)
    obs_disp_T = torch.tensor(obs_disp, dtype=torch.float32).to(Opt.device)

    # === model setting ===
    model = WGanGP(coords_sensor_new_T, Opt.resp_features)
    g_iter = 20000
    model.load_models(f"save_models/main_state/state_{g_iter}.pt")

    # === sampling procedure ===
    def logp(z):
        """ define log prob density """
        z_T = torch.tensor(z, dtype=torch.float32, requires_grad=True).to(Opt.device)
        fake_snapshot = model.generate_fake_data(z_T.view(1, -1), coords_sensor_new_T).data.cpu().numpy()
        fake_resp = fake_snapshot[:, Opt.params_features:]

        log_llh = np.sum(-(fake_resp - obs_disp) ** 2 / (2 * sigma ** 2)) - obs_disp.shape[0] * np.log(sigma)
        log_piror = np.sum(-(z - 0) ** 2 / (2 * 1.0 ** 2))

        return log_llh + log_piror

    print("find MAP position .........")

    # gradient MAP
    def map_loss(z):
        fake_snapshot = model.generate_fake_data(z.view(1, -1), coords_sensor_new_T)
        fake_resp = torch.squeeze(fake_snapshot[:, Opt.params_features:])
        llh_1 = torch.sum(-(fake_resp - obs_disp_T) ** 2 / (2 * sigma_T ** 2))
        llh_2 = - obs_disp_T.shape[0] * torch.log(sigma_T)
        log_llh = llh_1 + llh_2
        log_piror = torch.sum(-(z - 0) ** 2 / (2 * 1.0 ** 2))

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

    fake_map = model.generate_fake_data(z_T, coords_sensor_new_T).data.cpu().numpy()
    z_map = z_T.data.cpu().numpy()

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=[12, 3])
    ax0 = axes[0].scatter(coords_sensor_new[:, 0], coords_sensor_new[:, 1], c=fake_map[0, :n_points])
    plt.colorbar(ax0, ax=axes[0])
    ax1 = axes[1].scatter(coords_sensor_new[:, 0], coords_sensor_new[:, 1], c=fake_map[0, n_points:-n_points])
    plt.colorbar(ax1, ax=axes[1])
    ax2 = axes[2].scatter(coords_sensor_new[:, 0], coords_sensor_new[:, 1], c=fake_map[0, -n_points:])
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

    posterior_joint = model.generate_fake_data(position_array_T, coords_sensor_new_T).data.cpu().numpy()
    Opt.params_features = 625
    Opt.resp_features = 1250  # Changing the coordinates again requires changing the dimensions again
    posterior_joint_grid = model.generate_fake_data(position_array_T, coords_sensor_T).data.cpu().numpy()  # 25*25 grid
    posterior_joint_burned = posterior_joint[burn:, :]
    posterior_joint_grid_burned = posterior_joint_grid[burn:, :]
    posterior_params = posterior_joint_grid_burned[:, :625]
    posterior_resp = posterior_joint_grid_burned[:, 625:]
    posterior_params_mean = np.mean(posterior_params, axis=0)
    posterior_params_std = np.std(posterior_params, axis=0)
    posterior_resp_mean = np.mean(posterior_resp, axis=0)
    in3std = get_in3std(posterior_params_mean, posterior_params_std, ref_params)
    error_E = posterior_params_mean - ref_params


    def plot_axes_scatter(axes, xx, yy, zz, str):
        ax0 = axes.scatter(xx, yy, c=zz)
        plt.colorbar(ax0, ax=axes)
        axes.set_title(str)


    def plot_axes_pcolor(axes, xx, yy, zz, str):
        ax0 = axes.pcolor(xx, yy, zz.reshape(xx.shape[0], xx.shape[0]))
        plt.colorbar(ax0, ax=axes)
        axes.set_title(str)


    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=[16, 6])
    plot_axes_pcolor(axes[0][0], x_mesh_sensor, y_mesh_sensor, posterior_params_mean, "posterior_mean_E")
    plot_axes_pcolor(axes[0][1], x_mesh_sensor, y_mesh_sensor, ref_params, "reference_E")
    plot_axes_pcolor(axes[0][2], x_mesh_sensor, y_mesh_sensor, error_E, "error_E")
    plot_axes_pcolor(axes[0][3], x_mesh_sensor, y_mesh_sensor, posterior_params_std, "std_E")
    plot_axes_pcolor(axes[1][0], x_mesh_sensor, y_mesh_sensor, posterior_resp_mean[:625], "posterior_mean_disp_x")
    plot_axes_scatter(axes[1][1], coords_sensor_new[:, 0], coords_sensor_new[:, 1], obs_disp[:n_points],
                      "observation_x")
    plot_axes_pcolor(axes[1][2], x_mesh_sensor, y_mesh_sensor, posterior_resp_mean[625:], "posterior_mean_disp_y")
    plot_axes_scatter(axes[1][3], coords_sensor_new[:, 0], coords_sensor_new[:, 1], obs_disp[n_points:],
                      "observation_y")
    plt.savefig(f"training_result_images/posterior.png", dpi=300)
    #
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=[4, 3])
    ax = axes.pcolor(x_mesh_sensor, y_mesh_sensor, in3std.reshape(25, 25), cmap="coolwarm", vmax=1.0, vmin=0.0)
    plt.tight_layout()
    plt.savefig(f"training_result_images/in3std.png", dpi=300)

    error_l2_e = np.linalg.norm(posterior_params_mean - ref_params, 2) / np.linalg.norm(ref_params, 2)
    error_l2_x = np.linalg.norm(posterior_resp_mean[:625] - ref_disp[:625], 2) / np.linalg.norm(ref_disp[:625], 2)
    error_l2_y = np.linalg.norm(posterior_resp_mean[625:] - ref_disp[625:], 2) / np.linalg.norm(ref_disp[625:], 2)
    in3std_ration = in3std[in3std > 0].shape[0] / in3std.shape[0]
    print(f"Relative l2 error of elastic modulus field: {error_l2_e}, in 3-sigma ratio %: {round(in3std_ration, 2) * 100} %")
    print(f"Relative l2 error of x-displacement field : {error_l2_x}")
    print(f"Relative l2 error of y-displacement field : {error_l2_y}")

    # file_name = "training_result_images/postprocess_data.npz"
    # np.savez(file_name,
    #          coords=coords_sensor_new,
    #          fake_map=fake_map, z_map=z_map,
    #          sigma=sigma, obs_disp=obs_disp, burn=burn,
    #          position_array=position_array, ll_array=ll_array,
    #          posterior_joint=posterior_joint, posterior_joint_grid=posterior_joint_grid)
