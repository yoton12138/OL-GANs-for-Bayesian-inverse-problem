# DeepONet GANS prior for inverse problem
# resolution-independent

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
    data = np.load("data/meta_data_case3.npz")
    ref_params = data["ref_params"]
    ref_resp = data["ref_resp"]
    _, _, coords = data["x_mesh"], data["t_mesh"], data["coords"]

    # sparse resp sensors for plot
    nx = 33
    sensor_lin = np.linspace(0.0, 1.0, nx)
    x_mesh_sensor, t_mesh_sensor = np.meshgrid(sensor_lin, sensor_lin)
    coords_sensor = np.vstack([x_mesh_sensor.flatten(), t_mesh_sensor.flatten()]).T

    sigma = 0.2  # ensure the sigma consistent with the obs_resp

    # note 1: use the existed observation
    obs_resp = data["observation_02_r13_21"]  # choosing observation_01_r7_21, observation_02_r7_21, etc.
    coords_sensor_new = data["coords_r13_21"]  # consistent with the observation dim
    resp_sensor_new = new_observation_maker(coords_sensor_new, coords, ref_resp.reshape(1, -1))  # reference

    # note 2: use new sensor coords and observation, replace the coords_sensor_new and obs_resp
    # nt = 21  # changeable
    # nx = 13  # changeable
    # x_points = np.sort(lhs(1, samples=nx).squeeze())
    # # x_points = np.linspace(0, 1, nx)
    # t_points = np.linspace(0, 1, nt)
    # x_mesh_sensor_new, t_mesh_sensor_new = np.meshgrid(x_points, t_points)
    # coords_sensor_new = np.vstack([x_mesh_sensor_new.flatten(), t_mesh_sensor_new.flatten()]).T
    # resp_sensor_new = new_observation_maker(coords_sensor_new, coords, ref_resp.reshape(1, -1))  # reference
    # obs_resp = np.squeeze(resp_sensor_new) + sigma*np.random.randn(resp_sensor_new.shape[1])

    Opt.resp_features = resp_sensor_new.shape[1]

    # numpy to tensor
    coords_params_T = torch.tensor(np.linspace(0, 1, 100).reshape(-1, 1), dtype=torch.float32).to(Opt.device)
    coords_sensor_new_T = torch.tensor(coords_sensor_new, dtype=torch.float32, requires_grad=True).to(Opt.device)
    coords_sensor_T = torch.tensor(coords_sensor, dtype=torch.float32, requires_grad=True).to(Opt.device)
    sigma_T = torch.tensor(sigma, dtype=torch.float32).to(Opt.device)
    obs_resp_T = torch.tensor(obs_resp, dtype=torch.float32).to(Opt.device)

    # pca decomposition
    params_means_nn = Fnn([1, 20, 20, 20, 1], act_func=nn.Tanh()).to(Opt.device)
    params_components_nn = Fnn([1, 64, 64, 64, 10], act_func=nn.Tanh()).to(Opt.device)
    resp_means_nn = Fnn([2, 20, 20, 20, 1], act_func=nn.Tanh()).to(Opt.device)
    resp_components_nn = Fnn([2, 64, 64, 64, 10], act_func=nn.Tanh()).to(Opt.device)

    checkpoint = torch.load("save_models/pod_main_state/state_pod.pt")
    params_means_nn.load_state_dict(checkpoint["stat_dict_means_p"])
    resp_means_nn.load_state_dict(checkpoint["stat_dict_means_r"])
    params_components_nn.load_state_dict(checkpoint["stat_dict_components_p"])
    resp_components_nn.load_state_dict(checkpoint["stat_dict_components_r"])

    # === model setting ===
    model = WGanGP3(coords_params_T, coords_sensor_T, Opt.resp_features, params_means_nn, params_components_nn, resp_means_nn, resp_components_nn)
    model.load_models(f"save_models/pod_main_state/state_20000.pt")

    # === sampling procedure ===
    def logp(z):
        """ define log prob density """
        z_T = torch.tensor(z, dtype=torch.float32, requires_grad=True).to(Opt.device)
        fake_snapshot = model.generate_fake_data(z_T.view(1, -1), coords_params_T, coords_sensor_new_T).data.cpu().numpy()
        fake_resp = fake_snapshot[:, Opt.params_features:]

        log_llh = np.sum(-(fake_resp - obs_resp) ** 2 / (2 * sigma ** 2)) - obs_resp.shape[0] * np.log(sigma)
        log_piror = np.sum(-(z - 0) ** 2 / (2 * 1.0 ** 2))

        return log_llh + log_piror

    print("find MAP position .........")

    # gradient MAP
    def map_loss(z):
        fake_snapshot = model.generate_fake_data(z.view(1, -1), coords_params_T, coords_sensor_new_T)
        fake_resp = torch.squeeze(fake_snapshot[:, Opt.params_features:])
        llh_1 = torch.sum(-(fake_resp - obs_resp_T) ** 2 / (2 * sigma_T ** 2))
        llh_2 = - obs_resp_T.shape[0] * torch.log(sigma_T)
        log_llh = llh_1 + llh_2
        log_piror = torch.sum(-(z - 0) ** 2 / (2 * 1.0 ** 2))

        return log_llh + log_piror

    z_T = torch.nn.Parameter(torch.zeros(1, Opt.noise_features).to(Opt.device))
    lr = 0.05
    opt_iter = 1000
    iter_array = np.zeros((opt_iter, Opt.noise_features+1))
    map_optimizer = torch.optim.Adam([z_T], lr=lr, betas=(0.9, 0.999))

    for i in tqdm.tqdm(range(opt_iter)):
        map_optimizer.zero_grad()
        loss = - map_loss(z_T)
        loss.backward()
        map_optimizer.step()
        iter_array[i, :Opt.noise_features] = z_T.data.cpu().numpy()
        iter_array[i, -1] = loss.data.cpu().numpy()

    print(f"-MAP LOSS: {loss}")

    fake_map = model.generate_fake_data(z_T, coords_params_T, coords_sensor_new_T).data.cpu().numpy()
    z_map = z_T.data.cpu().numpy()

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=[8, 3])
    x_lin = np.linspace(0, 1, 100)
    ax0 = axes[0].plot(x_lin, fake_map[0, :Opt.params_features])
    resp_map = fake_map[0, Opt.params_features:]
    ax1 = axes[1].scatter(coords_sensor_new[:, 0], coords_sensor_new[:, 1], c=resp_map, s=20)
    plt.colorbar(ax1, ax=axes[1])
    plt.savefig("training_result_images/map.png", dpi=300)

    #  Metropolis Hasting Sampling
    init_position = z_map  # Max a posterior as initial position
    burn = 5000
    sampler = MHSampler(init_position, logp, Opt.noise_features)
    position_array = sampler.sampling(10000)
    ll_array = sampler.ll_array

    fig = plt.figure(figsize=[4, 3])
    plt.plot(np.asarray(ll_array))
    plt.tight_layout()
    plt.savefig("training_result_images/ll_new.png", dpi=300)

    position_array = np.asarray(position_array)
    position_array_T = torch.tensor(position_array, dtype=torch.float32).to(Opt.device)

    posterior_joint = model.generate_fake_data(position_array_T, coords_params_T,
                                               coords_sensor_new_T).data.cpu().numpy()
    Opt.resp_features = 1089  # 33*33
    posterior_joint_grid = model.generate_fake_data(position_array_T, coords_params_T,
                                                    coords_sensor_T).data.cpu().numpy()  # 33*33 grid
    posterior_joint_burned = posterior_joint[burn:, :]
    posterior_joint_grid_burned = posterior_joint_grid[burn:, :]
    posterior_params = posterior_joint_grid_burned[:, :Opt.params_features]
    posterior_resp = posterior_joint_grid_burned[:, Opt.params_features:]
    posterior_resp_random = posterior_joint_burned[:, Opt.params_features:]
    posterior_params_mean = np.mean(posterior_params, axis=0)
    posterior_params_std = np.std(posterior_params, axis=0)
    posterior_resp_mean = np.mean(posterior_resp, axis=0)
    posterior_resp_mean_random = np.mean(posterior_resp_random, axis=0)
    in3std = get_in3std(posterior_params_mean, posterior_params_std, ref_params)
    error_E = posterior_params_mean - ref_params
    upper = posterior_params_mean + 3 * posterior_params_std
    lower = posterior_params_mean - 3 * posterior_params_std


    def plot_axes(axes, xx, yy, zz, str):
        ax0 = axes.pcolor(xx, yy, zz.reshape(xx.shape[0], xx.shape[1]))
        plt.colorbar(ax0, ax=axes)
        axes.set_title(str)


    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=[12, 3])
    axes[0].plot(x_lin, ref_params, linestyle="-", label="Reference")
    axes[0].plot(x_lin, posterior_params_mean, linestyle="--", label="Posterior_mean")
    axes[0].fill_between(x_lin, lower.flatten(), upper.flatten(), alpha=0.5, rasterized=True,
                         label="Epistemic uncertainty")
    axes[0].legend()
    plot_axes(axes[1], x_mesh_sensor, t_mesh_sensor, posterior_resp_mean.reshape(33, 33), "Posterior_mean")
    ax1 = axes[2].scatter(coords_sensor_new[:, 0], coords_sensor_new[:, 1], c=obs_resp, s=20)
    axes[2].set_title("Observation")
    plt.savefig(f"training_result_images/posterior.png", dpi=300)
    #

    error_l2_params = np.linalg.norm(posterior_params_mean - ref_params, 2) / np.linalg.norm(ref_params, 2)
    r2 = r2_score(ref_params, posterior_params_mean)
    error_l2_resp = np.linalg.norm(posterior_resp_mean_random - resp_sensor_new.squeeze(), 2) / np.linalg.norm(
        resp_sensor_new.squeeze(), 2)
    in3std_ration = in3std[in3std > 0].shape[0] / in3std.shape[0]
    print(f"Relative l2 error of parameters： {error_l2_params}, r2: {r2}, in 3-sigma ratio %: {round(in3std_ration,2)*100} %")
    print(f"Relative l2 error of response  ： {error_l2_resp}")

    # save data
    # file_name = "training_result_images/postprocess_data.npz"
    # np.savez(file_name,
    #          fake_params=None, fake_resp=None,
    #          fake_map=fake_map, z_map=z_map,
    #          sigma=sigma, burn=burn,
    #          position_array=position_array, ll_array=ll_array,
    #          posterior_joint=posterior_joint, posterior_joint_grid=posterior_joint_grid)
