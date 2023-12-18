# DeepONet GANS prior for inverse problem
# resolution-independent, using the same network as main.py
# new sensor inversion, Different sensor arrangements can be tried to obtain inversion results
# sigma, coords_sensor_new, obs_resp should be consistent to ensure that the code runs.(changeable)

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
    data = np.load("data/meta_data_case1.npz")
    ref_params = data["ref_params"]
    ref_resp = data["ref_resp"]
    x_mesh, y_mesh, coords = data["x_mesh"], data["y_mesh"], data["coords"]

    sigma = 0.1  # ensure the sigma consistent with the obs_resp

    # note 1: use the existed observation
    obs_resp = data["observation_r64_01"]  # choosing observation_r64_001, observation_r36_001
    coords_sensor_new = data["coords_r64"]  # consistent with the observation dim
    resp_sensor_new = new_observation_maker(coords_sensor_new, coords, ref_resp.reshape(1, -1))  # reference

    # note 2: use new sensor coords and observation, replace the coords_sensor_new and obs_resp
    # ub = np.array([1.0, 1.0])
    # lb = np.array([0.0, 0.0])
    # n_points = 64  # changeable
    # coords_sensor_new = lb + (ub - lb) * lhs(2, samples=n_points)
    # resp_sensor_new = new_observation_maker(coords_sensor_new, coords, ref_resp.reshape(1, -1))  # reference
    # obs_resp = np.squeeze(resp_sensor_new) + sigma*np.random.randn(resp_sensor_new.shape[1])

    Opt.resp_features = resp_sensor_new.shape[1]

    # numpy to tensor
    coords_sensor_new_T = torch.tensor(coords_sensor_new, dtype=torch.float32, requires_grad=True).to(Opt.device)
    sigma_T = torch.tensor(sigma, dtype=torch.float32).to(Opt.device)
    obs_resp_T = torch.tensor(obs_resp, dtype=torch.float32).to(Opt.device)

    # ==== model setting ===
    model = WGanGP(coords_sensor_new_T, Opt.resp_features)
    model.load_models("save_models/main_state/state_20000.pt")

    # === sampling procedure ===
    def logp(z):
        """ define log probability density """
        z_T = torch.tensor(z, dtype=torch.float32, requires_grad=True).to(Opt.device)
        fake_snapshot = model.generate_fake_data(z_T.view(1, -1), coords_sensor_new_T).data.cpu().numpy()
        fake_resp = np.squeeze(fake_snapshot[:, Opt.params_features:])

        log_llh = np.sum(-(fake_resp - obs_resp)**2 / (2*sigma**2)) - obs_resp.shape[0]*np.log(sigma)
        log_piror = np.sum(-(z - 0)**2/(2*1.0**2))

        return log_llh + log_piror

    # gradient MAP
    def map_loss(z):
        fake_snapshot = model.generate_fake_data(z.view(1, -1), coords_sensor_new_T)
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

    fake_map = model.generate_fake_data(z_T, coords_sensor_new_T).data.cpu().numpy()
    z_map = z_T.data.cpu().numpy()

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
    position_array_T = torch.tensor(position_array, dtype=torch.float32, requires_grad=True).to(Opt.device)

    posterior_joint = model.generate_fake_data(position_array_T, coords_sensor_new_T).data.cpu().numpy()
    posterior_joint_burned = posterior_joint[burn:, :]
    posterior_params = posterior_joint_burned[:, :Opt.params_features]
    posterior_resp = posterior_joint_burned[:, Opt.params_features:]
    posterior_params_mean = np.mean(posterior_params, axis=0)
    posterior_params_std = np.std(posterior_params, axis=0)
    posterior_resp_mean = np.mean(posterior_resp, axis=0)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=[12, 3])
    ax0 = axes[0].scatter(coords_sensor_new[:, 0], coords_sensor_new[:, 1], c=posterior_resp_mean, marker="o", s=40)
    plt.colorbar(ax0, ax=axes[0])
    axes[0].set_title("posterior_mean")
    ax1 = axes[1].scatter(coords_sensor_new[:, 0], coords_sensor_new[:, 1], c=resp_sensor_new, marker="o", s=40)
    plt.colorbar(ax1, ax=axes[1])
    axes[1].set_title("reference")
    ax2 = axes[2].scatter(coords_sensor_new[:, 0], coords_sensor_new[:, 1], c=obs_resp, marker="o", s=40)
    plt.colorbar(ax2, ax=axes[2])
    axes[2].set_title("reference")
    plt.savefig("training_result_images/posterior_mean_resp_scatter.png", dpi=300)

    print("Plotting the joint distribution. Hold on...")
    # sns.set_theme(style="white")
    df = pd.DataFrame(posterior_params)
    g = sns.PairGrid(df, diag_sharey=False)
    g.map_upper(sns.scatterplot, s=15)
    g.map_lower(sns.kdeplot)
    g.map_diag(sns.kdeplot, lw=2)
    plt.savefig("training_result_images/pair_grid_new.png", dpi=300)

    print(f"param true:     {np.array2string(ref_params)}")
    print(f"posterior mean: {posterior_params_mean}")
    print(f"relative error: {np.array2string(np.abs(posterior_params_mean - ref_params) / ref_params)}")
    print(f"posterior std:  {posterior_params_std}")

    # save data
    # file_name = "training_result_images/postprocess_data.npz"
    # np.savez(file_name,
    #          coords=coords_sensor_new,
    #          fake_map=fake_map, z_map=z_map,
    #          sigma=sigma, burn=burn,
    #          position_array=position_array, ll_array=ll_array,
    #          posterior_joint=posterior_joint)
