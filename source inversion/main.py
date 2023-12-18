# DeepONet GANS prior for inverse problem
# sparse dataset, 9*9 grid based inverse procedure, noise levels: 0.01, 0.1
# 3 parameters and 81 sensors joint

import seaborn as sns
import pandas as pd
from utils import *
from networks import *
import tqdm

matplotlib.use("Agg")
plt.rcParams['font.family'] = 'Times New Roman'

seed_torch(616)
# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    data = np.load("data/meta_data_case1.npz")
    resp = data["resp"]
    params = data["params"]
    obs_resp = data["observation_01"]  # choosing observation_001, observation_01
    sigma = 0.1  # ensure the sigma consistent with the obs_resp
    ref_params = data["ref_params"]
    ref_resp = data["ref_resp"]
    x_mesh, y_mesh, coords = data["x_mesh"], data["y_mesh"], data["coords"]

    #  === make data set (loader)   sparse dataset ===
    sensor_lin = np.linspace(0.1, 0.9, 9)
    x_mesh_sensor, y_mesh_sensor, coords_sensor, resp_sensor = observation_maker(sensor_lin, resp)
    _, _, _, ref_resp_sensor = observation_maker(sensor_lin, ref_resp.reshape(1, -1))

    joint_snapshots = np.hstack((params, resp_sensor))
    Opt.resp_features = resp_sensor.shape[1]
    Opt.joint_features = joint_snapshots.shape[1]
    Opt.x_mesh_sensor, Opt.y_mesh_sensor = x_mesh_sensor, y_mesh_sensor

    # train test split
    train_snapshots, test_snapshots = joint_snapshots[:Opt.train_num, :], joint_snapshots[Opt.train_num:, :]

    # numpy to tensor
    coords_sensor_T = torch.tensor(coords_sensor, dtype=torch.float32, requires_grad=True).to(Opt.device)
    train_snapshots_T = torch.tensor(train_snapshots, dtype=torch.float32, requires_grad=True).to(Opt.device)
    test_snapshots_T = torch.tensor(test_snapshots, dtype=torch.float32, requires_grad=True).to(Opt.device)
    sigma_T = torch.tensor(sigma, dtype=torch.float32).to(Opt.device)
    obs_resp_T = torch.tensor(obs_resp, dtype=torch.float32).to(Opt.device)

    train_data = Dataset(train_snapshots_T)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=Opt.batch_size, drop_last=True, shuffle=False, num_workers=0)

    # === model setting ===
    g_iter = 20000
    model = WGanGP(coords_sensor_T, Opt.resp_features)
    model.load_models(f"save_models/main_state/state_{g_iter}.pt")  # choose to train a model or use an exist model
    # model.train(train_loader)  # choose to train a model or use an exist model

    z = torch.randn(2000, Opt.noise_features).to(Opt.device)
    fake_snapshots = model.generate_fake_data(z, coords_sensor_T).data.cpu().numpy()
    fake_params = fake_snapshots[:, :Opt.params_features]
    fake_resp = fake_snapshots[:, Opt.params_features:]

    ev_real = get_ev(resp_sensor)
    ev_fake = get_ev(fake_resp)

    fig = plt.figure(figsize=[4, 3])
    plt.plot(ev_real[:10], label="reference", marker="o")
    plt.plot(ev_fake[:10], label="generated", marker="o")
    plt.legend()
    plt.savefig("training_result_images/explained_variance_ratio_.png", dpi=300)

    # sns.set_theme(style="white")
    df = pd.DataFrame(fake_params)
    g = sns.PairGrid(df, diag_sharey=False)
    g.map_upper(sns.scatterplot, s=15)
    g.map_lower(sns.kdeplot)
    g.map_diag(sns.kdeplot, lw=2)
    plt.savefig("training_result_images/fake_params_dist.png", dpi=300)

    # === sampling procedure ===
    def logp(z):
        """ define log-probability density """
        z_T = torch.tensor(z, dtype=torch.float32, requires_grad=True).to(Opt.device)
        fake_snapshot = model.generate_fake_data(z_T.view(1, -1), coords_sensor_T).data.cpu().numpy()
        fake_resp = np.squeeze(fake_snapshot[:, Opt.params_features:])

        log_llh = np.sum(-(fake_resp - obs_resp)**2 / (2*sigma**2)) - obs_resp.shape[0]*np.log(sigma)
        log_piror = np.sum(-(z - 0)**2/(2*1.0**2))

        return log_llh + log_piror

    # gradient MAP
    def map_loss(z):
        fake_snapshot = model.generate_fake_data(z.view(1, -1), coords_sensor_T)
        fake_resp = torch.squeeze(fake_snapshot[:, Opt.params_features:])
        llh_1 = torch.sum(-(fake_resp - obs_resp_T)**2 / (2*sigma_T**2))
        llh_2 = - obs_resp_T.shape[0]*torch.log(sigma_T)
        log_llh = llh_1 + llh_2
        log_piror = torch.sum(-(z - 0)**2/(2*1.0**2))

        return log_llh + log_piror

    z_T = torch.nn.Parameter(torch.randn(1, Opt.noise_features).to(Opt.device))
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

    fake_map = model.generate_fake_data(z_T, coords_sensor_T).data.cpu().numpy()
    z_map = z_T.data.cpu().numpy()

    res = sensor_lin.shape[0]
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=[4, 3])
    ax0 = axes.pcolor(x_mesh_sensor, y_mesh_sensor, fake_map[0, Opt.params_features:].reshape(res, res))
    plt.colorbar(ax0, ax=axes)
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
    plt.savefig("training_result_images/ll.png", dpi=300)

    position_array = np.asarray(position_array)
    position_array_T = torch.tensor(position_array, dtype=torch.float32, requires_grad=True).to(Opt.device)

    posterior_joint = model.generate_fake_data(position_array_T, coords_sensor_T).data.cpu().numpy()
    posterior_joint_burned = posterior_joint[burn:, :]
    posterior_params = posterior_joint_burned[:, :Opt.params_features]
    posterior_resp = posterior_joint_burned[:, Opt.params_features:]
    posterior_params_mean = np.mean(posterior_params, axis=0)
    posterior_params_std = np.std(posterior_params, axis=0)
    posterior_resp_mean = np.mean(posterior_resp, axis=0)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=[12, 3])
    vmin = np.min(ref_resp_sensor)
    vmax = np.max(ref_resp_sensor)
    ax0 = axes[0].pcolor(x_mesh_sensor, y_mesh_sensor, posterior_resp_mean.reshape(res, res), vmin=vmin, vmax=vmax)
    plt.colorbar(ax0, ax=axes[0])
    axes[0].set_title("posterior_mean")
    ax1 = axes[1].pcolor(x_mesh_sensor, y_mesh_sensor, ref_resp_sensor.reshape(res, res), vmin=vmin, vmax=vmax)
    plt.colorbar(ax1, ax=axes[1])
    axes[1].set_title("reference")
    ax2 = axes[2].pcolor(x_mesh_sensor, y_mesh_sensor, obs_resp.reshape(res, res), vmin=vmin, vmax=vmax)
    plt.colorbar(ax2, ax=axes[2])
    axes[2].set_title("observed temperature")
    plt.savefig(f"training_result_images/posterior_mean_resp_{g_iter}.png", dpi=300)

    print("Plotting the joint distribution. Hold on...")
    # sns.set_theme(style="white")
    df = pd.DataFrame(posterior_params)
    g = sns.PairGrid(df, diag_sharey=False)
    g.map_upper(sns.scatterplot, s=15)
    g.map_lower(sns.kdeplot)
    g.map_diag(sns.kdeplot, lw=2)
    plt.savefig(f"training_result_images/pair_grid_{g_iter}.png", dpi=300)

    print(f"param true:     {np.array2string(ref_params)}")
    print(f"posterior mean: {posterior_params_mean}")
    print(f"relative error: {np.array2string(np.abs(posterior_params_mean - ref_params) / ref_params)}")
    print(f"posterior std:  {posterior_params_std}")

    # fake samples
    plt.rcParams['mathtext.default'] = 'regular'
    ax = plt.figure().add_subplot(projection='3d')
    ax.scatter(params[:1000, 0], params[:1000, 1], params[:1000, 2], label="original parameters")
    ax.scatter(fake_params[:1000, 0], fake_params[:1000, 1], fake_params[:1000, 2], color="red",
               label="reconstructed parameters")
    ax.set_xlabel("$c_{1}$", fontdict={'family': 'Times New Roman'})
    ax.set_ylabel("$c_{2}$", fontdict={'family': 'Times New Roman'})
    ax.set_zlabel("$c_{3}$", fontdict={'family': 'Times New Roman'})
    ax.legend()
    plt.savefig("training_result_images/OL-GAN.png", dpi=600, bbox_inches="tight")
    plt.savefig("training_result_images/OL-GAN.pdf", bbox_inches="tight")
    from scipy.stats import wasserstein_distance
    W1_distance_0 = wasserstein_distance(params[:1000, 0], fake_params[:1000, 0])
    W1_distance_1 = wasserstein_distance(params[:1000, 1], fake_params[:1000, 1])
    W1_distance_2 = wasserstein_distance(params[:1000, 2], fake_params[:1000, 2])
    print("wasserstein_distance:", W1_distance_0, W1_distance_1, W1_distance_2)

    # save data
    # file_name = "training_result_images/postprocess_data.npz"
    # np.savez(file_name,
    #          fake_params=fake_params, fake_resp=fake_resp,
    #          fake_map=fake_map, z_map=z_map, burn=burn,
    #          position_array=position_array, ll_array=ll_array,
    #          posterior_joint=posterior_joint)
