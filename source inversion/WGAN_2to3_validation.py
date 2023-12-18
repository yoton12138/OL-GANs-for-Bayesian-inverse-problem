# Appendix B
# Verify the ability of W-GAN to reconstruct 3D random uniform data.
# remember to change the network inputs in options. 2 for gen, 3 for disc

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
    params = data["params"]

    #  === make data set (loader)   sparse dataset ===
    train_params = params[:Opt.train_num, :]
    # numpy to tensor
    train_snapshots_T = torch.tensor(train_params, dtype=torch.float32).to(Opt.device)

    train_data = Dataset(train_snapshots_T)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=Opt.batch_size, drop_last=True, shuffle=False, num_workers=0)

    # === model setting ===
    g_iter = 20000
    model = WGanGP3("no coords needed", "no resp")
    model.load_models(f"save_models/WGAN-2to3_state/state_{g_iter}.pt")
    # model.train(train_loader)

    z = torch.randn(2000, Opt.noise_features).to(Opt.device)
    fake_params = model.generate_fake_data(z, "no coords needed").data.cpu().numpy()

    # generated samples
    plt.rcParams['mathtext.default'] = 'regular'
    ax = plt.figure().add_subplot(projection='3d')
    ax.scatter(params[:1000, 0], params[:1000, 1], params[:1000, 2], label="original parameters")
    ax.scatter(fake_params[:1000, 0], fake_params[:1000, 1], fake_params[:1000, 2], color="red",
               label="reconstructed parameters")
    ax.set_xlabel("$c_{1}$", fontdict={'family': 'Times New Roman'})
    ax.set_ylabel("$c_{2}$", fontdict={'family': 'Times New Roman'})
    ax.set_zlabel("$c_{3}$", fontdict={'family': 'Times New Roman'})
    ax.legend()
    plt.savefig("training_result_images/WGAN-GP.png", dpi=600, bbox_inches="tight")
    plt.savefig("training_result_images/WGAN-GP.pdf", bbox_inches="tight")
    from scipy.stats import wasserstein_distance
    W1_distance_0 = wasserstein_distance(params[:1000, 0], fake_params[:1000, 0])
    W1_distance_1 = wasserstein_distance(params[:1000, 1], fake_params[:1000, 1])
    W1_distance_2 = wasserstein_distance(params[:1000, 2], fake_params[:1000, 2])
    print("wasserstein_distance:", W1_distance_0, W1_distance_1, W1_distance_2)


