# Appendix B
# Verify the ability of AE and PCA to reconstruct 3D random uniform data.
# with 2 principal components or 2 latent variables

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pyDOE import lhs
import numpy as np
from sklearn.metrics import r2_score
from torch import nn
import torch
from networks import Fnn
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import wasserstein_distance

np.random.seed(616)

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.default'] = 'regular'
params = lhs(3, 2000)
params_train = params[:1000, :]
params_test = params[1000:, :]

pca = PCA()
pca.fit(params_train)

ub = np.array([0.8, 0.8, 0.15])
lb = np.array([0.2, 0.2, 0.05])
params_t = lb + (ub - lb)*params
params_train_t = params_t[:1000, :]
params_test_t = params_t[1000:, :]

pca_t = PCA(n_components=2)
pca_t.fit(params_train_t)

params_test_t_latent = pca_t.transform(params_test_t)
params_test_t_recons = pca_t.inverse_transform(params_test_t_latent)

print("reconstructed R2", r2_score(params_test_t, params_test_t_recons))

# fig, axes = plt.subplots(nrows=1, ncols=3, figsize=[12, 3])
# axes[0].scatter(params_test_t_recons[:, 1], params_test_t_recons[:, 0])
# axes[1].scatter(params_test_t_recons[:, 2], params_test_t_recons[:, 0])
# axes[2].scatter(params_test_t_recons[:, 2], params_test_t_recons[:, 1])
# plt.show()

ax = plt.figure().add_subplot(projection='3d')
ax.scatter(params_test_t[:, 0], params_test_t[:, 1], params_test_t[:, 2], label="original parameters")
ax.scatter(params_test_t_recons[:, 0], params_test_t_recons[:, 1], params_test_t_recons[:, 2], color="red", label="reconstructed parameters")
ax.set_xlabel("$c_{1}$", fontdict={'family': 'Times New Roman'})
ax.set_ylabel("$c_{2}$", fontdict={'family': 'Times New Roman'})
ax.set_zlabel("$c_{3}$", fontdict={'family': 'Times New Roman'})
ax.legend()
plt.savefig("training_result_images/PCA.png", dpi=600, bbox_inches="tight")
plt.savefig("training_result_images/PCA.pdf", bbox_inches="tight")

W1_distance_0 = wasserstein_distance(params_test_t[:, 0], params_test_t_recons[:, 0])
W1_distance_1 = wasserstein_distance(params_test_t[:, 1], params_test_t_recons[:, 1])
W1_distance_2 = wasserstein_distance(params_test_t[:, 2], params_test_t_recons[:, 2])
print("wasserstein_distance:", W1_distance_0, W1_distance_1, W1_distance_2)

encoder = Fnn(layers_size=[3, 32, 32, 32, 2], act_func=nn.Tanh())
decoder = Fnn(layers_size=[2, 32, 32, 32, 3], act_func=nn.Tanh())
epochs = 10000
mse_func = nn.MSELoss()
optimizer = torch.optim.Adam([{"params": encoder.parameters(), "lr": 0.001},
                             {"params": decoder.parameters(), "lr": 0.001}], betas=(0.9, 0.999))

params_train_t_T = torch.tensor(params_train_t, dtype=torch.float32)
params_test_t_T = torch.tensor(params_test_t, dtype=torch.float32)

for i in range(epochs):
    optimizer.zero_grad()
    z = encoder(params_train_t_T)
    params_train_recons = decoder(z)
    mse = mse_func(params_train_recons, params_train_t_T)
    mse.backward()
    optimizer.step()

    if i % 2000 == 0:
        z_test = encoder(params_test_t_T)
        params_test_t_recons_ae = decoder(z_test)
        params_test_t_recons_ae = params_test_t_recons_ae.data.cpu().numpy()
        print("AE reconstructed R2", r2_score(params_test_t, params_test_t_recons_ae))

ax = plt.figure().add_subplot(projection='3d')
ax.scatter(params_test_t[:, 0], params_test_t[:, 1], params_test_t[:, 2], label="original parameters")
ax.scatter(params_test_t_recons_ae[:, 0], params_test_t_recons_ae[:, 1], params_test_t_recons_ae[:, 2], color="red", label="reconstructed parameters")
ax.set_xlabel("$c_{1}$")
ax.set_ylabel("$c_{2}$")
ax.set_zlabel("$c_{3}$")
ax.legend()
plt.savefig("training_result_images/AE.png", dpi=600, bbox_inches="tight")
plt.savefig("training_result_images/AE.pdf", bbox_inches="tight")

W1_distance_0 = wasserstein_distance(params_test_t[:, 0], params_test_t_recons_ae[:, 0])
W1_distance_1 = wasserstein_distance(params_test_t[:, 1], params_test_t_recons_ae[:, 1])
W1_distance_2 = wasserstein_distance(params_test_t[:, 2], params_test_t_recons_ae[:, 2])
print("wasserstein_distance:", W1_distance_0, W1_distance_1, W1_distance_2)