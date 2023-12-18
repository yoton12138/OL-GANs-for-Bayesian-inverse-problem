# hyper-parameters
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("using device: ", torch.cuda.get_device_name(0))


class Opt(object):
    """ settings """
    total_data_num = 2500
    train_num = 1000
    batch_size = 250

    params_features = 3
    coords_features = 2
    noise_features = 3  # latent variables
    p = 10

    device = device
    epochs = 20000

    beta1 = 0.0
    beta2 = 0.999
    lr = 1e-4

    gp_lambda = 10
    critic_iter = 5

    layers_size_gen_params = [noise_features, 32, 32, 32, 3]
    layers_size_disc = [84, 128, 128, 128, 1]  # input 84: resolution of joint distribution, changeable

    SAVE_PER_EPOCH = 1000

