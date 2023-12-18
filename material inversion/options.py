# hyper-parameters
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("using device: ", torch.cuda.get_device_name(0))


class Opt(object):
    """ settings """
    total_data_num = 999
    train_num = 990
    batch_size = 330

    params_features = 625
    resp_features = 1250
    coords_features = 2
    noise_features = 6
    p = 10

    device = device
    epochs = 20000

    beta1 = 0.0
    beta2 = 0.999
    lr_g = 1e-4
    lr_d = 1e-4

    gp_lambda = 10
    critic_iter = 5

    layers_size_disc = [1875, 256, 256, 256, 1]

    SAVE_PER_EPOCH = 1000

