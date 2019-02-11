import torch

config_coeff = [0.2, 3.0, 2.0, 1.5]

step = 0.005
num_octave = 3
iteration = 20

max_loss = 10.0


cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'