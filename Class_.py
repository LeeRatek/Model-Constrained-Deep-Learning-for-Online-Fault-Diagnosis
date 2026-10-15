import torch
import torch.nn as nn


class CustomSigmoidFunc:
    # The authors use scale default = 1.8, shift default = 2.5
    # We set scale = 1, shift = 0 as default values for general use.
    def __init__(self, scale=1, shift=0):
        self.scale = scale
        self.shift = shift

    def forward(self, x):
        return self.shift + self.scale * torch.sigmoid(x)


class CombinedAE(nn.Module):
    def __init__(
        self,
        input_size,
        encode2_input_size,
        output_size,
        use_dx_in_forward: bool = True,
        add_one_output_layer: bool = False,
        activation_fn: CustomSigmoidFunc = None,
    ):
        super(CombinedAE, self).__init__()
        self.fc1 = nn.Linear(input_size, 1)
        self.fc2 = nn.Linear(encode2_input_size, 1)
        self.fc3 = nn.Linear(output_size, output_size)
        self.activation_fn = activation_fn
        self.use_dx_in_forward = use_dx_in_forward
        self.add_one_output_layer = add_one_output_layer
        if self.add_one_output_layer:
            self.fc4 = nn.Linear(output_size, output_size)

    def encode(self, x):
        return self.fc1(x)

    def encode2(self, x):
        return torch.sigmoid(self.fc2(x))

    def decode(self, z):
        return self.activation_fn.forward(self.fc3(z))

    def forward(self, x, dx, q, y=None):
        if not self.use_dx_in_forward:
            dx = dx * 0.0

        z = self.encode(x) + self.encode2(q) + dx
        if self.add_one_output_layer:
            z = self.fc4(z)
        re = self.decode(z)
        return re, z
