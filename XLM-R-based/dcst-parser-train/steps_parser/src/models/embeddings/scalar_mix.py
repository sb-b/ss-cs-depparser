# Adapted from https://github.com/Hyperparticle/udify/blob/b6a1173e7e5fc1e4c63f4a7cf1563b469268a3b8/udify/modules/scalar_mix.py
# (MIT License)
import torch
from torch.nn import Parameter


class ScalarMixWithDropout(torch.nn.Module):
    """
    Computes a parameterised scalar mixture of the dimensions of a tensor, ``mixture = gamma * sum(s_k * tensor_k)``
    where ``s = softmax(w)``, with ``w`` and ``gamma`` scalar parameters.
    If ``dropout > 0``, then for each scalar weight, adjust its softmax weight mass to 0 with
    the dropout probability (i.e., setting the unnormalized weight to -inf). This effectively
    should redistribute dropped probability mass to all other weights.
    """

    def __init__(self, mixture_size, trainable=True, initial_scalar_parameters=None, layer_dropout=None, layer_dropout_value=-1e20):
        """
        Args:
            mixture_size: Number of layers to mix.
            trainable: Whether to train the weights of the scalar mixture. Default: True.
            initial_scalar_parameters: Initial parameters (un-normalized weights) of the scalar mixture. If not
              provided, all initial weights are set to 0. Default: None.
            layer_dropout: Dropout ratio for entire layers of scalar mixture. Default: None.
            layer_dropout_value: Value to replace the unnormalized weight of dropped layers with. Should be "close" to
              negative infinity. Default: -1e20.
        """
        super(ScalarMixWithDropout, self).__init__()
        self.mixture_size = mixture_size
        self.layer_dropout = layer_dropout

        if initial_scalar_parameters is None:
            initial_scalar_parameters = [0.0] * mixture_size

        assert len(initial_scalar_parameters) == mixture_size

        self.scalar_parameters = Parameter(torch.FloatTensor(initial_scalar_parameters), requires_grad=trainable)
        self.gamma = Parameter(torch.FloatTensor([1.0]), requires_grad=trainable)

        if self.layer_dropout:
            layer_dropout_mask = torch.zeros(len(self.scalar_parameters))
            layer_dropout_fill = torch.empty(len(self.scalar_parameters)).fill_(layer_dropout_value)
            self.register_buffer("layer_dropout_mask", layer_dropout_mask)
            self.register_buffer("layer_dropout_fill", layer_dropout_fill)

    def forward(self, input_tensor):
        """Compute a weighted sum of the dimensions of ``input_tensor`` using the coefficients stored within the
        module.
        """
        assert input_tensor.shape[0] == self.mixture_size
        num_dim = len(input_tensor.shape)

        if self.layer_dropout and self.training:
            weights = torch.where(self.layer_dropout_mask.uniform_() > self.layer_dropout, self.scalar_parameters, self.layer_dropout_fill)
        else:
            weights = self.scalar_parameters

        normed_weights = torch.nn.functional.softmax(weights, dim=0)
        normed_weights = normed_weights[(...,) + (None,)*(num_dim-1)]  # Unsqueeze weight tensor for proper broadcasting

        return self.gamma * torch.sum(input_tensor * normed_weights, dim=0)
