import torch
import torch.nn as nn

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str, normalize=True):
        if normalize:
            # Check for NaNs in the input data before normalization
            assert not torch.isnan(x).any(), "NaN detected in input data before normalization"

            if mode == 'norm':
                self._get_statistics(x)
                x = self._normalize(x)
                assert not torch.isnan(x).any(), "NaN detected after normalization"
            elif mode == 'denorm':
                x = self._denormalize(x)
                assert not torch.isnan(x).any(), "NaN detected after denormalization"
            else:
                raise NotImplementedError
        return x


    def _init_params(self):
        # Initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
            print(f"Subtracting last value: {self.last}")
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
            print(f"Calculated mean: {self.mean}")
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()
        print(f"Calculated standard deviation: {self.stdev}")

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / (self.stdev + self.eps)  # Adding epsilon to avoid division by zero
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x