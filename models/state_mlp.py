import torch
from torch import nn, Tensor
from inspect import signature
import numpy as np

class TimeDependentSwish(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.beta = nn.Sequential(
            nn.Linear(1, min(64, dim * 4)),
            nn.Softplus(),
            nn.Linear(min(64, dim * 4), dim),
            nn.Softplus(),
        )

    def forward(self, t, x):
        beta = self.beta(t.reshape(-1, 1))
        return x * torch.sigmoid_(x * beta)

class ConcatLinear_v2(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ConcatLinear_v2, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(1, dim_out, bias=False)
        self._hyper_bias.weight.data.fill_(0.0)

    def forward(self, t, x):
        return self._layer(x) + self._hyper_bias(t.view(-1, 1))

class DiffEqWrapper(nn.Module):
    def __init__(self, module):
        super(DiffEqWrapper, self).__init__()
        self.module = module
        if len(signature(self.module.forward).parameters) == 1:
            self.diffeq = lambda t, y: self.module(y)
        elif len(signature(self.module.forward).parameters) == 2:
            self.diffeq = self.module
        else:
            raise ValueError("Differential equation needs to either take (t, y) or (y,) as input.")

    def forward(self, t, y):
        return self.diffeq(t, y)

    def __repr__(self):
        return self.diffeq.__repr__()


def diffeq_wrapper(layer):
    return DiffEqWrapper(layer)

class SequentialDiffEq(nn.Module):
    """A container for a sequential chain of layers. Supports both regular and diffeq layers."""

    def __init__(self, *layers):
        super(SequentialDiffEq, self).__init__()
        self.layers = nn.ModuleList([diffeq_wrapper(layer) for layer in layers])

    def forward(self, t, x):
        for layer in self.layers:
            x = layer(t, x)
        return x

def tMLP(d_in, d_out=None, d_model=256, num_layers=6, actfn="swish"):
    assert num_layers > 1, "No weak linear nets here"
    d_out = d_in if d_out is None else d_out
    actfn = TimeDependentSwish
    layers = [ConcatLinear_v2(d_in, d_model)]
    for _ in range(num_layers - 2):
        layers.append(actfn(d_model))
        layers.append(ConcatLinear_v2(d_model, d_model))
    layers.append(actfn(d_model))
    layers.append(ConcatLinear_v2(d_model, d_out))
    return SequentialDiffEq(*layers)

class StateMLP(nn.Module):
    def __init__(
        self,
        action_dim: int = 2,
        time_dim: int = 1,
        hidden_dim: int = 128,
        horizon_size: int = 5,
    ):
      super().__init__()

      self.action_dim = action_dim
      self.obs_dim = 2 * action_dim + 1 # o = [o_{τ−1}, o_c, τ − c]
      self.time_dim = time_dim
      self.hidden_dim = hidden_dim
      self.horizon_size = horizon_size

      self.model = tMLP(d_in=self.obs_dim + \
                             self.action_dim * self.horizon_size,
                        d_out=horizon_size * action_dim,
                        d_model=hidden_dim)

    def forward(self, obs: Tensor, a: Tensor, t: Tensor) -> Tensor:
      batch_size = obs.shape[0]

      obs = obs.view(batch_size, self.obs_dim)
      a = a.contiguous().view(batch_size, self.action_dim * self.horizon_size)
      t = t.view(batch_size, self.time_dim)

      h = torch.cat([obs, a], dim=1)
      output = self.model(t, h)

      return output.view(batch_size, self.horizon_size, self.action_dim)

class WrappedVF(nn.Module):
    def __init__(self, model, obs: torch.Tensor):
        super().__init__()
        self.model = model
        self.obs = obs
    def forward(self, a: torch.Tensor, t: torch.Tensor):
        return self.model(obs=self.obs, a=a, t=t)
    

if __name__ == '__main__':
    model = StateMLP(action_dim=2, 
                     time_dim=1, 
                     hidden_dim=64, 
                     horizon_size=8)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Learnable param number:", params)