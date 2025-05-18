import torch
from torch import nn, Tensor
from inspect import signature
import numpy as np
from flow_matching.utils.manifolds import Manifold, Sphere

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
        num_layers: int = 6,
        num_classes: int = 2,
        label_embedding_dim: int = 8,
    ):
      super().__init__()

      self.action_dim = action_dim
      self.obs_dim = 2 * action_dim + 1 # o = [o_{τ−1}, o_c, τ − c]
      self.time_dim = time_dim
      self.hidden_dim = hidden_dim
      self.horizon_size = horizon_size

      self.label_embedding_dim = label_embedding_dim

      self.label_embedding = nn.Embedding(num_classes, label_embedding_dim)

      self.model = tMLP(d_in=self.obs_dim + \
                             self.action_dim * self.horizon_size + \
                             label_embedding_dim,
                        d_out=horizon_size * action_dim,
                        d_model=hidden_dim,
                        num_layers=num_layers)

    def forward(self, obs: Tensor, x: Tensor, t: Tensor, label: Tensor) -> Tensor:
      batch_size = obs.shape[0]

      obs = obs.view(batch_size, self.obs_dim)
      x = x.contiguous().view(batch_size, self.action_dim * self.horizon_size)
      t = t.view(batch_size, self.time_dim)

      label_embedding = self.label_embedding(label)

      h = torch.cat([obs, x, label_embedding], dim=1)
      output = self.model(t, h)

      return output.view(batch_size, self.horizon_size, self.action_dim)

class WrappedVF(nn.Module):
    def __init__(self, model, obs: torch.Tensor, label=torch.Tensor):
        super().__init__()
        self.model = model
        self.obs = obs
        self.label = label
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        return self.model(obs=self.obs, label=self.label, x=x, t=t)
    
class ProjectToTangent(nn.Module):
    """Projects a vector field onto the tangent plane at the input."""

    def __init__(self, vecfield: nn.Module, manifold: Manifold=None, tangent_point=None):
        super().__init__()
        self.vecfield = vecfield
        self.manifold = manifold
        self.tangent_point = tangent_point

    def forward(self, obs: Tensor, x: Tensor, t: Tensor, label: Tensor) -> Tensor:
        if self.manifold:
            v = self.vecfield(obs=obs, x=x, t=t, label=label)
            if self.tangent_point is not None:
                v = self.manifold.proju(self.tangent_point, v)
            else:
                v = self.manifold.proju(x, v)
            return v
        else:
            return self.vecfield(obs=obs, x=x, t=t, label=label)
    

if __name__ == '__main__':
    # model = ProjectToTangent(
    #     vecfield=StateMLP(action_dim=3, 
    #                     time_dim=1, 
    #                     hidden_dim=64, 
    #                     horizon_size=8),
    #     manifold=Sphere)
    model = StateMLP(action_dim=3, 
                    time_dim=1, 
                    hidden_dim=64, 
                    horizon_size=8)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Learnable param number:", params)