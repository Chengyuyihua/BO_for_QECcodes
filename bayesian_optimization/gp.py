"""
    This file defines the Gaussian Process model used in Bayesian Optimization
"""
from typing import Optional
import torch
import torch.nn as nn
import gpytorch
from gpytorch.means import ConstantMean, LinearMean, Mean
from botorch.models.gpytorch import GPyTorchModel
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean,LinearMean
from gpytorch.models import ExactGP
from torch import Tensor
from gpytorch.priors import LogNormalPrior
from gpytorch.constraints import Interval
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
import math
device='cuda'

class NNMean(Mean):
    def __init__(self, input_size, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

    
class GaussianProcess_QEC_(ExactGP, GPyTorchModel):

    _num_outputs = 1  # to inform GPyTorchModel API

    def __init__(self, train_X, train_Y, kernel=None,encoder = None,embedding = None,train_Yvar: Optional[Tensor] = None,mean = 'linear',grakel=False,mean_input = 16,encoder2 = None,mean_module=None):
        # NOTE: This ignores train_Yvar and uses inferred noise instead.
        # squeeze output dim before passing train_Y to ExactGP
        likelihood =  likelihood = GaussianLikelihood(
                noise_prior=LogNormalPrior(math.log(1e-2), 0.5),
                noise_constraint=Interval(1e-6, 1.0)
        )
        super().__init__(train_X, train_Y.squeeze(-1),likelihood)
        self.grakel = grakel
        if encoder==None:
            def encode_(x):
                return x
            self.encode = encode_
        else:
            self.encode = encoder
        if encoder2==None:
            def encode_(x):
                return x
            self.encode2 = encode_
        else:
            self.encode2 = encoder2
        if embedding==None:
            def embed_(x):
                return x
            self.embed = embed_
        else:
            self.embed = embedding
        if mean in ['linear']:
            self.mean_module = LinearMean(mean_input)
        elif mean in ['nn','NN']:
            self.mean_module = NNMean(input_size=mean_input, hidden=64)
        elif mean in ['given']:
            self.mean_module = mean_module
        else :
            self.mean_module = ConstantMean()
        self.covar_module = kernel

        self.to(train_X)  

    def forward(self, x):
        
        # print(type(feats),len(feats))
        # print(feats)
        # print('end printing')
        # print("feats mean/std:", feats.mean().item(), feats.std().item())
        if self.grakel==True:
            mean_x  = self.mean_module(self.encode2(x)).squeeze(-1)
            feats = self.embed(self.encode(x))
        else: 
            feats = self.embed(self.encode(x))
            mean_x  = self.mean_module(feats)
        covar_x = self.covar_module(feats)
        
        # print(f'covar_x:{type(covar_x)},{covar_x}')
        
        return MultivariateNormal(mean_x, covar_x)

class GaussianProcess_QEC(gpytorch.models.ExactGP):
    """
    ExactGP that:
      X (continuous/int design) --(encoder.encode)-> list[views]
      --> embedder(list) -> z in R^d
      --> GP mean(z), cov(z)
    """

    def __init__(self, train_x, train_y,
                 *,
                 likelihood: gpytorch.likelihoods.GaussianLikelihood,
                 kernel: gpytorch.kernels.Kernel,
                 encoder,             # callable: encode(X_batch) -> list of views
                 embed: nn.Module,  # ChainComplexEmbedder
                 mean: str = 'constant',
                 mean_input: int = 64):
        # train_y must be 1D for ExactGP; we'll squeeze in caller
        super().__init__(train_x, train_y.squeeze(-1) if train_y.dim() == 2 else train_y, likelihood)
        self.encoder = encoder            # Python callable (non-differentiable wrt X)
        self.embed = embed            # nn.Module on device
        self.embed.to(device)

        d_model = next(self.embed.parameters()).shape[-1] if hasattr(self.embed, 'parameters') else 128
        d_model = getattr(self.embed, 'd_model', d_model)

        # Mean module
        if mean == 'zero':
            self.mean_module = gpytorch.means.ZeroMean()
        elif mean == 'constant':
            self.mean_module = gpytorch.means.ConstantMean()
        elif mean == 'linear':
            self.mean_module = gpytorch.means.LinearMean(input_size=d_model)
        elif mean == 'nn':
            # Small MLP mean head on top of z
            self.nn_mean = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, mean_input),
                nn.GELU(),
                nn.Linear(mean_input, 1),
            ).to(device)
            # wrap to a Mean that calls nn_mean (simple lambda-style)
            class _NNMean(gpytorch.means.Mean):
                def __init__(self, head): 
                    super().__init__(); self.head = head
                def forward(self, x):      # x: (N, d_model)
                    return self.head(x).squeeze(-1)
            self.mean_module = _NNMean(self.nn_mean)
        else:
            self.mean_module = gpytorch.means.ConstantMean()

        # Kernel (ScaleKernel wrapping a base kernel is recommended)
        self.covar_module = kernel

        # We’ll cache last z to avoid重复计算 in loss & pred (optional)
        self._cache_last_inputs = None
        self._cache_last_z = None

    def _x_to_z(self, X: torch.Tensor) -> torch.Tensor:
        """
        Convert a batch of design vectors X (B, D) to embedding z (B, d_model).
        """
        # Move to CPU numpy for code constructor if needed
        X_np = X.detach().cpu().numpy() if isinstance(X, torch.Tensor) else np.asarray(X)
        sample_views_list = self.encoder(X_np)            # list of views (len=B)
        # embedder.forward(list) packs internally and returns (B, d_model)
        z = self.embed(sample_views_list)                 # (B, d_model) on device
        return z

    def forward(self, X: torch.Tensor) -> MultivariateNormal:
        """
        Return latent f(X) distribution (MVN). GPyTorch's likelihood wraps this to p(y|X).
        """
        X = X.to(device)
        z = self._x_to_z(X)                               # (N, d_model)
        # cache
        self._cache_last_inputs = X
        self._cache_last_z = z

        mean_x = self.mean_module(z)                      # (N,)
        covar_x = self.covar_module(z)                    # kernel on z
        return MultivariateNormal(mean_x, covar_x)

    # 若你更喜欢 model.posterior(X) 的接口，可以保留：
    def posterior(self, X: torch.Tensor):
        self.eval(); self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # predictive distribution of y
            return self.likelihood(self(X.to(device)))
    
if __name__ == '__main__':
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    import gpytorch
    from gpytorch.kernels import ScaleKernel, RBFKernel
    from gpytorch.distributions import MultivariateNormal
    from gpytorch.means import ConstantMean
    from gpytorch.likelihoods import GaussianLikelihood
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from torch.optim import Adam
    def train_gpytorch_model(model, train_x, train_y, training_iter=100, lr=0.1):
        model.train()
        model.likelihood.train()
        optimizer = Adam(model.parameters(), lr=lr)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)

        for _ in range(training_iter):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y.squeeze(-1))
            loss.backward()
            optimizer.step()
        return model
    n_train = 20
    train_x = torch.linspace(0, 1, n_train).unsqueeze(-1)
    train_y = torch.sin(train_x.squeeze() * 2 * np.pi) + 0.2 * torch.randn(n_train)
    train_y = train_y.unsqueeze(-1)

    # 2. Instantiate and train the model
    kernel = ScaleKernel(RBFKernel(ard_num_dims=1))
    model = GaussianProcess_QEC(train_x, train_y, kernel=kernel)
    model = train_gpytorch_model(model, train_x, train_y)

    # 3. Generate test points and make predictions
    model.eval()
    model.likelihood.eval()
    test_x = torch.linspace(0, 1, 50).unsqueeze(-1)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred_dist = model.posterior(test_x)
        mean = pred_dist.mean
        stddev = pred_dist.stddev

    # 4. Print first few predictions
    print("First 10 mean predictions:\n", mean[:10].numpy())
    print("First 10 stddev predictions:\n", stddev[:10].numpy())
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred_dist = model.likelihood(model(test_x))
        mean = pred_dist.mean
        stddev = pred_dist.stddev

    # 4. Print first few predictions
    print("First 10 mean predictions:\n", mean[:10].numpy())
    print("First 10 stddev predictions:\n", stddev[:10].numpy())