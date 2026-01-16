device = 'cuda'
DEVICE = 'cuda'
import random
import numpy as np
import torch
from code_construction.code_construction import CodeConstructor
from bayesian_optimization.objective_function import ObjectiveFunction
from bayesian_optimization.encoder import *
from bayesian_optimization.chaincomplexembedding import *
from bayesian_optimization.gp import *
import gpytorch
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel, MaternKernel, SpectralMixtureKernel, ScaleKernel
import copy
from typing import Dict, Optional, Tuple, Any, List

import torch
import gpytorch
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.nn.utils import clip_grad_norm_
import math
from dataclasses import dataclass, asdict
from typing import Tuple, Union, Dict, Any

from torch.distributions.normal import Normal
from bayesian_optimization.bo import BO_on_QEC

def set_all_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
class Get_new_points_function():
    def __init__(self,method='qc-ldpc-hgp',hyperparameters = {'p': 2, 'q': 6, 'm': 2},encode='None'):
        self.method = method
        self.hyperparameters = hyperparameters
        self.encode = encode
        self.init = False

    def get_new_points_function(self,number):
        if self.method == 'qc-ldpc-hgp':
            new_points = self.get_new_points_HGP(number)
        elif self.method == 'bb':
            new_points = self.get_new_bb_vector(number)
        return new_points
        
    def get_new_points_HGP(self,number):
        return np.random.randint(0, self.hyperparameters['m'] + 1, (number, self.hyperparameters['p'] * self.hyperparameters['q']))

    def get_new_bb_vector(self,number):
        results = []
        l = self.hyperparameters['l']
        g = self.hyperparameters['g']
        if self.init == False and l==12 and g==999:
            print('best known bb code added to initial points')
            self.init=True

            a = np.zeros((l+g-1)*2)
            a[3]=1
            a[11+1]=1
            a[11+2]=1
            a[17+1]=1
            a[17+2]=1
            a[17+11+3]=1
            results.append(a)

        while number>0:
            new_point = np.random.randint(0,2, size=(l+g-1)*2)
            c = code_constructor.construct(new_point)
            if c.k==0:
                continue
            else:
                results.append(new_point)
                number -= 1
        return np.array(results)
if __name__ == '__main__':
    import pickle
    seed = 42
    set_all_seeds(seed)
    l = 12
    g = 6 # g here is m in Bravyi et al's paper
    para_dict = {'l':l,'g':g}
    code_class = 'bb'

    
    code_constructor = CodeConstructor(method=code_class,para_dict = para_dict)
    # define objective function
    pp=0.05
    Obj_Func = ObjectiveFunction(code_constructor, pp=pp,decoder_param={'trail':10_000})
    obj_func = Obj_Func.forward
    pl_to_obj = Obj_Func.pl_to_obj_with_std
    # method of sampling new points
    gnp = Get_new_points_function(method=code_class,hyperparameters = para_dict).get_new_points_function
    init_num = 20
    for i in range(5):
        print(f'generating new points...round {i}')
        X_init = gnp(init_num)
        y_init = []
        pl_init = []
        for x in X_init:

            y,pl = obj_func(x)
            y_init.append(y)
            pl_init.append(pl)
            print(f'x:{x},y:{y},pl:{pl}')
        data = {'X':X_init,'y':y_init,'pl':pl_init}
        with open(f"./data/BO_initial_points/BO_initial_points_{i+5}.pkl", "wb") as f:
            pickle.dump(data, f)