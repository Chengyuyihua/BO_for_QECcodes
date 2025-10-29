from typing import Callable, Tuple, List, Dict, Optional
import pickle
import numpy as np

from evaluation.decoder_based_evaluation import CSS_Evaluator
from code_construction.code_construction import CodeConstructor

import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import numpy as np
import random
from evaluation.circuit_level_noise import MonteCarloEstimationOfLogicalErrorRateUnderCircuitLevelNoise


import numpy as np
import torch
class Get_new_points_function():
    def __init__(self,method='qc-ldpc-hgp',hyperparameters = {'p': 2, 'q': 6, 'm': 2},encode='None',param={'p':0.5}):
        self.method = method
        self.hyperparameters = hyperparameters
        self.code_constructor = CodeConstructor(method,hyperparameters)
        self.encode = encode
        self.init = False
        self.param = param

    def get_new_points_function(self,number):
        if self.method == 'qc-ldpc-hgp':
            new_points = self.get_new_points_HGP(number)
        elif self.method == 'bb':
            new_points = self.get_new_bb_vector(number,self.param)
        return new_points
        
    def get_new_points_HGP(self,number):
        return np.random.randint(0, self.hyperparameters['m'] + 1, (number, self.hyperparameters['p'] * self.hyperparameters['q']))

    def get_new_bb_vector(self,number,param={'p':0.5}):
        density_expectation = param['p']
        results = []
        l = self.hyperparameters['l']
        g = self.hyperparameters['g']

        while number>0:
            new_point = np.random.choice([0,1], size=(l+g-1)*2,p=[1-density_expectation,density_expectation])
            c = self.code_constructor.construct(new_point)
            if c.k==0:
                continue
            else:
                results.append(new_point)
                number -= 1
        return np.array(results)
 



class QEC_Dataset(Dataset):
    def __init__(self,l,g,load = False,number = 100,save = True,p = 0.05,gnp_param = {'p':0.5},path = './data/codes/',noise_model='depolarizing',noise_and_decoder_param={}):
        # l = 6
        # g = 3
        para_dict = {'l':l,'g':g}
        self.noise_model = noise_model
        self.p = p
        
        self.codeconstructor = CodeConstructor(method='bb',para_dict = para_dict)
        self.gnp = Get_new_points_function(method='bb',hyperparameters = para_dict,param=gnp_param).get_new_points_function
        def objectivefunction(code):
            if noise_model == 'depolarizing':
                evaluator = CSS_Evaluator(code.hx, code.hz)
                pL = evaluator.Get_logical_error_rate_Monte_Carlo(
                    physical_error_rate=p,
                    xyz_bias=[1, 1, 1],
                    trail=noise_and_decoder_param.get('trail',10_000)
                )
            else:
                rounds = noise_and_decoder_param.get('rounds',12)
                decoder = noise_and_decoder_param.get('decoder','bplsd')
                custom_error_model = noise_and_decoder_param.get('custom_error_model',{})
                css = self.codeconstructor.construct(code)
                mc = MonteCarloEstimationOfLogicalErrorRateUnderCircuitLevelNoise(
                        css,
                        noise_model=noise_model,
                        p=p,
                        rounds=rounds,
                        custom_error_model=custom_error_model,
                        decoder=decoder
                        )
                pL = mc.run(shots= noise_and_decoder_param.get('trail',100_000), max_error = noise_and_decoder_param.get('max_error',100), num_workers = noise_and_decoder_param.get('num_worker',24))
            return pL
        self.objectfunction = objectivefunction
        # self.normalizer = Normalizer(mode='log_pos_trans',possitive=True)
        self.number = number
        self.load = load
        if load == False:
            
            self.get_a_dataset(l,g,number,save,path=path)
        else:
            self.load_dataset(l,g,number,path=path)
    def __len__(self):
        return self.number

    def get_a_dataset(self,l,g,number,save,path = './data/codes/'):
        def get_k(x):
            x = self.codeconstructor.construct(x)
            return x.k
        X = []
        y = []
        print('Generating dataset...')

        
        
        
        
        # target = 200
        # sample = 0
        number1 = number
        while number1>0:
            c = self.gnp(1)
            if get_k(c[0])!=0:
                # target-=1
                number1-=1
                X.append(c[0])
            # sample+=1
                pl = self.objectfunction(c[0])
                y.append(pl)
                print(f'code[{number-number1}],logical error rate:{pl}')
        self.X = np.array(X)
        self.y = np.array(y)

        if save:
            with open(path+f'{l}_{g}_{number}_{self.noise_model}.pkl','wb') as f:
                pickle.dump((X,y),f)
        return X,y

    def load_dataset(self,l,g,number,path = './data/codes/'):
        print('loading')
        file_name = path+f'{l}_{g}_{number}_{self.noise_model}.pkl'
        with open(file_name, 'rb') as f:
            X,y = pickle.load(f)
        self.X = X
        self.y = y

        print('successfully loaded')
            
    def __getitem__(self,idx):

        return self.X[idx], self.y[idx]
from torch.utils.data import Subset
class QECSubset(Subset):
    def __init__(self, base_dataset, indices):
        super().__init__(base_dataset,indices)
        self.X = base_dataset.X[indices]
        self.y = base_dataset.y[indices]
        self.indices = indices

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]