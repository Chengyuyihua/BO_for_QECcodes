import random
import numpy as np
import torch
from code_construction.code_construction import CodeConstructor
from bayesian_optimization.objective_function import ObjectiveFunction
import sys
def set_all_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
class Get_new_points_function():
    def __init__(self,method='qc-ldpc-hgp',code_constructor = None,encode='None'):
        self.method = method
        self.code_constructor = code_constructor
        self.encode = encode
        self.init = False

    def get_new_points_function(self,number):
        if self.method == 'qc-ldpc-hgp':
            new_points = self.get_new_points_HGP(number)
        elif self.method == 'bb':
            new_points = self.get_new_bb_vector(number)
        return new_points
        
    def get_new_points_HGP(self,number):
        return np.random.randint(0, code_constructor.para_dict['m'] + 1, (number, code_constructor.para_dict['p'] * code_constructor.para_dict['q']))

    def get_new_bb_vector(self,number):
        results = []
        l = self.code_constructor.para_dict['l']
        g = self.code_constructor.para_dict['g']
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
            c = self.code_constructor.construct(new_point)
            if c.k==0:
                continue
            else:
                results.append(new_point)
                number -= 1
        return np.array(results)
    
if __name__ == '__main__':
    import pickle
    
    
    if len(sys.argv) == 3:
        seed= int(sys.argv[1])
        dataset_index = int(sys.argv[2])
        lambda_ = 1
    elif len(sys.argv) >3:
        seed= int(sys.argv[1])
        dataset_index = int(sys.argv[2])
        lambda_ = float(sys.argv[3])
    else:
        seed = 42
        dataset_index = 0
        lambda_ = 1
    set_all_seeds(seed)
    l = 12
    g = 6 # g here is m in Bravyi et al's paper
    print(f'(l,g)=({l},{g}), dataset_index = {dataset_index}, seed={seed}, lambda = {lambda_}')
    
    para_dict = {'l':l,'g':g}
    code_class = 'bb'
    
    if l ==6 and g==3:
        init_data_file = f"./data/BO_initial_points/BO_initial_points_{dataset_index}_{lambda_}_63.pkl"
    else:
        init_data_file = f"./data/BO_initial_points/BO_initial_points_{dataset_index}_{lambda_}.pkl"
    # file with 63 suffix has (l,m)=(6,3). Otherwise (l,m)=(12,6)
    with open(init_data_file, "rb") as f:
        data = pickle.load(f)
        X_init = data['X']
        y_init = data['y']
        pl_init = data['pl']

    
    code_constructor = CodeConstructor(method=code_class,para_dict = para_dict)
    # define objective function
    pp=0.05
    Obj_Func = ObjectiveFunction(code_constructor,lambda_ = lambda_ ,pp=pp,decoder_param={'trail':10_000})
    obj_func = Obj_Func.forward
    pl_to_obj = Obj_Func.pl_to_obj_with_std
    # method of sampling new points
    gnp = Get_new_points_function(method=code_class,code_constructor=code_constructor).get_new_points_function
    X_random = gnp(4*50)
    flat3 = []
    best_x = None
    best_y = -999
    x_history = []
    pl_history = []
    for i in range(4*50):
        F,pL = obj_func(X_random[i])
        print(f'The {i}th point, obj_func:{F}')
        flat3.append(F)
        x_history.append(X_random[i])
        pl_history.append(pL)
        if F>=best_y:
            best_x = X_random[i]
            best_y = F


    flat3 = y_init + flat3
    with open(f'./data/BO_results/RS_{l}_{g}_{dataset_index}_{seed}_{lambda_}.pkl','wb') as f:
        results = {
            'best_x':best_x,
            'best_y':best_y,
            'evaluation_history':flat3
        }
        pickle.dump(results, f)
