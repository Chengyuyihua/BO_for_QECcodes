import random
import numpy as np
import torch
from code_construction.code_construction import CodeConstructor
from bayesian_optimization.objective_function import ObjectiveFunction
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
            c = self.code_constructor.construct(new_point)
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
    dataset_index = 0
    
    para_dict = {'l':l,'g':g}
    code_class = 'bb'
    if l ==6 and g==3:
        init_data_file = f"./data/BO_initial_points/BO_initial_points_{dataset_index}_63.pkl"
    else:
        init_data_file = f"./data/BO_initial_points/BO_initial_points_{dataset_index}.pkl"
    # file with 63 suffix has (l,m)=(6,3). Otherwise (l,m)=(12,6)
    with open(init_data_file, "rb") as f:
        data = pickle.load(f)
        X_init = data['X']
        y_init = data['y']
        pl_init = data['pl']

    
    code_constructor = CodeConstructor(method=code_class,para_dict = para_dict)
    # define objective function
    pp=0.05
    Obj_Func = ObjectiveFunction(code_constructor, pp=pp,decoder_param={'trail':10_000})
    obj_func = Obj_Func.forward
    pl_to_obj = Obj_Func.pl_to_obj_with_std
    # method of sampling new points
    gnp = Get_new_points_function(method=code_class,code_constructor=code_constructor).get_new_points_function
    X_random = gnp(4*50)
    flat3 = []
    best_x = None
    best_y = -999
    for i in range(4*50):
        F,_ = obj_func(X_random[i])
        flat3.append(F)
        if F>=best_y:
            best_x = X_random[i]
            best_y = F


    flat3 = y_init + flat3
    with open(f'./data/BO_results/EA_{l}_{g}_{dataset_index}_{seed}','wb') as f:
        results = {
            'best_x':best_x,
            'best_y':best_y,
            'evaluation_history':flat3
        }
        pickle.dump(results, f)
