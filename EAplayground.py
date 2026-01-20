# Evolutionary algorim
from code_construction.code_construction import CodeConstructor
import numpy as np
from evolutionary_algorithm.ea import BivariateBicycleCodeEvolutionaryOptimization
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.optimize import minimize
import pickle
from bayesian_optimization.objective_function import ObjectiveFunction

# constructor = CodeConstructor(method='canonical',para_dict={'n':5,'k':1,'r':4})
# stabilizer_code = constructor.construct(np.array([1,1,1,1,1,0,0,1,0,0,1,0,0,1,1,0,0,0]))
# undetectable_error_rate = Undetectable_error_rate(stabilizer_code,noise_model=(1,1,1))
# print(undetectable_error_rate.evaluate(p=0.01))
from pymoo.core.sampling import Sampling
import sys
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

l=12
g=6
print(f'(l,g)=({l},{g}), dataset_index = {dataset_index}, seed={seed}, lambda = {lambda_}')

class MySampling(Sampling):
    def __init__(self, init_samples):
        super().__init__()
        self.init_samples = init_samples

    def _do(self, problem, n_samples, **kwargs):
        return self.init_samples
para_dict = {'l':l,'g':g}
code_class = 'bb'


code_constructor = CodeConstructor(method=code_class,para_dict = para_dict)
# define objective function
pp=0.05
Obj_Func = ObjectiveFunction(code_constructor,lambda_=lambda_, pp=pp,decoder_param={'trail':10_000})
obj_func = Obj_Func.forward
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
problem = BivariateBicycleCodeEvolutionaryOptimization(l=l,m=g,obj_func=Obj_Func)        
algorithm = GA(
    pop_size=20,
    sampling=MySampling(X_init),
    crossover=UniformCrossover(prob=1.0),
    mutation=PM(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
    eliminate_duplicates=True
)

res = minimize(problem,
            algorithm,
            termination=('n_gen', 11),
            seed=seed,
)
flat2 = [-v for row in problem.evaluation_history for v in row]

print(problem.evaluation_history)
print(problem.best_result)
print(problem.best_parameters)
print("Best solution found: \nX = %s\nF = %s" % (res.X, -res.F))

with open(f'./data/BO_results/EA_{l}_{g}_{dataset_index}_{seed}_{lambda_}.pkl','wb') as f:
    results = {
        'best_x':problem.best_parameters,
        'best_y':problem.best_result,
        'evaluation_history':flat2,
    }
    pickle.dump(results, f)
