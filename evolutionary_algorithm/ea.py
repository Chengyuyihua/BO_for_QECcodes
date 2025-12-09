"""
    This file contains the implementation of the Evolutionary Algorithm on the QEC codes.
"""
from pymoo.core.problem import Problem
import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
# from bayesian_optimization.bo import ObjectiveFunction
from bayesian_optimization.objective_function import ObjectiveFunction
from code_construction.code_construction import CodeConstructor

class CanonicalCSSEvolutionaryOptimization(Problem):
    def __init__(self,nbits):
        super().__init__(n_var=nbits,         
                         n_obj=1,         
                         n_constr=0,
                         xl=np.zeros(nbits),  
                         xu=np.ones(nbits),   
                         type_var=int)     
                            

    def _evaluate(self, x, out, *args, **kwargs):
        # calculate the ebaluation function
        f = np.sum(x**2, axis=1)
        out["F"] = f



class BivariateBicycleCodeEvolutionaryOptimization(Problem):
    def __init__(self,l,m,obj_func):
        self.objective_function = obj_func
        self.evaluation_history = []
        self.best_result = 1
        self.best_parameters = []
        super().__init__(n_var=2*(l+m-1),         
                         n_obj=1,         
                         n_constr=0,
                         xl=np.zeros(2*(l+m-1)),  
                         xu=np.ones(2*(l+m-1)),   
                         type_var=int)     
                            

    def _evaluate(self, x, out, *args, **kwargs):
        # calculate the ebaluation function
        out_list = []
        for i in x:
            # stabilizer_code = self.constructor.construct(i)
            temp,_ = self.objective_function.forward(i)
            temp = -temp
            if self.best_result>= temp:
                self.best_parameters = i
                self.best_result = temp
            out_list.append(temp)
        print(f"This generation's result:{out_list}")
        out["F"] = np.array(out_list)
        self.evaluation_history.append(out_list)
