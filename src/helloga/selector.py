import numpy as np
import pandas as pd
from abc import ABC, abstractmethod 

class Selector(ABC):
    
    def __init__(self, constraints=None, feasible_ratio=0, min_pop=5) :
        self.constraints = [lambda x : True] if constraints is None else constraints
        self.feasible_ratio = feasible_ratio # keep a ratio of the infeasible population
        self.min_pop = min_pop # run selection step only if the total population is larger than this number

    def __feasible__(self, individual) :
        '''
        calculate all constraints for given individual, return True if every contraints evaluation result is true
        '''
        for cons in self.constraints : 
            if cons(individual) == False :
                return False
        return True

    @abstractmethod
    def __select__(self, df):
        pass

    def select(self, df) :
        if df.shape[0] > self.min_pop :
            return self.__select__(df)
        else : 
            return df

    def feasible(self, individuals, feasible_ratio=None) :
        '''
        run feasibility check for every individual and return the feasible ones for next iteration. 
        If `self.feasible_ratio` is 0.0, only feasible individuals can survive for next iteration, if 1.0, all the individuals could survive. 
        '''
        if feasible_ratio is None : 
            feasible_ratio = self.feasible_ratio

        # calculate feasibility result for each individual
        if self.constraints is not None :
            feasible = individuals.apply(func=lambda x : self.__feasible__(x))

        # if feasible_ratio is less than 1.0, keep random ratio of the infeasibile individuals. The other part will keep and reproduce further. 
        if feasible_ratio > 0 :
            n = np.math.floor(len(feasible[feasible!=True].index) * self.feasible_ratio)
            p = np.random.choice(feasible[feasible!=True].index, n, replace=False).tolist()
            f = feasible[feasible==True].index.tolist()
            return individuals[f+p]
        else : 
            # add this if-else to avoid random number drawing
            return individuals[feasible[feasible==True].index]


class LeadingSelector(Selector) : 
    '''
    Select top ratio individuals by fitness value.
    If ratio = 1, all will be selected. 
    '''
    def __init__(self, ratio=0.5, constraints=None, *args, **kwargs) :

        self.ratio = ratio
        Selector.__init__(self, constraints, *args, **kwargs)

    def __select__(self, df) :
        sorted_df = df.sort_values(by="fitness",ascending=False)
        n = np.math.ceil(sorted_df.shape[0]*self.ratio)
        selection = sorted_df.head(n)
        return selection

class RouletteWheelSelector(Selector) : 
    '''
    Select individuals by a probability with proportion to fitness value.
    If ratio = 1, same quantity of individuals will be selected with a probability by fitness value. There could be duplicated selection. 
    '''
    def __init__(self, ratio=1, negative_fitness=False, constraints=None, *args, **kwargs) :

        self.ratio = ratio
        self.negative_fitness = negative_fitness
        Selector.__init__(self, constraints, *args, **kwargs)

    def __select__(self, df) :
        # derive the selection probability by fitness value.
        f = np.abs(df['fitness'])
        N = abs(df['fitness'].sum())

        if not self.negative_fitness :
            prob = f / N
        else :
            prob = (N - f) / (N - f).sum()  # if df.shape[0] > 1 else 1

        # decide whether to select the individual 
        n = np.math.floor(df.shape[0] * self.ratio)
        selected_index = np.random.choice(df.index,n,replace=True, p=prob)

        selection      = df.loc[df.index.isin(selected_index), :]
        return selection

class LinearRankingSelector(Selector) : 
    '''
    Select individuals by a probability with proportion to linear rank of fitness value.
    If ratio = 1, it is still selected by the probability defined by the linear scale under alpha value.  
    If alpha = 0, it is uniform ranking selector, which means the probability is uniformed by the fitness ranking. If alpha > 0 and alpha < 1, it is scaled probability by the fitness ranking. If alpha = 1, all will be selected.
    '''
    def __init__(self, alpha=0.3, constraints=None, *args, **kwargs) :
        self.alpha = alpha
        Selector.__init__(self,constraints, *args, **kwargs)

    def __select__(self, df) :
        
        rank        = df["fitness"].rank(method='first',ascending=True)
        rank[rank==max(rank)-1]  = max(rank)
        prob        = self.alpha + (1 - self.alpha) * (rank - 1) / (rank.max() - 1)
        
        prob_bool   = prob.apply(func=lambda x : np.random.choice([0,1],1,p=[1-x, x]).astype(bool)).transpose()
        
        selection   = df.loc[prob_bool[prob_bool==True].index, :]
        return selection

