import numpy as np
import pandas as pd
from abc import ABC, abstractmethod 

class Selector(ABC):
    
    def __init__(self, constraints=None, feasible_ratio=1.0) :
        self.constraints = [lambda x : True] if constraints is None else constraints
        self.feasible_ratio = feasible_ratio

    def __feasible__(self, individual) :
        '''
        calculate all constraints for given individual, return True if every contraints evaluation result is true
        '''
        for cons in self.constraints : 
            if cons(individual) == False :
                return False
        return True

    @abstractmethod
    def select(self, df):
        pass
    
    def feasible(self, individuals) :
        '''
        run feasibility check for every individual and return the feasible ones for next iteration. 
        If `self.feasible_ratio` is 0.0, only feasible individuals can survive for next iteration, if 1.0, all the individuals could survive. 
        '''
        # calculate feasibility result for each individual
        if self.constraints is not None :
            feasible = individuals.apply(func=lambda x : self.__feasible__(x))

        # if feasible_ratio is less than 1.0, keep random ratio of the infeasibile individuals. The other part will keep and reproduce further. 
        if self.feasible_ratio < 1 :
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
    '''
    def __init__(self, ratio=0.5, constraints=None, *args, **kwargs) :

        Selector.__init__(self, constraints, *args, **kwargs)
        self.ratio = ratio

    
    def select(self, df) :
        sorted_df = df.sort_values(by="fitness",ascending=False)
        n_sel     = int(np.ceil(sorted_df.shape[0]*self.ratio))
        selection = sorted_df.iloc[:n_sel if n_sel > 1 else 2]
        return selection

class LinearRankingSelector(Selector) : 
    '''
    Select top ratio individuals by linear ranking of fitness value.
    '''
    def __init__(self, constraints=None, *args, **kwargs) :

        Selector.__init__(self,constraints, *args, **kwargs)

    def select(self, df) :
        
        rank        = df["fitness"].rank(method='first',ascending=True)
        rank[rank==max(rank)-1]  = max(rank)
        prob        = 0.3 + (1 - 0.3) * (rank - 1) / (rank.max() - 1)
        
        prob_bool   = prob.apply(func=lambda x : np.random.choice([0,1],1,p=[1-x, x]).astype(bool)).transpose()
        
        selection   = df.loc[prob_bool[prob_bool==True].index, :]
        return selection


class DenyLinearRankingSelector(LinearRankingSelector) : 
    def __init__(self, n_deny = 1) :

        Selector.__init__(self,constraints=[lambda ind : len(ind.indexOfPositive()) == len(ind) - n_deny])
        self.n_deny = n_deny

    def feasible(self, individuals) :
        # if the individual do not deny anything, make the lowest N weights to 0
        for i in individuals :
            r = pd.Series(i.chromosome).rank(method='first',ascending=True)
            i[r[r <= self.n_deny]] = 0
            i.reweigh()

        
        if self.constraints is not None :
            feasible  = individuals.apply(func=lambda x : self.__feasible__(x))
        
        
        return individuals[feasible[feasible==True].index]

