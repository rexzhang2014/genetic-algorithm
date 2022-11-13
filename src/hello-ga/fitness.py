import numpy as np
from abc import ABC, abstractmethod
from collections import Collection 
from individual import *

class Fitness(ABC) :
    @abstractmethod
    def run(self, individual) :
        pass
        
class SumFitness(Fitness) :
    '''
    Add the chromosome vector. For maximizing the selections.
    '''
    def run(self, x) :
        return x.sum()             

class WeightedSumFitness(Fitness) :
    '''
    Add the chromosome vector. For maximizing the total importance.
    '''
    def __init__(self, weights=[]) :
        self.weights = weights
    
    def run(self, individual : Individual) :
        a = np.array(individual.chromosome)
        b = np.array(self.weights)
        return np.dot(a, b.T)

class HeadingFitness(Fitness) :
    '''
    Maximize the gap between heading t elements and the tailing elements.
    '''
    def __init__(self, t=0) :
        self.t = t
    def run(self, individuals) :
        t = self.t
        return individuals[:t].sum() - individuals[t:].sum()

class ConstraintFitness(Fitness) :
    '''
    Add penalty in fitness functions if there is any constraints.  
    Notes: two way to implement constraints in GA. 1) put constraints in selector for excluding infeasible individuals every iteration. 2) put constraints in fitness function for assigning penalty on the fitness value, thus effecting on selection result. 
    '''
    def __init__(self, fx, hx=None, gx=None, alpha=1, beta=1) :
        self.fx = fx
        self.hx = hx
        self.gx = gx
        self.alpha = alpha
        self.beta = beta

    def run(self, individual) :
        
        # the overall fitness
        fitness = self.fx(individual) 

        # add bonus to the fitness
        if isinstance(self.hx, Collection) :
            for h in self.hx :
                fitness += self.alpha * h(individual)

        # minus from the fitness
        if isinstance(self.gx, Collection) :
            for g in self.gx :
                fitness -= self.beta * g(individual)

        return fitness

class SumConstraintFitness(ConstraintFitness) :
    '''
    Add penalty in fitness functions if there is any constraints.  
    Notes: two way to implement constraints in GA. 1) put constraints in selector for excluding infeasible individuals every iteration. 2) put constraints in fitness function for assigning penalty on the fitness value, thus effecting on selection result. 
    '''
    def __init__(self, total) :
        ConstraintFitness.__init__(self, fx=lambda x : x.sum() ** 2, hx=[lambda x : (x.sum()-total) ** 2] ) 
        self.total = total

class InSetFitness(ConstraintFitness) :
    '''
    Add penalty in fitness functions if there is any constraints.  
    Notes: two way to implement constraints in GA. 1) put constraints in selector for excluding infeasible individuals every iteration. 2) put constraints in fitness function for assigning penalty on the fitness value, thus effecting on selection result. 
    '''
    def _fx(self, individual) :
        avail_set = []
        for a in self.avail :
            avail_set.extend(a)
        avail_set = set(avail_set)
        f = sum( [
            1.0 * int(individual[i] in self.c_avail[i]) for i in range(len(individual))
        ] + [
            0.1 * int(individual[i] in avail_set) for i in range(len(individual))
        ])
        return f
    def _hx(self, individual) :
        return len(set(individual.chromosome))

    def _gx(self, individual) :
        return len(set(individual.chromosome)) - len(individual.chromosome)


    def __init__(self, c_avail, avail, r_c=None) :
        self.c_avail = c_avail if c_avail else None
        self.avail   = avail if avail else None
        self.r_c = r_c if r_c else None
        ConstraintFitness.__init__(self, fx=self._fx, hx=[self._hx], gx=[self._gx], alpha=1, beta=0)
        

