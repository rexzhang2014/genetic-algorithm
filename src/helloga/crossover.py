import numpy as np
from abc import ABC, abstractmethod

class CrossOver(ABC) :
    @abstractmethod
    def run(self, individuals) :
        pass


class SinglePointCrossOver(CrossOver) :
    '''
    Crossover at a single point
    Parameter
    -----------------
    No customized parameters necessary for this crossover operator

    Return
    -----------------
    A list of next generation offsprings. 

    Example
    -----------------
    000111 xo 111000 will result in two offsprings as 00100 110111 at crossover point 2

    '''
    def run(self, individuals) :
        offsprings = []
        for i in individuals :
            for j in individuals: 
                t = np.random.choice(range(len(individuals[0])),1)[0] 
                if i is j : 
                    # don't calculate twice
                    break
                else :
                    # swap the genes at position t and product 2 offsprings
                    offs1 = i[:t] + j[t:]
                    offs2 = j[:t] + i[t:]
                    offsprings.append(offs1)
                    offsprings.append(offs2)
        return offsprings                     

class MultiPointCrossOver(CrossOver) :
    '''
    Split the chromosome into N pieces and crossover. 

    Parameter
    -----------------
    N: int default 2

    The number of split points 

    Return
    -----------------
    A list of next generation offsprings. 

    Example
    -----------------
    000111 xo 111000 at N=2 will result in two offsprings as 110100 000111 at crossover points [2, 4]

    '''
    def __init__(self, N=2,) -> None:
        super().__init__()
        self.N = N

    def run(self, individuals) :
        offsprings = []
        for i in individuals :
            for j in individuals: 
                split_points = sorted(np.random.choice(range(len(individuals[0])), self.N, replace=False).tolist())
                
                if i is j : 
                    # don't calculate twice
                    break
                else :
                    for idx in range(self.N-1) :
                        offs1 = i.copy()
                        offs2 = j.copy()
                        p = split_points[idx]
                        p1 = split_points[idx+1]
                        if idx % 2 == 0 : 
                            offs1[p:p1] = j.chromosome[p:p1]
                            offs2[p:p1] = i.chromosome[p:p1]
                        offsprings.append(offs1)
                        offsprings.append(offs2)
        return offsprings                     

class UniformCrossOver(CrossOver) :
    '''
    Crossover for each position at uniform distribution probability
    Parameter
    -----------------
    prob: float default 0.2

    The probability that swap the value at each position. Must be a float number between [0,1]. Error will be raised by numpy module if the value is not applicable. 

    Return
    -----------------
    A list of next generation offsprings. 

    Example
    -----------------
    000111 xo 111000 with prob=1 will result in two offsprings as 111000 000111 because every position is swaping at a 1.0 prob.

    '''
    def __init__(self, prob=0.2) -> None:
        self.prob = prob
        super().__init__()

    def run(self, individuals) :
        offsprings = []
        
        N = len(individuals[0])

        # for each pair (i, j)
        for i in individuals :
            for j in individuals: 

                # assign a prob whether to swap the value at each index
                swap = np.random.rand(N) > self.prob

                if i is j : 
                    # don't calculate twice
                    break

                else :
                    # swap the genes at position t and product 2 offsprings
                    offs1 = i.copy()
                    offs2 = j.copy()
                    for s in range(swap) :
                        if swap[s] == True :
                            offs1[s] = j.chromosome[s]
                            offs2[s] = i.chromosome[s]
                    offsprings.append(offs1)
                    offsprings.append(offs2)

        return offsprings      

class ArithmeticCrossOver(CrossOver) :
    '''
    Only applicable for numeric individuals.
    Crossover by the value of two parents at a random portion. 
    '''
    def run(self, individuals) : 
        offsprings = []
        for i in individuals :
            for j in individuals: 
                if i is j : 
                    # don't calculate twice
                    break
                t = np.random.uniform(0.,1.,1)[0]
            
                # assign t as weight of parents (i,j) repectively
                offs1 = i * (1-t) + j * t
                offs1.reweigh()
                offs2 = j * (1-t) + i * t
                offs2.reweigh()
                offsprings.append(offs1)
                offsprings.append(offs2)

        return offsprings                                          


