import numpy as np
from abc import ABC, abstractmethod

class CrossOver(ABC) :
    @abstractmethod
    def run(self, individuals) :
        pass


class SinglePointCrossOver(CrossOver) :
    def run(self, individuals) :
        offsprings = []
        for i in individuals :
            for j in individuals: 
                t = np.random.choice(range(len(individuals[0])),1)[0]
                # np.random.rand(1)[0]
                if i is j : 
                    # don't calculate twice
                    break
                # elif i == j :
                #     continue
                else :
                    # swap the genes at position t and product 2 offsprings
                    offs1 = i[:t] + j[t:]
                    offs2 = j[:t] + i[t:]
                    offsprings.append(offs1)
                    offsprings.append(offs2)
        return offsprings                     

class UniformCrossOver(CrossOver) :
    def run(self, individuals) :
        offsprings = []
        for i in individuals :
            for j in individuals: 
                t = np.random.choice(range(len(individuals)),1)[0]
                # np.random.rand(1)[0]
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

class ArithmeticCrossOver(CrossOver) :
    '''
    Only applicable for numeric individuals
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


