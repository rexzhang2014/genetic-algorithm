import pandas as pd 
from crossover import SinglePointCrossOver
from selector import LinearRankingSelector, LeadingSelector
from fitness import SumFitness
from copy import deepcopy

import logging, sys, os, random, re

class Species() :
    _log_levels = {
        'info' : logging.INFO,
        'debug' : logging.DEBUG
    }

    def __init__(
        self, individuals=[], 
        selector=LeadingSelector(), 
        crossover=SinglePointCrossOver(), 
        fitness_func=SumFitness(),
        log_level='info'
    ) :
        self.individuals = pd.Series(individuals)
        self.crossover = crossover
        self.fitness_func = fitness_func
        self.selector = selector
        self.df = None
        self.fitness = None
        
        logging.basicConfig(level = self._log_levels[log_level],format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(str(self.__class__.__name__))

    def calculate_fitness(self, func=None) :
        # in this step, fitness is re-calculated as individuals have changed. df is rebuilt as individuals and fitness have changed
        func = self.fitness_func.run if func is None else func 
        fitness = self.individuals.apply(func=func)
        self.fitness = fitness
        self.df = pd.concat([self.individuals, fitness],axis=1)
        self.df.columns = ["individuals", "fitness"]
        self.logger.debug("FITNESS - top:{}; sum: {}; avg:{}; population:{}".format(self.fitness.max(), self.fitness.sum(), self.fitness.mean(),self.population()))
        return pd.concat([self.individuals, fitness],axis=1)

    def select(self, func=None, verbose=True) :
        func = self.selector.select if func is None else func 
        self.df = func(self.df) #Select with df
        self.individuals = self.df["individuals"]
        self.fitness     = self.df["fitness"]
        if verbose :
            self.logger.debug("SELECTION -- top:{}; sum: {}; avg:{}; population:{}".format( self.fitness.max(), self.fitness.sum(), self.fitness.mean(),self.population()))
            # x = self.sort()
            # print(x.iloc[0,0])
            # print(self.individuals.iloc[0])
        
        return self.df

    def feasible(self, func=None) :
        # at this step, only feasible individuals survive in feasibility constraints
        func = self.selector.feasible if func is None else func 
        self.individuals = func(self.individuals)
        # self.individuals = self.df["individuals"]
        # self.fitness     = self.df["fitness"]
        self.logger.debug("FEASIBLE -- top:{}; sum: {}; avg:{}; population:{}".format(self.fitness.max(), self.fitness.sum(), self.fitness.mean(),self.population()))
        return self.individuals

    def reproduce(self, func=None, *args, **kwargs ) :
        if len(args) > 1 :
            t = args[1] 
        elif "t" in kwargs.keys() :
            t = kwargs["t"]
        else :
            t = 0.5
        
        # get the mutations generation. The mutations are also individuals can join crossover step
        mutations = self.individuals.apply(func = lambda x : x.mutate(t))
        self.individuals = self.individuals.append(pd.Series(mutations)).reset_index(drop=True)
        self.logger.debug("MUTATION -- population: {}; generation: {}".format(self.population(), self.generations()))   
        # get the crossover generation
        func = self.crossover.run if func is None else func 
        offsprings = func(self.individuals)
        self.individuals = self.individuals.append(pd.Series(offsprings)).reset_index(drop=True)
        self.logger.debug("XOVER -- population: {}; generation: {}".format(self.population(), self.generations()))
        return mutations.append(pd.Series(offsprings)).reset_index(drop=True)

    def sort(self, func=None) :
        self.df.sort_values(by="fitness",ascending=False,inplace=True)
        return self.df

    def max(self) :
        if self.fitness is None :
            raise Exception("fitness has not been calculated")
        return self.fitness.max()

    def sum(self) :
        if self.fitness is None :
            raise Exception("fitness has not been calculated")
        return self.fitness.sum()

    def size(self) :
        return len(self.individuals)
    
    def population(self) :
        return len(self.individuals)

    def generations(self) :
        return max(self.individuals.apply(func=lambda x : x.generation))
    
    def diversity(self) :
        return 0 #len(set(self.individuals.values.tolist()))

    def copy(self) :
        return deepcopy(self)
    
    def top(self,k=1) :
        df_sorted = self.df.sort_values(by="fitness",ascending=False)
        return df_sorted.iloc[:k,:]
    
    def topD(self,k=1) :
        df_sorted = self.df.drop_duplicates(['individuals']).sort_values(by="fitness",ascending=False)
        return df_sorted['individuals'][:k]

    # def __str__(self) :
    #     return str(individuals)


        
class Environment():
    '''
    The environment is the central controller of the algorithm configuration and logic.  
    The initialization parameters contains parameters to form the Species instance and some for process control. 

    Parameters
    ------------
    selector: Selector. 
        It defines how the method how the environment keeps the populations.   
    crossover: CrossOver. 
        It defines how two individual exchange chromosome and reproduce next generations.  
    fitness_func: Fitness. 
        It defines how the individual is fitting for survival. Selector usually keeps the high fitness individuals.   
    mut_rate: float
        The new generation will have some rate to mutate. This will involve new values or patterns that never appeared in old generation.
    CAPACITY: int
        maximum number of individuals that an environment can hold. If the individuals growed unlimited, they would be culled by __punish__ 
    MAX_ITERATION: int
        maximum iterations of algorithm. The algorithm will stop when iteration reaches this number. 
    MAX_GENERATION: int
        maximum generation. The algorithm will stop when max generation in the species reaches this number.

    Methods
    ------------
    evolve: 
        Everything is setup at initialization, call env.evolve to run the algorithm. 
    getSolution:
        Find top K fitness solutions. If the algorithm has not been evolved, the 
    '''
    def __init__(self, individuals=[],
     selector=LinearRankingSelector(),
     crossover=SinglePointCrossOver(),
     fitness_func=SumFitness(),
     *args, **kwargs) :
        '''
        Initialize configurations and create a Species instance. Species is a set of individuals and interact with environment or operate on the set of individuals.
        '''

        self.CAPACITY = kwargs.get("CAPACITY", 1000)
        self.MAX_GENERATION = kwargs.get("MAX_GENERATION", 1000)
        self.MAX_ITERATION  = kwargs.get("MAX_ITERATION", 1000)
        self.mut_rate = kwargs.get("mut_rate", 0.1)
        self.epsilon = kwargs.get("epsilon", 1e-5)
        self.EXP_FITNESS = kwargs.get("EXP_FITNESS",1e6)
        self.verbose = kwargs.get('verbose', 0)
        # self.sel_rate = kwargs["sel_rate"] if "sel_rate" in kwargs.keys() else 0.5

        self.species = Species(
            individuals, 
            selector, 
            crossover, 
            fitness_func, 
            log_level='debug' if self.verbose>0 else 'info'
        )

    def __punish__(self) :
        pun = LinearRankingSelector(self.species.selector.constraints)
        # pun = LeadingSelector(0.3,self.species.selector.constraints)
        self.species.select(func=pun.select,verbose=False)
        # print("{} INFO: -- PUNISHMENT -- population: {}".format(self.species.population()))

    def evolute(self) :
        # last_fitness = -1e9
        for i in range(int(self.MAX_ITERATION)) :
            if i % 5 == 0 :
                self.species.logger.info("ITERATION START -- : {}".format(i))
            
            # calculate fitness
            self.species.calculate_fitness()
            # select from the fit individuals
            self.species.select()
            
            # if the population is beyond environment's capability, the environment will punish the species, forcing the non-opitma disappeared.
            while self.species.population() >= self.CAPACITY :
                self.__punish__()
            self.species.logger.debug("PUNISHMENT -- population: {}, diversity:{}".format(self.species.population(), self.species.diversity()) )

            if self.species.max() >= self.EXP_FITNESS :
                self.species.logger.info("EXP_FITNESS Achieved -- top fitness: {}".format(self.species.max()))
                break
            # Stop evolution if the generation has been exceeded MAX_GENERATION
            if self.species.generations() > self.MAX_GENERATION :
                self.species.logger.info("MAX_GENERATION Achieved -- top fitness: {}".format(self.species.max()))
                break
            
            # Reproduce next generation, including mutation and crossover
            self.species.reproduce(t=self.mut_rate)
            # Check or make the population be feasible under the pre-defined constraints
            self.species.feasible()
            
            # Stop evolution if total/maximum fitness converges
            # if self.species.sum() - last_fitness <= self.epsilon :
            #     break
            # else :
            #     last_fitness = self.species.sum()

            
        # self.species.select()

    def getSolution(self,k=1) :
        return self.species.topD(k)

    def evaluate(self) :
        # fitness, revenue, risk, cov = 
        return self.species.fitness_func.run(self.species.individuals)
