import numpy as np 
from copy import copy , deepcopy
from collections import Iterable, Collection
import pandas as pd
import random
from abc import ABC, abstractmethod
 
class Individual(ABC) :
    class KeyValueError(ValueError) :
        pass

    def __init__(self, *args, **kwargs) :
        if args is not None and isinstance(args[0], Individual) and len(args) == 1:
            tmp = args[0]
            self.chromosome = tmp.chromosome.copy()
            self.generation = tmp.generation
            self.age = tmp.age
            self.fitness = 0
        elif len(args) + len(kwargs) >= 3 :
            self.chromosome = args[0] if args[0] is not None else kwargs["chromosome"]
            self.generation = args[1] if args[1] is not None else kwargs["generation"]
            self.age = args[2] if args[2] is not None else kwargs["age"]
            self.fitness = 0
        else :
            raise Exception("non sufficient arguments")

    def __str__(self) :
        return str(self.chromosome) 
 
    def __setitem__(self,k,v) :
        pass
 
    def __getitem__(self,k) :
        pass
 
    def __add__(self, another) :
        pass
    
    def __truediv__(self, div) :
        pass
    def __mul__(self, mul) :
        pass

class BinaryIndividual(Individual) :
    '''
    Individual form for Binary Combinational Optimization problem. 
    eg. Whether to put i-th item into the bag.
    '''
    class KeyValueError(Exception) :
        def __init__(self, err_msg) :
            self.err_msg = err_msg
        def __str__(self) :
            str(self.err_msg)

    def __init__(self, *args, **kwargs) :
        if args is not None and isinstance(args[0], Individual) and len(args) == 1:
            tmp = args[0]
            self.chromosome = tmp.chromosome.copy()
            self.generation = tmp.generation
            self.age = tmp.age
            self.fitness = 0
        elif len(args) + len(kwargs) >= 3 :
            self.chromosome = args[0] if args[0] is not None else kwargs["chromosome"]
            self.generation = args[1] if args[1] is not None else kwargs["generation"]
            self.age = args[2] if args[2] is not None else kwargs["age"]
            self.fitness = 0
        else :
            raise Exception("non sufficient arguments")
    def mutate(self, t = 0.1, prob=None) :
        if prob is None :
            prob = np.random.rand(len(self.chromosome))
        
        factor = prob < t
        xor = lambda x, y : (x != y).astype(np.int32)

        chromosome = xor(np.array(self.chromosome), factor)
        generation = self.generation #+ 1
        age = 0
        
        self.grow()

        return BinaryIndividual(chromosome.tolist(), generation, age)

    def grow(self) :
        self.age += 1
    
    def __setitem__(self,k,v) :
        if type(k) == int :
            self.chromosome[k] = v
        elif type(k) == slice :
            self.chromosome[k] = v
        elif isinstance(k, Collection) :
            s = pd.Series(self.chromosome)
            s[k] = v
            self.chromosome = s.values.tolist()

        else : 
            raise Individual.KeyValueError("Cannot set chromosome with a key type is not int, slice or Collection")


    def __getitem__(self,k) :
        if type(k) == int :
            return BinaryIndividual(self.chromosome[k],self.generation,self.age)
        elif type(k) == slice :
            return BinaryIndividual(self.chromosome[k],self.generation,self.age)
        elif isinstance(k, Collection) :
            s = pd.Series(self.chromosome)
            return BinaryIndividual(s.values.tolist(), self.generation, self.age)
        
        else :
            raise Individual.KeyValueError("Cannot get chromosome with a key type is not int, slice or Collection")

    def __str__(self) :
        return str(self.chromosome)
    # def __eq__(self, obj) :
    #     return self.chromosome == obj.chromosome
    def __add__(self, another) :
        # implement concatenation of two chromosome so it is not mutable addition.
        chromosome = self.chromosome + another.chromosome
        generation = self.generation + 1
        age = 0
        
        self.grow()
        another.grow()

        return BinaryIndividual(chromosome, generation, age)

    def __truediv__(self, div) :
        # div : a number or a np.ndarray
        chromosome = (np.array(self.chromosome) / div).astype(float).tolist()
        generation = self.generation 
        age        = self.age

        return BinaryIndividual(chromosome, generation, age)
    def __mul__(self, mul) :
        # mul : a number or a np.ndarray
        chromosome = (np.array(self.chromosome) * mul).astype(float).tolist()
        generation = self.generation 
        age        = self.age

        return BinaryIndividual(chromosome, generation, age)

    def indexOf(self, vals) :
        if isinstance(vals, Collection) :
            indices = []
            for v in vals :
                indices.append(self.chromosome.index(v))
            return indices
        else :
            indices = []
            for i in range(len(self.chromosome)) :
                if self.chromosome[i] == vals :
                    indices.append(i)
            return indices
    def indexOfPositive(self) :
        indices = []
        for i in range(len(self.chromosome)) :
            if self.chromosome[i] > 0 :
                indices.append(i)
        return indices

    def copy(self) :
        return deepcopy(self)

    def sum(self) :
        return sum(self.chromosome)

    def __len__(self) :
        return len(self.chromosome)

    # def __eq__(self, another) :
    #     return "".join(self.chromosome) == "".join(another.chromosome)


class IntegerIndividual(Individual) :
    '''
    Individual form for Integer Combinational Optimization problem.  
    eg. Assign M people for i-th task.
    '''
    def __init__(self, *args, **kwargs) :
        Individual.__init__(self, *args, **kwargs)
        if "domain" not in kwargs : 
            raise ValueError("")
        self.domain = kwargs["domain"] # a list of integers 
        self.upper = max(self.domain)
        self.lower = min(self.domain)

    def mutate(self, t = 0.1, prob=None) :
        
        # if prob is None :
        #    prob = random.sample(self.domain, 1)[0]

        chromosome = self.chromosome.copy()
        chr_len = len(self.chromosome)
        if prob is None :
            prob = np.random.rand(chr_len)

        factor = prob < t 
        for i in range(chr_len) :
            if factor[i] :
                chr_lst = self.domain.copy()
                chr_lst.remove(chromosome[i])
                chromosome[i] = random.choice(chr_lst)
  
        generation = self.generation + 1

        age = 0
        
        self.grow()

        return IntegerIndividual(chromosome, generation, age, domain=self.domain)
    
    def __getitem__(self,k) :
        if type(k) == int :
            return self.chromosome[k] #IntegerIndividual(self.chromosome[k],self.generation,self.age, domain=self.domain)
        if type(k) == slice :
            # return self.chromosome[k] 
            return IntegerIndividual(self.chromosome[k],self.generation,self.age, domain=self.domain)
        elif isinstance(k, Collection) :
            s = pd.Series(self.chromosome)
            return IntegerIndividual(s.values.tolist(), self.generation, self.age)
        
        else :
            raise Individual.KeyValueError("Cannot get chromosome with a key type is not int, slice or Collection")

