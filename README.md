# hello-genetic-algorithm
A basic implementation of genetic algorithm. It fits for small scale experiments like college labs or GA beginners.
  
Use this command to install  
`pip install hello-genetic-algorithm`

## Algorithm Parameters  
- Essential parameters (must set up by user)
    - Individual class 
    - Fitness function
- Optional parameters (with default value if not set) 
    - Mutation ratio
    - Crossover
    - Constraints
    - Selection ratio
- Execution parameters (config stop criteria or logging level)
    - Max iteration
    - Max generation
    - Environment Capacity
    - Logging level
## Introduction by example
### Formulate the Problem 
Assume we have 12 goods to be packed into a travel suitcase. The suitcase can hold at list 250 kg goods. We have measured the weights for each of the goods and define importance so that we should packed more important goods as much as possible in the limit of weight constraint.   

Define the chromosome as a 0-1 vector to represent each box. The problem is to find out a vector that maximize sum of box ***I***mportance where sum of box ***W***eights are less than or equal to 250. eq.   
$$ \max \sum_{i}{I_i} \quad where \quad i \in \{0, \dots, 11\}  $$
$$ s.t. \sum_{i}{W_i} <= 250 $$
### Formulation to parameters
This formulation can be mapping to
- a `BinaryIndividual` class to represent whether to select a goods
- a predefined `WeightedSumFitness` :
- a user-defined `constraints` : total weights of goods should be less or equal to 250. 

### Algorithm Steps
1. Initialize: input parameters and create algorithm instance
1. Calculate fitness: calculate fitness value for every individual
1. Select: keep only the individuals fulfill the constrants
1. Reproduce: generate new individuals by mutation and crossover operator
1. Exit criteria: check if stop criteria is fulfilled. If yes, stop the progress, otherwise repeat from step 2.  

### Example Code
```py
from helloga.environment import Environment
from helloga.individual import BinaryIndividual 
from helloga.crossover import SinglePointCrossOver
from helloga.selector import LeadingSelector
from helloga.fitness import WeightedSumFitness

# Define fitness importence and contraint weights 
box_importance = [6, 5, 8, 7, 6, 9, 4, 5, 4, 9, 2, 1]
box_weights = [20, 30, 60, 90, 50, 70, 30, 30, 70, 20, 20, 60]

# Define constraints
def total_size_lt250(individual, size=np.array([])) :
    def total_size(individual, size=np.array([])) :
        chr_arr = np.array(individual.chromosome)
        siz_arr = np.array(size)
        total = np.dot(chr_arr, siz_arr.T)
        return total 

    total = total_size(individual, size)
    return total <= 250

# Initialize by random individuals.
# Each element in the vector represents whether to select the i-th goods or not
individuals = [ 
    BinaryIndividual([1,1,1,0,0,0,0,0,0,0,0,1],0,0),
    BinaryIndividual([1,0,0,0,1,0,0,0,0,0,0,1],0,0),
    BinaryIndividual([0,0,0,0,0,1,1,0,0,1,0,0],0,0),
    BinaryIndividual([0,0,1,0,0,0,0,0,1,0,0,1],0,0),
    BinaryIndividual([0,1,0,0,1,0,0,0,0,0,0,1],0,0),
]    

# Define selector
sel = LeadingSelector(
    ratio = 0.5,
    constraints=[lambda x: total_size_lt250(x, box_weights)]
)

# Define fitness function
fit = WeightedSumFitness(weights = box_importance)

# Define selector
xo = SinglePointCrossOver()

# Create environment
env = Environment(
    individuals,
    selector=sel,
    crossover=xo, 
    fitness_func=fit,
    MAX_GENERATION=50,
    CAPACITY=100, 
    MAX_ITERATION=100,
    log_level='info'
)

# Run genetic algorithm
env.evolute()

# Print population and generation after the algorithm stopped
print(env.species.population(), env.species.generations())

# Print best 3 solutions to the problem
print('The best 3 solutions are: ')
for sol in env.getSolution(3) :
    print(sol) 

```