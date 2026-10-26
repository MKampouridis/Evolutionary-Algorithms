# Port.Optimisation PROBLEM IMPLEMENTATION. ESSENTIALLY THE ONLY THING WE NEED TO DO IS DEFINE THE PROBLEM-SPECIFIC
# FITNESS FUNCTION. ALL OTHER GA FUNCTIONS (E.G. CROSSOVER, MUTATION) ARE WRITTEN IN THE PARENT CLASSES EA AND GA. WE
# CAN OF COURSE OVER-WRITE THEM HERE, IF WE WANT A DIFFERENT IMPLEMENTATION, E.G. A DIFFERENT CROSSOVER OPERATOR.
from EA import Problem
from GA.GA import GA
import numpy as np
import pandas as pd
from random import seed, randint, uniform

seed(111)  # Using a seed for reproducibility purposes, as GA is a stochastic algorithm.
nRUNS = 50
POPULATION_SIZE = 50
nGENS = 50  # Number of generations
XOVER_PROB = 0.75
TOURNAMENT_SIZE = 4
GENES_TYPE = "float"  # Define the type of the GA genes, if e.g. they will be int or float
LB = 0  # Lower boundary (inclusive) of the potential gene values that are generated
UB = 1  # Upper boundary (inclusive) of the potential gene values that are generated
PROBLEM = Problem.MAXIMISATION  # Setup as maximisation problem
data = np.array(pd.read_csv("returns.csv"))
expected_returns = np.mean(data, axis=0)
GENES = data.shape[1]  # Number of genes (cells) in each chromosome


class PortfolioOptimisation(GA):

    def __init__(self):
        super().__init__(type_of_genes=GENES_TYPE, lower_bound=LB, upper_bound=UB,
                         pop_size=POPULATION_SIZE, no_of_genes=GENES, problem=PROBLEM)

    # Problem-specific fitness function calculation. Flag is not used here.
    def evaluate(self, individual, dataset, flag=None):
        # Some explanations: 1. we do np.array(individual.model), because after xover the individuals end up as list
        # Normally there's no problem, but we cannot do the matrix multiplication for lists, so we convert it to array.
        # 2. We reshape both the weights (individual.model) and the returns arrays, so that their dimensions are
        # compatible for the matrix multiplication. 3. In this simple example, fitness is the return of the portfolio.
        return_per_individual = np.matmul(np.array(individual.model).reshape(1, len(individual.model)),
                                          dataset.reshape(len(dataset), 1))  # weights*returns = the individual's return
        individual.fitness = return_per_individual[0, 0]  # Obtain the value from the single-element array

    # One-point crossover. Overwrites parent method to ensure that weights are always normalised in the [0, 1) range.
    # Strictly speaking, I should have done the same during population initialisation. But it doesn't really matter,
    # as even without normalisation, the weights still show preferences between assets. And during crossover/mutation,
    # I normalise the whole population, so from generation 1 onwards we have normalised weights.
    def crossover(self, parent1, parent2):
        xover_point = randint(1, len(parent1) - 1)
        child = []
        for i in range(0, xover_point):
            child.append(parent1[i])
        for i in range(xover_point, len(parent2)):
            child.append(parent2[i])

        return child / np.sum(child)  # weights normalisation

    # Point mutation. Overwrites parent method to ensure that weights are always normalised in the [0, 1) range.
    def mutate(self, parent):
        mutation_point = randint(0, len(parent) - 1)
        parent_copy = parent.copy()
        parent_copy[mutation_point] = uniform(self.lower_bound, self.upper_bound)

        return parent_copy / np.sum(parent)  # weights normalisation


# Runs the EA algorithm over a number of independent runs and prints and saves results.
if __name__ == '__main__':
    task = PortfolioOptimisation()
    task.run(population_size=POPULATION_SIZE, xover_prob=XOVER_PROB, tournament_size=TOURNAMENT_SIZE,
             no_of_gens=nGENS, no_of_runs=nRUNS, training_dataset=expected_returns, test_dataset=expected_returns,
             class_object=PortfolioOptimisation.Individual(""))
    # DUMMY TEST DATASET HAS BEEN PASSED ABOVE. SHOULD REPLACE IT WITH AN ACTUAL TEST DATASET.
