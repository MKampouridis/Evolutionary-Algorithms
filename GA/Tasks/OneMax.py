# OneMax PROBLEM IMPLEMENTATION. ESSENTIALLY THE ONLY THING WE NEED TO DO IS DEFINE THE PROBLEM-SPECIFIC FITNESS
# FUNCTION. ALL OTHER GA FUNCTIONS (E.G. CROSSOVER, MUTATION) ARE WRITTEN IN THE PARENT CLASSES EA AND GA. WE CAN OF
# COURSE OVER-WRITE THEM HERE, IF WE WANT A DIFFERENT IMPLEMENTATION, E.G. A DIFFERENT CROSSOVER OPERATOR.
from EA import Problem
from GA.GA import GA
from random import seed
import numpy as np

seed(111)  # Using a seed for reproducibility purposes, as GA is a stochastic algorithm.
nRUNS = 1
POPULATION_SIZE = 10
nGENS = 5  # Number of generations
GENES = 10  # Number of genes (cells) in each chromosome
XOVER_PROB = 0.75
TOURNAMENT_SIZE = 2
GENES_TYPE = "int"  # Define the type of the GA genes, if e.g. they will be int or float
LB = 0  # Lower boundary (inclusive) of the potential gene values that are generated
UB = 1  # Upper boundary (inclusive) of the potential gene values that are generated
PROBLEM = Problem.MAXIMISATION  # Setup as maximisation problem


class OneMax(GA):

    def __init__(self):
        super().__init__(type_of_genes=GENES_TYPE, lower_bound=LB, upper_bound=UB, pop_size=POPULATION_SIZE,
                         no_of_genes=GENES, problem=PROBLEM)

    # Fitness function: sum of genes. The dataset and flag argument is not needed in the OneMax problem.
    def evaluate(self, individual, dataset=None, flag=None):
        individual.fitness = np.array(individual.model).sum()


# Runs the EA algorithm over a number of independent runs and prints and saves results.
if __name__ == '__main__':
    task = OneMax()
    task.run(population_size=POPULATION_SIZE, xover_prob=XOVER_PROB, tournament_size=TOURNAMENT_SIZE,
             no_of_gens=nGENS, no_of_runs=nRUNS, training_dataset=None, test_dataset=None,
             class_object=OneMax.Individual(""))
