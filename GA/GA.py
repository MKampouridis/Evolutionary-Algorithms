# GENETIC ALGORITHM METHODS.
from abc import ABC, abstractmethod
from random import randint, uniform
from EA import EA
import numpy as np


class GA(EA, ABC):
    def __init__(self, type_of_genes, lower_bound, upper_bound, pop_size, no_of_genes, problem):
        self.type_of_genes = type_of_genes
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.pop_size = pop_size
        self.no_of_genes = no_of_genes
        super().__init__(problem)

    @abstractmethod
    def evaluate(self, *args):
        pass

    # Generic random population initialisation. Creates either integers or floats, depending on the given type_of_genes.
    def initialise_population(self, *args):
        chromosomes = []

        if not args or args[0] is None:
            args = ()
        else:
            for list in args:
                chromosomes.extend(list)

        if self.type_of_genes == "int":
            chromosomes += [[randint(self.lower_bound, self.upper_bound) for _ in range(self.no_of_genes)] for _
                            in range(self.pop_size - len(chromosomes))]
        elif self.type_of_genes == "float":
            chromosomes = [[uniform(self.lower_bound, self.upper_bound) for _ in range(self.no_of_genes)] for _
                           in range(self.pop_size)]
        else:
            print("SHOULD NOT REACH HERE!")
            exit()
        population = [self.Individual(chromosome) for chromosome in chromosomes]

        return population

    # One-point crossover
    def crossover(self, parent1, parent2):
        xover_point = randint(1, len(parent1) - 1)
        child = []
        for i in range(0, xover_point):
            child.append(parent1[i])
        for i in range(xover_point, len(parent2)):
            child.append(parent2[i])

        return child

    # Point mutation. Differentiates between integer and float gene types.
    def mutate(self, parent):
        mutation_point = randint(0, len(parent) - 1)
        parent_copy = parent.copy()
        if self.type_of_genes == "int":
            parent_copy[mutation_point] = randint(self.lower_bound, self.upper_bound)
        elif self.type_of_genes == "float":
            parent_copy[mutation_point] = uniform(self.lower_bound, self.upper_bound)

        return parent_copy

