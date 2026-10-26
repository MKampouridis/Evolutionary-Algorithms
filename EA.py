# GENERIC EVOLUTIONARY ALGORITHM ABSTRACT CLASS. CAN BE USED FOR BOTH GENETIC ALGORITHMS AND GENETIC PROGRAMMING. THE
# FUNCTIONS DEFINED HERE CAN BE USED BY ANY EVOLUTIONARY ALGORITHM. WE ALSO DEFINE METHODS, SUCH AS CROSSOVER AND
# MUTATION, WHICH ARE ALGORITHM-SPECIFIC (I.E. THERE IS A NEED FOR A DIFFERENT IMPLEMENTATION OF THESE METHODS IN A GA
# AND A DIFFERENT IN A GP).
from abc import ABC, abstractmethod
from enum import Enum, auto
from statistics import mean
from random import random, randint
from Auxiliary import FileWriter
import numpy as np

# from GP.Tasks.FCFSymbolicRegression import FCFSymbolicRegression


class Problem(Enum):
    MAXIMISATION = auto()
    MINIMISATION = auto()


class EA(ABC):

    def __init__(self, problem):
        self.problem = problem

    class Individual:
        fitness = 0

        def __init__(self, model):
            self.model = model

    @abstractmethod
    def initialise_population(self, *args):
        pass

    @abstractmethod
    def crossover(self, *args):
        pass

    @abstractmethod
    def mutate(self, *args):
        pass

    @abstractmethod
    def evaluate(self, *args):
        pass

    # Tournament selection
    def tournament(self, TOURNAMENT_SIZE, population):
        tournament_participants = []  # Initialise empty list of random tournament participants.
        for i in range(TOURNAMENT_SIZE):  # select population individuals at random for tournament selection
            random_index = randint(0, len(population)-1)  # Random index to obtain a tournament participant
            tournament_participants.append(population[random_index])  # Place this individual in the list

        tournament_participants.sort(key=lambda x: x.fitness, reverse=False) if self.problem == Problem.MINIMISATION \
            else tournament_participants.sort(key=lambda x: x.fitness, reverse=True)

        return tournament_participants[0].model  # Return the individual with the best fitness


    # Main evolutionary process
    def evolve(self, nGENS, population, tournament_size, xover_prob, current_run, dataset, flag):
        FileWriter.save_logger("Log", current_run)
        for g in range(0, nGENS):
            print("---------Generation ", g, "-----------")
            for individual in population:
                self.evaluate(individual, dataset, flag)  # Calculate fitness for each  individual in the population
            # Sort them by fitness value.
            population.sort(key=lambda x: x.fitness, reverse=False) if self.problem == Problem.MINIMISATION \
                else population.sort(key=lambda x: x.fitness, reverse=True)  # Sort population by fitness, best on top
            mean_pop_fitness = mean([p.fitness for p in population])
            FileWriter.generation_printouts(  # Print results in the terminal and save them in the log file
                best_fitness=population[0].fitness, mean_fitness=mean_pop_fitness,
                worse_fitness=population[-1].fitness, current_generation=g, current_run=current_run,
                best_model=population[0].model)

            if g < nGENS - 1:  # Breeding. Evolution happens up to the generation before the last.
                # Elitism; copy the best individual into the next generation
                intermediate_pop = [population[0].model]

                for i in range(1, len(population)):  # Evolutionary process.
                    parent1 = EA.tournament(self, tournament_size, population)
                    r = random()
                    if r < xover_prob:  # Crossover
                        parent2 = self.tournament(tournament_size, population)
                        child = self.crossover(parent1, parent2)
                        intermediate_pop.append(child)
                    else:  # Mutation
                        intermediate_pop.append(self.mutate(parent1))

                for individual, intermediate in zip(population, intermediate_pop):
                    individual.model = intermediate  # Copy the intermediate population into the actual population list

    # Main method to run the GA
    def run(self, population_size, xover_prob, tournament_size, no_of_gens, no_of_runs, training_dataset, test_dataset,
            class_object, caller_class='', injected_individuals = None):
        FileWriter.log_experimental_setup(population_size, xover_prob, tournament_size, no_of_gens)
        FileWriter.create_results_files(class_object)

        # All Tasks apart from FCFSymbolicRegression go next
        if not str(caller_class).__contains__("FCFSymbolicRegression"):
            for n in range(no_of_runs):
                print("--------------- Run ", n, " ----------------")
                population = (self.initialise_population() if injected_individuals is None
                              else self.initialise_population(injected_individuals))
                self.evolve(no_of_gens, population, tournament_size, xover_prob, n, training_dataset, flag=0)
                FileWriter.run_printouts(population[0], n, flag=0)
                self.evaluate(population[0], test_dataset, flag=1)
                FileWriter.run_printouts(population[0], n, flag=1)
        else: # This is for FCFSymbolicRegression, as it uses an ***expanding window approach***
            for n in range(no_of_runs):
                print("--------------- Run ", n, " ----------------")
                population = self.initialise_population()
                training_subset = np.copy(training_dataset)

                # Train per window and make a prediction on a single (the next) data point on the test set; then
                # expand training window and at the same time remove that single point from the test set.
                for k in range(len(test_dataset)):
                    test_subset = test_dataset[k]
                    self.evolve(no_of_gens, population, tournament_size, xover_prob, n, training_subset, flag=0)
                    self.evaluate(population[0], test_subset[:, np.newaxis].T, flag=2)
                    training_subset = np.vstack((training_subset, test_dataset[k]))
                    print("****************************************", k)

                # After training across all windows is complete, calculate the average error across the test set
                # predictions on the different windows
                FileWriter.run_printouts(population[0], n, flag=0)
                self.evaluate(population[0], test_dataset, flag=1)
                FileWriter.run_printouts(population[0], n, flag=1)

