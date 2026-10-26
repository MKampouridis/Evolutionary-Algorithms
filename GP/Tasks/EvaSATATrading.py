from Auxiliary import FileWriter
from EA import Problem, EA
from GP.GP import GP, Util
import numpy as np
import pandas as pd
from random import seed, random, randint, uniform
from statistics import mean, stdev
from multipledispatch import dispatch

seed(111)
problem = Problem.MAXIMISATION  # Setup the current problem as a maximisation one (i.e. maximise the fitness value).
terminalNodeCrossBias = 0.1  # Probability to select a terminal node during crossover. Value given by Koza (1992).
nRuns = 1  # Individual and independent GP runs
INITIAL_DEPTH = 2  # Initial tree depth during population initialisation
MAX_DEPTH = 4  # Maximum tree depth
POP_SIZE = 50  # Population size
XOVER_PROB = 0.85  # Crossover probability. Mutation probability is (1 - XOVER_PROB)
FLIP_MUT_PROB = 0.5  # Probability of selecting a tree node for mutation during the point mutation operator
TOURNAMENT_SIZE = 10  # Tournament size
nGENS = 5  # Number of generations
ERC_LB = -1  # Lower bound for ERC numbers
ERC_UB = 1  # Upper bound for ERC numbers
THETA = 0.05  # Threshold used as part of the fitness function. Look in there for more details.
MIN_NO_OF_TRADES = 6  # Forces a minimum number of trades during trading.
TYPE = "Predicate"  # Options are: Predicate, Arithmetic, and Constrained_Predicate. The latter forces types
# in tree branches, i.e. left branch is for Sentiment Analysis, while right branch is for Technical Analysis.
current_run = 0  # Tracks the current GP run
training_data = np.array(pd.read_csv("./AAPL_Tr.csv"))  # Read training file
test_data = np.array(pd.read_csv("./AAPL_V.csv"))  # Read test file
F_SET = ["AND", "SA_AND", "TA_AND", "SA_OR", "TA_OR", "SA_GT", "TA_GT", "SA_LT", "TA_LT"]  # Function Set
# Number of variables  in the terminal set depends on number of features in the excel input file
# ERC: Ephemeral Random Constant; it should always be in the last place of the array
T_SET = ["Var0", "Var1", "Var2", "Var3", "Var4", "Var5", "Var6", "Var7", "ERC"]
grammar = {  # Acts as a lookup type table, finding what are the compatible types and their values
    "Root": ["AND"],  # Force the root of the trees to be one of these functions
    "AND": F_SET,  # These are the compatible types when the parent node is AND
    "SA_AND": ["SA_AND", "SA_OR", "SA_GT", "SA_LT"],  # SA-related functions compatible with type SA_AND
    "SA_OR": ["SA_AND", "SA_OR", "SA_GT", "SA_LT"],
    "TA_AND": ["TA_AND", "TA_OR", "TA_GT", "TA_LT"],  # TA-related functions compatible with type TA_AND
    "TA_OR": ["TA_AND", "TA_OR", "TA_GT", "TA_LT"],
    "SA_LT": ["Var0", "Var1", "Var2", "Var3", "ERC"],  # Place here all SA-related indicators compatible with SA_LT
    "SA_GT": ["Var0", "Var1", "Var2", "Var3", "ERC"],  # As above
    "TA_LT": ["Var4", "Var5", "Var6", "Var7", "ERC"],  # Here all the TA-related indicators
    "TA_GT": ["Var4", "Var5", "Var6", "Var7", "ERC"]  # As above
}

types_lookup = {  # Defines the types of each node; the function set is booleans, whereas the terminal set is arithmetic
    "Boolean": F_SET,
    "Arithmetic1": ["Var0", "Var1", "Var2", "Var3"],  # All SA-related indicators
    "Arithmetic2": ["Var4", "Var5", "Var6", "Var7"],  # All TA-related indicators
    "Arithmetic3": ["ERC"]  # Dummy. When needed ERC is going to be replaced by a numeric value. But I want to have the
    # Arithmetic3 key, which I will use later in the crossover.
}


class EvaSATATrading(GP):
    # Properties of an individual. An individual is a trading strategy represented as a prefix expression.
    class Individual(EA.Individual):  # Extends the EA Individual class
        no_of_trades = 0
        sharpe_ratio = 0
        rate_of_return = 0
        risk = 0  # standard deviation of returns
        signal = []  # array holding the predictions of the individual

    def __init__(self):
        super().__init__(POP_SIZE, INITIAL_DEPTH, MAX_DEPTH, grammar, types_lookup, F_SET, T_SET, terminalNodeCrossBias,
                         FLIP_MUT_PROB, ERC_LB, ERC_UB, TYPE, problem)

    # Generates trees that have different types on the left branch (SA) and on the right branch (TA). Overrides parent
    # method.
    def generate_tree(self, current_depth, expression, node, method):
        # Following if-statement as per Poli's "A Field Guide to Genetic Programming", p.14: If grow method,
        # choose a leaf when a certain probability occurs
        terminal_prob = len(self.terminal_set) / (len(self.terminal_set) + len(self.function_set))
        no_of_children = 2  # All current functions have an arity of 2. This can of course change in the future
        # if other functions (e.g. NOT) are added.
        if node in self.terminal_set:
            no_of_children = 0  # Terminal sets, no children
            if node == "ERC":
                expression.append(uniform(self.erc_lb, self.erc_ub))  # Random number uniformly distributed
            else:
                expression.append(node)
        else:
            expression.append(node)

        potential_children = self.grammar.get(node)
        child = ""

        if node == "AND":  # Force left branch with SA and right branch with TA
            potential_children = [x for x in potential_children if "TA" not in x and x != "AND"]  # Obtain SA-related
            # list of functions
            child = potential_children[randint(0, len(potential_children) - 1)]
            self.generate_tree(current_depth - 1, expression, child, method)

            potential_children = self.grammar.get(node)
            potential_children = [x for x in potential_children if "SA" not in x and x != "AND"]  # TA list of functions
            child = potential_children[randint(0, len(potential_children) - 1)]
            self.generate_tree(current_depth - 1, expression, child, method)
        elif "LT" in node or "GT" in node:
            # Force first branch to be a non-ERC terminal
            child = potential_children[randint(0, len(potential_children) - 2)]
            self.generate_tree(current_depth - 1, expression, child, method)
            child = "ERC"  # Force second branch to be ERC
            self.generate_tree(current_depth - 1, expression, child, method)
        else:
            for i in range(no_of_children):
                if potential_children:  # checks that potential_children has elements (i.e, it is not empty)
                    child = potential_children[randint(0, len(potential_children) - 1)]

                # Need to have remaining depth of at least 2 to consider all functions; OR
                # if method is grow, and the node is not already GT/LT, then with a certain probability
                # choose LT/GT to end the tree.
                # Otherwise force LT or GT [unless we are already dealing with a terminal]
                if (current_depth < 2 and child not in self.terminal_set) or (
                        method == "grow" and current_depth != self.max_depth and
                        random() < terminal_prob and "LT" not in node and "GT" not in node):
                    if "SA" in str(node):  # Ensure when parent is SA that child is also SA
                        child = "SA_LT" if randint(0, 1) == 0 else "SA_GT"
                    else:
                        child = "TA_LT" if randint(0, 1) == 0 else "TA_GT"

                self.generate_tree(current_depth - 1, expression, child, method)

        return expression

    # Subtree crossover. Overrides parent method. Forces crossover on the SA-side of the tree. The crossover point is
    # an SA function (SA_AND, SA_OR, SA_GT, SA_LT). Crossover point will not happen on the terminals (Var, ERC), but
    # that's ok, because these will be taken care by the mutation operator.
    @dispatch(list, list, str)
    def crossover(self, parent1, parent2, branch_type):
        # Parent 1
        xo1_start = randint(1, len(parent1) - 2)  # crossover point from index 1 (not 0, thus excluding the root)

        # Select terminal node for crossover, under a certain probability; OR when length is 3, ie tree depth is 1; in
        # such cases there's only one function at the root of the tree. So there's no point for crossover at the root,
        # as we would be replacing the whole tree. So instead in these cases, which only usually happen in generation 0,
        # we force the crossover point to be a terminal.
        if random() < self.terminal_node_xover_bias or len(parent1) == 3:
            while parent1[xo1_start] in self.function_set:  # Ensure crossover point is a terminal
                xo1_start = randint(1, len(parent1) - 2)
        else:
            while parent1[xo1_start] not in self.function_set:  # Ensure crossover point is a function
                xo1_start = randint(1, len(parent1) - 2)

        root1 = parent1[xo1_start]

        # We force the crossover point to be an SA-related or a TA-related function, depending on the value of the
        # branch_type ("SA" or "TA").
        while branch_type not in str(root1):
            xo1_start = randint(1, len(parent1) - 2)
            root1 = parent1[xo1_start]

        # Parent 2
        xo2_start = randint(1, len(parent2) - 2)  # crossover point from index 1 (not 0, thus excluding the root)
        root2 = parent2[xo2_start]

        # Find the key that contains the value of root1 from the types_lookup table. The result is a single key, which
        # is saved in a list (root1_type or root2_type). Thus to obtain the element, we call root1_type[0]
        root1_type = [key for key, values in self.types_lookup.items() if root1 in values]
        if not root1_type and isinstance(root1,
                                         float):  # When ERC, a float appears in the tree, so no key was found.
            root1_type.append('Arithmetic3')  # When ERC, since it's a float, we set the type to Arithmetic3.
        root2_type = [key for key, values in self.types_lookup.items() if root2 in values]
        if not root2_type and isinstance(root2, float):
            root2_type.append('Arithmetic3')

        while root1_type[0] != root2_type[0] or (str(root1)[:2] != str(root2)[:2] and type(root1) is not float and
                                                 type(root2) is not float):  # We want the two roots to be of the
            # same (hence compatible) type. We also want the two roots to be of the same SA or TA type, so we check the
            # first two characters of the root name, which is always going to be SA or TA. We don't want them to be
            # floats, as this would indicate they're ERCs, and if both of them are ERCs, then the crossover is ok to go
            # ahead.
            xo2_start = randint(1, len(parent2) - 2)  # XOVER point from index 1 (not 0, thus excluding the root)
            root2 = parent2[xo2_start]
            root2_type = [key for key, values in self.types_lookup.items() if root2 in values]
            if not root2_type and isinstance(root2, float):
                root2_type.append('Arithmetic3')

        temp = Util.get_valid_subtree(parent1, xo1_start, self.function_set)  # subtree to be exchanged from parent1
        temp2 = Util.get_valid_subtree(parent2, xo2_start,
                                       self.function_set)  # subtree to be exchanged from parent2

        child = Util.copy_subtree(xo1_start, parent1, temp, temp2)  # Copy subtree from parent2 into parent1

        if Util.is_valid(child, self.function_set) is False:
            print("SHOULD NEVER REACH HERE!")
            exit()

        return child

    # Subtree crossover. Overrides parent method. Ensures legal operations for SA and TA types. There's only a single
    # crossover point in any given parent tree.
    @dispatch(list, list)
    def crossover(self, parent1, parent2):
        # Parent 1
        xo1_start = randint(1, len(parent1) - 2)  # crossover point from index 1 (not 0, thus excluding the root)
        # Select terminal node for crossover, under a certain probability; OR when length is 3, ie tree depth is 1; in
        # such cases there's only one function at the root of the tree. So there's no point for crossover at the root,
        # as we would be replacing the whole tree. So instead in these cases, which only usually happen in generation 0,
        # we force the crossover point to be a terminal.
        if random() < self.terminal_node_xover_bias or len(parent1) == 3:
            while parent1[xo1_start] in self.function_set:  # Ensure crossover point is a terminal
                xo1_start = randint(1, len(parent1) - 2)
        else:
            while parent1[xo1_start] not in self.function_set:  # Ensure crossover point is a function
                xo1_start = randint(1, len(parent1) - 2)

        root1 = parent1[xo1_start]

        # Parent 2
        xo2_start = randint(1, len(parent2) - 2)  # crossover point from index 1 (not 0, thus excluding the root)
        root2 = parent2[xo2_start]

        # Find the key that contains the value of root1 from the types_lookup table. The result is a single key, which
        # is saved in a list (root1_type or root2_type). Thus to obtain the element, we call root1_type[0]
        root1_type = [key for key, values in self.types_lookup.items() if root1 in values]
        if not root1_type and isinstance(root1,
                                         float):  # When ERC, a float appears in the tree, so no key was found.
            root1_type.append('Arithmetic3')  # When ERC, since it's a float, we set the type to Arithmetic3.
        root2_type = [key for key, values in self.types_lookup.items() if root2 in values]
        if not root2_type and isinstance(root2, float):
            root2_type.append('Arithmetic3')

        while root1_type[0] != root2_type[0] or (str(root1)[:2] != str(root2)[:2] and type(root1) is not float and
                                                 type(root2) is not float):  # We want the two roots to be of the
            # same (hence compatible) type. We also want the two roots to be of the same SA or TA type, so we check the
            # first two characters of the root name, which is always going to be SA or TA. We don't want them to be
            # floats, as this would indicate they're ERCs, and if both of them are ERCs, then the crossover is ok to go
            # ahead.
            xo2_start = randint(1, len(parent2) - 2)  # XOVER point from index 1 (not 0, thus excluding the root)
            root2 = parent2[xo2_start]
            root2_type = [key for key, values in self.types_lookup.items() if root2 in values]
            if not root2_type and isinstance(root2, float):
                root2_type.append('Arithmetic3')

        temp = Util.get_valid_subtree(parent1, xo1_start, self.function_set)  # subtree to be exchanged from parent1
        temp2 = Util.get_valid_subtree(parent2, xo2_start,
                                       self.function_set)  # subtree to be exchanged from parent2

        child = Util.copy_subtree(xo1_start, parent1, temp, temp2)  # Copy subtree from parent2 into parent1

        if Util.is_valid(child, self.function_set) is False:
            print("SHOULD NEVER REACH HERE!")
            exit()

        return child

    # Point mutation. Overrides parent method to ensure legal mutations (i.e. SA with SA, TA with TA)
    def mutate(self, expression):
        copy = expression.copy()  # create a copy
        for index in range(len(copy)):
            if random() < self.flip_mutation_probability:  # Apply point mutation at each node with probability
                s = copy[index]
                if s == "AND":  # AND can only be point-mutated to OR; OR only to AND; GT only to LT; LT only to GT.
                    copy[index] = "AND"  # Root node remains unchanged
                elif s == "SA_AND":
                    copy[index] = "SA_OR"
                elif s == "SA_OR":
                    copy[index] = "SA_AND"
                elif s == "SA_GT":
                    copy[index] = "SA_LT"
                elif s == "SA_LT":
                    copy[index] = "SA_GT"
                elif s == "TA_AND":
                    copy[index] = "TA_OR"
                elif s == "TA_OR":
                    copy[index] = "TA_AND"
                elif s == "TA_GT":
                    copy[index] = "TA_LT"
                elif s == "TA_LT":
                    copy[index] = "TA_GT"
                else:  # Terminal set cases
                    if "Var" not in str(s):  # No Var, thus ERC; we force ERC to be mutated with another ERC
                        copy[index] = uniform(self.erc_lb, self.erc_ub)  # Random number uniformly distributed
                    else:  # It's a Var. We force a Var to be mutated with another Var of the same type (SA or TA)
                        if s in grammar.get('SA_GT'):  # Both SA_GT and SA_LT have the same VarX children, so it doesn't
                            # matter if I use SA_GT or SA_LT
                            number = randint(0, len(self.grammar.get('SA_GT')) - 2)  # Get a random index, excluding ERC
                            while str(number) in s:  # Re-do this, if we ended up mutating to the same variable
                                number = randint(0, len(self.grammar.get('SA_GT')) - 2)
                            copy[index] = self.grammar.get('SA_GT')[number]
                        elif s in grammar.get('TA_GT'):  # Now do the same for the TA variables.
                            number = randint(0, len(self.grammar.get('TA_GT')) - 2)  # Get a random index, excluding ERC
                            while str(number) in s:  # Re-do this, if we ended up mutating to the same T_SET variable
                                number = randint(0, len(self.grammar.get('TA_GT')) - 2)
                            copy[index] = self.grammar.get('TA_GT')[number]
                        else:
                            print("SHOULD NEVER REACH HERE!")

        return copy

    # Fitness evaluation based on trading metrics. Flag is used to denote training or test dataset.
    @staticmethod
    def evaluate(individual, data, flag):
        global current_run

        rate_of_return = []
        q = 0  # Shares quantity. For now we allow a single quantity only.
        no_of_trades = 0
        price_at_buy = data[0, 0]  # initialise value to the first price in the time series
        signal = Util.interpret(individual.model,
                                data)  # boolean depending on the evaluation of the expression (tree)

        for i in range(len(data)):
            price = data[i, 0]  # Price is Column 0 in the dataset
            if signal[i]:  # If expression output is True, then take an action; otherwise take a different action.
                if q == 0:
                    q += 1  # Open position
                    no_of_trades += 1
                    price_at_buy = price
            else:
                # Ensure we have quantity to sell&that the sell price is greater than the purchase price by a threshold.
                if q == 1 and (price - price_at_buy) / price_at_buy > THETA:
                    q -= 1  # CLose position
                    no_of_trades += 1
                    rate_of_return.append((price - price_at_buy) / price_at_buy)

        individual.no_of_trades = no_of_trades
        if flag == 0:  # Training set calculation
            # Encourage individuals to close open positions (no_of_trades is an even number) & have a min no of trades
            individual.rate_of_return = mean(rate_of_return) \
                if (no_of_trades % 2 == 0 and no_of_trades >= MIN_NO_OF_TRADES) else 0
        else:  # Test set calculation
            individual.rate_of_return = mean(
                rate_of_return) if rate_of_return else 0  # Calculate mean if list not empty
            individual.signal = signal
            FileWriter.save_results_to_file("SignalsBestTest.txt", signal,
                                            current_run)  # Signals from best model in test set
            current_run += 1

        # Standard deviation calculation requires at least 2 values, which are derived from a minimum of 4 trades
        individual.risk = stdev(rate_of_return) if no_of_trades >= 4 else 0
        individual.sharpe_ratio = individual.rate_of_return / individual.risk \
            if individual.risk != 0 else individual.rate_of_return  # Sharpe ratio calculation
        individual.fitness = individual.sharpe_ratio  # Fitness function value


    # Main evolutionary process. I've had to override it to allow for the different crossover operators.
    def evolve(self, generations, population, tournament_size, xover_prob, run_number, dataset, flag):
        FileWriter.save_logger("Log", run_number)
        for g in range(0, generations):
            for individual in population:
                self.evaluate(individual, dataset, flag)  # Calculate fitness for each  individual in the population
            print("---------Generation ", g, "-----------")
            # Sort them by fitness value.
            population.sort(key=lambda x: x.fitness, reverse=False) if self.problem == Problem.MINIMISATION \
                else population.sort(key=lambda x: x.fitness, reverse=True)  # Sort population by fitness, best on top
            mean_pop_fitness = mean([p.fitness for p in population])
            FileWriter.generation_printouts(  # Print results in the terminal and save them in the log file
                best_fitness=population[0].fitness, mean_fitness=mean_pop_fitness,
                worse_fitness=population[-1].fitness, current_generation=g, current_run=run_number)

            if g < generations - 1:  # Breeding. Evolution happens up to the generation before the last.
                # Elitism; copy the best individual into the next generation
                intermediate_pop = [population[0].model]

                for i in range(1, len(population)):  # Evolutionary process.
                    parent1 = EA.tournament(self, tournament_size, population)
                    r = random()
                    if r < xover_prob:  # Crossover
                        parent2 = self.tournament(tournament_size, population)
                        # When I'm doing crossover on both SA and TA sides (thus two crossover points), I need to do it
                        # over 2 steps: in the first step, we crossover the SA side and obtain an intermediate_child.
                        # From this child, the SA side is new as a result of the crossover, while the TA side is still
                        # the same, as it was from parent1. So in step 2 we crossover the intermediate_child's TA side,
                        # which as we've said is parent1's TA side with parent2's TA side, and thus obtain the final
                        # child.
                        intermediate_child = self.crossover(parent1, parent2, "SA")
                        child = self.crossover(intermediate_child, parent2, "TA")
                        # child = self.crossover(parent1, parent2)  # This is crossover with a single crossover point,
                        # # which can be anywhere in the tree (SA or TA side). This is an alternative type of crossover.
                        intermediate_pop.append(child)
                    else:  # Mutation
                        intermediate_pop.append(self.mutate(parent1))

                for individual, intermediate in zip(population, intermediate_pop):
                    individual.model = intermediate  # Copy the intermediate population into the actual population list


# Runs the EA algorithm over a number of independent runs and prints and saves results.
if __name__ == '__main__':
    task = EvaSATATrading()
    task.run(population_size=POP_SIZE, xover_prob=XOVER_PROB, tournament_size=TOURNAMENT_SIZE,
             no_of_gens=nGENS, no_of_runs=nRuns, training_dataset=training_data, test_dataset=test_data,
             class_object=EvaSATATrading.Individual(""))
    FileWriter.log_experimental_setup_gp(INITIAL_DEPTH, MAX_DEPTH, FLIP_MUT_PROB, F_SET, T_SET)
