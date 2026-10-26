from Auxiliary import FileWriter
from EA import Problem, EA
from GP.GP import GP, Util
import numpy as np
import pandas as pd
from random import seed
from statistics import mean, stdev

seed(111)
problem = Problem.MAXIMISATION  # Setup the current problem as a maximisation one (i.e. maximise the fitness value).
terminalNodeCrossBias = 0.1  # Probability to select a terminal node during crossover. Value given by Koza (1992).
nRuns = 10  # Individual and independent GP runs
INITIAL_DEPTH = 2  # Initial tree depth during population initialisation
MAX_DEPTH = 4  # Maximum tree depth
POP_SIZE = 500  # Population size
XOVER_PROB = 0.95  # Crossover probability. Mutation probability is (1 - XOVER_PROB)
FLIP_MUT_PROB = 0.5  # Probability of selecting a tree node for mutation during the point mutation operator
TOURNAMENT_SIZE = 4  # Tournament size
nGENS = 25  # Number of generations
ERC_LB = -1  # Lower bound for ERC numbers
hh = 20
gg = 0.02
ERC_UB = 1  # Upper bound for ERC numbers
THETA = 0.05  # Threshold used as part of the fitness function. Look in there for more details.
MIN_NO_OF_TRADES = 6  # Forces a minimum number of trades during trading.
TYPE = "Predicate"
current_run = 0  # Tracks the current GP run
training_data = np.array(pd.read_csv("./GAData/AUDUSD/window_1/train.csv"))  # Read training file
test_data = np.array(pd.read_csv("./GAData/AUDUSD/window_1/val.csv"))  # Read test file
F_SET = ["AND", "OR", "GT", "LT"]  # Function Set
# Number of variables  in the terminal set depends on number of features in the excel input file
# ERC: Ephemeral Random Constant; it should always be in the last place of the array
# T_SET = ["Var0", "Var1", "Var2", "Var3", "Var4", "Var5", "Var6", "Var7", "Var8", "Var9", "Var10", "Var11", "Var12", "Var13", "Var14", "Var15", "Var16", "Var17", "Var18", "Var19", "Var20", "Var21", "Var22", "Var23",  "ERC"]
T_SET = ["Var0", "Var1", "Var2", "Var3", "Var4", "Var5", "Var6", "Var7", "Var8", "Var9", "Var10", "ERC"]
grammar = {  # Acts as a lookup type table, finding what are the compatible types and their values
    "Root": ["AND", "OR"],  # Force the root of the trees to be one of these functions
    "AND": F_SET,  # These are the compatible types when the parent node is AND
    "OR": F_SET,  # These are the compatible types when the parent node is OR
    "LT": T_SET,  # These are the compatible types when the parent node is LT
    "GT": T_SET  # These are the compatible types when the parent node is GT
}

types_lookup = {  # Defines the types of each node; the function set is booleans, whereas the terminal set is arithmetic
    "Boolean": F_SET,
    "Arithmetic": T_SET
}


class EvaTrading(GP):
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

    # Fitness evaluation based on trading metrics. Flag is used to denote training or test dataset.
    @staticmethod
    def evaluate(individual, data, flag):
        global current_run

        rate_of_return = []
        transaction_cost = 0.00025
        riskfree = 0.00022
        q = 0  # Shares quantity. For now we allow a single quantity only.
        no_of_trades = 0
        price_at_buy = data[0, 0]  # initialise value to the first price in the time series
        signal = Util.interpret(individual.model,
                                data)  # boolean depending on the evaluation of the expression (tree)

        for i in range(len(data)):
            price = data[i, 0]  # Price is Column 0 in the dataset
            if signal[i]:  # If expression output is True, then take an action; otherwise take a different action.
                no_of_trades += 1
                price_at_buy = price
                all_prices = data[i:i + hh, 0]  # all_prices = data.iloc[i:i+n,0]
                temp = 0.00
                for count, price in enumerate(all_prices):
                    temp = price
                    all_prices[count::]  # reducing the list size to latest first element
                    for compare_value in all_prices:
                        if (compare_value - temp) / temp > gg:
                            no_of_trades += 1
                            rate_of_return.append((compare_value - price_at_buy) / price_at_buy - transaction_cost)
                            break
                    if no_of_trades % 2 != 0:
                        no_of_trades += 1
                        price_at_sell = all_prices[-1]
                        rate_of_return.append((price_at_sell - price_at_buy) / price_at_buy - transaction_cost)
                    break

        individual.no_of_trades = no_of_trades
        if flag == 0:  # Training set calculation
            # Encourage individuals to close open positions (no_of_trades is an even number) & have a min no of trades
            individual.rate_of_return = mean(rate_of_return) \
                if (no_of_trades % 2 == 0 and no_of_trades >= MIN_NO_OF_TRADES) else 0
        else:  # Test set calculation
            individual.rate_of_return = mean(
                rate_of_return) if rate_of_return else 0  # Calculate mean if list not empty
            individual.signal = signal
            # FileWriter.save_results_to_file("SignalsBestTest.txt", signal,
            #                                current_run,pop_size,xover_prob,tournament_size,max_depth,no_of_gens)  # Signals from best model in test set
            current_run += 1

        # Standard deviation calculation requires at least 2 values, which are derived from a minimum of 4 trades
        individual.risk = stdev(rate_of_return) if no_of_trades >= 4 else 0
        individual.sharpe_ratio = individual.rate_of_return / individual.risk \
            if individual.risk != 0 else individual.rate_of_return  # Sharpe ratio calculation
        individual.fitness = individual.sharpe_ratio  # Fitness function value


# Runs the EA algorithm over a number of independent runs and prints and saves results.
if __name__ == '__main__':
    task = EvaTrading()
    task.run(population_size=POP_SIZE, xover_prob=XOVER_PROB, tournament_size=TOURNAMENT_SIZE,
             no_of_gens=nGENS, no_of_runs=nRuns, training_dataset=training_data, test_dataset=test_data,
             class_object=EvaTrading.Individual(""))
    FileWriter.log_experimental_setup_gp(INITIAL_DEPTH, MAX_DEPTH, FLIP_MUT_PROB, F_SET, T_SET)