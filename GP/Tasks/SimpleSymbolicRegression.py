from Auxiliary import FileWriter
from EA import Problem
from GP.GP import GP, Util
import numpy as np
import pandas as pd
from random import seed

seed(111)
problem = Problem.MINIMISATION  # Setup the current problem as a maximisation one (i.e. maximise the fitness value).
terminalNodeCrossBias = 0.1  # Probability to select a terminal node during crossover. Value given by Koza (1992).
nRuns = 1  # Individual and independent GP runs
INITIAL_DEPTH = 3  # Initial tree depth during population initialisation
MAX_DEPTH = 4  # Maximum tree depth
POP_SIZE = 10  # Population size
XOVER_PROB = 0.85  # Crossover probability. Mutation probability is (1 - XOVER_PROB)
FLIP_MUT_PROB = 0.5  # Probability of selecting a tree node for mutation during the point mutation operator
TOURNAMENT_SIZE = 10  # Tournament size
nGENS = 5  # Number of generations
ERC_LB = -1  # Lower bound for ERC numbers
ERC_UB = 1  # Upper bound for ERC numbers
TYPE = "Arithmetic"
training_data = np.array(pd.read_csv("./1301TW5indicatorstraining.csv"))  # Read training file
test_data = np.array(pd.read_csv("./1301TW5indicatorstest.csv"))  # Read test file
F_SET = ["ADD", "SUB", "MUL", "DIV"]  # Function Set
T_SET = ["Var0", "Var1", "Var2", "Var3", "Var4", "Var5", "Var6", "Var7",
         "Var8", "Var9", "Var10", "Var11", "Var12", "Var13", "Var14", "Var15",
         "Var16", "Var17", "Var18", "Var19", "Var20", "Var21", "Var22", "Var23",
         "Var24", "Var25", "Var26", "Var27", "Var28", "Var29", "Var30", "Var31",
         "Var32", "Var33", "Var34", "Var35", "Var36", "Var37", "Var38", "Var39",
         "Var40", "Var41", "Var42", "Var43", "Var44", "Var45", "Var46", "Var47",
         "Var48", "Var49", "Var50", "Var51", "Var52", "Var53", "Var54", "Var55",
         "Var56", "Var57", "Var58", "Var59", "Var60", "Var61", "Var62", "Var63",
         "Var64", "Var65", "Var66", "Var67", "Var68", "Var69", "Var70", "Var71",
         "Var72", "Var73", "Var74", "Var75", "Var76", "Var77", "Var78", "Var79",
         "Var80", "Var81", "Var82", "Var83", "Var84", "Var85", "Var86", "Var87",
         "Var88", "Var89", "Var90", "Var91", "Var92", "Var93", "Var94", "Var95",
         "Var96", "Var97", "Var98", "ERC"]
grammar = {  # Acts as a lookup type table, finding what are the compatible types and their values
    "Root": ["ADD", "SUB", "MUL", "DIV"],  # Force the root of the trees to be one of these functions
    "ADD": F_SET + T_SET,  # These are the compatible types when the parent node is ADD
    "SUB": F_SET + T_SET,  # These are the compatible types when the parent node is SUB
    "MUL": F_SET + T_SET,  # These are the compatible types when the parent node is MUL
    "DIV": F_SET + T_SET  # These are the compatible types when the parent node is DIV
}

types_lookup = {
    "Boolean": ["AND", "OR", "GT", "LT"],
    "Arithmetic": F_SET + T_SET
}


class SimpleSymbolicRegression(GP):
    def __init__(self):
        super().__init__(POP_SIZE, INITIAL_DEPTH, MAX_DEPTH, grammar, types_lookup, function_set=F_SET,
                         terminal_set=T_SET, terminal_node_xover_bias=terminalNodeCrossBias,
                         flip_mutation_probability=FLIP_MUT_PROB, TYPE=TYPE, erc_lb=ERC_LB, erc_ub=ERC_UB,
                         problem=problem)

    # RMSE fitness evaluation. Flag not used here (it's there to denote if we are evaluating on training or test data)
    def evaluate(self, individual, dataset, flag):
        predictions = Util.interpret(individual.model, dataset)
        reality = dataset[:, 0]  # First column is (target) prices
        mse = np.square(np.subtract(reality, predictions)).mean()
        rmse = np.sqrt(mse)
        individual.fitness = rmse  # Root mean squared error


# Runs the EA algorithm over a number of independent runs and prints and saves results.
if __name__ == '__main__':
    task = SimpleSymbolicRegression()
    task.run(population_size=POP_SIZE, xover_prob=XOVER_PROB, tournament_size=TOURNAMENT_SIZE,
             no_of_gens=nGENS, no_of_runs=nRuns, training_dataset=training_data, test_dataset=test_data,
             class_object=SimpleSymbolicRegression.Individual(""))
    FileWriter.log_experimental_setup_gp(INITIAL_DEPTH, MAX_DEPTH, FLIP_MUT_PROB, F_SET, T_SET)
