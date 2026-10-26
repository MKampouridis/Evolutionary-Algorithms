from abc import ABC
from random import random, randint, uniform
from EA import EA
from GP import Util
from multipledispatch import dispatch


class GP(EA, ABC):
    def __init__(self, pop_size, initial_depth, max_depth, grammar, types_lookup, function_set, terminal_set,
                 terminal_node_xover_bias, flip_mutation_probability, erc_lb, erc_ub, TYPE, problem):
        self.pop_size = pop_size
        self.initial_depth = initial_depth
        self.max_depth = max_depth
        self.grammar = grammar
        self.types_lookup = types_lookup
        self.function_set = function_set
        self.terminal_set = terminal_set
        self.terminal_node_xover_bias = terminal_node_xover_bias
        self.flip_mutation_probability = flip_mutation_probability
        self.erc_lb = erc_lb
        self.erc_ub = erc_ub
        self.TYPE = TYPE
        super().__init__(problem=problem)

    # Initial (random) population. Generates half trees using the full method, and the other half using the grow
    # The size (depth) of the trees is uniformly created across the population, i.e. some trees get a depth of 3,
    # some get a depth of 4, etc up to the given MAX_DEPTH
    def initialise_population(self):
        population = []  # Re-set population; this is necessary to start anew in each individual GP run.
        no_of_individuals = int(self.pop_size / self.max_depth)
        cursor = 0
        for i in range(self.initial_depth, self.max_depth):
            for j in range(no_of_individuals):
                root = self.grammar.get("Root")  # Get a random function from the ones designated as Root
                function = root[randint(0, len(root) - 1)]
                if j % 2 == 0:  # Full method tree generation
                    if self.TYPE == "Predicate":
                        expression = self.generate_tree(i, [], function, "full")
                    else:
                        expression = self.generate_tree(i, i, [], "full")
                else:  # Grow method tree generation
                    if self.TYPE == "Predicate":
                        expression = self.generate_tree(i, [], function, "grow")
                    else:
                        expression = self.generate_tree(i, i, [], "grow")

                population.append(
                    self.Individual(expression))  # Add new individual with generated expression and initial
                # fitness and no of trades of zero
                cursor += 1

        while cursor < self.pop_size:  # Fill in any remaining population individuals
            method = "full" if random() < 0.5 else "grow"
            rand_depth = randint(2,
                                 self.max_depth - 2)  # Start from +2 so that it doesn't create trees of depth of 0 or 1
            root = self.grammar.get("Root")  # Obtain a list of root-compatible functions
            function = root[randint(0, len(root) - 1)]  # Get a random function from the ones designated as root
            if self.TYPE == "Predicate":
                population.append(self.Individual(
                    self.generate_tree(rand_depth, [], function, method)))
            else:
                population.append(self.Individual(
                    self.generate_tree(rand_depth, rand_depth, [], method)))
            cursor += 1

        return population

    # Generates *predicate* trees
    # Method for recursively generating a single expression (tree)
    # @param current_depth
    # @param max_depth
    # @param expression The tree that will (eventually) be returned by the method. New nodes are added recursively.
    # @node current tree node that is getting created
    # @method Grow or Full tree initialisation
    # @returns expression The full expression (tree)
    @dispatch(int, list, str, str)
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

        if node == "LT" or node == "GT":
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
                        random() < terminal_prob and node != "LT" and node != "GT"):
                    child = "LT" if randint(0, 1) == 0 else "GT"  # We only have two comparisons: LT and GT; so pick
                # at random either LT or GT

                self.generate_tree(current_depth - 1, expression, child, method)

        return expression

    # Generic method of generating trees. Applicable to Arithmetic-based trees.
    @dispatch(int, int, list, str)
    def generate_tree(self, current_depth, max_depth, expression, method):
        # Following if-statement as per Poli's "A Field Guide to Genetic Programming", p.14: If grow method,
        # choose a leaf when a certain probability occurs
        terminal_prob = len(self.terminal_set) / (len(self.terminal_set) + len(self.function_set))
        index = randint(0, len(self.terminal_set) - 1)
        if current_depth == 0 or (method == "grow" and current_depth != max_depth and random() < terminal_prob):
            if self.terminal_set[index] == "ERC":
                expression.append(uniform(self.erc_lb, self.erc_ub))
            else:
                expression.append(self.terminal_set[index])
        else:
            expression.append(self.function_set[randint(0, len(self.function_set) - 1)])
            self.generate_tree(current_depth - 1, max_depth, expression, method)  # Branch 1
            self.generate_tree(current_depth - 1, max_depth, expression, method)  # Branch 2
            # (all functions have arity of 2)

        return expression

    # Point mutation
    def mutate(self, expression):
        copy = expression.copy()  # create a copy
        for index in range(len(copy)):
            if random() < self.flip_mutation_probability:  # Apply point mutation at each node with probability
                s = copy[index]
                if s == "AND":  # AND can only be point-mutated to OR; OR only to AND; GT only to LT; LT only to GT.
                    copy[index] = "OR"
                elif s == "OR":
                    copy[index] = "AND"
                elif s == "GT":
                    copy[index] = "LT"
                elif s == "LT":
                    copy[index] = "GT"
                elif s == "ADD" or s == "SUB" or s == "MUL" or s == "DIV" or s == "POW":
                    copy[index] = self.function_set[randint(0, len(self.function_set) - 1)]
                else:  # Terminal set cases
                    if "Var" not in str(s):  # No Var, thus ERC; we force ERC to be mutated with another ERC
                        copy[index] = uniform(self.erc_lb, self.erc_ub)  # Random number uniformly distributed
                    else:  # It's a Var. We force a Var to be mutated with another Var
                        number = randint(0, len(self.terminal_set) - 2)  # Get a random index from T_SET, excluding ERC
                        while str(number) in s:  # Re-do this, if we ended up mutating to the same T_SET variable
                            number = randint(0, len(self.terminal_set) - 2)
                        copy[index] = self.terminal_set[number]

        return copy

    # Subtree crossover
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
        if not root1_type and isinstance(root1, float):  # When ERC, a float appears in the tree, so no key was found.
            root1_type.append('Arithmetic')  # When ERC, since it's a float, we set the type to Arithmetic.
        root2_type = [key for key, values in self.types_lookup.items() if root2 in values]
        if not root2_type and isinstance(root2, float):
            root2_type.append('Arithmetic')

        while root1_type[0] != root2_type[0]:  # We want the two roots to be of the same (hence compatible) type
            xo2_start = randint(1, len(parent2) - 2)  # XOVER point from index 1 (not 0, thus excluding the root)
            root2 = parent2[xo2_start]
            root2_type = [key for key, values in self.types_lookup.items() if root2 in values]
            if not root2_type and isinstance(root2, float):
                root2_type.append('Arithmetic')

        temp = Util.get_valid_subtree(parent1, xo1_start, self.function_set)  # subtree to be exchanged from parent1
        temp2 = Util.get_valid_subtree(parent2, xo2_start, self.function_set)  # subtree to be exchanged from parent2

        child = Util.copy_subtree(xo1_start, parent1, temp, temp2)  # Copy subtree from parent2 into parent1

        if Util.is_valid(child, self.function_set) is False:
            print("SHOULD NEVER REACH HERE!")
            exit()

        return child
