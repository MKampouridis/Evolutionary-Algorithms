# AUXILIARY MODULE FOR PERFORMING GENERIC OPERATIONS ON EXPRESSIONS (TREES)
import math

import numpy as np


# Prefix interpreter. Its logic is based here: https://www.geeksforgeeks.org/evaluation-prefix-expressions/
# Uses Numpy vectorisation for quick matrix operations without the need of for-loops
def interpret(expression, data):
    stack = []
    for i in range(len(expression) - 1, -1, -1):  # reverse iteration over the expression elements
        if "AND" in str(expression[i]) or "OR" in str(expression[i]):
            v1 = stack.pop()
            v2 = stack.pop()
            if "AND" in expression[i]:
                stack.append(np.logical_and(v1, v2))
            elif "OR" in expression[i]:
                stack.append(np.logical_or(v1, v2))
        elif "GT" in str(expression[i]) or "LT" in str(expression[i]) or expression[i] in ["ADD", "SUB", "MUL", "DIV",
                                                                                           "POW"]:
            v1 = stack.pop()
            v2 = stack.pop()
            if "Var" in str(v1):
                # Get Var number (e.g. "1" from Var1)
                index = int(str(v1)[len(str(v1)) - 1]) \
                    if len(str(v1)) <= 4 \
                    else int(str(v1)[len(str(v1)) - 2:len(str(v1))])  # Var1-Var9 or ERC; len<=4. Var10-Var99; length 5.
                x = data[:, index + 1]  # Obtain Var arithmetic value; index+1, as the first column is prices.
            elif isinstance(v1, np.ndarray):  # v1 is an array of inputs
                x = v1
            else:
                x = np.full(shape=len(data), fill_value=v1)  # v1 is a single value (thus it was an ERC)

            if "Var" in str(v2):
                index = int(str(v2)[len(str(v2)) - 1]) if len(str(v2)) <= 4 \
                    else int(str(v2)[len(str(v2)) - 2:len(str(v2))])
                y = data[:, index + 1]
            elif isinstance(v2, np.ndarray):  # v2 is an array of inputs
                y = v2
            else:
                y = np.full(shape=len(data), fill_value=v2)

            if "GT" in str(expression[i]):
                stack.append(x > y)
            elif "LT" in str(expression[i]):
                stack.append(x < y)
            elif expression[i] == "ADD":
                stack.append(x + y)
            elif expression[i] == "SUB":
                stack.append(x - y)
            elif expression[i] == "MUL":
                stack.append(x * y)
            elif expression[i] == "DIV":
                result = x / y
                np.seterr(divide='ignore', invalid='ignore')  # Suppress warnings for division by 0
                result[y == 0] = 999999  # Protected division. Return a very large number when we divide by 0.
                stack.append(result)
            elif expression[i] == "POW":
                stack.append(math.pow(x,y))
        else:
            stack.append(expression[i])

    return stack.pop()


# Checks if given expression (tree) is valid
def is_valid(expression, F_SET):
    # If expression is *not* part of the Function Set (thus a single terminal/number), return true; otherwise false.
    if len(expression) == 1:
        if expression[0] not in F_SET:
            return True
        else:
            return False

    # It's valid prefix if first element is a function AND the last two are *not* a function
    to_return = False
    for i in range(0, len(F_SET)):
        if expression[0] == F_SET[i] and \
                expression[len(expression) - 1] != F_SET[i] and \
                expression[len(expression) - 2] != F_SET[i]:
            to_return = True

    function_counter = 0  # how many functions in the expression
    terminal_counter = 0  # how many terminals in the expression
    for i in range(0, len(expression)):
        if expression[i] in F_SET:
            function_counter += 1
        else:
            terminal_counter += 1

    # The number of terminals should always be one more than the number of functions
    if terminal_counter - function_counter != 1:
        to_return = False

    return to_return


# Method for obtaining a valid subtree from an expression
# @Param expression The expression from which we're going to obtain a subtree
# @Param index The index which is going to form the root of the subtree
# @Param F_SET The GP function set
# @Returns a valid subtree
def get_valid_subtree(expression, index, F_SET):
    for i in range(1, len(expression)):
        temp = []
        for j in range(0, i):
            temp.append(expression[index + j])

        if is_valid(temp, F_SET):  # subtree is a valid prefix expression, so we can use it for crossover
            return temp


# Copy subtree into the given expression at the provided index point
# @Param index The index of expression where subtree1 will be removed and subtree2 will be copied into
# @Param expression The expression
# @Param subtree1 The subtree that will be removed from the given expression
# @Param subtree2 The subtree that will replace subtree1 in the expression
# @Returns The newly created expression, after copying subtree2 into the place of subtree1
def copy_subtree(index, expression, subtree1, subtree2):
    new_expression = []  # Put together the child solution into a list
    for i in range(0, index):
        new_expression.append(expression[i])
    for i in range(0, len(subtree2)):
        new_expression.append(subtree2[i])
    for i in range(index + len(subtree1), len(expression)):
        new_expression.append(expression[i])

    return new_expression


def hamming_distance(signal_returned, true_signal):
    return abs(signal_returned - true_signal)
