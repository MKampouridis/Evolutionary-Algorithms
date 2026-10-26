from unittest import TestCase
import Util

class Test(TestCase):
    def test_interpret(self):
        tree = ["AND", "GT", 0.5, 0.2, "LT", 0.1, 0.154]
        res = Util.interpret(tree, [0])

        tree2 = ["SA_AND", "SA_GT", 0.5, 0.2, "SA_LT", 0.1, 0.154]
        res2 = Util.interpret(tree2, [0])

        if res and res2:
            pass
        else:
            self.fail()
