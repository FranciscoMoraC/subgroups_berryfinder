# -*- coding: utf-8 -*-

# Contributors:

"""Tests of the functionality contained in the file 'algorithms/berryfinder.py'.
"""

from os import remove
from pandas import DataFrame
from subgroups.algorithms.subgroup_sets.berryfinder import BerryFinder
from subgroups.data_structures.bfinder_node import BFinderNode
from subgroups.core.operator import Operator
from subgroups.core.pattern import Pattern
from subgroups.core.selector import Selector
from subgroups.core.subgroup import Subgroup
import unittest
from math import inf

from subgroups.exceptions import InconsistentMethodParametersError

class TestQFinder(unittest.TestCase):
    
    def test_BerryFinder_init_method1(self):
        # Test with valid parameters.
        obj = BerryFinder(cats=3, max_complexity=10, coverage_thld=0.5,
                           or_thld=1.0, p_val_thld=0.1, abs_contribution_thld=0.3,
                           contribution_thld=4, write_results_in_file=True,
                           file_path='results.txt', min_rank=3)
        # Assert that the object has been created correctly.
        self.assertIsNotNone(obj)

    def test_BerryFinder_init_method2(self):
        # Test without the optional parameters.
        obj = BerryFinder()
        # Assert that the object has been created correctly.
        self.assertIsNotNone(obj)

    def test_BerryFinder_init_method3(self):
        # Test with invalid parameters.
        #Invalida types
        with self.assertRaises(TypeError):
            obj = BerryFinder(cats='3')
        with self.assertRaises(TypeError):
            obj = BerryFinder(max_complexity='10')
        with self.assertRaises(TypeError):
            obj = BerryFinder(coverage_thld='0.5')
        with self.assertRaises(TypeError):
            obj = BerryFinder(or_thld='1.0')
        with self.assertRaises(TypeError):
            obj = BerryFinder(p_val_thld='0.1')
        with self.assertRaises(TypeError):
            obj = BerryFinder(abs_contribution_thld='0.3')
        with self.assertRaises(TypeError):
            obj = BerryFinder(contribution_thld='4')
        with self.assertRaises(TypeError):
            obj = BerryFinder(write_results_in_file='True')
        with self.assertRaises(TypeError):
            obj = BerryFinder(file_path=1)
        with self.assertRaises(TypeError):
            obj = BerryFinder(min_rank='3')
        #Invalid values
        with self.assertRaises(ValueError):
            # 'write_results_in_file' is True but 'file_path' is None
            BerryFinder(write_results_in_file=True)
        with self.assertRaises(TypeError):
            # Invalid type for file_path
            BerryFinder(file_path=5)
        with self.assertRaises(InconsistentMethodParametersError):
            # cats is less tan -1
            BerryFinder(cats=-3)
        with self.assertRaises(InconsistentMethodParametersError):
            # cats is zero
            BerryFinder(cats=0)
        with self.assertRaises(InconsistentMethodParametersError):
            # coverage_thld is not in the range [0, 1]
            BerryFinder(coverage_thld=1.5)
        with self.assertRaises(InconsistentMethodParametersError):
            # or_thld is negative
            BerryFinder(or_thld=-1.0)
        with self.assertRaises(InconsistentMethodParametersError):
            # p_val_thld is not in the range [0, 1]
            BerryFinder(p_val_thld=1.5)
        with self.assertRaises(InconsistentMethodParametersError):
            # abs_contribution_thld is not greater than 0
            BerryFinder(abs_contribution_thld=-1.5)
        with self.assertRaises(InconsistentMethodParametersError):
            # contribution_thld is negative
            BerryFinder(contribution_thld=-4)
        with self.assertRaises(InconsistentMethodParametersError):
            # min_rank is negative
            BerryFinder(min_rank=-3) 
        with self.assertRaises(InconsistentMethodParametersError):
            # min_rank is too high
            BerryFinder(min_rank=10)       

    def test_BerryFinder_reduce_categories(self):
        # Check that the 'other' value is added to the reduced dataframe
        obj = BerryFinder(cats = 2)
        df = DataFrame({'a': ["1", "2", "1", "1", "3"], 'b': ["1", "2", "1", "1", "2"], 'c':["other", "2", "other", "other", "3"], 'class': ["1", "1", "0", "0", "1"]})
        target = ("class", "1")
        reduced_df = obj._reduce_categories(df, target)
        self.assertIn('other', reduced_df['a'].unique())
        self.assertNotIn("other", reduced_df['b'].unique())
        self.assertIn("other_", reduced_df['c'].unique())
    
    def test_BerryFinder_generate_selectors(self):
        # Check that the method returns the correct selectors
        obj = BerryFinder(cats = 2)
        df = DataFrame({'a': ["1", "2", "1", "1", "3"], 'b': ["1", "2", "1", "1", "2"], 'c':["other", "2", "other", "other", "3"], 'class': ["1", "1", "0", "0", "1"]})
        target = ("class", "1")
        reduced_df = obj._reduce_categories(df, target)
        obj._TP = reduced_df[reduced_df[target[0]] == target[1]].shape[0]
        obj._FP = reduced_df[reduced_df[target[0]] != target[1]].shape[0]
        selectors = obj._generate_selectors(reduced_df, target)
        self.assertIn(Selector("a", Operator.EQUAL, "1"), selectors)
        self.assertIn(Selector("a", Operator.EQUAL, "other"), selectors)
        self.assertIn(Selector("b", Operator.EQUAL, "1"), selectors)
        self.assertIn(Selector("b", Operator.EQUAL, "2"), selectors)
        self.assertIn(Selector("c", Operator.EQUAL, "other"), selectors)
        self.assertIn(Selector("c", Operator.EQUAL, "other_"), selectors)

    def test_BerryFinder_compute_rank(self):
        model = BerryFinder()
        self.assertEqual(model._compute_rank([1,0,1,1,1]), 1)
        self.assertEqual(model._compute_rank([0,1,1,1,1]), 0)
        self.assertEqual(model._compute_rank([0,0,1,1,1]), 0)
        self.assertEqual(model._compute_rank([1,1,1,1,1]), 5)
        self.assertEqual(model._compute_rank([1,1,1,1,0]), 4)

    def test_BerryFinder_add_to_graph1(self):
        sd = BerryFinder()
        str_base_selectors = ["a = 1", "b = 1", "c = 1", "d = 1", "e = 1"]
        base_selectors = [Selector.generate_from_str(sel) for sel in str_base_selectors]
        index_patterns_to_add = [[0,1,2,3], [0,1,2],[0,1,4],[0,1],[0],[1,2,3],[1,2],[1],[4]]
        sd._selectors = base_selectors
        sd._root_node = BFinderNode(set(range(len(base_selectors))), -1)
        for pat in index_patterns_to_add:
            pat = [base_selectors[i] for i in pat]
            pattern = Pattern(pat)
            sd._add_to_graph(pattern, 1)
        # The graph should look like this:
        #     "*" : {"[a = 1, b = 1, c = 1, d = 1]", "[a = 1, b = 1, e = 1]"},
        #     "[a = 1, b = 1, c = 1, d = 1]" : {"[a = 1, b = 1, c = 1]", "[b = 1, c = 1, d = 1]"},
        #     "[a = 1, b = 1, e = 1]" : {"[a = 1, b = 1]", "[e = 1]"},
        #     "[a = 1, b = 1, c = 1]" : {"[a = 1, b = 1]", "[b = 1, c = 1]"},
        #     "[b = 1, c = 1, d = 1]" : {"[b = 1, c = 1]"},
        #     "[a = 1, b = 1]" : {"[a = 1]", "[b = 1]"},
        #     "[e = 1]" : set(),
        #     "[b = 1, c = 1]" : {"[b = 1]"},
        #     "[a = 1]" : set(),
        #     "[b = 1]" : set(),
        target_graph = {
            "{0, 1, 2, 3, 4}" : [{0,1,2,3}, {0,1,4}],
            "{0, 1, 2, 3}" : [{0,1,2}, {1,2,3}],
            "{0, 1, 4}" : [{0,1}, {4}],
            "{0, 1, 2}" : [{0,1}, {1,2}],
            "{1, 2, 3}" : [{1,2}],
            "{0, 1}" : [{0}, {1}],
            "{4}" : [],
            "{1, 2}" : [{1}],
            "{0}" : [],
            "{1}" : []
        }
        nodes_to_check = [sd._root_node]
        while len(nodes_to_check) > 0:
            node = nodes_to_check.pop()
            self.assertEqual(node.descendant_indexes, target_graph[str(node.selector_indexes)])
            if node == sd._root_node:
                self.assertEqual(node.rank, -1)
            else:
                self.assertEqual(node.rank, 1)
            nodes_to_check.extend(node.descendant_nodes)

    def test_BerryFinder_add_to_graph2(self):
        # We will test the pruning technique applied in BerryFinder by  adjusting the ranks of each pattern in the search space.
        # For this, we duplicate the BerryFinder class and modify the _handle_individual_result method to return the desired rank.
        attributes = {
            key: value
            for key, value in BerryFinder.__dict__.items()
            if key not in ('__dict__', '__weakref__') and key not in BerryFinder.__slots__
        }
        # Include __slots__ in the new class attributes
        if '__slots__' in BerryFinder.__dict__:
            attributes['__slots__'] = BerryFinder.__slots__
        # Create the duplicate class
        DuplicateBerryFinder = type(
            BerryFinder.__name__ + "Duplicate",
            BerryFinder.__bases__,
            attributes
        )
        str_base_selectors = ["a = '1'", "b = '1'", "c = '1'", "d = '1'"]
        base_selectors = [Selector.generate_from_str(sel) for sel in str_base_selectors]
        pattern_ranks = {
            str(Pattern([])): 1,
            str(Pattern([base_selectors[0]])): 6,  # [a]
            str(Pattern([base_selectors[1]])): 4,  # [b]
            str(Pattern([base_selectors[2]])): 4,  # [c]
            str(Pattern([base_selectors[3]])): 6,  # [d]
            str(Pattern([base_selectors[0], base_selectors[1]])): 4,  # [a, b]
            str(Pattern([base_selectors[0], base_selectors[1], base_selectors[3]])): 4,  # [a, b, d]
            str(Pattern([base_selectors[0], base_selectors[1], base_selectors[2], base_selectors[3]])): 4,  # [a, b, c, d]
            str(Pattern([base_selectors[0], base_selectors[2]])): 4,  # [a, c]
            str(Pattern([base_selectors[0], base_selectors[3]])): 4,  # [a, d]
            str(Pattern([base_selectors[1], base_selectors[2]])): 6,  # [b, c]
            str(Pattern([base_selectors[1], base_selectors[2], base_selectors[3]])): 3,  # [b, c, d]
            str(Pattern([base_selectors[2], base_selectors[3]])): 3,  # [c, d]
        }
        # Define modified _handle_individual_result method
        def _handle_individual_result(self, target_column, pattern: Pattern, appearance):
            if str(pattern) in pattern_ranks:
                return pattern_ranks[str(pattern)]
            return 1
        # Include the modified method in the new class
        DuplicateBerryFinder._handle_individual_result = _handle_individual_result
        # Create the model
        sd = DuplicateBerryFinder(min_rank=3)
        # Dataframe containing the target column and prevoiusly defined selectors
        df = DataFrame({"a": '1', "b": '1', "c": '1', "d": '1', "class": '1'}, index = [0])
        target = ("class", '1')
        sd.fit(df, target)
        # The graph should look like this:
        #     "*" : {"[a = '1']", "[b = '1', c = '1']", "[d = '1']"},
        #     "[a = '1']" : set(),
        #     "[d = '1']" : set()
        #     "[b = '1', c = '1']" : {"[b = '1']", "[c = '1']"},
        #     "[b = '1']" : set(),
        #     "[c = '1']" : set(),
        #     However, the selectores will be resorted: [a = '1'], [d = '1'], [b = '1'], [c = '1'] 
        target_graph = {
            "{0, 1, 2, 3}" : [{0}, {1}, {2, 3}],
            "{0}" : [],
            "{1}" : [],
            "{2, 3}" : [{2}, {3}],
            "{2}" : [],
            "{3}" : [],
        }
        nodes_to_check = [sd._root_node]
        while len(nodes_to_check) > 0:
            node = nodes_to_check.pop()
            self.assertEqual(node.descendant_indexes, target_graph[str(node.selector_indexes)])
            nodes_to_check.extend(node.descendant_nodes)

    
    def test_BerryFinder_check_redundancies(self):
        sd = BerryFinder()
        str_base_selectors = ["a = 1", "b = 1", "c = 1", "d = 1", "e = 1"]
        base_selectors = [Selector.generate_from_str(sel) for sel in str_base_selectors]
        sd._selectors = base_selectors
        sd._root_node = BFinderNode(set(range(len(base_selectors))), -1)
        index_patterns_to_add = [[0,1,2,3], [0,1,2],[0,1,4],[0,1],[0],[1,2,3],[1,2],[1],[4]]
        pattern_ranks = [2,5,5,4,4,1,5,3,3]
        for i in range(len(index_patterns_to_add)):
            pat = index_patterns_to_add[i]
            pat = [base_selectors[i] for i in pat]
            rank = pattern_ranks[i]
            pattern = Pattern(pat)
            sd._add_to_graph(pattern, rank)
        sd._check_redundancy(sd._root_node, ("f", "1"))
        # Only the following nodes should remain:
        #     "[a = 1]",
        #     "[b = 1]",
        #     "[e = 1]",
        #     "[b = 1, c = 1]",
        #     "[a = 1, b = 1, e = 1]",
        target_indexes = [{0}, {1}, {4}, {1,2}, {0,1,4}]
        nodes_to_check = [sd._root_node]
        while len(nodes_to_check) > 0:
            node = nodes_to_check.pop()
            if node.selector_indexes in target_indexes:
                self.assertFalse(node.is_discarded())
            else:
                self.assertTrue(node.is_discarded())
            nodes_to_check.extend(node.descendant_nodes)

    def test_BerryFinder_fit1(self):
        # Minimum threshold, all patterns have rank 5 and only the empty pattern is selected
        df = DataFrame({'bread': {0: 'yes', 1: 'yes', 2: 'no', 3: 'yes', 4: 'yes', 5: 'yes', 6: 'yes'}, 'milk': {0: 'yes', 1: 'no', 2: 'yes', 3: 'yes', 4: 'yes', 5: 'yes', 6: 'yes'}, 'beer': {0: 'no', 1: 'yes', 2: 'yes', 3: 'yes', 4: 'no', 5: 'yes', 6: 'no'}, 'coke': {0: 'no', 1: 'no', 2: 'yes', 3: 'no', 4: 'yes', 5: 'no', 6: 'yes'}, 'diaper': {0: 'no', 1: 'yes', 2: 'yes', 3: 'yes', 4: 'yes', 5: 'yes', 6: 'yes'}})        
        target = ("diaper", "yes")
        model = BerryFinder(
                        coverage_thld=0,
                        or_thld=0,
                        p_val_thld=1.0,
                        abs_contribution_thld=0,
                        contribution_thld=inf,
                        min_rank=0,
                        write_results_in_file=True,
                        file_path='./results.txt'
                        )
        model.fit(df, target)
        file_to_parse = open('./results.txt', 'r')
        list_of_written_results = file_to_parse.readlines()
        self.assertEqual(len(list_of_written_results), 1)

    def test_BerryFinder_fit2(self):
        df = DataFrame({'bread': {0: 'yes', 1: 'yes', 2: 'no', 3: 'yes', 4: 'yes', 5: 'yes', 6: 'yes'}, 'milk': {0: 'yes', 1: 'no', 2: 'yes', 3: 'yes', 4: 'yes', 5: 'yes', 6: 'yes'}, 'beer': {0: 'no', 1: 'yes', 2: 'yes', 3: 'yes', 4: 'no', 5: 'yes', 6: 'no'}, 'coke': {0: 'no', 1: 'no', 2: 'yes', 3: 'no', 4: 'yes', 5: 'no', 6: 'yes'}, 'diaper': {0: 'no', 1: 'yes', 2: 'yes', 3: 'yes', 4: 'yes', 5: 'yes', 6: 'yes'}})        
        target = ("diaper", "yes")
        model = BerryFinder(write_results_in_file=True, file_path='./results.txt', min_rank=1)
        model.fit(df, target)
        # Description: [bread = 'no'], Target: diaper = 'yes' ; Rank : 2 ; 
        # Description: [milk = 'no'], Target: diaper = 'yes' ; Rank : 2 ; 
        # Description: [beer = 'yes'], Target: diaper = 'yes' ; Rank : 2 ; 
        # Description: [coke = 'yes'], Target: diaper = 'yes' ; Rank : 2 ; 
        # Description: [], Target: diaper = 'yes' ; Rank : 1 ; 
        file_to_parse = open('./results.txt', 'r')
        list_of_written_results = []
        for line in file_to_parse:
            list_of_written_results.append(line)
        list_of_subgroups = [Subgroup.generate_from_str(elem.split(";")[0][:-1]) for elem in list_of_written_results]
        self.assertEqual(len(list_of_subgroups), 5)
        self.assertIn(Subgroup.generate_from_str("Description: [bread = 'no'], Target: diaper = 'yes'"), list_of_subgroups)
        self.assertIn(Subgroup.generate_from_str("Description: [milk = 'no'], Target: diaper = 'yes'"), list_of_subgroups)
        self.assertIn(Subgroup.generate_from_str("Description: [beer = 'yes'], Target: diaper = 'yes'"), list_of_subgroups)
        self.assertIn(Subgroup.generate_from_str("Description: [coke = 'yes'], Target: diaper = 'yes'"), list_of_subgroups)
        self.assertIn(Subgroup.generate_from_str("Description: [], Target: diaper = 'yes'"), list_of_subgroups)
        file_to_parse.close()
        remove('./results.txt')

