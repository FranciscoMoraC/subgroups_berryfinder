"""Tests for the ExCover algorithm."""

import unittest

from pandas import DataFrame

from subgroups.algorithms import ExCover
from subgroups.core.operator import Operator
from subgroups.core.pattern import Pattern
from subgroups.core.selector import Selector
from subgroups.datasets import load_mushroom_csv


class TestExCover(unittest.TestCase):

    def test_paper_example(self):
        dataframe = DataFrame([
            ["+", "A B D E"], ["+", "A B C D E"], ["+", "A C D E"],
            ["+", "A B C"], ["+", "B"], ["-", "A B D E"],
            ["-", "B C D E"], ["-", "C D E"], ["-", "A D E"],
            ["-", "A D"],
        ], columns=["class", "items"])
        for item in "ABCDE":
            dataframe[item] = dataframe["items"].map(
                lambda value, item=item: "yes" if item in value.split() else "no")

        model = ExCover()
        model.fit(dataframe.drop(columns=["items"]), ("class", "+"))

        expected = {
            str(Pattern([Selector("B", Operator.EQUAL, "yes")])),
            str(Pattern([
                Selector("A", Operator.EQUAL, "yes"),
                Selector("C", Operator.EQUAL, "yes"),
            ])),
        }
        self.assertEqual({str(pattern) for pattern in model.top_patterns}, expected)
        self.assertGreater(model.pruned_subgroups, 0)
        self.assertLess(model.visited_subgroups, 52)

    def test_max_complexity(self):
        model = ExCover(max_depth=1)
        self.assertEqual(model._max_depth, 1)

    def test_filter_uses_class_normalized_support(self):
        dataframe = DataFrame({
            "class": ["+", "+", "+", "-", "-", "-"],
            "feature": ["yes", "yes", "no", "yes", "yes", "yes"],
        })
        model = ExCover()
        model.fit(dataframe, ("class", "+"))

        self.assertNotIn(
            str(Pattern([Selector("feature", Operator.EQUAL, "yes")])),
            {str(pattern) for pattern in model.top_patterns},
        )

    def test_mushroom_example(self):
        model = ExCover()
        model.fit(load_mushroom_csv(), ("class", "e"))

        self.assertEqual(
            {str(pattern) for pattern in model.top_patterns},
            {
                "[gill-size = 'b', stalk-surface-above-ring = 's']",
                "[odor = 'n']",
                "[stalk-surface-above-ring = 's']",
            },
        )

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            ExCover(cats=0)
        with self.assertRaises(ValueError):
            ExCover(max_depth=0)
        with self.assertRaises(ValueError):
            ExCover(write_results_in_file=True)
