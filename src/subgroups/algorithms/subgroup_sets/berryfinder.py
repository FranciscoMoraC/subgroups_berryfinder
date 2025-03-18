# -*- coding: utf-8 -*-

# Contributors:

"""This file contains the implementation of BerryFinder
"""

from typing import Union
from pandas import DataFrame, Series
from pandas.api.types import is_string_dtype
from subgroups.algorithms.algorithm import Algorithm
from subgroups.exceptions import InconsistentMethodParametersError, DatasetAttributeTypeError
from subgroups.core.pattern import Pattern
from subgroups.core.operator import Operator
from subgroups.core.selector import Selector
from subgroups.core.subgroup import Subgroup
from subgroups.quality_measures.coverage import Coverage
from subgroups.quality_measures.ppv import PPV
from subgroups.credibility_measures.odds_ratio_stat import OddsRatioStatistic
from subgroups.credibility_measures.p_value_independence import PValueIndependence
from subgroups.credibility_measures.selector_contribution import SelectorContribution
from subgroups.data_structures.bfinder_node import BFinderNode
from math import inf
import operator

class BerryFinder(Algorithm):
    """
    This class implements the BerryFinder algorithm. 

    :param cats: the maximum number of categories per column. If cats = -1, we take all the values. Otherwise, we take the cats-1 most frequent ones and the rest of them are grouped in the "other" value.
    :param max_complexity: the maximum complexity (size) of the patterns. If max_complexity = -1, we do not limit the complexity of the patterns.
    :param coverage_thld: the minimum coverage threshold.
    :param ppv_thld: the minimum positive predictive value threshold.
    :param or_thld: the minimum odds ratio threshold.
    :param p_val_thld: the maximum p-value threshold.
    :param abs_contribution_thld: the minimum absolute contribution threshold.
    :param contribution_thld: the maximum contribution ratio threshold.
    :param write_results_in_file: a boolean which indicates if the results are written in a file.
    :param file_path: the path of the file where the results are written.
    :param min_rank: the minimum rank of the pattern to be credible.
    """

    _credibility_criterions = {
        "coverage" :  operator.ge,
        "odds_ratio" : operator.ge,
        "p_value" : operator.le,
        "ppv" : operator.ge,
        "absolute_contribution" : operator.ge,
        "contribution_ratio" : operator.le,
    }

    __slots__ = ['_cats', '_file', '_visited_subgroups', '_max_complexity', '_selected_subgroups', '_pruned_subgroups', '_credible_subgroups', '_selectors', '_thresholds','_file_path','_TP','_FP','_N','_min_rank','_entry_template','_selector_appearances','_odds_ratio_measure','_p_value_measure','_selector_contribution_measure','_coverage_measure', '_ppv_measure','_root_node']

    def __init__(self, cats : int = -1, max_complexity: int = -1, coverage_thld: float = 0.1, ppv_thld = 0.6, or_thld: float = 1.2, p_val_thld: float = 0.05, abs_contribution_thld: float = 0.2, contribution_thld: float = 5, write_results_in_file: bool = False, file_path: Union[str,None] = None, min_rank : int = 1) -> None:
        if type(cats) is not int:
            raise TypeError("The type of the parameter 'cats' must be 'int'.")
        if type(max_complexity) is not int:
            raise TypeError("The type of the parameter 'max_complexity' must be 'int'.")
        if type(coverage_thld) is not float and type(coverage_thld) is not int:
            raise TypeError("The type of the parameter 'coverage_thld' must be 'float'.")
        if type(ppv_thld) is not float and type(ppv_thld) is not int:
            raise TypeError("The type of the parameter 'ppv_thld' must be 'float'.")
        if type(or_thld) is not float and type(or_thld) is not int:
            raise TypeError("The type of the parameter 'or_thld' must be 'float'.")
        if type(p_val_thld) is not float and type(p_val_thld) is not int:
            raise TypeError("The type of the parameter 'p_val_thld' must be 'float'.")
        if type(abs_contribution_thld) is not float and type(abs_contribution_thld) is not int:
            raise TypeError("The type of the parameter 'abs_contribution_thld' must be 'float'.")
        if type(contribution_thld) is not float and type(contribution_thld) is not int:
            raise TypeError("The type of the parameter 'contribution_thld' must be 'float'.")
        if type(min_rank) is not int:
            raise TypeError("The type of the parameter 'min_rank' must be 'int'.")
        if type(write_results_in_file) is not bool:
            raise TypeError("The type of the parameter 'write_results_in_file' must be 'bool'.")
        if (type(file_path) is not str) and (file_path is not None):
            raise TypeError("The type of the parameter 'file_path' must be 'str' or 'NoneType'.")
        # We check that that the parameter values are valid.
        if (cats < -1 or cats == 0):
            raise InconsistentMethodParametersError("The parameter 'cats' must be greater than zero or equal to -1.")
        if (coverage_thld < 0 or coverage_thld > 1):
            raise InconsistentMethodParametersError("The parameter 'coverage_thld' must be between 0 and 1.")
        if (ppv_thld < 0 or ppv_thld > 1):
            raise InconsistentMethodParametersError("The parameter 'ppv_thld' must be between 0 and 1.")
        if (or_thld < 0):
            raise InconsistentMethodParametersError("The parameter 'or_thld' must be greater than or equal to 0.")
        if (p_val_thld < 0 or p_val_thld > 1):
            raise InconsistentMethodParametersError("The parameter 'p_val_thld' must be between 0 and 1.")
        if (abs_contribution_thld < 0):
            raise InconsistentMethodParametersError("The parameter 'abs_contribution_thld' must be greater than or equal to 0.")
        if (contribution_thld < 0):
            raise InconsistentMethodParametersError("The parameter 'contribution_thld' must be greater than or equal to 0.")
        # If 'write_results_in_file' is True, 'file_path' must not be None.
        if (write_results_in_file) and (file_path is None):
            raise ValueError("If the parameter 'write_results_in_file' is True, the parameter 'file_path' must not be None.")
        if min_rank < 0 or min_rank > len(BerryFinder._credibility_criterions):
            raise InconsistentMethodParametersError("The parameter 'min_rank' must be between 0 and the number of credibility measures ("+ str(len(BerryFinder._credibility_criterions))+").")
        self._visited_subgroups = 0
        self._selected_subgroups = 0
        self._pruned_subgroups = 0
        self._credible_subgroups = 0
        self._cats = cats
        self._max_complexity = max_complexity
        if (write_results_in_file):
            self._file_path = file_path
        else:
            self._file_path = None
        self._file = None
        self._selectors = []
        # Thresholds for each credibility measure.
        self._thresholds = {
            "coverage" : coverage_thld,
            "odds_ratio" : or_thld,
            "p_value" : p_val_thld,
            "ppv" : ppv_thld,
            "absolute_contribution" : abs_contribution_thld,
            "contribution_ratio" : contribution_thld,
        }
        self._min_rank = min_rank
        # Dictionary used to save the appearance of each selector in the dataset.
        self._selector_appearances = dict()
        # We initialize the credibility measures objects.
        self._coverage_measure = Coverage()
        self._ppv_measure = PPV()
        self._odds_ratio_measure = OddsRatioStatistic()
        self._p_value_measure = PValueIndependence()
        self._selector_contribution_measure = SelectorContribution()

    def _get_selected_subgroups(self) -> int:
        return self._selected_subgroups

    def _get_unselected_subgroups(self) -> int:
        return self._visited_subgroups - self._selected_subgroups

    def _get_visited_subgroups(self) -> int:
        return self._visited_subgroups

    def _get_pruned_subgroups(self) -> int:
        return self._pruned_subgroups
    
    def _get_credible_subgroups(self) -> int:
        return self._credible_subgroups
    
    selected_subgroups = property(_get_selected_subgroups, None, None, "The number of selected subgroups.")
    unselected_subgroups = property(_get_unselected_subgroups, None, None, "The number of unselected subgroups.")
    visited_subgroups = property(_get_visited_subgroups, None, None, "The number of visited subgroups.")
    pruned_subgroups = property(_get_pruned_subgroups, None, None, "The number of pruned subgroups by its coverage.")
    credible_subgroups = property(_get_credible_subgroups, None, None, "The number of subgroups that meet the minimum rank.")

    def _reduce_categories(self,df : DataFrame,tuple_target_attribute_value: tuple) -> DataFrame:
        """ Method to reduce the number of different values for the categorical attributes. If cats = -1, we take all the values.
        Otherwise, we take the cats-1 most frequent ones and the rest of them are grouped in the "other" value.

        :param df: the dataset.
        :param tuple_target_attribute_value: the tuple which contains the target attribute name and the target attribute values.
        :return: the dataset with the reduced number of different values for the categorical attributes.
        
        """

        # If we don't have to limit the number of values, we take all of them.
        if self._cats == -1:
            return df
        
        df_without_target = df.drop(columns=[tuple_target_attribute_value[0]])
        for column in df_without_target:
            # Number of different values for the current column.
            n_values = df_without_target[column].nunique()
            # If we don't have to limit the number of values, we take all of them.
            if (n_values <= self._cats):
                continue
            else:
                value_counts = df_without_target[column].value_counts()
                # We edit our copy of the dataset to set the "other" value to the rows which have a value that is not in the top_values.
                # If the "other" value is already in the dataset, we need to make sure that the added value is different.
                other_string = "other"
                while other_string in value_counts.index:
                    other_string += "_"
                # Least frequent values. These values will be grouped in the "other" value.
                other_values = value_counts.nsmallest(n_values - self._cats+1).index
                # We edit our copy of the dataset to set the "other" value to the rows which have a value that is not in the top_values.
                df.loc[df_without_target[column].isin(other_values), column] = other_string
        return df
        

    def _generate_selectors(self,df : DataFrame,tuple_target_attribute_value: tuple) -> Series:
        """ 
        Method to generate the list of selectors in a given dataset.
        We use this function after reducing the number of categories.

        :param df: the dataset.
        :param tuple_target_attribute_value: the tuple which contains the target attribute name and the target attribute values.
        """
        max_rank_selectors = []
        other_selectors = []
        # Target appearance is a boolean Series which is True if the target attribute is equal to the target value.
        # It will be used to compute the rank of the selectors (in order to sort them).
        target_appearance = df[tuple_target_attribute_value[0]] == tuple_target_attribute_value[1]
        for column in df.columns:
            if column != tuple_target_attribute_value[0]:
                for value in df[column].unique():
                    sel = Selector(column,Operator.EQUAL,value)
                    # selectors.
                    appearance = df[column] == value
                    self._selector_appearances[sel] = appearance
                    sel_as_pattern = Pattern([sel])
                    # We compute the rank of the selector.
                    rank = self._handle_individual_result(target_appearance, sel_as_pattern, appearance)
                    # If the rank of the selector is the maximum possible rank, we add this selector to the list of selectors with the maximum rank.
                    if rank == len(BerryFinder._credibility_criterions):
                        max_rank_selectors.append(sel)
                    else:
                        other_selectors.append(sel)
        # Max rank selectors go first (break ties by coverage), other selectors are sorted by coverage (ascending).
        # This is done to prune as much as possible the branches of the tree.
        max_rank_selectors.sort(key = lambda x: self._selector_appearances[x].sum(), reverse = True)
        other_selectors.sort(key = lambda x: self._selector_appearances[x].sum())
        return max_rank_selectors + other_selectors
    
    def _handle_individual_result(self, target_column: Series, pattern: Pattern, appearance: Series) -> int:
        """ Method to compute the credibility measures of a pattern and its rank.
        
        :param target_column: the target column of the dataset represented as a pandas Series of booleans (True if equal to the target value, False otherwise).
        :param pattern: the pattern to be evaluated.
        :param appearance: pandas boolean Series containing the rows that satisfy the pattern.
        :return: the rank of the pattern.
        """
        # Most credibility measures are computed using a logistic regression model.
        tp = (target_column & appearance).sum()
        fp = appearance.sum() - tp
        # Parameters for each credibility measure. Contributions are computed at the same time.
        credibility_parameters = {
            "coverage" : {"tp": tp, "fp": fp, "TP": self._TP, "FP": self._FP},
            "odds_ratio" : {"tp": tp, "fp": fp, "TP": self._TP, "FP": self._FP},
            "p_value" : {"tp": tp, "fp": fp, "TP": self._TP, "FP": self._FP},
            "ppv" : {"tp": tp, "fp": fp},
            "contributions" : {"pattern": pattern, "target_appearance": target_column, "selector_appearances": self._selector_appearances,
                               "odds_ratio_definition": "statistic", "absolute_contribution_threshold": self._thresholds["absolute_contribution"]},
        }
        # If the credibility measure does not meet the threshold, we do not compute the rest of the credibility measures.
        # We store the credibility values in a dictionary and initialize them as the worst possible values.
        credibility_values = {
            "coverage": -inf,
            "odds_ratio": -inf,
            "p_value": inf,
            "ppv": -inf,
            "absolute_contribution": -inf,
            "contribution_ratio": inf
        }
        for cred in credibility_parameters:
            if cred != "contributions":
                measure_value = getattr(self,"_" + cred + "_measure").compute(credibility_parameters[cred])
                credibility_values[cred] = measure_value
                # If the credibility measure does not meet the threshold, we do not need to compute the rest of the credibility measures.
                # This is because the rank is computed as the number of consecutive True values in the credibility list.
                if not BerryFinder._credibility_criterions[cred](credibility_values[cred],self._thresholds[cred]):
                    break
            # Contribution measures are computed both at the same time and at the last iteration.
            else:
                absolute_contribution, contribution_ratio = self._selector_contribution_measure.compute(credibility_parameters[cred])
                credibility_values["absolute_contribution"] = absolute_contribution
                credibility_values["contribution_ratio"] = contribution_ratio
        # The credibility of a patterns is represented as a list of booleans,
        # where each element is True if the credibility measure meets the threshold and False otherwise.
        credibility = []
        for cred in BerryFinder._credibility_criterions:
            # The credibility_criterions dictionary contains the function to compare the credibility measure with the threshold.
            # The dictionaries credibility_values, thresholds and credibility_criterions have the same keys (the credibility measures).
            credibility.append(BerryFinder._credibility_criterions[cred](credibility_values[cred],self._thresholds[cred]))
        # We compute the numerical rank of the pattern given its credibility.
        return self._compute_rank(credibility)

    
    def _compute_rank(self,credibility: list) -> int:
        """Method to compute the rank of a pattern.

        :param credibility: the list of booleans representing the pattern's credibility.
        :return: the rank of the pattern.
        """
        rank = 0
        # We iterate over the credibility list to find the first False value.
        # The rank is the number of True values before the first False value (or the end of the list if there is no False value).
        for i in range(len(credibility)):
            if not credibility[i]:
                break
            rank = i+1
        return rank        

    def _add_to_graph(self,pattern: Pattern, rank : int) -> None:
        """Method to add a pattern to the graph of subgroups.

        :param pattern: the pattern to be added.
        :param rank: the rank of the pattern.
        """
        # Set of selector indexes that the pattern contains.
        selector_indexes = {self._selectors.index(selector) for selector in pattern}
        # Patterns that refine the new pattern, initially only the root node.
        refined = {self._root_node}
        # New node to be added to the graph
        new_node = BFinderNode(selector_indexes,rank)
        # While the list of refined patterns is not empty, we iterate over the patterns in the list.
        while refined:
            for x in refined:
                descendants = x.get_refining_descendants(selector_indexes)
                # If no descendant of x refines the new pattern, add the new pattern as a descendant of x
                if not descendants:
                    x.add_descendant(new_node)
            # Update refined set with descendants of the current refined set that refine the new pattern
            refined = {
                descendant
                for x in refined
                for descendant in x.get_refining_descendants(selector_indexes)
            }
    
    def _grow_tree(self,df : DataFrame,tuple_target_attribute_value: tuple,selectors: list[Selector], pattern:Pattern, pattern_appearance: Series) -> None:
        """ Recurssive method to grow the tree of patterns.
        :param df: the dataset.
        :param tuple_target_attribute_value: the tuple which contains the target attribute name and the target attribute values.
        :param selectors: the list of candidate selectors for the current iteration.
        :param pattern: the current pattern (node of the tree).
        :param pattern_appearance: the appearance of the current pattern in the dataset.
        """
        
        # We count the number of visited subgroups.
        self._visited_subgroups += 1
        coverage = pattern_appearance.sum() / len(pattern_appearance)
        # If the pattern does not appear in the database (coverage = 0), we prune this branch.
        if coverage == 0:
            self._pruned_subgroups += 1
            return
        # We check if the pattern meets the minimum coverage threshold.
        meets_coverage_threshold = coverage >= self._thresholds["coverage"]
        # Since the coverage is antimonotonic, we can prune the branch if the rank of this pattern is 0 and
        # we look for patterns with a minimum rank greater than 0.
        if self._min_rank > 0 and not meets_coverage_threshold:
            self._pruned_subgroups += 1
            return
        # We evaluate the pattern and compute its rank.
        rank = self._handle_individual_result(df[tuple_target_attribute_value[0]] == tuple_target_attribute_value[1], pattern, pattern_appearance)
        # If the rank of the pattern is the maximum possible rank, we add this pattern to the graph and prune this branch, since all the descendants of this pattern will be discarded when checking redundancies.
        if rank == len(BerryFinder._credibility_criterions):
            self._add_to_graph(pattern, rank)
            return
        # If we have not pruned the branch and we have not reached the maximum depth, we continue growing the tree.
        if len(pattern) < self._max_complexity:
            for i in range(len(selectors)):
                # Pattern with the new selector.
                new_pattern = pattern.copy()
                new_pattern.add_selector(selectors[i])
                # The new pattern appearance is the intersection of the appearance of the current pattern and the appearance of the new selector.
                new_pattern_appearance = pattern_appearance & self._selector_appearances[selectors[i]]
                # We do not use the full list of patterns in each call to avoid repeating the same patterns.
                self._grow_tree(df, tuple_target_attribute_value, selectors[i+1:], new_pattern, new_pattern_appearance)
        # We only add the pattern to the graph if it meets the minimum rank threshold.
        if rank >= self._min_rank:
            # Update the number of credible subgroups.
            self._credible_subgroups += 1
            # We add the pattern to the graph of subgroups.
            self._add_to_graph(pattern, rank)

    def _check_redundancy(self, node : BFinderNode, tuple_target_attribute_value : tuple) -> None:
        """Method to remove redundancies from the graph of subgroups. This method checks redundancies for all descendants of the node given as a parameter and outputs the subgroups if it is not discarded.
        :param node: the node to be checked for redundancies.
        :param tuple_target_attribute_value: the tuple which contains the target attribute name and the target attribute values.
        """
        # Descendants of the current node.
        descendants = node.descendant_nodes
        # Check redundancies for all descendants of the current node.
        for x in descendants:
            if not x.redundancies_checked:
                self._check_redundancy(x, tuple_target_attribute_value)
        # Maximum rank found in the descendants of the current node.
        max_rank = -1
        for x in descendants:
            max_rank = max(max_rank, x.rank, x.max_descendant_rank)
        # Set the maximum rank of the current node.
        node.max_descendant_rank = max_rank
        # Output the node if it is not discarded.
        if not node.is_discarded() and self._file is not None:
            pattern = Pattern([self._selectors[i] for i in node.selector_indexes])
            subgroup = Subgroup(pattern, Selector(tuple_target_attribute_value[0],Operator.EQUAL,tuple_target_attribute_value[1]))
            self._file.write(f"{str(subgroup)} ; Rank : {node.rank}\n")
        # Update the selected subgroups counter.
        if not node.is_discarded():
            self._selected_subgroups += 1
        # Mark the node as checked for redundancies.
        node.redundancies_checked = True

    def fit(self, pandas_dataframe: DataFrame, tuple_target_attribute_value: tuple) -> None:
        """Main method to run the QFinder algorithm. This algorithm only supports nominal attributes (i.e., type 'str'). IMPORTANT: missing values are not supported yet.
        
        :param data: the DataFrame which is scanned. This algorithm only supports nominal attributes (i.e., type 'str'). IMPORTANT: missing values are not supported yet.
        :param target: a tuple with 2 elements: the target attribute name and the target value.
        """
        if type(pandas_dataframe) != DataFrame:
            raise TypeError("The dataset must be a pandas DataFrame.")
        if type(tuple_target_attribute_value) != tuple:
            raise TypeError("The target must be a tuple.")
        for column in pandas_dataframe.columns:
            if not is_string_dtype(pandas_dataframe[column]):
                raise DatasetAttributeTypeError("Error in attribute '" + str(column) + "'. This algorithm only supports nominal attributes (i.e., type 'str').")
        if tuple_target_attribute_value[0] not in pandas_dataframe.columns:
            raise ValueError("The target attribute must be in the dataset.")
        if tuple_target_attribute_value[1] not in pandas_dataframe[tuple_target_attribute_value[0]].unique():
            raise ValueError("The target value must be in the target attribute.")
        # We compute TP, FP and N for this dataset and target
        self._TP = (pandas_dataframe[tuple_target_attribute_value[0]] == tuple_target_attribute_value[1]).sum()
        self._FP = len(pandas_dataframe) - self._TP
        self._N = len(pandas_dataframe)
        # We copy the DataFrame to avoid modifying the original when dealing with "other" values.
        df = pandas_dataframe.copy()
        # We reduce the number of categories per column according to 'cats' and generate the list of selectors.
        self._reduce_categories(df, tuple_target_attribute_value)
        # If max_complexity = -1, we take all non-empty patterns. For this reason, we set max_complexity = len(df.columns) - 1 (we do not consider the target attribute).
        if self._max_complexity == -1:
            self._max_complexity = len(df.columns) - 1
        self._selectors = self._generate_selectors(df, tuple_target_attribute_value)
        self._root_node = BFinderNode(set(range(len(self._selectors))),-1)
        self._grow_tree(df, tuple_target_attribute_value, self._selectors, Pattern([]), Series(True, index = df.index))
        if self._file_path is not None:
            self._file = open(self._file_path,"w")
        self._check_redundancy(self._root_node, tuple_target_attribute_value)
        if self._file is not None:
            self._file.close()




