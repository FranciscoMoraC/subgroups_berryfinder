"""ExCover: exhaustive covering for non-redundant discriminative itemsets."""

from typing import Union

from pandas import DataFrame

from subgroups.algorithms.algorithm import Algorithm
from subgroups.core.operator import Operator
from subgroups.core.pattern import Pattern
from subgroups.core.selector import Selector
from subgroups.core.subgroup import Subgroup
from subgroups.quality_measures.f1_score import F1Score


class ExCover(Algorithm):
    """Find patterns that cover at least one positive transaction best.

    Patterns are closed on the positive transactions. The default quality
    measure is F1 score; custom callable quality measures may be supplied.
    """

    __slots__ = (
        "_quality_measure", "_cats", "_max_depth", "_file_path",
        "_top_patterns", "_visited_subgroups", "_pruned_subgroups",
        "_selector_appearances", "_target", "_target_name", "_positive_count",
        "_negative_count", "_df", "_candidate_patterns", "_selectors",
        "_attribute_selectors",
    )

    def __init__(self, quality_measure=None, cats: int = -1,
                 max_depth: int = -1, write_results_in_file: bool = False,
                 file_path: Union[str, None] = None) -> None:
        if quality_measure is None:
            quality_measure = F1Score()
        if not callable(quality_measure):
            raise TypeError("The parameter 'quality_measure' must be callable.")
        if type(cats) is not int:
            raise TypeError("The type of the parameter 'cats' must be 'int'.")
        if type(max_depth) is not int:
            raise TypeError("The type of the parameter 'max_complexity' must be 'int'.")
        if type(write_results_in_file) is not bool:
            raise TypeError("The type of the parameter 'write_results_in_file' must be 'bool'.")
        if file_path is not None and type(file_path) is not str:
            raise TypeError("The type of the parameter 'file_path' must be 'str' or 'NoneType'.")
        if cats == 0 or cats < -1:
            raise ValueError("The parameter 'cats' must be greater than zero or equal to -1.")
        if max_depth == 0 or max_depth < -1:
            raise ValueError("The parameter 'max_depth' must be greater than zero or equal to -1.")
        if write_results_in_file and file_path is None:
            raise ValueError("If 'write_results_in_file' is True, 'file_path' must not be None.")
        self._quality_measure = quality_measure
        self._cats = cats
        self._max_depth = max_depth
        self._file_path = file_path if write_results_in_file else None
        self._top_patterns = []
        self._visited_subgroups = 0
        self._pruned_subgroups = 0

    def _get_top_patterns(self) -> list[Pattern]:
        return self._top_patterns.copy()

    def _get_selected_subgroups(self) -> int:
        return len(self._top_patterns)

    top_patterns = property(_get_top_patterns)
    selected_subgroups = property(_get_selected_subgroups)
    visited_subgroups = property(lambda self: self._visited_subgroups)
    pruned_subgroups = property(lambda self: self._pruned_subgroups)

    def _reduce_categories(self, dataframe: DataFrame, target: tuple) -> DataFrame:
        if self._cats == -1:
            return dataframe
        result = dataframe.copy()
        for column in result.columns:
            if column == target[0] or result[column].nunique() <= self._cats:
                continue
            counts = result[column].value_counts()
            retained = set(counts.nlargest(self._cats).index)
            other = "other"
            while other in counts.index:
                other += "_"
            result.loc[~result[column].isin(retained), column] = other
        return result

    def _quality(self, mask) -> float:
        tp = int((mask & self._target).sum())
        fp = int(mask.sum()) - tp
        return self._quality_measure({"tp": tp, "fp": fp, "TP": self._positive_count})

    def _upper_bound(self, mask) -> float:
        """Return the F1 upper bound after replacing all future FP by zero."""
        if not isinstance(self._quality_measure, F1Score):
            return float("inf")
        true_positives = int((mask & self._target).sum())
        return (2 * true_positives) / (self._positive_count + true_positives)

    def _closure(self, mask) -> Pattern:
        positive_mask = mask & self._target
        return Pattern([
            selector for selector, appearance in self._selector_appearances.items()
            if bool(appearance[positive_mask].all())
        ])

    def _add(self, pattern: Pattern, mask, quality: float) -> None:
        for index in mask[mask & self._target].index:
            candidates = self._candidate_patterns[index]
            if not candidates or quality > candidates[0][1]:
                self._candidate_patterns[index] = [(pattern, quality)]
            elif quality == candidates[0][1] and not any(
                    candidate.is_refinement(pattern, False) for candidate, _ in candidates):
                candidates.append((pattern, quality))

    def _visit(self, pattern: Pattern) -> None:
        self._visited_subgroups += 1
        closed_mask = pattern.is_contained(self._df.drop(columns=[self._target_name]))
        positive_support = int((closed_mask & self._target).sum())
        negative_support = int((closed_mask & ~self._target).sum())
        if (positive_support / self._positive_count) >= (negative_support / self._negative_count):
            self._add(pattern, closed_mask, self._quality(closed_mask))

    def _should_prune(self, mask) -> bool:
        covered_positive = mask & self._target
        if not bool(covered_positive.any()):
            self._pruned_subgroups += 1
            return True
        upper_bound = self._upper_bound(mask)
        covered_indices = covered_positive[covered_positive].index
        existing_candidates = [
            self._candidate_patterns[index][0][1]
            for index in covered_indices if self._candidate_patterns[index]
        ]
        if existing_candidates:
            best_existing = min(existing_candidates)
            if upper_bound < best_existing:
                self._pruned_subgroups += 1
                return True
        return False

    def _enumerate(self, last_index: int, mask, core_selectors: list[Selector], depth: int) -> None:
        if self._max_depth != -1 and depth >= self._max_depth:
            self._pruned_subgroups += 1
            return
        candidate_indices = range(len(self._selectors)) if last_index == -1 else range(last_index)
        for index in candidate_indices:
            selector = self._selectors[index]
            if selector.attribute_name in {item.attribute_name for item in core_selectors}:
                continue
            next_mask = mask & self._selector_appearances[selector]
            if not bool(next_mask.any()):
                self._pruned_subgroups += 1
                continue
            if self._should_prune(next_mask):
                continue
            closure = self._closure(next_mask)
            closure_indices = [self._selectors.index(item) for item in closure]
            # SPC keeps only closures whose suffix ends at the core selector.
            existing_selectors = set(core_selectors)
            new_successors = [
                closure_item for closure_item, closure_index in zip(closure, closure_indices)
                if closure_index > index and closure_item not in existing_selectors
            ]
            if new_successors:
                self._pruned_subgroups += 1
                continue
            self._visit(closure)
            closed_mask = closure.is_contained(self._df.drop(columns=[self._target_name]))
            self._enumerate(index, closed_mask, core_selectors + list(closure), depth + 1)

    def fit(self, pandas_dataframe: DataFrame, tuple_target_attribute_value: tuple) -> None:
        if type(pandas_dataframe) is not DataFrame:
            raise TypeError("The dataset must be a pandas DataFrame.")
        if type(tuple_target_attribute_value) is not tuple or len(tuple_target_attribute_value) != 2:
            raise TypeError("The target must be a tuple with two elements.")
        target_name, target_value = tuple_target_attribute_value
        if target_name not in pandas_dataframe.columns:
            raise KeyError(target_name)
        self._df = self._reduce_categories(pandas_dataframe, tuple_target_attribute_value)
        self._target_name = target_name
        self._target = self._df[target_name] == target_value
        self._positive_count = int(self._target.sum())
        if self._positive_count == 0:
            raise ValueError("The target class must occur at least once.")
        self._negative_count = len(self._df) - self._positive_count
        if self._negative_count == 0:
            raise ValueError("The dataset must contain at least one negative example.")
        self._selector_appearances = {}
        self._attribute_selectors = {}
        for column in self._df.columns:
            if column == target_name:
                continue
            self._attribute_selectors[column] = []
            for value in self._df[column].unique():
                selector = Selector(column, Operator.EQUAL, value)
                self._attribute_selectors[column].append(selector)
                self._selector_appearances[selector] = self._df[column] == value
        self._selectors = [selector for selectors in self._attribute_selectors.values() for selector in selectors]
        self._selectors.sort(key=lambda selector: self._quality(self._selector_appearances[selector]), reverse=True)
        self._candidate_patterns = {index: [] for index in self._df.index} # L
        self._visited_subgroups = 0
        self._pruned_subgroups = 0
        self._enumerate(-1, self._df[target_name].notna(), [], 0)
        self._top_patterns = []
        for candidates in self._candidate_patterns.values():
            for pattern, _ in candidates:
                if pattern not in self._top_patterns:
                    self._top_patterns.append(pattern)
        if self._file_path is not None:
            target = Selector(target_name, Operator.EQUAL, target_value)
            with open(self._file_path, "w") as output:
                for pattern in self._top_patterns:
                    output.write(str(Subgroup(pattern, target)) + "\n")