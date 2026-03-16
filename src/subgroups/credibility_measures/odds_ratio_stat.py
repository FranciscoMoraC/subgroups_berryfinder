# -*- coding: utf-8 -*-

# Contributors:
#    Francisco Mora-Caselles: <fmora@um.es>

"""This file contains the implementation of odds ratio credibility measure computed using the contingency table.
"""

from subgroups.credibility_measures.credibility_measure import CredibilityMeasure
from subgroups.exceptions import ParameterNotFoundError
from math import inf

# Python annotations.
from typing import Union

class OddsRatioStatistic(CredibilityMeasure):
    """This class defines the odds ratio credibility measure computed using the contingency table.
    """

    _singleton = None
    __slots__ = ("_threshold","_initialized")

    def __new__(cls, threshold: float = None) -> 'OddsRatioStatistic':
        if OddsRatioStatistic._singleton is None:
            OddsRatioStatistic._singleton = object().__new__(cls)
        elif threshold is not None:
            OddsRatioStatistic._singleton._threshold = threshold
        return OddsRatioStatistic._singleton
    
    def __init__(self, threshold: float = None) -> None:
        """Constructor of the class OddsRatioStatistic.

        :param threshold: threshold for the credibility measure (default: None).
        """
        if getattr(self, "_initialized", False):
            return
        self._threshold = threshold
        self._initialized = True
    
    def compute(self, dict_of_parameters: dict[str, int | float]) -> float:
        """Method to compute the odds ratio credibility measure using the contingency table (you can also call to the instance for this purpose).

        :param dict_of_parameters: python dictionary which contains all the necessary parameters used to compute this credibility measure.
        :return: the computed value for the odds ratio credibility measure.
        """

        if type(dict_of_parameters) is not dict:
            raise TypeError("The type of the parameter 'dict_of_parameters' must be 'dict'.")
        # Required parameters for the computation of the credibility measure. We need either 'tp', 'fp', 'TP' and 'FP' or 'appearance' and 'target_appearance'.
        if ("tp" not in dict_of_parameters or "fp" not in dict_of_parameters or "TP" not in dict_of_parameters or "FP" not in dict_of_parameters) and ("appearance" not in dict_of_parameters or "target_appearance" not in dict_of_parameters):
            raise ParameterNotFoundError("All the parameters 'tp', 'fp', 'TP' and 'FP' or 'appearance' and 'target_appearance' must be included in 'dict_of_parameters'.")
        # If the base statistics are provided, we use them to compute the odds ratio.
        if "tp" in dict_of_parameters:
            tp = dict_of_parameters["tp"]
            fp = dict_of_parameters["fp"]
            TP = dict_of_parameters["TP"]
            FP = dict_of_parameters["FP"]
        # If the appearance and target appearance vectors are provided, we compute the base statistics from them.
        else:
            appearance = dict_of_parameters["appearance"]
            target_appearance = dict_of_parameters["target_appearance"]
            tp = (appearance & target_appearance).sum()
            fp = (appearance & ~target_appearance).sum()
            TP = target_appearance.sum()
            FP = (~target_appearance).sum()
        # If the pattern covers all instances, we assign the minimum possible odds ratio so the pattern is not selected.
        if tp == TP and fp == FP:
            return 0
        # If the pattern covers all positive instances, we assign the maximum possible odds ratio.
        if tp == TP:
            return inf
        # If the pattern covers all negative instances but not all instances, we assign the minimum possible odds ratio so the pattern is not selected.
        if fp == FP:
            return 0
        # If the pattern does not cover any instance, we assign the minimum possible odds ratio so the pattern is not selected.
        if tp == 0 and fp == 0:
            return 0
        # If the pattern only covers positive instances, we assign the maximum possible odds ratio.
        if fp == 0:
            return inf
        return (tp/fp)/((TP-tp)/(FP-fp))
    
    def get_name(self) -> str:
        """Method to get the credibility measure name (equal to the class name).
        """
        return "OddsRatioStatistic"
    
    def __call__(self, dict_of_parameters: dict[str, int | float]) -> bool:
        """Compute the odds ratio credibility measure using the contingency table.
        :param dict_of_parameters: python dictionary which contains all the necessary parameters used to compute this credibility measure.
        :return: True if the credibility measure meets the threshold, False otherwise.
        """
        # print("Computing odds ratio with threshold", self._threshold, "and value ", self.compute(dict_of_parameters))
        if self._threshold == 0 and self.compute(dict_of_parameters) == 0:
            print("0,0 case. Rv is", self.compute(dict_of_parameters) >= self._threshold) 
        if self._threshold is None:
            raise ValueError("The threshold for the odds ratio credibility measure is not set.")
        return self.compute(dict_of_parameters) >= self._threshold