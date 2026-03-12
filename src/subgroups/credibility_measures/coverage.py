# -*- coding: utf-8 -*-

# Contributors:
#    Antonio López Martínez-Carrasco <antoniolopezmc1995@gmail.com>
#    Francisco Mora-Caselles: <fmora@um.es>

"""This file contains the implementation of the Coverage credibility measure.
"""

from subgroups.credibility_measures.credibility_measure import CredibilityMeasure
from subgroups.exceptions import SubgroupParameterNotFoundError

# Python annotations.
from typing import Union

class Coverage(CredibilityMeasure):
    """This class defines the Coverage credibility measure.
    """
    
    _singleton = None
    __slots__ = ("_threshold",)
    
    def __new__(cls, threshold: float = None) -> 'Coverage':
        if Coverage._singleton is None:
            Coverage._singleton = object().__new__(cls)
        return Coverage._singleton

    def __init__(self, threshold: float = None) -> None:
        """Constructor of the class Coverage.

        :param threshold: threshold for the credibility measure (default: None).
        """
        self._threshold = threshold

    def compute(self, dict_of_parameters : dict[str, Union[int, float]]) -> float:
        """Method to compute the Coverage credibility measure (you can also call to the instance for this purpose).
        
        :param dict_of_parameters: python dictionary which contains all the necessary parameters used to compute this credibility measure.
        :return: the computed value for the Coverage credibility measure.
        """
        if type(dict_of_parameters) is not dict:
            raise TypeError("The type of the parameter 'dict_of_parameters' must be 'dict'.")
        if ("tp" not in dict_of_parameters):
            raise SubgroupParameterNotFoundError("The subgroup parameter 'tp' is not in 'dict_of_parameters'.")
        if ("fp" not in dict_of_parameters):
            raise SubgroupParameterNotFoundError("The subgroup parameter 'fp' is not in 'dict_of_parameters'.")
        if ("TP" not in dict_of_parameters):
            raise SubgroupParameterNotFoundError("The subgroup parameter 'TP' is not in 'dict_of_parameters'.")
        if ("FP" not in dict_of_parameters):
            raise SubgroupParameterNotFoundError("The subgroup parameter 'FP' is not in 'dict_of_parameters'.")
        tp = dict_of_parameters["tp"]
        fp = dict_of_parameters["fp"]
        TP = dict_of_parameters["TP"]
        FP = dict_of_parameters["FP"]
        return ( tp + fp ) / ( TP + FP )
    
    def get_name(self) -> str:
        """Method to get the quality measure name (equal to the class name).
        """
        return "Coverage"
    
    def optimistic_estimate_of(self) -> dict[str, CredibilityMeasure]:
        """Method to get a python dictionary with the quality measures of which this one is an optimistic estimate.
        
        :return: a python dictionary in which the keys are the quality measure names and the values are the instances of those quality measures.
        """
        return dict()
    
    def __call__(self, dict_of_parameters : dict[str, Union[int, float]]) -> bool:
        """Compute the Coverage credibility measure and checks if it meets the threshold.
        
        :param dict_of_parameters: python dictionary which contains all the needed parameters with which to compute this credibility measure.
        :return: wether the computed value for the Coverage credibility measure meets the threshold or not.
        """
        if self._threshold is None:
            raise ValueError("The threshold for the coverage credibility measure is not set.")
        return self.compute(dict_of_parameters) >= self._threshold