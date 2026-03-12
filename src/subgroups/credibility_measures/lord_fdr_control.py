# -*- coding: utf-8 -*-

# Contributors:
#    Francisco Mora-Caselles: <fmora@um.es>

"""This file contains the implementation of the LORD FDR control that can be applied to the significance credibility measure computed using statistical independece hypothesis test.
This strategy was proposed in the paper "On Online Control of False Discovery Rate" by A. Javanmard et al. (https://www.jstor.org/stable/26542797).
"""

# Python annotations.
from typing import Union


class LordFDRControl:
    """This class defines the LORD FDR control that can be applied to the significance credibility measure computed using statistical independece hypothesis test.
    """

    _singleton = None
    __slots__ = ("_w0", "_b0", "_test_number", "_last_rejections", "_max_size", "_waiting_for_rejection")

    def __new__(cls, w0 : float  = 0.025, b0 : float = 0.025, max_size : int = 5) -> 'LordFDRControl':
        if LordFDRControl._singleton is None:
            LordFDRControl._singleton = object().__new__(cls)
        return LordFDRControl._singleton

    def __init__(self, w0 : float  = 0.025, b0 : float = 0.025, max_size : int = 5) -> None:
        """Constructor of the class LordFDRControl.

        :param w0: initial weight for LORD FDR control (default: 0.025).
        :param b0: initial budget for LORD FDR control (default: 0.025).
        :param max_size: maximum size of the queue for storing last rejections (default: 5).
        """
        self._w0 = w0
        self._b0 = b0
        self._test_number = 0
        self._last_rejections = []
        self._max_size = max_size
        self._waiting_for_rejection = False

    def _gamma(self, i: int) -> float:
        """Method to compute the value of gamma_i.

        :param i: the index of the test.
        :return: the computed value for gamma_i.
        """
        return 1/2**i
    
    def get_threshold(self) -> float:
        """Method to compute the threshold for the next test.

        :return: the computed threshold for the next test.
        """
        if self._waiting_for_rejection:
            raise Exception("LORD FDR control is waiting for an answer for the previous test. You cannot compute the threshold for the next test until a rejection/non rejection is made.")
        self._test_number += 1
        curr_gamma = self._gamma(self._test_number)
        self._waiting_for_rejection = True
        rv =  self._w0 * curr_gamma
        if len(self._last_rejections) == 0:
            return rv
        for i in range(len(self._last_rejections)):
            rv += self._b0 * self._gamma(self._test_number - self._last_rejections[i])
        return rv
    
    def rejection(self, rejected: bool) -> None:
        """Method to update the internal state of the LORD FDR control after a test is performed.

        :param rejected: boolean value indicating whether the null hypothesis has been rejected or not.
        """
        if not self._waiting_for_rejection:
            raise Exception("LORD FDR control is not waiting for an answer for the previous test. You cannot update the internal state until a threshold for the next test is computed.")
        if rejected:
            self._last_rejections.append(self._test_number)
            if len(self._last_rejections) > self._max_size:
                self._last_rejections.pop(0)
        self._waiting_for_rejection = False




