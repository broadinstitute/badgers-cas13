"""Define landscape for the Cas13a guide design task"""
import os
from typing import Dict

import numpy as np
from flexs import Landscape


class Cas13Landscape(Landscape):
    """
    Defines landscape for the Cas13a guide design task

    We do not have a 'ground truth' value for the fitness of each sequence 
    (because it would not be possible to experimentally test the performance each of the guide-target pairs evaluated during the design process.
    So, a nan fitness value is returned instead.
    """

    def __init__(self):
        """
        Create a Cas13 binding landscape from the test dataset
        """
        super().__init__(name="Cas13Landscape")


    def _fitness_function(self, sequences) -> np.ndarray:
        return np.array([np.nan] * len(sequences))