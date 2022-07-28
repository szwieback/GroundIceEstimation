import pyximport; pyximport.install(language_level=3)

from simulation.stefan import stefan_integral_balance
from simulation.stratigraphy import (
    StefanStratigraphySmoothingSpline, StratigraphyMultiple, StefanStratigraphyConstantE,
    StefanStratigraphyPrescribedConstantE)