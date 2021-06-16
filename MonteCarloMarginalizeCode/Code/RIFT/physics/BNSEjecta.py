import numpy as np
import lal
import lalsimulation as lalsim
import scipy.constants as constants
from functools import lru_cache

from . import EOSManager

class EjectaManager:

    def __init__(self, eos_name):
        self.eos_name = eos_name
        self.eos = EOSManager.EOSLALSimulation(self.eos_name)
        self.eos_fam = self.eos.eos_fam

    def _radius(self, m):
        # lalsimulation expects mass in kg, assume we're given mass in Msun
        return lalsim.SimNeutronStarRadius(m * lal.MSUN_SI, self.eos_fam)

    def _tidal_deformability(self, m):
        return self.eos.lambda_from_m(m)

    @lru_cache(maxsize=32)
    def _compactness(self, m, r=None):
        if r is None:
            r = self._radius(m)
        return constants.G * m * lal.MSUN_SI / (r * constants.c**2)

    def mass_dynamical(self, m1, m2, r1=None, r2=None):
        # Eq. 6 in Kruger and Foucart (2020) http://arxiv.org/abs/2002.07728
        a = -9.3335
        b = 114.17
        c = -337.56
        n = 1.5465
        C1 = self._compactness(m1, r=r1)
        C2 = self._compactness(m2, r=r2)
        return 1.e-3 * ((a / C1 + b * (m2 / m1)**n + c * C1) * m1 + (a / C2 + b * (m1 / m2)**n + c * C2) * m2)

    def mass_disk(self, m1, m2, r1=None, r2=None):
        # Eq. 4 in Kruger and Foucart (2020) http://arxiv.org/abs/2002.07728 (doesn't include m2 term)
        # or Eq. 18 in Nedora et al. (2020) http://arxiv.org/abs/2011.11110 (does include m2 term)
        a = -8.1324
        c = 1.4820
        d = 1.7784
        C1 = self._compactness(m1, r=r1)
        C2 = self._compactness(m2, r=r2)
        return m1 * max(5.e-4, (a * C1 + c)**d) + m2 * max(4.e-5, (a * C2 + c)**d)

    def velocity_dynamical(self, m1, m2, r1=None, r2=None):
        # Eq. 22 in Radice et al. (2018) https://iopscience.iop.org/article/10.3847/1538-4357/aaf054
        a = -0.287
        b = 0.494
        c = -3.
        C1 = self._compactness(m1, r=r1)
        C2 = self._compactness(m2, r=r2)
        return a * (m1 / m2) * (1. + c * C1) + a * (m2 / m1) * (1. + c * C2) + b

    def velocity_disk(self, m1, m2, r1=None, r2=None):
        # I'm having a hard time finding anything about disk/wind velocity in the literature, so for now I'm hard-coding it to 0.1
        return 0.1
