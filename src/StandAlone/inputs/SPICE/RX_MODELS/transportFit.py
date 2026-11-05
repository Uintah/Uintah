#!/usr/bin/env python3
"""
transportFit.py
=================

Reads a Cantera-format YAML mechanism file and computes the same 5-coefficient
transport polynomials that Cantera's GasTransport class generates internally
(viscosity, thermal conductivity, binary diffusion coefficients) -- entirely
from the raw kinetic-theory parameters in the YAML file (Lennard-Jones
diameter/well-depth, dipole moment, polarizability, rotational relaxation
number, geometry) plus the NASA7 thermo polynomials.

This is a from-scratch reimplementation of Cantera's algorithm
(src/transport/GasTransport.cpp, src/transport/MMCollisionInt.cpp). It does
NOT import or call Cantera. The physics is the standard Chapman-Enskog /
Hirschfelder-Curtiss-Bird kinetic theory with the Monchick & Mason (1961)
tabulated collision integrals and the modified-Eucken thermal-conductivity
correction used in the CHEMKIN transport report (Kee, Dixon-Lewis, Warnatz,
Coltrin & Miller, SAND86-8246) and reproduced in Kee, Coltrin & Glarborg,
"Chemically Reacting Flow".

Units: everything is handled internally in SI. The YAML "customary" units
(diameter in Angstrom, well-depth in Kelvin, dipole in Debye, polarizability
in cubic Angstrom) are converted on read, exactly as Cantera's
GasTransportData::setCustomaryUnits does.

Usage
-----
    python3 transportFit.py mechanism.yaml
    python3 transportFit.py mechanism.yaml --tmin 300 --tmax 3000 --json out.json

Written by James Karr July 2026
"""

import argparse
import json
import math
import sys

import numpy as np
import yaml

# --------------------------------------------------------------------------
# Physical constants (matching Cantera's ct_defs.h / physical_constants)
# --------------------------------------------------------------------------
BOLTZMANN = 1.380649e-23        # J/K
AVOGADRO = 6.02214076e23        # 1/mol
GAS_CONSTANT = BOLTZMANN * AVOGADRO   # J/(mol K)
PI = math.pi
EPSILON_0 = 8.8541878128e-12    # F/m (vacuum permittivity)
LIGHT_SPEED = 2.99792458e8      # m/s

# "Customary unit" conversions used by Cantera's GasTransportData
ANGSTROM = 1.0e-10              # m
CUBIC_ANGSTROM = 1.0e-30        # m^3
DEBYE_TO_CM = 1.0e-21 / LIGHT_SPEED   # Debye -> C*m (Cantera's exact conversion)

# Standard atomic weights [g/mol] -- IUPAC 2021 conventional values, enough
# for typical combustion mechanisms. If the YAML "elements" section provides
# custom atomic weights (isotope-specific mechanisms), those take priority.
ATOMIC_WEIGHTS = {
    'H': 1.00794, 'D': 2.01410, 'He': 4.002602, 'Li': 6.941, 'Be': 9.012182,
    'B': 10.811, 'C': 12.0107, 'N': 14.0067, 'O': 15.9994, 'F': 18.9984032,
    'Ne': 20.1797, 'Na': 22.98976928, 'Mg': 24.305, 'Al': 26.9815386,
    'Si': 28.0855, 'P': 30.973762, 'S': 32.065, 'Cl': 35.453, 'Ar': 39.948,
    'K': 39.0983, 'Ca': 40.078, 'Ti': 47.867, 'Cr': 51.9961, 'Mn': 54.938045,
    'Fe': 55.845, 'Ni': 58.6934, 'Cu': 63.546, 'Zn': 65.38, 'Br': 79.904,
    'Kr': 83.798, 'Rb': 85.4678, 'Sr': 87.62, 'Y': 88.90585, 'Zr': 91.224,
    'Mo': 95.96, 'Ag': 107.8682, 'Cd': 112.411, 'Sn': 118.71, 'I': 126.90447,
    'Xe': 131.293, 'Cs': 132.9054519, 'Ba': 137.327, 'W': 183.84,
    'Pt': 195.084, 'Au': 196.966569, 'Hg': 200.59, 'Pb': 207.2,
    'E': 0.000548579909,  # electron
}

COLL_INT_POLY_DEGREE = 8  # unused directly here (see note in README section)
NP_FIT_POINTS = 50         # number of T points Cantera samples when fitting
FIT_DEGREE = 4             # 5 coefficients (non-CK mode)


# ==========================================================================
# Monchick & Mason (1961) tabulated reduced collision integrals
# Exact tables transcribed from Cantera's MMCollisionInt.cpp
# ==========================================================================

DELTA_GRID = np.array([0.0, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0, 2.5])

TSTAR22 = np.array([
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
    1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0,
    5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 14.0, 16.0,
    18.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0, 75.0, 100.0
])

OMEGA22_TABLE = np.array([
    [4.1005, 4.266, 4.833, 5.742, 6.729, 8.624, 10.34, 11.89],
    [3.2626, 3.305, 3.516, 3.914, 4.433, 5.57, 6.637, 7.618],
    [2.8399, 2.836, 2.936, 3.168, 3.511, 4.329, 5.126, 5.874],
    [2.531, 2.522, 2.586, 2.749, 3.004, 3.64, 4.282, 4.895],
    [2.2837, 2.277, 2.329, 2.46, 2.665, 3.187, 3.727, 4.249],
    [2.0838, 2.081, 2.13, 2.243, 2.417, 2.862, 3.329, 3.786],
    [1.922, 1.924, 1.97, 2.072, 2.225, 2.614, 3.028, 3.435],
    [1.7902, 1.795, 1.84, 1.934, 2.07, 2.417, 2.788, 3.156],
    [1.6823, 1.689, 1.733, 1.82, 1.944, 2.258, 2.596, 2.933],
    [1.5929, 1.601, 1.644, 1.725, 1.838, 2.124, 2.435, 2.746],
    [1.4551, 1.465, 1.504, 1.574, 1.67, 1.913, 2.181, 2.451],
    [1.3551, 1.365, 1.4, 1.461, 1.544, 1.754, 1.989, 2.228],
    [1.28, 1.289, 1.321, 1.374, 1.447, 1.63, 1.838, 2.053],
    [1.2219, 1.231, 1.259, 1.306, 1.37, 1.532, 1.718, 1.912],
    [1.1757, 1.184, 1.209, 1.251, 1.307, 1.451, 1.618, 1.795],
    [1.0933, 1.1, 1.119, 1.15, 1.193, 1.304, 1.435, 1.578],
    [1.0388, 1.044, 1.059, 1.083, 1.117, 1.204, 1.31, 1.428],
    [0.99963, 1.004, 1.016, 1.035, 1.062, 1.133, 1.22, 1.319],
    [0.96988, 0.9732, 0.983, 0.9991, 1.021, 1.079, 1.153, 1.236],
    [0.92676, 0.9291, 0.936, 0.9473, 0.9628, 1.005, 1.058, 1.121],
    [0.89616, 0.8979, 0.903, 0.9114, 0.923, 0.9545, 0.9955, 1.044],
    [0.87272, 0.8741, 0.878, 0.8845, 0.8935, 0.9181, 0.9505, 0.9893],
    [0.85379, 0.8549, 0.858, 0.8632, 0.8703, 0.8901, 0.9164, 0.9482],
    [0.83795, 0.8388, 0.8414, 0.8456, 0.8515, 0.8678, 0.8895, 0.916],
    [0.82435, 0.8251, 0.8273, 0.8308, 0.8356, 0.8493, 0.8676, 0.8901],
    [0.80184, 0.8024, 0.8039, 0.8065, 0.8101, 0.8201, 0.8337, 0.8504],
    [0.78363, 0.784, 0.7852, 0.7872, 0.7899, 0.7976, 0.8081, 0.8212],
    [0.76834, 0.7687, 0.7696, 0.7712, 0.7733, 0.7794, 0.7878, 0.7983],
    [0.75518, 0.7554, 0.7562, 0.7575, 0.7592, 0.7642, 0.7711, 0.7797],
    [0.74364, 0.7438, 0.7445, 0.7455, 0.747, 0.7512, 0.7569, 0.7642],
    [0.71982, 0.72, 0.7204, 0.7211, 0.7221, 0.725, 0.7289, 0.7339],
    [0.70097, 0.7011, 0.7014, 0.7019, 0.7026, 0.7047, 0.7076, 0.7112],
    [0.68545, 0.6855, 0.6858, 0.6861, 0.6867, 0.6883, 0.6905, 0.6932],
    [0.67232, 0.6724, 0.6726, 0.6728, 0.6733, 0.6743, 0.6762, 0.6784],
    [0.65099, 0.651, 0.6512, 0.6513, 0.6516, 0.6524, 0.6534, 0.6546],
    [0.61397, 0.6141, 0.6143, 0.6145, 0.6147, 0.6148, 0.6148, 0.6147],
    [0.5887, 0.5889, 0.5894, 0.59, 0.5903, 0.5901, 0.5895, 0.5885],
])

# tstar grid for astar/bstar/cstar has an extra leading (0.0) and trailing
# (500.0) point; the runtime interpolation only ever uses the interior 37
# points, which line up exactly with TSTAR22.
ASTAR_TABLE = np.array([
    [1.0231, 1.0660, 1.0380, 1.0400, 1.0430, 1.0500, 1.0520, 1.0510],
    [1.0424, 1.0450, 1.0480, 1.0520, 1.0560, 1.0650, 1.0660, 1.0640],
    [1.0719, 1.0670, 1.0600, 1.0550, 1.0580, 1.0680, 1.0710, 1.0710],
    [1.0936, 1.0870, 1.0770, 1.0690, 1.0680, 1.0750, 1.0780, 1.0780],
    [1.1053, 1.0980, 1.0880, 1.0800, 1.0780, 1.0820, 1.0840, 1.0840],
    [1.1104, 1.1040, 1.0960, 1.0890, 1.0860, 1.0890, 1.0900, 1.0900],
    [1.1114, 1.1070, 1.1000, 1.0950, 1.0930, 1.0950, 1.0960, 1.0950],
    [1.1104, 1.1070, 1.1020, 1.0990, 1.0980, 1.1000, 1.1000, 1.0990],
    [1.1086, 1.1060, 1.1020, 1.1010, 1.1010, 1.1050, 1.1050, 1.1040],
    [1.1063, 1.1040, 1.1030, 1.1030, 1.1040, 1.1080, 1.1090, 1.1080],
    [1.1020, 1.1020, 1.1030, 1.1050, 1.1070, 1.1120, 1.1150, 1.1150],
    [1.0985, 1.0990, 1.1010, 1.1040, 1.1080, 1.1150, 1.1190, 1.1200],
    [1.0960, 1.0960, 1.0990, 1.1030, 1.1080, 1.1160, 1.1210, 1.1240],
    [1.0943, 1.0950, 1.0990, 1.1020, 1.1080, 1.1170, 1.1230, 1.1260],
    [1.0934, 1.0940, 1.0970, 1.1020, 1.1070, 1.1160, 1.1230, 1.1280],
    [1.0926, 1.0940, 1.0970, 1.0990, 1.1050, 1.1150, 1.1230, 1.1300],
    [1.0934, 1.0950, 1.0970, 1.0990, 1.1040, 1.1130, 1.1220, 1.1290],
    [1.0948, 1.0960, 1.0980, 1.1000, 1.1030, 1.1120, 1.1190, 1.1270],
    [1.0965, 1.0970, 1.0990, 1.1010, 1.1040, 1.1100, 1.1180, 1.1260],
    [1.0997, 1.1000, 1.1010, 1.1020, 1.1050, 1.1100, 1.1160, 1.1230],
    [1.1025, 1.1030, 1.1040, 1.1050, 1.1060, 1.1100, 1.1150, 1.1210],
    [1.1050, 1.1050, 1.1060, 1.1070, 1.1080, 1.1110, 1.1150, 1.1200],
    [1.1072, 1.1070, 1.1080, 1.1080, 1.1090, 1.1120, 1.1150, 1.1190],
    [1.1091, 1.1090, 1.1090, 1.1100, 1.1110, 1.1130, 1.1150, 1.1190],
    [1.1107, 1.1110, 1.1110, 1.1110, 1.1120, 1.1140, 1.1160, 1.1190],
    [1.1133, 1.1140, 1.1130, 1.1140, 1.1140, 1.1150, 1.1170, 1.1190],
    [1.1154, 1.1150, 1.1160, 1.1160, 1.1160, 1.1170, 1.1180, 1.1200],
    [1.1172, 1.1170, 1.1170, 1.1180, 1.1180, 1.1180, 1.1190, 1.1200],
    [1.1186, 1.1190, 1.1190, 1.1190, 1.1190, 1.1190, 1.1200, 1.1210],
    [1.1199, 1.1200, 1.1200, 1.1200, 1.1200, 1.1210, 1.1210, 1.1220],
    [1.1223, 1.1220, 1.1220, 1.1220, 1.1220, 1.1230, 1.1230, 1.1240],
    [1.1243, 1.1240, 1.1240, 1.1240, 1.1240, 1.1240, 1.1250, 1.1250],
    [1.1259, 1.1260, 1.1260, 1.1260, 1.1260, 1.1260, 1.1260, 1.1260],
    [1.1273, 1.1270, 1.1270, 1.1270, 1.1270, 1.1270, 1.1270, 1.1280],
    [1.1297, 1.1300, 1.1300, 1.1300, 1.1300, 1.1300, 1.1300, 1.1290],
    [1.1339, 1.1340, 1.1340, 1.1350, 1.1350, 1.1340, 1.1340, 1.1320],
    [1.1364, 1.1370, 1.1370, 1.1380, 1.1390, 1.1380, 1.1370, 1.1350],
])

BSTAR_TABLE = np.array([
    [1.1960, 1.216, 1.237, 1.269, 1.285, 1.290, 1.297, 1.294],
    [1.2451, 1.257, 1.340, 1.389, 1.366, 1.327, 1.314, 1.278],
    [1.2900, 1.294, 1.272, 1.258, 1.262, 1.282, 1.290, 1.299],
    [1.2986, 1.291, 1.284, 1.278, 1.277, 1.288, 1.294, 1.297],
    [1.2865, 1.281, 1.276, 1.272, 1.277, 1.286, 1.292, 1.298],
    [1.2665, 1.264, 1.261, 1.263, 1.269, 1.284, 1.292, 1.298],
    [1.2455, 1.244, 1.248, 1.255, 1.262, 1.278, 1.289, 1.296],
    [1.2253, 1.225, 1.234, 1.240, 1.252, 1.271, 1.284, 1.295],
    [1.2078, 1.210, 1.216, 1.227, 1.242, 1.264, 1.281, 1.292],
    [1.1919, 1.192, 1.205, 1.216, 1.230, 1.256, 1.273, 1.287],
    [1.1678, 1.172, 1.181, 1.195, 1.209, 1.237, 1.261, 1.277],
    [1.1496, 1.155, 1.161, 1.174, 1.189, 1.221, 1.246, 1.266],
    [1.1366, 1.141, 1.147, 1.159, 1.174, 1.202, 1.231, 1.256],
    [1.1270, 1.130, 1.138, 1.148, 1.162, 1.191, 1.218, 1.242],
    [1.1197, 1.122, 1.129, 1.140, 1.149, 1.178, 1.205, 1.231],
    [1.1080, 1.110, 1.116, 1.122, 1.132, 1.154, 1.180, 1.205],
    [1.1016, 1.103, 1.107, 1.112, 1.120, 1.138, 1.160, 1.183],
    [1.0980, 1.099, 1.102, 1.106, 1.112, 1.127, 1.145, 1.165],
    [1.0958, 1.097, 1.099, 1.102, 1.107, 1.119, 1.135, 1.153],
    [1.0935, 1.094, 1.095, 1.097, 1.100, 1.109, 1.120, 1.134],
    [1.0925, 1.092, 1.094, 1.095, 1.098, 1.104, 1.112, 1.122],
    [1.0922, 1.092, 1.093, 1.094, 1.096, 1.100, 1.106, 1.115],
    [1.0922, 1.092, 1.093, 1.093, 1.095, 1.098, 1.103, 1.110],
    [1.0923, 1.092, 1.093, 1.093, 1.094, 1.097, 1.101, 1.106],
    [1.0923, 1.092, 1.092, 1.093, 1.094, 1.096, 1.099, 1.103],
    [1.0927, 1.093, 1.093, 1.093, 1.094, 1.095, 1.098, 1.101],
    [1.0930, 1.093, 1.093, 1.093, 1.094, 1.094, 1.096, 1.099],
    [1.0933, 1.094, 1.093, 1.094, 1.094, 1.095, 1.096, 1.098],
    [1.0937, 1.093, 1.094, 1.094, 1.094, 1.094, 1.096, 1.097],
    [1.0939, 1.094, 1.094, 1.094, 1.094, 1.095, 1.095, 1.097],
    [1.0943, 1.094, 1.094, 1.094, 1.095, 1.095, 1.096, 1.096],
    [1.0944, 1.095, 1.094, 1.094, 1.094, 1.095, 1.095, 1.096],
    [1.0944, 1.094, 1.095, 1.094, 1.094, 1.095, 1.096, 1.096],
    [1.0943, 1.095, 1.094, 1.094, 1.095, 1.095, 1.095, 1.095],
    [1.0941, 1.094, 1.094, 1.094, 1.094, 1.094, 1.094, 1.096],
    [1.0947, 1.095, 1.094, 1.094, 1.093, 1.093, 1.094, 1.095],
    [1.0957, 1.095, 1.094, 1.093, 1.092, 1.093, 1.093, 1.094],
])

CSTAR_TABLE = np.array([
    [0.88575, 0.8988, 0.8378, 0.8029, 0.7876, 0.7805, 0.7799, 0.7801],
    [0.87268, 0.8692, 0.8647, 0.8479, 0.8237, 0.7975, 0.7881, 0.7784],
    [0.85182, 0.8525, 0.8366, 0.8198, 0.8054, 0.7903, 0.7839, 0.782],
    [0.83542, 0.8362, 0.8306, 0.8196, 0.8076, 0.7918, 0.7842, 0.7806],
    [0.82629, 0.8278, 0.8252, 0.8169, 0.8074, 0.7916, 0.7838, 0.7802],
    [0.82299, 0.8249, 0.823, 0.8165, 0.8072, 0.7922, 0.7839, 0.7798],
    [0.82357, 0.8257, 0.8241, 0.8178, 0.8084, 0.7927, 0.7839, 0.7794],
    [0.82657, 0.828, 0.8264, 0.8199, 0.8107, 0.7939, 0.7842, 0.7796],
    [0.8311, 0.8234, 0.8295, 0.8228, 0.8136, 0.796, 0.7854, 0.7798],
    [0.8363, 0.8366, 0.8342, 0.8267, 0.8168, 0.7986, 0.7864, 0.7805],
    [0.84762, 0.8474, 0.8438, 0.8358, 0.825, 0.8041, 0.7904, 0.7822],
    [0.85846, 0.8583, 0.853, 0.8444, 0.8336, 0.8118, 0.7957, 0.7854],
    [0.8684, 0.8674, 0.8619, 0.8531, 0.8423, 0.8186, 0.8011, 0.7898],
    [0.87713, 0.8755, 0.8709, 0.8616, 0.8504, 0.8265, 0.8072, 0.7939],
    [0.88479, 0.8831, 0.8779, 0.8695, 0.8578, 0.8338, 0.8133, 0.799],
    [0.89972, 0.8986, 0.8936, 0.8846, 0.8742, 0.8504, 0.8294, 0.8125],
    [0.91028, 0.9089, 0.9043, 0.8967, 0.8869, 0.8649, 0.8438, 0.8253],
    [0.91793, 0.9166, 0.9125, 0.9058, 0.897, 0.8768, 0.8557, 0.8372],
    [0.92371, 0.9226, 0.9189, 0.9128, 0.905, 0.8861, 0.8664, 0.8484],
    [0.93135, 0.9304, 0.9274, 0.9226, 0.9164, 0.9006, 0.8833, 0.8662],
    [0.93607, 0.9353, 0.9329, 0.9291, 0.924, 0.9109, 0.8958, 0.8802],
    [0.93927, 0.9387, 0.9366, 0.9334, 0.9292, 0.9162, 0.905, 0.8911],
    [0.94149, 0.9409, 0.9393, 0.9366, 0.9331, 0.9236, 0.9122, 0.8997],
    [0.94306, 0.9426, 0.9412, 0.9388, 0.9357, 0.9276, 0.9175, 0.9065],
    [0.94419, 0.9437, 0.9425, 0.9406, 0.938, 0.9308, 0.9219, 0.9119],
    [0.94571, 0.9455, 0.9445, 0.943, 0.9409, 0.9353, 0.9283, 0.9201],
    [0.94662, 0.9464, 0.9456, 0.9444, 0.9428, 0.9382, 0.9325, 0.9258],
    [0.94723, 0.9471, 0.9464, 0.9455, 0.9442, 0.9405, 0.9355, 0.9298],
    [0.94764, 0.9474, 0.9469, 0.9462, 0.945, 0.9418, 0.9378, 0.9328],
    [0.9479, 0.9478, 0.9474, 0.9465, 0.9457, 0.943, 0.9394, 0.9352],
    [0.94827, 0.9481, 0.948, 0.9472, 0.9467, 0.9447, 0.9422, 0.9391],
    [0.94842, 0.9484, 0.9481, 0.9478, 0.9472, 0.9458, 0.9437, 0.9415],
    [0.94852, 0.9484, 0.9483, 0.948, 0.9475, 0.9465, 0.9449, 0.943],
    [0.94861, 0.9487, 0.9484, 0.9481, 0.9479, 0.9468, 0.9455, 0.943],
    [0.94872, 0.9486, 0.9486, 0.9483, 0.9482, 0.9475, 0.9464, 0.9452],
    [0.94881, 0.9488, 0.9489, 0.949, 0.9487, 0.9482, 0.9476, 0.9468],
    [0.94863, 0.9487, 0.9489, 0.9491, 0.9493, 0.9491, 0.9483, 0.9476],
])

LOG_TSTAR22 = np.log(TSTAR22)


def _fit_delta_row(table_row, degree=6):
    """Fit a degree-6 polynomial across the 8-point delta* grid for one
    tstar row (Cantera's MMCollisionInt::fitDelta)."""
    return np.polynomial.polynomial.polyfit(DELTA_GRID, table_row, degree)


# Pre-fit the delta* polynomials for every tstar row, for each of the four
# tables, mirroring MMCollisionInt::init().
_OMEGA22_DPOLY = [_fit_delta_row(row) for row in OMEGA22_TABLE]
_ASTAR_DPOLY = [_fit_delta_row(row) for row in ASTAR_TABLE]
_BSTAR_DPOLY = [_fit_delta_row(row) for row in BSTAR_TABLE]
_CSTAR_DPOLY = [_fit_delta_row(row) for row in CSTAR_TABLE]


def _quad_interp(x0, x, y):
    """3-point quadratic (Lagrange-form) interpolation, exactly matching
    Cantera's MMCollisionInt::quadInterp."""
    dx21 = x[1] - x[0]
    dx32 = x[2] - x[1]
    dx31 = dx21 + dx32
    dy32 = y[2] - y[1]
    dy21 = y[1] - y[0]
    a = (dx21 * dy32 - dy21 * dx32) / (dx21 * dx31 * dx32)
    return a * (x0 - x[0]) * (x0 - x[1]) + (dy21 / dx21) * (x0 - x[1]) + y[1]


def _lookup(ts, deltastar, table, dpoly):
    """Generic table lookup + quadratic interpolation in ln(T*), matching
    the pattern shared by omega22()/astar()/bstar()/cstar() in
    MMCollisionInt.cpp."""
    n = len(TSTAR22)
    i = np.searchsorted(TSTAR22, ts, side='right')
    # Cantera: for i in range(37): if ts < tstar22[i]: break  (i lands on
    # the first index whose value exceeds ts; searchsorted 'right' matches)
    i1 = max(i - 1, 0)
    i2 = i1 + 3
    if i2 > n - 1:
        i2 = n - 1
        i1 = i2 - 3
    idxs = range(i1, i2)
    if deltastar == 0.0:
        values = [table[k, 0] for k in idxs]
    else:
        values = [np.polynomial.polynomial.polyval(deltastar, dpoly[k]) for k in idxs]
    return _quad_interp(math.log(ts), LOG_TSTAR22[i1:i1 + 3], np.array(values))


def omega22(ts, deltastar):
    return _lookup(ts, deltastar, OMEGA22_TABLE, _OMEGA22_DPOLY)


def astar(ts, deltastar):
    return _lookup(ts, deltastar, ASTAR_TABLE, _ASTAR_DPOLY)


def bstar(ts, deltastar):
    return _lookup(ts, deltastar, BSTAR_TABLE, _BSTAR_DPOLY)


def cstar(ts, deltastar):
    return _lookup(ts, deltastar, CSTAR_TABLE, _CSTAR_DPOLY)


def omega11(ts, deltastar):
    return omega22(ts, deltastar) / astar(ts, deltastar)


# ==========================================================================
# YAML mechanism parsing
# ==========================================================================

class SpeciesData:
    def __init__(self, name):
        self.name = name
        self.mw = None            # kg/mol
        self.geometry = 'atom'
        self.sigma = 0.0          # m
        self.eps = 0.0            # J  (well depth * kB)
        self.dipole = 0.0         # C*m
        self.polar = False
        self.alpha = 0.0          # m^3 (polarizability)
        self.zrot = 0.0
        self.crot = 0.0           # dimensionless rotational cv/R contribution
        self.nasa_ranges = None   # [Tlow, Tmid, Thigh]
        self.nasa_low = None      # 7 coeffs
        self.nasa_high = None     # 7 coeffs


def _atomic_weights_from_yaml(doc):
    weights = dict(ATOMIC_WEIGHTS)
    for el in doc.get('elements', []):
        if isinstance(el, dict) and 'symbol' in el and 'atomic-weight' in el:
            weights[el['symbol']] = float(el['atomic-weight'])
    return weights


class _MechLoader(yaml.SafeLoader):
    """YAML loader that does NOT treat bare 'NO'/'YES'/'ON'/'OFF' as
    booleans (YAML 1.1 behavior). Cantera mechanisms contain species
    literally named 'NO' (nitric oxide), which PyYAML's default SafeLoader
    would otherwise silently parse as the boolean False."""


_MechLoader.yaml_implicit_resolvers = {
    k: [(tag, rx) for tag, rx in v if tag != 'tag:yaml.org,2002:bool']
    for k, v in yaml.SafeLoader.yaml_implicit_resolvers.items()
}


def load_mechanism(path):
    with open(path, 'r') as f:
        doc = yaml.load(f, Loader=_MechLoader)

    weights = _atomic_weights_from_yaml(doc)
    species_list = []

    for sp in doc.get('species', []):
        s = SpeciesData(sp['name'])

        # --- molecular weight from composition ---
        comp = sp.get('composition', {})
        mw = 0.0
        for el, count in comp.items():
            if el not in weights:
                raise ValueError(
                    f"Unknown element '{el}' for species {sp['name']}; "
                    "add it to ATOMIC_WEIGHTS or the YAML 'elements' section."
                )
            mw += weights[el] * count
        s.mw = mw / 1000.0  # g/mol -> kg/mol

        # --- thermo (NASA7) ---
        th = sp.get('thermo', {})
        if th.get('model', 'NASA7').upper() != 'NASA7':
            raise ValueError(
                f"Species {sp['name']}: only NASA7 thermo is supported "
                f"by this script (found '{th.get('model')}')."
            )
        s.nasa_ranges = th['temperature-ranges']
        data = th['data']
        s.nasa_low = data[0]
        s.nasa_high = data[1] if len(data) > 1 else data[0]

        # --- transport ---
        tr = sp.get('transport')
        if tr is None:
            raise ValueError(f"Species {sp['name']} has no 'transport' block.")
        s.geometry = tr.get('geometry', 'atom')
        s.crot = {'atom': 0.0, 'linear': 1.0, 'nonlinear': 1.5}[s.geometry]
        s.sigma = tr.get('diameter', 0.0) * ANGSTROM
        s.eps = tr.get('well-depth', 0.0) * BOLTZMANN
        dipole_debye = tr.get('dipole', 0.0)
        s.dipole = dipole_debye * DEBYE_TO_CM
        s.polar = dipole_debye > 0.0
        s.alpha = tr.get('polarizability', 0.0) * CUBIC_ANGSTROM
        s.zrot = tr.get('rotational-relaxation', 0.0)

        species_list.append(s)

    return species_list


# ==========================================================================
# NASA7 thermo evaluation
# ==========================================================================

def cp_R(species, T):
    """Dimensionless cp/R from the NASA7 polynomial."""
    Tlow, Tmid, Thigh = species.nasa_ranges
    a = species.nasa_low if T < Tmid else species.nasa_high
    return a[0] + a[1] * T + a[2] * T**2 + a[3] * T**3 + a[4] * T**4


# ==========================================================================
# Pairwise collision parameters (combining rules + polar correction)
# ==========================================================================

def _polar_correction(sp_i, sp_j):
    """Returns (f_eps, f_sigma) scale factors for one polar / one nonpolar
    pair, matching GasTransport::makePolarCorrections. (1,1) if both polar
    or both nonpolar."""
    if sp_i.polar == sp_j.polar:
        return 1.0, 1.0
    polar, nonpolar = (sp_i, sp_j) if sp_i.polar else (sp_j, sp_i)
    d3_np = nonpolar.sigma ** 3
    d3_p = polar.sigma ** 3
    alpha_star = nonpolar.alpha / d3_np
    mu_p_star = polar.dipole / math.sqrt(4 * PI * EPSILON_0 * d3_p * polar.eps)
    xi = 1.0 + 0.25 * alpha_star * mu_p_star**2 * math.sqrt(polar.eps / nonpolar.eps)
    f_sigma = xi ** (-1.0 / 6.0)
    f_eps = xi * xi
    return f_eps, f_sigma


def pair_params(sp_i, sp_j):
    """Returns dict with reduced mass, sigma_ij, eps_ij, delta_star_ij for
    the (i,j) collision, exactly as GasTransport::setupCollisionParameters."""
    mu_ij = sp_i.mw * sp_j.mw / (AVOGADRO * (sp_i.mw + sp_j.mw))
    sigma_ij = 0.5 * (sp_i.sigma + sp_j.sigma)
    eps_ij = math.sqrt(sp_i.eps * sp_j.eps)
    dipole_ij = math.sqrt(sp_i.dipole * sp_j.dipole)  # zero unless i==j or both polar... (see note)
    delta_ij = 0.0
    if sigma_ij > 0 and eps_ij > 0:
        delta_ij = 0.5 * dipole_ij**2 / (4 * PI * EPSILON_0 * eps_ij * sigma_ij**3)
    f_eps, f_sigma = _polar_correction(sp_i, sp_j)
    sigma_ij *= f_sigma
    eps_ij *= f_eps
    return dict(reduced_mass=mu_ij, sigma=sigma_ij, eps=eps_ij, delta=delta_ij)


# ==========================================================================
# Raw kinetic-theory property evaluation (Chapman-Enskog + modified Eucken)
# ==========================================================================

def _self_delta(sp):
    """delta*(k,k): for i==j, makePolarCorrections is a no-op (a species is
    never 'polar vs itself'), so sigma/eps are unchanged -- but delta* itself
    is still computed from the species' own (possibly nonzero) dipole
    moment. This matters for polar species like H2O."""
    if sp.sigma <= 0 or sp.eps <= 0:
        return 0.0
    return 0.5 * sp.dipole**2 / (4 * PI * EPSILON_0 * sp.eps * sp.sigma**3)


def pure_viscosity(sp, T):
    tstar = BOLTZMANN * T / sp.eps
    om22 = omega22(tstar, _self_delta(sp))
    return (5.0 / 16.0) * math.sqrt(PI * sp.mw * BOLTZMANN * T / AVOGADRO) / \
        (om22 * PI * sp.sigma**2)


def pure_self_diffusion(sp, T):
    tstar = BOLTZMANN * T / sp.eps
    om11 = omega11(tstar, _self_delta(sp))
    mu_kk = sp.mw * sp.mw / (AVOGADRO * (sp.mw + sp.mw))
    return (3.0 / 16.0) * math.sqrt(2.0 * PI / mu_kk) * (BOLTZMANN * T) ** 1.5 / \
        (PI * sp.sigma**2 * om11)


def pure_conductivity(sp, T, cpR):
    """Modified-Eucken thermal conductivity (Kee, Coltrin & Glarborg
    Eq. 12.112-style formula), matching GasTransport::fitProperties."""
    visc = pure_viscosity(sp, T)
    diffcoeff = pure_self_diffusion(sp, T)
    tstar = BOLTZMANN * T / sp.eps
    tstar_298 = BOLTZMANN * 298.0 / sp.eps

    def fz(ts):
        return 1.0 + PI**1.5 / math.sqrt(ts) * (0.5 + 1.0 / ts) + (0.25 * PI**2 + 2) / ts

    fz_298 = fz(tstar_298)
    fz_t = fz(tstar)

    f_int = sp.mw / (GAS_CONSTANT * T) * diffcoeff / visc
    cv_rot = sp.crot
    A = 2.5 - f_int
    B = sp.zrot * fz_298 / fz_t + (2.0 / PI) * (5.0 / 3.0 * cv_rot + f_int)
    c1 = (2.0 / PI) * A / B
    cv_int = cpR - 2.5 - cv_rot
    f_rot = f_int * (1.0 + c1)
    f_trans = 2.5 * (1.0 - c1 * cv_rot / 1.5)
    return (visc / sp.mw) * GAS_CONSTANT * (f_trans * 1.5 + f_rot * cv_rot + f_int * cv_int)


def binary_diffusion(sp_i, sp_j, T, params):
    tstar = BOLTZMANN * T / params['eps']
    om11 = omega11(tstar, params['delta'])
    return (3.0 / 16.0) * math.sqrt(2.0 * PI / params['reduced_mass']) * \
        (BOLTZMANN * T) ** 1.5 / (PI * params['sigma']**2 * om11)


# ==========================================================================
# Weighted least-squares polynomial fitting (matches Cantera's polyfit
# weight convention: weight passed to Cantera's routine is 1/value^2; numpy
# wants the sqrt of that, i.e. 1/|value|)
# ==========================================================================

def _weighted_fit(lnT, values, degree=FIT_DEGREE):
    weights = 1.0 / np.abs(values)
    coeffs = np.polynomial.polynomial.polyfit(lnT, values, degree, w=weights)
    return coeffs  # ascending order: c0 + c1*x + c2*x^2 + ...


# ==========================================================================
# Main driver: build all polynomials for a mechanism
# ==========================================================================

def fit_transport_polynomials(species_list, T_min=200.0, T_max=3500.0, n_points=NP_FIT_POINTS):
    T_grid = np.linspace(T_min, T_max, n_points)
    lnT = np.log(T_grid)

    results = {
        'viscosity': {},      # species name -> 5 coeffs, for sqrt(eta/sqrt(T))
        'conductivity': {},   # species name -> 5 coeffs, for lambda/sqrt(T)
        'binary_diffusion': {},  # (name_i, name_j) -> 5 coeffs, for D/T^1.5
    }

    # --- pure-species viscosity & conductivity ---
    for sp in species_list:
        visc_vals = np.empty(n_points)
        cond_vals = np.empty(n_points)
        for n, T in enumerate(T_grid):
            eta = pure_viscosity(sp, T)
            cpR = cp_R(sp, T)
            lam = pure_conductivity(sp, T, cpR)
            visc_vals[n] = math.sqrt(eta / math.sqrt(T))
            cond_vals[n] = lam / math.sqrt(T)
        results['viscosity'][sp.name] = _weighted_fit(lnT, visc_vals)
        results['conductivity'][sp.name] = _weighted_fit(lnT, cond_vals)

    # --- binary diffusion coefficients (unit pressure), i<=j ---
    for i, sp_i in enumerate(species_list):
        for j in range(i, len(species_list)):
            sp_j = species_list[j]
            params = pair_params(sp_i, sp_j)
            diff_vals = np.empty(n_points)
            for n, T in enumerate(T_grid):
                D = binary_diffusion(sp_i, sp_j, T, params)
                diff_vals[n] = D / T**1.5
            results['binary_diffusion'][(sp_i.name, sp_j.name)] = _weighted_fit(lnT, diff_vals)

    return results


# ==========================================================================
# Polynomial evaluation helpers (reconstruct properties from coefficients)
# ==========================================================================

def eval_viscosity(T, coeffs):
    lnT = math.log(T)
    s = np.polynomial.polynomial.polyval(lnT, coeffs)
    return math.sqrt(T) * s * s


def eval_conductivity(T, coeffs):
    lnT = math.log(T)
    return math.sqrt(T) * np.polynomial.polynomial.polyval(lnT, coeffs)


def eval_binary_diffusion(T, coeffs):
    lnT = math.log(T)
    return T**1.5 * np.polynomial.polynomial.polyval(lnT, coeffs)


# --------------------------------------------------------------------------
# Bonus: mixture-averaged combination rules (NOT polynomials -- these are
# composition-dependent and must be evaluated at each state; Cantera doesn't
# fit polynomials for these either. Included so the binary-diffusion
# polynomials above are directly usable.)
# --------------------------------------------------------------------------

def mixture_viscosity(T, mole_fractions, species_list, visc_coeffs):
    """Wilke (1950) mixture rule, matching GasTransport::viscosity()/
    updateViscosity_T()."""
    n = len(species_list)
    eta = np.array([eval_viscosity(T, visc_coeffs[sp.name]) for sp in species_list])
    mw = np.array([sp.mw for sp in species_list])
    X = np.array(mole_fractions)
    phi = np.zeros((n, n))
    for j in range(n):
        for k in range(n):
            num = (1.0 + math.sqrt(eta[k] / eta[j]) * (mw[j] / mw[k]) ** 0.25) ** 2
            den = math.sqrt(8.0 * (1.0 + mw[k] / mw[j]))
            phi[k, j] = num / den
    denom = phi @ X
    return float(np.sum(X * eta / denom))


def mixture_diffusion_coeffs_mole(T, P, mole_fractions, species_list, diff_coeffs):
    """Mole-based mixture-averaged diffusion coefficients, matching
    GasTransport::getMixDiffCoeffsMole(): D_k = (1-X_k) / sum_{j!=k}(X_j/D_kj)."""
    n = len(species_list)
    names = [sp.name for sp in species_list]

    def Dij(i, j):
        key = (names[i], names[j]) if (names[i], names[j]) in diff_coeffs else (names[j], names[i])
        return eval_binary_diffusion(T, diff_coeffs[key]) / P

    X = np.array(mole_fractions)
    D_mix = np.zeros(n)
    for k in range(n):
        s = sum(X[j] / Dij(k, j) for j in range(n) if j != k)
        D_mix[k] = Dij(k, k) if s <= 0 else (1.0 - X[k]) / s
    return D_mix


# ==========================================================================
# CLI
# ==========================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mechanism', help='Path to a Cantera-format YAML mechanism file')
    ap.add_argument('--tmin', type=float, default=200.0, help='Fit range min T [K]')
    ap.add_argument('--tmax', type=float, default=3500.0, help='Fit range max T [K]')
    ap.add_argument('--json', default=None, help='Optional path to dump results as JSON')
    args = ap.parse_args()

    species_list = load_mechanism(args.mechanism)
    results = fit_transport_polynomials(species_list, args.tmin, args.tmax)

    print(f"Fit range: {args.tmin} - {args.tmax} K, {len(species_list)} species\n")
    print("Viscosity polynomials [sqrt(eta/sqrt(T)) = c0 + c1*lnT + ... + c4*lnT^4]")
    for name, c in results['viscosity'].items():
        print(f"  {name:12s} {c}")

    print("\nConductivity polynomials [lambda/sqrt(T) = c0 + c1*lnT + ... + c4*lnT^4]")
    for name, c in results['conductivity'].items():
        print(f"  {name:12s} {c}")

    print("\nBinary diffusion polynomials [D/T^1.5 = c0 + c1*lnT + ... + c4*lnT^4] (unit pressure, Pa)")
    for (ni, nj), c in results['binary_diffusion'].items():
        print(f"  {ni:10s}-{nj:10s} {c}")

    if args.json:
        out = {
            'fit_range_K': [args.tmin, args.tmax],
            'viscosity': {k: v.tolist() for k, v in results['viscosity'].items()},
            'conductivity': {k: v.tolist() for k, v in results['conductivity'].items()},
            'binary_diffusion': {f"{i}:{j}": v.tolist() for (i, j), v in results['binary_diffusion'].items()},
        }
        with open(args.json, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"\nWrote {args.json}")


if __name__ == '__main__':
    main()
