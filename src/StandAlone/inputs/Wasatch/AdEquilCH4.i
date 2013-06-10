# this file should be used with:
#   TabProps/test/gri30.cti
#   TabProps/test/elements.xml

#--- Reaction Model

REACTION MODEL = AdiabaticEquilibrium

#--- Specification for the AdiabaticEquilibrium model
  AdiabaticEquilibrium.CanteraInputFile = gri30.cti

  AdiabaticEquilibrium.FuelComposition.MoleFraction.CH4 = 0.8
  AdiabaticEquilibrium.FuelComposition.MoleFraction.H2O = 0.2
  AdiabaticEquilibrium.FuelTemperature = 300

  AdiabaticEquilibrium.OxidizerComposition.MoleFraction.O2 = 0.21
  AdiabaticEquilibrium.OxidizerComposition.MoleFraction.N2 = 0.79
  AdiabaticEquilibrium.OxidizerTemperature = 400

  AdiabaticEquilibrium.SelectForOutput = temperature density MolecularWeight enthalpy Viscosity Conductivity

  # pick some mole fractions required for obtaining radiative transport properties
  AdiabaticEquilibrium.SelectMoleFracForOutput = CO2 H2O

  AdiabaticEquilibrium.nfpts = 100   # number of points in mixture fraction
