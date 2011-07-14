# this file should be used with:
#   TabProps/test/h2o2.cti
#   TabProps/test/elements.xml

#--- Reaction Model

REACTION MODEL = AdiabaticEquilibrium

#--- Specification for the AdiabaticEquilibrium model
  AdiabaticEquilibrium.CanteraInputFile = h2o2.cti

  AdiabaticEquilibrium.FuelComposition.MoleFraction.H2 = 0.8
  AdiabaticEquilibrium.FuelComposition.MoleFraction.H2O = 0.2
  AdiabaticEquilibrium.FuelTemperature = 300

  AdiabaticEquilibrium.OxidizerComposition.MoleFraction.O2 = 0.21
  AdiabaticEquilibrium.OxidizerComposition.MoleFraction.AR = 0.79
  AdiabaticEquilibrium.OxidizerTemperature = 400

  AdiabaticEquilibrium.SelectForOutput = temperature density MolecularWeight enthalpy Viscosity Conductivity

  AdiabaticEquilibrium.nfpts = 100   # number of points in mixture fraction
