<ReactionModel type="AdiabaticEquilibrium">
    <CanteraInputFile>h2o2.cti</CanteraInputFile>
  <FuelComposition type="MoleFraction">
    <Species name="H2" >0.8</Species>
    <Species name="H2O">0.2</Species>
  </FuelComposition>
  <FuelTemperature>300</FuelTemperature>
  <OxidizerComposition type="MoleFraction">
    <Species name="O2">0.21</Species>
    <Species name="AR">0.79</Species>
  </OxidizerComposition>
  <OxidizerTemperature>400</OxidizerTemperature>
  <order>1</order>
  <nfpts>100</nfpts>
  <SelectForOutput>
    temperature 
    density 
    MolecularWeight
    enthalpy 
    Viscosity 
    Conductivity
  </SelectForOutput>
</ReactionModel>

<!-- old format:
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
-->