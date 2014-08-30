<!-- new format: -->
<ReactionModel type="Equilibrium">
  <order>1</order>
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
  <OxidizerTemperature>300</OxidizerTemperature>
  <order>1</order>
  <nfpts>21</nfpts>
  <NHeatLossPts>11</NHeatLossPts>
  <SelectForOutput>temperature molecularweight enthalpy</SelectForOutput>
  <SelectSpeciesForOutput>H2 O2 H2O AR</SelectSpeciesForOutput>
</ReactionModel>
  

  <!-- old format:
  REACTION MODEL = Equilibrium

#--- Specification for the Equilibrium model
  Equilibrium.CanteraInputFile = h2o2.cti

  Equilibrium.FuelComposition.MoleFraction.H2 = 0.8
  Equilibrium.FuelComposition.MoleFraction.H2O = 0.2
  Equilibrium.FuelTemperature = 300

  Equilibrium.OxidizerComposition.MoleFraction.O2 = 0.21
  Equilibrium.OxidizerComposition.MoleFraction.AR = 0.79
  Equilibrium.OxidizerTemperature = 400

  # choose output (output all species):
  Equilibrium.SelectForOutput = temperature molecularweight enthalpy
  Equilibrium.SelectSpeciesForOutput = H2 O2 H2O AR

  Equilibrium.nfpts = 21         # number of points in mixture fraction
  Equilibrium.NHeatLossPts = 11  # number of points in the heat loss dimension

  -->