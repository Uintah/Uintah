<!-- new format: -->
<ReactionModel type="FastChem">
  <CanteraInputFile>h2o2.cti</CanteraInputFile>
  <FuelComposition type="MoleFraction">
    <Species name="H2" >0.8</Species>
    <Species name="H2O">0.2</Species>
  </FuelComposition>
  <OxidizerComposition type="MoleFraction">
    <Species name="O2">0.21</Species>
    <Species name="AR">0.79</Species>
  </OxidizerComposition>
  <FuelTemperature>300</FuelTemperature>
  <OxidizerTemperature>400</OxidizerTemperature>
  <SelectForOutput>
    temperature molecularweight enthalpy
  </SelectForOutput>
  <SelectSpeciesForOutput>H2 O2 H2O AR</SelectSpeciesForOutput>
  <nfpts>75</nfpts>
  <GridStretchFactor>3.0</GridStretchFactor>
  <NHeatLossPts>51</NHeatLossPts>
  <order>1</order>
</ReactionModel>



<!-- old format:
REACTION MODEL = FastChem

#--- Specification for the FastChem model
  FastChem.CanteraInputFile = h2o2.cti

  FastChem.FuelComposition.MoleFraction.H2 = 0.8
  FastChem.FuelComposition.MoleFraction.H2O = 0.2
  FastChem.FuelTemperature = 300

  FastChem.OxidizerComposition.MoleFraction.O2 = 0.21
  FastChem.OxidizerComposition.MoleFraction.AR = 0.79
  FastChem.OxidizerTemperature = 400

  # choose output:
  FastChem.SelectForOutput = temperature molecularweight enthalpy
  FastChem.SelectSpeciesForOutput = H2 O2 H2O AR

  FastChem.nfpts = 75        # number of points in mixture fraction
  FastChem.GridStretchFactor = 3.0
  FastChem.NHeatLossPts = 51  # number of points in the heat loss dimension
-->