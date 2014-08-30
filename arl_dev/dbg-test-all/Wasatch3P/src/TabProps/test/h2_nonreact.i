<!-- new format: -->
<ReactionModel type="NonReacting">
  <order>1</order>
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
  <SelectSpeciesForOutput>
    H2 O2 H2O AR
  </SelectSpeciesForOutput>
  <nfpts>175</nfpts>
</ReactionModel>


<!-- old format: 
REACTION MODEL = NonReacting

#--- Specification for the NonReacting model
  NonReacting.CanteraInputFile = h2o2.cti

  NonReacting.FuelComposition.MoleFraction.H2 = 0.8
  NonReacting.FuelComposition.MoleFraction.H2O = 0.2
  NonReacting.FuelTemperature = 300

  NonReacting.OxidizerComposition.MoleFraction.O2 = 0.21
  NonReacting.OxidizerComposition.MoleFraction.AR = 0.79
  NonReacting.OxidizerTemperature = 400

  # choose output (output all species):
  NonReacting.SelectForOutput = temperature molecularweight enthalpy
  NonReacting.SelectSpeciesForOutput = H2 O2 H2O AR

  NonReacting.nfpts = 175   # number of points in mixture fraction
-->