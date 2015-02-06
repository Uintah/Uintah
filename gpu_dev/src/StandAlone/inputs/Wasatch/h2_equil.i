<ReactionModel type="Equilibrium">
  <CanteraInputFile>h2o2.cti</CanteraInputFile>
  <order>3</order>
  <FuelComposition type="MoleFraction">
    <Species name="H2" >0.8</Species>
    <Species name="H2O">0.2</Species>
  </FuelComposition>
  <OxidizerComposition type="MoleFraction">
    <Species name="O2">0.21</Species>
    <Species name="AR">0.79</Species>
  </OxidizerComposition>
  <FuelTemperature>300</FuelTemperature>
  <OxidizerTemperature>300</OxidizerTemperature>
  <SelectForOutput>temperature viscosity density enthalpy conductivity specificheat</SelectForOutput>
  <SelectSpeciesForOutput>H2 O2 H2O AR</SelectSpeciesForOutput>
  <nfpts>176</nfpts>
  <NHeatLossPts>50</NHeatLossPts>
</ReactionModel>
