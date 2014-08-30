<!-- new format: -->
<ReactionModel type="Equilibrium">
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
  <nfpts>21</nfpts>
  <NHeatLossPts>11</NHeatLossPts>
  <SelectForOutput>temperature molecularweight enthalpy</SelectForOutput>
  <SelectSpeciesForOutput>H2 O2 H2O AR</SelectSpeciesForOutput>
</ReactionModel>

<MixingModel type="ClipGauss">
  <order>2</order>
  <ConvolutionVariable>MixtureFraction</ConvolutionVariable>
  <ReactionModelFileName>Equilibrium</ReactionModelFileName>
  <MixtureFraction>
    <npts>11</npts>
    <!-- default bounds [0,1] -->
  </MixtureFraction>
  <HeatLoss>
    <npts>10</npts>
    <min>-1</min>
    <max>1</max>
  </HeatLoss>
</MixingModel>

<!-- old format: 
#--- Reaction Model

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



#------------------------------------------------
#--------------- The Mixing Model ---------------

# The clipped-gaussian mixing model is recommended, since the
# integrators have a hard time integrating the beta-pdf accurately.
# Just as with the reaction models, one may specify multiple mixing
# models here to activate them.

MIXING MODEL = ClipGauss

# by default, the min and max are [0,1].
# Otherwise, we must specify them as with the heat loss below.
  ClipGauss.MixtureFraction.npts = 11

  ClipGauss.HeatLoss.npts = 10
  ClipGauss.HeatLoss.min = -1.0
  ClipGauss.HeatLoss.max = 1.0

  ClipGauss.MixtureFractionVariance.npts = 11

# specify the variable that we are applying the mixing model on.
# this variable name comes from the reaction model table.
  ClipGauss.ConvolutionVariable = MixtureFraction

# specify the name of the HDF5 database for use in the mixing model.
  ClipGauss.ReactionModelFileName = Equilibrium
#------------------------------------------------
 -->