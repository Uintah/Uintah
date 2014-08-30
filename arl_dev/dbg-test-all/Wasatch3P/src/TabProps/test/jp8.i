<!-- new format -->
<ReactionModel type="AdiabaticEquilibrium" >
  <order>1</order>
  <CanteraInputFile>heptane.cti</CanteraInputFile>
  <CanteraGroupName>equil_gas</CanteraGroupName>
  <FuelComposition type="MoleFraction">
    <Species name="NC7H16">1.0</Species>
  </FuelComposition>  
  <OxidizerComposition type="MoleFraction">
    <Species name="O2">0.21</Species>
    <Species name="N2">0.79</Species>
  </OxidizerComposition>
  <FuelTemperature>298</FuelTemperature>
  <OxidizerTemperature>298</OxidizerTemperature>
  <SelectForOutput>temperature density viscosity</SelectForOutput>
  <nfpts>200</nfpts>
</ReactionModel>

<ReactionModel type="Equilibrium" >
  <order>1</order>
  <CanteraInputFile>heptane.cti</CanteraInputFile>
  <CanteraGroupName>equil_gas</CanteraGroupName>
  <FuelComposition type="MoleFraction">
    <Species name="NC7H16">1.0</Species>
  </FuelComposition>  
  <OxidizerComposition type="MoleFraction">
    <Species name="O2">0.21</Species>
    <Species name="N2">0.79</Species>
  </OxidizerComposition>
  <FuelTemperature>300</FuelTemperature>
  <OxidizerTemperature>300</OxidizerTemperature>
  <GridStretchFactor>4.0</GridStretchFactor>
  <NHeatLossPts>50</NHeatLossPts>
  <SelectForOutput>temperature</SelectForOutput>
  <SelectSpeciesForOutput>CO CO2 H2O OH NC7H16</SelectSpeciesForOutput>
</ReactionModel>


<!-- ------------ The Mixing Model ------------

    The clipped-gaussian mixing model is recommended, since the
    integrators have a hard time integrating the beta-pdf accurately.
    Just as with the reaction models, one may specify multiple mixing
    models here to activate them.
-->
<MixingModel type="ClipGauss">

<!-- Specify the variable that we are applying the mixing model on.
     this variable name comes from the reaction model table. -->
  <ConvolutionVariable>MixtureFraction</ConvolutionVariable>
  
  <ReactionModelFileName>AdiabaticEquil</ReactionModelFileName>
  
</MixingModel>







<!-- old format:
#-------------------------------------------------
#--------------- The Reaction Models -------------

# Activate some reaction models.  Note that you can have multiple
# entries, but you must only have a single "REACTION MODEL" line.
# Also, a model may be turned off by removing it from this list, even
# if it is in the rest of the input file.
REACTION MODEL = AdiabaticEquilibrium Equilibrium

# specify the name of the input file for cantera, as well as
# the name of the gas mixture from the cantera input file
 AdiabaticEquilibrium.CanteraInputFile = heptane.cti
 AdiabaticEquilibrium.CanteraGroupName = equil_gas

# specify the composition and temperature of the fuel stream
 AdiabaticEquilibrium.FuelComposition.MoleFraction.NC7H16 = 1.0
 AdiabaticEquilibrium.FuelTemperature = 298.0

# specify the composition and temperature of the oxidizer stream
 AdiabaticEquilibrium.OxidizerComposition.MoleFraction.O2 = 0.21
 AdiabaticEquilibrium.OxidizerComposition.MoleFraction.N2 = 0.79
 AdiabaticEquilibrium.OxidizerTemperature = 298.0

# specify the variables for output from the reaction model.
# all of these variables will be put through the mixing model.
 AdiabaticEquilibrium.SelectForOutput = temperature density viscosity
 AdiabaticEquilibrium.nfpts = 200

#-- The Equilibrium Model
 Equilibrium.CanteraInputFile = heptane.cti
 Equilibrium.CanteraGroupName = equil_gas
 Equilibrium.FuelComposition.MoleFraction.NC7H16 = 1.0
 Equilibrium.FuelTemperature = 300
 Equilibrium.OxidizerTemperature = 300
 Equilibrium.OxidizerComposition.MoleFraction.O2 = 0.21
 Equilibrium.OxidizerComposition.MoleFraction.N2 = 0.79
 Equilibrium.nfpts = 201
 Equilibrium.GridStretchFactor = 4.0
 Equilibrium.NHeatLossPts = 50
 Equilibrium.SelectForOutput = temperature
 Equilibrium.SelectSpeciesForOutput = CO CO2 H2O OH NC7H16
#-------------------------------------------------


#------------------------------------------------
#--------------- The Mixing Model ---------------

# The clipped-gaussian mixing model is recommended, since the
# integrators have a hard time integrating the beta-pdf accurately.
# Just as with the reaction models, one may specify multiple mixing
# models here to activate them.

 MIXING MODEL = ClipGauss

# specify the variable that we are applying the mixing model on.
# this variable name comes from the reaction model table.
 ClipGauss.ConvolutionVariable = MixtureFraction

# specify the name of the HDF5 database for use in the mixing model.
 ClipGauss.ReactionModelFileName = AdiabaticEquil
#------------------------------------------------


# NOTE: to run this, use the "mixrxn" executable

-->
