<!-- new format: -->
<ReactionModel type="AdiabaticEquilibrium">
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
  <OxidizerTemperature>300</OxidizerTemperature>
  <SelectForOutput>
    temperature density MolecularWeight enthalpy species ReactionRate SpeciesEnthalpy
  </SelectForOutput>
  <nfpts>176</nfpts>
</ReactionModel>

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
  <OxidizerTemperature>300</OxidizerTemperature>
  <SelectForOutput>
    temperature MolecularWeight enthalpy
  </SelectForOutput>
  <SelectSpeciesForOutput>H2 O2 H2O AR</SelectSpeciesForOutput>
  <nfpts>176</nfpts>
  <GridStretchFactor>3.0</GridStretchFactor>
  <NHeatLossPts>50</NHeatLossPts>
</ReactionModel>

<ReactionModel type="Equilibrium">
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
  <OxidizerTemperature>300</OxidizerTemperature>
  <SelectForOutput>temperature MolecularWeight enthalpy</SelectForOutput>
  <SelectSpeciesForOutput>H2 O2 H2O AR</SelectSpeciesForOutput>
  <nfpts>176</nfpts>
  <NHeatLossPts>50</NHeatLossPts>
</ReactionModel>

<MixingModel type="ClipGauss">
  <ConvolutionVariable>MixtureFraction</ConvolutionVariable>
  <ReactionModelFileName>Equilibrium</ReactionModelFileName>
  <MixtureFraction> <npts>51</npts> </MixtureFraction>
	<HeatLoss>
		<npts>30</npts>
	  <min>-1.0</min>
	  <max>1.0</max>
	</HeatLoss>
	<MixtureFractionVariance> <npts>21</npts> </MixtureFractionVariance>
</MixingModel>


<!-- old format:
#--- Reaction Model

REACTION MODEL = AdiabaticEquilibrium FastChem Equilibrium


#--- Specification for the AdiabaticEquilibrium model
  AdiabaticEquilibrium.CanteraInputFile = h2o2.cti

  AdiabaticEquilibrium.FuelComposition.MoleFraction.H2 = 0.8
  AdiabaticEquilibrium.FuelComposition.MoleFraction.H2O = 0.2
  AdiabaticEquilibrium.FuelTemperature = 300

  AdiabaticEquilibrium.OxidizerComposition.MoleFraction.O2 = 0.21
  AdiabaticEquilibrium.OxidizerComposition.MoleFraction.AR = 0.79
  AdiabaticEquilibrium.OxidizerTemperature = 300

  AdiabaticEquilibrium.SelectForOutput = temperature density MolecularWeight enthalpy species ReactionRate SpeciesEnthalpy

 # output a set of species:
  AdiabaticEquilibrium.nfpts = 176   # number of points in mixture fraction

#--- Specification for the FastChem model
  FastChem.CanteraInputFile = h2o2.cti

  FastChem.FuelComposition.MoleFraction.H2 = 0.8
  FastChem.FuelComposition.MoleFraction.H2O = 0.2
  FastChem.FuelTemperature = 300

  FastChem.OxidizerComposition.MoleFraction.O2 = 0.21
  FastChem.OxidizerComposition.MoleFraction.AR = 0.79
  FastChem.OxidizerTemperature = 300

  # choose output (output all species):
  FastChem.SelectForOutput = temperature molecularweight enthalpy
  FastChem.SelectSpeciesForOutput = H2 O2 H2O AR

  FastChem.nfpts = 176        # number of points in mixture fraction
  FastChem.GridStretchFactor = 3.0
  FastChem.NHeatLossPts = 50  # number of points in the heat loss dimension

#--- Specification for the Equilibrium model
  Equilibrium.CanteraInputFile = h2o2.cti

  Equilibrium.FuelComposition.MoleFraction.H2 = 0.8
  Equilibrium.FuelComposition.MoleFraction.H2O = 0.2
  Equilibrium.FuelTemperature = 300

  Equilibrium.OxidizerComposition.MoleFraction.O2 = 0.21
  Equilibrium.OxidizerComposition.MoleFraction.AR = 0.79
  Equilibrium.OxidizerTemperature = 300

  # choose output (output all species):
  Equilibrium.SelectForOutput = temperature molecularweight enthalpy
  Equilibrium.SelectSpeciesForOutput = H2 O2 H2O AR

  Equilibrium.nfpts = 176        # number of points in mixture fraction
  Equilibrium.NHeatLossPts = 50  # number of points in the heat loss dimension



#------------------------------------------------
#--------------- The Mixing Model ---------------

# The clipped-gaussian mixing model is recommended, since the
# integrators have a hard time integrating the beta-pdf accurately.
# Just as with the reaction models, one may specify multiple mixing
# models here to activate them.

MIXING MODEL = ClipGauss

#ClipGauss.MixtureFraction.npts = 21
#ClipGauss.HeatLoss.npts = 21
#ClipGauss.MixtureFractionVariance.npts = 21

ClipGauss.MixtureFraction.npts = 51
ClipGauss.HeatLoss.npts = 30
ClipGauss.HeatLoss.min = -1.0
ClipGauss.HeatLoss.max = 1.0
ClipGauss.MixtureFractionVariance.npts = 21

# specify the variable that we are applying the mixing model on.
# this variable name comes from the reaction model table.
ClipGauss.ConvolutionVariable = MixtureFraction

# specify the name of the HDF5 database for use in the mixing model.
ClipGauss.ReactionModelFileName = Equilibrium
#------------------------------------------------

-->