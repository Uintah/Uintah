#--- Reaction Model
 REACTION MODEL = NonReacting
#--- Specification for the NonReacting model
 NonReacting.CanteraInputFile = gri30.cti

 NonReacting.FuelComposition.MoleFraction.H2  = 1.0
 NonReacting.FuelTemperature                 = 300

 NonReacting.OxidizerComposition.MoleFraction.O2 = 1.0
 NonReacting.OxidizerTemperature                 = 300

 NonReacting.SelectForOutput = density temperature MolecularWeight

 NonReacting.nfpts = 150   # number of points in mixture fraction 
 NonReacting.order = 3     # order of the polynomial interpolation