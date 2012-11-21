Instructions for calculating and using Planck mean absorption coefficients

CAUTION: Planck mean absorption coefficients are valid only in optically thin media. Hence, they must be opted for only when the medium conditions warrant their usage, for instance small pool fires (diameter < 0.5 m)

1. First, you must have the files RGAMMA, SDCOTWO and SDWATER in your working directory. These files are located in the same directory as this file.

2. Go to: 

SCIRun/src/Packages/Uintah/CCA/Components/Arches/Radiation/DORadiationModel.cc

and set the logical "loptthin" to true:

  loptthin = true;

3. Re-compile the code

Written by Gautham Krishnamoorthy (gautham@crsim.utah.edu)