//----- RadiationConstVariables.h -----------------------------------------------
#include <Packages/Uintah/Core/Exceptions/PetscError.h>

#ifndef Uintah_Components_Models_RadiationConstVariables_h
#define Uintah_Components_Models_RadiationConstVariables_h

#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>

/**************************************
CLASS
   RadiationConstVariables
   
   Class RadiationConstVariables creates and stores the ConstVariables 
   that are used in Components/Models/Radiation 

GENERAL INFORMATION
   RadiationConstVariables.h - declaration of the class
   
   Author: Seshadri Kumar (skumar@crsim.utah.edu)
   
   Creation Date: April 12, 2005

   C-SAFE 
   
   Copyright U of U 2005

KEYWORDS

DESCRIPTION
   This routine is based on ArchesConstVariables, written by Rajesh Rawat 
   (2000) in the Components/Arches directory.      

WARNING
   none

************************************************************************/

namespace Uintah {

  class RadiationConstVariables {
    public:
      RadiationConstVariables();
      ~RadiationConstVariables();
      constCCVariable<double> absorption;
      constCCVariable<double> co2;
      constCCVariable<double> c2h2;
      constCCVariable<int> cellType;
      constCCVariable<double> density;
      constCCVariable<double> h2o;
      constCCVariable<double> mixfrac;
      constCCVariable<double> sootVF;
      constCCVariable<double> temperature;

    }; // End class RadiationConstVariables
} // End namespace Uintah

#endif
