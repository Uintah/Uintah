//----- RadiationVariables.h -----------------------------------------------

#ifndef Uintah_Components_Models_RadiationVariables_h
#define Uintah_Components_Models_RadiationVariables_h

#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>

/**************************************
CLASS
   RadiationVariables
   
   Class RadiationVariables creates and stores the 
   Variables that are used in Components/Models/Radiation

GENERAL INFORMATION
   RadiationVariables.h - declaration of the class
   
   Author: Seshadri Kumar (skumar@crsim.utah.edu)
   
   Creation Date:   April 11, 2005
   
   C-SAFE 
   
   Copyright University of Utah 2005

KEYWORDS

DESCRIPTION
   This routine is based on ArchesVariables, written by Rajesh Rawat (2000) 
   in the Components/Arches directory.   

WARNING
   none

************************************************************************/

namespace Uintah {

  class RadiationVariables {
    public:
      RadiationVariables();
      ~RadiationVariables();

      CCVariable<double> ABSKG;
      CCVariable<int> cellType;
      CCVariable<double> cenint;
      CCVariable<double> co2;
      CCVariable<double> h2o;
      CCVariable<double> mixfrac;
      CCVariable<double> cp;
      CCVariable<double> ESRCG;
      CCVariable<double> qfluxe;
      CCVariable<double> qfluxw;
      CCVariable<double> qfluxn;
      CCVariable<double> qfluxs;
      CCVariable<double> qfluxt;
      CCVariable<double> qfluxb;
      CCVariable<double> sootVF;
      CCVariable<double> shgamma;
      CCVariable<double> src;
      CCVariable<double> temperature;

    }; // End class RadiationVariables
   
} // End namespace Uintah

#endif
