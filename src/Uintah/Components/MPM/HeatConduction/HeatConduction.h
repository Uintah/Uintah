#ifndef __HeatConduction__
#define __HeatConduction__

#include <Uintah/Interface/DataWarehouseP.h>
#include <Uintah/Grid/SimulationState.h>
#include <Uintah/Grid/SimulationStateP.h>
#include <Uintah/Grid/VarLabel.h>
#include <Uintah/Grid/NCVariable.h>
#include <Uintah/Grid/ReductionVariable.h>
#include <Uintah/Grid/SimulationStateP.h>
#include <Uintah/Grid/VarTypes.h>
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Math/MinMax.h>

#include <math.h>

namespace Uintah {
   using SCICore::Geometry::Vector;
   using SCICore::Math::Min;

   class ProcessorGroup;
   class Patch;
   class VarLabel;
   class Task;
   namespace MPM {
     class MPMMaterial;

/**************************************

CLASS
   HeatConduction
   
   Handle heat conduction in solid using MPM.

GENERAL INFORMATION

   HeatConduction.h

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   HeatConduction_Model

DESCRIPTION
   This is for handling with heat conduction using MPM.  Most of the
   algorithm are mingled in MPM class.
  
WARNING

****************************************/

class HeatConduction {
public:
  // Constructor
  HeatConduction(ProblemSpecP& ps,SimulationStateP& d_sS);

  double getThermalConductivity() const;
  double getSpecificHeat() const;
  double getHeatTransferCoefficient() const;

private:
  SimulationStateP d_sharedState;

  double d_thermalConductivity;
  double d_specificHeat;
  double d_heatTransferCoefficient;
         
};
      
} // end namespace MPM
} // end namespace Uintah

#endif // __HeatConduction__

// $Log$
// Revision 1.2  2000/06/22 22:36:07  tan
// Moved heat conduction physical parameters (thermalConductivity, specificHeat,
// and heatTransferCoefficient) from MPMMaterial class to HeatConduction class.
//
// Revision 1.1  2000/06/20 17:59:02  tan
// Heat Conduction model created to move heat conduction part of code from MPM.
// Thus make MPM clean and easy to maintain.
//
