#ifndef __HeatConduction__
#define __HeatConduction__

#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/SimulationStateP.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>
#include <Packages/Uintah/Core/Grid/SimulationStateP.h>
#include <Core/Geometry/Vector.h>
#include <Core/Math/MinMax.h>

#include <math.h>

namespace Uintah {

using namespace SCIRun;

   class ProcessorGroup;
   class Patch;
   class VarLabel;
   class Task;

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


private:
  SimulationStateP d_sharedState;
};
      
} // End namespace Uintah

#endif // __HeatConduction__

