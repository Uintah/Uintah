#ifndef __ThermalContact__
#define __ThermalContact__

#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/SimulationStateP.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>
#include <Packages/Uintah/Core/Grid/ReductionVariable.h>
#include <Packages/Uintah/Core/Grid/SimulationStateP.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Core/Geometry/Vector.h>
#include <Core/Math/MinMax.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>
#include <math.h>

namespace Uintah {
using namespace SCIRun;

   class ProcessorGroup;
   class Patch;
   class VarLabel;
   class Task;

/**************************************

CLASS
   ThermalContact
   
   Short description...

GENERAL INFORMATION

   ThermalContact.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   ThermalContact_Model

DESCRIPTION
   Long description...
  
WARNING

****************************************/

class ThermalContact {
public:
  // Constructor
  ThermalContact(ProblemSpecP& ps,SimulationStateP& d_sS);
  ~ThermalContact();

  void computeHeatExchange(const ProcessorGroup*,
                           const Patch* patch,
                           DataWarehouseP& old_dw,
                           DataWarehouseP& new_dw);
	 
  void initializeThermalContact(const Patch* patch,
				int vfindex,
				DataWarehouseP& new_dw);

  void addComputesAndRequires(Task* task,
                              const MPMMaterial* matl,
                              const Patch* patch,
                              DataWarehouseP& old_dw,
                              DataWarehouseP& new_dw) const;

private:
  SimulationStateP d_sharedState;
  MPMLabel* lb;
};
      
} // End namespace Uintah

#endif // __ThermalContact__
