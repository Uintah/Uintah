#ifndef __ThermalContact__
#define __ThermalContact__

#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/SimulationStateP.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>
#include <Packages/Uintah/Core/Grid/SimulationStateP.h>
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
    ThermalContact();
    virtual ~ThermalContact();

    virtual void computeHeatExchange(const ProcessorGroup*,
                           const PatchSubset* patches,
                           const MaterialSubset* matls,
                           DataWarehouse* old_dw,
                           DataWarehouse* new_dw) = 0;
	 
    virtual void initializeThermalContact(const Patch* patch,
				int vfindex,
				DataWarehouse* new_dw) = 0;

    virtual void addComputesAndRequires(Task* task,
                              const PatchSet* patches,
			      const MaterialSet* matls) const = 0;

  private:
    MPMLabel* lb;
  };
      
} // End namespace Uintah

#endif // __ThermalContact__
