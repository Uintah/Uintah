#ifndef __NullThermalContact__
#define __NullThermalContact__

#include <Packages/Uintah/CCA/Components/MPM/ThermalContact/ThermalContact.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>
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
   NullThermalContact
   
   This version of thermal contact drives the temperatures
   of two materials to the same value at each grid point.

GENERAL INFORMATION

   NullThermalContact.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   NullThermalContact_Model

DESCRIPTION
   Long description...
  
WARNING

****************************************/

  class NullThermalContact : public ThermalContact {
    public:
    // Constructor
    NullThermalContact(ProblemSpecP& ps,SimulationStateP& d_sS, MPMLabel* lb);

    // Destructor
    virtual ~NullThermalContact();

    virtual void computeHeatExchange(const ProcessorGroup*,
                           const PatchSubset* patches,
                           const MaterialSubset* matls,
                           DataWarehouse* old_dw,
                           DataWarehouse* new_dw);
	 
    virtual void initializeThermalContact(const Patch* patch,
				int vfindex,
				DataWarehouse* new_dw);

    virtual void addComputesAndRequires(Task* task,
                              const PatchSet* patches,
			      const MaterialSet* matls) const;

    private:
      SimulationStateP d_sharedState;
      MPMLabel* lb;
      // Prevent copying of this class
      // copy constructor
      NullThermalContact(const NullThermalContact &con);
      NullThermalContact& operator=(const NullThermalContact &con);
  };
      
} // End namespace Uintah

#endif // __NullThermalContact__
