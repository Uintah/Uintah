#ifndef __NullThermalContact__
#define __NullThermalContact__

#include <CCA/Components/MPM/ThermalContact/ThermalContact.h>
#include <CCA/Ports/DataWarehouseP.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/SimulationStateP.h>
#include <SCIRun/Core/Geometry/Vector.h>
#include <SCIRun/Core/Math/MinMax.h>
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
    NullThermalContact(ProblemSpecP& ps,SimulationStateP& d_sS, MPMLabel* lb,
		       MPMFlags* MFlag);

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

    virtual void outputProblemSpec(ProblemSpecP& ps);

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
