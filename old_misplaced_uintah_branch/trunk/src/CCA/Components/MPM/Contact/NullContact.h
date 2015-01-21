// NullContact.h

#ifndef __NULL_CONTACT_H__
#define __NULL_CONTACT_H__

#include <CCA/Components/MPM/Contact/Contact.h>
#include <CCA/Ports/DataWarehouseP.h>
#include <Core/Parallel/UintahParallelComponent.h>
#include <Core/Grid/GridP.h>
#include <Core/Grid/LevelP.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/SimulationStateP.h>

namespace Uintah {
/**************************************

CLASS
   NullContact
   
   Short description...

GENERAL INFORMATION

   NullContact.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Contact_Model_Null

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

    class NullContact : public Contact {
    private:
      
      // Prevent copying of this class
      // copy constructor
      NullContact(const NullContact &con);
      NullContact& operator=(const NullContact &con);

      SimulationStateP d_sharedState;
      
    public:
      // Constructor
      NullContact(const ProcessorGroup* myworld,
                  SimulationStateP& ss, MPMLabel* lb,
		  MPMFlags* MFlag);
      
      // Destructor
      virtual ~NullContact();

      virtual void outputProblemSpec(ProblemSpecP& ps);

      // Basic contact methods
      virtual void exMomInterpolated(const ProcessorGroup*,
				     const PatchSubset* patches,
				     const MaterialSubset* matls,
				     DataWarehouse* old_dw,
				     DataWarehouse* new_dw);
      

      virtual void exMomIntegrated(const ProcessorGroup*,
				   const PatchSubset* patches,
				   const MaterialSubset* matls,
				   DataWarehouse* old_dw,
				   DataWarehouse* new_dw);
      
      virtual void addComputesAndRequiresInterpolated(SchedulerP & sched,
					     const PatchSet* patches,
					     const MaterialSet* matls);

      virtual void addComputesAndRequiresIntegrated(SchedulerP & sched,
					     const PatchSet* patches,
					     const MaterialSet* matls);

    };
} // End namespace Uintah
    


#endif /* __NULL_CONTACT_H__ */

