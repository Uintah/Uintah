// NullContact.h

#ifndef __NULL_CONTACT_H__
#define __NULL_CONTACT_H__

#include <Packages/Uintah/CCA/Components/MPM/Contact/Contact.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>
#include <Packages/Uintah/Core/Grid/GridP.h>
#include <Packages/Uintah/Core/Grid/LevelP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/SimulationStateP.h>

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
      NullContact(ProblemSpecP& ps,SimulationStateP& ss, MPMLabel* lb);
      
      // Destructor
      virtual ~NullContact();

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
      
      virtual void addComputesAndRequiresInterpolated(Task* task,
					     const PatchSet* patches,
					     const MaterialSet* matls) const;

      virtual void addComputesAndRequiresIntegrated(Task* task,
					     const PatchSet* patches,
					     const MaterialSet* matls) const;

    };
} // End namespace Uintah
    


#endif /* __NULL_CONTACT_H__ */

