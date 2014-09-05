#ifndef __CONTACT_H__
#define __CONTACT_H__

#include <Packages/Uintah/CCA/Components/MPM/Contact/ContactMaterialSpec.h>
#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>
#include <Packages/Uintah/Core/Grid/Variables/ComputeSet.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/CCA/Ports/SchedulerP.h>

#include <math.h>

namespace Uintah {
using namespace SCIRun;
  class DataWarehouse;
  class MPMLabel;
  class MPMFlags;
  class ProcessorGroup;
  class Patch;
  class VarLabel;
  class Task;

/**************************************

CLASS
   Contact
   
   Short description...

GENERAL INFORMATION

   Contact.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Contact_Model

DESCRIPTION
   Long description...
  
WARNING

****************************************/

  class Contact : public UintahParallelComponent {
      public:
         // Constructor
         Contact(const ProcessorGroup* myworld, MPMLabel* Mlb, MPMFlags* MFlag, ProblemSpecP ps);
	 virtual ~Contact();

	 // Basic contact methods
	 virtual void exMomInterpolated(const ProcessorGroup*,
					const PatchSubset* patches,
					const MaterialSubset* matls,
					DataWarehouse* old_dw,
					DataWarehouse* new_dw) = 0;
	 
	 virtual void exMomIntegrated(const ProcessorGroup*,
				      const PatchSubset* patches,
				      const MaterialSubset* matls,
				      DataWarehouse* old_dw,
				      DataWarehouse* new_dw) = 0;
         
         virtual void addComputesAndRequiresInterpolated(SchedulerP & sched,
                                      const PatchSet* patches,
                                      const MaterialSet* matls) = 0;
	 
         virtual void addComputesAndRequiresIntegrated(SchedulerP & sched,
				      const PatchSet* patches,
				      const MaterialSet* matls) = 0;
         
      protected:
	 MPMLabel* lb;
	 MPMFlags* flag;
	 
         ContactMaterialSpec d_matls;
      };
      
      inline bool compare(double num1, double num2) {
	    double EPSILON=1.e-12;
	    
	    return (fabs(num1-num2) <= EPSILON);
      }

} // End namespace Uintah

#endif // __CONTACT_H__
