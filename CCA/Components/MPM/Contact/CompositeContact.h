#ifndef __COMPOSITE_CONTACT_H__
#define __COMPOSITE_CONTACT_H__

#include <Packages/Uintah/CCA/Components/MPM/Contact/Contact.h>
#include <list>

namespace Uintah {
using namespace SCIRun;

/**************************************

CLASS
   CompositeContact
   
GENERAL INFORMATION

   CompositeContact.h

   Andrew Brydon
   Los Alamos National Laboratory
 
   Copyright (C) 2005 SCI Group

KEYWORDS
   Contact_Model Composite

DESCRIPTION
   Long description...
  
WARNING

****************************************/

    class CompositeContact :public Contact {
      public:
         // Constructor
         CompositeContact(const ProcessorGroup* myworld, MPMLabel* Mlb, MPMFlags* MFlag);
	 virtual ~CompositeContact();
         
         // memory deleted on destruction of composite
         void add(Contact * m);
         
         // how many 
         size_t size() const { return d_m.size(); }
         
	 // Basic contact methods
	 void exMomInterpolated(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset* matls,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw);
	 
	 void exMomIntegrated(const ProcessorGroup*,
                              const PatchSubset* patches,
                              const MaterialSubset* matls,
                              DataWarehouse* old_dw,
                              DataWarehouse* new_dw);
         
         void addComputesAndRequiresInterpolated(SchedulerP & sched,
                                                 const PatchSet* patches,
                                                 const MaterialSet* matls);
	 
         void addComputesAndRequiresIntegrated(SchedulerP & sched,
                                               const PatchSet* patches,
                                               const MaterialSet* matls);

         void initFriction(const ProcessorGroup*,
                           const PatchSubset*,
                           const MaterialSubset* matls,
                           DataWarehouse*,
                           DataWarehouse* new_dw);
         
      private: // hide
         CompositeContact(const CompositeContact &);
         CompositeContact& operator=(const CompositeContact &);


         
      protected: // data
         std::list< Contact * > d_m;
      };
      
} // End namespace Uintah

#endif // __COMPOSITE_CONTACT_H__
