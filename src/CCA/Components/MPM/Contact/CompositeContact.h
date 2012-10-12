/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef __COMPOSITE_CONTACT_H__
#define __COMPOSITE_CONTACT_H__

#include <CCA/Components/MPM/Contact/Contact.h>
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
 

KEYWORDS
   Contact_Model Composite

DESCRIPTION
   Long description...
  
WARNING

****************************************/

    class CompositeContact :public Contact {
      public:
         // Constructor
         CompositeContact(const ProcessorGroup* myworld, MPMLabel* Mlb, 
                          MPMFlags* MFlag);
         virtual ~CompositeContact();

         virtual void outputProblemSpec(ProblemSpecP& ps);
         
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
