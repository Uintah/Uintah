/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef __CONTACT_H__
#define __CONTACT_H__

#include <CCA/Components/MPM/Contact/ContactMaterialSpec.h>
#include <Core/Parallel/UintahParallelComponent.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/SchedulerP.h>
#include <cmath>

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
  

KEYWORDS
   Contact_Model

DESCRIPTION
   Long description...
  
WARNING

****************************************/

  class Contact : public UintahParallelComponent {
      public:
         // Constructor
         Contact(const ProcessorGroup* myworld, MPMLabel* Mlb, MPMFlags* MFlag,
                 ProblemSpecP ps);
         virtual ~Contact();

         virtual void outputProblemSpec(ProblemSpecP& ps) = 0;

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
            double EPSILON=1.e-14;
            
            return (fabs(num1-num2) <= EPSILON);
      }

} // End namespace Uintah

#endif // __CONTACT_H__
