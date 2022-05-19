/*
 * The MIT License
 *
 * Copyright (c) 1997-2021 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
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

// ContactStressIndependent.h

#ifndef __CONTACT_STRESS_INDEPENDENT
#define __CONTACT_STRESS_INDEPENDENT

#include <CCA/Components/MPM/Materials/Dissolution/Dissolution.h>
#include <CCA/Components/MPM/Materials/Dissolution/DissolutionMaterialSpec.h> 
#include <CCA/Ports/DataWarehouseP.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/GridP.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/MaterialManagerP.h>
#include <Core/Grid/Task.h>

namespace Uintah {
/**************************************

CLASS
   ContactStressIndependent
   
   Short description...

GENERAL INFORMATION

   ContactStressIndependent.h

   James E. Guilkey
   Laird Avenue Consulting/University of Utah

KEYWORDS
   Dissolution_Model_ContactStressIndependent

DESCRIPTION
  One of the derived Dissolution classes.
WARNING
  
****************************************/

      class ContactStressIndependent : public Dissolution {
      private:

        // Prevent copying of this class
        // copy constructor
        ContactStressIndependent(const ContactStressIndependent &ci);
        ContactStressIndependent& operator=(const ContactStressIndependent &ci);

        MaterialManagerP    d_materialManager;

        // Dissolution rate
        double d_Vm;
        double d_R;
        double d_StressThresh;
        double d_maxCemThickness;  // Diss doesn't occur for thicker overgrowth
        double d_Ao;
        double d_Ea;
        double d_Ao_clay; // Modified value in the presence of clay
        double d_Ea_clay; // Modified value in the presence of clay
        // master material
        int    d_masterModalID;
        int    d_inContactWithModalID;

      public:
         // Constructor
         ContactStressIndependent(const ProcessorGroup* myworld,
                          ProblemSpecP& ps,MaterialManagerP& d_sS,MPMLabel* lb);

         // Destructor
         virtual ~ContactStressIndependent();

         virtual void outputProblemSpec(ProblemSpecP& ps);

         // Dissolution methods
         virtual void computeMassBurnFraction(const ProcessorGroup*,
                                              const PatchSubset* patches,
                                              const MaterialSubset* matls,
                                              DataWarehouse* old_dw,
                                              DataWarehouse* new_dw);

         virtual void addComputesAndRequiresMassBurnFrac(SchedulerP & sched,
                                                    const PatchSet* patches,
                                                    const MaterialSet* matls);
      };
} // End namespace Uintah

#endif /* __CONTACT_STRESS_INDEPENDENT */
