/*
 * The MIT License
 *
 * Copyright (c) 1997-2025 The University of Utah
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

// QuartzOvergrowth.h

#ifndef __QUARTZ_OVERGROWTH_MODEL
#define __QUARTZ_OVERGROWTH_MODEL

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
   QuartzOvergrowth
   
   Short description...

GENERAL INFORMATION

   QuartzOvergrowth.h

   James E. Guilkey
   Laird Avenue Consulting/University of Utah

KEYWORDS
   Dissolution_Model_QuartzOvergrowth

DESCRIPTION
  One of the derived Dissolution classes.
WARNING
  
****************************************/

      class QuartzOvergrowth : public Dissolution {
      private:
         
        // Prevent copying of this class
        // copy constructor
        QuartzOvergrowth(const QuartzOvergrowth &con);
        QuartzOvergrowth& operator=(const QuartzOvergrowth &con);

        MaterialManagerP    d_materialManager;

        // Growth rate
        double d_growthRate;
        double d_growthRateClay;
        // master material
        int    d_masterModalID;

      public:
         // Constructor
         QuartzOvergrowth(const ProcessorGroup* myworld,
                          ProblemSpecP& ps,MaterialManagerP& d_sS,MPMLabel* lb);

         // Destructor
         virtual ~QuartzOvergrowth();

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

#endif /* __QUARTZ_OVERGROWTH_MODEL */
