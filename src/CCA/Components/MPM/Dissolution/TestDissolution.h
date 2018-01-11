/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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

// TestDissolution.h

#ifndef __TEST_DISSOLUTION_H__
#define __TEST_DISSOLUTION_H__

#include <CCA/Components/MPM/Dissolution/Dissolution.h>
#include <CCA/Components/MPM/Dissolution/DissolutionMaterialSpec.h> 
#include <CCA/Components/MPM/MPMFlags.h>
#include <CCA/Ports/DataWarehouseP.h>
#include <Core/Parallel/UintahParallelComponent.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/GridP.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/Grid/Task.h>



namespace Uintah {
/**************************************

CLASS
   TestDissolution
   
   Short description...

GENERAL INFORMATION

   TestDissolution.h

   James E. Guilkey
   Laird Avenue Consulting/University of Utah

KEYWORDS
   Dissolution_Model_TestDissolution

DESCRIPTION
  One of the derived Dissolution classes.
WARNING
  
****************************************/

      class TestDissolution : public Dissolution {
      private:
         
         // Prevent copying of this class
         // copy constructor
         TestDissolution(const TestDissolution &con);
         TestDissolution& operator=(const TestDissolution &con);
         
         SimulationStateP    d_sharedState;
         
         // Dissolution rate
         double d_rate;
         double d_PressThresh;
         // master material
         int       d_material;

      public:
         // Constructor
         TestDissolution(const ProcessorGroup* myworld,
                          ProblemSpecP& ps,SimulationStateP& d_sS,MPMLabel* lb,
                          MPMFlags* MFlag);
         
         // Destructor
         virtual ~TestDissolution();

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

#endif /* __TEST_DISSOLUTION_H__ */
