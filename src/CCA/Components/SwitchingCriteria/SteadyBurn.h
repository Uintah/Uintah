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

#ifndef Packages_Uintah_CCA_Components_Switching_SteadyBurn_h
#define Packages_Uintah_CCA_Components_Switching_SteadyBurn_h

#include <Core/ProblemSpec/ProblemSpecP.h>
#include <CCA/Ports/SwitchingCriteria.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Labels/MPMLabel.h>
#include <Core/Labels/MPMICELabel.h>
#include <Core/Labels/ICELabel.h>

namespace Uintah {

  class ProcessorGroup;
  class DataWarehouse;

  class SteadyBurnCriteria : public SwitchingCriteria
    {
    public:
      SteadyBurnCriteria(ProblemSpecP& ps);
      virtual ~SteadyBurnCriteria();
      
      virtual void problemSetup(const ProblemSpecP& ps, 
                                const ProblemSpecP& restart_prob_spec, 
                                SimulationStateP& sharedState);

      virtual void scheduleSwitchTest(const LevelP& level, SchedulerP& sched);

      void switchTest(const ProcessorGroup*, const PatchSubset* patches,
                      const MaterialSubset* matls, DataWarehouse*,
                      DataWarehouse*);


    private:
      unsigned int d_material;
      double d_temperature;   
      double d_BP;              // Number of Particles at Boundary
      
      SimulationStateP d_sharedState; 
      MPMLabel* Mlb;
      MPMICELabel* MIlb;
      ICELabel* Ilb;
      #define d_SMALL_NUM 1e-100
    };
} // End namespace Uintah

#endif 
