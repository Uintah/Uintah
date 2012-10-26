/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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


#ifndef Packages_Uintah_CCA_Components_Examples_Wave_h
#define Packages_Uintah_CCA_Components_Examples_Wave_h

#include <Core/Parallel/UintahParallelComponent.h>
#include <CCA/Ports/SimulationInterface.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Task.h>

namespace Uintah {
  class SimpleMaterial;
  class ExamplesLabel;

/**************************************

CLASS
   Wave
   
   Wave simulation

GENERAL INFORMATION

   Wave.h

   Steven Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   
KEYWORDS
   Wave

DESCRIPTION
  
WARNING
  
****************************************/

  class VarLabel;
  class Wave : public UintahParallelComponent, public SimulationInterface {
  public:
    Wave(const ProcessorGroup* myworld);
    virtual ~Wave();

    virtual void problemSetup(const ProblemSpecP& params, 
                              const ProblemSpecP& restart_prob_spec,
                              GridP& grid, SimulationStateP&);
    virtual void scheduleInitialize(const LevelP& level,
				    SchedulerP& sched);
    virtual void scheduleComputeStableTimestep(const LevelP& level,
					       SchedulerP&);
    virtual void scheduleTimeAdvance( const LevelP& level, 
				      SchedulerP&);
  protected:
    struct Step {
      Task::WhichDW cur_dw;
      const VarLabel* curphi_label;
      const VarLabel* curpi_label;
      const VarLabel* newphi_label;
      const VarLabel* newpi_label;
      double stepweight;
      double totalweight;
    };


    inline double curl(constCCVariable<double>& phi, const IntVector& c, double sumdx2, Vector inv_dx2) 
    {
      return sumdx2 * phi[c]
        + (phi[c+IntVector(1,0,0)] + phi[c-IntVector(1,0,0)]) * inv_dx2.x()
        + (phi[c+IntVector(0,1,0)] + phi[c-IntVector(0,1,0)]) * inv_dx2.y()
        + (phi[c+IntVector(0,0,1)] + phi[c-IntVector(0,0,1)]) * inv_dx2.z();
    }

    void initialize(const ProcessorGroup*,
		    const PatchSubset* patches, const MaterialSubset* matls,
		    DataWarehouse* old_dw, DataWarehouse* new_dw);
    void computeStableTimestep(const ProcessorGroup*,
			       const PatchSubset* patches,
			       const MaterialSubset* matls,
			       DataWarehouse* old_dw, DataWarehouse* new_dw);
    void timeAdvanceEuler(const ProcessorGroup*,
		     const PatchSubset* patches,
		     const MaterialSubset* matls,
                          DataWarehouse* old_dw, DataWarehouse* new_dw);
    void setupRK4(const ProcessorGroup*,
                  const PatchSubset* patches,
                  const MaterialSubset* matls,
                  DataWarehouse* old_dw, DataWarehouse* new_dw);
    void timeAdvanceRK4(const ProcessorGroup*,
                        const PatchSubset* patches,
                        const MaterialSubset* matls,
                        DataWarehouse* old_dw, DataWarehouse* new_dw, Step* s);
    virtual void addRefineDependencies( Task* /*task*/, const VarLabel* /*label*/,bool needCoarseOld, bool needCoarseNew) {}
    virtual void refineFaces(const Patch* /*finePatch*/, const Level* /*fineLevel*/, const Level* /*coarseLevel*/, 
                             CCVariable<double>& /*finevar*/, const VarLabel* /*label*/,
                             int /*matl*/, DataWarehouse* /*coarse_old_dw*/, DataWarehouse* /*coarse_new_dw*/) {}

    const VarLabel* phi_label;
    const VarLabel* pi_label;
    double r0;
    string initial_condition;
    string integration;
    SimulationStateP sharedState_;
    SimpleMaterial* mymat_;
    Step rk4steps[4];

    Wave(const Wave&);
    Wave& operator=(const Wave&);
	 
  };
}

#endif
