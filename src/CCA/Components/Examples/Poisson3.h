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


#ifndef Packages_Uintah_CCA_Components_Examples_Poisson3_h
#define Packages_Uintah_CCA_Components_Examples_Poisson3_h

#include <CCA/Components/Examples/Interpolator.h>
#include <Core/Parallel/UintahParallelComponent.h>
#include <CCA/Ports/SimulationInterface.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/VarLabel.h>

namespace Uintah {
  class SimpleMaterial;


/**************************************

CLASS
   Poisson3
   
   Poisson3 simulation

GENERAL INFORMATION

   Poisson3.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   
KEYWORDS
   Poisson3

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/




  class Poisson3 : public UintahParallelComponent, public SimulationInterface {
  public:
    Poisson3(const ProcessorGroup* myworld);
    virtual ~Poisson3();

    virtual void problemSetup(const ProblemSpecP& params, 
                              const ProblemSpecP& restart_prob_spec, 
                              GridP& grid, SimulationStateP&);
    virtual void scheduleInitialize(const LevelP& level,
				    SchedulerP& sched);
    virtual void scheduleComputeStableTimestep(const LevelP& level,
					       SchedulerP&);
    virtual void scheduleTimeAdvance( const LevelP& level, 
				      SchedulerP&);

    // New functions
    virtual void scheduleRefine(const LevelP& fineLevel, SchedulerP& sched);
    void refine(const ProcessorGroup* pg,
                const PatchSubset* finePatches, 
		const MaterialSubset* matls,
                DataWarehouse*, 
                DataWarehouse* newDW);

    virtual void scheduleRefineInterface(const LevelP& fineLevel,
					 SchedulerP& scheduler,
					 bool needCoarseOld, bool needCoarseNew);
    void refineInterface(const ProcessorGroup*,
			 const PatchSubset* finePatches, 
			 const MaterialSubset* matls,
			 DataWarehouse* fineDW, 
			 DataWarehouse* coarseDW);

    virtual void scheduleCoarsen(const LevelP& coarseLevel, SchedulerP& sched);
    void coarsen(const ProcessorGroup* pg,
	         const PatchSubset* finePatches, 
		 const MaterialSubset* matls,
                 DataWarehouse* coarseDW, 
                 DataWarehouse* fineDW);

  private:
    void initialize(const ProcessorGroup*,
		    const PatchSubset* patches, const MaterialSubset* matls,
		    DataWarehouse* old_dw, DataWarehouse* new_dw);
    void computeStableTimestep(const ProcessorGroup*,
			       const PatchSubset* patches,
			       const MaterialSubset* matls,
			       DataWarehouse* old_dw, DataWarehouse* new_dw);
    void timeAdvance(const ProcessorGroup*,
		     const PatchSubset* patches,
		     const MaterialSubset* matls,
		     DataWarehouse* old_dw, DataWarehouse* new_dw,
		     bool modify);

    const VarLabel* phi_label;
    const VarLabel* residual_label;
    SimulationStateP sharedState_;
    double delt_;
    SimpleMaterial* mymat_;
    Interpolator interpolator_;
    int max_int_support_;

    Poisson3(const Poisson3&);
    Poisson3& operator=(const Poisson3&);
	 
  };
}



#endif
