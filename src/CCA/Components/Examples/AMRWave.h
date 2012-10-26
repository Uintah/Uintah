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


#ifndef Packages_Uintah_CCA_Components_Examples_AMRWave_h
#define Packages_Uintah_CCA_Components_Examples_AMRWave_h

#include <CCA/Components/Examples/Wave.h>

namespace Uintah {
  class SimpleMaterial;
  class ExamplesLabel;

/**************************************

CLASS
   AMRWave
   
   AMRWave simulation

GENERAL INFORMATION

   AMRWave.h

   Steven Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   
KEYWORDS
   AMRWave

DESCRIPTION
  
WARNING
  
****************************************/

  class VarLabel;
  class AMRWave : public Wave {
  public:
    AMRWave(const ProcessorGroup* myworld);
    virtual ~AMRWave();

    virtual void problemSetup(const ProblemSpecP& params, 
                              const ProblemSpecP& restart_prob_spec, 
                              GridP& grid, SimulationStateP& sharedState);
    virtual void scheduleRefineInterface(const LevelP& fineLevel,
					 SchedulerP& scheduler,
					 bool needCoarseOld, bool needCoarseNew);
    virtual void scheduleCoarsen(const LevelP& coarseLevel, SchedulerP& sched);
    virtual void scheduleRefine (const PatchSet* patches, SchedulerP& sched);

    virtual void scheduleErrorEstimate(const LevelP& coarseLevel,
				       SchedulerP& sched);
    virtual void scheduleInitialErrorEstimate(const LevelP& coarseLevel,
                                              SchedulerP& sched);
    virtual void scheduleTimeAdvance( const LevelP& level, SchedulerP&);
  protected:
    virtual void addRefineDependencies( Task* /*task*/, const VarLabel* /*label*/, bool needCoarseOld, bool needCoarseNew);
    virtual void refineFaces(const Patch* finePatch, const Level* fineLevel, const Level* coarseLevel, 
                      CCVariable<double>& finevar, const VarLabel* label,
                      int matl, DataWarehouse* coarse_old_dw, DataWarehouse* coarse_new_dw);
  private:
    void errorEstimate(const ProcessorGroup*,
		       const PatchSubset* patches,
		       const MaterialSubset* matls,
		       DataWarehouse*, DataWarehouse* new_dw);
    void refine(const ProcessorGroup*,
                const PatchSubset* patches,
                const MaterialSubset* matls,
                DataWarehouse*, DataWarehouse* new_dw);
    void coarsen(const ProcessorGroup*,
                 const PatchSubset* patches,
                 const MaterialSubset* matls,
                 DataWarehouse*, DataWarehouse* new_dw);

    void refineCell(CCVariable<double>& finevar, constCCVariable<double>& coarsevar, IntVector fineIndex,
                    const Level* fineLevel, const Level* coarseLevel); 
    void coarsenCell(CCVariable<double>& coarsevar, constCCVariable<double>& finevar, IntVector coarseIndex,
                    const Level* fineLevel, const Level* coarseLevel); 

    AMRWave(const AMRWave&);
    AMRWave& operator=(const AMRWave&);
	 
    double refine_threshold;
    bool do_refineFaces;
    bool do_refine;
    bool do_coarsen;
  };
}

#endif
