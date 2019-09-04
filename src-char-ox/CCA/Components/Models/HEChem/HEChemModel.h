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

#ifndef UINTAH_HOMEBREW_HECHEMMODEL_H
#define UINTAH_HOMEBREW_HECHEMMODEL_H

#include <CCA/Ports/ModelInterface.h>
#include <CCA/Ports/SchedulerP.h>

#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/GridP.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/ProblemSpec/ProblemSpecP.h>

/**************************************

CLASS
   HEChemModel
   
   Short description...

GENERAL INFORMATION

   HEChemModel.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Model of Accidental Fires and Explosions (C-SAFE)
  
   
KEYWORDS
   HEChem Model

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

namespace Uintah {

  class DataWarehouse;
  class ProcessorGroup;
  
  //________________________________________________
  class HEChemModel : public ModelInterface {
  public:
    HEChemModel(const ProcessorGroup* myworld,
		const SimulationStateP sharedState)
      : ModelInterface(myworld, sharedState) {};

    virtual ~HEChemModel() {};

    virtual void problemSetup(GridP& grid, const bool isRestart) = 0;
      
    virtual void outputProblemSpec(ProblemSpecP& ps) = 0;

    virtual void scheduleInitialize(SchedulerP& scheduler,
				    const LevelP& level) = 0;

    virtual void scheduleComputeStableTimeStep(SchedulerP& scheduler,
					       const LevelP& level) = 0;

    // Used by DDT1 ONLY.
    virtual void scheduleRefine(const PatchSet* patches,
				SchedulerP& sched) {};

    virtual void scheduleComputeModelSources(SchedulerP& scheduler,
					     const LevelP& level) = 0;

    // Used by LightTime ONLY.
    virtual void scheduleErrorEstimate(const LevelP& coarseLevel,
                                       SchedulerP& sched) {};
                                               
  private:     
    HEChemModel(const HEChemModel&);
    HEChemModel& operator=(const HEChemModel&);
  };
} // End namespace Uintah
   
#endif
