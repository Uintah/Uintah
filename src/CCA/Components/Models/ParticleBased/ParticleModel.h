/*
 * The MIT License
 *
 * Copyright (c) 1997-2023 The University of Utah
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

#ifndef UINTAH_HOMEBREW_PARTICLEMODEL_H
#define UINTAH_HOMEBREW_PARTICLEMODEL_H

#include <CCA/Ports/ModelInterface.h>
#include <CCA/Ports/SchedulerP.h>

#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/GridP.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/MaterialManagerP.h>
#include <Core/ProblemSpec/ProblemSpecP.h>

//______________________________________________________________________
//
namespace Uintah {

  class DataWarehouse;
  class ProcessorGroup;

  //________________________________________________
  class ParticleModel : public ModelInterface {
  public:
    ParticleModel(const ProcessorGroup* myworld,
                  const MaterialManagerP materialManager)
      : ModelInterface(myworld, materialManager) {};

    virtual ~ParticleModel() {};

    virtual void problemSetup(GridP& grid, const bool isRestart) = 0;

    virtual void outputProblemSpec(ProblemSpecP& ps) = 0;

    virtual void scheduleInitialize(SchedulerP  & scheduler,
                                    const LevelP& level) = 0;
                                    
    virtual void scheduleRestartInitialize(SchedulerP&,
                                           const LevelP& level){};

    virtual void scheduleComputeStableTimeStep(SchedulerP   & scheduler,
                                               const LevelP & level){};

    virtual void scheduleRefine(const PatchSet * patches,
                                SchedulerP     & sched){};

    virtual void scheduleComputeModelSources(SchedulerP  & scheduler,
                                             const LevelP& level) = 0;


    const MaterialSubset* d_matl_mss;

    // used for particle relocation
    std::vector<const VarLabel* > d_newLabels;
    std::vector<const VarLabel* > d_oldLabels;
    VarLabel * pXLabel;           // particle position label
    VarLabel * pXLabel_preReloc;
    
    VarLabel * pIDLabel;          // particle ID label, of type long64
    VarLabel * pIDLabel_preReloc;

  protected:
    const Material* d_matl;
    MaterialSet*    d_matl_set;

    const TypeDescription* d_Part_point  = ParticleVariable<Point>::getTypeDescription();
    const TypeDescription* d_Part_Vector = ParticleVariable<Vector>::getTypeDescription();
    const TypeDescription* d_Part_double = ParticleVariable<double>::getTypeDescription();
    const TypeDescription* d_Part_long64 = ParticleVariable<long64>::getTypeDescription();

  private:
    ParticleModel(const ParticleModel&);
    ParticleModel& operator=(const ParticleModel&);
  };
} // End namespace Uintah

#endif
