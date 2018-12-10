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


#ifndef ON_THE_FLY_RADIOMETER_H
#define ON_THE_FLY_RADIOMETER_H
#include <CCA/Components/OnTheFlyAnalysis/AnalysisModule.h>
#include <CCA/Components/Models/Radiation/RMCRT/RMCRTCommon.h>
#include <CCA/Components/Models/Radiation/RMCRT/Radiometer.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/LevelP.h>

namespace Uintah {
/*______________________________________________________________________

  Class:        On-the-Fly radiometer

  Author:       Todd Harman

  Description:  This call the RMCRT: radiometer methods.  This
                allows the user to compare directly with experimental
                data regardless of what method is used to solve
                the radiative transfer equation.

  Dependencies:  This assumes that the component has precomputed
                     - temperature
                     - aborption coefficient
                     - cellType
_____________________________________________________________________*/
  class OnTheFly_radiometer : public AnalysisModule {
  public:
    OnTheFly_radiometer(const ProcessorGroup* myworld,
                        const MaterialManagerP materialManager,
                        const ProblemSpecP& module_spec);

    OnTheFly_radiometer();

    virtual ~OnTheFly_radiometer();

    virtual void problemSetup(const ProblemSpecP& prob_spec,
                              const ProblemSpecP& restart_prob_spec,
                              GridP& grid,
                              std::vector<std::vector<const VarLabel* > > &PState,
                              std::vector<std::vector<const VarLabel* > > &PState_preReloc);

    virtual void outputProblemSpec(ProblemSpecP& ps){};

    virtual void scheduleInitialize(SchedulerP& sched,
                                    const LevelP& level);
                                    
    virtual void scheduleRestartInitialize(SchedulerP& sched,
                                           const LevelP& level){};

    virtual void restartInitialize();

    virtual void scheduleDoAnalysis(SchedulerP& sched,
                                    const LevelP& level);

    void scheduleDoAnalysis_preReloc(SchedulerP& sched,
                                    const LevelP& level) {};
  //______________________________________________________________________

  private:

    void initialize(const ProcessorGroup*,
                    const PatchSubset* patches,
                    const MaterialSubset*,
                    DataWarehouse*,
                    DataWarehouse* new_dw);

    void doAnalysis(const ProcessorGroup* pg,
                    const PatchSubset* patches,
                    const MaterialSubset*,
                    DataWarehouse*,
                    DataWarehouse* new_dw);

    Radiometer* d_radiometer;
  };
}

#endif
