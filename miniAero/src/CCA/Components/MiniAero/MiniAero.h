/*
 * The MIT License
 *
 * Copyright (c) 1997-2014 The University of Utah
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

#ifndef CCA_COMPONENTS_MINIAERO_H
#define CCA_COMPONENTS_MINIAERO_H

#include <CCA/Ports/SimulationInterface.h>
#include <Core/Parallel/UintahParallelComponent.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/Variables/VarLabel.h>

namespace Uintah {

class SimpleMaterial;

/**************************************

 CLASS
 MiniAero

 MiniAero simulation


 GENERAL INFORMATION

 MiniAero.h


 KEYWORDS
 MiniAero


 DESCRIPTION
 2D implementation of MiniAero's Algorithm using Euler's method.

 ****************************************/

class MiniAero : public UintahParallelComponent, public SimulationInterface {

  public:

    MiniAero(const ProcessorGroup* myworld);

    virtual ~MiniAero();

    virtual void problemSetup(const ProblemSpecP& params,
                              const ProblemSpecP& restart_prob_spec,
                              GridP& grid,
                              SimulationStateP&);

    virtual void scheduleInitialize(const LevelP& level,
                                    SchedulerP& sched);

    virtual void scheduleComputeStableTimestep(const LevelP& level,
                                               SchedulerP&);

    virtual void scheduleTimeAdvance(const LevelP& level,
                                     SchedulerP&);

  private:

    void initialize(const ProcessorGroup*,
                    const PatchSubset* patches,
                    const MaterialSubset* matls,
                    DataWarehouse* old_dw,
                    DataWarehouse* new_dw);

    void computeStableTimestep(const ProcessorGroup*,
                               const PatchSubset* patches,
                               const MaterialSubset* matls,
                               DataWarehouse* old_dw,
                               DataWarehouse* new_dw);

    void timeAdvance(const ProcessorGroup*,
                     const PatchSubset* patches,
                     const MaterialSubset* matls,
                     DataWarehouse* old_dw,
                     DataWarehouse* new_dw);


    void schedConvertOutput(const LevelP& level,SchedulerP& sched);
    
    void convertOutput(const ProcessorGroup*,
                     const PatchSubset* patches,
                     const MaterialSubset* matls,
                     DataWarehouse* old_dw,
                     DataWarehouse* new_dw);

    const VarLabel* conserved_label;
    const VarLabel* rho_CClabel;
    const VarLabel* vel_CClabel;
    const VarLabel* press_CClabel;
    const VarLabel* temp_CClabel;
    SimulationStateP sharedState_;
    double delt_;
    SimpleMaterial* mymat_;

    MiniAero(const MiniAero&);
    MiniAero& operator=(const MiniAero&);

};  // end class MiniAero

}  // end namespace Uintah

#endif  // end CCA_COMPONENTS_MINIAERO_H
