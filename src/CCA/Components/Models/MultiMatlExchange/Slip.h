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

#ifndef Models_MultiMatlExchange_Slip_h
#define Models_MultiMatlExchange_Slip_h

#include <CCA/Components/Models/MultiMatlExchange/ExchangeModel.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/Grid/Variables/ComputeSet.h>

namespace Uintah {
namespace ExchangeModels{

//______________________________________________________________________
//
  class SlipExch : public ExchangeModel {

  public:
    SlipExch(const ProblemSpecP     & prob_spec,
             const SimulationStateP & sharedState );

    virtual ~SlipExch();

    virtual void problemSetup();


    virtual void scheduleAddExchangeToMomentumAndEnergy(SchedulerP           & sched,
                                                        const PatchSet       * patches,
                                                        const MaterialSubset * ice_matls,
                                                        const MaterialSubset * mpm_matls,
                                                        const MaterialSubset * press_matl,
                                                        const MaterialSet    * all_matls);

    virtual void addExchangeToMomentumAndEnergy( const ProcessorGroup * pg,
                                                 const PatchSubset    * patches,
                                                 const MaterialSubset * matls,
                                                 DataWarehouse        * old_dw,
                                                 DataWarehouse        * new_dw);

  private:
    int    d_fluidMatlIndx = -9;
    int    d_solidMatlIndx = -9;
    double d_momentum_accommodation_coeff = -9;
    double d_thermal_accommodation_coeff  = -9;
    ProblemSpecP d_prob_spec;

  };
}
}

#endif
