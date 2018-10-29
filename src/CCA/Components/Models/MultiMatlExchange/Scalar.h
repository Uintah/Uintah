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
#ifndef Models_MultiMatlExchange_Scalar_h
#define Models_MultiMatlExchange_Scalar_h

#include <CCA/Components/ICE/Core/ICELabel.h>
#include <CCA/Components/Models/MultiMatlExchange/ExchangeCoefficients.h>
#include <CCA/Components/Models/MultiMatlExchange/ExchangeModel.h>
#include <CCA/Components/MPM/Core/MPMLabel.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/MaterialManagerP.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Math/FastMatrix.h>

namespace Uintah {
namespace ExchangeModels{

//______________________________________________________________________
//  This exchange model is the Kashiwa model from CFDLIB

  class ScalarExch : public ExchangeModel {

  public:
    ScalarExch(const ProblemSpecP     & prob_spec,
               const MaterialManagerP & materialManager );

    virtual ~ScalarExch();

    virtual void problemSetup(const ProblemSpecP & prob_spec);

    virtual void outputProblemSpec(ProblemSpecP & prob_spec);

    virtual void sched_PreExchangeTasks(SchedulerP           & sched,
                                        const PatchSet       * patches,     
                                        const MaterialSubset * iceMatls,    
                                        const MaterialSet    * allMatls){};
                                        
    virtual void addExchangeModelRequires ( Task* t,
                                            const MaterialSubset * zeroMatls,
                                            const MaterialSubset * iceMatls,
                                            const MaterialSubset * mpmMatls){};  

    virtual void sched_AddExch_VelFC(SchedulerP           & sched,
                                     const PatchSet       * patches,
                                     const MaterialSubset * iceMatls,
                                     const MaterialSubset * mpmMatls,
                                     const MaterialSet    * allMatls,
                                     customBC_globalVars  * BC_globalVars,
                                     const bool recursion);


    virtual void addExch_VelFC(const ProcessorGroup * pg,
                               const PatchSubset    * patch,
                               const MaterialSubset * matls,
                               DataWarehouse        * old_dw,
                               DataWarehouse        * new_dw,
                               customBC_globalVars  * BC_globalVars,
                               const bool recursion);

    virtual void sched_AddExch_Vel_Temp_CC(SchedulerP           & sched,
                                           const PatchSet       * patches,
                                           const MaterialSubset * ice_matls,
                                           const MaterialSubset * mpm_matls,
                                           const MaterialSet    * all_matls,
                                           customBC_globalVars  * BC_globalVars);

    virtual void addExch_Vel_Temp_CC( const ProcessorGroup * pg,
                                      const PatchSubset    * patches,
                                      const MaterialSubset * matls,
                                      DataWarehouse        * old_dw,
                                      DataWarehouse        * new_dw,
                                      customBC_globalVars  * BC_globalVars);
  //__________________________________
  private:
    template<class constSFC, class SFC>
    void vel_FC_exchange( CellIterator  iter,
                          IntVector     adj_offset,
                          int           numMatls,
                          FastMatrix  & K,
                          double        delT,
                          std::vector<constCCVariable<double> >& vol_frac_CC,
                          std::vector<constCCVariable<double> >& sp_vol_CC,
                          std::vector< constSFC>               & vel_FC,
                          std::vector< SFC >                   & sp_vol_FC,
                          std::vector< SFC >                   & vel_FCME);

    void addExch_Vel_Temp_CC_1matl(const ProcessorGroup * pg,
                                   const PatchSubset    * patches,
                                   const MaterialSubset * matls,
                                   DataWarehouse        * old_dw,
                                   DataWarehouse        * new_dws,
                                   customBC_globalVars  * BC_globalVars);

    //__________________________________
    //  variables
    ExchangeCoefficients* d_exchCoeff;
    MPMLabel* Mlb;
    ICELabel* Ilb;
  };
}
}

#endif
