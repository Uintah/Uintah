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

#ifndef Models_MultiMatlExchange_Exchange_h
#define Models_MultiMatlExchange_Exchange_h



#include <CCA/Components/ICE/CustomBCs/C_BC_driver.h>
#include <CCA/Components/MPM/Core/MPMLabel.h>
#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Ports/SchedulerP.h>
#include <Core/Grid/DbgOutput.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/MaterialManagerP.h>
#include <Core/Grid/Variables/ComputeSet.h>

namespace Uintah {

  class DataWarehouse;
  class Material;
  class Patch;
  class ExchangeModel {

  public:
    ExchangeModel(const ProblemSpecP     & prob_spec,
                  const MaterialManagerP & materialManager );

    virtual ~ExchangeModel();

    virtual void problemSetup(const ProblemSpecP & prob_spec ) = 0;
    
    virtual void outputProblemSpec(ProblemSpecP & prob_spec ) = 0;
  
    virtual void sched_PreExchangeTasks(SchedulerP           & sched,
                                        const PatchSet       * patches,     
                                        const MaterialSubset * iceMatls,    
                                        const MaterialSet    * allMatls) = 0;
                                        
    virtual void addExchangeModelRequires ( Task* t,
                                            const MaterialSubset * zeroMatls,
                                            const MaterialSubset * iceMatls,
                                            const MaterialSubset * mpmMatls) = 0;

    virtual void sched_AddExch_VelFC(SchedulerP           & sched,
                                     const PatchSet       * patches,
                                     const MaterialSubset * iceMatls,
                                     const MaterialSubset * mpmMatls,
                                     const MaterialSet    * allMatls,
                                     customBC_globalVars  * BC_globalVars,
                                     const bool recursion) = 0;


    virtual void addExch_VelFC(const ProcessorGroup  * pg,
                               const PatchSubset     * patch,
                               const MaterialSubset  * matls,
                               DataWarehouse         * old_dw,
                               DataWarehouse         * new_dw,
                               customBC_globalVars   * BC_globalVars,
                               const bool recursion) = 0;

    virtual void sched_AddExch_Vel_Temp_CC(SchedulerP           & sched,
                                           const PatchSet       * patches,
                                           const MaterialSubset * ice_matls,
                                           const MaterialSubset * mpm_matls,
                                           const MaterialSet    * all_matls,
                                           customBC_globalVars  * BC_globalVars) = 0;

    virtual void addExch_Vel_Temp_CC( const ProcessorGroup * pg,
                                      const PatchSubset    * patches,
                                      const MaterialSubset * matls,
                                      DataWarehouse        * old_dw,
                                      DataWarehouse        * new_dw,
                                      customBC_globalVars  * BC_globalVars) = 0;

    void schedComputeSurfaceNormal( SchedulerP     & sched,
                                    const PatchSet * patches );

    void ComputeSurfaceNormal( const ProcessorGroup *,
                               const PatchSubset    * patches,
                               const MaterialSubset *,
                               DataWarehouse        * old_dw,
                               DataWarehouse        * new_dw );

    //__________________________________
    // variables & objects needed by
    // the different exchange models.
    const VarLabel* d_surfaceNormLabel;
    const VarLabel* d_isSurfaceCellLabel;

    double d_SMALL_NUM = 1.0e-100;
    int    d_numMatls  = -9;
    MaterialManagerP  d_materialManager;
    MaterialSubset * d_zero_matl;

  private:
    MPMLabel* Mlb;

  };
}

#endif

