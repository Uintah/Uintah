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


#ifndef UINTAH_CCA_COMPONENTS_FVM_ELECTROSTATICSOLVE_H
#define UINTAH_CCA_COMPONENTS_FVM_ELECTROSTATICSOLVE_H

#include <CCA/Components/Application/ApplicationCommon.h>

#include <Core/Grid/Grid.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/MaterialManagerP.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/ProblemSpec/ProblemSpec.h>

namespace Uintah {

/**************************************

CLASS
   ElectrostaticSolv

GENERAL INFORMATION
   ElectrostaticSolv.h

KEYWORDS
   ElectrostaticSolv

DESCRIPTION
   A solver for electrostatic potential.
  
WARNING
  
****************************************/
  class FVMLabel;

  class SolverInterface;

  class DataWarehouse;
  class ProcessorGroup;

  class ElectrostaticSolve : public ApplicationCommon {
  public:
    ElectrostaticSolve(const ProcessorGroup* myworld,
		       const MaterialManagerP materialManager);
    virtual ~ElectrostaticSolve();

    virtual void problemSetup(const ProblemSpecP& params,
                              const ProblemSpecP& restart_prob_spec,
			      GridP& grid);

    virtual void outputProblemSpec(ProblemSpecP& ps);
                              
    virtual void scheduleInitialize(const LevelP& level,
                                    SchedulerP& sched);
                                    
    virtual void scheduleRestartInitialize(const LevelP& level,
                                           SchedulerP& sched);
                                           
    virtual void scheduleComputeStableTimeStep(const LevelP& level,
                                               SchedulerP&);
    virtual void scheduleTimeAdvance( const LevelP& level, 
                                      SchedulerP&);

    virtual void setWithMPM(bool value) { d_with_mpm = value; }

  protected:
    FVMLabel* d_lb;
    friend class ESMPM;
    friend class ESMPM2;

  private:
    void initialize(const ProcessorGroup*,
                    const PatchSubset* patches, const MaterialSubset* matls,
                    DataWarehouse* old_dw, DataWarehouse* new_dw);
    void computeStableTimeStep(const ProcessorGroup*,
                               const PatchSubset* patches,
                               const MaterialSubset* matls,
                               DataWarehouse* old_dw, DataWarehouse* new_dw);

    void scheduleComputeConductivity(SchedulerP& sched, const LevelP& level,
                                     const MaterialSet* fvm_matls);

    void computeConductivity(const ProcessorGroup* pg,const PatchSubset* patches,
                             const MaterialSubset* fvm_matls,DataWarehouse* old_dw,
                             DataWarehouse* new_dw);

    void scheduleComputeFCConductivity(SchedulerP& sched, const LevelP& level,
                                       const MaterialSet* es_matl);

    void computeFCConductivity(const ProcessorGroup* pg,const PatchSubset* patches,
                               const MaterialSubset* es_matls,DataWarehouse* old_dw,
                               DataWarehouse* new_dw);

    void scheduleBuildMatrixAndRhs(SchedulerP& sched, const LevelP& level,
                                   const MaterialSet* es_matls);

    void buildMatrixAndRhs(const ProcessorGroup*,
                           const PatchSubset* patches,
                           const MaterialSubset* es_matl,
                           DataWarehouse* old_dw, DataWarehouse* new_dw,
                           LevelP, Scheduler*);

    void scheduleSolve(SchedulerP& sched, const LevelP& level,
                       const MaterialSet* es_matlset);

    void scheduleUpdateESPotential(SchedulerP& sched, const LevelP& level,
                                   const MaterialSet* es_matl);

    void updateESPotential(const ProcessorGroup*,
                           const PatchSubset* patches,
                           const MaterialSubset* matls,
                           DataWarehouse* old_dw, DataWarehouse* new_dw,
                           LevelP, Scheduler*);

    void scheduleComputeCurrent(SchedulerP& sched, const LevelP& level,
                               const MaterialSet* es_matl);

    void computeCurrent(const ProcessorGroup*,
                        const PatchSubset* patches,
                        const MaterialSubset* matls,
                        DataWarehouse* old_dw, DataWarehouse* new_dw,
                        LevelP, Scheduler*);

    double d_delt;
    SolverInterface* d_solver;
    MaterialSet* d_es_matlset;
    MaterialSubset* d_es_matl;
    bool d_with_mpm;
    
    ElectrostaticSolve(const ElectrostaticSolve&);
    ElectrostaticSolve& operator=(const ElectrostaticSolve&);         
  };
}

#endif
