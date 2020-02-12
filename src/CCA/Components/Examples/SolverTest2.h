/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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


#ifndef Packages_Uintah_CCA_Components_Examples_SolverTest2_h
#define Packages_Uintah_CCA_Components_Examples_SolverTest2_h

#include <CCA/Components/Application/ApplicationCommon.h>

#include <Core/Util/RefCounted.h>
#include <Core/Util/Handle.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <CCA/Ports/SolverInterface.h>

namespace Uintah {
  class SimpleMaterial;
  class ExamplesLabel;

/**************************************

CLASS
   SolverTest2
   
   SolverTest2 simulation

GENERAL INFORMATION

   SolverTest2.h

   A Solver test for the Hypre Linear-Algebraic Interface (IJ)
  
   
KEYWORDS
   SolverTest2

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  class SolverTest2 : public ApplicationCommon {
  public:
    SolverTest2(const ProcessorGroup* myworld,
                const MaterialManagerP materialManager);
    
    virtual ~SolverTest2();

    virtual void problemSetup(const ProblemSpecP& params,
                              const ProblemSpecP& restart_prob_spec,
                              GridP& grid);
                              
    virtual void scheduleInitialize(const LevelP& level,
                                    SchedulerP& sched);
                                    
    virtual void scheduleRestartInitialize(const LevelP& level,
                                           SchedulerP& sched);
                                           
    virtual void scheduleComputeStableTimeStep(const LevelP& level,
                                               SchedulerP&);
    virtual void scheduleTimeAdvance( const LevelP& level, 
                                      SchedulerP&);
  private:
    void initialize(const ProcessorGroup*,
                    const PatchSubset* patches, const MaterialSubset* matls,
                    DataWarehouse* old_dw, DataWarehouse* new_dw);
    void computeStableTimeStep(const ProcessorGroup*,
                               const PatchSubset* patches,
                               const MaterialSubset* matls,
                               DataWarehouse* old_dw, DataWarehouse* new_dw);
    void timeAdvance(const ProcessorGroup*,
                     const PatchSubset* patches,
                     const MaterialSubset* matls,
                     DataWarehouse* old_dw, DataWarehouse* new_dw,
                     LevelP, Scheduler*);


    ExamplesLabel* lb_;
    double delt_;
    SimpleMaterial* mymat_;
    //SolverInterface* solver;
    //SolverParameters* solver_parameters;
    bool x_laplacian, y_laplacian, z_laplacian;
    
    SolverTest2(const SolverTest2&);
    SolverTest2& operator=(const SolverTest2&);
         
  };
}

#endif
