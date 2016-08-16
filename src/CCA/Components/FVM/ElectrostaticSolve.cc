/*
 * The MIT License
 *
 * Copyright (c) 1997-2016 The University of Utah
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


#include <CCA/Components/FVM/ElectrostaticSolve.h>
#include <CCA/Components/FVM/FVMBoundCond.h>
#include <CCA/Ports/LoadBalancer.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/Variables/Stencil7.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/SoleVariable.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Geometry/Vector.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Malloc/Allocator.h>

#include <iostream>

using namespace Uintah;

ElectrostaticSolve::ElectrostaticSolve(const ProcessorGroup* myworld)
  : UintahParallelComponent(myworld)
{
  d_lb = scinew FVMLabel();

  d_solver_parameters = 0;
  d_delt = 0;
  d_solver = 0;
  d_fvmmat = 0;
  d_shared_state = 0;
}
//__________________________________
//
ElectrostaticSolve::~ElectrostaticSolve()
{
  delete d_lb;
  delete d_solver_parameters;
}
//__________________________________
//
void ElectrostaticSolve::problemSetup(const ProblemSpecP& prob_spec,
                                      const ProblemSpecP& restart_prob_spec,
                                      GridP& grid,
                                      SimulationStateP& shared_state)
{

  d_shared_state = shared_state;
  d_solver = dynamic_cast<SolverInterface*>(getPort("solver"));
  if(!d_solver) {
    throw InternalError("ST1:couldn't get solver port", __FILE__, __LINE__);
  }
  
  ProblemSpecP st_ps = prob_spec->findBlock("FVM");
  d_solver_parameters = d_solver->readParameters(st_ps, "electrostatic_solver",
                                             d_shared_state);
  d_solver_parameters->setSolveOnExtraCells(false);
    
  st_ps->require("delt", d_delt);

  d_fvmmat = scinew FVMMaterial();
  d_shared_state->registerFVMMaterial(d_fvmmat);
}
//__________________________________
// 
void ElectrostaticSolve::scheduleInitialize(const LevelP& level,
                               SchedulerP& sched)
{
  d_solver->scheduleInitialize(level,sched, d_shared_state->allMaterials());
}
//__________________________________
//
void ElectrostaticSolve::scheduleRestartInitialize(const LevelP& level,
                                            SchedulerP& sched)
{
}
//__________________________________
// 
void ElectrostaticSolve::scheduleComputeStableTimestep(const LevelP& level,
                                          SchedulerP& sched)
{
  Task* task = scinew Task("computeStableTimestep",this, 
                           &ElectrostaticSolve::computeStableTimestep);
  task->computes(sharedState_->get_delt_label(),level.get_rep());
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());
}
//__________________________________
//
void
ElectrostaticSolve::scheduleTimeAdvance( const LevelP& level, SchedulerP& sched)
{
  Task* task = scinew Task("buildMatrixAndRhs",
                           this, &ElectrostaticSolve::buildMatrixAndRhs,
                           level, sched.get_rep());
  task->computes(d_lb->ccESPotentialMatrix);
  task->computes(d_lb->ccRHS_ESPotential);

  sched->addTask(task, level->eachPatch(), d_shared_state->allMaterials());

  d_solver->scheduleSolve(level, sched, d_shared_state->allMaterials(),
                        d_lb->ccESPotentialMatrix, Task::NewDW, d_lb->ccESPotential,
                        false, d_lb->ccRHS_ESPotential, Task::NewDW, 0, Task::OldDW,
                        d_solver_parameters,false);

}
//__________________________________
//
void ElectrostaticSolve::computeStableTimestep(const ProcessorGroup*,
                                  const PatchSubset* pss,
                                  const MaterialSubset*,
                                  DataWarehouse*, DataWarehouse* new_dw)
{
  new_dw->put(delt_vartype(delt_), sharedState_->get_delt_label(),getLevel(pss));
}
//__________________________________
//
void ElectrostaticSolve::initialize(const ProcessorGroup*,
                       const PatchSubset* patches,
                       const MaterialSubset* matls,
                       DataWarehouse*, DataWarehouse* new_dw)
{
}
//______________________________________________________________________
//
void ElectrostaticSolve::buildMatrixAndRhs(const ProcessorGroup* pg,
                           const PatchSubset* patches,
                           const MaterialSubset* matls,
                           DataWarehouse* old_dw, DataWarehouse* new_dw,
                           LevelP level, Scheduler* sched)
{
  FVMBoundCond bc;

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    Vector dx = patch->dCell();

    double a_n = dx.x() * dx.z(); double a_s = dx.x() * dx.z();
    double a_e = dx.y() * dx.z(); double a_w = dx.y() * dx.z();
    double a_t = dx.x() * dx.y(); double a_b = dx.x() * dx.y();
    // double vol = dx.x() * dx.y() * dx.z();

    double n = a_n / dx.y(); double s = a_s / dx.y();
    double e = a_e / dx.x(); double w = a_w / dx.x();
    double t = a_t / dx.z(); double b = a_b / dx.z();
    double center = n + s + e + w + t + b;

    IntVector low_idx  = patch->getCellLowIndex();
    IntVector high_idx = patch->getCellHighIndex();

    std::cout << "low_idx: " << low_idx << std::endl;
    std::cout << "high_idx: " << high_idx << std::endl;


    IntVector low_offset = IntVector(patch->getBCType(Patch::xminus) == Patch::Neighbor?0:1,
                                     patch->getBCType(Patch::yminus) == Patch::Neighbor?0:1,
                                     patch->getBCType(Patch::zminus) == Patch::Neighbor?0:1);


    IntVector high_offset = IntVector(patch->getBCType(Patch::xplus)  == Patch::Neighbor?0:1,
                                      patch->getBCType(Patch::yplus)  == Patch::Neighbor?0:1,
                                      patch->getBCType(Patch::zplus)  == Patch::Neighbor?0:1);

    //IntVector low_interior  = low_idx + low_offset;
    //IntVector high_interior = high_idx - high_offset;
    std::cout << "low_offset: " << low_offset << std::endl;
    std::cout << "high_offset: " << high_offset << std::endl;
    std::cout << "material size: " << matls->size() << std::endl;

    CCVariable<Stencil7> A;
    CCVariable<double> rhs;

    for(int m = 0;m<matls->size();m++){
      int matl = matls->get(m);
      new_dw->allocateAndPut(A,   d_lb->ccESPotentialMatrix, matl, patch);
      new_dw->allocateAndPut(rhs, d_lb->ccRHS_ESPotential,   matl, patch);
      // iterate over cells;
      for(CellIterator iter(low_idx, high_idx); !iter.done(); iter++){
        IntVector c = *iter;
        Stencil7&  A_tmp=A[c];

        A_tmp.p = -center;
        A_tmp.n = n;   A_tmp.s = s;
        A_tmp.e = e;   A_tmp.w = w;
        A_tmp.t = t;   A_tmp.b = b;
        rhs[c] = 0;

      } // End CellIterator
      bc.setESBoundaryConditions(patch, matl, A, rhs);
    } // End Materials
  } // End patches
}
