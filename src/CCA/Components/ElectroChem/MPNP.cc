/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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


#include <CCA/Components/ElectroChem/MPNP.h>
#include <CCA/Components/FVM/FVMBoundCond.h>
#include <CCA/Ports/LoadBalancerPort.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/Variables/Stencil7.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Grid/Variables/SoleVariable.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Ghost.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Geometry/Vector.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Malloc/Allocator.h>

#include <iostream>

using namespace Uintah;

MPNP::MPNP(const ProcessorGroup* myworld)
  : UintahParallelComponent(myworld)
{
  d_lb = scinew FVMLabel();

  d_solver_parameters = 0;
  d_delt = 0;
  d_unit_charge = 0;
  d_permittivity = 1.0;
  d_solver = 0;
  d_shared_state = 0;

  d_one_matl_subset  = scinew MaterialSubset();
  d_one_matl_subset->add(0);
  d_one_matl_subset->addReference();

  d_one_matl_set  = scinew MaterialSet();
  d_one_matl_set->add(0);
  d_one_matl_set->addReference();

  std::cout << "************MPNP Constructor*********" << std::endl;
}
//__________________________________
//
MPNP::~MPNP()
{
  delete d_lb;
  delete d_solver_parameters;

  if (d_one_matl_subset && d_one_matl_subset->removeReference()){
    delete d_one_matl_subset;
  }

  if (d_one_matl_set && d_one_matl_set->removeReference()){
    delete d_one_matl_set;
  }
}
//__________________________________
//
void MPNP::problemSetup(const ProblemSpecP& prob_spec,
                              const ProblemSpecP& restart_prob_spec,
                              GridP& grid,
                              SimulationStateP& shared_state)
{
  std::cout << "************MPNP Start Problem Setup*********" << std::endl;
  d_shared_state = shared_state;
  
  d_solver = dynamic_cast<SolverInterface*>(getPort("solver"));
  if(!d_solver) {
    throw InternalError("ST1:couldn't get solver port", __FILE__, __LINE__);
  }

  ProblemSpecP root_ps = 0;
  if (restart_prob_spec){
    root_ps = restart_prob_spec;
  } else{
    root_ps = prob_spec;
  }

  ProblemSpecP fvm_ps = prob_spec->findBlock("FVM");

  d_solver_parameters = d_solver->readParameters(fvm_ps, "electrostatic_solver",
                                                 d_shared_state);
  d_solver_parameters->setSolveOnExtraCells(false);

  fvm_ps->require("delt", d_delt);
  fvm_ps->require("unit_charge", d_unit_charge);
  fvm_ps->require("permittivity", d_permittivity);

  ProblemSpecP mat_ps = root_ps->findBlockWithOutAttribute("MaterialProperties");
  ProblemSpecP fvm_mat_ps = mat_ps->findBlock("FVM");

  for ( ProblemSpecP ps = fvm_mat_ps->findBlock("material"); ps != nullptr;
                     ps = ps->findNextBlock("material") ) {
    FVMMaterial *mat = scinew FVMMaterial(ps, d_shared_state, FVMMaterial::PNP);
    d_shared_state->registerFVMMaterial(mat);
  }
  std::cout << "************MPNP End Problem Setup***********" << std::endl;
}

void
MPNP::outputProblemSpec(ProblemSpecP& ps)
{

}

//__________________________________
// 
void
MPNP::scheduleInitialize( const LevelP&     level,
                                SchedulerP& sched )
{
  const MaterialSet* fvm_matls = d_shared_state->allFVMMaterials();

    Task* t = scinew Task("MPNP::initialize", this,
                          &MPNP::initialize);
    t->computes(d_lb->ccRelativePermittivity);
    t->computes(d_lb->ccPosCharge);
    t->computes(d_lb->ccNegCharge);
    t->computes(d_lb->ccMatId,         d_one_matl_subset, Task::OutOfDomain);
    t->computes(d_lb->ccInterfaceCell, d_one_matl_subset, Task::OutOfDomain);

    sched->addTask(t, level->eachPatch(), fvm_matls);

    d_solver->scheduleInitialize(level,sched, fvm_matls);
}
//__________________________________
//
void MPNP::initialize(const ProcessorGroup*,
                      const PatchSubset* patches,
                      const MaterialSubset* matls,
                            DataWarehouse* old_dw,
                            DataWarehouse* new_dw)
{
  FVMBoundCond bc;
  int num_matls = d_shared_state->getNumFVMMatls();

  for (int p = 0; p < patches->size(); p++){
    const Patch* patch = patches->get(p);
    CCVariable<int> mat_id;
    CCVariable<int> interface_cell;
    new_dw->allocateAndPut(mat_id,         d_lb->ccMatId,         0, patch);
    new_dw->allocateAndPut(interface_cell, d_lb->ccInterfaceCell, 0, patch);
    mat_id.initialize(-1);
    interface_cell.initialize(0);

    for(int m = 0; m < num_matls; m++){
      FVMMaterial* fvm_matl = d_shared_state->getFVMMaterial(m);
      int idx = fvm_matl->getDWIndex();

      CCVariable<double> rel_permittivity;
      CCVariable<double> pos_charge;
      CCVariable<double> neg_charge;

      new_dw->allocateAndPut(rel_permittivity, d_lb->ccRelativePermittivity, idx, patch);
      new_dw->allocateAndPut(pos_charge,       d_lb->ccPosCharge,            idx, patch);
      new_dw->allocateAndPut(neg_charge,       d_lb->ccNegCharge,            idx, patch);

      fvm_matl->initializeMPNPValues(idx, patch, rel_permittivity, pos_charge,
                                     neg_charge, mat_id, interface_cell);

      //bc.setConductivityBC(patch, idx, conductivity);

    }
  }
}
//__________________________________
//
void MPNP::scheduleRestartInitialize(const LevelP&     level,
                                           SchedulerP& sched)
{
}
//__________________________________
// 
void MPNP::scheduleComputeStableTimestep(const LevelP& level,
                                               SchedulerP& sched)
{
  Task* task = scinew Task("MPNP::computeStableTimestep",this,
                           &MPNP::computeStableTimestep);
  task->computes(d_shared_state->get_delt_label(),level.get_rep());
  sched->addTask(task, level->eachPatch(), d_shared_state->allFVMMaterials());
}
//__________________________________
//
void MPNP::computeStableTimestep(const ProcessorGroup* pg,
                                 const PatchSubset* patches,
                                 const MaterialSubset* matls,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw)
{
  new_dw->put(delt_vartype(d_delt), d_shared_state->get_delt_label(),getLevel(patches));
}
//__________________________________
//
void
MPNP::scheduleTimeAdvance( const LevelP& level, SchedulerP& sched)
{

}
