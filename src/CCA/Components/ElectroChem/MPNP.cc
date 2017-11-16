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
#include <Core/Util/DebugStream.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Malloc/Allocator.h>

#include <iostream>

using namespace Uintah;

static DebugStream debug1("MPNP", false);

MPNP::MPNP(const ProcessorGroup* myworld)
  : UintahParallelComponent(myworld)
{
  d_lb = scinew FVMLabel();

  d_solver_parameters = 0;
  d_delt = 0;
  d_unit_charge = 0;
  d_permittivity = 1.0;
  d_boltzmanns = 1.38e-23;
  d_temp = 300;
  d_solver = 0;
  d_shared_state = 0;

  d_alpha = 0.0;
  d_beta  = 0.0;
  d_gamma = 0.0;

  d_one_matl_subset  = scinew MaterialSubset();
  d_one_matl_subset->add(0);
  d_one_matl_subset->addReference();

  d_one_matl_set  = scinew MaterialSet();
  d_one_matl_set->add(0);
  d_one_matl_set->addReference();

  debug1 << "************MPNP Constructor*********" << std::endl;
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
  debug1 << "************MPNP Start Problem Setup*********" << std::endl;
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
  fvm_ps->require("boltzmanns_const", d_boltzmanns);
  fvm_ps->require("temperature", d_temp);

  // Still need to add code to throw error message
  if(d_permittivity > 0){
    d_alpha = d_unit_charge/d_permittivity;
  }

  if(d_temp > 0 && d_boltzmanns > 0){
    d_beta = d_unit_charge/(d_boltzmanns * d_temp);
  }

  std::cout << "Beta: " << d_beta << std::endl;

  ProblemSpecP mat_ps = root_ps->findBlockWithOutAttribute("MaterialProperties");
  ProblemSpecP fvm_mat_ps = mat_ps->findBlock("FVM");

  for ( ProblemSpecP ps = fvm_mat_ps->findBlock("material"); ps != nullptr;
                     ps = ps->findNextBlock("material") ) {
    FVMMaterial *mat = scinew FVMMaterial(ps, d_shared_state, FVMMaterial::PNP);
    d_shared_state->registerFVMMaterial(mat);
  }
  debug1 << "************MPNP End Problem Setup***********" << std::endl;
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
    t->computes(d_lb->ccBoundaryCell,  d_one_matl_subset, Task::OutOfDomain);

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
  debug1 << "************MPNP Start Initialize***********" << std::endl;
  FVMBoundCond bc;
  int num_matls = d_shared_state->getNumFVMMatls();

  for (int p = 0; p < patches->size(); p++){
    const Patch* patch = patches->get(p);
    CCVariable<int> mat_id;
    CCVariable<int> interface_cell;
    CCVariable<int> boundary_cell;
    new_dw->allocateAndPut(mat_id,         d_lb->ccMatId,         0, patch);
    new_dw->allocateAndPut(interface_cell, d_lb->ccInterfaceCell, 0, patch);
    new_dw->allocateAndPut(boundary_cell,  d_lb->ccBoundaryCell,  0, patch);
    mat_id.initialize(-1);
    interface_cell.initialize(0);
    boundary_cell.initialize(0);

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
                                     neg_charge, mat_id, interface_cell, boundary_cell);

      //bc.setConductivityBC(patch, idx, conductivity);

    }
  }
  debug1 << "************MPNP End Initialize*************" << std::endl;
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
  debug1 << "************MPNP Start TimeAdvance***********" << std::endl;
  const MaterialSet* fvm_matls = d_shared_state->allFVMMaterials();

  scheduleComputeMPNPValues(   sched, level, fvm_matls);
  scheduleComputeFCPermittivity( sched, level, d_one_matl_set);

  scheduleBuildMatrixAndRhs(     sched, level, d_one_matl_set);

  d_solver->scheduleSolve(level, sched, d_one_matl_set,
                          d_lb->ccESPotentialMatrix, Task::NewDW,
                          d_lb->ccESPotential, false,
                          d_lb->ccRHS_ESPotential, Task::NewDW,
                          0, Task::OldDW,
                          d_solver_parameters,false);

  /**
  scheduleUpdateESPotential(sched, level, d_es_matlset);
  scheduleComputeCurrent(sched, level, d_es_matlset);
  */
  scheduleComputeDcDt(     sched, level, fvm_matls);
  scheduleUpdateMPNPValues(sched, level, fvm_matls);
  debug1 << "************MPNP End TimeAdvance*************" << std::endl;
}
//______________________________________________________________________
//
void MPNP::scheduleComputeMPNPValues(SchedulerP& sched,
                                       const LevelP& level,
                                       const MaterialSet* fvm_matls)
{
  Task* t = scinew Task("MPNP::computeMPNPValues", this,
                        &MPNP::computeMPNPValues);

  t->requires(Task::OldDW, d_lb->ccRelativePermittivity, d_gac, 1);
  t->requires(Task::OldDW, d_lb->ccPosCharge,            d_gac, 1);
  t->requires(Task::OldDW, d_lb->ccNegCharge,            d_gac, 1);

  t->computes(d_lb->ccGridPermittivity, d_one_matl_subset, Task::OutOfDomain);
  t->computes(d_lb->ccGridTotalCharge,  d_one_matl_subset, Task::OutOfDomain);
  sched->addTask(t, level->eachPatch(), fvm_matls);
}
//______________________________________________________________________
//
void MPNP::computeMPNPValues(const ProcessorGroup* pg,
                             const PatchSubset* patches,
                             const MaterialSubset* fvm_matls,
                                   DataWarehouse* old_dw,
                                   DataWarehouse* new_dw)
{
  debug1 << "**********MPNP Start ComputeMPNPValues***********" << std::endl;
  int num_matls = d_shared_state->getNumFVMMatls();
  for (int p = 0; p < patches->size(); p++){
    const Patch* patch = patches->get(p);

    constCCVariable<int> old_mat_id;
    constCCVariable<int> old_interface_cell;

    CCVariable<double> grid_permittivity;
    CCVariable<double> grid_total_charge;

    new_dw->allocateAndPut(grid_permittivity, d_lb->ccGridPermittivity, 0, patch);
    new_dw->allocateAndPut(grid_total_charge, d_lb->ccGridTotalCharge,  0, patch);

    grid_permittivity.initialize(0.0);
    grid_total_charge.initialize(0.0);

    for(int m = 0; m < num_matls; m++){
      FVMMaterial* fvm_matl = d_shared_state->getFVMMaterial(m);
      int idx = fvm_matl->getDWIndex();

      constCCVariable<double> permittivity;
      constCCVariable<double> pos_charge;
      constCCVariable<double> neg_charge;

      old_dw->get(permittivity, d_lb->ccRelativePermittivity, idx, patch, d_gac, 1);
      old_dw->get(pos_charge,   d_lb->ccPosCharge,            idx, patch, d_gac, 1);
      old_dw->get(neg_charge,   d_lb->ccNegCharge,            idx, patch, d_gac, 1);

      for(CellIterator iter = patch->getExtraCellIterator();!iter.done();iter++){
        IntVector c = *iter;
        if(permittivity[c] > 0.0){
          grid_permittivity[c] = permittivity[c];
        }

        grid_total_charge[c] = d_alpha * (pos_charge[c] - neg_charge[c]);
      }
    } // material loop
  } // patch loop
  debug1 << "**********MPNP End ComputeMPNPValues*************" << std::endl;
}
//______________________________________________________________________
//
void MPNP::scheduleComputeFCPermittivity(SchedulerP& sched,
                                         const LevelP& level,
                                         const MaterialSet* es_matls)
{
  Task* t = scinew Task("MPNP::computeFCPermittivity", this,
                        &MPNP::computeFCPermittivity);

    t->requires(Task::NewDW, d_lb->ccGridPermittivity, d_gac, 1);
    t->computes(d_lb->fcxRelativePermittivity, d_one_matl_subset, Task::OutOfDomain);
    t->computes(d_lb->fcyRelativePermittivity, d_one_matl_subset, Task::OutOfDomain);
    t->computes(d_lb->fczRelativePermittivity, d_one_matl_subset, Task::OutOfDomain);
    sched->addTask(t, level->eachPatch(), es_matls);
}

void MPNP::computeFCPermittivity(const ProcessorGroup* pg,
                                 const PatchSubset* patches,
                                 const MaterialSubset* es_matls,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw)
{
  debug1 << "********MPNP Start ComputeFCPermittivity**********" << std::endl;
  for (int p = 0; p < patches->size(); p++){
    const Patch* patch = patches->get(p);

    constCCVariable<double> grid_permittivity;

    SFCXVariable<double> fcx_permittivity;
    SFCYVariable<double> fcy_permittivity;
    SFCZVariable<double> fcz_permittivity;


    new_dw->get(grid_permittivity, d_lb->ccGridPermittivity, 0, patch, d_gac, 1);
    new_dw->allocateAndPut(fcx_permittivity, d_lb->fcxRelativePermittivity, 0, patch);
    new_dw->allocateAndPut(fcy_permittivity, d_lb->fcyRelativePermittivity, 0, patch);
    new_dw->allocateAndPut(fcz_permittivity, d_lb->fczRelativePermittivity, 0, patch);

    fcx_permittivity.initialize(0.0);
    fcy_permittivity.initialize(0.0);
    fcz_permittivity.initialize(0.0);

    for(CellIterator iter = patch->getSFCXIterator(); !iter.done(); iter++){
      IntVector c = *iter;
      IntVector offset(-1,0,0);
      fcx_permittivity[c] = .5*(grid_permittivity[c] + grid_permittivity[c + offset]);
    }

    for(CellIterator iter = patch->getSFCYIterator(); !iter.done(); iter++){
      IntVector c = *iter;
      IntVector offset(0,-1,0);
      fcy_permittivity[c] = .5*(grid_permittivity[c] + grid_permittivity[c + offset]);
    }

    for(CellIterator iter = patch->getSFCZIterator(); !iter.done(); iter++){
      IntVector c = *iter;
      IntVector offset(0,0,-1);
      fcz_permittivity[c] = .5*(grid_permittivity[c] + grid_permittivity[c + offset]);
    }
  } // patch loop
  debug1 << "********MPNP End ComputeFCPermittivity************" << std::endl;
}
//______________________________________________________________________
//
void MPNP::scheduleBuildMatrixAndRhs(SchedulerP& sched,
                                     const LevelP& level,
                                     const MaterialSet* es_matl)
{
  Task* task = scinew Task("MPNP::buildMatrixAndRhs", this,
                           &MPNP::buildMatrixAndRhs,
                           level, sched.get_rep());

  task->requires(Task::NewDW, d_lb->fcxRelativePermittivity, d_gac, 1);
  task->requires(Task::NewDW, d_lb->fcyRelativePermittivity, d_gac, 1);
  task->requires(Task::NewDW, d_lb->fczRelativePermittivity, d_gac, 1);
  task->requires(Task::NewDW, d_lb->ccGridTotalCharge,       d_gac, 1);

  task->computes(d_lb->ccESPotentialMatrix, d_one_matl_subset, Task::OutOfDomain);
  task->computes(d_lb->ccRHS_ESPotential,   d_one_matl_subset, Task::OutOfDomain);

  sched->addTask(task, level->eachPatch(), es_matl);
}
//______________________________________________________________________
//
void MPNP::buildMatrixAndRhs(const ProcessorGroup* pg,
                             const PatchSubset* patches,
                             const MaterialSubset* ,
                                   DataWarehouse* old_dw,
                                   DataWarehouse* new_dw,
                                   LevelP level,
                                   Scheduler* sched)
{
  debug1 << "********MPNP Start BuildMatrixAndRHS************" << std::endl;
  FVMBoundCond bc;
  IntVector xoffset(1,0,0);
  IntVector yoffset(0,1,0);
  IntVector zoffset(0,0,1);

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

    constSFCXVariable<double> fcx_permittivity;
    constSFCYVariable<double> fcy_permittivity;
    constSFCZVariable<double> fcz_permittivity;
    constCCVariable<double>   total_charge;

    new_dw->get(fcx_permittivity, d_lb->fcxRelativePermittivity,   0, patch, d_gac, 1);
    new_dw->get(fcy_permittivity, d_lb->fcyRelativePermittivity,   0, patch, d_gac, 1);
    new_dw->get(fcz_permittivity, d_lb->fczRelativePermittivity,   0, patch, d_gac, 1);
    new_dw->get(total_charge,     d_lb->ccGridTotalCharge, 0, patch, d_gac, 1);

    CCVariable<Stencil7> A;
    CCVariable<double> rhs;
    new_dw->allocateAndPut(A,   d_lb->ccESPotentialMatrix, 0, patch);
    new_dw->allocateAndPut(rhs, d_lb->ccRHS_ESPotential,   0, patch);

    //__________________________________
    //  Initialize A
    for(CellIterator iter(patch->getExtraCellIterator()); !iter.done(); iter++){
      IntVector c = *iter;
      Stencil7&  A_tmp=A[c];
      A_tmp.p = 0.0;
      A_tmp.n = 0.0;   A_tmp.s = 0.0;
      A_tmp.e = 0.0;   A_tmp.w = 0.0;
      A_tmp.t = 0.0;   A_tmp.b = 0.0;
      rhs[c] = 0;
    }

    // iterate over cells;
    for(CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
      IntVector c = *iter;
      Stencil7&  A_tmp=A[c];

      double efc = e*fcx_permittivity[c + xoffset];
      double wfc = w*fcx_permittivity[c];
      double nfc = n*fcy_permittivity[c + yoffset];
      double sfc = s*fcy_permittivity[c];
      double tfc = t*fcz_permittivity[c + zoffset];
      double bfc = b*fcz_permittivity[c];
      double center = efc + wfc + nfc + sfc + tfc + bfc;

      A_tmp.p = -center;
      A_tmp.n = nfc;   A_tmp.s = sfc;
      A_tmp.e = efc;   A_tmp.w = wfc;
      A_tmp.t = tfc;   A_tmp.b = bfc;

      rhs[c] = total_charge[c];
    } // End CellIterator

    bc.setESBoundaryConditions(patch, 0, A, rhs,
                               fcx_permittivity, fcy_permittivity, fcz_permittivity);

  } // End patches
  debug1 << "********MPNP End BuildMatrixAndRHS**************" << std::endl;
}
//______________________________________________________________________
//
void MPNP::scheduleComputeDcDt(SchedulerP&        sched,
                               const LevelP&      level,
                               const MaterialSet* fvm_matls)
{
  Task* t = scinew Task("MPNP::computeDcDt", this, &MPNP::computeDcDt);

  t->requires(Task::OldDW, d_lb->ccPosCharge, d_gac, 1);
  t->requires(Task::OldDW, d_lb->ccNegCharge, d_gac, 1);

  t->requires(Task::OldDW, d_lb->ccMatId,         d_one_matl_subset, d_gac, 1);
  t->requires(Task::OldDW, d_lb->ccInterfaceCell, d_one_matl_subset, d_gac, 1);
  t->requires(Task::NewDW, d_lb->ccESPotential,   d_one_matl_subset, d_gac, 1);

  t->computes(d_lb->ccDPosChargeDt);
  t->computes(d_lb->ccDNegChargeDt);

  sched->addTask(t, level->eachPatch(), fvm_matls);
}
//______________________________________________________________________
//
void MPNP::computeDcDt(const ProcessorGroup* pg,
                       const PatchSubset* patches,
                       const MaterialSubset* fvm_matls,
                             DataWarehouse* old_dw,
                             DataWarehouse* new_dw)
{
  debug1 << "********MPNP Start ComputesDcDt**************" << std::endl;
  int num_matls = d_shared_state->getNumFVMMatls();
  IntVector xoff = IntVector(1,0,0);
  IntVector yoff = IntVector(0,1,0);
  IntVector zoff = IntVector(0,0,1);

  double pflux[6];
  double nflux[6];
  double es_p;
  double es_v[6];
  double p_p;
  double p_v[6];
  double n_p;
  double n_v[6];

  double diff;

  for (int p = 0; p < patches->size(); p++){
    const Patch* patch = patches->get(p);
    Vector dx = patch->dCell();
    IntVector inner_low;
    IntVector inner_high;

    getInnerRange(patch, inner_low, inner_high);
    std::cout << "low: " << patch->getCellLowIndex() << ", high: " << patch->getCellHighIndex() << std::endl;
    std::cout << "Low: " << inner_low << ", High: " << inner_high << std::endl;

    double a_n = dx.x() * dx.z();
    double a_e = dx.y() * dx.z();
    double a_t = dx.x() * dx.y();

    constCCVariable<double> es_potential;
    constCCVariable<int>    mat_id;
    constCCVariable<int>    cell_interface;
    new_dw->get(es_potential,   d_lb->ccESPotential,   0, patch, d_gac, 1);
    old_dw->get(mat_id,         d_lb->ccMatId,         0, patch, d_gac, 1);
    old_dw->get(cell_interface, d_lb->ccInterfaceCell, 0, patch, d_gac, 1);

    for(int m = 0; m < num_matls; m++){
      FVMMaterial* fvm_matl = d_shared_state->getFVMMaterial(m);
      int idx = fvm_matl->getDWIndex();
      diff = fvm_matl->getDiffusivity();

      constCCVariable<double> pos_charge;
      constCCVariable<double> neg_charge;

      CCVariable<double> dpdt;
      CCVariable<double> dndt;

      old_dw->get(pos_charge, d_lb->ccPosCharge, idx, patch, d_gac, 1);
      old_dw->get(neg_charge, d_lb->ccNegCharge, idx, patch, d_gac, 1);

      new_dw->allocateAndPut(dpdt, d_lb->ccDPosChargeDt, idx, patch);
      new_dw->allocateAndPut(dndt, d_lb->ccDNegChargeDt, idx, patch);

      dpdt.initialize(0.0);
      dndt.initialize(0.0);
      for(CellIterator iter(inner_low, inner_high); !iter.done(); iter++){
        IntVector c = *iter;
        if(mat_id[c] == idx){
          es_p = es_potential[c];
          es_v[0] = es_potential[c+xoff]; es_v[1] = es_potential[c-xoff];
          es_v[2] = es_potential[c+yoff]; es_v[3] = es_potential[c-yoff];
          es_v[4] = es_potential[c+zoff]; es_v[5] = es_potential[c-zoff];

          p_p = pos_charge[c];
          p_v[0] = pos_charge[c+xoff]; p_v[1] = pos_charge[c-xoff];
          p_v[2] = pos_charge[c+yoff]; p_v[3] = pos_charge[c-yoff];
          p_v[4] = pos_charge[c+zoff]; p_v[5] = pos_charge[c-zoff];

          n_p = neg_charge[c];
          n_v[0] = neg_charge[c+xoff]; n_v[1] = neg_charge[c-xoff];
          n_v[2] = neg_charge[c+yoff]; n_v[3] = neg_charge[c-yoff];
          n_v[4] = neg_charge[c+zoff]; n_v[5] = neg_charge[c-zoff];

          for(int i = 0; i < 6; i++){
            if((cell_interface[c] & (1<<i)) == (1<<i)){
              pflux[i] = 0;
              nflux[i] = 0;
            }else{
              pflux[i] = diff*(p_v[i]-p_p) + .5*diff*d_beta*(p_v[i]+p_p)*(es_v[i]-es_p);
              nflux[i] = diff*(n_v[i]-n_p) - .5*diff*d_beta*(n_v[i]+n_p)*(es_v[i]-es_p);
              if(i == 0 || i == 1){
                pflux[i] /= dx.x();
                nflux[i] /= dx.x();
              }else if(i == 2 || i == 3){
                pflux[i] /= dx.y();
                nflux[i] /= dx.y();
              }else{
                pflux[i] /= dx.z();
                nflux[i] /= dx.z();
              }
            }
          }
          dpdt[c] = a_e*(pflux[0]+pflux[1]) + a_n*(pflux[2]+pflux[3]) + a_t*(pflux[4]+pflux[5]);
          dndt[c] = a_e*(nflux[0]+nflux[1]) + a_n*(nflux[2]+nflux[3]) + a_t*(nflux[4]+nflux[5]);
        }
      } // End Cell iterator
    }
  }
  debug1 << "********MPNP End ComputesDcDt****************" << std::endl;
}
//______________________________________________________________________
//
void MPNP::scheduleUpdateMPNPValues(SchedulerP& sched,
                                    const LevelP& level,
                                    const MaterialSet* fvm_matls)
{
  Task* t = scinew Task("MPNP::updateMPNPValues", this,
                        &MPNP::updateMPNPValues);

  t->requires(Task::OldDW, d_lb->ccRelativePermittivity, d_gac, 0);
  t->requires(Task::OldDW, d_lb->ccPosCharge,            d_gac, 0);
  t->requires(Task::OldDW, d_lb->ccNegCharge,            d_gac, 0);
  t->requires(Task::NewDW, d_lb->ccDPosChargeDt,         d_gac, 0);
  t->requires(Task::NewDW, d_lb->ccDNegChargeDt,         d_gac, 0);

  t->requires(Task::OldDW, d_lb->ccMatId,         d_one_matl_subset, d_gac, 0);
  t->requires(Task::OldDW, d_lb->ccInterfaceCell, d_one_matl_subset, d_gac, 0);

  t->computes(d_lb->ccRelativePermittivity);
  t->computes(d_lb->ccPosCharge);
  t->computes(d_lb->ccNegCharge);
  t->computes(d_lb->ccMatId,         d_one_matl_subset, Task::OutOfDomain);
  t->computes(d_lb->ccInterfaceCell, d_one_matl_subset, Task::OutOfDomain);
  sched->addTask(t, level->eachPatch(), fvm_matls);
}
//______________________________________________________________________
//
void MPNP::updateMPNPValues(const ProcessorGroup* pg,
                            const PatchSubset* patches,
                            const MaterialSubset* fvm_matls,
                                  DataWarehouse* old_dw,
                                  DataWarehouse* new_dw)
{
  debug1 << "********MPNP Start UpdateMPNPValues************" << std::endl;
  int num_matls = d_shared_state->getNumFVMMatls();
  for (int p = 0; p < patches->size(); p++){
    const Patch* patch = patches->get(p);
    Vector dx = patch->dCell();
    double vol = dx.x() * dx.y() * dx.z();

    constCCVariable<int> old_mat_id;
    constCCVariable<int> old_interface_cell;

    CCVariable<int> mat_id;
    CCVariable<int> interface_cell;

    old_dw->get(old_mat_id,         d_lb->ccMatId,         0, patch, d_gac, 0);
    old_dw->get(old_interface_cell, d_lb->ccInterfaceCell, 0, patch, d_gac, 0);

    new_dw->allocateAndPut(mat_id,         d_lb->ccMatId,         0, patch);
    new_dw->allocateAndPut(interface_cell, d_lb->ccInterfaceCell, 0, patch);

    mat_id.initialize(-1);
    interface_cell.initialize(0);

    for(int m = 0; m < num_matls; m++){
      FVMMaterial* fvm_matl = d_shared_state->getFVMMaterial(m);
      int idx = fvm_matl->getDWIndex();

      constCCVariable<double> old_permittivity;
      constCCVariable<double> old_pos_charge;
      constCCVariable<double> old_neg_charge;
      constCCVariable<double> dposdt;
      constCCVariable<double> dnegdt;

      CCVariable<double> permittivity;
      CCVariable<double> pos_charge;
      CCVariable<double> neg_charge;

      old_dw->get(old_permittivity, d_lb->ccRelativePermittivity, idx, patch, d_gac, 0);
      old_dw->get(old_pos_charge,   d_lb->ccPosCharge,    idx, patch, d_gac, 0);
      old_dw->get(old_neg_charge,   d_lb->ccNegCharge,    idx, patch, d_gac, 0);
      new_dw->get(dposdt,           d_lb->ccDPosChargeDt, idx, patch, d_gac, 0);
      new_dw->get(dnegdt,           d_lb->ccDNegChargeDt, idx, patch, d_gac, 0);

      new_dw->allocateAndPut(permittivity, d_lb->ccRelativePermittivity, idx, patch);
      new_dw->allocateAndPut(pos_charge,   d_lb->ccPosCharge,            idx, patch);
      new_dw->allocateAndPut(neg_charge,   d_lb->ccNegCharge,            idx, patch);

      for(CellIterator iter = patch->getCellIterator();!iter.done();iter++){
        IntVector c = *iter;

        permittivity[c]   = old_permittivity[c];
        pos_charge[c]     = old_pos_charge[c] + d_delt * dposdt[c]/vol;
        neg_charge[c]     = old_neg_charge[c] + d_delt * dnegdt[c]/vol;

        if(old_mat_id[c] == idx){
          mat_id[c] = old_mat_id[c];
          interface_cell[c] = old_interface_cell[c];
        }
      }
    } // material loop
  } // patch loop
  debug1 << "********MPNP End UpdateMPNPValues**************" << std::endl;
}
