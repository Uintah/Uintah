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

#include <CCA/Components/FVM/GaussSolve.h>

#include <CCA/Components/FVM/FVMBoundCond.h>
#include <CCA/Components/FVM/FVMLabel.h>
#include <CCA/Components/FVM/FVMMaterial.h>

#include <CCA/Ports/Scheduler.h>

#include <Core/Geometry/IntVector.h>
#include <Core/Geometry/Vector.h>
#include <Core/Grid/Ghost.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/Stencil7.h>
#include <Core/Grid/Task.h>

#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Parallel/ProcessorGroup.h>

using namespace Uintah;

GaussSolve::GaussSolve(const ProcessorGroup* myworld,
		       const MaterialManagerP materialManager)
  : ApplicationCommon(myworld, materialManager)
{
  d_lb = scinew FVMLabel();

  d_delt = 0;
  d_solver = 0;
  d_with_mpm = false;
  d_elem_charge = 0.0;

  d_es_matl  = scinew MaterialSubset();
  d_es_matl->add(0);
  d_es_matl->addReference();

  d_es_matlset  = scinew MaterialSet();
  d_es_matlset->add(0);
  d_es_matlset->addReference();
}
//__________________________________
//
GaussSolve::~GaussSolve()
{
  delete d_lb;

  if (d_es_matl && d_es_matl->removeReference()){
    delete d_es_matl;
  }

  if (d_es_matlset && d_es_matlset->removeReference()){
    delete d_es_matlset;
  }
}
//__________________________________
//
void GaussSolve::problemSetup(const ProblemSpecP& prob_spec,
                              const ProblemSpecP& restart_prob_spec,
                              GridP& grid)
{
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

  d_solver->readParameters(fvm_ps, "gauss1_solver");
  d_solver->getParameters()->setSolveOnExtraCells(false);
    
  fvm_ps->require("delt", d_delt);
  fvm_ps->require("unit_charge", d_elem_charge);

  ProblemSpecP mat_ps = root_ps->findBlockWithOutAttribute("MaterialProperties");
  ProblemSpecP fvm_mat_ps = mat_ps->findBlock("FVM");

  if( !d_with_mpm ){
    for ( ProblemSpecP ps = fvm_mat_ps->findBlock("material"); ps != nullptr; ps = ps->findNextBlock("material") ) {

      FVMMaterial *mat = scinew FVMMaterial( ps, m_materialManager, FVMMaterial::Gauss );
      m_materialManager->registerMaterial( "FVM",  mat );
    }
  }
}

void
GaussSolve::outputProblemSpec( ProblemSpecP& ps )
{
}

//__________________________________
// 
void
GaussSolve::scheduleInitialize( const LevelP     & level,
                                      SchedulerP & sched )
{
  const MaterialSet* fvm_matls = m_materialManager->allMaterials( "FVM" );

  Task* t = scinew Task( "GaussSolve::initialize", this, &GaussSolve::initialize );

  t->computes(d_lb->ccPosCharge);
  t->computes(d_lb->ccNegCharge);
  t->computes(d_lb->ccPermittivity);
  sched->addTask(t, level->eachPatch(), fvm_matls);

  d_solver->scheduleInitialize(level,sched, fvm_matls);
}

//__________________________________
//
void
GaussSolve::initialize( const ProcessorGroup *,
                        const PatchSubset    * patches,
                        const MaterialSubset * matls,
                              DataWarehouse  * /* old_dw */,
                              DataWarehouse  * new_dw )
{
  FVMBoundCond bc;
  int num_matls = m_materialManager->getNumMatls( "FVM" );

  for (int p = 0; p < patches->size(); p++){
    const Patch* patch = patches->get(p);
    CCVariable<double> pos_charge;
    CCVariable<double> neg_charge;
    CCVariable<double> permittivity;
    new_dw->allocateAndPut(pos_charge,   d_lb->ccPosCharge,    0, patch);
    new_dw->allocateAndPut(neg_charge,   d_lb->ccNegCharge,    0, patch);
    new_dw->allocateAndPut(permittivity, d_lb->ccPermittivity, 0, patch);

    pos_charge.initialize(0.0);
    neg_charge.initialize(0.0);
    permittivity.initialize(0.0);
    for(int m = 0; m < num_matls; m++){
      FVMMaterial* fvm_matl = (FVMMaterial* ) m_materialManager->getMaterial( "FVM", m);
      fvm_matl->initializePermittivityAndCharge(permittivity, pos_charge,
                                                neg_charge, patch);
      //bc.setConductivityBC(patch, idx, conductivity);
    }
  }
}
//__________________________________
//
void GaussSolve::scheduleRestartInitialize(const LevelP& level,
                                            SchedulerP& sched)
{
}
//__________________________________
// 
void GaussSolve::scheduleComputeStableTimeStep(const LevelP& level,
                                          SchedulerP& sched)
{
  Task* task = scinew Task("computeStableTimeStep",this, 
                           &GaussSolve::computeStableTimeStep);
  task->computes(getDelTLabel(),level.get_rep());
  sched->addTask(task, level->eachPatch(), m_materialManager->allMaterials( "FVM" ));
}
//__________________________________
//
void
GaussSolve::scheduleTimeAdvance( const LevelP& level, SchedulerP& sched)
{
  scheduleComputeCharge(     sched, level, d_es_matlset);
  scheduleBuildMatrixAndRhs( sched, level, d_es_matlset);

  d_solver->scheduleSolve(level, sched, d_es_matlset,
                          d_lb->ccESPotentialMatrix, Task::NewDW,
                          d_lb->ccESPotential, false,
                          d_lb->ccRHS_ESPotential, Task::NewDW,
                          0, Task::OldDW,false);

  scheduleUpdateESPotential( sched, level, d_es_matlset);
}
//__________________________________
//

void
GaussSolve::computeStableTimeStep( const ProcessorGroup *,
                                   const PatchSubset    * pss,
                                   const MaterialSubset *,
                                         DataWarehouse  *,
                                         DataWarehouse  * new_dw )
{
  new_dw->put(delt_vartype(d_delt), getDelTLabel(),getLevel(pss));
}


//______________________________________________________________________
//
void
GaussSolve::scheduleBuildMatrixAndRhs(       SchedulerP  & sched,
                                       const LevelP      & level,
                                       const MaterialSet * es_matl )
{
  Task* task = scinew Task("GaussSolve::buildMatrixAndRhs", this,
                           &GaussSolve::buildMatrixAndRhs,
                           level, sched.get_rep());

  task->requires(Task::NewDW, d_lb->ccPosCharge,    Ghost::AroundCells, 1);
  task->requires(Task::NewDW, d_lb->ccNegCharge,    Ghost::AroundCells, 1);
  task->requires(Task::NewDW, d_lb->ccPermittivity, Ghost::AroundCells, 1);
  task->computes(d_lb->ccESPotentialMatrix, d_es_matl, Task::OutOfDomain);
  task->computes(d_lb->ccRHS_ESPotential,   d_es_matl, Task::OutOfDomain);
  task->computes(d_lb->ccTotalCharge,       d_es_matl, Task::OutOfDomain);

  sched->addTask(task, level->eachPatch(), es_matl);
}
//______________________________________________________________________
//

void
GaussSolve::buildMatrixAndRhs( const ProcessorGroup * pg,
                               const PatchSubset    * patches,
                               const MaterialSubset *,
                                     DataWarehouse  * old_dw,
                                     DataWarehouse  * new_dw,
                                     LevelP           level,
                                     Scheduler      * sched )
{
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

    constCCVariable<double> cc_pos_charge;
    constCCVariable<double> cc_neg_charge;
    constCCVariable<double> cc_permit;

    new_dw->get(cc_pos_charge, d_lb->ccPosCharge,    0, patch, Ghost::AroundCells, 1);
    new_dw->get(cc_neg_charge, d_lb->ccNegCharge,    0, patch, Ghost::AroundCells, 1);
    new_dw->get(cc_permit,     d_lb->ccPermittivity, 0, patch, Ghost::AroundCells, 1);

    CCVariable<Stencil7> A;
    CCVariable<double> rhs;
    CCVariable<double> total_charge;
    new_dw->allocateAndPut(A, d_lb->ccESPotentialMatrix, 0, patch);
    new_dw->allocateAndPut(rhs, d_lb->ccRHS_ESPotential, 0, patch);
    new_dw->allocateAndPut(total_charge, d_lb->ccTotalCharge, 0, patch);

    total_charge.initialize(0.0);
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
      double center = e + w + n + s + t + b;

      A_tmp.p = -center;
      A_tmp.n = n;   A_tmp.s = s;
      A_tmp.e = e;   A_tmp.w = w;
      A_tmp.t = t;   A_tmp.b = b;

      total_charge[c] = cc_pos_charge[c] - cc_neg_charge[c];
      rhs[c] = d_elem_charge * total_charge[c]/cc_permit[c];
    } // End CellIterator

    bc.setG1BoundaryConditions(patch, 0, A, rhs);

  } // End patches
}

void GaussSolve::scheduleSolve(SchedulerP& sched,
                                       const LevelP& level,
                                       const MaterialSet* es_matlset)
{
  d_solver->scheduleSolve(level, sched, d_es_matlset,
                          d_lb->ccESPotentialMatrix, Task::NewDW,
                          d_lb->ccESPotential, false,
                          d_lb->ccRHS_ESPotential, Task::NewDW,
                          d_lb->ccESPotential, Task::OldDW,false);
}

void GaussSolve::scheduleComputeCharge(SchedulerP& sched,
                                       const LevelP& level,
                                       const MaterialSet* fvm_matls)
{
  Task* t = scinew Task("GaussSolve::computeCharge", this,
                        &GaussSolve::computeCharge);

  t->requires(Task::OldDW, d_lb->ccPosCharge,    Ghost::AroundCells, 1);
  t->requires(Task::OldDW, d_lb->ccNegCharge,    Ghost::AroundCells, 1);
  t->requires(Task::OldDW, d_lb->ccPermittivity, Ghost::AroundCells, 1);
  t->computes(d_lb->ccPosCharge);
  t->computes(d_lb->ccNegCharge);
  t->computes(d_lb->ccPermittivity);

  sched->addTask(t, level->eachPatch(), fvm_matls);
}

//______________________________________________________________________
//
void GaussSolve::computeCharge(const ProcessorGroup* pg,
                               const PatchSubset* patches,
                               const MaterialSubset* fvm_matls,
                               DataWarehouse* old_dw,
                               DataWarehouse* new_dw)
{
  for (int p = 0; p < patches->size(); p++){
    const Patch* patch = patches->get(p);

    constCCVariable<double> old_pos_charge;
    constCCVariable<double> old_neg_charge;
    constCCVariable<double> old_permittivity;

    CCVariable<double> pos_charge;
    CCVariable<double> neg_charge;
    CCVariable<double> permittivity;

    old_dw->get(old_pos_charge,   d_lb->ccPosCharge,    0, patch, Ghost::AroundCells, 1);
    old_dw->get(old_neg_charge,   d_lb->ccNegCharge,    0, patch, Ghost::AroundCells, 1);
    old_dw->get(old_permittivity, d_lb->ccPermittivity, 0, patch, Ghost::AroundCells, 1);
    new_dw->allocateAndPut(pos_charge,   d_lb->ccPosCharge, 0, patch);
    new_dw->allocateAndPut(neg_charge,   d_lb->ccNegCharge, 0, patch);
    new_dw->allocateAndPut(permittivity, d_lb->ccPermittivity, 0, patch);

    for(CellIterator iter = patch->getExtraCellIterator();!iter.done();iter++){
      IntVector c = *iter;
      pos_charge[c] = old_pos_charge[c];
      neg_charge[c] = old_neg_charge[c];
      permittivity[c] = old_permittivity[c];
    }
  } // patch loop
}

void GaussSolve::scheduleUpdateESPotential(SchedulerP& sched, const LevelP& level,
                                                  const MaterialSet* es_matl)
{
  Task* task = scinew Task("GaussSolve::updateESPotential", this,
                           &GaussSolve::updateESPotential,
                           level, sched.get_rep());

  task->modifies(d_lb->ccESPotential , d_es_matl);
  sched->addTask(task, level->eachPatch(), es_matl);
}

void GaussSolve::updateESPotential(const ProcessorGroup*, const PatchSubset* patches,
                                   const MaterialSubset* es_matls,
                                   DataWarehouse* old_dw, DataWarehouse* new_dw,
                                   LevelP, Scheduler*)
{
  FVMBoundCond bc;

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    CCVariable<double> es_potential;
    new_dw->getModifiable(es_potential, d_lb->ccESPotential, 0, patch);

    bc.setESPotentialBC(patch, 0, es_potential);

  } // end patch loop

}
