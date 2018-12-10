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

#include <CCA/Components/FVM/ElectrostaticSolve.h>
#include <CCA/Components/FVM/FVMLabel.h>
#include <CCA/Components/FVM/FVMMaterial.h>
#include <CCA/Components/FVM/FVMBoundCond.h>

#include <CCA/Ports/Scheduler.h>

#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Geometry/Vector.h>
#include <Core/Grid/Ghost.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/Stencil7.h>
#include <Core/Grid/Task.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/ProblemSpec/ProblemSpecP.h>

using namespace Uintah;

ElectrostaticSolve::ElectrostaticSolve(const ProcessorGroup* myworld,
                                       const MaterialManagerP materialManager)
  : ApplicationCommon(myworld, materialManager)
{
  d_lb = scinew FVMLabel();

  d_delt = 0;
  d_solver = 0;
  d_with_mpm = false;

  d_es_matl  = scinew MaterialSubset();
  d_es_matl->add(0);
  d_es_matl->addReference();

  d_es_matlset  = scinew MaterialSet();
  d_es_matlset->add(0);
  d_es_matlset->addReference();
}
//__________________________________
//
ElectrostaticSolve::~ElectrostaticSolve()
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
void ElectrostaticSolve::problemSetup(const ProblemSpecP& prob_spec,
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

  if( !fvm_ps ) {
    throw ProblemSetupException("ERROR: Cannot find the FVM block",
                                __FILE__, __LINE__);
  }
  
  d_solver->readParameters(fvm_ps, "electrostatic_solver");

  d_solver->getParameters()->setSolveOnExtraCells(false);
    
  fvm_ps->require("delt", d_delt);

  if( !d_with_mpm ) {

    ProblemSpecP mat_ps =
      root_ps->findBlockWithOutAttribute("MaterialProperties");
  
    if( !mat_ps ) {
      throw ProblemSetupException("ERROR: Cannot find the Material Properties block",
                                  __FILE__, __LINE__);
    }

    ProblemSpecP fvm_mat_ps = mat_ps->findBlock("FVM");

    if( !fvm_mat_ps ) {
      throw ProblemSetupException("ERROR: Cannot find the FVM Materials Properties block",
                                  __FILE__, __LINE__);
    }

    for ( ProblemSpecP ps = fvm_mat_ps->findBlock("material"); ps != nullptr; ps = ps->findNextBlock("material") ) {

      FVMMaterial *mat = scinew FVMMaterial(ps, m_materialManager, FVMMaterial::ESPotential);
      m_materialManager->registerMaterial( "FVM", mat);
    }
  }
}

void
ElectrostaticSolve::outputProblemSpec(ProblemSpecP& ps)
{

}

//__________________________________
// 
void
ElectrostaticSolve::scheduleInitialize( const LevelP     & level,
                                              SchedulerP & sched )
{
  const MaterialSet* fvm_matls = m_materialManager->allMaterials( "FVM" );

  Task* t = scinew Task("ElectrostaticSolve::initialize", this,
                        &ElectrostaticSolve::initialize);

  t->computes(d_lb->ccConductivity);
  sched->addTask(t, level->eachPatch(), fvm_matls);

  d_solver->scheduleInitialize(level,sched, fvm_matls);
}
//__________________________________
//
void ElectrostaticSolve::scheduleRestartInitialize(const LevelP& level,
                                            SchedulerP& sched)
{
}
//__________________________________
// 
void ElectrostaticSolve::scheduleComputeStableTimeStep(const LevelP& level,
                                          SchedulerP& sched)
{
  Task* task = scinew Task("computeStableTimeStep",this, 
                           &ElectrostaticSolve::computeStableTimeStep);
  task->computes(getDelTLabel(),level.get_rep());
  sched->addTask(task, level->eachPatch(), m_materialManager->allMaterials( "FVM" ));
}
//__________________________________
//
void
ElectrostaticSolve::scheduleTimeAdvance( const LevelP& level, SchedulerP& sched)
{
  const MaterialSet* fvm_matls = m_materialManager->allMaterials( "FVM" );
  // const MaterialSet* all_matls = m_materialManager->allMaterials();

  scheduleComputeConductivity(   sched, level, fvm_matls);
  scheduleComputeFCConductivity( sched, level, d_es_matlset);
  scheduleBuildMatrixAndRhs(     sched, level, d_es_matlset);

  d_solver->scheduleSolve(level, sched, d_es_matlset,
                          d_lb->ccESPotentialMatrix, Task::NewDW,
                          d_lb->ccESPotential, false,
                          d_lb->ccRHS_ESPotential, Task::NewDW,
                          0, Task::OldDW,false);

  scheduleUpdateESPotential(sched, level, d_es_matlset);
  scheduleComputeCurrent(sched, level, d_es_matlset);

}
//__________________________________
//
void ElectrostaticSolve::computeStableTimeStep(const ProcessorGroup*,
                                  const PatchSubset* pss,
                                  const MaterialSubset*,
                                  DataWarehouse*, DataWarehouse* new_dw)
{
  new_dw->put(delt_vartype(d_delt), getDelTLabel(),getLevel(pss));
}
//__________________________________
//
void ElectrostaticSolve::initialize(const ProcessorGroup*,
                       const PatchSubset* patches,
                       const MaterialSubset* matls,
                       DataWarehouse*, DataWarehouse* new_dw)
{
  FVMBoundCond bc;
  int num_matls = m_materialManager->getNumMatls( "FVM" );

  for (int p = 0; p < patches->size(); p++){
    const Patch* patch = patches->get(p);
    for(int m = 0; m < num_matls; m++){
      FVMMaterial* fvm_matl = (FVMMaterial* ) m_materialManager->getMaterial( "FVM", m);
      int idx = fvm_matl->getDWIndex();

      CCVariable<double> conductivity;
      new_dw->allocateAndPut(conductivity, d_lb->ccConductivity, idx, patch);

      fvm_matl->initializeConductivity(conductivity, patch);

      bc.setConductivityBC(patch, idx, conductivity);

    }
  }
}

//______________________________________________________________________
//
void ElectrostaticSolve::scheduleComputeConductivity(SchedulerP& sched,
                                                     const LevelP& level,
                                                     const MaterialSet* fvm_matls)
{
  Task* t = scinew Task("ElectrostaticSolve::computeConductivity", this,
                           &ElectrostaticSolve::computeConductivity);

  t->requires(Task::OldDW, d_lb->ccConductivity, Ghost::AroundCells, 1);
  t->computes(d_lb->ccConductivity);
  t->computes(d_lb->ccGridConductivity, d_es_matl, Task::OutOfDomain);

  sched->addTask(t, level->eachPatch(), fvm_matls);
}

//______________________________________________________________________
//
void ElectrostaticSolve::computeConductivity(const ProcessorGroup* pg,
                                             const PatchSubset* patches,
                                             const MaterialSubset* fvm_matls,
                                             DataWarehouse* old_dw,
                                             DataWarehouse* new_dw)
{
  int num_matls = m_materialManager->getNumMatls( "FVM" );
  for (int p = 0; p < patches->size(); p++){
    const Patch* patch = patches->get(p);

    CCVariable<double>   grid_conductivity;

    new_dw->allocateAndPut(grid_conductivity, d_lb->ccGridConductivity, 0, patch);

    grid_conductivity.initialize(0.0);

    for(int m = 0; m < num_matls; m++){
      FVMMaterial* fvm_matl = (FVMMaterial* ) m_materialManager->getMaterial( "FVM", m);
      int idx = fvm_matl->getDWIndex();

      constCCVariable<double> old_conductivty;
      CCVariable<double> conductivity;

      old_dw->get(old_conductivty, d_lb->ccConductivity, idx, patch, Ghost::AroundCells, 1);
      new_dw->allocateAndPut(conductivity, d_lb->ccConductivity,  idx, patch);

      for(CellIterator iter = patch->getExtraCellIterator();!iter.done();iter++){
        IntVector c = *iter;
        conductivity[c] = old_conductivty[c];
        if(conductivity[c] > 0.0){
          grid_conductivity[c] = conductivity[c];
        }
      }
    } // material loop
  } // patch loop
}

void ElectrostaticSolve::scheduleComputeFCConductivity(SchedulerP& sched, const LevelP& level,
                                                       const MaterialSet* es_matls)
{
  Task* t = scinew Task("ElectrostaticSolve::computeFCConductivity", this,
                        &ElectrostaticSolve::computeFCConductivity);

    t->requires(Task::NewDW, d_lb->ccGridConductivity, Ghost::AroundCells, 1);
    t->computes(d_lb->fcxConductivity,    d_es_matl, Task::OutOfDomain);
    t->computes(d_lb->fcyConductivity,    d_es_matl, Task::OutOfDomain);
    t->computes(d_lb->fczConductivity,    d_es_matl, Task::OutOfDomain);
    sched->addTask(t, level->eachPatch(), es_matls);
}

void ElectrostaticSolve::computeFCConductivity(const ProcessorGroup* pg,const PatchSubset* patches,
                                               const MaterialSubset* es_matls,DataWarehouse* old_dw,
                                               DataWarehouse* new_dw)
{
  for (int p = 0; p < patches->size(); p++){
    const Patch* patch = patches->get(p);

    constCCVariable<double>  grid_conductivity;

    SFCXVariable<double> fcx_conductivity;
    SFCYVariable<double> fcy_conductivity;
    SFCZVariable<double> fcz_conductivity;


    new_dw->get(grid_conductivity, d_lb->ccGridConductivity, 0, patch, Ghost::AroundCells, 1);
    new_dw->allocateAndPut(fcx_conductivity, d_lb->fcxConductivity,    0, patch);
    new_dw->allocateAndPut(fcy_conductivity, d_lb->fcyConductivity,    0, patch);
    new_dw->allocateAndPut(fcz_conductivity, d_lb->fczConductivity,    0, patch);

    fcx_conductivity.initialize(0.0);
    fcy_conductivity.initialize(0.0);
    fcz_conductivity.initialize(0.0);

    for(CellIterator iter = patch->getSFCXIterator(); !iter.done(); iter++){
      IntVector c = *iter;
      IntVector offset(-1,0,0);
      fcx_conductivity[c] = .5*(grid_conductivity[c] + grid_conductivity[c + offset]);
    }

    for(CellIterator iter = patch->getSFCYIterator(); !iter.done(); iter++){
      IntVector c = *iter;
      IntVector offset(0,-1,0);
      fcy_conductivity[c] = .5*(grid_conductivity[c] + grid_conductivity[c + offset]);
    }

    for(CellIterator iter = patch->getSFCZIterator(); !iter.done(); iter++){
      IntVector c = *iter;
      IntVector offset(0,0,-1);
      fcz_conductivity[c] = .5*(grid_conductivity[c] + grid_conductivity[c + offset]);
    }
  } // patch loop
}
//______________________________________________________________________
//
void ElectrostaticSolve::scheduleBuildMatrixAndRhs(SchedulerP& sched,
                                                   const LevelP& level,
                                                   const MaterialSet* es_matl)
{
  Task* task = scinew Task("ElectrostaticSolve::buildMatrixAndRhs", this,
                           &ElectrostaticSolve::buildMatrixAndRhs,
                           level, sched.get_rep());

  task->requires(Task::NewDW, d_lb->fcxConductivity , Ghost::AroundCells, 1);
  task->requires(Task::NewDW, d_lb->fcyConductivity , Ghost::AroundCells, 1);
  task->requires(Task::NewDW, d_lb->fczConductivity , Ghost::AroundCells, 1);

  task->computes(d_lb->ccESPotentialMatrix, d_es_matl, Task::OutOfDomain);
  task->computes(d_lb->ccRHS_ESPotential,   d_es_matl, Task::OutOfDomain);

  sched->addTask(task, level->eachPatch(), es_matl);
}
//______________________________________________________________________
//
void ElectrostaticSolve::buildMatrixAndRhs(const ProcessorGroup* pg,
                           const PatchSubset* patches,
                           const MaterialSubset* ,
                           DataWarehouse* old_dw, DataWarehouse* new_dw,
                           LevelP level, Scheduler* sched)
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

    constSFCXVariable<double> fcx_conductivity;
    constSFCYVariable<double> fcy_conductivity;
    constSFCZVariable<double> fcz_conductivity;

    new_dw->get(fcx_conductivity, d_lb->fcxConductivity, 0, patch, Ghost::AroundCells, 1);
    new_dw->get(fcy_conductivity, d_lb->fcyConductivity, 0, patch, Ghost::AroundCells, 1);
    new_dw->get(fcz_conductivity, d_lb->fczConductivity, 0, patch, Ghost::AroundCells, 1);

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
      double efc = e*fcx_conductivity[c + xoffset];
      double wfc = w*fcx_conductivity[c];
      double nfc = n*fcy_conductivity[c + yoffset];
      double sfc = s*fcy_conductivity[c];
      double tfc = t*fcz_conductivity[c + zoffset];
      double bfc = b*fcz_conductivity[c];
      double center = efc + wfc + nfc + sfc + tfc + bfc;

      A_tmp.p = -center;
      A_tmp.n = nfc;   A_tmp.s = sfc;
      A_tmp.e = efc;   A_tmp.w = wfc;
      A_tmp.t = tfc;   A_tmp.b = bfc;
    } // End CellIterator

    bc.setESBoundaryConditions(patch, 0, A, rhs, fcx_conductivity, fcy_conductivity, fcz_conductivity);

  } // End patches
}

void ElectrostaticSolve::scheduleSolve(SchedulerP& sched,
                                       const LevelP& level,
                                       const MaterialSet* es_matlset)
{
  d_solver->scheduleSolve(level, sched, d_es_matlset,
                          d_lb->ccESPotentialMatrix, Task::NewDW,
                          d_lb->ccESPotential, false,
                          d_lb->ccRHS_ESPotential, Task::NewDW,
                          d_lb->ccESPotential, Task::OldDW,false);
}

void ElectrostaticSolve::scheduleUpdateESPotential(SchedulerP& sched, const LevelP& level,
                                                  const MaterialSet* es_matl)
{
  Task* task = scinew Task("ElectrostaticSolve::updateESPotential", this,
                           &ElectrostaticSolve::updateESPotential,
                           level, sched.get_rep());

  task->requires(Task::NewDW, d_lb->fcxConductivity , Ghost::AroundCells, 1);
  task->requires(Task::NewDW, d_lb->fcyConductivity , Ghost::AroundCells, 1);
  task->requires(Task::NewDW, d_lb->fczConductivity , Ghost::AroundCells, 1);

  task->modifies(d_lb->ccESPotential , d_es_matl);
  sched->addTask(task, level->eachPatch(), es_matl);
}

void ElectrostaticSolve::updateESPotential(const ProcessorGroup*, const PatchSubset* patches,
                                           const MaterialSubset* es_matls,
                                           DataWarehouse* old_dw, DataWarehouse* new_dw,
                                           LevelP, Scheduler*)
{
  FVMBoundCond bc;

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    constSFCXVariable<double> fcx_conductivity;
    constSFCYVariable<double> fcy_conductivity;
    constSFCZVariable<double> fcz_conductivity;

    new_dw->get(fcx_conductivity, d_lb->fcxConductivity, 0, patch, Ghost::AroundCells, 1);
    new_dw->get(fcy_conductivity, d_lb->fcyConductivity, 0, patch, Ghost::AroundCells, 1);
    new_dw->get(fcz_conductivity, d_lb->fczConductivity, 0, patch, Ghost::AroundCells, 1);

    CCVariable<double> es_potential;
    new_dw->getModifiable(es_potential, d_lb->ccESPotential, 0, patch);

    bc.setESPotentialBC(patch, 0, es_potential,
                        fcx_conductivity, fcy_conductivity, fcz_conductivity);

  } // end patch loop
}

void ElectrostaticSolve::scheduleComputeCurrent(SchedulerP& sched,
                                                const LevelP& level,
                                                const MaterialSet* es_matl)
{
  Task* t = scinew Task("ElectrostaticSolve::computeCurrent", this,
                        &ElectrostaticSolve::computeCurrent,
                        level, sched.get_rep());

  t->requires(Task::NewDW, d_lb->ccGridConductivity, Ghost::AroundCells, 1);
  t->requires(Task::NewDW, d_lb->ccESPotential,      Ghost::AroundCells, 1);

  t->computes(d_lb->ccCurrent, d_es_matl, Task::OutOfDomain);

  sched->addTask(t, level->eachPatch(), es_matl);
}

void ElectrostaticSolve::computeCurrent(const ProcessorGroup*,
                                        const PatchSubset* patches,
                                        const MaterialSubset* matls,
                                        DataWarehouse* old_dw,
                                        DataWarehouse* new_dw,
                                        LevelP, Scheduler*)
{
  IntVector xoffset(1,0,0);
  IntVector yoffset(0,1,0);
  IntVector zoffset(0,0,1);

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    Vector dx = patch->dCell();

    double hx2 = 2 * dx.x();
    double hy2 = 2 * dx.y();
    double hz2 = 2 * dx.z();

    constCCVariable<double> cc_cond;
    constCCVariable<double> cc_pot;

    CCVariable<Vector> cell_current;

    new_dw->get(cc_cond, d_lb->ccConductivity, 0, patch, Ghost::AroundCells, 1);
    new_dw->get(cc_pot,  d_lb->ccESPotential,  0, patch, Ghost::AroundCells, 1);

    new_dw->allocateAndPut(cell_current, d_lb->ccCurrent,  0, patch);

    for(CellIterator iter(patch->getCellIterator()); !iter.done(); iter++){
      IntVector c = *iter;
      Vector current;
      current.x( cc_cond[c] * (cc_pot[c + xoffset] - cc_pot[c - xoffset])/hx2 );
      current.y( cc_cond[c] * (cc_pot[c + yoffset] - cc_pot[c - yoffset])/hy2 );
      current.z( cc_cond[c] * (cc_pot[c + zoffset] - cc_pot[c - zoffset])/hz2) ;
      cell_current[c] = current;
    }
  } // end patch loop
}
