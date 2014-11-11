/*
 * The MIT License
 *
 * Copyright (c) 1997-2014 The University of Utah
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

#include <CCA/Components/MiniAero/MiniAero.h>
#include <CCA/Components/Examples/ExamplesLabel.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/SimpleMaterial.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Math/Matrix3.h>
#include <Core/Grid/Variables/Stencil7.h>
#include <Core/Grid/BoundaryConditions/BCDataArray.h>
#include <Core/Grid/BoundaryConditions/BoundCond.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Malloc/Allocator.h>

using namespace std;
using namespace Uintah;

//______________________________________________________________________
//  Preliminary
MiniAero::MiniAero(const ProcessorGroup* myworld)
    : UintahParallelComponent(myworld)
{
  conserved_label = VarLabel::create("conserved", CCVariable<Stencil7>::getTypeDescription());
  rho_CClabel = VarLabel::create("density", CCVariable<double>::getTypeDescription());
  vel_CClabel = VarLabel::create("velocity", CCVariable<Vector>::getTypeDescription());
  press_CClabel = VarLabel::create("pressure", CCVariable<double>::getTypeDescription());
  temp_CClabel = VarLabel::create("temperature", CCVariable<double>::getTypeDescription());
  flux_mass_CClabel = VarLabel::create("flux_mass", CCVariable<Vector>::getTypeDescription());
  flux_mom_CClabel = VarLabel::create("flux_mom", CCVariable<Matrix3>::getTypeDescription());
  flux_energy_CClabel = VarLabel::create("flux_energy", CCVariable<Vector>::getTypeDescription());
}

MiniAero::~MiniAero()
{
  VarLabel::destroy(conserved_label);
  VarLabel::destroy(rho_CClabel);
  VarLabel::destroy(vel_CClabel);
  VarLabel::destroy(press_CClabel);
  VarLabel::destroy(temp_CClabel);
}

//______________________________________________________________________
//
void MiniAero::problemSetup(const ProblemSpecP& params,
                            const ProblemSpecP& restart_prob_spec,
                            GridP& /*grid*/,
                            SimulationStateP& sharedState)
{
  sharedState_ = sharedState;
  ProblemSpecP miniaero = params->findBlock("MiniAero");
  miniaero->require("delt", delt_);
  mymat_ = scinew SimpleMaterial();
  sharedState->registerSimpleMaterial(mymat_);
}

//______________________________________________________________________
// 
void MiniAero::scheduleInitialize(const LevelP& level,
                                  SchedulerP& sched)
{
  Task* t = scinew Task("MiniAero::initialize", this, &MiniAero::initialize);
  t->computes(conserved_label);

  t->computes( vel_CClabel );
  t->computes( rho_CClabel );
  t->computes( press_CClabel );
  t->computes( temp_CClabel );

  sched->addTask(t, level->eachPatch(), sharedState_->allMaterials());
}

//______________________________________________________________________
// 
void MiniAero::scheduleComputeStableTimestep(const LevelP& level,
                                             SchedulerP& sched)
{
  Task* task = scinew Task("MiniAero::computeStableTimestep", this, &MiniAero::computeStableTimestep);

  task->computes(sharedState_->get_delt_label(), level.get_rep());
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());
}

//______________________________________________________________________
//
void MiniAero::scheduleTimeAdvance(const LevelP& level,
                                   SchedulerP& sched)
{
  Task* task = scinew Task("MiniAero::timeAdvance", this, &MiniAero::timeAdvance);

  task->requires(Task::OldDW, conserved_label, Ghost::AroundCells, 1);
  task->requires(Task::OldDW, sharedState_->get_delt_label());

  task->computes(conserved_label);
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());

  schedConvertOutput(level,sched);
}

void MiniAero::schedConvertOutput(const LevelP& level,
                                   SchedulerP& sched)
{

  Task* task = scinew Task("MiniAero::convertOutput", this, 
                           &MiniAero::convertOutput);


  task->requires(Task::NewDW, conserved_label,Ghost::None);

  task->computes(rho_CClabel);
  task->computes(vel_CClabel);
  task->computes(press_CClabel);

  sched->addTask(task,level->eachPatch(),sharedState_->allMaterials());

}

//______________________________________________________________________
//
void MiniAero::schedCellCenteredFlux(const LevelP& level,
                                   SchedulerP& sched)
{

  Task* task = scinew Task("MiniAero::cellCenteredFlux", this, 
                           &MiniAero::cellCenteredFlux);



  task->requires(Task::NewDW,rho_CClabel,Ghost::None);
  task->requires(Task::NewDW,vel_CClabel,Ghost::None);
  task->requires(Task::NewDW,press_CClabel,Ghost::None);

  task->computes(flux_mass_CClabel);
  task->computes(flux_mom_CClabel);
  task->computes(flux_energy_CClabel);

  sched->addTask(task,level->eachPatch(),sharedState_->allMaterials());

}

//______________________________________________________________________
//
void MiniAero::computeStableTimestep(const ProcessorGroup*,
                                     const PatchSubset* patches,
                                     const MaterialSubset*,
                                     DataWarehouse*,
                                     DataWarehouse* new_dw)
{
  new_dw->put(delt_vartype(delt_), sharedState_->get_delt_label(), getLevel(patches));
}

//______________________________________________________________________
//
void MiniAero::initialize(const ProcessorGroup*,
                          const PatchSubset* patches,
                          const MaterialSubset* matls,
                          DataWarehouse*,
                          DataWarehouse* new_dw)
{
  int matl = 0;
  int size = patches->size();
  for (int p = 0; p < size; p++) {
    const Patch* patch = patches->get(p);

    CCVariable<Stencil7> u;
    new_dw->allocateAndPut(u, conserved_label, matl, patch);
    
    //Initialize

    for (CellIterator iter = patch->getCellIterator(); !iter.done(); iter++) {
      IntVector c = *iter;
      u[c][0]=1.;
      u[c][1]=500.;
      u[c][2]=0.;
      u[c][3]=0.;
      u[c][4]=100000.;
    }
  }
}

//______________________________________________________________________
//
void MiniAero::timeAdvance(const ProcessorGroup*,
                           const PatchSubset* patches,
                           const MaterialSubset* matls,
                           DataWarehouse* old_dw,
                           DataWarehouse* new_dw)
{
  int matl = 0;

  //Loop for all patches on this processor
  int size = patches->size();
  for (int p = 0; p < size; p++) {
    const Patch* patch = patches->get(p);
    
    //  Get data from the data warehouse including 1 layer of
    // "ghost" cells from surrounding patches
    constCCVariable<Stencil7> u;
    old_dw->get(u, conserved_label, matl, patch, Ghost::AroundCells, 1);

    // dt, dx
    Vector dx = patch->getLevel()->dCell();
    delt_vartype dt;
    old_dw->get(dt, sharedState_->get_delt_label());

    // allocate memory
    CCVariable<Stencil7> new_u;
    new_dw->allocateAndPut(new_u, conserved_label, matl, patch);

    //Iterate through all the nodes
    for(CellIterator iter=patch->getCellIterator(); !iter.done();iter++) {
      IntVector c = *iter;
      new_u[c][0] = u[c][0];
      new_u[c][1] = u[c][1] + 1.;
      new_u[c][2] = u[c][2] ;
      new_u[c][3] = u[c][3] ;
      new_u[c][4] = u[c][4] ;      
    }

    //__________________________________
    // Boundary conditions: Neumann
    // Iterate over the faces encompassing the domain
    vector<Patch::FaceType>::const_iterator iter;
    vector<Patch::FaceType> bf;
    patch->getBoundaryFaces(bf);
    for (iter = bf.begin(); iter != bf.end(); ++iter) {
      Patch::FaceType face = *iter;

      IntVector axes = patch->getFaceAxes(face);
      int P_dir = axes[0];  // find the principal dir of that face

      IntVector offset(0, 0, 0);
      if (face == Patch::xminus || face == Patch::yminus || face == Patch::zminus) {
        offset[P_dir] += 1;
      }
      if (face == Patch::xplus || face == Patch::yplus || face == Patch::zplus) {
        offset[P_dir] -= 1;
      }
      Patch::FaceIteratorType PEC = Patch::ExtraPlusEdgeCells;
      for (CellIterator iter = patch->getFaceIterator(face, PEC); !iter.done(); iter++) {
        IntVector n = *iter;
        new_u[n][0] = new_u[n + offset][0];
        new_u[n][1] = new_u[n + offset][1];
        new_u[n][2] = new_u[n + offset][2];
        new_u[n][3] = new_u[n + offset][3];
        new_u[n][4] = new_u[n + offset][4];
      }
    }
  }
}


void MiniAero::convertOutput(const ProcessorGroup* /*pg*/,
                             const PatchSubset* patches,
                             const MaterialSubset* /*matls*/,
                             DataWarehouse* old_dw,
                             DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    Ghost::GhostType  gn  = Ghost::None;

    
    CCVariable<double> rho_CC, pressure_CC;
    CCVariable<Vector> vel_CC;
    constCCVariable<Stencil7> conserved;

    new_dw->get( conserved,  conserved_label, 0, patch, gn, 0 );

    new_dw->allocateAndPut( rho_CC, rho_CClabel,   0,patch );
    new_dw->allocateAndPut( vel_CC, vel_CClabel,   0,patch );
    new_dw->allocateAndPut(pressure_CC,press_CClabel,     0,patch );

    //__________________________________
    // Backout primitive quantities from
    // the conserved ones.
    for(CellIterator iter = patch->getExtraCellIterator(); !iter.done(); iter++) {
      IntVector c = *iter;
      rho_CC[c]    = conserved[c][0];
      vel_CC[c].x(conserved[c][1]/rho_CC[c]);
      vel_CC[c].y(conserved[c][2]/rho_CC[c]);
      vel_CC[c].z(conserved[c][3]/rho_CC[c]);
      pressure_CC[c]=conserved[c][4];
      
    }
  }
}


void MiniAero::cellCenteredFlux(const ProcessorGroup* /*pg*/,
                             const PatchSubset* patches,
                             const MaterialSubset* /*matls*/,
                             DataWarehouse* old_dw,
                             DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    //FIXME
    double d_gamma=1.4;
    Ghost::GhostType  gn  = Ghost::None;

    
    constCCVariable<double> rho_CC, pressure_CC;
    constCCVariable<Vector> vel_CC;
    CCVariable<Vector> flux_mass_CC;
    CCVariable<Matrix3> flux_mom_CC;
    CCVariable<Vector> flux_energy_CC;

    new_dw->get( rho_CC,  rho_CClabel, 0, patch, gn, 0 );
    new_dw->get( vel_CC,  vel_CClabel, 0, patch, gn, 0 );
    new_dw->get( pressure_CC,  press_CClabel, 0, patch, gn, 0 );

    new_dw->allocateAndPut( flux_mass_CC, flux_mass_CClabel,   0,patch );
    new_dw->allocateAndPut( flux_mom_CC, flux_mom_CClabel,   0,patch );
    new_dw->allocateAndPut( flux_energy_CC, flux_energy_CClabel,   0,patch );

    //__________________________________
    // Backout primitive quantities from
    // the conserved ones.
    for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++) {
      IntVector c = *iter;
      for (int idim=0; idim < 3; ++idim) {
        flux_mass_CC[c][idim]            = rho_CC[c]*vel_CC[c][idim];
        double KE=0;
        for (int jdim=0; jdim < 3; ++jdim) {
          KE += 0.5*rho_CC[c]*vel_CC[c][jdim]*vel_CC[c][jdim];
          flux_mom_CC[c](idim,jdim)      = rho_CC[c]*vel_CC[c][idim]*vel_CC[c][jdim];
        }
        flux_mom_CC[c](idim, idim)      += pressure_CC[c];
        flux_energy_CC[c][idim]            = KE + pressure_CC[c]*(d_gamma/(d_gamma-1) );
      }
    }
  }
}
