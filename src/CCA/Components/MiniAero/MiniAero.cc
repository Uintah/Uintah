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
  u_label = VarLabel::create("u", CCVariable<double>::getTypeDescription());
}

MiniAero::~MiniAero()
{
  VarLabel::destroy(u_label);
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
  Task* task = scinew Task("MiniAero::initialize", this, &MiniAero::initialize);
  task->computes(u_label);
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());
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

  task->requires(Task::OldDW, u_label, Ghost::AroundCells, 1);
  task->requires(Task::OldDW, sharedState_->get_delt_label());

  task->computes(u_label);
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());
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

    CCVariable<double> u;
    new_dw->allocateAndPut(u, u_label, matl, patch);
    u.initialize(0.);

    //Initialize
    // u = sin( pi*x ) + sin( pi*2*y ) + sin(pi*3z )

    for (CellIterator iter = patch->getCellIterator(); !iter.done(); iter++) {
      IntVector c = *iter;
      Point p = patch->cellPosition(c);
      u[c] = sin(p.x() * 3.14159265358) + sin(p.y() * 2 * 3.14159265358) + sin(p.z() * 3 * 3.14159265358);
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
    constCCVariable<double> u;
    old_dw->get(u, u_label, matl, patch, Ghost::AroundCells, 1);

    // dt, dx
    Vector dx = patch->getLevel()->dCell();
    delt_vartype dt;
    old_dw->get(dt, sharedState_->get_delt_label());

    // allocate memory
    CCVariable<double> new_u;
    new_dw->allocateAndPut(new_u, u_label, matl, patch);

    //Iterate through all the nodes
    for(CellIterator iter=patch->getCellIterator(); !iter.done();iter++) {
      IntVector c = *iter;

      double dudx = (u[c + IntVector(1, 0, 0)] - u[c - IntVector(1, 0, 0)]) / (2.0 * dx.x());
      double dudy = (u[c + IntVector(0, 1, 0)] - u[c - IntVector(0, 1, 0)]) / (2.0 * dx.y());
      double dudz = (u[c + IntVector(0, 0, 1)] - u[c - IntVector(0, 0, 1)]) / (2.0 * dx.z());
      double du = -u[c] * dt * (dudx + dudy + dudz);
      new_u[c] = u[c] + du;
      
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
        new_u[n] = new_u[n + offset];
      }
    }
  }
}
