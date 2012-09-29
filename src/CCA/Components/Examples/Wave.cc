/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */


#include <CCA/Components/Examples/Wave.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/SimpleMaterial.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/DebugStream.h>

// TODO - don't do the step on the last RK4 step
// TODO - optimize?

using namespace Uintah;

static DebugStream wave("Wave", false);

Wave::Wave(const ProcessorGroup* myworld)
  : UintahParallelComponent(myworld)
{
  phi_label = VarLabel::create("phi", CCVariable<double>::getTypeDescription(), IntVector(1,1,1));
  pi_label = VarLabel::create("pi", CCVariable<double>::getTypeDescription());

  rk4steps[0].cur_dw = Task::OldDW;
  rk4steps[0].curphi_label = VarLabel::create("phi", CCVariable<double>::getTypeDescription(), IntVector(1,1,1));
  rk4steps[0].curpi_label = VarLabel::create("pi", CCVariable<double>::getTypeDescription());
  rk4steps[0].newphi_label = VarLabel::create("phi1", CCVariable<double>::getTypeDescription(), IntVector(1,1,1));
  rk4steps[0].newpi_label = VarLabel::create("pi1", CCVariable<double>::getTypeDescription());
  rk4steps[0].stepweight = 0.5;
  rk4steps[0].totalweight = 1/6.0;

  rk4steps[1].cur_dw = Task::NewDW;
  rk4steps[1].curphi_label = VarLabel::create("phi1", CCVariable<double>::getTypeDescription(), IntVector(1,1,1));
  rk4steps[1].curpi_label = VarLabel::create("pi1", CCVariable<double>::getTypeDescription());
  rk4steps[1].newphi_label = VarLabel::create("phi2", CCVariable<double>::getTypeDescription(), IntVector(1,1,1));
  rk4steps[1].newpi_label = VarLabel::create("pi2", CCVariable<double>::getTypeDescription());
  rk4steps[1].stepweight = 0.5;
  rk4steps[1].totalweight = 1/3.0;

  rk4steps[2].cur_dw = Task::NewDW;
  rk4steps[2].curphi_label = VarLabel::create("phi2", CCVariable<double>::getTypeDescription(), IntVector(1,1,1));
  rk4steps[2].curpi_label = VarLabel::create("pi2", CCVariable<double>::getTypeDescription());
  rk4steps[2].newphi_label = VarLabel::create("phi3", CCVariable<double>::getTypeDescription(), IntVector(1,1,1));
  rk4steps[2].newpi_label = VarLabel::create("pi3", CCVariable<double>::getTypeDescription());
  rk4steps[2].stepweight = 1.0;
  rk4steps[2].totalweight = 1/3.0;

  rk4steps[3].cur_dw = Task::NewDW;
  rk4steps[3].curphi_label = VarLabel::create("phi3", CCVariable<double>::getTypeDescription(), IntVector(1,1,1));
  rk4steps[3].curpi_label = VarLabel::create("pi3", CCVariable<double>::getTypeDescription());
  rk4steps[3].newphi_label = VarLabel::create("phi4", CCVariable<double>::getTypeDescription(), IntVector(1,1,1));
  rk4steps[3].newpi_label = VarLabel::create("pi4", CCVariable<double>::getTypeDescription());
  rk4steps[3].stepweight = 0.0;
  rk4steps[3].totalweight = 1/6.0;

}
//______________________________________________________________________
//
Wave::~Wave()
{
  VarLabel::destroy(phi_label);
  VarLabel::destroy(pi_label);
  for(int i=0;i<4;i++){
    VarLabel::destroy(rk4steps[i].curphi_label);
    VarLabel::destroy(rk4steps[i].curpi_label);
    VarLabel::destroy(rk4steps[i].newphi_label);
    VarLabel::destroy(rk4steps[i].newpi_label);
  }
}
//______________________________________________________________________
//
void Wave::problemSetup(const ProblemSpecP& params, 
                        const ProblemSpecP& restart_prob_spec, 
                        GridP& /*grid*/,  SimulationStateP& sharedState)
{
  sharedState_ = sharedState;
  ProblemSpecP wave = params->findBlock("Wave");
  wave->require("initial_condition", initial_condition);
  if(initial_condition == "Chombo"){
    wave->require("radius", r0);
  } else {
    throw ProblemSetupException("Unknown initial condition for Wave", __FILE__, __LINE__);
  }
  wave->require("integration", integration);
  if(integration != "Euler" && integration != "RK4")
    throw ProblemSetupException("Unknown integration method for Wave", __FILE__, __LINE__);
  mymat_ = scinew SimpleMaterial();
  sharedState->registerSimpleMaterial(mymat_);

}
//______________________________________________________________________
//
void Wave::scheduleInitialize(const LevelP& level,
			       SchedulerP& sched)
{
  Task* task = scinew Task("initialize",
			   this, &Wave::initialize);
  task->computes(phi_label);
  task->computes(pi_label);
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());
}
//______________________________________________________________________
// 
void Wave::scheduleComputeStableTimestep(const LevelP& level,
					  SchedulerP& sched)
{
  Task* task = scinew Task("computeStableTimestep",
			   this, &Wave::computeStableTimestep);
  task->computes(sharedState_->get_delt_label(),level.get_rep());
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());
}
//______________________________________________________________________
//
void
Wave::scheduleTimeAdvance( const LevelP& level, SchedulerP& sched)
{
  if(integration == "Euler"){
    Task* task = scinew Task("timeAdvance",
                             this, &Wave::timeAdvanceEuler);
    task->requires(Task::OldDW, phi_label, Ghost::AroundCells, 1);
    task->requires(Task::OldDW, pi_label,  Ghost::None, 0);
    if(level->getIndex()>0){        // REFINE 
      addRefineDependencies(task, phi_label, true, true);
    }
    //task->requires(Task::OldDW, sharedState_->get_delt_label());
    task->computes(phi_label);
    task->computes(pi_label);
    sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());
  } else if(integration == "RK4"){
    Task* task = scinew Task("setupRK4",
                             this, &Wave::setupRK4);
    task->requires(Task::OldDW, phi_label, Ghost::AroundCells, 1);
    task->requires(Task::OldDW, pi_label,  Ghost::None, 0);
    if(level->getIndex()>0){        // REFINE 
      // TODO, fix calls to addRefineDependencies and refineFaces
      addRefineDependencies(task, phi_label, true, true);
    }
    task->computes(phi_label);
    task->computes(pi_label);
    sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());

    for(int i=0;i<4;i++){
      Step* s = &rk4steps[i];
      Task* task = scinew Task("timeAdvance",
                               this, &Wave::timeAdvanceRK4, s);
                               
      task->requires(Task::OldDW, sharedState_->get_delt_label(), level.get_rep());
      task->requires(Task::OldDW, phi_label,      Ghost::None);
      task->requires(Task::OldDW, pi_label,       Ghost::None);
      task->requires(s->cur_dw, s->curphi_label,  Ghost::AroundCells, 1);
      task->requires(s->cur_dw, s->curpi_label,   Ghost::None, 0);

      if(level->getIndex()>0){        // REFINE 
        addRefineDependencies(task, s->curphi_label, true, true);
      }
      task->computes(s->newphi_label);
      task->computes(s->newpi_label);
      task->modifies(phi_label);
      task->modifies(pi_label);
      sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());
    }
  } else {
    throw ProblemSetupException("Unknown integration method for wave", __FILE__, __LINE__);
  }
}
//______________________________________________________________________
//
void Wave::initialize(const ProcessorGroup*,
			const PatchSubset* patches,
			const MaterialSubset* matls,
			DataWarehouse*, DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    for(int m = 0;m<matls->size();m++){
      int matl = matls->get(m);
      CCVariable<double> pi;
      new_dw->allocateAndPut(pi, pi_label, matl, patch);
      pi.initialize(0);

      CCVariable<double> phi;
      new_dw->allocateAndPut(phi, phi_label, matl, patch);

      if(initial_condition == "Chombo"){
        // Initial conditions to  mimic AMRWaveEqn from Chombo
        // Only matches when the domain is [-.5,-.5,-.5] to [.5,.5,.5]
        for(CellIterator iter(phi.getLowIndex(), phi.getHighIndex()); !iter.done(); iter++){
          Point pos = patch->nodePosition(*iter);
          double dist = (pos.asVector().length2());
          phi[*iter] = exp(-dist/(r0*r0))/(r0*r0*r0);
        }
      } else {
        throw ProblemSetupException("Unknown initial condition for Wave", __FILE__, __LINE__);
      }
    }
  }
}
//______________________________________________________________________
//
void Wave::computeStableTimestep(const ProcessorGroup*,
				  const PatchSubset* patches,
				  const MaterialSubset*,
				  DataWarehouse*, DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    double delt = patch->dCell().minComponent();
    const Level* level = getLevel(patches);
    new_dw->put(delt_vartype(delt), sharedState_->get_delt_label(), level);
  }
}

//______________________________________________________________________
// This could be done with the RK4 version below, but this is simpler...
void Wave::timeAdvanceEuler(const ProcessorGroup*,
                            const PatchSubset* patches,
                            const MaterialSubset* matls,
                            DataWarehouse* old_dw, DataWarehouse* new_dw)
{
  const Level* level = getLevel(patches);
  const Level* coarseLevel = 0;
  DataWarehouse* codw = 0;
  DataWarehouse* cndw = 0;
  if (level->getIndex() > 0) {
    coarseLevel = level->getCoarserLevel().get_rep();
    codw = old_dw->getOtherDataWarehouse(Task::CoarseOldDW);
    cndw = old_dw->getOtherDataWarehouse(Task::CoarseNewDW);
  }
  //Loop for all patches on this processor
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    for(int m = 0;m<matls->size();m++){
      int matl = matls->get(m);

      // cout << " Doing Wave::timeAdvanceEuler on patch " << patch->getID() << ", matl " << matl << endl;
      delt_vartype dt;
      old_dw->get(dt, sharedState_->get_delt_label(), level);

      constCCVariable<double> oldPhi;
      old_dw->get(oldPhi, phi_label, matl, patch, Ghost::AroundCells, 1);

      if(level->getIndex() > 0){        // REFINE
        refineFaces(patch, level, coarseLevel, oldPhi.castOffConst(),
                    phi_label, matl, codw, cndw);
      }                      

      constCCVariable<double> oldPi;
      old_dw->get(oldPi, pi_label, matl, patch, Ghost::None, 0);

      CCVariable<double> newPhi;
      //new_dw->allocateAndPut(newPhi, phi_label, matl, patch, Ghost::AroundCells, 1);
      new_dw->allocateAndPut(newPhi, phi_label, matl, patch);

      CCVariable<double> newPi;
      new_dw->allocateAndPut(newPi, pi_label, matl, patch);

      newPhi.initialize(0);
      newPi.initialize(0);

      // No boundary conditions - only works with periodic grids...

      double sumPhi = 0;
      Vector dx = patch->dCell();
      double sumdx2 = -2 / (dx.x()*dx.x()) -2/(dx.y()*dx.y()) - 2/(dx.z()*dx.z());
      Vector inv_dx2(1./(dx.x()*dx.x()), 1./(dx.y()*dx.y()), 1./(dx.z()*dx.z()));
      double maxphi = 0;
      double delt = dt;
      for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
        const IntVector& c = *iter;

        // Compute curl
        double curlPhi = curl(oldPhi, c, sumdx2, inv_dx2);

        // Integrate
        newPhi[c] = oldPhi[c] + oldPi[c] * delt;
        newPi[c] = oldPi[c] + curlPhi * delt;

        wave << "Index: " << c << " Phi " << newPhi[c] << " Pi " << newPi[c] << endl;

        sumPhi += newPhi[c];
        if(newPhi[c] > maxphi)
          maxphi = newPhi[c];
      }
    }
  }
}
//______________________________________________________________________
//
void Wave::setupRK4(const ProcessorGroup*,
                    const PatchSubset* patches,
                    const MaterialSubset* matls,
                    DataWarehouse* old_dw, DataWarehouse* new_dw)
{
  const Level* level = getLevel(patches);
  const Level* coarseLevel = 0;
  DataWarehouse* codw = 0;
  DataWarehouse* cndw = 0;
  if (level->getIndex() > 0) {
    coarseLevel = level->getCoarserLevel().get_rep();
    codw = old_dw->getOtherDataWarehouse(Task::CoarseOldDW);
    cndw = old_dw->getOtherDataWarehouse(Task::CoarseNewDW);
  }

  //Loop for all patches on this processor
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    for(int m = 0;m<matls->size();m++){
      int matl = matls->get(m);

      constCCVariable<double> oldPhi;
      old_dw->get(oldPhi, phi_label, matl, patch, Ghost::AroundCells, 1);

      if(level->getIndex() > 0){        // REFINE
        refineFaces(patch, level, coarseLevel, oldPhi.castOffConst(),
                    phi_label, matl, codw, cndw);
      }                      

      constCCVariable<double> oldPi;
      old_dw->get(oldPi, pi_label, matl, patch, Ghost::None, 0);

      CCVariable<double> newPhi;
      new_dw->allocateAndPut(newPhi, phi_label, matl, patch);
      CCVariable<double> newPi;
      new_dw->allocateAndPut(newPi, pi_label, matl, patch);


      CellIterator iter = patch->getCellIterator();
      newPhi.copyPatch(oldPhi, patch->getCellLowIndex(), patch->getCellHighIndex());
      newPi.copyPatch(oldPi, patch->getCellLowIndex(), patch->getCellHighIndex());
    }
  }
}
//______________________________________________________________________
//
void Wave::timeAdvanceRK4(const ProcessorGroup*,
                          const PatchSubset* patches,
                          const MaterialSubset* matls,
                          DataWarehouse* old_dw, DataWarehouse* new_dw,
                          Wave::Step* s)
{
  const Level* level = getLevel(patches);
  const Level* coarseLevel = 0;
  DataWarehouse* codw = 0;
  DataWarehouse* cndw = 0;
  if (level->getIndex() > 0) {
    coarseLevel = level->getCoarserLevel().get_rep();
    codw = old_dw->getOtherDataWarehouse(Task::CoarseOldDW);
    cndw = old_dw->getOtherDataWarehouse(Task::CoarseNewDW);
  }

  //Loop for all patches on this processor
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    for(int m = 0;m<matls->size();m++){
      int matl = matls->get(m);

      //cout << " Doing Wave::timeAdvanceRK4 on patch " << patch->getID() << ", matl " << matl << endl;
      delt_vartype dt;
      old_dw->get(dt, sharedState_->get_delt_label(), level);

      DataWarehouse* cur_dw = new_dw->getOtherDataWarehouse(s->cur_dw);
      constCCVariable<double> curPhi;
      cur_dw->get(curPhi, s->curphi_label, matl, patch, Ghost::AroundCells, 1);

      if(level->getIndex() > 0){        // REFINE
        refineFaces(patch, level, coarseLevel, curPhi.castOffConst(),
                    phi_label, matl, codw, cndw);
      }                      


      constCCVariable<double> curPi;
      cur_dw->get(curPi, s->curpi_label, matl, patch, Ghost::None, 0);

      CCVariable<double> newPhi;
      new_dw->allocateAndPut(newPhi, s->newphi_label, matl, patch);
      CCVariable<double> newPi;
      new_dw->allocateAndPut(newPi, s->newpi_label, matl, patch);

      constCCVariable<double> oldPhi;
      old_dw->get(oldPhi, phi_label, matl, patch, Ghost::AroundCells, 1);

      if(level->getIndex() > 0){        // REFINE
        refineFaces(patch, level, coarseLevel, oldPhi.castOffConst(),
                    phi_label, matl, codw, cndw);
      }                      

      constCCVariable<double> oldPi;
      old_dw->get(oldPi, pi_label, matl, patch, Ghost::None, 0);

      CCVariable<double> totalPhi;
      new_dw->getModifiable(totalPhi, phi_label, matl, patch);
      CCVariable<double> totalPi;
      new_dw->getModifiable(totalPi, pi_label, matl, patch);

      CCVariable<double> curlPhi;
      new_dw->allocateTemporary(curlPhi, patch);

      // No boundary conditions - only works with periodic grids...

      //double sumPhi = 0;
      Vector dx = patch->dCell();
      double sumdx2 = -2 / (dx.x()*dx.x()) -2/(dx.y()*dx.y()) - 2/(dx.z()*dx.z());
      Vector inv_dx2(1./(dx.x()*dx.x()), 1./(dx.y()*dx.y()), 1./(dx.z()*dx.z()));
      double dtstep = dt * s->stepweight;
      double dttotal = dt * s->totalweight;
      for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
        const IntVector& c = *iter;

        // Compute curl
        double curlPhi = curl(curPhi, c, sumdx2, inv_dx2);

        // Integrate
        newPhi[c] = oldPhi[c] + curPi[c] * dtstep;
        newPi[c] = oldPi[c] + curlPhi * dtstep;

        totalPhi[c] += curPi[c] * dttotal;
        totalPi[c] += curlPhi * dttotal;

        //cerr << c << "rhs phi=" << curPi[c] << ", rhs pi=" << curlPhi << ", phi=" << newPhi[c] << ", pi=" << newPi[c] << ", total phi=" << totalPhi[c] << ", total pi=" << totalPi[c] << ", dt=" << dt << ", " << dtstep << ", " << dttotal << '\n';
        //sumPhi += newPhi[c];
      }
      //cerr << "sumPhi=" << sumPhi << '\n';
    }
  }
}
