
#include <Packages/Uintah/CCA/Components/Examples/Wave.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/SimpleMaterial.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Core/Malloc/Allocator.h>

// TODO - don't do the step on the last RK4 step
// TODO - optimize?

using namespace Uintah;

Wave::Wave(const ProcessorGroup* myworld)
  : UintahParallelComponent(myworld)
{
  phi_label = VarLabel::create("phi", CCVariable<double>::getTypeDescription());
  pi_label = VarLabel::create("pi", CCVariable<double>::getTypeDescription());

  rk4steps[0].cur_dw = Task::OldDW;
  rk4steps[0].curphi_label = VarLabel::create("phi", CCVariable<double>::getTypeDescription());
  rk4steps[0].curpi_label = VarLabel::create("pi", CCVariable<double>::getTypeDescription());
  rk4steps[0].newphi_label = VarLabel::create("phi1", CCVariable<double>::getTypeDescription());
  rk4steps[0].newpi_label = VarLabel::create("pi1", CCVariable<double>::getTypeDescription());
  rk4steps[0].stepweight = 0.5;
  rk4steps[0].totalweight = 1/6.0;

  rk4steps[1].cur_dw = Task::NewDW;
  rk4steps[1].curphi_label = VarLabel::create("phi1", CCVariable<double>::getTypeDescription());
  rk4steps[1].curpi_label = VarLabel::create("pi1", CCVariable<double>::getTypeDescription());
  rk4steps[1].newphi_label = VarLabel::create("phi2", CCVariable<double>::getTypeDescription());
  rk4steps[1].newpi_label = VarLabel::create("pi2", CCVariable<double>::getTypeDescription());
  rk4steps[1].stepweight = 0.5;
  rk4steps[1].totalweight = 1/3.0;

  rk4steps[2].cur_dw = Task::NewDW;
  rk4steps[2].curphi_label = VarLabel::create("phi2", CCVariable<double>::getTypeDescription());
  rk4steps[2].curpi_label = VarLabel::create("pi2", CCVariable<double>::getTypeDescription());
  rk4steps[2].newphi_label = VarLabel::create("phi3", CCVariable<double>::getTypeDescription());
  rk4steps[2].newpi_label = VarLabel::create("pi3", CCVariable<double>::getTypeDescription());
  rk4steps[2].stepweight = 1.0;
  rk4steps[2].totalweight = 1/3.0;

  rk4steps[3].cur_dw = Task::NewDW;
  rk4steps[3].curphi_label = VarLabel::create("phi3", CCVariable<double>::getTypeDescription());
  rk4steps[3].curpi_label = VarLabel::create("pi3", CCVariable<double>::getTypeDescription());
  rk4steps[3].newphi_label = VarLabel::create("phi4", CCVariable<double>::getTypeDescription());
  rk4steps[3].newpi_label = VarLabel::create("pi4", CCVariable<double>::getTypeDescription());
  rk4steps[3].stepweight = 0.0;
  rk4steps[3].totalweight = 1/6.0;

}

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

void Wave::problemSetup(const ProblemSpecP& params, GridP& /*grid*/,
			 SimulationStateP& sharedState)
{
  sharedState_ = sharedState;
  ProblemSpecP wave = params->findBlock("Wave");
  wave->require("initial_condition", initial_condition);
  if(initial_condition == "Chombo"){
    wave->require("radius", r0);
  } else {
    throw ProblemSetupException("Unknown initial condition for Wave");
  }
  wave->require("integration", integration);
  if(integration != "Euler" && integration != "RK4")
    throw ProblemSetupException("Unknown integration method for Wave");
  mymat_ = new SimpleMaterial();
  sharedState->registerSimpleMaterial(mymat_);

}
 
void Wave::scheduleInitialize(const LevelP& level,
			       SchedulerP& sched)
{
  Task* task = scinew Task("initialize",
			   this, &Wave::initialize);
  task->computes(phi_label);
  task->computes(pi_label);
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());
}
 
void Wave::scheduleComputeStableTimestep(const LevelP& level,
					  SchedulerP& sched)
{
  Task* task = scinew Task("computeStableTimestep",
			   this, &Wave::computeStableTimestep);
  task->computes(sharedState_->get_delt_label());
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());
}

void
Wave::scheduleTimeAdvance( const LevelP& level, SchedulerP& sched, int, int )
{
  if(integration == "Euler"){
    Task* task = scinew Task("timeAdvance",
                             this, &Wave::timeAdvanceEuler);
    task->requires(Task::OldDW, phi_label, Ghost::AroundNodes, 1);
    task->requires(Task::OldDW, pi_label, Ghost::AroundNodes, 1);
    task->requires(Task::OldDW, sharedState_->get_delt_label());
    task->computes(phi_label);
    task->computes(pi_label);
    sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());
  } else if(integration == "RK4"){
    Task* task = scinew Task("setupRK4",
                             this, &Wave::setupRK4);
    task->requires(Task::OldDW, phi_label, Ghost::AroundNodes, 1);
    task->requires(Task::OldDW, pi_label, Ghost::AroundNodes, 1);
    task->computes(phi_label);
    task->computes(pi_label);
    sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());

    for(int i=0;i<4;i++){
      Step* step = &rk4steps[i];
      Task* task = scinew Task("timeAdvance",
                               this, &Wave::timeAdvanceRK4, step);
      task->requires(Task::OldDW, sharedState_->get_delt_label());
      task->requires(Task::OldDW, phi_label, Ghost::None);
      task->requires(Task::OldDW, pi_label, Ghost::None);
      task->requires(step->cur_dw, step->curphi_label, Ghost::AroundCells, 1);
      task->requires(step->cur_dw, step->curpi_label, Ghost::AroundCells, 1);
      task->computes(step->newphi_label);
      task->computes(step->newpi_label);
      task->modifies(phi_label);
      task->modifies(pi_label);
      sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());
    }
  } else {
    throw ProblemSetupException("Unknown integration method for wave");
  }
}

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
        for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
          Point pos = patch->nodePosition(*iter);
          double dist = (pos.asVector().length2());
          phi[*iter] = exp(-dist/(r0*r0))/(r0*r0*r0);
        }
      } else {
        throw ProblemSetupException("Unknown initial condition for Wave");
      }
    }
  }
}

void Wave::computeStableTimestep(const ProcessorGroup*,
				  const PatchSubset* patches,
				  const MaterialSubset*,
				  DataWarehouse*, DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    double delt = patch->dCell().minComponent();
    new_dw->put(delt_vartype(delt), sharedState_->get_delt_label());
  }
}

// This could be done with the RK4 version below, but this is simpler...
void Wave::timeAdvanceEuler(const ProcessorGroup*,
			const PatchSubset* patches,
			const MaterialSubset* matls,
			DataWarehouse* old_dw, DataWarehouse* new_dw)
{
  //Loop for all patches on this processor
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    for(int m = 0;m<matls->size();m++){
      int matl = matls->get(m);

      delt_vartype dt;
      old_dw->get(dt, sharedState_->get_delt_label());

      constCCVariable<double> oldPhi;
      old_dw->get(oldPhi, phi_label, matl, patch, Ghost::AroundCells, 1);
      constCCVariable<double> oldPi;
      old_dw->get(oldPi, pi_label, matl, patch, Ghost::AroundCells, 1);

      CCVariable<double> newPhi;
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
        double curlPhi = sumdx2 * oldPhi[c]
          + (oldPhi[c+IntVector(1,0,0)] + oldPhi[c-IntVector(1,0,0)]) * inv_dx2.x()
          + (oldPhi[c+IntVector(0,1,0)] + oldPhi[c-IntVector(0,1,0)]) * inv_dx2.y()
          + (oldPhi[c+IntVector(0,0,1)] + oldPhi[c-IntVector(0,0,1)]) * inv_dx2.z();

        // Integrate
        newPhi[c] = oldPhi[c] + oldPi[c] * delt;
        newPi[c] = oldPi[c] + curlPhi * delt;

        cerr << c << ", phi=" << newPhi[c] << ", pi=" << newPi[c] << '\n';
        sumPhi += newPhi[c];
        if(newPhi[c] > maxphi)
          maxphi = newPhi[c];
      }
      cerr << "sumPhi=" << sumPhi << '\n';
      cerr << "maxPhi=" << maxphi << '\n';
    }
  }
}

void Wave::setupRK4(const ProcessorGroup*,
                    const PatchSubset* patches,
                    const MaterialSubset* matls,
                    DataWarehouse* old_dw, DataWarehouse* new_dw)
{
  //Loop for all patches on this processor
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    for(int m = 0;m<matls->size();m++){
      int matl = matls->get(m);

      constCCVariable<double> oldPhi;
      old_dw->get(oldPhi, phi_label, matl, patch, Ghost::AroundCells, 1);
      constCCVariable<double> oldPi;
      old_dw->get(oldPi, pi_label, matl, patch, Ghost::AroundCells, 1);

      CCVariable<double> newPhi;
      new_dw->allocateAndPut(newPhi, phi_label, matl, patch);
      CCVariable<double> newPi;
      new_dw->allocateAndPut(newPi, pi_label, matl, patch);


      CellIterator iter = patch->getCellIterator();
      newPhi.copyPatch(oldPhi, patch->getLowIndex(), patch->getHighIndex());
      newPi.copyPatch(oldPi, patch->getLowIndex(), patch->getHighIndex());
    }
  }
}

void Wave::timeAdvanceRK4(const ProcessorGroup*,
                          const PatchSubset* patches,
                          const MaterialSubset* matls,
                          DataWarehouse* old_dw, DataWarehouse* new_dw,
                          Wave::Step* step)
{
  //Loop for all patches on this processor
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    for(int m = 0;m<matls->size();m++){
      int matl = matls->get(m);

      delt_vartype dt;
      old_dw->get(dt, sharedState_->get_delt_label());

      DataWarehouse* cur_dw = new_dw->getOtherDataWarehouse(step->cur_dw);
      constCCVariable<double> curPhi;
      cur_dw->get(curPhi, step->curphi_label, matl, patch, Ghost::AroundCells, 1);
      constCCVariable<double> curPi;
      cur_dw->get(curPi, step->curpi_label, matl, patch, Ghost::AroundCells, 1);

      CCVariable<double> newPhi;
      new_dw->allocateAndPut(newPhi, step->newphi_label, matl, patch);
      CCVariable<double> newPi;
      new_dw->allocateAndPut(newPi, step->newpi_label, matl, patch);

      constCCVariable<double> oldPhi;
      old_dw->get(oldPhi, phi_label, matl, patch, Ghost::AroundCells, 1);
      constCCVariable<double> oldPi;
      old_dw->get(oldPi, pi_label, matl, patch, Ghost::AroundCells, 1);

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
      double dtstep = dt * step->stepweight;
      double dttotal = dt * step->totalweight;
      //cerr << "STEP: " << step-&rk4steps[0] << '\n';
      for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
        const IntVector& c = *iter;

        // Compute curl
        double curlPhi = sumdx2 * curPhi[c]
          + (curPhi[c+IntVector(1,0,0)] + curPhi[c-IntVector(1,0,0)]) * inv_dx2.x()
          + (curPhi[c+IntVector(0,1,0)] + curPhi[c-IntVector(0,1,0)]) * inv_dx2.y()
          + (curPhi[c+IntVector(0,0,1)] + curPhi[c-IntVector(0,0,1)]) * inv_dx2.z();

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
