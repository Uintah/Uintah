
#include <Packages/Uintah/CCA/Components/Examples/ParticleTest1.h>
#include <Packages/Uintah/CCA/Components/Examples/ExamplesLabel.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Grid/Variables/NCVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/NodeIterator.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/SimpleMaterial.h>
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Core/Malloc/Allocator.h>

#include <iostream>

using namespace std;

using namespace Uintah;

ParticleTest1::ParticleTest1(const ProcessorGroup* myworld)
  : UintahParallelComponent(myworld)
{
  lb_ = scinew ExamplesLabel();
}

ParticleTest1::~ParticleTest1()
{
  delete lb_;
}

void ParticleTest1::problemSetup(const ProblemSpecP& params, GridP& /*grid*/,
			 SimulationStateP& sharedState)
{
  sharedState_ = sharedState;
  ProblemSpecP pt1 = params->findBlock("ParticleTest1");
  pt1->getWithDefault("doOutput", doOutput_, 0);
  pt1->getWithDefault("doGhostCells", doGhostCells_ , 0);
  
  mymat_ = new SimpleMaterial();
  sharedState_->registerSimpleMaterial(mymat_);

}
 
void ParticleTest1::scheduleInitialize(const LevelP& level,
			       SchedulerP& sched)
{
  Task* task = scinew Task("initialize",
			   this, &ParticleTest1::initialize);
  task->computes(lb_->pXLabel);
  task->computes(lb_->pMassLabel);
  task->computes(lb_->pParticleIDLabel);
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());
}
 
void ParticleTest1::scheduleComputeStableTimestep(const LevelP& level,
					  SchedulerP& sched)
{
  Task* task = scinew Task("computeStableTimestep",
			   this, &ParticleTest1::computeStableTimestep);
  task->computes(sharedState_->get_delt_label());
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());

}

void
ParticleTest1::scheduleTimeAdvance( const LevelP& level, SchedulerP& sched,
			       int, int )
{
  const MaterialSet* matls = sharedState_->allMaterials();

  Task* task = scinew Task("timeAdvance",
			   this, &ParticleTest1::timeAdvance);

  // set this in problemSetup.  0 is no ghost cells, 1 is all with 1 ghost
  // atound-node, and 2 mixes them
  if (doGhostCells_ == 0) {
    task->requires(Task::OldDW, lb_->pParticleIDLabel, Ghost::None, 0);
    task->requires(Task::OldDW, lb_->pXLabel, Ghost::None, 0);
    task->requires(Task::OldDW, lb_->pMassLabel, Ghost::None, 0);
  }
  
  else if (doGhostCells_ == 1) {
    task->requires(Task::OldDW, lb_->pXLabel, Ghost::AroundNodes, 1);
    task->requires(Task::OldDW, lb_->pMassLabel, Ghost::AroundNodes, 1);
    task->requires(Task::OldDW, lb_->pParticleIDLabel, Ghost::AroundNodes, 1);
  }
  else if (doGhostCells_ == 2) {
    task->requires(Task::OldDW, lb_->pXLabel, Ghost::None, 0);
    task->requires(Task::OldDW, lb_->pMassLabel, Ghost::AroundNodes, 1);
    task->requires(Task::OldDW, lb_->pParticleIDLabel, Ghost::None, 0);
  }

  task->computes(lb_->pXLabel_preReloc);
  task->computes(lb_->pMassLabel_preReloc);
  task->computes(lb_->pParticleIDLabel_preReloc);
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());

  lb_->d_particleState.clear();
  lb_->d_particleState_preReloc.clear();
  for (int m = 0; m < matls->size(); m++) {
    vector<const VarLabel*> vars;
    vector<const VarLabel*> vars_preReloc;

    vars.push_back(lb_->pMassLabel);
    vars.push_back(lb_->pParticleIDLabel);

    vars_preReloc.push_back(lb_->pMassLabel_preReloc);
    vars_preReloc.push_back(lb_->pParticleIDLabel_preReloc);
    lb_->d_particleState.push_back(vars);
    lb_->d_particleState_preReloc.push_back(vars_preReloc);
  }

  sched->scheduleParticleRelocation(level, lb_->pXLabel_preReloc,
				    lb_->d_particleState_preReloc,
				    lb_->pXLabel, lb_->d_particleState,
				    lb_->pParticleIDLabel, matls);

}

void ParticleTest1::computeStableTimestep(const ProcessorGroup* /*pg*/,
				     const PatchSubset* /*patches*/,
				     const MaterialSubset* /*matls*/,
				     DataWarehouse*,
				     DataWarehouse* new_dw)
{
  new_dw->put(delt_vartype(1), sharedState_->get_delt_label());
}

void ParticleTest1::initialize(const ProcessorGroup*,
			  const PatchSubset* patches,
			  const MaterialSubset* matls,
			  DataWarehouse* /*old_dw*/, DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    Point low = patch->cellPosition(patch->getLowIndex());
    Point high = patch->cellPosition(patch->getHighIndex());
    for(int m = 0;m<matls->size();m++){
      srand(1);
      int numParticles = 10;
      int matl = matls->get(m);

      ParticleVariable<Point> px;
      ParticleVariable<double> pmass;
      ParticleVariable<long64> pids;


      ParticleSubset* subset = new_dw->createParticleSubset(numParticles,matl,patch);
      new_dw->allocateAndPut(px,       lb_->pXLabel,             subset);
      new_dw->allocateAndPut(pmass,          lb_->pMassLabel,          subset);
      new_dw->allocateAndPut(pids,    lb_->pParticleIDLabel,    subset);

      for (int i = 0; i < numParticles; i++) {
        Point pos( (((float) rand()) / RAND_MAX * ( high.x() - low.x()-1) + low.x()),
          (((float) rand()) / RAND_MAX * ( high.y() - low.y()-1) + low.y()),
          (((float) rand()) / RAND_MAX * ( high.z() - low.z()-1) + low.z()));
        px[i] = pos;
        pids[i] = patch->getID()*numParticles+i;
        pmass[i] = ((float) rand()) / RAND_MAX * 10;
      }
    }
  }
}

void ParticleTest1::timeAdvance(const ProcessorGroup*,
			const PatchSubset* patches,
			const MaterialSubset* matls,
			DataWarehouse* old_dw, DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    for(int m = 0;m<matls->size();m++){
      int matl = matls->get(m);
      ParticleSubset* pset = old_dw->getParticleSubset(matl, patch);
      ParticleSubset* delset = scinew ParticleSubset(pset->getParticleSet(),
                                                     false,matl,patch, 0);

      // Get the arrays of particle values to be changed
      constParticleVariable<Point> px;
      ParticleVariable<Point> pxnew;
      constParticleVariable<double> pmass;
      ParticleVariable<double> pmassnew;
      constParticleVariable<long64> pids;
      ParticleVariable<long64> pidsnew;

      old_dw->get(pmass, lb_->pMassLabel,               pset);
      old_dw->get(px,    lb_->pXLabel,                  pset);
      old_dw->get(pids,  lb_->pParticleIDLabel,         pset);

      new_dw->allocateAndPut(pmassnew, lb_->pMassLabel_preReloc,       pset);
      new_dw->allocateAndPut(pxnew,    lb_->pXLabel_preReloc,          pset);
      new_dw->allocateAndPut(pidsnew,  lb_->pParticleIDLabel_preReloc, pset);

      // every timestep, move down the +x axis, and decay the mass a little bit
      for (int i = 0; i < pset->numParticles(); i++) {
        Point pos( px[i].x() + .25, px[i].y(), px[i].z());
        pxnew[i] = pos;
        pidsnew[i] = pids[i];
        pmassnew[i] = pmass[i] *.9;
        if (doOutput_)
          cout << " Patch " << patch->getID() << ": ID " 
               << pidsnew[i] << ", pos " << pxnew[i] 
               << ", mass " << pmassnew[i] << endl;
      }
      new_dw->deleteParticles(delset);
    }
  }
}
