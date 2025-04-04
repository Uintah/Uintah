/*
 * The MIT License
 *
 * Copyright (c) 1997-2025 The University of Utah
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


#include <CCA/Components/Examples/ParticleTest1.h>
#include <CCA/Components/Examples/ExamplesLabel.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/NodeIterator.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/SimpleMaterial.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Malloc/Allocator.h>

#include <iostream>

using namespace std;

using namespace Uintah;
//______________________________________________________________________
//
ParticleTest1::ParticleTest1(const ProcessorGroup* myworld,
                             const MaterialManagerP materialManager)
  : ApplicationCommon(myworld, materialManager)
{
  lb_ = scinew ExamplesLabel();
  lb_->timeStepLabel = VarLabel::find(timeStep_name);
}
//______________________________________________________________________
//
ParticleTest1::~ParticleTest1()
{
  delete lb_;
}
//______________________________________________________________________
//
void ParticleTest1::problemSetup(const ProblemSpecP& params, 
                                 const ProblemSpecP& restart_prob_spec, 
                                 GridP& /*grid*/)
{
  m_scheduler->setPositionVar(lb_->pXLabel);
  
  ProblemSpecP pt1 = params->findBlock("ParticleTest1");
  pt1->getWithDefault("doOutput", doOutput_, 0);
  pt1->getWithDefault("doGhostCells", doGhostCells_ , 0);
  
  mymat_ = scinew SimpleMaterial();
  m_materialManager->registerSimpleMaterial(mymat_);
}

//______________________________________________________________________
//
void ParticleTest1::scheduleInitialize(const LevelP& level,
                                       SchedulerP& sched)
{
  Task* task = scinew Task("initialize",
                           this, &ParticleTest1::initialize);
  task->computesVar(lb_->pXLabel);
  task->computesVar(lb_->pMassLabel);
  task->computesVar(lb_->pParticleIDLabel);
  task->computesVar(lb_->dummyLabel);
  
  sched->addTask(task, level->eachPatch(), m_materialManager->allMaterials());
}

//______________________________________________________________________
//
void ParticleTest1::scheduleRestartInitialize(const LevelP& level,
                                              SchedulerP& sched)
{
}

//______________________________________________________________________
//
void ParticleTest1::scheduleComputeStableTimeStep(const LevelP& level,
                                                  SchedulerP& sched)
{
  Task* task = scinew Task("computeStableTimeStep",
                           this, &ParticleTest1::computeStableTimeStep);
  task->computesVar(getDelTLabel(),level.get_rep());
  sched->addTask(task, level->eachPatch(), m_materialManager->allMaterials());

}

//______________________________________________________________________
//
void
ParticleTest1::scheduleTimeAdvance( const LevelP& level, 
                                    SchedulerP& sched)
{
  const MaterialSet* matls = m_materialManager->allMaterials();

  Task* task = scinew Task("timeAdvance",
                           this, &ParticleTest1::timeAdvance);

  // set this in problemSetup.  0 is no ghost cells, 1 is all with 1 ghost
  // atound-node, and 2 mixes them
  if (doGhostCells_ == 0) {
    task->requiresVar(Task::OldDW, lb_->pParticleIDLabel, Ghost::None, 0);
    task->requiresVar(Task::OldDW, lb_->pXLabel, Ghost::None, 0);
    task->requiresVar(Task::OldDW, lb_->pMassLabel, Ghost::None, 0);
  }
  
  else if (doGhostCells_ == 1) {
    task->requiresVar(Task::OldDW, lb_->pXLabel, Ghost::AroundNodes, 1);
    task->requiresVar(Task::OldDW, lb_->pMassLabel, Ghost::AroundNodes, 1);
    task->requiresVar(Task::OldDW, lb_->pParticleIDLabel, Ghost::AroundNodes, 1);
  }
  else if (doGhostCells_ == 2) {
    task->requiresVar(Task::OldDW, lb_->pXLabel, Ghost::None, 0);
    task->requiresVar(Task::OldDW, lb_->pMassLabel, Ghost::AroundNodes, 1);
    task->requiresVar(Task::OldDW, lb_->pParticleIDLabel, Ghost::None, 0);
  }

  task->requiresVar(Task::OldDW, lb_->timeStepLabel);
  
  task->computesVar(lb_->dummyLabel);
  task->computesVar(lb_->pXLabel_preReloc);
  task->computesVar(lb_->pMassLabel_preReloc);
  task->computesVar(lb_->pParticleIDLabel_preReloc);
  sched->addTask(task, level->eachPatch(), m_materialManager->allMaterials());

  //__________________________________
  //  schedule relocate
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

//______________________________________________________________________
//
void ParticleTest1::computeStableTimeStep(const ProcessorGroup* /*pg*/,
                                     const PatchSubset* patches,
                                     const MaterialSubset* /*matls*/,
                                     DataWarehouse*,
                                     DataWarehouse* new_dw)
{
  new_dw->put(delt_vartype(1), getDelTLabel(),getLevel(patches));
}

//______________________________________________________________________
//
void ParticleTest1::initialize(const ProcessorGroup*,
                          const PatchSubset* patches,
                          const MaterialSubset* matls,
                          DataWarehouse* /*old_dw*/, 
                          DataWarehouse* new_dw)
{
  for( int p=0; p<patches->size(); ++p ){
    const Patch* patch = patches->get(p);
    const Point low = patch->cellPosition(patch->getCellLowIndex());
    const Point high = patch->cellPosition(patch->getCellHighIndex());
    for(int m = 0;m<matls->size();m++){
      srand(1);
      const int numParticles = 10;
      const int matl = matls->get(m);

      ParticleVariable<Point> px;
      ParticleVariable<double> pmass;
      ParticleVariable<long64> pids;
      
      CCVariable<double> dummy;  // used for testing outputNthProc
      new_dw->allocateAndPut( dummy, lb_->dummyLabel, matl, patch);
      dummy.initialize(0.0);

      ParticleSubset* subset = new_dw->createParticleSubset(numParticles,matl,patch);
      new_dw->allocateAndPut( px,    lb_->pXLabel,          subset );
      new_dw->allocateAndPut( pmass, lb_->pMassLabel,       subset );
      new_dw->allocateAndPut( pids,  lb_->pParticleIDLabel, subset );

      for( int i = 0; i < numParticles; ++i ){
        const Point pos( (((float) rand()) / RAND_MAX * ( high.x() - low.x()-1) + low.x()),
                         (((float) rand()) / RAND_MAX * ( high.y() - low.y()-1) + low.y()),
                         (((float) rand()) / RAND_MAX * ( high.z() - low.z()-1) + low.z()) );
        px[i] = pos;
        pids[i] = patch->getID()*numParticles+i;
        pmass[i] = ((float) rand()) / RAND_MAX * 10;
      }
    }
  }
}

//______________________________________________________________________
//
void ParticleTest1::timeAdvance(const ProcessorGroup*,
                        const PatchSubset* patches,
                        const MaterialSubset* matls,
                        DataWarehouse* old_dw, 
                        DataWarehouse* new_dw)
{
  for( int p=0; p<patches->size(); ++p ){
    const Patch* patch = patches->get(p);
    for( int m = 0; m<matls->size(); ++m ){
      int matl = matls->get(m);
      
      timeStep_vartype timeStep;
      old_dw->get(timeStep, lb_->timeStepLabel );
  
      CCVariable<double> dummy;  // used for testing outputNthProc
      new_dw->allocateAndPut( dummy, lb_->dummyLabel, matl, patch);
      dummy.initialize((int) timeStep);
      
      ParticleSubset* pset = old_dw->getParticleSubset(matl, patch);
      ParticleSubset* delset = scinew ParticleSubset(0,matl,patch);

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
      for( unsigned i = 0; i < pset->numParticles(); ++i ){
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
