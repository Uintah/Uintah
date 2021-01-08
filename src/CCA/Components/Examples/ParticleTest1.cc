/*
 * The MIT License
 *
 * Copyright (c) 1997-2021 The University of Utah
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

#include <CCA/Ports/Scheduler.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/SimpleMaterial.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/NodeIterator.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Util/DOUT.hpp>

#include <iostream>

#define ONE 1                // This changes the number of ghost cells requested in task1.

using namespace std;

using namespace Uintah;

ParticleTest1::ParticleTest1(const ProcessorGroup* myworld,
                             const MaterialManagerP materialManager)
  : ApplicationCommon(myworld, materialManager)
{
  lb_ = scinew ExamplesLabel();
}

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
  pt1->getWithDefault("doOutput",     m_doOutput, 0);
  pt1->getWithDefault("nGhostCells",  m_NumGC,    0);

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
  task->computes(lb_->pXLabel);
  task->computes(lb_->pMassLabel);
  task->computes(lb_->pParticleIDLabel);
  sched->addTask(task, level->eachPatch(), m_materialManager->allMaterials());
}

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
  task->computes(getDelTLabel(),level.get_rep());
  sched->addTask(task, level->eachPatch(), m_materialManager->allMaterials());

}

//______________________________________________________________________
//
void
ParticleTest1::scheduleTimeAdvance( const LevelP& level,
                                    SchedulerP  & sched)
{
  schedTask1( level, sched);

  schedTask2( level, sched);

  //__________________________________
  //
  lb_->d_particleState.clear();
  lb_->d_particleState_preReloc.clear();

  const MaterialSet* matls = m_materialManager->allMaterials();

  for (int m = 0; m < matls->size(); m++) {
    vector<const VarLabel*> vars;
    vector<const VarLabel*> vars_preReloc;

    vars.push_back( lb_->pMassLabel );
    vars.push_back( lb_->pParticleIDLabel );

    vars_preReloc.push_back( lb_->pMassLabel_preReloc );
    vars_preReloc.push_back( lb_->pParticleIDLabel_preReloc );
    lb_->d_particleState.push_back(vars);
    lb_->d_particleState_preReloc.push_back(vars_preReloc);
  }

  sched->scheduleParticleRelocation(level,
                                    lb_->pXLabel_preReloc,
                                    lb_->d_particleState_preReloc,
                                    lb_->pXLabel,
                                    lb_->d_particleState,
                                    lb_->pParticleIDLabel,
                                    matls);

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
                          DataWarehouse* /*old_dw*/, DataWarehouse* new_dw)
{

  for( int p=0; p<patches->size(); ++p ){
    const Patch* patch = patches->get(p);
    const Point low  = patch->getNodePosition(patch->getCellLowIndex());
    const Point high = patch->getNodePosition(patch->getCellHighIndex());
    const Vector dx  = patch->dCell();

    cout << " low: " << low << " high: " << high << endl;

    for(int m = 0;m<matls->size();m++){
      srand(1);
      const int numParticles = 10;
      const int matl = matls->get(m);

      ParticleVariable<Point> px;
      ParticleVariable<double> pmass;
      ParticleVariable<long64> pids;

      ParticleSubset* subset = new_dw->createParticleSubset(numParticles,matl,patch);
      new_dw->allocateAndPut( px,    lb_->pXLabel,          subset );
      new_dw->allocateAndPut( pmass, lb_->pMassLabel,       subset );
      new_dw->allocateAndPut( pids,  lb_->pParticleIDLabel, subset );

      for( int i = 0; i < numParticles; ++i ){
        //const Point pos( (((float) rand()) / RAND_MAX * ( high.x() - low.x()-1) + low.x()),
        //                 (((float) rand()) / RAND_MAX * ( high.y() - low.y()-1) + low.y()),
       //                  (((float) rand()) / RAND_MAX * ( high.z() - low.z()-1) + low.z()) );

        const Point pos( ( ((double) i/numParticles)   *  ( high.x() - low.x() ) + low.x() + dx.x()/2 ),
                         ( ((float) rand()) / RAND_MAX) * ( high.y() - low.y() ) + low.y(),
                         ( ((float) rand()) / RAND_MAX) * ( high.z() - low.z() ) + low.z()  );
        px[i] = pos;
        pids[i] = patch->getID()*numParticles+i;
        pmass[i] = ((float) rand()) / RAND_MAX * 10;
        
        if (m_doOutput) {
           DOUTR( true, " initialization Patch " << patch->getID()
                        << ": i " << i
                        << ": ID "   << pids[i]
                        << ", pos "  << px[i]
                        << ", mass " << pmass[i] );
        }
      }
    }
  }
}

//______________________________________________________________________
//
void
ParticleTest1::schedTask1( const LevelP & level,
                           SchedulerP   & sched)
{
  Task* task = scinew Task("task1", this, &ParticleTest1::task1);

  if (m_NumGC == 0) {
    task->requires( Task::OldDW, lb_->pParticleIDLabel, m_gn, 0 );
    task->requires( Task::OldDW, lb_->pXLabel,          m_gn, 0 );
    task->requires( Task::OldDW, lb_->pMassLabel,       m_gn, 0 );
  }
  else if (m_NumGC > 0) {
    task->requires( Task::OldDW, lb_->pParticleIDLabel, m_gn, 0 );
    task->requires( Task::OldDW, lb_->pXLabel,          m_gac, m_NumGC+ONE );
    task->requires( Task::OldDW, lb_->pMassLabel,       m_gac, m_NumGC+ONE );
  }

  task->computes( lb_->pXLabel_1 );
  task->computes( lb_->pMassLabel_1 );

  sched->addTask(task, level->eachPatch(), m_materialManager->allMaterials());
}

//______________________________________________________________________
//
void ParticleTest1::task1(const ProcessorGroup * pg,
                          const PatchSubset    * patches,
                          const MaterialSubset * matls,
                          DataWarehouse        * old_dw,
                          DataWarehouse        * new_dw)
{
  for( int p=0; p<patches->size(); ++p ){
    const Patch* patch = patches->get(p);

    for( int m = 0; m<matls->size(); ++m ){
      int matl = matls->get(m);

      // Get the arrays of particle values to be changed
      constParticleVariable<Point> px;
      ParticleVariable<Point> px_1;

      constParticleVariable<double> pmass;
      ParticleVariable<double> pmass_1;
      constParticleVariable<long64> pids;

      //__________________________________
      // particle sets
      ParticleSubset* pset_gc = nullptr;
      ParticleSubset* pset_gn = old_dw->getParticleSubset(matl, patch);

      if (m_NumGC == 0) {
        pset_gc = pset_gn;
      }
      else if (m_NumGC >0 ) {
        pset_gc = old_dw->getParticleSubset( matl, patch, m_gac, m_NumGC+ONE, lb_->pXLabel );
      }

      int rank = pg->myRank();
//      DOUTR( true, "-Task1: pset_GC: " << *pset_gc);
//      DOUTR( true, "-Task1: pset_GN: " << *pset_gn);
      ParticleSubset* delset = scinew ParticleSubset(0,matl,patch);


      old_dw->get( pmass, lb_->pMassLabel,       pset_gc);
      old_dw->get( px,    lb_->pXLabel,          pset_gc);
      old_dw->get( pids,  lb_->pParticleIDLabel, pset_gn);

      new_dw->allocateAndPut( pmass_1, lb_->pMassLabel_1, pset_gn);
      new_dw->allocateAndPut( px_1,    lb_->pXLabel_1,    pset_gn);

      //__________________________________
      //
      auto i = pset_gn->begin();

      for(auto iter = pset_gc->begin(); iter != pset_gc->end(); iter++){
        particleIndex idx = *iter;

        if ( patch->containsPoint(px[idx]) ){
          // do nothing
          px_1[*i]    = px[idx];
          pmass_1[*i] = pmass[idx];
#if 0
          if (m_doOutput) {
           DOUTR( true, "] Task 1 Patch " << patch->getID()
                        << ": i " << *i
                        << ": ID "   << pids[*i]
                        << ", pos "  << px_1[*i].x()
                        << ", mass " << pmass_1[*i] );

          }
#endif
          ++i;
        }
      }
      new_dw->deleteParticles(delset);
    }
  }
}

//______________________________________________________________________
//
void
ParticleTest1::schedTask2( const LevelP & level,
                           SchedulerP   & sched)
{
  Task* task = scinew Task("task2", this, &ParticleTest1::task2);

  // set this in problemSetup.  0 is no ghost cells, 1 is all with 1 ghost
  // atound-node, and 2 mixes them
  if (m_NumGC == 0) {
    task->requires( Task::NewDW, lb_->pXLabel_1,        m_gn, 0);
    task->requires( Task::NewDW, lb_->pMassLabel,       m_gn, 0);
    task->requires( Task::NewDW, lb_->pParticleIDLabel, m_gn, 0);
  }

  else if (m_NumGC >0) {
    task->requires( Task::OldDW, lb_->pParticleIDLabel, m_gn, 0);
    task->requires( Task::OldDW, lb_->pXLabel,          m_gac, m_NumGC);
    task->requires( Task::NewDW, lb_->pXLabel_1,        m_gac, m_NumGC);
    task->requires( Task::OldDW, lb_->pMassLabel,       m_gac, m_NumGC);
  }

  task->computes(lb_->pXLabel_preReloc);
  task->computes(lb_->pMassLabel_preReloc);
  task->computes( lb_->pParticleIDLabel_preReloc );

  sched->addTask(task, level->eachPatch(), m_materialManager->allMaterials());
}

//______________________________________________________________________
//
void ParticleTest1::task2(const ProcessorGroup * pg,
                          const PatchSubset    * patches,
                          const MaterialSubset * matls,
                          DataWarehouse        * old_dw,
                          DataWarehouse        * new_dw)
{
  for( int p=0; p<patches->size(); ++p ){
    const Patch* patch = patches->get(p);

    for( int m = 0; m<matls->size(); ++m ){
      int matl = matls->get(m);

      constParticleVariable<Point> px_old;
      constParticleVariable<Point> px_1;
      ParticleVariable<Point>      px_new;

      constParticleVariable<double> pmass;
      ParticleVariable<double>      pmass_new;
      constParticleVariable<long64> pids;
      ParticleVariable<long64>      pids_new;

      //__________________________________
      // particle sets
      ParticleSubset* pset_gc = nullptr;
      ParticleSubset* pset_gn = old_dw->getParticleSubset(matl, patch);

      if ( m_NumGC == 0 ) {
        pset_gc = pset_gn;
      }
      else if ( m_NumGC > 0 ) {
        pset_gc = old_dw->getParticleSubset(matl, patch, m_gac, m_NumGC, lb_->pXLabel);
      }

      int rank = pg->myRank();
      DOUTR( true, "-Task2 pset_gc: " << *pset_gc);
      DOUTR( true, "-Task2 pset_GN: " << *pset_gn);

      old_dw->get( pmass,   lb_->pMassLabel,         pset_gc);
      old_dw->get( px_old,  lb_->pXLabel,            pset_gc );
      new_dw->get( px_1,    lb_->pXLabel_1,          pset_gc);
      old_dw->get( pids,    lb_->pParticleIDLabel,   pset_gn);

      new_dw->allocateAndPut( pmass_new, lb_->pMassLabel_preReloc,       pset_gn);
      new_dw->allocateAndPut( px_new,    lb_->pXLabel_preReloc,          pset_gn);
      new_dw->allocateAndPut( pids_new,  lb_->pParticleIDLabel_preReloc, pset_gn);

      pids_new.copyData(pids);


      //__________________________________
      //  Decrement the mass
      auto i = pset_gn->begin();

      for(auto iter = pset_gc->begin(); iter != pset_gc->end(); iter++){
        particleIndex idx = *iter;

        if ( patch->containsPoint(px_1[idx]) ){
          pmass_new[*i] = pmass[idx] *.9;

         ++i;
        }
      }

      //__________________________________
      // Move particles along +x axis
      i = pset_gn->begin();

      for(auto iter = pset_gc->begin(); iter != pset_gc->end(); iter++){
        particleIndex idx = *iter;

        if ( patch->containsPoint(px_1[idx]) ){
          //Point pos_new( px_1[idx].x() + 0.5, px_1[idx].y(), px_1[idx].z());
          //px_new[*i]  = pos_new;
          
          px_new[*i] = px_1[idx];

         ++i;
        }
      }
      //__________________________________
      //
      if (m_doOutput) {
        for(auto iter = pset_gn->begin(); iter != pset_gn->end(); iter++){
          particleIndex idx = *iter;

           DOUTR( true, "-Task2 Patch " << patch->getID()
                        << ": idx " << idx
                        << ": ID " << pids[idx]
                        << ": px " << px_new[idx].x()
                        << ", mass " << pmass_new[idx] );
          }
      }


      if( i != pset_gn->end() ){
        throw InternalError("task2:: there's a problem with the pset_gn iterator", __FILE__, __LINE__);
      }
    }
  }
}
