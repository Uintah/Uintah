/**
 *  \file   ParticlesHelper.cc
 *  \date   June, 2014
 *  \author "Tony Saad"
 *
 *
 * The MIT License
 *
 * Copyright (c) 2013-2014 The University of Utah
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

#include "ParticlesHelper.h"

//-- Boost includes --//
#include <boost/foreach.hpp>

//-- ExprLib includes --//
#include <expression/Expression.h>
#include <expression/ExpressionFactory.h>

//-- Wasatch includes --//
#include <CCA/Components/Wasatch/FieldAdaptor.h>
#include <CCA/Components/Wasatch/Wasatch.h>
#include <CCA/Components/Wasatch/FieldTypes.h>
#include <CCA/Components/Wasatch/Expressions/ReductionBase.h>
#include <CCA/Components/Wasatch/Expressions/Reduction.h>
#include <CCA/Components/Wasatch/ParseTools.h>

//-- Uintah Includes --//
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/Variables/VarTypes.h>

namespace SS = SpatialOps::structured;

std::vector<const Expr::Tag> Wasatch::ParticlesHelper::otherParticleTags_;

namespace Wasatch {
  
  //==================================================================

  ParticlesHelper::ParticlesHelper()
  {
    wasatchSync_ = false;
    pPosLabel_ = Uintah::VarLabel::create("p.x",
                                         Uintah::ParticleVariable<Uintah::Point>::getTypeDescription(),
                                         SCIRun::IntVector(0,0,0),
                                         Uintah::VarLabel::PositionVariable );
    destroyMe_.push_back(pPosLabel_);
  }

  //------------------------------------------------------------------

  ParticlesHelper&
  ParticlesHelper::self()
  {
    static ParticlesHelper partHelp;
    return partHelp;
  }

  //------------------------------------------------------------------

  ParticlesHelper::~ParticlesHelper()
  {
    using namespace Uintah;
    std::vector<const VarLabel*>::iterator it = destroyMe_.begin();
    while (it!=destroyMe_.end()) {
      Uintah::VarLabel::destroy(*it);
      ++it;
    }
  }
  
  //--------------------------------------------------------------------
  
  void ParticlesHelper::schedule_initialize_particles(const Uintah::LevelP& level,
                                                      Uintah::SchedulerP& sched)
  {
    using namespace Uintah;
    if( !wasatchSync_ ){
      std::ostringstream msg;
      msg << "ParticlesHelper error: must call sync_with_wasatch() prior to initializing particles!" << std::endl;
      throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
    }

    //
    // set the position varlabels
    particleEqsSpec_ = wasatch_->get_wasatch_spec()->findBlock("ParticleTransportEquations");
    particleEqsSpec_->get("NumberOfInitialParticles",nParticles_);

    ProblemSpecP pPosSpec = particleEqsSpec_->findBlock("ParticlePosition");
    std::string px, py, pz;
    pPosSpec->getAttribute("x",px);
    pPosSpec->getAttribute("y",py);
    pPosSpec->getAttribute("z",pz);

    pXLabel_ = VarLabel::find(px);
    pYLabel_ = VarLabel::find(py);
    pZLabel_ = VarLabel::find(pz);

    if (!pXLabel_) {
      pXLabel_ = Uintah::VarLabel::create(px,
                                          Uintah::ParticleVariable<double>::getTypeDescription() );
      destroyMe_.push_back(pXLabel_);
    }
    if (!pYLabel_) {
      pYLabel_ = Uintah::VarLabel::create(py,
                                          Uintah::ParticleVariable<double>::getTypeDescription() );
      destroyMe_.push_back(pYLabel_);
    }
    if (!pZLabel_) {
      pZLabel_ = Uintah::VarLabel::create(pz,
                                          Uintah::ParticleVariable<double>::getTypeDescription() );
      destroyMe_.push_back(pZLabel_);
    }

    //

    // this task will allocate a particle subset and create particle positions
    Uintah::Task* task = scinew Uintah::Task("initialize particles",
                                             this, &ParticlesHelper::initialize_particles);
    task->computes(pPosLabel_);
    sched->addTask(task, level->eachPatch(), wasatch_->get_wasatch_materials());
  }
  
  //--------------------------------------------------------------------

  // this will create the particle subset
  void ParticlesHelper::initialize_particles(const Uintah::ProcessorGroup*,
                                     const Uintah::PatchSubset* patches, const Uintah::MaterialSubset* matls,
                                     Uintah::DataWarehouse* old_dw, Uintah::DataWarehouse* new_dw)
  {
    using namespace Uintah;
    particleEqsSpec_ = wasatch_->get_wasatch_spec()->findBlock("ParticleTransportEquations");
    particleEqsSpec_->get("NumberOfInitialParticles",nParticles_);
    
    for(int p=0;p<patches->size();p++){
      const Patch* patch = patches->get(p);
      for(int m = 0;m<matls->size();m++){
        int matl = matls->get(m);
        ParticleVariable<Point> ppos;
        ParticleSubset* subset = new_dw->createParticleSubset(nParticles_,matl,patch);
        new_dw->allocateAndPut(ppos,    pPosLabel_,           subset);
      }
    }
  }

  //------------------------------------------------------------------
  
  void
  ParticlesHelper::schedule_relocate_particles( const Uintah::LevelP& level,
                                                Uintah::SchedulerP& sched    )
  {
    using namespace std;
    using namespace Uintah;
    const MaterialSet* const materials = wasatch_->get_wasatch_materials();
    
    // first go through the list of particle expressions and check whether Uintah manages those
    // or note. We need this for particle relocation.
    vector<const VarLabel*> otherParticleVarLabels;
    vector<const Expr::Tag>::iterator tagIter = otherParticleTags_.begin();
    while (tagIter != otherParticleTags_.end())
    {
      if (VarLabel::find( (*tagIter).name() ) ) {
        VarLabel* theVarLabel = VarLabel::find( (*tagIter).name() );
        
        if (std::find(otherParticleVarLabels.begin(), otherParticleVarLabels.end(),theVarLabel) == otherParticleVarLabels.end())
        {
          otherParticleVarLabels.push_back(VarLabel::find( (*tagIter).name() ));
        }
      }
      ++tagIter;
    }
    
    vector< vector<const VarLabel*> > otherParticleVars;
    for (int m = 0; m < materials->size(); m++) {
      otherParticleVars.push_back(otherParticleVarLabels);
    }
    sched->scheduleParticleRelocation(level, pPosLabel_, otherParticleVars, materials);
  }

  //------------------------------------------------------------------
  
  void
  ParticlesHelper::sync_with_wasatch( Wasatch* const wasatch )
  {
    wasatch_ = wasatch;
    wasatchSync_ = true;
  }

  //------------------------------------------------------------------
  
  void
  ParticlesHelper::add_particle_variable( const Expr::Tag& varTag )
  {
    otherParticleTags_.push_back(varTag);
  }

  //--------------------------------------------------------------------
  // this task will sync particle position with wasatch computed values
  void ParticlesHelper::schedule_sync_particle_position(const Uintah::LevelP& level,
                                        Uintah::SchedulerP& sched, const bool initialization)
  {
    using namespace Uintah;
    Uintah::Task* task = scinew Uintah::Task("sync particles",
                                             this, &ParticlesHelper::sync_particle_position, initialization);
    if (initialization) {
      task->modifies(pPosLabel_);
    } else {
      task->computes(pPosLabel_);
    }
    
    task->requires(Task::NewDW, pXLabel_, Uintah::Ghost::None, 0);
    task->requires(Task::NewDW, pYLabel_, Uintah::Ghost::None, 0);
    task->requires(Task::NewDW, pZLabel_, Uintah::Ghost::None, 0);
    sched->addTask(task, level->eachPatch(), wasatch_->get_wasatch_materials());
  }
  
  //--------------------------------------------------------------------
  void ParticlesHelper::sync_particle_position(const Uintah::ProcessorGroup*,
                               const Uintah::PatchSubset* patches, const Uintah::MaterialSubset* matls,
                               Uintah::DataWarehouse* old_dw, Uintah::DataWarehouse* new_dw, const bool initialization)
  {
    using namespace Uintah;
    for(int p=0;p<patches->size();p++){
      const Patch* patch = patches->get(p);
      Point low = patch->cellPosition(patch->getCellLowIndex());
      Point high = patch->cellPosition(patch->getCellHighIndex());
      for(int m = 0; m<matls->size(); m++){
        const int matl = matls->get(m);
        ParticleSubset* pset = initialization ? new_dw->getParticleSubset(matl, patch) : old_dw->getParticleSubset(matl, patch);
        const int numParticles =pset->numParticles();
        
        ParticleSubset* delset = scinew ParticleSubset(0,matl,patch);
        new_dw->deleteParticles(delset);
        
        ParticleVariable<Point> ppos; // Uintah particle position

        // Wasatch particle positions
        constParticleVariable<double> px;
        constParticleVariable<double> py;
        constParticleVariable<double> pz;
        
        new_dw->get(px,    pXLabel_,                  pset);
        new_dw->get(py,    pYLabel_,                  pset);
        new_dw->get(pz,    pZLabel_,                  pset);
        if (initialization) {
          //          ParticleVariable<Point> pxtmp;
          //          new_dw->allocateTemporary(pxtmp, pset);
          //          new_dw->put(pxtmp, pPosLabel, true);
          new_dw->getModifiable(ppos,    pPosLabel_,          pset);
        } else {
          new_dw->allocateAndPut(ppos,    pPosLabel_,          pset);
        }
        
        for (int i = 0; i < numParticles; i++) {
          ppos[i].x(px[i]);
          ppos[i].y(py[i]);
          ppos[i].z(pz[i]);
        }
      }
    }
  }

//  //--------------------------------------------------------------------
//  
//  void ParticlesHelper::schedule_add_particles( const Uintah::LevelP& level,
//                                                Uintah::SchedulerP& sched )
//  {
//    // this task will allocate a particle subset and create particle positions
//    Uintah::Task* task = scinew Uintah::Task( "add particles",
//                                              this, &ParticlesHelper::add_particles );
//    sched->addTask(task, level->eachPatch(), wasatch_->get_wasatch_materials());
//  }
//  
//  //--------------------------------------------------------------------
//  void ParticlesHelper::add_particles( const Uintah::ProcessorGroup*,
//                                       const Uintah::PatchSubset* patches, const Uintah::MaterialSubset* matls,
//                                       Uintah::DataWarehouse* old_dw, Uintah::DataWarehouse* new_dw )
//  {
//    using namespace Uintah;
//    for(int p=0; p<patches->size(); p++)
//    {
//      const Patch* patch = patches->get(p);
//      for(int m = 0;m<matls->size();m++)
//      {
//        int matl = matls->get(m);
//        ParticleSubset* pset = new_dw->haveParticleSubset(matl,patch) ? new_dw->getParticleSubset(matl, patch) : old_dw->getParticleSubset(matl, patch);
//        pset->addParticles(20);
//      }
//    }
//  }

  //--------------------------------------------------------------------

} /* namespace Wasatch */