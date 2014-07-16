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

//-- Wasatch includes --//
#include <CCA/Components/Wasatch/Wasatch.h>

//-- Uintah Includes --//
#include <Core/Grid/Box.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/Variables/VarTypes.h>

std::vector<std::string> Uintah::ParticlesHelper::otherParticleVarNames_;

namespace Uintah {
  
  //==================================================================
  
  ParticlesHelper::ParticlesHelper() :
  isValidState_(false),
  pPerCell_(0.0),
  maxParticles_(0x10000u) // 2^32 maximum particles per patch
  {
    pPosLabel_ = VarLabel::create("p.x",
                                  ParticleVariable<Uintah::Point>::getTypeDescription(),
                                  SCIRun::IntVector(0,0,0),
                                  VarLabel::PositionVariable );
    pIDLabel_ = Uintah::VarLabel::create("p.particleID",
                                         ParticleVariable<long64>::getTypeDescription());
    
    destroyMe_.push_back(pPosLabel_);
    destroyMe_.push_back(pIDLabel_);
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
  
  //------------------------------------------------------------------
  
  void ParticlesHelper::problem_setup(Uintah::ProblemSpecP particleEqsSpec)
  {
    using namespace Uintah;
    particleEqsSpec_ = particleEqsSpec;

    //
    // set the position varlabels
    particleEqsSpec_->get("ParticlesPerCell",pPerCell_);
    particleEqsSpec_->get("MaximumParticles",maxParticles_);
    
    ProblemSpecP pPosSpec = particleEqsSpec_->findBlock("ParticlePosition");

    if (!pPosSpec) {
      std::ostringstream msg;
      msg << "ParticlesHelper Error: It looks like your particle specification does not include an xml block for ParticlePosition. In order for the \
      ParticlesHelper class to work properly, you must specify a ParticlePosition xml block with x, y, and z attributes \
      denoting the particle position varlabels." << std::endl;
      throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
    }
    
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
    
  }

  //--------------------------------------------------------------------
  
  void ParticlesHelper::schedule_initialize (const Uintah::LevelP& level,
                                             Uintah::SchedulerP& sched)
  {
    if (!isValidState_) {
      std::ostringstream msg;
      msg << "ParticlesHelper error: you must call problem_setup and set_materials prior to initializing particles!" << std::endl;
      throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
    }
    // this task will allocate a particle subset and create particle positions
    Uintah::Task* task = scinew Uintah::Task("initialize particles memory",
                                             this, &ParticlesHelper::initialize);
    task->computes(pPosLabel_);
    task->computes(pIDLabel_);
    sched->addTask(task, level->eachPatch(), materials_);
  }

  //--------------------------------------------------------------------
  
  // this will create the particle subset
  void ParticlesHelper::initialize( const Uintah::ProcessorGroup*,
                                   const Uintah::PatchSubset* patches, const Uintah::MaterialSubset* matls,
                                   Uintah::DataWarehouse* old_dw, Uintah::DataWarehouse* new_dw)
  {
    using namespace Uintah;
    particleEqsSpec_->get("ParticlesPerCell",pPerCell_);
    
    for(int p=0;p<patches->size();p++){
      const Patch* patch = patches->get(p);
      for(int m = 0;m<matls->size();m++){
        int matl = matls->get(m);
        
        deleteSet_.insert( std::pair<int, ParticleSubset*>(patch->getID(), scinew ParticleSubset(0,matl,patch)));
        
        // create a subset with the correct number of particles. This will serve as the initial memory
        // block for particles
        int nParticles = pPerCell_ * patch->getNumCells();
        ParticleSubset* subset = new_dw->createParticleSubset(nParticles,matl,patch);
        
        // allocate memory for Uintah particle position and particle IDs
        ParticleVariable<Point>  ppos;
        ParticleVariable<long64> pid;
        new_dw->allocateAndPut(ppos,    pPosLabel_,           subset);
        new_dw->allocateAndPut(pid,    pIDLabel_,           subset);
        for (int i=0; i < nParticles; i++) {
          pid[i] = i + patch->getID();
        }
      }
    }
  }

  //--------------------------------------------------------------------
  
  void ParticlesHelper::schedule_restart_initialize (const Uintah::LevelP& level,
                                                     Uintah::SchedulerP& sched)
  {
    // this task will allocate a particle subset and create particle positions
    Uintah::Task* task = scinew Uintah::Task("restart initialize particles",
                                             this, &ParticlesHelper::restart_initialize);
    sched->addTask(task, level->eachPatch(), materials_);
  }

  //--------------------------------------------------------------------
  
  // This is needed to reallocate memory for the deleteset
  void ParticlesHelper::restart_initialize( const Uintah::ProcessorGroup*,
                                            const Uintah::PatchSubset* patches, const Uintah::MaterialSubset* matls,
                                             Uintah::DataWarehouse* old_dw, Uintah::DataWarehouse* new_dw)
  {
    using namespace Uintah;

    if (!deleteSet_.empty())
    {
      return;
    }
    
    for(int p=0;p<patches->size();p++){
      const Patch* patch = patches->get(p);
      for(int m = 0;m<matls->size();m++){
        int matl = matls->get(m);
        deleteSet_.insert( std::pair<int, ParticleSubset*>(patch->getID(), scinew ParticleSubset(0,matl,patch)));
      }
    }
  }
  
  //--------------------------------------------------------------------
  
  void ParticlesHelper::schedule_delete_outside_particles(const Uintah::LevelP& level,
                                                          Uintah::SchedulerP& sched)
  {
    using namespace Uintah;
    Uintah::Task* task = scinew Uintah::Task("delete outside particles",
                                             this, &ParticlesHelper::delete_outside_particles);
    task->modifies(pXLabel_);
    task->modifies(pYLabel_);
    task->modifies(pZLabel_);
    sched->addTask(task, level->eachPatch(), materials_);
  }
  
  //--------------------------------------------------------------------
  
  // this will delete outside particles
  void ParticlesHelper::delete_outside_particles(const Uintah::ProcessorGroup*,
                                                 const Uintah::PatchSubset* patches, const Uintah::MaterialSubset* matls,
                                                 Uintah::DataWarehouse* old_dw, Uintah::DataWarehouse* new_dw)
  {
    using namespace Uintah;
    for(int p=0;p<patches->size();p++){
      const Patch* patch = patches->get(p);
      for(int m = 0; m<matls->size(); m++){
        const int matl = matls->get(m);
        ParticleSubset* pset = new_dw->getParticleSubset(matl, patch);
        ParticleSubset* delset = deleteSet_[patch->getID()];

        Point low  = patch->getBox().lower();
        Point high = patch->getBox().upper();;
        
        // Wasatch particle positions
        ParticleVariable<double> px;
        ParticleVariable<double> py;
        ParticleVariable<double> pz;
        
        new_dw->getModifiable(px,    pXLabel_,                  pset);
        new_dw->getModifiable(py,    pYLabel_,                  pset);
        new_dw->getModifiable(pz,    pZLabel_,                  pset);
        
        
        for(ParticleSubset::iterator iter = pset->begin();
            iter != pset->end(); iter++)
        {
          particleIndex idx = *iter;
          // delete particles that are outside this patch
          if (   px[idx] >= high.x() || px[idx] <= low.x()
              || py[idx] >= high.y() || py[idx] <= low.y()
              || pz[idx] >= high.z() || pz[idx] <= low.z()){
            px[idx] = low.x() + (high.x() - low.x())/2.0;
            py[idx] = low.y() + (high.y() - low.y())/2.0;
            pz[idx] = low.z() + (high.z() - low.z())/2.0;;
            delset->addParticle(idx);
          }
        } // particles
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
    
    // first go through the list of particle expressions and check whether Uintah manages those
    // or note. We need this for particle relocation.
    vector<const VarLabel*> otherParticleVarLabels;
    vector<string>::iterator varNameIter = otherParticleVarNames_.begin();
    for (; varNameIter != otherParticleVarNames_.end(); ++varNameIter) {
      if (VarLabel::find( *varNameIter ) ) {
        VarLabel* theVarLabel = VarLabel::find( *varNameIter );
        
        if (std::find(otherParticleVarLabels.begin(), otherParticleVarLabels.end(),theVarLabel) == otherParticleVarLabels.end())
        {
          otherParticleVarLabels.push_back(VarLabel::find( *varNameIter ));
        }
      }
    }

    // add the particle ID label!
    otherParticleVarLabels.push_back(pIDLabel_);
    
    vector< vector<const VarLabel*> > otherParticleVars;
    for (int m = 0; m < materials_->size(); m++) {
      otherParticleVars.push_back(otherParticleVarLabels);
    }
    sched->scheduleParticleRelocation(level, pPosLabel_, otherParticleVars, materials_);
    
    // clean the delete set
    Task* task = scinew Task("cleanup deleteset",
                              this, &ParticlesHelper::clear_deleteset);
    sched->addTask(task, level->eachPatch(), materials_);
  }
  
  //--------------------------------------------------------------------
  
  // this will create the particle subset
  void ParticlesHelper::clear_deleteset(const Uintah::ProcessorGroup*,
                                         const Uintah::PatchSubset* patches, const Uintah::MaterialSubset* matls,
                                         Uintah::DataWarehouse* old_dw, Uintah::DataWarehouse* new_dw)
  {
    using namespace Uintah;
    for(int p=0;p<patches->size();p++){
      const Patch* patch = patches->get(p);
      for(int m = 0; m<matls->size(); m++){
        const int matl = matls->get(m);
        
        ParticleSubset* existingDelset = deleteSet_[patch->getID()];
        if (existingDelset->numParticles() > 0)
        {
          ParticleSubset* delset = scinew ParticleSubset(0,matl, patch);
          deleteSet_[patch->getID()] = delset;
        }
      }
    }
  }
  
  //------------------------------------------------------------------
  
  void
  ParticlesHelper::add_particle_variable(const std::string& varName )
  {
    otherParticleVarNames_.push_back(varName);
  }
  
  //--------------------------------------------------------------------
  // this task will sync particle position with wasatch computed values
  void ParticlesHelper::schedule_transfer_particle_ids(const Uintah::LevelP& level,
                                                       Uintah::SchedulerP& sched)
  {
    using namespace Uintah;
    Uintah::Task* task = scinew Uintah::Task("transfer particles IDs",
                                             this, &ParticlesHelper::transfer_particle_ids);
    task->computes(pIDLabel_);
    task->requires(Task::OldDW, pIDLabel_, Uintah::Ghost::None, 0);
    sched->addTask(task, level->eachPatch(), materials_);
  }
  
  //--------------------------------------------------------------------
  
  void ParticlesHelper::transfer_particle_ids(const Uintah::ProcessorGroup*,
                                              const Uintah::PatchSubset* patches, const Uintah::MaterialSubset* matls,
                                              Uintah::DataWarehouse* old_dw, Uintah::DataWarehouse* new_dw )
  {
    using namespace Uintah;
    for(int p=0;p<patches->size();p++){
      const Patch* patch = patches->get(p);
      for(int m = 0; m<matls->size(); m++){
        const int matl = matls->get(m);
        ParticleSubset* pset = old_dw->getParticleSubset(matl, patch);
        ParticleVariable<long64> pid;
        constParticleVariable<long64> pidOld;
        new_dw->allocateAndPut(pid,    pIDLabel_,          pset);
        old_dw->get(pidOld,pIDLabel_,pset);
        pid.copyData(pidOld);
      }
    }
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
    sched->addTask(task, level->eachPatch(), materials_);
  }
  
  //--------------------------------------------------------------------
  void ParticlesHelper::sync_particle_position(const Uintah::ProcessorGroup*,
                                               const Uintah::PatchSubset* patches, const Uintah::MaterialSubset* matls,
                                               Uintah::DataWarehouse* old_dw, Uintah::DataWarehouse* new_dw, const bool initialization)
  {
    using namespace Uintah;
    for(int p=0;p<patches->size();p++){
      const Patch* patch = patches->get(p);
      for(int m = 0; m<matls->size(); m++){
        const int matl = matls->get(m);
        ParticleSubset* pset = initialization ? new_dw->getParticleSubset(matl, patch) : old_dw->getParticleSubset(matl, patch);
        const int numParticles =pset->numParticles();
        
        //ParticleSubset* delset = scinew ParticleSubset(0,matl,patch);
        new_dw->deleteParticles(deleteSet_[patch->getID()]);
        
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