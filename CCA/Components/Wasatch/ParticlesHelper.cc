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
#include <Core/Grid/Box.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/Variables/VarTypes.h>

std::vector<Expr::Tag> Wasatch::ParticlesHelper::otherParticleTags_;

namespace Wasatch {
  
  //==================================================================
  
  ParticlesHelper::ParticlesHelper()
  {
    wasatchSync_ = false;
    using namespace Uintah;
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
  
  void ParticlesHelper::problem_setup()
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
    
  }

  //--------------------------------------------------------------------
  
  void ParticlesHelper::schedule_initialize (const Uintah::LevelP& level,
                                             Uintah::SchedulerP& sched)
  {
    // this task will allocate a particle subset and create particle positions
    Uintah::Task* task = scinew Uintah::Task("initialize particles",
                                             this, &ParticlesHelper::initialize);
    task->computes(pPosLabel_);
    task->computes(pIDLabel_);
    sched->addTask(task, level->eachPatch(), wasatch_->get_wasatch_materials());
  }

  //--------------------------------------------------------------------
  
  // this will create the particle subset
  void ParticlesHelper::initialize( const Uintah::ProcessorGroup*,
                                   const Uintah::PatchSubset* patches, const Uintah::MaterialSubset* matls,
                                   Uintah::DataWarehouse* old_dw, Uintah::DataWarehouse* new_dw)
  {
    using namespace Uintah;
    particleEqsSpec_->get("NumberOfInitialParticles",nParticles_);
    
    
    //____________________________________________
    /* In certain cases of particle initialization, a patch will NOT have any particles in it. For example,
     given a domain where x[0, 1] with patch0 [0,0.5] and patch1[0.5,1], assume one wants to initialize
     particles in the region x[0.55,1]. The process runnin patch will create the correct positions for these
     particles (since they are bounded by that patch, however, the process running on patch0 will
     create particles that are OUTSIDE its bounds. This is incorrect and usually leads to problems when
     performing particle interpolation. There are three remedies to this:
     (1) Relocate particles that are outside a given patch. This is NOT an option at the moment because
     particle relocation doesn't work with the initialization task graph
     (2) Delete the particles that are outside a given patch (on initialization only!)
     (3) Check particle initialization spec and create a subset with 0 particles on patches that should
     not have particles in them.
     The code that follows does option (3).
     */
    double xmin=-DBL_MAX, xmax=DBL_MAX,
    ymin=-DBL_MAX, ymax=DBL_MAX,
    zmin=-DBL_MAX, zmax=DBL_MAX;
    
    bool bounded=false;
    for( Uintah::ProblemSpecP exprParams = wasatch_->get_wasatch_spec()->findBlock("BasicExpression");
        exprParams != 0;
        exprParams = exprParams->findNextBlock("BasicExpression") )
    {
      if (exprParams->findBlock("ParticlePositionIC")) {
        Uintah::ProblemSpecP pICSpec = exprParams->findBlock("ParticlePositionIC");
        // check what type of bounds we are using: specified or patch based?
        std::string boundsType;
        pICSpec->getAttribute("bounds",boundsType);
        bounded = bounded || (boundsType == "SPECIFIED");
        if (bounded) {
          double lo = 0.0, hi = 1.0;
          pICSpec->findBlock("Bounds")->getAttribute("low", lo);
          pICSpec->findBlock("Bounds")->getAttribute("high", hi);
          // parse coordinate
          std::string coord;
          pICSpec->getAttribute("coordinate",coord);
          if (coord == "X") {xmin = lo; xmax = hi;}
          if (coord == "Y") {ymin = lo; ymax = hi;}
          if (coord == "Z") {zmin = lo; zmax = hi;}
        }
      }
    }
    
    for(int p=0;p<patches->size();p++){
      const Patch* patch = patches->get(p);
      for(int m = 0;m<matls->size();m++){
        int matl = matls->get(m);
        
        deleteSet_.insert( std::pair<int, ParticleSubset*>(patch->getID(), scinew ParticleSubset(0,matl,patch)));
        
        // If the particle position initialization is bounded, make sure that the bounds are within
        // this patch. If the bounds are NOT, then set the number of particles on this patch to 0.
        if (bounded) {
          Point low = patch->getBox().lower();
          Point high = patch->getBox().upper();
          if (   xmin >= high.x() || ymin >= high.y() || zmin >= high.z()
              || xmax <= low.x()  || ymax <= low.y()  || zmax <= low.z()  ) {
            // no particles will be created in this patch
            nParticles_ = 0;
          }
        }
        
        // create a subset with the correct number of particles. This will serve as the initial memory
        // block for particles
        ParticleSubset* subset = new_dw->createParticleSubset(nParticles_,matl,patch);
        
        // allocate memory for Uintah particle position and particle IDs
        ParticleVariable<Point>  ppos;
        ParticleVariable<long64> pid;
        new_dw->allocateAndPut(ppos,    pPosLabel_,           subset);
        new_dw->allocateAndPut(pid,    pIDLabel_,           subset);
        for (int i=0; i < nParticles_; i++) {
          pid[i] = i + patch->getID() * nParticles_;
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
    sched->addTask(task, level->eachPatch(), wasatch_->get_wasatch_materials());
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
    sched->addTask(task, level->eachPatch(), wasatch_->get_wasatch_materials());
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
    const MaterialSet* const materials = wasatch_->get_wasatch_materials();
    
    // first go through the list of particle expressions and check whether Uintah manages those
    // or note. We need this for particle relocation.
    vector<const VarLabel*> otherParticleVarLabels;
    vector<Expr::Tag>::iterator tagIter = otherParticleTags_.begin();
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
    
    otherParticleVarLabels.push_back(pIDLabel_);
    
    vector< vector<const VarLabel*> > otherParticleVars;
    for (int m = 0; m < materials->size(); m++) {
      otherParticleVars.push_back(otherParticleVarLabels);
    }
    sched->scheduleParticleRelocation(level, pPosLabel_, otherParticleVars, materials);
    
    // clean the delete set
    Task* task = scinew Task("cleanup deleteset",
                              this, &ParticlesHelper::clear_deleteset);
    sched->addTask(task, level->eachPatch(), wasatch_->get_wasatch_materials());
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
  ParticlesHelper::sync_with_wasatch( Wasatch* const wasatch )
  {
    wasatch_ = wasatch;
    wasatchSync_ = true;
  }
  
  //------------------------------------------------------------------
  
  void
  ParticlesHelper::add_particle_variable(const Expr::Tag& varTag )
  {
    otherParticleTags_.push_back(varTag);
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
    sched->addTask(task, level->eachPatch(), wasatch_->get_wasatch_materials());
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