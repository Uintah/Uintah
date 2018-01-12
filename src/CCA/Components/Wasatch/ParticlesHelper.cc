/**
 *  \file   ParticlesHelper.cc
 *  \date   June, 2014
 *  \author "Tony Saad"
 *
 *
 * The MIT License
 *
 * Copyright (c) 2013-2018 The University of Utah
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

/**************************************************************************************
 _______    ______   _______  ________  ______   ______   __        ________   ______
 |       \  /      \ |       \|        \|      \ /      \ |  \      |        \ /      \
 | $$$$$$$\|  $$$$$$\| $$$$$$$\\$$$$$$$$ \$$$$$$|  $$$$$$\| $$      | $$$$$$$$|  $$$$$$\
 | $$__/ $$| $$__| $$| $$__| $$  | $$     | $$  | $$   \$$| $$      | $$__    | $$___\$$
 | $$    $$| $$    $$| $$    $$  | $$     | $$  | $$      | $$      | $$  \    \$$    \
 | $$$$$$$ | $$$$$$$$| $$$$$$$\  | $$     | $$  | $$   __ | $$      | $$$$$    _\$$$$$$\
 | $$      | $$  | $$| $$  | $$  | $$    _| $$_ | $$__/  \| $$_____ | $$_____ |  \__| $$
 | $$      | $$  | $$| $$  | $$  | $$   |   $$ \ \$$    $$| $$     \| $$     \ \$$    $$
 \ $$      \ $$  \ $$ \$$   \$$   \$$    \$$$$$$  \$$$$$$  \$$$$$$$$ \$$$$$$$$  \$$$$$$
 **************************************************************************************/

#include <CCA/Components/Wasatch/ParticlesHelper.h>

//-- Uintah Includes --//
#include <Core/Grid/Box.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/BoundaryConditions/BCDataArray.h>
#include <Core/Grid/BoundaryConditions/BoundCond.h>

std::vector<std::string> Uintah::ParticlesHelper::needsRelocation_;
std::vector<std::string> Uintah::ParticlesHelper::needsBC_;
std::map<std::string, std::map<int, std::vector<int> > > Uintah::ParticlesHelper::bndParticlesMap_;
std::string Uintah::ParticlesHelper::pPosName_;
std::string Uintah::ParticlesHelper::pIDName_;

namespace Uintah {
  
  //==================================================================
  
  const std::vector<std::string>&
  ParticlesHelper::get_relocatable_particle_varnames()
  {
    return needsRelocation_;
  }
  
  //------------------------------------------------------------------
  
  void
  ParticlesHelper::mark_for_relocation(const std::string& varName )
  {
    using namespace std;
    // disallow duplicates
    vector<string>::iterator it = find(needsRelocation_.begin(), needsRelocation_.end(), varName);
    if (it == needsRelocation_.end()) {
      needsRelocation_.push_back(varName);
    }
  }

  //------------------------------------------------------------------
  
  void ParticlesHelper::initialize_internal(const int matSize)
  {
    // allocate proper sizes for delete sets and particle IDs
    if (lastPIDPerMaterialPerPatch_.size() == 0) {
      for (int m = 0; m<matSize; ++m ) {
        std::map<int,long64> lastPIDPerPatch;
        lastPIDPerMaterialPerPatch_.push_back(lastPIDPerPatch);
      }
    }
    if (deleteSets_.size() == 0) {
      for (int m = 0; m<matSize; ++m ) {
        std::map<int,ParticleSubset*> thisMaterialDeleteSet;
        deleteSets_.push_back(thisMaterialDeleteSet);
      }
    }
  }
  
  //------------------------------------------------------------------
  
  void
  ParticlesHelper::needs_boundary_condition(const std::string& varName )
  {
    using namespace std;

    // disallow addition of particle position vector and particle ID to the list of
    // boundary conditions for particles. Those are handled internally.
    if (varName == pPosName_ ||
        varName == pIDName_     ) {
      return;
    }

    // disallow duplicates
    vector<string>::iterator it = find(needsBC_.begin(), needsBC_.end(), varName);
    if (it == needsBC_.end()) {
      needsBC_.push_back(varName);
    }
  }

  //------------------------------------------------------------------
  
  ParticlesHelper::ParticlesHelper()
    : isValidState_(false),
      pPerCell_(0.0),
      maxParticles_(0x10000u) // 2^32 ~ 4.3 billion particles per patch - maximum
  {
    pXLabel_ = nullptr;
    pYLabel_ = nullptr;
    pZLabel_ = nullptr;
    materials_ = nullptr;

    // delta t
    VarLabel* nonconstDelT =
      VarLabel::create(delT_name, delt_vartype::getTypeDescription() );
    nonconstDelT->allowMultipleComputes();
    delTLabel_ = nonconstDelT;
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

    VarLabel::destroy(delTLabel_);
  }
  
  //------------------------------------------------------------------
  
  void ParticlesHelper::problem_setup(Uintah::ProblemSpecP uintahSpec,
                                      Uintah::ProblemSpecP particleEqsSpec)
  {
    using namespace Uintah;
    
    ProblemSpecP uintahPPosSpec = uintahSpec->findBlock("ParticlePosition");
    if (uintahPPosSpec) uintahPPosSpec->getAttribute("label",pPosName_);
    else pPosName_ = "p.x";
    Uintah::VarLabel::setParticlePositionName(pPosName_);
    
    pPosLabel_ = VarLabel::create(pPosName_,
                                  ParticleVariable<Uintah::Point>::getTypeDescription(),
                                  Uintah::IntVector(0,0,0),
                                  VarLabel::PositionVariable );
    pIDLabel_ = Uintah::VarLabel::create("p.particleID",
                                         ParticleVariable<long64>::getTypeDescription());
    
    pIDName_  = pIDLabel_->getName();
    
    destroyMe_.push_back(pPosLabel_);
    destroyMe_.push_back(pIDLabel_);

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
    
    needs_boundary_condition(px);
    needs_boundary_condition(py);
    needs_boundary_condition(pz);
    
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
    parse_boundary_conditions(level, sched);
  }
  
  //--------------------------------------------------------------------
  
  // this will create the particle subset
  void ParticlesHelper::initialize( const Uintah::ProcessorGroup*,
                                    const Uintah::PatchSubset* patches,
                                    const Uintah::MaterialSubset* matls,
                                    Uintah::DataWarehouse* old_dw,
                                    Uintah::DataWarehouse* new_dw )
  {
    using namespace Uintah;
    
    initialize_internal(matls->size());
    
    particleEqsSpec_->get("ParticlesPerCell",pPerCell_);
    
    for( int m = 0; m<matls->size(); ++m ){
      const int matl = matls->get(m);
      std::map<int,long64>& lastPIDPerPatch = lastPIDPerMaterialPerPatch_[m];
      std::map<int,ParticleSubset*>& thisMaterialDeleteSet = deleteSets_[m];
      for( int p=0; p<patches->size(); ++p ){
        const Patch* patch = patches->get(p);
        const int patchID = patch->getID();
        
        lastPIDPerPatch.insert( std::pair<int, long64>(patchID, 0 ) );
        thisMaterialDeleteSet.insert( std::pair<int, ParticleSubset*>(patchID, scinew ParticleSubset(0,matl,patch)));
        
        // create a subset with the correct number of particles. This will serve as the initial memory
        // block for particles
        const int nParticles = pPerCell_ * patch->getNumCells();
        ParticleSubset* subset = new_dw->createParticleSubset(nParticles,matl,patch);
        
        // allocate memory for Uintah particle position and particle IDs
        ParticleVariable<Point>  ppos;
        ParticleVariable<long64> pid;
        new_dw->allocateAndPut(ppos,    pPosLabel_,           subset);
        new_dw->allocateAndPut(pid,    pIDLabel_,           subset);
        for (int i=0; i < nParticles; i++) {
          pid[i] = i + patchID;
        }
        lastPIDPerPatch[patchID] = nParticles > 0 ? pid[nParticles-1] : 0;
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
    task->requires(Task::OldDW, pIDLabel_, Uintah::Ghost::None, 0);
    sched->addTask(task, level->eachPatch(), materials_);
    parse_boundary_conditions(level, sched);
  }
  
  //--------------------------------------------------------------------
  
  // This is needed to reallocate memory for the deleteset
  void ParticlesHelper::restart_initialize( const Uintah::ProcessorGroup*,
                                           const Uintah::PatchSubset* patches, const Uintah::MaterialSubset* matls,
                                           Uintah::DataWarehouse* old_dw, Uintah::DataWarehouse* new_dw)
  {
    using namespace Uintah;
    initialize_internal(matls->size());
    
    for(int m = 0;m<matls->size();m++){
      int matl = matls->get(m);
      std::map<int,long64>& lastPIDPerPatch = lastPIDPerMaterialPerPatch_[m];
      std::map<int,ParticleSubset*>& thisMaterialDeleteSet = deleteSets_[m];
      for(int p=0;p<patches->size();p++){
        const Patch* patch = patches->get(p);
        const int patchID = patch->getID();
        
        lastPIDPerPatch.insert( std::pair<int, long64>(patchID, 0 ) );
        thisMaterialDeleteSet.insert( std::pair<int, ParticleSubset*>(patchID, scinew ParticleSubset(0,matl,patch)));
        
        ParticleSubset* pset = new_dw->haveParticleSubset(matl,patch) ? new_dw->getParticleSubset(matl, patch) : old_dw->getParticleSubset(matl, patch);
        
        constParticleVariable<long64> pids;
        old_dw->get(pids, pIDLabel_, pset);
        long64 largestPID = 0;
        for( ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); ++iter ){
          particleIndex idx = *iter;
          if (largestPID < pids[idx]) {
            largestPID = pids[idx];
          }
        }
        lastPIDPerPatch[patchID] = largestPID;
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
    for( int m = 0; m<matls->size(); ++m ){
      const int matl = matls->get(m);
      std::map<int, ParticleSubset*> thisMatDelSet;
      for( int p=0; p<patches->size(); ++p ){
        const Patch* patch = patches->get(p);
        ParticleSubset* pset = new_dw->getParticleSubset(matl, patch);
        ParticleSubset* delset = thisMatDelSet[patch->getID()];
        
        const Point low  = patch->getBox().lower();
        const Point high = patch->getBox().upper();;
        
        // Component particle positions
        ParticleVariable<double> px;
        ParticleVariable<double> py;
        ParticleVariable<double> pz;
        
        new_dw->getModifiable( px, pXLabel_, pset);
        new_dw->getModifiable( py, pYLabel_, pset);
        new_dw->getModifiable( pz, pZLabel_, pset);

        for( ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); ++iter ){
          particleIndex idx = *iter;
          // delete particles that are outside this patch
          if(   px[idx] >= high.x() || px[idx] <= low.x()
             || py[idx] >= high.y() || py[idx] <= low.y()
             || pz[idx] >= high.z() || pz[idx] <= low.z() )
          {
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
                                                Uintah::SchedulerP& sched )
  {
    using namespace std;
    using namespace Uintah;
    
    // first go through the list of particle expressions and check whether Uintah manages those
    // or not. We need this for particle relocation.
    vector<const VarLabel*> otherParticleVarLabels;
    const vector<string>& otherParticleVarNames = ParticlesHelper::get_relocatable_particle_varnames();
    vector<string>::const_iterator varNameIter = otherParticleVarNames.begin();
    //    vector<string>::iterator varNameIter = needsRelocation_.begin();
    for( ; varNameIter != otherParticleVarNames.end(); ++varNameIter ){
      if( VarLabel::find( *varNameIter ) ){
        const VarLabel* theVarLabel = VarLabel::find( *varNameIter );
        
        if (std::find(otherParticleVarLabels.begin(), otherParticleVarLabels.end(),theVarLabel) == otherParticleVarLabels.end() ){
          otherParticleVarLabels.push_back(VarLabel::find( *varNameIter ));
        }
      }
    }

    // add the particle ID label!
    otherParticleVarLabels.push_back(pIDLabel_);
    
    vector< vector<const VarLabel*> > otherParticleVars;
    for( int m = 0; m < materials_->size(); ++m ){
      otherParticleVars.push_back( otherParticleVarLabels );
    }
    sched->scheduleParticleRelocation(level, pPosLabel_, otherParticleVars, materials_);
    
    // clean the delete set
    Task* task = scinew Task("cleanup deleteset", this, &ParticlesHelper::clear_deleteset);
    sched->addTask(task, level->eachPatch(), materials_);
    
    // after particle relocation, one must sync the Uintah particle position back
    // with the component particle positions. This is important in periodic
    // problems so that one recovers the correct particle positions as particles
    // go through the periodic boundaries
    using namespace Uintah;
    Uintah::Task* periodictask = scinew Uintah::Task("sync particles for periodic boundaries", this, &ParticlesHelper::sync_particle_position_periodic );
    periodictask->requires(Task::NewDW, pPosLabel_, Uintah::Ghost::None, 0);
    periodictask->modifies(pXLabel_);
    periodictask->modifies(pYLabel_);
    periodictask->modifies(pZLabel_);
    sched->addTask(periodictask, level->eachPatch(), materials_);
  }
  
  //--------------------------------------------------------------------
  
  // this will create the particle subset
  void ParticlesHelper::clear_deleteset( const Uintah::ProcessorGroup*,
                                         const Uintah::PatchSubset* patches,
                                         const Uintah::MaterialSubset* matls,
                                         Uintah::DataWarehouse* old_dw,
                                         Uintah::DataWarehouse* new_dw )
  {
    using namespace Uintah;
    for( int m = 0; m<matls->size(); ++m ){
      const int matl = matls->get(m);
      std::map<int,ParticleSubset*>& thisMatDelSet = deleteSets_[m];
      for( int p=0; p<patches->size(); ++p ){
        const Patch* patch = patches->get(p);
        ParticleSubset* existingDelset = thisMatDelSet[patch->getID()];
        if( existingDelset->numParticles() > 0 ){
          ParticleSubset* delset = scinew ParticleSubset(0,matl, patch);
          thisMatDelSet[patch->getID()] = delset;
        }
      }
    }
  }
  
  //--------------------------------------------------------------------
  
  // this task will sync particle position with component computed values
  void ParticlesHelper::schedule_transfer_particle_ids(const Uintah::LevelP& level,
                                                       Uintah::SchedulerP& sched)
  {
    using namespace Uintah;
    Uintah::Task* task = scinew Uintah::Task( "transfer particles IDs",
                                              this, &ParticlesHelper::transfer_particle_ids );
    task->computes(pIDLabel_);
    task->requires(Task::OldDW, pIDLabel_, Uintah::Ghost::None, 0);
    sched->addTask(task, level->eachPatch(), materials_);
  }
  
  //--------------------------------------------------------------------
  
  void ParticlesHelper::transfer_particle_ids( const Uintah::ProcessorGroup*,
                                               const Uintah::PatchSubset* patches,
                                               const Uintah::MaterialSubset* matls,
                                               Uintah::DataWarehouse* old_dw,
                                               Uintah::DataWarehouse* new_dw )
  {
    using namespace Uintah;
    for( int m = 0; m<matls->size(); ++m ){
      const int matl = matls->get(m);
      for( int p=0; p<patches->size(); ++p ){
        const Patch* patch = patches->get(p);
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
  
  void ParticlesHelper::sync_particle_position_periodic(const Uintah::ProcessorGroup*,
                                               const Uintah::PatchSubset* patches, const Uintah::MaterialSubset* matls,
                                               Uintah::DataWarehouse* old_dw, Uintah::DataWarehouse* new_dw)
  {
    using namespace Uintah;
    for(int m = 0; m<matls->size(); m++){
      const int matl = matls->get(m);
//      std::map<int,ParticleSubset*>& thisMatDelSet = deleteSets_[m];
      for(int p=0;p<patches->size();p++){
        const Patch* patch = patches->get(p);
        ParticleSubset* pset = new_dw->getParticleSubset(matl, patch);
        const int numParticles =pset->numParticles();
        
        //new_dw->deleteParticles(thisMatDelSet[patch->getID()]);
        
        constParticleVariable<Point> ppos; // Uintah particle position
        
        // component particle positions
        ParticleVariable<double> px;
        ParticleVariable<double> py;
        ParticleVariable<double> pz;
        
        new_dw->getModifiable(px,    pXLabel_,                  pset);
        new_dw->getModifiable(py,    pYLabel_,                  pset);
        new_dw->getModifiable(pz,    pZLabel_,                  pset);
        new_dw->get          (ppos,  pPosLabel_,                pset);
        
        for (int i = 0; i < numParticles; i++) {
          px[i] = ppos[i].x();
          py[i] = ppos[i].y();
          pz[i] = ppos[i].z();
        }
      }
    }
  }

  
  
  //--------------------------------------------------------------------

  // this task will sync particle position with component computed values
  void ParticlesHelper::schedule_sync_particle_position( const Uintah::LevelP& level,
                                                         Uintah::SchedulerP& sched,
                                                         const bool initialization )
  {
    using namespace Uintah;
    Uintah::Task* task = scinew Uintah::Task("sync particles", this, &ParticlesHelper::sync_particle_position, initialization );
    if( initialization ){
      task->modifies( pPosLabel_ );
    }
    else{
      task->computes( pPosLabel_ );
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
    for(int m = 0; m<matls->size(); m++){
      const int matl = matls->get(m);
      std::map<int,ParticleSubset*>& thisMatDelSet = deleteSets_[m];
      for(int p=0;p<patches->size();p++){
        const Patch* patch = patches->get(p);
        ParticleSubset* pset = initialization ? new_dw->getParticleSubset(matl, patch) : old_dw->getParticleSubset(matl, patch);
        const int numParticles =pset->numParticles();
        
        new_dw->deleteParticles(thisMatDelSet[patch->getID()]);
        
        ParticleVariable<Point> ppos; // Uintah particle position
        
        // component particle positions
        constParticleVariable<double> px;
        constParticleVariable<double> py;
        constParticleVariable<double> pz;
        
        new_dw->get(px,    pXLabel_,                  pset);
        new_dw->get(py,    pYLabel_,                  pset);
        new_dw->get(pz,    pZLabel_,                  pset);
        if (initialization) {
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
  
  //------------------------------------------------------------------------------------------------
  
  void ParticlesHelper::allocate_boundary_particles_vector(const std::string& bndName,
                                                           const int& patchID                )
  {
    using namespace std;
    vector<int> bndParticlesIndicesEmpty;
    typedef map<int, vector<int> > patchIDIterMapT;
    if ( bndParticlesMap_.find(bndName) != bndParticlesMap_.end() ) {
      // if this boundary was already added then check if the patchID was added
      patchIDIterMapT& myMap = (*bndParticlesMap_.find(bndName)).second;
      if ( myMap.find(patchID) != myMap.end() ) {
        // if the patchID was already added, simply update the particle indices with new ones from myIters
        std::ostringstream msg;
        msg << "ERROR: While trying to allocate memory to store the list of particles near boundary " << bndName <<
        " and on patch " << patchID << ". It looks like memory was already allocated for that patch." << std::endl;
        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
      } else {
        // if the patchID was not added, add a new particle indices vector
        (*bndParticlesMap_.find(bndName)).second.insert( pair<int, vector<int> >(patchID, bndParticlesIndicesEmpty));
      }
    } else {
      patchIDIterMapT patchIDIterMap;
      patchIDIterMap.insert(pair< int, vector<int> >(patchID, bndParticlesIndicesEmpty));
      bndParticlesMap_.insert( pair< string, patchIDIterMapT >(bndName, patchIDIterMap ) );
    }
  }
  
  //------------------------------------------------------------------------------------------------
  
  void ParticlesHelper::update_boundary_particles_vector( const std::vector<int>& myIters,
                                                         const std::string& bndName,
                                                         const int& patchID                )
  {
    using namespace std;
    typedef map<int, vector<int> > patchIDIterMapT;
    if ( bndParticlesMap_.find(bndName) != bndParticlesMap_.end() ) {
      // if this boundary was already added then check if the patchID was added
      patchIDIterMapT& myMap = (*bndParticlesMap_.find(bndName)).second;
      if ( myMap.find(patchID) != myMap.end() ) {
        // if the patchID was already added, simply update the particle indices with new ones from myIters
        (*myMap.find(patchID)).second = myIters;
      } else {
        // if the patchID was not added, add a new particle indices vector
        (*bndParticlesMap_.find(bndName)).second.insert( pair<int, std::vector<int> >(patchID, myIters));
      }
    } else {
      patchIDIterMapT patchIDIterMap;
      patchIDIterMap.insert(pair< int, std::vector<int> >(patchID, myIters));
      bndParticlesMap_.insert( pair< string, patchIDIterMapT >(bndName, patchIDIterMap ) );
    }
  }
  
  //------------------------------------------------------------------------------------------------
  
  const std::vector<int>* ParticlesHelper::get_boundary_particles( const std::string& bndName,
                                                                  const int patchID )
  {
    using namespace std;
    if ( bndParticlesMap_.find(bndName) != bndParticlesMap_.end() ) {
      std::map<int, std::vector<int> >& temp = (*bndParticlesMap_.find(bndName)).second;
      return &(temp.find(patchID)->second);
    }
    return nullptr;
  }
  
  //--------------------------------------------------------------------
  
  void ParticlesHelper::schedule_find_boundary_particles( const Uintah::LevelP& level,
                                                         Uintah::SchedulerP& sched    )
  {
    
    // this task will allocate a particle subset and create particle positions
    Uintah::Task* task = scinew Uintah::Task("find boundary particles",
                                             this, &ParticlesHelper::find_boundary_particles);
    task->requires(Task::OldDW, pPosLabel_, Uintah::Ghost::None, 0);
    sched->addTask(task, level->eachPatch(), materials_);
  }
  
  //--------------------------------------------------------------------
  
  void ParticlesHelper::find_boundary_particles( const Uintah::ProcessorGroup*,
                                                const Uintah::PatchSubset* patches, const Uintah::MaterialSubset* matls,
                                                Uintah::DataWarehouse* old_dw, Uintah::DataWarehouse* new_dw )
  {
    using namespace std;
    using namespace Uintah;
    
    // loop over the material subset
    for (int m=0; m<matls->size(); m++) {
      const int materialID = matls->get(m);
      for (int p=0; p<patches->size(); p++){
        const Patch* patch = patches->get(p);
        const int patchID = patch->getID();
        
        std::vector<Uintah::Patch::FaceType> bndFaces;
        patch->getBoundaryFaces(bndFaces);
        
        // loop over the physical boundaries of this patch. These are the LOGICAL boundaries
        // and do NOT include intrusions
        for (size_t f=0; f < bndFaces.size(); f++) {
          Patch::FaceType face = bndFaces[f];
          // for a full boundary face, get the list of particles that are near that boundary
          IntVector low, high;
          patch->getFaceCells(face,-1,low,high);
          
          ParticleSubset* bndParticleSubset = old_dw->getParticleSubset(materialID, patch);
          
          constParticleVariable<Point> pos;
          old_dw->get(pos,pPosLabel_,bndParticleSubset);
          
          // map that holds particle indices per boundary cell: boundary cell index -> (vector of particles in that cell)
          std::map<Uintah::IntVector, std::vector<int> > bndCIdxPIdx;
          
          // loop over all particles in this patch
          for (ParticleSubset::iterator it = bndParticleSubset->begin(); it!=bndParticleSubset->end();++it)
          {
            // get particle index
            particleIndex idx = *it;
            // get the cell index in which this particle lives
            const Uintah::IntVector cellIdx = patch->getCellIndex(pos[idx]);
            // if this cell is part of the boundary face, then add it to the list of boundary cells and particles
            if(Patch::containsIndex(low,high,cellIdx) )
            {
              map<IntVector, vector<int> >::iterator it = bndCIdxPIdx.find(cellIdx);
              map<IntVector, vector<int> >::iterator iend = bndCIdxPIdx.end();
              if ( it == iend ) {
                bndCIdxPIdx.insert(pair<IntVector, vector<int> >(cellIdx, vector<int>(1,idx)));
              } else {
                it->second.push_back(idx);
              }
              
            }
          }
          
          // Get the number of "boundaries" (children) specified on this boundary face.
          // example: x- boundary face has a circle specified as inlet while the rest of the
          // face is specified as wall. This results in two "boundaries" or children.
          // the BCDataArray will store this list of children
          const Uintah::BCDataArray* bcDataArray = patch->getBCDataArray(face);
          
          // Grab the number of children on this boundary face
          const int numChildren = bcDataArray->getNumberChildren(materialID);
          
          const Uintah::IntVector unitNormal = patch->faceDirection(face); // this is needed to construct interior cells
          // now go over every child-boundary (sub-boundary) specified on this domain boundary face
          for( int chid = 0; chid<numChildren; ++chid ) {
            
            // here is where the fun starts. Now we can get information about this boundary condition.
            // The BCDataArray stores information related to its children as BCGeomBase objects.
            // Each child is associated with a BCGeomBase object. Grab that
            Uintah::BCGeomBase* thisGeom = bcDataArray->getChild(materialID,chid);
            const std::string bndName = thisGeom->getBCName();
            if (bndName=="NotSet") {
              std::ostringstream msg;
              msg << "ERROR: It looks like you have not set a name for one of your boundary conditions! "
              << "You MUST specify a name for your <Face> spec boundary condition. Please revise your input file." << std::endl;
              throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
            }
            
            //__________________________________________________________________________________
            Uintah::Iterator bndIter; // allocate iterator
            std::vector<int> childBndParticles;
            // get the iterator for the extracells for this child
            bcDataArray->getCellFaceIterator(materialID, bndIter, chid);
            
            // now that we found the list of particles that are near the parent boundary face,
            // lets go over every child and find which of these particles belong to that child.
            // this is very expensive.
            // for every boundary point on this child, see which particles belong
            for( bndIter.reset(); !bndIter.done(); ++bndIter ) {
              const Uintah::IntVector bcPointIJK = *bndIter - unitNormal;
              // find this boundary cell in the bndCIdxPIdx
              map<IntVector, vector<int> >::iterator it = bndCIdxPIdx.find(bcPointIJK);
              map<IntVector, vector<int> >::iterator iend = bndCIdxPIdx.end();
              if ( it != iend ) {
                childBndParticles.insert(childBndParticles.end(),it->second.begin(), it->second.end());
              }
            }
            update_boundary_particles_vector( childBndParticles, bndName, patchID );
          } // boundary child loop (note, a boundary child is what we think of as a boundary condition
        } // boundary faces loop
      } // patch loop
    } // patch subset loop
  } // material subset loop

  //--------------------------------------------------------------------
  
  void ParticlesHelper::parse_boundary_conditions( const Uintah::LevelP& level,
                                                  Uintah::SchedulerP& sched)
  {
    const Uintah::PatchSet* const allPatches = sched->getLoadBalancer()->getPerProcessorPatchSet(level);
    const Uintah::PatchSubset* const localPatches = allPatches->getSubset( Uintah::Parallel::getMPIRank() );
    Uintah::PatchSet* patches = new Uintah::PatchSet;
    patches->addEach( localPatches->getVector() );
    parse_boundary_conditions(patches);
    delete patches;
  }
  
  //--------------------------------------------------------------------
  
  void ParticlesHelper::parse_boundary_conditions( const Uintah::PatchSet* const localPatches)
  {
    using namespace std;
    using namespace Uintah;

    // loop over the material set
    for (int ms = 0; ms < materials_->size(); ms++) {
      const Uintah::MaterialSubset* matSubSet = materials_->getSubset(ms);
      // loop over materials
      for( int im=0; im < matSubSet->size(); ++im ) {
        
        const int materialID = matSubSet->get(im);
        
        // loop over local patches
        for (int ps = 0; ps < localPatches->size(); ps++) {
          const Uintah::PatchSubset* const patches = localPatches->getSubset(ps);
          // loop over every patch in the patch subset
          for (int p=0; p<patches->size(); p++) {
            const Uintah::Patch* const patch = patches->get(p);
            const int patchID = patch->getID();
            
            std::vector<Uintah::Patch::FaceType> bndFaces;
            patch->getBoundaryFaces(bndFaces);
            
            // loop over the physical boundaries of this patch. These are the LOGICAL boundaries
            // and do NOT include intrusions
            for (size_t f=0; f<bndFaces.size(); f++) {
              const Patch::FaceType face = bndFaces[f];
              // for a full boundary face, get the list of particles that are near that boundary
              IntVector low, high;
              patch->getFaceCells(face,-1,low,high);
              
              // Get the number of "boundaries" (children) specified on this boundary face.
              // example: x- boundary face has a circle specified as inlet while the rest of the
              // face is specified as wall. This results in two "boundaries" or children.
              // the BCDataArray will store this list of children
              const Uintah::BCDataArray* bcDataArray = patch->getBCDataArray(face);
              
              // Grab the number of children on this boundary face
              const int numChildren = bcDataArray->getNumberChildren(materialID);
              
              // now go over every child-boundary (sub-boundary) specified on this domain boundary face
              for( int chid = 0; chid<numChildren; ++chid ) {
                
                // here is where the fun starts. Now we can get information about this boundary condition.
                // The BCDataArray stores information related to its children as BCGeomBase objects.
                // Each child is associated with a BCGeomBase object. Grab that
                Uintah::BCGeomBase* thisGeom = bcDataArray->getChild(materialID,chid);
                const std::string bndName = thisGeom->getBCName();
                if (bndName=="NotSet") {
                  std::ostringstream msg;
                  msg << "ERROR: It looks like you have not set a name for one of your boundary conditions! "
                  << "You MUST specify a name for your <Face> spec boundary condition. Please revise your input file." << std::endl;
                  throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
                }                
                // for every child, allocate a new vector for boundary particles. this vector will
                // be referenced by the boundary condition expressions/tasks.
                allocate_boundary_particles_vector(bndName, patchID );
              } // boundary child loop (note, a boundary child is what we think of as a boundary condition
            } // boundary faces loop
          } // patch loop
        } // patch subset loop
      } // material loop
    }
  }
  
  //--------------------------------------------------------------------
  
  void ParticlesHelper::schedule_add_particles( const Uintah::LevelP& level,
                                               Uintah::SchedulerP& sched )
  {
    if( needsBC_.size() == 0 ) return;
    // this task will allocate a particle subset and create particle positions
    Uintah::Task* task = scinew Uintah::Task( "add particles",
                                              this, &ParticlesHelper::add_particles );
    for( size_t i=0; i<needsBC_.size(); ++i ){
      task->modifies(Uintah::VarLabel::find(needsBC_[i]));
    }
    task->modifies(pIDLabel_ );
    task->modifies(pPosLabel_);
    task->requires(Task::OldDW, delTLabel_);
    sched->addTask(task, level->eachPatch(), materials_);
  }
  
  //--------------------------------------------------------------------
  
  void ParticlesHelper::add_particles( const Uintah::ProcessorGroup*,
                                       const Uintah::PatchSubset* patches,
                                       const Uintah::MaterialSubset* matls,
                                       Uintah::DataWarehouse* old_dw,
                                       Uintah::DataWarehouse* new_dw )
  {
    using namespace Uintah;
    
    delt_vartype DT;
    old_dw->get(DT, delTLabel_);
    const double dt = DT;

    for( int m=0; m<matls->size(); ++m ){
      const int matl = matls->get(m);
      std::map<int,long64>& lastPIDPerPatch = lastPIDPerMaterialPerPatch_[m];
      for( int p=0; p<patches->size(); ++p ){
        const Patch* patch = patches->get( p );
        const int patchID = patch->getID();
        // get the last particle ID created by this patch. will be used further down.
        long64& lastPID = lastPIDPerPatch[patchID];
        const long64 pidoffset = patchID * PIDOFFSET;
        
        std::vector<Uintah::Patch::FaceType> bndFaces;
        patch->getBoundaryFaces(bndFaces);
        
        // loop over the physical boundaries of this patch. These are the LOGICAL boundaries
        // and do NOT include intrusions
        for( size_t f=0; f < bndFaces.size(); ++f ){
          Patch::FaceType face = bndFaces[f];
          
          // Get the number of "boundaries" (children) specified on this boundary face.
          // example: x- boundary face has a circle specified as inlet while the rest of the
          // face is specified as wall. This results in two "boundaries" or children.
          // the BCDataArray will store this list of children
          const Uintah::BCDataArray* bcDataArray = patch->getBCDataArray(face);
          
          // Grab the number of children on this boundary face
          const int numChildren = bcDataArray->getNumberChildren(matl);
          
          const Uintah::IntVector unitNormal = patch->faceDirection(face); // this is needed to construct interior cells
          // now go over every child-boundary (sub-boundary) specified on this domain boundary face
          for( int chid = 0; chid<numChildren; ++chid ) {
            
            // here is where the fun starts. Now we can get information about this boundary condition.
            // The BCDataArray stores information related to its children as BCGeomBase objects.
            // Each child is associated with a BCGeomBase object. Grab that
            Uintah::BCGeomBase* thisGeom = bcDataArray->getChild(matl,chid);
            const std::string bndName = thisGeom->getBCName();
            if( bndName=="NotSet" ){
              std::ostringstream msg;
              msg << "ERROR: It looks like you have not set a name for one of your boundary conditions! "
              << "You MUST specify a name for your <Face> spec boundary condition. Please revise your input file." << std::endl;
              throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
            }
            
            const Uintah::BCGeomBase::ParticleBndSpec& pBndSpec = thisGeom->getParticleBndSpec();
            if( pBndSpec.hasParticleBC() ){
              if( pBndSpec.bndType == Uintah::BCGeomBase::ParticleBndSpec::INLET ){
                // This is a particle inlet. get the number of boundary particles per second
                const double pPerSec = pBndSpec.particlesPerSec;

                //__________________________________________________________________________________
                Uintah::Iterator bndIter; // allocate iterator
                // get the iterator for the extracells for this child
                bcDataArray->getCellFaceIterator(matl, bndIter, chid);
                if( bndIter.done() ) continue; // go to the next child if this iterator is empty
                // get the number of cells on this boundary
                const unsigned int nCells = bndIter.size();
                // if the number of cells is zero, then return. this is extra proofing
                if( nCells == 0 ) continue;
                
                ParticleSubset* pset = new_dw->haveParticleSubset(matl,patch) ? new_dw->getParticleSubset(matl, patch) : old_dw->getParticleSubset(matl, patch);

                const unsigned int newNParticles = dt * pPerSec;
                const unsigned int oldNParticles = pset->addParticles(newNParticles);
                
                // deal with particles IDs separately from other particle variables
                ParticleVariable<long64> pids;
                new_dw->getModifiable(pids, pIDLabel_, pset);

                // deal with particles IDs separately from other particle variables
                ParticleVariable<Uintah::Point> ppos;
                new_dw->getModifiable(ppos, pPosLabel_, pset);

                // deal with the rest of the particle variables below
                const unsigned int nVars = needsBC_.size();
                std::vector< Uintah::VarLabel* > needsBCLabels; // vector of varlabels that need bcs

		std::vector< ParticleVariable<double> > allVars(nVars);
		std::vector< ParticleVariable<double> > tmpVars(nVars);
                for( size_t i=0; i<needsBC_.size(); ++i ){
                  needsBCLabels.push_back( VarLabel::find(needsBC_[i]) );
                  new_dw->getModifiable( allVars[(unsigned)i], needsBCLabels[i], pset );
                  new_dw->allocateTemporary( tmpVars[(unsigned)i], pset );
                }
                
                //__________________________________________________________________________________
                // now allocate temporary variables of size new particlesubset
                
                ParticleVariable<long64> pidstmp;
                new_dw->allocateTemporary(pidstmp, pset);

                ParticleVariable<Uintah::Point> ppostmp;
                new_dw->allocateTemporary(ppostmp, pset);

                // copy the data from old variables to temporary vars
                for( unsigned i=0; i<nVars; ++i ){
                  ParticleVariable<double>& oldVar = allVars[i];
                  ParticleVariable<double>& tmpVar = tmpVars[i];
                  for( unsigned int p=0; p<oldNParticles; ++p ){
                    tmpVar[p] = oldVar[p];
                  }
                }
                // copy data from old variables for particle IDs and the position vector
                for( unsigned int p=0; p<oldNParticles; ++p ){
                  pidstmp[p] = pids[p];
                  ppostmp[p] = ppos[p];
                }
                
                // find out which variables are the x, y, and z position variables
                std::vector<VarLabel*>::iterator itx = std::find (needsBCLabels.begin(), needsBCLabels.end(), pXLabel_);
                const int ix = std::distance(needsBCLabels.begin(), itx);
                ParticleVariable<double>& pxtmp = tmpVars[ix];
                
                std::vector<VarLabel*>::iterator ity = std::find (needsBCLabels.begin(), needsBCLabels.end(), pYLabel_);
                const int iy = std::distance(needsBCLabels.begin(), ity);
                ParticleVariable<double>& pytmp = tmpVars[iy];
                
                std::vector<VarLabel*>::iterator itz = std::find (needsBCLabels.begin(), needsBCLabels.end(), pZLabel_);
                const int iz = std::distance(needsBCLabels.begin(), itz);
                ParticleVariable<double>& pztmp = tmpVars[iz];
                
                // inject particles. This will place particles randomly on the injecting boundary
                unsigned int i = oldNParticles;
                Vector spacing = patch->dCell()/2.0;
                for( unsigned int j=0; j<newNParticles; ++j, ++i ){
                  
                  // pick a random cell on this boundary
                  const unsigned int r1 = rand() % nCells;
                  bndIter.reset();
                  for( unsigned int t = 0; t < r1; ++t ) bndIter++;
                  
                  // get the interior cell
                  const IntVector bcPointIJK = *bndIter - unitNormal;
                  const Point        bcPoint = patch->getCellPosition(bcPointIJK);

                  // get the bounds of this cell
                  const Point low (bcPoint - spacing);
                  const Point high(bcPoint + spacing);
                  
                  // generate a random point inside this cell
                  const Point pos( (((float) rand()) / RAND_MAX * ( high.x() - low.x()) + low.x()),
                                   (((float) rand()) / RAND_MAX * ( high.y() - low.y()) + low.y()),
                                   (((float) rand()) / RAND_MAX * ( high.z() - low.z()) + low.z()) );

                  // set the particle ID
                  pidstmp[i] = lastPID + j + 1 + pidoffset;

                  // set the particle positions
                  ppostmp[i] = pos;
                  pxtmp[i]   = pos.x();
                  pytmp[i]   = pos.y();
                  pztmp[i]   = pos.z();
                }
                // save the last particle ID used on this patch
                lastPID = pidstmp[oldNParticles + newNParticles - 1];
                
                // go through the list of particle variables specified at this boundary
                //__________________________________________________________________________________
                // Now, each BCGeomObject has BCData associated with it. This BCData contains the list
                // of variables and types (Dirichlet, etc...), and values that the user specified
                // through the input file!
                Uintah::BCData bcData;
                thisGeom->getBCData(bcData);
                
                // loop over the particle variables for which a BC has been specified
                for( size_t ivar=0; ivar < needsBC_.size(); ++ivar ){
                  const std::string varName = needsBC_[ivar];
                  const Uintah::BoundCondBase* bndCondBase = bcData.getBCValues(varName);
                  int p = oldNParticles;
                  ParticleVariable<double>& pvar = tmpVars[(unsigned)ivar];
                  if( bndCondBase ){
                    const Uintah::BoundCond<double>* const new_bc = dynamic_cast<const Uintah::BoundCond<double>*>(bndCondBase);
                    const double doubleVal = new_bc->getValue();
                    // right now, we only support constant boundary conditions
                    for( unsigned int j=0; j<newNParticles; ++j, ++p ){
                      // pvar[p] = ((double) rand()/RAND_MAX)*(doubleVal*1.2 - doubleVal*0.8) + doubleVal*0.8;
                      pvar[p] = doubleVal;
                    }
                  }
                  else if( varName != pXLabel_->getName() &&
                           varName != pYLabel_->getName() &&
                           varName != pZLabel_->getName() )
                  {
                    // for all particle variables that do not have bcs specified in the input file, initialize them to zero
                    for( unsigned int j=0; j<newNParticles; ++j, ++p ) {
                      pvar[p] = 0.0;
                    }
                  }
                }
                
                // put the temporary data back in the original variables
                new_dw->put(pidstmp, pIDLabel_, true);
                new_dw->put(ppostmp, pPosLabel_, true);
                for( size_t ivar=0; ivar < needsBCLabels.size(); ++ivar ){
                  new_dw->put(tmpVars[(unsigned)ivar],needsBCLabels[ivar],true);
                }
              }
            }
          } // boundary child loop
        }
      }
    }
  }
  
  //--------------------------------------------------------------------
  
} /* namespace Uintah */
