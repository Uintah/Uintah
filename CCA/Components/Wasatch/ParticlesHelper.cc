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

/**************************************************************************************
 _______    ______   _______  ________  ______   ______   __        ________   ______
 |       \  /      \ |       \|        \|      \ /      \ |  \      |        \ /      \
 | $$$$$$$\|  $$$$$$\| $$$$$$$\\$$$$$$$$ \$$$$$$|  $$$$$$\| $$      | $$$$$$$$|  $$$$$$\
 | $$__/ $$| $$__| $$| $$__| $$  | $$     | $$  | $$   \$$| $$      | $$__    | $$___\$$
 | $$    $$| $$    $$| $$    $$  | $$     | $$  | $$      | $$      | $$  \    \$$    \
 | $$$$$$$ | $$$$$$$$| $$$$$$$\  | $$     | $$  | $$   __ | $$      | $$$$$    _\$$$$$$\
 | $$      | $$  | $$| $$  | $$  | $$    _| $$_ | $$__/  \| $$_____ | $$_____ |  \__| $$
 | $$      | $$  | $$| $$  | $$  | $$   |   $$ \ \$$    $$| $$     \| $$     \ \$$    $$
 \$$       \$$   \$$ \$$   \$$   \$$    \$$$$$$  \$$$$$$  \$$$$$$$$ \$$$$$$$$  \$$$$$$
 **************************************************************************************/

#include "ParticlesHelper.h"

//-- Wasatch includes --//
#include <CCA/Components/Wasatch/Wasatch.h>

//-- Uintah Includes --//
#include <Core/Grid/Box.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Exceptions/ProblemSetupException.h>

std::vector<std::string> Uintah::ParticlesHelper::otherParticleVarNames_;
std::map<std::string, std::map<int, std::vector<int> > > Uintah::ParticlesHelper::bndParticlesMap_;

namespace Uintah {
  
  //==================================================================
  
  const std::vector<std::string>&
  ParticlesHelper::get_relocatable_particle_varnames()
  {
    return otherParticleVarNames_;
  }
  
  //------------------------------------------------------------------
  
  void
  ParticlesHelper::add_particle_variable(const std::string& varName )
  {
    otherParticleVarNames_.push_back(varName);
  }

  //------------------------------------------------------------------
  
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
    parse_boundary_conditions(level, sched);
  }
  
  //--------------------------------------------------------------------
  
  // this will create the particle subset
  void ParticlesHelper::initialize( const Uintah::ProcessorGroup*,
                                   const Uintah::PatchSubset* patches, const Uintah::MaterialSubset* matls,
                                   Uintah::DataWarehouse* old_dw, Uintah::DataWarehouse* new_dw)
  {
    using namespace Uintah;
    particleEqsSpec_->get("ParticlesPerCell",pPerCell_);
    
    for(int m = 0;m<matls->size();m++){
      const int matl = matls->get(m);
      for(int p=0;p<patches->size();p++){
        const Patch* patch = patches->get(p);
        
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
    
    for(int m = 0;m<matls->size();m++){
      int matl = matls->get(m);
      for(int p=0;p<patches->size();p++){
        const Patch* patch = patches->get(p);
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
    for(int m = 0; m<matls->size(); m++){
      const int matl = matls->get(m);
      for(int p=0;p<patches->size();p++){
        const Patch* patch = patches->get(p);
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
    const vector<string>& otherParticleVarNames = ParticlesHelper::get_relocatable_particle_varnames();
    vector<string>::const_iterator varNameIter = otherParticleVarNames.begin();
    //    vector<string>::iterator varNameIter = otherParticleVarNames_.begin();
    for (; varNameIter != otherParticleVarNames.end(); ++varNameIter) {
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
    for(int m = 0; m<matls->size(); m++){
      const int matl = matls->get(m);
      for(int p=0;p<patches->size();p++){
        const Patch* patch = patches->get(p);
        ParticleSubset* existingDelset = deleteSet_[patch->getID()];
        if (existingDelset->numParticles() > 0)
        {
          ParticleSubset* delset = scinew ParticleSubset(0,matl, patch);
          deleteSet_[patch->getID()] = delset;
        }
      }
    }
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
    for(int m = 0; m<matls->size(); m++){
      const int matl = matls->get(m);
      for(int p=0;p<patches->size();p++){
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
    for(int m = 0; m<matls->size(); m++){
      const int matl = matls->get(m);
      for(int p=0;p<patches->size();p++){
        const Patch* patch = patches->get(p);
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
    return NULL;
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
    using namespace SCIRun;
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
        for (int f=0; f < bndFaces.size(); f++) {
          Patch::FaceType face = bndFaces[f];
          // for a full boundary face, get the list of particles that are near that boundary
          IntVector low, high;
          patch->getFaceCells(face,-1,low,high);
          
          ParticleSubset* bndParticleSubset = old_dw->getParticleSubset(materialID, patch);
          
          constParticleVariable<Point> pos;
          old_dw->get(pos,pPosLabel_,bndParticleSubset);
          
          std::vector<int> bndParticlesIndices;
          std::vector<Uintah::IntVector> bndCellIndices;
          for (ParticleSubset::iterator it = bndParticleSubset->begin(); it!=bndParticleSubset->end();++it)
          {
            particleIndex idx = *it;
            const Uintah::IntVector cellIdx = patch->getCellIndex(pos[idx]);
            if(Patch::containsIndex(low,high,cellIdx) )
            {
              bndParticlesIndices.push_back(idx);
              bndCellIndices.push_back(cellIdx);
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
              vector<Uintah::IntVector>::iterator cit = bndCellIndices.begin();
              vector<int>::iterator pit = bndParticlesIndices.begin();
              for (; cit != bndCellIndices.end() && pit != bndParticlesIndices.end(); ++cit, ++pit)
              {
                if (*cit == bcPointIJK) {
                  childBndParticles.push_back(*pit);
                }
              }
            }
            update_boundary_particles_vector( childBndParticles, bndName, patchID );
          } // boundary child loop (note, a boundary child is what Wasatch thinks of as a boundary condition
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
  }
  //--------------------------------------------------------------------
  
  void ParticlesHelper::parse_boundary_conditions( const Uintah::PatchSet* const localPatches)
  {
    using namespace std;
    using namespace SCIRun;
    using namespace Uintah;
    // loop over the material set
    BOOST_FOREACH( const Uintah::MaterialSubset* matSubSet, materials_->getVector() ) {
      
      // loop over materials
      for( int im=0; im<matSubSet->size(); ++im ) {
        
        const int materialID = matSubSet->get(im);
        
        // loop over local patches
        BOOST_FOREACH( const Uintah::PatchSubset* const patches, localPatches->getVector() ) {
          
          // loop over every patch in the patch subset
          BOOST_FOREACH( const Uintah::Patch* const patch, patches->getVector() ) {
            
            const int patchID = patch->getID();
            
            std::vector<Uintah::Patch::FaceType> bndFaces;
            patch->getBoundaryFaces(bndFaces);
            
            // loop over the physical boundaries of this patch. These are the LOGICAL boundaries
            // and do NOT include intrusions
            BOOST_FOREACH(const Uintah::Patch::FaceType face, bndFaces) {
              
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
                // be referenced by the boundary condition expressions.
                allocate_boundary_particles_vector(bndName, patchID );
              } // boundary child loop (note, a boundary child is what Wasatch thinks of as a boundary condition
            } // boundary faces loop
          } // patch loop
        } // patch subset loop
      } // material loop
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