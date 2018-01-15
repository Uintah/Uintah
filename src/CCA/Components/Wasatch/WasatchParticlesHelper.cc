/**
 *  \file   WasatchParticlesHelper.cc
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

#include <CCA/Components/Wasatch/WasatchParticlesHelper.h>

//-- Wasatch includes --//
#include <CCA/Components/Wasatch/Wasatch.h>

//-- Uintah Includes --//
#include <Core/Grid/Box.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/GeometryPiece/GeometryPiece.h>
#include <Core/GeometryPiece/GeometryPieceFactory.h>

namespace WasatchCore {
  
  //==================================================================
  
  WasatchParticlesHelper::WasatchParticlesHelper() :
  Uintah::ParticlesHelper()
  {
    wasatch_ = nullptr;
    wasatchSync_ = false;
  }
  
  //------------------------------------------------------------------
  
  WasatchParticlesHelper::~WasatchParticlesHelper()
  {}
  
  //--------------------------------------------------------------------
  
  void WasatchParticlesHelper::schedule_initialize( const Uintah::LevelP& level,
                                                    Uintah::SchedulerP& sched )
  {
    // this task will allocate a particle subset and create particle positions
    Uintah::Task* task = scinew Uintah::Task( "initialize particles",
                                              this, &WasatchParticlesHelper::initialize );
    task->computes(pPosLabel_);
    task->computes(pIDLabel_);
    sched->addTask(task, level->eachPatch(), materials_);    
    parse_boundary_conditions(level, sched);
  }

  //--------------------------------------------------------------------
  
  // this will create the particle subset
  void WasatchParticlesHelper::initialize( const Uintah::ProcessorGroup*,
                                           const Uintah::PatchSubset* patches,
                                           const Uintah::MaterialSubset* matls,
                                           Uintah::DataWarehouse* old_dw,
                                           Uintah::DataWarehouse* new_dw)
  {
    assert( wasatchSync_     );
    assert( wasatch_ != nullptr );

    using namespace Uintah;
    initialize_internal(matls->size());
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


    std::vector <GeometryPieceP > geomObjects;
    
    bool hasGeom = false;
    bool bounded=false;
    for( ProblemSpecP exprParams = wasatch_->get_wasatch_spec()->findBlock("BasicExpression");
        exprParams != nullptr;
        exprParams = exprParams->findNextBlock("BasicExpression") )
    {
      // look for ParticlePositionIC xml Blocks (initial condition). These specify the kind of
      // initial condition that the user wants to apply on particles.
      if (exprParams->findBlock("ParticlePositionIC")) {
        ProblemSpecP pICSpec = exprParams->findBlock("ParticlePositionIC");
        // check what type of bounds we are using: specified or patch based?
        std::string boundsType;
        // Check the kinds of bounds that the user has specified (based on patch bounds or user specified)
        pICSpec->getAttribute("bounds",boundsType);
        bounded = bounded || (boundsType == "SPECIFIED");
        if (bounded) {
          // if the user specified the bounds, then check those out and save them in xmin, xmax etc...
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
        
        if ( pICSpec->findBlock("Geometry") && geomObjects.size() == 0 ) { // only allow one vector of geom objects to be created. this is because all particle initialization with geometry shapes must have the same geometry
          hasGeom = true;
          ProblemSpecP geomBasedSpec = pICSpec->findBlock("Geometry");
          double seed = 0.0;
          geomBasedSpec->getAttribute("seed",seed);
          // parse all intrusions
          for( ProblemSpecP intrusionParams = geomBasedSpec->findBlock("geom_object");
              intrusionParams != nullptr;
              intrusionParams = intrusionParams->findNextBlock("geom_object") )
          {
            GeometryPieceFactory::create(intrusionParams, geomObjects);
          }
        }
      }
    }
    
    for(int m = 0;m<matls->size();m++){
      const int matl = matls->get(m);
      std::map<int,long64>& lastPIDPerPatch = lastPIDPerMaterialPerPatch_[m];
      std::map<int,ParticleSubset*>& thisMatDelSet = deleteSets_[m];
      for(int p=0;p<patches->size();p++){
        const Patch* patch = patches->get(p);
        const int patchID = patch->getID();
    
        // create lastPIDPerPatch_ map
        lastPIDPerPatch.insert( std::pair<int, long64>(patchID, 0 ) );
        // create an empty delete set
        thisMatDelSet.insert( std::pair<int, ParticleSubset*>(patch->getID(), scinew ParticleSubset(0,matl,patch)));
        
        // If the particle position initialization is bounded, make sure that the bounds are within
        // this patch. If the bounds are NOT, then set the number of particles on this patch to 0.
        // Also, since the user specifies the number of particles per cell, we need to count the number
        // of cells that fall within the specified bounds of the initialization and multiply that
        // by the number of particles per cell to get the total number of particles in this patch
        unsigned int nCells = 0;
        if( bounded ){
          const Point low  = patch->getBox().lower();
          const Point high = patch->getBox().upper();
          if(   xmin >= high.x() || ymin >= high.y() || zmin >= high.z()
             || xmax <= low.x()  || ymax <= low.y()  || zmax <= low.z()  ){
            // no particles will be created in this patch
            nCells = 0;
          }
          else {
            // count the number of cells that we will initialize particles in
            for( CellIterator iter(patch->getCellIterator()); !iter.done(); ++iter ){
              const IntVector iCell = *iter;
              const Point p = patch->getCellPosition(iCell);
              if( p.x() <= xmax && p.x() >= xmin &&
                  p.y() <= ymax && p.y() >= ymin &&
                  p.z() <= zmax && p.z() >= zmin ){
                nCells++;
              }
            }
          }
        }
        else {
          nCells = patch->getNumCells();
        }
        
        if( hasGeom ){
          int nPatchCells = 0;
          std::vector<GeometryPieceP>::iterator geomIter;
          // get the total cells inside the geometries
          nCells = 0;
          bool isInside;
          for(CellIterator iter(patch->getCellIterator()); !iter.done(); iter++){
            nPatchCells++;
            IntVector iCell = *iter;
            // loop over all geometry objects
            geomIter = geomObjects.begin();
            Uintah::Point p = patch->getCellPosition(iCell);
            while( geomIter != geomObjects.end() ){
              isInside = (*geomIter)->inside(p);
              if( isInside ) nCells++;
              ++geomIter;
            }
          }
        }
        
        nCells = std::min(maxParticles_, nCells);
        
        const int nParticles = pPerCell_ * nCells;
        // create a subset with the correct number of particles. This will serve as the initial memory
        // block for particles
        ParticleSubset* subset = new_dw->createParticleSubset(nParticles,matl,patch);
        
        // allocate memory for Uintah particle position and particle IDs
        ParticleVariable<Point>  ppos;
        ParticleVariable<long64> pid;
        new_dw->allocateAndPut(ppos,    pPosLabel_,           subset);
        new_dw->allocateAndPut(pid,    pIDLabel_,           subset);
        for( int i=0; i < nParticles; i++ ){
          pid[i] = i + patch->getID() * nParticles;
        }
        lastPIDPerPatch[patchID] = nParticles > 0 ? pid[nParticles-1] : 0;
      }
    }
  }
  
  //------------------------------------------------------------------
  
  void
  WasatchParticlesHelper::sync_with_wasatch( Wasatch* const wasatch )
  {
    wasatch_ = wasatch;
    wasatchSync_ = true;
  }
  
  //--------------------------------------------------------------------
  
} /* namespace WasatchCore */
