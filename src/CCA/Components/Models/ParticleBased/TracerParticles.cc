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

#include <CCA/Components/ICE/Core/ICELabel.h>
#include <CCA/Components/Models/ParticleBased/TracerParticles.h>
#include <Core/Grid/Variables/PerPatchVars.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/GeometryPiece/GeometryPieceFactory.h>
#include <Core/GeometryPiece/UnionGeometryPiece.h>
#include <Core/Grid/Box.h>
#include <Core/Grid/DbgOutput.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Material.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Variables/PerPatch.h>

#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Exceptions/ParameterNotFound.h>
#include <Core/Util/DOUT.hpp>

#include <iostream>
#include <cstdio>

using namespace Uintah;
using namespace std;

#define proc0cout_cmp(X,Y) if( isProc0_macro && X == Y) std::cout

Dout dout_models_tp("Models_tracerParticles", "Models::TracerParticles", "Models::TracerParticles debug stream", false);
//______________________________________________________________________
TracerParticles::TracerParticles(const ProcessorGroup  * myworld,
                                 const MaterialManagerP& materialManager,
                                 const ProblemSpecP    & params)
    : ParticleModel(myworld, materialManager), d_params(params)
{
  d_matl_set = {nullptr};
  Ilb  = scinew ICELabel();
  
  
  pXLabel             = VarLabel::create( "p.x",  // The varlabel is hardcoded to p.x to match MPM.
                              d_Part_point,
                              IntVector(0,0,0), 
                              VarLabel::PositionVariable);

  pXLabel_preReloc    = VarLabel::create( "p.x+",
                              d_Part_point,
                              IntVector(0,0,0), 
                              VarLabel::PositionVariable);
                              
  pDispLabel          = VarLabel::create( "p.displacement",  d_Part_Vector );
  pDispLabel_preReloc = VarLabel::create( "p.displacement+", d_Part_Vector ); 
  pIDLabel            = VarLabel::create( "p.particleID",    d_Part_long64 );
  pIDLabel_preReloc   = VarLabel::create( "p.particleID+",   d_Part_long64 );

  d_oldLabels.push_back( pIDLabel_preReloc );
  d_oldLabels.push_back( pDispLabel_preReloc );
  
  d_newLabels.push_back( pIDLabel );
  d_newLabels.push_back( pDispLabel );
  
  nPPCLabel = VarLabel::create("nPPC", CCVariable<int>::getTypeDescription() );
  
}

//______________________________________________________________________
//
TracerParticles::~TracerParticles()
{
  if(d_matl_set && d_matl_set->removeReference()) {
    delete d_matl_set;
  }
  
  // VarLabels
  VarLabel::destroy( pXLabel );
  VarLabel::destroy( pXLabel_preReloc );
  
  VarLabel::destroy( pDispLabel );
  VarLabel::destroy( pDispLabel_preReloc );
  
  VarLabel::destroy( pIDLabel );
  VarLabel::destroy( pIDLabel_preReloc );
  VarLabel::destroy( nPPCLabel ) ;

  delete Ilb;

  // regions used during initialization
  for(vector<Region*>::iterator iter = d_tracer->regions.begin();
                                iter != d_tracer->regions.end(); iter++){
    Region* region = *iter;
    delete region;
  }

  // Interior regions
  for(vector<interiorRegion*>::iterator iter = d_tracer->interiorRegions.begin();
                                        iter != d_tracer->interiorRegions.end(); iter++){
    interiorRegion* region = *iter;
    delete region;
  }

  delete d_tracer;

}

//______________________________________________________________________
//
TracerParticles::Region::Region(GeometryPieceP piece, 
                                ProblemSpecP & ps)
  : piece(piece)
{
  ps->require("res", ppc);

}

//______________________________________________________________________
//
TracerParticles::interiorRegion::interiorRegion(GeometryPieceP piece, 
                                                ProblemSpecP  & ps)
  : piece(piece)
{
  ps->require("res", ppc);
}


//______________________________________________________________________
//     P R O B L E M   S E T U P
void TracerParticles::problemSetup( GridP&, 
                                    const bool isRestart)
{
  DOUT(dout_models_tp, "Doing racerParticles::problemSetup" );

  ProblemSpecP PS_ps = d_params->findBlock("TracerParticles");
  d_matl = m_materialManager->parseAndLookupMaterial(PS_ps, "material");

  vector<int> m(1);
  m[0] = d_matl->getDWIndex();
  d_matl_set = scinew MaterialSet();
  d_matl_set->addAll(m);
  d_matl_set->addReference();
  d_matl_mss = d_matl_set->getUnion();




  //__________________________________
  //
  ProblemSpecP tracer_ps = PS_ps->findBlock("tracer");
  if (!tracer_ps){
    throw ProblemSetupException("TracerParticles: Couldn't find tracer tag", __FILE__, __LINE__);
  }

  std::string name {""};
  tracer_ps->getAttribute( "name", name );
  if ( name ==""){
    throw ProblemSetupException("TracerParticles: the tracer tag must have a valid name  <tracer name=X>", __FILE__, __LINE__);
  }
  std::string fullName = "tracer-"+name;

  d_tracer = scinew Tracer();
  d_tracer->name     = name;
  d_tracer->fullName = fullName;

  //__________________________________
  //  Initialization: Read in the geometry objects for the scalar
  ProblemSpecP init_ps = tracer_ps->findBlock("initialization");

  if( !isRestart ){

   for ( ProblemSpecP geom_obj_ps = init_ps->findBlock("geom_object"); geom_obj_ps != nullptr; geom_obj_ps = geom_obj_ps->findNextBlock("geom_object") ) {

    vector<GeometryPieceP> pieces;
    GeometryPieceFactory::create(geom_obj_ps, pieces);

    GeometryPieceP mainpiece;
    if(pieces.size() == 0){
      throw ParameterNotFound("No piece specified in geom_object", __FILE__, __LINE__);
    }
    else if(pieces.size() > 1){
      mainpiece = scinew UnionGeometryPiece(pieces);
    }
    else {
      mainpiece = pieces[0];
    }

    d_tracer->regions.push_back( scinew TracerParticles::Region(mainpiece, geom_obj_ps) );
   }
  }

  if( d_tracer->regions.size() == 0 && !isRestart) {
    throw ProblemSetupException("Variable: "+fullName +" does not have any initial value regions", __FILE__, __LINE__);
  }

  //__________________________________
  //  Read in interior geometry objects for injecting a scalar in the domain
  ProblemSpecP srcs_ps = tracer_ps->findBlock("interiorSources");

  if( srcs_ps ) {

    for (ProblemSpecP geom_obj_ps = srcs_ps->findBlock("geom_object"); geom_obj_ps != nullptr; geom_obj_ps = geom_obj_ps->findNextBlock("geom_object") ) {
      vector<GeometryPieceP> pieces;
      GeometryPieceFactory::create(geom_obj_ps, pieces);

      GeometryPieceP mainpiece;
      if(pieces.size() == 0){
        throw ParameterNotFound("No piece specified in geom_object", __FILE__, __LINE__);
      }
      else if(pieces.size() > 1){
        mainpiece = scinew UnionGeometryPiece(pieces);
      }
      else {
        mainpiece = pieces[0];
      }

      d_tracer->interiorRegions.push_back( scinew TracerParticles::interiorRegion(mainpiece, geom_obj_ps) );
    }
  }
}

//______________________________________________________________________
//  Function:  TracerParticles::outputProblemSpec
//  Purpose:   Output to the checkpoints variables needed for a restart
//______________________________________________________________________
void TracerParticles::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP model_ps = ps->appendChild("Model");
  model_ps->setAttribute( "type","TracerParticles" );
  ProblemSpecP PS_ps = model_ps->appendChild( "TracerParticles" );

  PS_ps->appendElement( "material", d_matl->getName() );
  ProblemSpecP scalar_ps = PS_ps->appendChild( "tracer" );
  scalar_ps->setAttribute( "name", d_tracer->name );


  //__________________________________
  //  initialization regions
  ProblemSpecP init_ps = scalar_ps->appendChild( "initialization" );

  vector<Region*>::const_iterator iter;
  for ( iter = d_tracer->regions.begin(); iter != d_tracer->regions.end(); iter++) {
    ProblemSpecP geom_ps = init_ps->appendChild( "geom_object" );

    (*iter)->piece->outputProblemSpec(geom_ps);
    geom_ps->appendElement( "res",(*iter)->ppc );
  }


  //__________________________________
  //  regions inside the domain
  if( d_tracer->interiorRegions.size() > 0 ){
    ProblemSpecP int_ps = scalar_ps->appendChild( "interiorSources" );

    vector<interiorRegion*>::const_iterator itr;
    for ( itr = d_tracer->interiorRegions.begin(); itr != d_tracer->interiorRegions.end(); itr++) {
      ProblemSpecP geom_ps = int_ps->appendChild("geom_object");
      (*itr)->piece->outputProblemSpec( geom_ps );

      geom_ps->appendElement( "res",(*iter)->ppc );
    }
  }
}


//______________________________________________________________________
//      S C H E D U L E   I N I T I A L I Z E
void TracerParticles::scheduleInitialize(SchedulerP   & sched,
                                         const LevelP & level)
{
  const string taskName = "TracerParticles::scheduleInitialize_("+ d_tracer->fullName+")";
  printSchedule(level,dout_models_tp,taskName);

  Task* t = scinew Task(taskName, this, &TracerParticles::initialize);
  
  t->computes( nPPCLabel,  d_matl_mss );
  t->computes( pXLabel,    d_matl_mss );
  t->computes( pDispLabel, d_matl_mss );
  t->computes( pIDLabel,   d_matl_mss );
  
  sched->addTask(t, level->eachPatch(), d_matl_set);

}

//______________________________________________________________________
//   Function:  TracerParticles::countParticles
//   Purpose:    Count the number of particles on a patch that are in 
//               all the user defined regions.
//______________________________________________________________________
unsigned int TracerParticles::countParticles( const Patch   * patch,
                                              regionPoints  & pPositions)
{
  unsigned int count = 0;
  
  for(vector<Region*>::iterator iter = d_tracer->regions.begin();
                                iter != d_tracer->regions.end(); iter++){
    Region* region = *iter;
    
    // is this region contained on this patch
    GeometryPieceP piece = region->piece;
    Box b1 = piece->getBoundingBox();
    Box b2 = patch->getExtraBox();
    Box b  = b1.intersect(b2);
    if( b.degenerate() ){
       return 0;
    }
    
    IntVector ppc     = region->ppc;
    Vector    dxpp    = patch->dCell()/ppc;
    Vector    dcorner = dxpp*0.5;

    for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
      Point lower = patch->nodePosition(*iter) + dcorner;
      IntVector c = *iter;

      for(int ix=0;ix < ppc.x(); ix++){
        for(int iy=0;iy < ppc.y(); iy++){
          for(int iz=0;iz < ppc.z(); iz++){

            Vector idx( (double)ix, 
                        (double)iy, 
                        (double)iz);

            Point p = lower + dxpp*idx;

            if ( !b2.contains(p) ){
              throw InternalError("Particle created outside of patch?", __FILE__, __LINE__);
            }

            if ( piece->inside(p,true) ){ 
              pPositions[region].push_back(p);
              count++;
            }
          }  // z
        }  // y
      }  // x
    }  // cell
  }  // region
  return count;
}



//______________________________________________________________________
//  Task:  TracerParticles::initialize
//  Purpose:  Create the particles on all patches 
//______________________________________________________________________
void TracerParticles::initialize(const ProcessorGroup *,
                                 const PatchSubset    * patches,
                                 const MaterialSubset * matl_mss,
                                 DataWarehouse        *,
                                 DataWarehouse        * new_dw)
{
  //__________________________________
  // Patches loop
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    const string msg = "Doing TracerParticles::initialize_("+ d_tracer->fullName+")";
    printTask(patches, patch, dout_models_tp, msg);

    //__________________________________
    //  Count the number of particles in all the regions on this patch
    //  Return the positions of the particles
    regionPoints pPositions;
    unsigned int nParticles;
    nParticles = TracerParticles::countParticles( patch, pPositions );

    //__________________________________
    //  allocate the particle variables
    ParticleVariable<Point>  pX;
    ParticleVariable<Vector> pDisp;
    ParticleVariable<long64> pID;
    CCVariable<int>          nPPC;    // number of particles per cell
    
    int indx = d_matl->getDWIndex();
    ParticleSubset* part_ss = new_dw->createParticleSubset( nParticles, indx, patch );
    
    new_dw->allocateAndPut( pX,    pXLabel,     part_ss );
    new_dw->allocateAndPut( pDisp, pDispLabel,  part_ss );
    new_dw->allocateAndPut( pID,   pIDLabel,    part_ss );
    new_dw->allocateAndPut( nPPC,  nPPCLabel, indx, patch );
    nPPC.initialize(0);
    
    int pIndx = 0;
    //__________________________________
    //  Region loop
    for( vector<Region*>::iterator iter = d_tracer->regions.begin();
                                   iter != d_tracer->regions.end(); iter++){
      Region* region = *iter;
      
      // ignore if region isn't contained in the patch
      GeometryPieceP piece = region->piece;
      Box b2 = patch->getExtraBox();
      Box b1 = piece->getBoundingBox();
      Box b  = b1.intersect(b2);
      
      if( b.degenerate() ) {
        continue;
      }      
      
      vector<Point>::const_iterator itr;
      for(itr=pPositions[region].begin();itr!=pPositions[region].end(); ++itr){
        const Point pos = *itr;
        
        IntVector cell_indx;
        if ( !patch->findCell( pos,cell_indx ) ) {
          continue;
        }
        if (!patch->containsPoint( pos )) {
          continue;
        }
        
        pX[pIndx] = pos;
        pDisp[pIndx] = Vector(0);
        
        ASSERT(cell_indx.x() <= 0xffff && 
               cell_indx.y() <= 0xffff && 
               cell_indx.z() <= 0xffff);
         
        long64 cellID = ((long64)cell_indx.x() << 16) | 
                        ((long64)cell_indx.y() << 32) | 
                        ((long64)cell_indx.z() << 48);
        
        int& my_nPPC = nPPC[cell_indx];
        pID[pIndx] = (cellID | (long64) my_nPPC);
        ASSERT(my_nPPC < 0x7fff);
        my_nPPC++;
         
        pIndx++;
        
      }  // particles
    }  // regions
  }  // patches
}


//______________________________________________________________________
void TracerParticles::scheduleComputeModelSources(SchedulerP  & sched,
                                                  const LevelP& level)
{
  const string taskName = "TracerParticles::scheduleComputeModelSources_("+ d_tracer->fullName+")";
  printSchedule(level,dout_models_tp, taskName);

  sched_updateParticles( sched, level );
}

//______________________________________________________________________
//  Task: updateParticles
//  Purpose:  Update the particles:
//              - position
//              - displacement
//              - delete any particles outside the domain
//______________________________________________________________________
void TracerParticles::sched_updateParticles(SchedulerP  & sched,
                                            const LevelP& level)
{

  const string schedName = "sched_updateParticles_("+ d_tracer->fullName+")";
  printSchedule( level ,dout_models_tp, schedName);

  const string taskName = "TracerParticles::updateParticles_("+ d_tracer->fullName+")";
  Task* t = scinew Task( taskName, this, &TracerParticles::updateParticles);

  t->requires( Task::OldDW, Ilb->delTLabel, level.get_rep() );

  t->requires( Task::OldDW, pXLabel,     d_matl_mss, d_gn );
  t->requires( Task::OldDW, pDispLabel,  d_matl_mss, d_gn );
  t->requires( Task::OldDW, pIDLabel,    d_matl_mss, d_gn );
    
  t->requires( Task::OldDW, Ilb->vel_CCLabel, d_matl_mss, d_gn );
  
  t->computes( pXLabel_preReloc,    d_matl_mss ); 
  t->computes( pDispLabel_preReloc, d_matl_mss );
  t->computes( pIDLabel_preReloc,   d_matl_mss );

  sched->addTask(t, level->eachPatch(), d_matl_set);
}

//______________________________________________________________________
//
//______________________________________________________________________
void TracerParticles::updateParticles(const ProcessorGroup  *,
                                      const PatchSubset     * patches,
                                      const MaterialSubset  * matls,
                                      DataWarehouse         * old_dw,
                                      DataWarehouse         * new_dw)
{
  const Level* level = getLevel(patches);
  delt_vartype delT;
  old_dw->get(delT, Ilb->delTLabel, level);

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    const string msg = "Doing TracerParticles::updateParticles_("+ d_tracer->fullName+")";
    printTask(patches, patch, dout_models_tp, msg);

    //__________________________________
    //  
    ParticleVariable<Point>  pX;
    ParticleVariable<Vector> pDisp;
    ParticleVariable<long64> pID;

    constParticleVariable<Point>  pX_old;
    constParticleVariable<Vector> pDisp_old;
    constParticleVariable<long64> pID_old;
    constCCVariable<Vector>       vel_CC;
    
    int matlIndx = d_matl->getDWIndex();
    ParticleSubset* pset   = old_dw->getParticleSubset( matlIndx, patch );
    ParticleSubset* delset = scinew ParticleSubset(0, matlIndx, patch);
    
    old_dw->get( vel_CC,           Ilb->vel_CCLabel, matlIndx, patch, d_gn, 0);
    old_dw->get( pX_old,           pXLabel,              pset );           
    old_dw->get( pDisp_old,        pDispLabel,           pset );   
    old_dw->get( pID_old,          pIDLabel,             pset );            
    
    new_dw->allocateAndPut( pX,    pXLabel_preReloc,     pset );             
    new_dw->allocateAndPut( pDisp, pDispLabel_preReloc,  pset );   
    new_dw->allocateAndPut( pID,   pIDLabel_preReloc,    pset );         
    pID.copyData( pID_old );

    BBox compDomain;
    GridP grid = level->getGrid();
    grid->getInteriorSpatialRange(compDomain);
   
    //__________________________________
    //  Update particle postion and displacement
    for(ParticleSubset::iterator iter = pset->begin();iter != pset->end(); iter++){
      particleIndex idx = *iter;
      
      IntVector cell_indx;
      if ( !patch->findCell( pX_old[idx],cell_indx ) ) {
        continue;
      }
       
      pX[idx]    = pX_old[idx] + vel_CC[cell_indx]*delT;
      pDisp[idx] = pDisp_old[idx]  + ( pX[idx] - pX_old[idx] );
      
      //__________________________________
      //  delete particles that are ouside the domain
      if ( ! compDomain.inside( pX[idx] ) ){
        delset->addParticle(idx);
      }
      
    }
    new_dw->deleteParticles(delset);
  }
}
