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

#include <Core/Math/MersenneTwister.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/GeometryPiece/GeometryPieceFactory.h>
#include <Core/GeometryPiece/UnionGeometryPiece.h>
#include <Core/Grid/Box.h>
#include <Core/Grid/DbgOutput.h>
#include <Core/Grid/Material.h>
#include <Core/Math/MiscMath.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Variables/PerPatchVars.h>

#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Exceptions/ParameterNotFound.h>
#include <Core/Util/DOUT.hpp>

#include <iostream>

using namespace Uintah;
using std::string;
using std::vector;
using std::ostringstream;

//______________________________________________________________________
//To Do:
//  -Add start and stop times.
//  -Test restarts
//  -Can only setVariables from old_DW
//  - Must initialze variables after adding them.
#define proc0cout_eq(X,Y) if( isProc0_macro && X == Y) std::cout
#define proc0cout_gt(X,Y) if( isProc0_macro && X >= Y) std::cout
#define proc0cout_lt(X,Y) if( isProc0_macro && X <= Y) std::cout

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
  for(vector<Region*>::iterator iter = d_tracer->interiorRegions.begin();
                                iter != d_tracer->interiorRegions.end(); iter++){
    Region* region = *iter;
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
  ps->get( "particlesPerCell",          particlesPerCell );
  ps->get( "particlesPerCellPerSecond", particlesPerCellPerSecond );

  string nodeName = ps->getParent()->getNodeName();
  if (nodeName == "interiorSources" ){
    isInteriorRegion = true;
  }
}
//______________________________________________________________________
//  "That C++11 doesn't include make_unique is partly an oversight, and it will
//   almost certainly be added in the future. In the meantime, use the one provided below."
//     - Herb Sutter, chair of the C++ standardization committee
//
//   Once C++14 is adpoted delete this
template<typename T, typename ...Args>
std::unique_ptr<T> TracerParticles::make_unique( Args&& ...args )
{
  return std::unique_ptr<T>( new T( std::forward<Args>(args)... ) );
}

//______________________________________________________________________
//     P R O B L E M   S E T U P
void TracerParticles::problemSetup( GridP&,
                                    const bool isRestart)
{
  DOUT(dout_models_tp, "Doing racerParticles::problemSetup" );

  ProblemSpecP TP_ps = d_params->findBlock("TracerParticles");
  d_matl = m_materialManager->parseAndLookupMaterial(TP_ps, "material");

  vector<int> m(1);
  m[0] = d_matl->getDWIndex();
  d_matl_set = scinew MaterialSet();
  d_matl_set->addAll(m);
  d_matl_set->addReference();
  d_matl_mss = d_matl_set->getUnion();

  //__________________________________
  //
  ProblemSpecP tracer_ps = TP_ps->findBlock("tracer");
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

    Region* region = scinew TracerParticles::Region(mainpiece, geom_obj_ps);
    d_tracer->regions.push_back( region );
   }
  }

  // bulletproofing
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

      Region* region = scinew TracerParticles::Region(mainpiece, geom_obj_ps);
      d_tracer->interiorRegions.push_back( region );
    }
  }

  //__________________________________
  //  CCVariable values to be copied
  ProblemSpecP vars_ps = TP_ps->findBlock("newVariables");
  if ( vars_ps ){

    static int count=1;

    proc0cout_eq( count, 1 ) << "\n__________________________________TracerParticles\n";

    for( ProblemSpecP label_spec = vars_ps->findBlock( "CCVarLabel" ); label_spec != nullptr; label_spec = label_spec->findNextBlock( "CCVarLabel" ) ) {

      std::map<string,string> attribute;
      label_spec->getAttributes( attribute );

      //__________________________________
      // label name
      string labelName = attribute["label"];
      VarLabel* label = VarLabel::find( labelName, "ERROR  TracerParticles::problemSetup" );

      //__________________________________
      //  bulletproofing
      const TypeDescription* td      = label->typeDescription();
      const TypeDescription* subtype = td->getSubType();

      const TypeDescription::Type baseType = td->getType();
      const TypeDescription::Type subType  = subtype->getType();

      // CC Variables and only doubles
      if(baseType != TypeDescription::CCVariable &&
         subType  != TypeDescription::double_type   ){
        ostringstream warn;
        warn << "ERROR:TracerParticles: ("<<label->getName() << " "
             << " only CCVariable<double> variables work";
        throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
      }

      //__________________________________
      // define particle quantity label names
      std::string L1 = "p."+labelName+"+";
      std::string L2 = "p."+labelName;
      VarLabel* QLabel_preReloc   = VarLabel::create( L1, d_Part_double );
      VarLabel* QLabel            = VarLabel::create( L2, d_Part_double );
      proc0cout_eq( count, 1 ) << "   Created labels (" << L1 << ") (" << L2 << ")\n";

      //__________________________________
      //  populate the vector of particle variables
      auto me               = make_unique< Qvar >();
      me->CCVarLabel        = label;
      me->pQLabel_preReloc  = QLabel_preReloc;
      me->pQLabel           = QLabel;
      d_Qvars.push_back( move(me) );

      // used for the relocation task
      d_oldLabels.push_back( QLabel_preReloc );
      d_newLabels.push_back( QLabel );
    }
    proc0cout_eq( count, 1 ) << "__________________________________\n";
    count ++;
  } else {
    proc0cout << "\nTracerParticles:WARNING Couldn't find <Variables> tag.\n";
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
    Region* region = *iter;

    ProblemSpecP geom_ps = init_ps->appendChild( "geom_object" );

    region->piece->outputProblemSpec(geom_ps);
    geom_ps->appendElement( "isInteriorRegion",   region->isInteriorRegion );
    geom_ps->appendElement( "particlesPerCell",   region->particlesPerCell );
  }


  //__________________________________
  //  regions for particle injection
  if( d_tracer->interiorRegions.size() > 0 ){
    ProblemSpecP int_ps = scalar_ps->appendChild( "interiorSources" );

    vector<Region*>::const_iterator itr;
    for ( iter = d_tracer->interiorRegions.begin(); iter != d_tracer->interiorRegions.end(); iter++) {
      Region* region = *iter;

      ProblemSpecP geom_ps = int_ps->appendChild("geom_object");
      region->piece->outputProblemSpec( geom_ps );

      geom_ps->appendElement( "isInteriorRegion",         region->isInteriorRegion );
      geom_ps->appendElement( "particlesPerCellPerSecond",region->particlesPerCellPerSecond );
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

  for ( size_t i=0 ; i<d_Qvars.size(); i++ ) {
    std::shared_ptr<Qvar> Q = d_Qvars[i];
    t->computes ( Q->pQLabel, d_matl_mss );
  }

  sched->addTask(t, level->eachPatch(), d_matl_set);
}

//______________________________________________________________________
//   Function:  TracerParticles::countParticles
//   Purpose:    Count the number of particles on a patch inside
//                user defined regions.
//              - randomly lay down particles in each cell in each region
//______________________________________________________________________
unsigned int TracerParticles::distributeParticles( const Patch   * patch,
                                                   const double    delT,
                                                   const std::vector<Region*> regions,
                                                   regionPoints  & pPositions)
{
  unsigned int count = 0;

  for( auto r_iter = regions.begin(); r_iter != regions.end(); r_iter++){
    Region* region = *r_iter;

    // is this region contained on this patch
    GeometryPieceP piece = region->piece;
    Box b1 = piece->getBoundingBox();
    Box b2 = patch->getExtraBox();
    Box b  = b1.intersect(b2);
    if( b.degenerate() ){
       continue;
    }

    Vector dx = patch->dCell();
    Vector dx_2 = dx/Vector(2);
    MTRand mTwister;

    int nParticlesPerCell = region->particlesPerCell;

    // for interior regions compute PPC
    if( region->isInteriorRegion ) {

      region->elapsedTime += delT;
      double elapsedTime = region->elapsedTime;

      nParticlesPerCell = Uintah::RoundDown(region->particlesPerCellPerSecond * elapsedTime);

      // reset if particles are added
      if( nParticlesPerCell > 0 ){
        region->elapsedTime = 0;
      }
    }

    //__________________________________
    //  Loop over all cells and laydown particles
    for(CellIterator c_iter = patch->getCellIterator(); !c_iter.done(); c_iter++){
      IntVector c = *c_iter;

      Point CC_pos =patch->cellPosition(c);

      if ( piece->inside(CC_pos,true) ){

        Point lower = CC_pos - dx_2;

        for(int i=0; i<nParticlesPerCell; i++){

          if ( !b2.contains(CC_pos) ){
            throw InternalError("Particle created outside of patch?", __FILE__, __LINE__);
          }

          // generate a random point inside this cell
          double x = mTwister.rand() * dx.x();
          double y = mTwister.rand() * dx.y();
          double z = mTwister.rand() * dx.z();
          Point p;
          p.x( lower.x()  + x );
          p.y( lower.y()  + y );
          p.z( lower.z()  + z );

          pPositions[region].push_back(p);
          count++;
        }
      }  //inside
    }  // cell
  }  // region

  if( count > 0 ){
    ostringstream msg;
    msg <<" TracerParticles::addParticles  patch-"<<patch->getID()
        << " adding " << count << " particles ";
    DOUTR( true, msg.str() );
  }
  return count;
}


//______________________________________________________________________
//  function:  initializeRegions
//______________________________________________________________________
void TracerParticles::initializeRegions( const Patch   * patch,
                                         unsigned int    pIndx,
                                         regionPoints  & pPositions,
                                         std::vector<Region*> regions,
                                         ParticleVariable<Point> & pX,
                                         ParticleVariable<Vector>& pDisp,
                                         ParticleVariable<long64>& pID,
                                         CCVariable<int>         & nPPC )
{
  //__________________________________
  //  Region loop
  for( vector<Region*>::iterator iter = regions.begin(); iter !=regions.end(); iter++){
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

      if (!patch->containsPoint( pos )) {
        continue;
      }

      IntVector cell_indx;
      if ( !patch->findCell( pos,cell_indx ) ) {
        continue;
      }

      DOUTR( true, " initializeRegions: patch: " << patch->getID() << " pIdx: " << pIndx << " pos " << pos );

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
    // For each Region on this patch
    //  - Laydown particles in each region.
    //  - return the positions of the particles
    //  - Count the number of particles
    //
    regionPoints pPositions;
    unsigned int nParticles;
    const double ignore = -9;
    nParticles = TracerParticles::distributeParticles( patch,
                                                       ignore,
                                                       d_tracer->regions,
                                                       pPositions );

    //__________________________________
    //  allocate the particle variables
    ParticleVariable<Point>  pX;
    ParticleVariable<Vector> pDisp;
    ParticleVariable<long64> pID;
    CCVariable<int>          nPPC;    // number of particles per cell

    int indx = d_matl->getDWIndex();
    ParticleSubset* pset = new_dw->createParticleSubset( nParticles, indx, patch );

    new_dw->allocateAndPut( pX,    pXLabel,     pset );
    new_dw->allocateAndPut( pDisp, pDispLabel,  pset );
    new_dw->allocateAndPut( pID,   pIDLabel,    pset );
    new_dw->allocateAndPut( nPPC,  nPPCLabel, indx, patch );
    nPPC.initialize(0);

    if (nParticles == 0 ) {
      continue;
    }

    int pIndx = 0;

    initializeRegions(  patch, pIndx, pPositions, d_tracer->regions,
                          pX,  pDisp, pID, nPPC );

    //__________________________________
    //  The additional varialbs
    for ( size_t i=0 ; i<d_Qvars.size(); i++ ) {
      std::shared_ptr<Qvar> Q = d_Qvars[i];

      ParticleVariable<double> pQ;
      new_dw->allocateAndPut( pQ, Q->pQLabel_preReloc, pset );

      for(ParticleSubset::iterator iter = pset->begin();iter != pset->end(); iter++){
        particleIndex idx = *iter;
        pQ[idx] = 0.0;
      }
    }  // Qvars loop
  }  // patches
}


//______________________________________________________________________
void TracerParticles::scheduleComputeModelSources(SchedulerP  & sched,
                                                  const LevelP& level)
{
  const string taskName = "TracerParticles::scheduleComputeModelSources_("+ d_tracer->fullName+")";
  printSchedule(level,dout_models_tp, taskName);

  sched_moveParticles(   sched, level );

  sched_setParticleVars( sched, level );

  sched_addParticles(    sched, level );
}

//______________________________________________________________________
//  Task: moveParticles
//  Purpose:  Update the particles:
//              - position
//              - displacement
//              - delete any particles outside the domain
//______________________________________________________________________
void TracerParticles::sched_moveParticles(SchedulerP  & sched,
                                          const LevelP& level)
{

  const string schedName = "sched_moveParticles_("+ d_tracer->fullName+")";
  printSchedule( level ,dout_models_tp, schedName);

  const string taskName = "TracerParticles::moveParticles_("+ d_tracer->fullName+")";
  Task* t = scinew Task( taskName, this, &TracerParticles::moveParticles);

  t->requires( Task::OldDW, Ilb->delTLabel, level.get_rep() );

  t->requires( Task::OldDW, pXLabel,     d_matl_mss, d_gn );
  t->requires( Task::OldDW, pDispLabel,  d_matl_mss, d_gn );
  t->requires( Task::OldDW, pIDLabel,    d_matl_mss, d_gn );

  t->requires( Task::OldDW, Ilb->vel_CCLabel, d_matl_mss, d_gn );   // hardwired to use ICE's velocity

  t->computes( pXLabel_preReloc,    d_matl_mss );
  t->computes( pDispLabel_preReloc, d_matl_mss );
  t->computes( pIDLabel_preReloc,   d_matl_mss );

  sched->addTask(t, level->eachPatch(), d_matl_set);
}

//______________________________________________________________________
//
//______________________________________________________________________
void TracerParticles::moveParticles(const ProcessorGroup  *,
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

    const string msg = "Doing TracerParticles::moveParticles_("+ d_tracer->fullName+")";
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
    grid->getInteriorSpatialRange( compDomain );

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

//______________________________________________________________________
//  Task: addParticles
//  Purpose:  add particles to user defined regions at a user defined rate:
//            This task is called after particles have been moved
//______________________________________________________________________
void TracerParticles::sched_addParticles( SchedulerP  & sched,
                                          const LevelP& level)
{
  if ( d_tracer->interiorRegions.size() == 0 ){
    return;
  }

  const string schedName = "sched_addParticles_("+ d_tracer->fullName+")";
  printSchedule( level ,dout_models_tp, schedName);

  const string taskName = "TracerParticles::addParticles_("+ d_tracer->fullName+")";
  Task* t = scinew Task( taskName, this, &TracerParticles::addParticles);

  t->requires( Task::OldDW, Ilb->delTLabel, level.get_rep() );
  t->requires( Task::OldDW, nPPCLabel,     d_matl_mss, d_gn );

  t->modifies( pXLabel_preReloc,    d_matl_mss );
  t->modifies( pDispLabel_preReloc, d_matl_mss );
  t->modifies( pIDLabel_preReloc,   d_matl_mss );
  t->computes( nPPCLabel,           d_matl_mss );

  sched->addTask(t, level->eachPatch(), d_matl_set);
}

//______________________________________________________________________
//
void TracerParticles::addParticles(const ProcessorGroup  *,
                                   const PatchSubset     * patches,
                                   const MaterialSubset  * ,
                                   DataWarehouse         * old_dw,
                                   DataWarehouse         * new_dw)
{
  const Level* level = getLevel(patches);
  delt_vartype delT;
  old_dw->get(delT, Ilb->delTLabel, level);

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    const string msg = "Doing TracerParticles::addParticles_("+ d_tracer->fullName+")";
    printTask(patches, patch, dout_models_tp, msg);

    //__________________________________
    // For each interiorRegion on this patch
    //  - distribute particles in each region.
    //  - return the positions of the particles
    //  - Count the number of particles
    //
    regionPoints pPositions;
    unsigned int nParticles;

    nParticles = TracerParticles::distributeParticles( patch,
                                                       delT,
                                                       d_tracer->interiorRegions,
                                                       pPositions );
    //__________________________________
    //
    int matlIndx = d_matl->getDWIndex();

    constCCVariable<int> nPPC_old;
    CCVariable<int>      nPPC;    // number of particles per cell

    old_dw->get(            nPPC_old, nPPCLabel, matlIndx, patch, d_gn, 0);
    new_dw->allocateAndPut( nPPC,     nPPCLabel, matlIndx, patch );
    nPPC.copyData( nPPC_old );

    ParticleVariable<Point>  pX;
    ParticleVariable<Vector> pDisp;
    ParticleVariable<long64> pID;

    ParticleSubset* pset  = old_dw->getParticleSubset( matlIndx, patch );
    unsigned int oldNumPar   = pset->addParticles( nParticles );

    new_dw->getModifiable( pX,    pXLabel_preReloc,     pset );
    new_dw->getModifiable( pDisp, pDispLabel_preReloc,  pset );
    new_dw->getModifiable( pID,   pIDLabel_preReloc,    pset );

    //__________________________________
    //  Allocate temp variables and populate them
    ParticleVariable<Point>  pX_tmp;
    ParticleVariable<Vector> pDisp_tmp;
    ParticleVariable<long64> pID_tmp;

    new_dw->allocateTemporary( pX_tmp,     pset );
    new_dw->allocateTemporary( pDisp_tmp,  pset );
    new_dw->allocateTemporary( pID_tmp,    pset );

    for( unsigned int idx=0; idx<oldNumPar; ++idx ){
      pX_tmp[idx]    = pX[idx];
      pDisp_tmp[idx] = pDisp[idx];
      pID_tmp[idx]   = pID[idx];
    }

    initializeRegions(  patch, oldNumPar, pPositions, d_tracer->interiorRegions,
                        pX_tmp,  pDisp_tmp, pID_tmp, nPPC );

     new_dw->put( pX_tmp,    pXLabel_preReloc,    true);
     new_dw->put( pDisp_tmp, pDispLabel_preReloc, true);
     new_dw->put( pID_tmp,   pIDLabel_preReloc,   true);
  }
}


//______________________________________________________________________
//  Task: setParticleVars
//  Purpose:  set the particle quantities to the corresponding CCVariables
//______________________________________________________________________
void TracerParticles::sched_setParticleVars( SchedulerP  & sched,
                                             const LevelP& level)
{
  const string schedName = "sched_setParticleVars_("+ d_tracer->fullName+")";
  printSchedule( level ,dout_models_tp, schedName);

  const string taskName = "TracerParticles::setParticleVars_("+ d_tracer->fullName+")";
  Task* t = scinew Task( taskName, this, &TracerParticles::setParticleVars);

  t->requires( Task::NewDW, pXLabel_preReloc, d_matl_mss, d_gn, 0 );

  for ( size_t i=0 ; i<d_Qvars.size(); i++ ) {
    std::shared_ptr<Qvar> Q = d_Qvars[i];

    t->requires( Task::OldDW, Q->CCVarLabel, d_matl_mss, d_gn, 0 );
    t->computes ( Q->pQLabel_preReloc, d_matl_mss );
  }

  sched->addTask(t, level->eachPatch(), d_matl_set);
}


//______________________________________________________________________
//
void TracerParticles::setParticleVars(const ProcessorGroup  *,
                                      const PatchSubset     * patches,
                                      const MaterialSubset  * ,
                                      DataWarehouse         * old_dw,
                                      DataWarehouse         * new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    const string msg = "Doing TracerParticles::setParticleVars("+ d_tracer->fullName+")";
    printTask(patches, patch, dout_models_tp, msg);

    //__________________________________
    //
    int matlIndx = d_matl->getDWIndex();
    ParticleSubset* pset  = old_dw->getParticleSubset( matlIndx, patch );

    constParticleVariable<Point>  pX;
    new_dw->get( pX, pXLabel_preReloc, pset );

    //__________________________________
    //
    for ( size_t i=0 ; i<d_Qvars.size(); i++ ) {
      std::shared_ptr<Qvar> Q = d_Qvars[i];

      constCCVariable<double>  Q_CC;
      ParticleVariable<double> pQ;

      old_dw->get( Q_CC, Q->CCVarLabel, matlIndx, patch,  d_gn, 0 );
      new_dw->allocateAndPut( pQ,   Q->pQLabel_preReloc, pset );

      for(ParticleSubset::iterator iter = pset->begin();iter != pset->end(); iter++){
        particleIndex idx = *iter;

        IntVector cell;
        if ( !patch->findCell( pX[idx],cell ) ) {
          continue;
        }

        pQ[idx] = Q_CC[cell];
      }
    }  // Qvars loop
  }  // patches
}
