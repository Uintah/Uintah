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
#include <CCA/Components/Models/FluidsBased/PassiveScalar.h>

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
using std::cout;
using std::endl;

//______________________________________________________________________
//To Do:
//  - Can only setVariables from old_DW
//  - Change name from cloneVar to cloneVar
//  - Optimization:  Read in the variable decay coefficient from the passive scalar
//    model if it's running.
//
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

  pVelocityLabel           = VarLabel::create( "p.velocity",      d_Part_Vector );
  pVelocityLabel_preReloc  = VarLabel::create( "p.velocity+",     d_Part_Vector );

  pIDLabel            = VarLabel::create( "p.particleID",    d_Part_long64 );
  pIDLabel_preReloc   = VarLabel::create( "p.particleID+",   d_Part_long64 );
  simTimeLabel        = VarLabel::create( simTime_name, simTime_vartype::getTypeDescription() );

  d_oldLabels.push_back( pIDLabel_preReloc );
  d_oldLabels.push_back( pDispLabel_preReloc );
  d_oldLabels.push_back( pVelocityLabel_preReloc );

  d_newLabels.push_back( pIDLabel );
  d_newLabels.push_back( pDispLabel );
  d_newLabels.push_back( pVelocityLabel );

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

  VarLabel::destroy( pVelocityLabel );
  VarLabel::destroy( pVelocityLabel_preReloc );

  VarLabel::destroy( pIDLabel );
  VarLabel::destroy( pIDLabel_preReloc );
  VarLabel::destroy( nPPCLabel ) ;
  VarLabel::destroy( simTimeLabel );

  delete Ilb;

  // regions used during initialization
  for(vector<Region*>::iterator iter = d_tracer->initializeRegions.begin();
                                iter != d_tracer->initializeRegions.end(); iter++){
    Region* region = *iter;
    delete region;
  }

  // Interior regions
  for(vector<Region*>::iterator iter = d_tracer->injectionRegions.begin();
                                iter != d_tracer->injectionRegions.end(); iter++){
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
  DOUT(dout_models_tp, "Doing TracerParticles::problemSetup" );

  ProblemSpecP TP_ps = d_params->findBlock("TracerParticles");

  TP_ps->get( "modelPreviouslyInitialized", d_previouslyInitialized );
  TP_ps->getWithDefault("reinitializeDomain",  d_reinitializeDomain,  false);

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

  TP_ps->getWithDefault( "timeStart", d_tracer->timeStart, 0.0);
  TP_ps->getWithDefault( "timeStop",  d_tracer->timeStop, 9.e99);


  //__________________________________
  //  bulletproofing

  if( !m_output->isLabelSaved( "p.particleID" ) ){
    throw ProblemSetupException("TracerParticles: ERROR you must add <save label=\"p.particleID\"/> to the ups file", __FILE__, __LINE__);
  }

  //__________________________________
  //  Read in all geometry objects/pieces in the <Material> node of the ups file.
  //  Needed since the user may referec
  if( d_reinitializeDomain ){

    ProblemSpecP root_ps = d_params->getRootNode();
    ProblemSpecP mat_ps = root_ps->findBlockWithOutAttribute( "MaterialProperties" );

    // find all of the geom_objects problem specs
    std::vector<ProblemSpecP> geom_objs = mat_ps->findBlocksRecursive("geom_object");

    // create geom piece if needed
    for( size_t i=0; i<geom_objs.size(); i++){

      ProblemSpecP geo_obj_ps = geom_objs[i];

      if( GeometryPieceFactory::geometryPieceExists( geo_obj_ps ) < 0 ){

        vector<GeometryPieceP> pieces;
        GeometryPieceFactory::create( geo_obj_ps, pieces );
      }
    }
  } 

  //__________________________________
  //  Initialization: Read in the geometry pieces
  if( !d_previouslyInitialized || d_reinitializeDomain ){

    ProblemSpecP init_ps = tracer_ps->findBlock("initialization");

    for ( ProblemSpecP geom_obj_ps = init_ps->findBlock("geom_object");
                       geom_obj_ps != nullptr;
                       geom_obj_ps = geom_obj_ps->findNextBlock("geom_object") ) {

      vector<GeometryPieceP> pieces;
      GeometryPieceFactory::create(geom_obj_ps, pieces);

      GeometryPieceP mainpiece;
      if( pieces.size() == 0 ){
        throw ParameterNotFound("No piece specified in geom_object", __FILE__, __LINE__);
      }
      else if( pieces.size() > 1){
        mainpiece = scinew UnionGeometryPiece(pieces);
      }
      else {
        mainpiece = pieces[0];
      }

      Region* region = scinew TracerParticles::Region(mainpiece, geom_obj_ps);
      d_tracer->initializeRegions.push_back( region );
    }
  }

  // bulletproofing
  if( d_tracer->initializeRegions.size() == 0 && !isRestart) {
    throw ProblemSetupException("Variable: "+fullName +" does not have any initial value regions", __FILE__, __LINE__);
  }

  //__________________________________
  //  Read in interior geometry pieces for injecting a scalar in the domain
  ProblemSpecP srcs_ps = tracer_ps->findBlock("interiorSources");

  if( srcs_ps ) {

    for (ProblemSpecP geom_obj_ps = srcs_ps->findBlock("geom_object");
                      geom_obj_ps != nullptr;
                      geom_obj_ps = geom_obj_ps->findNextBlock("geom_object") ) {

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
      d_tracer->injectionRegions.push_back( region );
    }
  }

  //__________________________________
  //  CCVariable values to be copied
  ProblemSpecP vars_ps = TP_ps->findBlock("cloneVariables");
  if ( vars_ps ){

    static int cntr=1;

    proc0cout_eq( cntr, 1 ) << "\n__________________________________TracerParticles\n";

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
      std::string L1 = "p.clone-"+labelName+"+";
      std::string L2 = "p.clone-"+labelName;
      VarLabel* QLabel_preReloc   = VarLabel::create( L1, d_Part_double );
      VarLabel* QLabel            = VarLabel::create( L2, d_Part_double );
      proc0cout_eq( cntr, 1 ) << "   Created labels (" << L1 << ") (" << L2 << ")\n";

      //__________________________________
      //  populate the vector of particle variables
      auto me               = make_unique< cloneVar >();
      me->CCVarName         = labelName;
      me->CCVarLabel        = label;
      me->pQLabel_preReloc  = QLabel_preReloc;
      me->pQLabel           = QLabel;
      d_cloneVars.push_back( move(me) );

      // used for the relocation task
      d_oldLabels.push_back( QLabel_preReloc );
      d_newLabels.push_back( QLabel );
    }
    cntr ++;
  }
  else {
    proc0cout << "\nTracerParticles:WARNING Couldn't find <Variables> tag.\n";
  }

  //__________________________________
  //  create scalar particles whose value can decay exponentially
  //  This is identical to what passiveScalar
  for( ProblemSpecP scalar_ps = TP_ps->findBlock( "scalar" ); scalar_ps != nullptr; scalar_ps = scalar_ps->findNextBlock( "scalar" ) ) {
    if ( scalar_ps ){

      static int cntr=1;

      std::string name {""};
      scalar_ps->getAttribute( "name", name );
      if ( name ==""){
        throw ProblemSetupException("TracerParticles: the scalar tag must have a valid name  <scalar name=X>", __FILE__, __LINE__);
      }

      // create the labels associated with this scalar
      auto S  = make_unique< scalar >();
      std::string ln  = name;
      std::string lnP = ln  +"+";

      S->labelName        = ln;
      S->label_preReloc   = VarLabel::create( lnP, d_Part_double );
      S->label            = VarLabel::create( ln,  d_Part_double );

      proc0cout_eq( cntr, 1 ) << "   Created labels (" << lnP << ") (" << ln << ")\n";

      // labels needed for the relocation task
      d_oldLabels.push_back( S->label_preReloc );
      d_newLabels.push_back( S->label );

      scalar_ps->require( "initialValue",  S->initialValue );

      // container for preReloc label and initial value
      std::pair< VarLabel*, double > map ( S->label_preReloc, S->initialValue );
      S->label_value.insert( map );


      //__________________________________
      // exponential Decay
      ProblemSpecP exp_ps = scalar_ps->findBlock("exponentialDecay");

      if( exp_ps ) {

                         // create labels associated with the exponental decay model
        std::string L1 = name +"_expDecayCoef";
        std::string L2 = name + "_totalDecay";
        std::string L3 = L2 + "+";

        S->expDecayCoefLabel = VarLabel::create( L1, CCVariable<double>::getTypeDescription());
        S->totalDecayLabel          = VarLabel::create( L2, d_Part_double );
        S->totalDecayLabel_preReloc = VarLabel::create( L3, d_Part_double );

        // labels needed for the relocation task
        d_oldLabels.push_back( S->totalDecayLabel_preReloc );
        d_newLabels.push_back( S->totalDecayLabel );

        // container for preReloc label and initial value
        std::pair< VarLabel*, double > map ( S->totalDecayLabel_preReloc, 0.0 );
        S->label_value.insert( map );


        proc0cout_eq( cntr, 1 ) << "   Created labels (" << L1 << ") (" << L2 << ") (" << L3 <<")\n";

        S->withExpDecayModel = true;
        exp_ps->require(        "c1", S->c1);
        exp_ps->getWithDefault( "c3", S->c3, 0.0 );

        // The c2 coefficient type can be either a constant or read from a table
        ProblemSpecP c2_ps = exp_ps->findBlock("c2");
        std::string type = "";
        c2_ps->getAttribute( "type", type );

        if( type == "variable"){    // read c2 from table
          S->decayCoefType = scalar::variable;
          c2_ps->require( "filename", S->c2_filename );
        }
        else{           // c2 is a constant
          S->decayCoefType = scalar::constant;
          c2_ps->require( "value", S->c2);
        }

        if( S->decayCoefType == scalar::none ){
          throw ProblemSetupException("TracerParticles: the tag c2 must have either a constant value or a filenaS", __FILE__, __LINE__);
        }
      }

      d_scalars.push_back( move(S) );
      proc0cout_eq( cntr, 1 ) << "__________________________________\n";
      cntr ++;
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
  ProblemSpecP tp_ps = model_ps->appendChild( "TracerParticles" );

  tp_ps->appendElement( "modelPreviouslyInitialized", d_previouslyInitialized );
  tp_ps->appendElement("reinitializeDomain",          "false" );                    // The user must manually overide in checkpoint

  tp_ps->appendElement( "material",  d_matl->getName() );
  tp_ps->appendElement( "timeStart", d_tracer->timeStart );
  tp_ps->appendElement( "timeStop",  d_tracer->timeStop );

  //__________________________________
  //  Clone of CC variables
  if( d_cloneVars.size() > 0 ){
    ProblemSpecP nv_ps = tp_ps->appendChild( "cloneVariables" );

    for ( size_t i=0 ; i<d_cloneVars.size(); i++ ) {
      std::shared_ptr<cloneVar> Q = d_cloneVars[i];

      ProblemSpecP CC_ps = nv_ps->appendChild( "CCVarLabel" );
      CC_ps->setAttribute( "label", Q->CCVarName );
    }
  }

  //__________________________________
  //  scalars
  if( d_scalars.size() > 0 ){

    for ( size_t i=0 ; i<d_scalars.size(); i++ ) {
      std::shared_ptr<scalar> S = d_scalars[i];

      ProblemSpecP scalar_ps = tp_ps->appendChild( "scalar" );
      scalar_ps->setAttribute( "name",S->labelName );

      scalar_ps->appendElement( "initialValue", S->initialValue );


      if (S->withExpDecayModel ){
        ProblemSpecP exp_ps = scalar_ps->appendChild( "exponentialDecay" );
        exp_ps->appendElement( "c1", S->c1 );

                        // The c2 coefficient type can be either a constant or read from a table
        ProblemSpecP c2_ps = exp_ps->appendChild("c2");

                        // read c2 from table
        if( S->decayCoefType == scalar::variable){
          c2_ps->setAttribute( "type", "variable" );
          c2_ps->appendElement( "filename", S->c2_filename );
        }
        else{           // c2 is a constant
          c2_ps->setAttribute( "type", "constant" );
          c2_ps->appendElement( "value", S->c2 );
        }
        exp_ps->appendElement( "c3", S->c3 );
      }
    }
  }

  //__________________________________
  //
  ProblemSpecP tracer_ps = tp_ps->appendChild( "tracer" );
  tracer_ps->setAttribute( "name", d_tracer->name );

  //__________________________________
  //  initialization regions
  ProblemSpecP init_ps = tracer_ps->appendChild( "initialization" );

  vector<Region*>::const_iterator iter;
  for ( iter = d_tracer->initializeRegions.begin(); iter != d_tracer->initializeRegions.end(); iter++) {
    Region* region = *iter;

    ProblemSpecP geom_ps = init_ps->appendChild( "geom_object" );

    region->piece->outputProblemSpec(geom_ps);
    geom_ps->appendElement( "isInteriorRegion",   region->isInteriorRegion );
    geom_ps->appendElement( "particlesPerCell",   region->particlesPerCell );
  }

  //__________________________________
  //  regions for particle injection
  if( d_tracer->injectionRegions.size() > 0 ){
    ProblemSpecP int_ps = tracer_ps->appendChild( "interiorSources" );

    vector<Region*>::const_iterator itr;
    for ( iter = d_tracer->injectionRegions.begin(); iter != d_tracer->injectionRegions.end(); iter++) {
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
  const string schedName = "TracerParticles::scheduleInitialize_("+ d_tracer->fullName+")";
  printSchedule(level,dout_models_tp,schedName);

  const string taskName = "TracerParticles::initializeTask_("+ d_tracer->fullName+")";
  Task* t = scinew Task(taskName, this, &TracerParticles::initializeTask);

  t->requires( Task::OldDW,    simTimeLabel );
  t->computes( nPPCLabel,      d_matl_mss );
  t->computes( pXLabel,        d_matl_mss );
  t->computes( pDispLabel,     d_matl_mss );
  t->computes( pVelocityLabel, d_matl_mss );
  t->computes( pIDLabel,       d_matl_mss );

  for ( size_t i=0 ; i<d_cloneVars.size(); i++ ) {
    std::shared_ptr<cloneVar> Q = d_cloneVars[i];
    t->computes ( Q->pQLabel, d_matl_mss );
  }

  //__________________________________
  //  scalars
  for ( size_t i=0 ; i<d_scalars.size(); i++ ) {
    std::shared_ptr<scalar> S = d_scalars[i];
    t->computes( S->label, d_matl_mss );

    if( S->withExpDecayModel ){
      t->computes( S->expDecayCoefLabel, d_matl_mss );
      t->computes( S->totalDecayLabel,   d_matl_mss );
    }
  }

  sched->addTask(t, level->eachPatch(), d_matl_set);
}

//______________________________________________________________________
//    Task:  On a restart if the values are going to be modified there must first
//           be computes( label ) before it can be modified.  This is a hack
void TracerParticles::sched_restartInitializeHACK( SchedulerP   & sched,
                                                   const LevelP & level)
{
  const string schedName = "TracerParticles::sched_restartInitializeHACK("+ d_tracer->fullName+")";
  printSchedule(level,dout_models_tp,schedName);

  const string taskName = "TracerParticles::restartInitializeHACK_("+ d_tracer->fullName+")";
  Task* t = scinew Task(taskName, this, &TracerParticles::restartInitializeHACK);

  //__________________________________
  //     core variables
  t->computes( nPPCLabel,      d_matl_mss );
  t->computes( pXLabel,        d_matl_mss );
  t->computes( pDispLabel,     d_matl_mss );
  t->computes( pVelocityLabel, d_matl_mss );
  t->computes( pIDLabel,       d_matl_mss );
  
  //__________________________________
  //      clone Q_CC vars
  for ( size_t i=0 ; i<d_cloneVars.size(); i++ ) {
    std::shared_ptr<cloneVar> Q = d_cloneVars[i];
    t->computes ( Q->pQLabel, d_matl_mss );
  }

  //__________________________________
  //      scalars
  for ( size_t i=0 ; i<d_scalars.size(); i++ ) {
    std::shared_ptr<scalar> S = d_scalars[i];
    t->computes( S->label, d_matl_mss );

    if( S->withExpDecayModel ){
      t->computes( S->expDecayCoefLabel, d_matl_mss );
      t->computes( S->totalDecayLabel,   d_matl_mss );
    }
  }

  sched->addTask(t, level->eachPatch(), d_matl_set);
}


//______________________________________________________________________
//    Task:  schedule restartInitialize
//           Only execute this if the domain was not previously initialized
//           or the user requests that the domain should be reinitialized
void TracerParticles::scheduleRestartInitialize(SchedulerP   & sched,
                                                const LevelP & level)
{

  //__________________________________
  //  if the user turned on the model in a checkpoint
  if( !d_previouslyInitialized ){
    scheduleInitialize( sched, level );
    return;
  }

  if( !d_reinitializeDomain ){
    return;
  }

  //__________________________________
  //
  sched_restartInitializeHACK( sched, level );


  //__________________________________
  //
  const string schedName = "TracerParticles::scheduleRestartInitialize_("+ d_tracer->fullName+")";
  printSchedule(level,dout_models_tp,schedName);

  const string taskName = "TracerParticles::restartInitializeTask_("+ d_tracer->fullName+")";
  Task* t = scinew Task(taskName, this, &TracerParticles::restartInitializeTask);

  t->requires( Task::OldDW,    simTimeLabel );
  t->modifies( nPPCLabel,      d_matl_mss );
  t->modifies( pXLabel,        d_matl_mss );
  t->modifies( pDispLabel,     d_matl_mss );
  t->modifies( pVelocityLabel, d_matl_mss );
  t->modifies( pIDLabel,       d_matl_mss );
  
  //__________________________________
  //    clone variables
  for ( size_t i=0 ; i<d_cloneVars.size(); i++ ) {
    std::shared_ptr<cloneVar> Q = d_cloneVars[i];
    t->modifies ( Q->pQLabel, d_matl_mss );
  }

  //__________________________________
  //    tracer scalars
  for ( size_t i=0 ; i<d_scalars.size(); i++ ) {
    std::shared_ptr<scalar> S = d_scalars[i];
    t->modifies( S->label, d_matl_mss );

    if( S->withExpDecayModel ){
      t->modifies( S->expDecayCoefLabel, d_matl_mss );
      t->modifies( S->totalDecayLabel,   d_matl_mss );
    }
  }

  sched->addTask(t, level->eachPatch(), d_matl_set);
}

//______________________________________________________________________
//   Function:  TracerParticles::distributeParticles
//   Purpose:    Count the number of particles on a patch inside
//                user defined regions.
//              - randomly lay down particles in each cell in each region
//______________________________________________________________________
unsigned int TracerParticles::distributeParticles( const Patch   * patch,
                                                   const double    simTime,
                                                   const double    delT,
                                                   const std::vector<Region*> regions,
                                                   regionPoints  & pPositions)
{
  unsigned int count = 0;

  bool isItTime = (( simTime >= d_tracer->timeStart ) && ( simTime <= d_tracer->timeStop) );

  if( !isItTime ){
    return count;
  }

  for( auto r_iter = regions.begin(); r_iter != regions.end(); r_iter++){
    Region* region = *r_iter;

    // is this region contained on this patch
    GeometryPieceP piece = region->piece;
    Box b1 = piece->getBoundingBox();
    Box b2 = patch->getExtraBox();
    Box b  = b1.intersect(b2);

   #if 0
      cout << " patch: " << patch->getID() << " piece: " << piece->getName() << " degenerate: " << b.degenerate() << " piece bounding box; " << b1 << " patch Bonding box: " << b2
          << " intersect: " << b << endl;
    #endif

    if( b.degenerate() ){
       continue;
    }

    Vector dx = patch->dCell();
    Vector dx_2 = dx/Vector(2);
    MTRand mTwister;

    int nParticlesPerCell = region->particlesPerCell;

    // for interior regions compute PPC
    if( region->isInteriorRegion ) {

      region->elapsedTime[patch] += delT;
      double elapsedTime = region->elapsedTime[patch];

      nParticlesPerCell = Uintah::RoundDown(region->particlesPerCellPerSecond * elapsedTime);

      // reset if particles are added
      if( nParticlesPerCell > 0 ){
        region->elapsedTime[patch] = 0;
      }
    }

    //__________________________________
    //  Loop over all cells and laydown particles
    for(CellIterator c_iter = patch->getCellIterator(); !c_iter.done(); c_iter++){
      IntVector c = *c_iter;

      Point CC_pos =patch->cellPosition(c);

      if ( piece->inside(CC_pos,true) ){

        Point lower = CC_pos - dx_2;
        mTwister.seed((c.x() + c.y() + c.z()));

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
void TracerParticles::initializeCoreVariables( const Patch   * patch,
                                               unsigned int    pIndx,
                                               regionPoints  & pPositions,
                                               std::vector<Region*> regions,
                                               ParticleVariable<Point> & pX,
                                               ParticleVariable<Vector>& pDisp,
                                               ParticleVariable<Vector>& pVelocity,
                                               ParticleVariable<long64>& pID,
                                               CCVariable<int>         & nPPC )
{

  if( pPositions.size() == 0 ){
    return;
  }

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

      IntVector cellIndx;
      if ( !patch->findCell( pos,cellIndx ) ) {
        continue;
      }

      pX[pIndx]        = pos;
      pDisp[pIndx]     = Vector(0);
      pVelocity[pIndx] = Vector(0.0);

      ASSERT(cellIndx.x() <= 0xffff &&
             cellIndx.y() <= 0xffff &&
             cellIndx.z() <= 0xffff);

      long64 cellID = ((long64)cellIndx.x() << 16) |
                      ((long64)cellIndx.y() << 32) |
                      ((long64)cellIndx.z() << 48);

      int& my_nPPC = nPPC[cellIndx];

      pID[pIndx] = (cellID | (long64) my_nPPC);
      ASSERT(my_nPPC < 0x7fff);

      my_nPPC++;

      DOUTR( dout_models_tp, " initializeRegions: patch: " << patch->getID() << " Cell: " << cellIndx << " nPPC: "
                             << my_nPPC << " pIdx: " << pIndx << " pID: " << pID[pIndx] << " pos: " << pos );

      pIndx++;

    }  // particles
  }  // regions
}
//______________________________________________________________________
//  If you modify the particle variables you must do extra work
//______________________________________________________________________
void TracerParticles::initializeScalarVars( ParticleSubset * pset,
                                               const Patch    * patch,
                                               const int        indx,
                                               DataWarehouse  * new_dw,
                                               const modifiesComputes which)
{

  if( which == TracerParticles::computesVar){
  //      
    for ( size_t i=0 ; i<d_scalars.size(); i++ ) {
      std::shared_ptr<scalar> S = d_scalars[i];

      ParticleVariable<double> pS;
      new_dw->allocateAndPut( pS, S->label, pset );

      for(ParticleSubset::iterator iter = pset->begin();iter != pset->end(); iter++){
        particleIndex idx = *iter;
        pS[idx] = S->initialValue;
      }

      //__________________________________
      //  Initialize coefficient used in exponential decay model
      if( S->withExpDecayModel ){
      
        int id = patch->getID();

        proc0cout_eq(id, 0)
                << "________________________TracerParticles\n"
                << "  Coefficient c1: " << S->c1 << "\n";        
      
        CCVariable<double> c2;
        new_dw->allocateAndPut( c2, S->expDecayCoefLabel, indx, patch, d_gn, 0);

        c2.initialize(0.0);

                          // constant value
        if ( S->decayCoefType == scalar::constant ){
          c2.initialize( S->c2 );
          proc0cout_eq(id, 0)
              << "  Coefficient c2: " << S->c2 << "\n";
        }
        else{             // read in from a file
          const Level* level = patch->getLevel();
          PassiveScalar::readTable( patch, level, S->c2_filename, c2 );
        }
        
        proc0cout_eq(id, 0)
              << "  Coefficient c3: " << S->c3 << "\n"
              << "__________________________________\n";

                          // initialize quantities
        ParticleVariable<double> pTotalDecay;
        new_dw->allocateAndPut( pTotalDecay, S->totalDecayLabel, pset );

        for(ParticleSubset::iterator iter = pset->begin();iter != pset->end(); iter++){
          particleIndex idx = *iter;
          pTotalDecay[idx] = 0.0;
        }
      }  // exp decay
    }  // scalars loop
  } 
  
  //__________________________________
  //        modify variables
  if( which == TracerParticles::modifiesVar){
    
    const bool replace {true};
    
    for ( size_t i=0 ; i<d_scalars.size(); i++ ) {
      std::shared_ptr<scalar> S = d_scalars[i];

      ParticleVariable<double> pS;
      new_dw->allocateTemporary( pS, pset );

      for(ParticleSubset::iterator iter = pset->begin();iter != pset->end(); iter++){
        particleIndex idx = *iter;
        pS[idx] = S->initialValue;
      }

      new_dw->put( pS,  S->label, replace);
      //__________________________________
      //  Initialize coefficient used in exponential decay model
      if( S->withExpDecayModel ){
        int id = patch->getID();

        proc0cout_eq(id, 0)
                << "________________________TracerParticles\n"
                << "  Coefficient c1: " << S->c1 << "\n";      
      
        CCVariable<double> c2;

        new_dw->getModifiable( c2, S->expDecayCoefLabel, indx, patch, d_gn, 0);
        c2.initialize(0.0);

                          // constant value
        if ( S->decayCoefType == scalar::constant ){
          c2.initialize( S->c2 );
          proc0cout_eq( id, 0)
              << "  Coefficient c2: " << S->c2 << "\n";
        }
        else{             // read in from a file
          const Level* level = patch->getLevel();
          PassiveScalar::readTable( patch, level, S->c2_filename, c2 );
        }
        proc0cout_eq(id, 0)
              << "  Coefficient c3: " << S->c3 << "\n"
              << "__________________________________\n";

                          // initialize quantities
        ParticleVariable<double> pTotalDecay;
        new_dw->allocateTemporary( pTotalDecay, pset );

        for(ParticleSubset::iterator iter = pset->begin();iter != pset->end(); iter++){
          particleIndex idx = *iter;
          pTotalDecay[idx] = 0.0;
        }
        new_dw->put( pTotalDecay, S->totalDecayLabel,    replace );
      }  // exp decay
    }  // scalars loop
  }
}


//______________________________________________________________________
//
//______________________________________________________________________
void TracerParticles::initializeCloneVars( ParticleSubset * pset,
                                           const Patch    * patch,
                                           const int        indx,
                                           DataWarehouse  * new_dw)
{
  //__________________________________
  //      Clone variables
  for ( size_t i=0 ; i<d_cloneVars.size(); i++ ) {
    std::shared_ptr<cloneVar> Q = d_cloneVars[i];

    ParticleVariable<double> pQ;
    new_dw->allocateAndPut( pQ, Q->pQLabel, pset );

    for(ParticleSubset::iterator iter = pset->begin();iter != pset->end(); iter++){
      particleIndex idx = *iter;
      pQ[idx] = 0.0;
    }
  }  // cloneVars loop
}


//______________________________________________________________________
//  Task:  TracerParticles::initialize
//  Purpose:  Create the particles on all patches
//______________________________________________________________________
void TracerParticles::initializeTask(const ProcessorGroup *,
                                     const PatchSubset    * patches,
                                     const MaterialSubset * matl_mss,
                                     DataWarehouse        * old_dw,
                                     DataWarehouse        * new_dw)
{

  simTime_vartype simTime;
  new_dw->get( simTime, simTimeLabel );
  d_previouslyInitialized = true;

  //__________________________________
  // Patches loop
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    const string msg = "Doing TracerParticles::initializeTask_("+ d_tracer->fullName+")";
    printTask(patches, patch, dout_models_tp, msg);

    //__________________________________
    // For each Region on this patch
    //  - initialize timer
    //  - Laydown particles in each region.
    //  - return the positions of the particles
    //  - Count the number of particles
    //

    for( auto r_iter = d_tracer->initializeRegions.begin();
             r_iter != d_tracer->initializeRegions.end(); r_iter++){
      Region* region = *r_iter;
      region->elapsedTime[patch] = 0.0;
    }

    regionPoints pPositions;
    unsigned int nParticles;
    const double ignore = -9;
    nParticles = TracerParticles::distributeParticles( patch,
                                                       simTime,
                                                       ignore,
                                                       d_tracer->initializeRegions,
                                                       pPositions );

    //__________________________________
    //  allocate the particle variables
    ParticleVariable<Point>  pX;
    ParticleVariable<Vector> pDisp;
    ParticleVariable<Vector> pVel;
    ParticleVariable<long64> pID;
    CCVariable<int>          nPPC;    // number of particles per cell

    int indx = d_matl->getDWIndex();

    ParticleSubset* pset = new_dw->createParticleSubset( nParticles, indx, patch );

    new_dw->allocateAndPut( pX,    pXLabel,        pset );
    new_dw->allocateAndPut( pDisp, pDispLabel,     pset );
    new_dw->allocateAndPut( pVel,  pVelocityLabel, pset );
    new_dw->allocateAndPut( pID,   pIDLabel,       pset );
    new_dw->allocateAndPut( nPPC,  nPPCLabel, indx, patch );
    nPPC.initialize(0);

    int pIndx = 0;

    initializeCoreVariables(  patch, pIndx, pPositions,
                              d_tracer->initializeRegions,
                              pX,  pDisp, pVel, pID, nPPC );

    initializeScalarVars(  pset, patch, indx, new_dw, TracerParticles::computesVar);
    
    initializeCloneVars(  pset, patch, indx, new_dw);

  }  // patches
}

//______________________________________________________________________
//  Task:  TracerParticles::restartInitialize
//  Purpose:  delete all the existing particles and reinitialize them
//            Only execute this task when d_reinitializeDomain is set
//______________________________________________________________________
void TracerParticles::restartInitializeTask(const ProcessorGroup *,
                                            const PatchSubset    * patches,
                                            const MaterialSubset * matl_mss,
                                            DataWarehouse        * old_dw,
                                            DataWarehouse        * new_dw)
{

  simTime_vartype simTime;
  new_dw->get( simTime, simTimeLabel );
  d_previouslyInitialized = true;

  //__________________________________
  // Patches loop
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    const string msg = "Doing TracerParticles::restartInitialize_("+ d_tracer->fullName+")";
    printTask(patches, patch, dout_models_tp, msg);

    //__________________________________
    // For each Region on this patch
    //  - initialize timer
    //  - Laydown particles in each region.
    //  - return the positions of the particles
    //  - Count the number of particles
    //

    for( auto r_iter = d_tracer->initializeRegions.begin();
             r_iter != d_tracer->initializeRegions.end(); r_iter++){
      Region* region = *r_iter;
      region->elapsedTime[patch] = 0.0;
    }

    regionPoints pPositions;
    unsigned int nParticles;
    const double ignore = -9;
    nParticles = TracerParticles::distributeParticles( patch,
                                                       simTime,
                                                       ignore,
                                                       d_tracer->initializeRegions,
                                                       pPositions );

    //__________________________________
    //  allocate the  core particle variables
    ParticleVariable<Point>  pX;
    ParticleVariable<Vector> pDisp;
    ParticleVariable<Vector> pVel;
    ParticleVariable<long64> pID;
    CCVariable<int>          nPPC;    // number of particles per cell

    int indx = d_matl->getDWIndex();

                                      // delete the old pset and create a new one
    ParticleSubset* old_pset = new_dw->getParticleSubset( indx, patch );
    new_dw->deleteParticleSubset( old_pset );

    ParticleSubset* pset = new_dw->createParticleSubset( nParticles, indx, patch );

    DOUTR(true, "TracerParticles: restartInitialize patch: " << patch->getID() << " deleting old Particle subset and creating a new one" );
    DOUTR(true, "TracerParticles: restartInitialize patch: " << patch->getID() << " numParticles: " << nParticles << " new pset.size() " << pset->numParticles() );

    new_dw->allocateTemporary( pX,     pset );
    new_dw->allocateTemporary( pDisp,  pset );
    new_dw->allocateTemporary( pVel,   pset );
    new_dw->allocateTemporary( pID,    pset );

    new_dw->getModifiable( nPPC,  nPPCLabel, indx, patch );
    nPPC.initialize(0);

    const int pIndx = 0;

    initializeCoreVariables(  patch, pIndx, pPositions,
                              d_tracer->initializeRegions,
                              pX,  pDisp, pVel, pID, nPPC );

    const bool replace = true;
    new_dw->put( pX,    pXLabel,        replace );
    new_dw->put( pDisp, pDispLabel,     replace );
    new_dw->put( pVel,  pVelocityLabel, replace );
    new_dw->put( pID,   pIDLabel,       replace );

    //__________________________________
    //    Initialize all the scalar variables
    initializeScalarVars(  pset, patch, indx, new_dw, TracerParticles::modifiesVar  );

  }  // patches
}




//______________________________________________________________________
void TracerParticles::scheduleComputeModelSources(SchedulerP  & sched,
                                                  const LevelP& level)
{
  const string taskName = "TracerParticles::scheduleComputeModelSources_("+ d_tracer->fullName+")";
  printSchedule(level,dout_models_tp, taskName);

  sched_setParticleVars( sched, level );

  sched_moveParticles(   sched, level );

  sched_addParticles(    sched, level );

}

//______________________________________________________________________
//  Task: moveParticles
//  Purpose:  Update the particles:
//              - position
//              - displacement
//              - velocity
//              - delete any particles outside the domain
//______________________________________________________________________
void TracerParticles::sched_moveParticles(SchedulerP  & sched,
                                          const LevelP& level)
{

  const string schedName = "TracerParticles::sched_moveParticles_("+ d_tracer->fullName+")";
  printSchedule( level ,dout_models_tp, schedName);

  const string taskName = "TracerParticles::moveParticles_("+ d_tracer->fullName+")";
  Task* t = scinew Task( taskName, this, &TracerParticles::moveParticles);

  t->requires( Task::OldDW, Ilb->delTLabel, level.get_rep() );
  t->requires( Task::OldDW, simTimeLabel );

  t->requires( Task::OldDW, pXLabel,        d_matl_mss, d_gn );
  t->requires( Task::OldDW, pDispLabel,     d_matl_mss, d_gn );
  t->requires( Task::OldDW, pIDLabel,       d_matl_mss, d_gn );

  t->requires( Task::OldDW, Ilb->vel_CCLabel, d_matl_mss, d_gn );   // hardwired to use ICE's velocity

  t->computes( pXLabel_preReloc,        d_matl_mss );
  t->computes( pDispLabel_preReloc,     d_matl_mss );
  t->computes( pVelocityLabel_preReloc, d_matl_mss );
  t->computes( pIDLabel_preReloc,       d_matl_mss );

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
    ParticleVariable<Vector> pVel;
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

    new_dw->allocateAndPut( pX,    pXLabel_preReloc,        pset );
    new_dw->allocateAndPut( pDisp, pDispLabel_preReloc,     pset );
    new_dw->allocateAndPut( pVel,  pVelocityLabel_preReloc, pset );
    new_dw->allocateAndPut( pID,   pIDLabel_preReloc,       pset );
    pID.copyData( pID_old );

    BBox compDomain;
    GridP grid = level->getGrid();
    grid->getInteriorSpatialRange( compDomain );

    //__________________________________
    //  Update particle postion and displacement
    for(ParticleSubset::iterator iter = pset->begin();iter != pset->end(); iter++){
      particleIndex idx = *iter;

      IntVector cellIndx;
      if ( !patch->findCell( pX_old[idx],cellIndx ) ) {
        continue;
      }

      pX[idx]    = pX_old[idx]     + vel_CC[cellIndx]*delT;
      pDisp[idx] = pDisp_old[idx]  + ( pX[idx] - pX_old[idx] );
      pVel[idx]  = vel_CC[cellIndx];

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
  if ( d_tracer->injectionRegions.size() == 0 ){
    return;
  }

  const string schedName = "TracerParticles::sched_addParticles_("+ d_tracer->fullName+")";
  printSchedule( level ,dout_models_tp, schedName);

  const string taskName = "TracerParticles::addParticles_("+ d_tracer->fullName+")";
  Task* t = scinew Task( taskName, this, &TracerParticles::addParticles);

  t->requires( Task::OldDW, Ilb->delTLabel, level.get_rep() );
  t->requires( Task::OldDW, nPPCLabel,     d_matl_mss, d_gn );

  t->modifies( pXLabel_preReloc,        d_matl_mss );
  t->modifies( pDispLabel_preReloc,     d_matl_mss );
  t->modifies( pVelocityLabel_preReloc, d_matl_mss );
  t->modifies( pIDLabel_preReloc,       d_matl_mss );
  t->modifies( nPPCLabel,               d_matl_mss );

                // Clone of CC variables
  for ( size_t i=0 ; i<d_cloneVars.size(); i++ ) {
    std::shared_ptr<cloneVar> Q = d_cloneVars[i];
    t->modifies( Q->pQLabel_preReloc,   d_matl_mss );
  }

                // scalar variables
  for ( size_t i=0 ; i<d_scalars.size(); i++ ) {
    std::shared_ptr<scalar> S = d_scalars[i];
    t->modifies( S->label_preReloc,           d_matl_mss );
    t->modifies( S->totalDecayLabel_preReloc, d_matl_mss );
  }

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
  simTime_vartype simTime;
  old_dw->get( delT,    Ilb->delTLabel, level);
  old_dw->get( simTime, simTimeLabel);

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
                                                       simTime,
                                                       delT,
                                                       d_tracer->injectionRegions,
                                                       pPositions );
    //__________________________________
    //
    int matlIndx = d_matl->getDWIndex();

    CCVariable<int> nPPC;    // number of particles per cell
    new_dw->getModifiable( nPPC,     nPPCLabel, matlIndx, patch, d_gn, 0 );

    ParticleVariable<Point>  pX;
    ParticleVariable<Vector> pDisp;
    ParticleVariable<Vector> pVel;
    ParticleVariable<long64> pID;

    ParticleSubset* pset  = old_dw->getParticleSubset( matlIndx, patch );
    unsigned int oldNumPar   = pset->addParticles( nParticles );

    new_dw->getModifiable( pX,    pXLabel_preReloc,       pset );
    new_dw->getModifiable( pDisp, pDispLabel_preReloc,    pset );
    new_dw->getModifiable( pVel,  pVelocityLabel_preReloc,pset );
    new_dw->getModifiable( pID,   pIDLabel_preReloc,      pset );

    //__________________________________
    //  Allocate temp variables and populate them
    ParticleVariable<Point>  pX_tmp;
    ParticleVariable<Vector> pDisp_tmp;
    ParticleVariable<Vector> pVel_tmp;
    ParticleVariable<long64> pID_tmp;

    new_dw->allocateTemporary( pX_tmp,     pset );
    new_dw->allocateTemporary( pDisp_tmp,  pset );
    new_dw->allocateTemporary( pVel_tmp,   pset );
    new_dw->allocateTemporary( pID_tmp,    pset );

    for( unsigned int idx=0; idx<oldNumPar; ++idx ){
      pX_tmp[idx]    = pX[idx];
      pDisp_tmp[idx] = pDisp[idx];
      pVel_tmp[idx]  = pVel[idx];
      pID_tmp[idx]   = pID[idx];
    }

    // update their values
    initializeCoreVariables(  patch, oldNumPar, pPositions,
                              d_tracer->injectionRegions,
                              pX_tmp, pDisp_tmp, pVel_tmp,
                              pID_tmp, nPPC );

    const bool replace = true;
    new_dw->put( pX_tmp,    pXLabel_preReloc,        replace );
    new_dw->put( pDisp_tmp, pDispLabel_preReloc,     replace );
    new_dw->put( pVel_tmp,  pVelocityLabel_preReloc, replace );
    new_dw->put( pID_tmp,   pIDLabel_preReloc,       replace );

    //__________________________________
    // Initialize Clone
    for ( size_t i=0 ; i<d_cloneVars.size(); i++ ) {
     std::shared_ptr<cloneVar> Q = d_cloneVars[i];

     constCCVariable<double>  Q_CC;
     old_dw->get( Q_CC, Q->CCVarLabel, matlIndx, patch,  d_gn, 0 );

     ParticleVariable<double> pQ;
     ParticleVariable<double> pQ_tmp;

     new_dw->getModifiable( pQ,   Q->pQLabel_preReloc, pset );
     new_dw->allocateTemporary( pQ_tmp,     pset );

     for( unsigned int idx=0; idx<oldNumPar; ++idx ){
       pQ_tmp[idx] = pQ[idx];
     }

     initializeRegions(  patch, oldNumPar, pPositions,
                          d_tracer->injectionRegions,
                          Q_CC, -9, pQ_tmp );

     new_dw->put( pQ_tmp,  Q->pQLabel_preReloc, true);
    }

    //__________________________________
    // Initialize the scalar particle variables:
    //  label_preReloc and totalDecayLabel_preReloc
    for ( size_t i=0 ; i<d_scalars.size(); i++ ) {
      std::shared_ptr<scalar> S = d_scalars[i];

      std::multimap<VarLabel*, double>::iterator itr;
      for (itr = S->label_value.begin(); itr != S->label_value.end(); ++itr) {

        VarLabel* label_preReloc = itr->first;
        double initialValue      = itr->second;

        constCCVariable<double>  empty_CC;  // not used
        ParticleVariable<double> pS;
        ParticleVariable<double> pS_tmp;

        new_dw->getModifiable(  pS, label_preReloc, pset );
        new_dw->allocateTemporary( pS_tmp,  pset );

        for( unsigned int idx=0; idx<oldNumPar; ++idx ){
          pS_tmp[idx] = pS[idx];
        }

        initializeRegions(  patch, oldNumPar, pPositions,
                             d_tracer->injectionRegions,
                             empty_CC, initialValue, pS_tmp );

        new_dw->put( pS_tmp,  label_preReloc, true);

      }
    }  // scalar loop
  }  //  patches
}


//______________________________________________________________________
//  Task: setParticleVars
//  Purpose:  set the particle quantities to the corresponding CCVariables
//            and any particle scalar variables
//______________________________________________________________________
void TracerParticles::sched_setParticleVars( SchedulerP  & sched,
                                             const LevelP& level)
{
  const string schedName = "TracerParticles::sched_setParticleVars_("+ d_tracer->fullName+")";
  printSchedule( level ,dout_models_tp, schedName);

  const string taskName = "TracerParticles::setParticleVars_("+ d_tracer->fullName+")";
  Task* t = scinew Task( taskName, this, &TracerParticles::setParticleVars);

  t->requires( Task::OldDW, pXLabel,   d_matl_mss, d_gn, 0 );
  t->requires( Task::OldDW, nPPCLabel, d_matl_mss, d_gn, 0 );
  t->computes( nPPCLabel,              d_matl_mss );

  //__________________________________
  //    clone variables
  for ( size_t i=0 ; i<d_cloneVars.size(); i++ ) {
    std::shared_ptr<cloneVar> Q = d_cloneVars[i];

    t->requires( Task::OldDW, Q->CCVarLabel, d_matl_mss, d_gn, 0 );
    t->computes ( Q->pQLabel_preReloc, d_matl_mss );
  }

  //__________________________________
  // scalars: update if decay model is enabled
  for ( size_t i=0 ; i<d_scalars.size(); i++ ) {
    std::shared_ptr<scalar> S = d_scalars[i];

    if ( S->withExpDecayModel ){
      t->requires( Task::OldDW, Ilb->delTLabel, level.get_rep() );
      t->requires( Task::OldDW, S->label,             d_matl_mss, d_gn, 0 );
      t->requires( Task::OldDW, S->expDecayCoefLabel, d_matl_mss, d_gn, 0 );
      t->requires( Task::OldDW, S->totalDecayLabel,   d_matl_mss, d_gn, 0 );

      t->computes( S->totalDecayLabel_preReloc, d_matl_mss );
      t->computes( S->expDecayCoefLabel,         d_matl_mss );
      t->computes( S->label_preReloc,            d_matl_mss );
    }
  }

  sched->addTask(t, level->eachPatch(), d_matl_set);
}


//______________________________________________________________________
//
void TracerParticles::setParticleVars(const ProcessorGroup  *,
                                      const PatchSubset     * patches,
                                      const MaterialSubset  * matls,
                                      DataWarehouse         * old_dw,
                                      DataWarehouse         * new_dw)
{
  const Level* level = getLevel(patches);

  new_dw->transferFrom( old_dw, nPPCLabel, patches, d_matl_mss );

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    const string msg = "Doing TracerParticles::setParticleVars("+ d_tracer->fullName+")";
    printTask(patches, patch, dout_models_tp, msg);



    //__________________________________
    //
    int matlIndx = d_matl->getDWIndex();
    ParticleSubset* pset  = old_dw->getParticleSubset( matlIndx, patch );

    constParticleVariable<Point>  pX;
    old_dw->get( pX, pXLabel, pset );

    //__________________________________
    //    clone CC variables
    for ( size_t i=0 ; i<d_cloneVars.size(); i++ ) {
      std::shared_ptr<cloneVar> Q = d_cloneVars[i];

      constCCVariable<double>  Q_CC;
      ParticleVariable<double> pQ;

      old_dw->get( Q_CC, Q->CCVarLabel, matlIndx, patch,  d_gn, 0 );
      new_dw->allocateAndPut( pQ,   Q->pQLabel_preReloc, pset );

      for(ParticleSubset::iterator iter = pset->begin();iter != pset->end(); iter++){
        particleIndex idx = *iter;

        pQ[idx] = -9;

        IntVector cell;
        if ( !patch->findCell( pX[idx],cell ) ) {

          DOUTR(true, " setParticleVars, pX: " << pX[idx] << " cell: " << cell << " patch: " << patch->getID() );
          continue;
        }

        pQ[idx] = Q_CC[cell];
      }
    }  // cloneVars loop


    //__________________________________
    // Update the particle scalar if decay model is enabled
    for ( size_t i=0 ; i<d_scalars.size(); i++ ) {
      std::shared_ptr<scalar> S = d_scalars[i];

      if ( S->withExpDecayModel ){
        delt_vartype delT;
        old_dw->get( delT, Ilb->delTLabel, level);

        constParticleVariable<double>  s_old;
        constParticleVariable<double>  totalDecay_old;

        ParticleVariable<double> s;
        ParticleVariable<double> totalDecay;

        old_dw->get( s_old,          S->label,           matlIndx, patch );
        old_dw->get( totalDecay_old, S->totalDecayLabel, matlIndx, patch );

        new_dw->allocateAndPut( s,          S->label_preReloc,            pset );
        new_dw->allocateAndPut( totalDecay, S->totalDecayLabel_preReloc,  pset );

                            // exponential decay coefficient
        constCCVariable<double> c2;
        old_dw->get( c2, S->expDecayCoefLabel, matlIndx, patch, d_gn,0);
        new_dw->transferFrom( old_dw, S->expDecayCoefLabel, patches, matls );

        const double c1 = S->c1;
        const double c3 = S->c3;

        //__________________________________
        //
        for(ParticleSubset::iterator iter = pset->begin();iter != pset->end(); iter++){
          particleIndex idx = *iter;

          s[idx]          = -9;
          totalDecay[idx] = -9;

          IntVector c;
          if ( !patch->findCell( pX[idx],c ) ) {

            DOUTR(true, " setParticleVars, pX: " << pX[idx] << " cell: " << c << " patch: " << patch->getID() );
            continue;
          }
          double exposure = c2[c] * delT;
          totalDecay[idx] = totalDecay_old[idx] + exposure;
          s[idx]          = s_old[idx] * exp( -(c1 * c2[c] + c3) * delT );
        }
      }
    }
  }  // patches
}



//______________________________________________________________________
//  function:  initializeRegions
//  purpose:   Set the values in the newly created particles
//______________________________________________________________________
void TracerParticles::initializeRegions( const Patch             *  patch,
                                          unsigned int               pIndx,
                                          regionPoints             & pPositions,
                                          std::vector<Region*>       regions,
                                          constCCVariable<double>  & Q_CC,
                                          const double               initialValue,
                                          ParticleVariable<double> & pQ_tmp  )
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

      IntVector cell;
      if ( !patch->findCell( pos,cell ) ) {
        continue;
      }

                  // set to the CC quantity
      if( Q_CC.getDataSize() != 0 ){
        pQ_tmp[pIndx] = Q_CC[cell];
        std::cout << " Adding particle in cell: " << cell << " Value:  " << Q_CC[cell] << std::endl;
      }
      else{       // set to a constant
        std::cout << " Adding scalar particle in cell: " << cell << " Value: " << initialValue << std::endl;
        pQ_tmp[pIndx] = initialValue;
      }

      pIndx++;
    }  // particles
  }  // regions
}
