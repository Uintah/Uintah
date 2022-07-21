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


#include <CCA/Components/ICE/Core/ConservationTest.h>
#include <CCA/Components/ICE/Core/Diffusion.h>
#include <CCA/Components/ICE/CustomBCs/BoundaryCond.h>
#include <CCA/Components/Models/FluidsBased/PassiveScalar.h>
#include <CCA/Components/Models/FluidsBased/FluidsBasedModel.h>
#include <Core/Grid/Variables/PerPatchVars.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/Regridder.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/GeometryPiece/GeometryPieceFactory.h>
#include <Core/GeometryPiece/UnionGeometryPiece.h>
#include <Core/Grid/Box.h>
#include <Core/Grid/DbgOutput.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Material.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/PerPatch.h>

#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/IO/UintahZlibUtil.h>
#include <Core/IO/UintahIFStreamUtil.h>
#include <Core/Exceptions/ParameterNotFound.h>
#include <Core/Util/DOUT.hpp>

#include <iostream>
#include <cstdio>

using namespace Uintah;
using namespace std;

#define proc0cout_cmp(X,Y) if( isProc0_macro && X == Y) std::cout
Dout dout_models_ps("MODELS_DOING_COUT", "Models::PassiveScalar", "Models::PassiveScalar debug stream", false);
//______________________________________________________________________
PassiveScalar::PassiveScalar(const ProcessorGroup* myworld,
                             const MaterialManagerP& materialManager,
                             const ProblemSpecP& params)
  : FluidsBasedModel(myworld, materialManager), d_params(params)
{
  m_modelComputesThermoTransportProps = true;

  d_matl_set = 0;
  Ilb  = scinew ICELabel();
}

//__________________________________
PassiveScalar::~PassiveScalar()
{
  if(d_matl_set && d_matl_set->removeReference()) {
    delete d_matl_set;
  }

  VarLabel::destroy( d_scalar->Q_CCLabel );
  VarLabel::destroy( d_scalar->Q_src_CCLabel );
  VarLabel::destroy( d_scalar->diffusionCoef_CCLabel );
  VarLabel::destroy( d_scalar->mag_grad_Q_CCLabel );
  VarLabel::destroy( d_scalar->sum_Q_CCLabel );
  VarLabel::destroy( d_scalar->expDecayCoefLabel );

  delete Ilb;

  // regions used during initialization
  for(vector<Region*>::iterator iter = d_scalar->regions.begin();
                                iter != d_scalar->regions.end(); iter++){
    Region* region = *iter;
    delete region;
  }

  // Interior regions
  for(vector<interiorRegion*>::iterator iter = d_scalar->interiorRegions.begin();
                                        iter != d_scalar->interiorRegions.end(); iter++){
    interiorRegion* region = *iter;
    delete region;
  }

  delete d_scalar;

}

//______________________________________________________________________
//
PassiveScalar::Region::Region(GeometryPieceP piece, ProblemSpecP& ps)
  : piece(piece)
{
  ps->require("scalar", initialScalar);
  ps->getWithDefault( "sinusoidalInitialize",      sinusoidalInitialize,     false);
  ps->getWithDefault( "linearInitialize",          linearInitialize,         false);
  ps->getWithDefault( "cubicInitialize",           cubicInitialize,          false);
  ps->getWithDefault( "quadraticInitialize",       quadraticInitialize,      false);
  ps->getWithDefault( "exponentialInitialize_1D",  exponentialInitialize_1D, false);
  ps->getWithDefault( "exponentialInitialize_2D",  exponentialInitialize_2D, false);
  ps->getWithDefault( "triangularInitialize",      triangularInitialize,     false);

  if(sinusoidalInitialize){
    ps->getWithDefault("freq",freq,IntVector(0,0,0));
  }
  if(linearInitialize || triangularInitialize){
    ps->getWithDefault("slope",slope,Vector(0,0,0));
  }
  if(quadraticInitialize || exponentialInitialize_1D || exponentialInitialize_2D){
    ps->getWithDefault("coeff",coeff,Vector(0,0,0));
  }

  uniformInitialize = true;
  if(sinusoidalInitialize    || linearInitialize ||
     quadraticInitialize     || cubicInitialize ||
     exponentialInitialize_1D|| exponentialInitialize_2D ||
     triangularInitialize){
    uniformInitialize = false;
  }
}

//______________________________________________________________________
//
PassiveScalar::interiorRegion::interiorRegion(GeometryPieceP piece, ProblemSpecP& ps)
  : piece(piece)
{
  ps->require("scalar", value);
  ps->getWithDefault( "maxScalar" , clampValue, DBL_MAX );
}


//______________________________________________________________________
//     P R O B L E M   S E T U P
void PassiveScalar::problemSetup(GridP&, const bool isRestart)
{
  DOUT(dout_models_ps, "Doing problemSetup \t\t\t\tPASSIVE_SCALAR" );

  ProblemSpecP PS_ps = d_params->findBlock("PassiveScalar");
  d_matl = m_materialManager->parseAndLookupMaterial(PS_ps, "material");

  vector<int> m(1);
  m[0] = d_matl->getDWIndex();
  d_matl_set = scinew MaterialSet();
  d_matl_set->addAll(m);
  d_matl_set->addReference();
  d_matl_sub = d_matl_set->getUnion();

  //__________________________________
  // - create Label names
  // - register the scalar to be transported

  ProblemSpecP scalar_ps = PS_ps->findBlock("scalar");
  if (!scalar_ps){
    throw ProblemSetupException("PassiveScalar: Couldn't find scalar tag", __FILE__, __LINE__);
  }

  std::string name {""};
  scalar_ps->getAttribute( "name", name );
  if ( name ==""){
    throw ProblemSetupException("PassiveScalar: the scalar tag must have a valid name  <scalar name=X>", __FILE__, __LINE__);
  }
  std::string fullName = "scalar-"+name;

  d_scalar = scinew Scalar();
  d_scalar->index    = 0;
  d_scalar->name     = name;
  d_scalar->fullName = fullName;

  const TypeDescription* td_CCdouble = CCVariable<double>::getTypeDescription();


  d_scalar->Q_CCLabel                = VarLabel::create( fullName,                  td_CCdouble);
  d_scalar->diffusionCoef_CCLabel    = VarLabel::create( fullName +"_diffCoef",     td_CCdouble);
  d_scalar->Q_src_CCLabel            = VarLabel::create( fullName +"_src",          td_CCdouble);
  d_scalar->expDecayCoefLabel        = VarLabel::create( fullName +"_expDecayCoef", td_CCdouble);

  d_scalar->mag_grad_Q_CCLabel       = VarLabel::create( "mag_grad_"+fullName,  td_CCdouble);
  d_scalar->sum_Q_CCLabel            = VarLabel::create( "totalSum_"+fullName,  sum_vartype::getTypeDescription());

  registerTransportedVariable( d_matl_set,
                              d_scalar->Q_CCLabel,
                              d_scalar->Q_src_CCLabel);

  //__________________________________
  //  register the AMRrefluxing variables
  if(m_AMR){
    registerAMRRefluxVariable( d_matl_set, d_scalar->Q_CCLabel);
  }

  //__________________________________
  // Read in the constants for the scalar
  scalar_ps->getWithDefault("test_conservation",   d_runConservationTask, false);
  scalar_ps->getWithDefault("reinitializeDomain",  d_reinitializeDomain,  false);

  ProblemSpecP const_ps = scalar_ps->findBlock("constants");
  if(!const_ps) {
    throw ProblemSetupException("PassiveScalar: Couldn't find constants tag", __FILE__, __LINE__);
  }

  const_ps->getWithDefault( "decayRate",               d_scalar->decayRate,  0.0);
  const_ps->getWithDefault( "diffusivity",             d_scalar->diff_coeff, 0.0);
  const_ps->getWithDefault( "AMR_Refinement_Criteria", d_scalar->refineCriteria,1e100);

  //__________________________________
  // exponential Decay
  ProblemSpecP exp_ps = scalar_ps->findBlock("exponentialDecay");

  if( exp_ps ) {
    d_withExpDecayModel = true;
    exp_ps->require(        "c1", d_scalar->c1);
    exp_ps->getWithDefault( "c3", d_scalar->c3, 0.0 );

    // The c2 coefficient type can be either a constant or read from a table
    ProblemSpecP c2_ps = exp_ps->findBlock("c2");
    std::string type = "";
    c2_ps->getAttribute( "type", name );

    if( name == "variable"){    // read c2 from table
      d_decayCoef = variable;
      c2_ps->require( "filename", d_scalar->c2_filename );
    }
    else{           // c2 is a constant
      d_decayCoef = constant;
      c2_ps->require( "value", d_scalar->c2);
    }

    if( d_decayCoef == none ){
      throw ProblemSetupException("PassiveScalar: the scalar tag c2 must have either a constant value or a filename", __FILE__, __LINE__);
    }
  }

  //__________________________________
  //  Read in all geometry objects in the <Material> node.
  //  They may be referenenced.

  //__________________________________
  //  Read in all geometry objects/pieces in the <Material> node of the ups file.
  //  Needed since the user may referec
  if( isRestart || d_reinitializeDomain ){

    ProblemSpecP root_ps = d_params->getRootNode();
    ProblemSpecP mat_ps = root_ps->findBlockWithOutAttribute( "MaterialProperties" );

    // find all of the geom_objects
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
  //  Initialization: Read in the geometry objects for the scalar

  if( !isRestart || d_reinitializeDomain ){

    ProblemSpecP init_ps = scalar_ps->findBlock("initialization");

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

    d_scalar->regions.push_back(scinew Region(mainpiece, geom_obj_ps));
   }
  }

  if( d_scalar->regions.size() == 0 && !isRestart) {
    throw ProblemSetupException("Variable: "+fullName +" does not have any initial value regions", __FILE__, __LINE__);
  }

  //__________________________________
  //  Read in interior geometry objects for injecting a scalar in the domain
  ProblemSpecP srcs_ps = scalar_ps->findBlock("interiorSources");

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

      d_scalar->interiorRegions.push_back(scinew interiorRegion(mainpiece, geom_obj_ps));
    }
  }
}

//______________________________________________________________________
//
void PassiveScalar::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP model_ps = ps->appendChild("Model");
  model_ps->setAttribute( "type","PassiveScalar" );
  ProblemSpecP PS_ps = model_ps->appendChild( "PassiveScalar" );

  PS_ps->appendElement( "material",d_matl->getName() );
  ProblemSpecP scalar_ps = PS_ps->appendChild( "scalar" );
  scalar_ps->setAttribute( "name", d_scalar->name );

  scalar_ps->appendElement( "test_conservation",  d_runConservationTask );
  scalar_ps->appendElement( "reinitializeDomain", "false" );              // the user must manually override 

  ProblemSpecP const_ps = scalar_ps->appendChild( "constants" );
  const_ps->appendElement( "decayRate",                d_scalar->decayRate );
  const_ps->appendElement( "diffusivity",              d_scalar->diff_coeff );
  const_ps->appendElement( "AMR_Refinement_Criteria",  d_scalar->refineCriteria );

  if(d_withExpDecayModel){
    ProblemSpecP exp_ps = scalar_ps->appendChild( "exponentialDecay" );
    exp_ps->appendElement( "c1", d_scalar->c1 );

                    // The c2 coefficient type can be either a constant or read from a table
    ProblemSpecP c2_ps = exp_ps->appendChild("c2");

                    // read c2 from table
    if( d_decayCoef == variable){    
      c2_ps->setAttribute( "type", "variable" );
      c2_ps->appendElement( "filename", d_scalar->c2_filename );
    }
    else{           // c2 is a constant
      c2_ps->setAttribute( "type", "constant" );
      c2_ps->appendElement( "value", d_scalar->c2 );
    }
    exp_ps->appendElement( "c3", d_scalar->c3 );
  }

  //__________________________________
  //  initialization regions
  ProblemSpecP init_ps = scalar_ps->appendChild( "initialization" );

  vector<Region*>::const_iterator iter;
  for ( iter = d_scalar->regions.begin(); iter != d_scalar->regions.end(); iter++) {
    ProblemSpecP geom_ps = init_ps->appendChild( "geom_object" );

    (*iter)->piece->outputProblemSpec(geom_ps);
    geom_ps->appendElement( "scalar",(*iter)->initialScalar );
  }


  //__________________________________
  //  regions inside the domain
  if( d_scalar->interiorRegions.size() > 0 ){
    ProblemSpecP int_ps = scalar_ps->appendChild( "interiorSources" );

    vector<interiorRegion*>::const_iterator itr;
    for ( itr = d_scalar->interiorRegions.begin(); itr != d_scalar->interiorRegions.end(); itr++) {
      ProblemSpecP geom_ps = int_ps->appendChild("geom_object");
      (*itr)->piece->outputProblemSpec( geom_ps );

      geom_ps->appendElement( "scalar",   (*itr)->value );
      geom_ps->appendElement( "maxScalar",(*itr)->clampValue );
    }
  }
}


//______________________________________________________________________
// Read in a csv formatted file <x>,<y>,<z> value
void PassiveScalar::readTable( const Patch * patch,
                               const Level * level,
                               const std::string filename,
                               CCVariable<double>& c2 )
{
  static int count=1;

  std::ifstream ifs( filename.c_str() );
  if( !ifs ) {
    throw ProblemSetupException("ERROR: PassiveScalar::readTable: Unable to open the input file: " + filename, __FILE__, __LINE__);
  }

  proc0cout_cmp(patch->getID(), 0)
        << "  Reading in table ("<< filename <<") for variable coefficient c2\n";


  string line;            // row from the file
  int fpos = ifs.tellg(); // file fposition

  //__________________________________
  //  ignore header lines (#)
  while ( getline( ifs, line ) ){
    if ( line[0] != '#'){
      break;
    }
    fpos = ifs.tellg();
  }

  // rewind one line
  ifs.seekg( fpos );

  int nCells_read = 0;

  double x_CC;
  double y_CC;
  double z_CC;
  double phi;
  std::string str;

  //__________________________________
  //  read the rest of the file
  while ( std::getline( ifs, str, ',' ) ) {
    x_CC = std::stod( str );

    std::getline( ifs, str, ',' );
    y_CC = std::stod( str );

    std::getline( ifs, str, ',' );
    z_CC = std::stod( str );

    getline( ifs, str );
    phi = std::stod( str );

    Point pos( x_CC,y_CC,z_CC );

    if(level->containsPointIncludingExtraCells(pos)){
      IntVector c = level->getCellIndex( pos );

      if( patch->containsCell( c ) ){
        c2[c] = phi;
        nCells_read ++;
        //DOUTR( true, "x: " << x_CC << " y: " << y_CC << " z: " << z_CC << " cellIndex " << c << " phi: " << phi );
      }
    }
    else {
      ostringstream msg;
      msg << " ERROR: PassiveScalar::readTable: The coordinate ("<<pos<< ") is not contained on this level";
      throw ProblemSetupException( msg.str(), __FILE__, __LINE__ );
    }
  }

  // bulletproofing
  int patchNumCells = patch->getNumCells();

  if ( patchNumCells != nCells_read ){
    ostringstream msg;
    msg << " ERROR: PassiveScalar::readTable: number of cells read ("<< nCells_read
        << " != number of cells in the patch: " << patchNumCells;
    throw ProblemSetupException( msg.str(), __FILE__, __LINE__ );
  }

  count ++;
}



//______________________________________________________________________
//      S C H E D U L E   I N I T I A L I Z E
void PassiveScalar::scheduleInitialize(SchedulerP& sched,
                                       const LevelP& level)
{
  const string schedName = "PassiveScalar::scheduleInitialize_("+ d_scalar->fullName+")";
  printSchedule( level, dout_models_ps, schedName );


  const string taskName = "PassiveScalar::initialize_("+ d_scalar->fullName+")";
  Task* t = scinew Task(taskName, this, &PassiveScalar::initialize);

  t->requires(Task::NewDW, Ilb->timeStepLabel );

  if( d_withExpDecayModel ){
    t->computes( d_scalar->expDecayCoefLabel );
  }
  t->computes(d_scalar->Q_CCLabel);

  sched->addTask(t, level->eachPatch(), d_matl_set);

}


//______________________________________________________________________
//       I N I T I A L I Z E
void PassiveScalar::initialize(const ProcessorGroup*,
                               const PatchSubset* patches,
                               const MaterialSubset*,
                               DataWarehouse*,
                               DataWarehouse* new_dw)
{
  timeStep_vartype timeStep;
  new_dw->get(timeStep, VarLabel::find( timeStep_name) );

  bool isNotInitialTimeStep = (timeStep > 0);

  const Level* level = getLevel(patches);

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    const string msg = "Doing PassiveScalar::initialize_("+ d_scalar->fullName+")";
    printTask(patches, patch, dout_models_ps, msg);

    int indx = d_matl->getDWIndex();

    //__________________________________
    //  Initialize coefficient used in exponential decay model
    if( d_withExpDecayModel ){

      int id = patch->getID();

      proc0cout_cmp( id, 0)
              << "________________________PassiveScalar\n"
              << "  Coefficient c1: " << d_scalar->c1 << "\n";

      CCVariable<double> c2;
      new_dw->allocateAndPut(c2, d_scalar->expDecayCoefLabel, indx, patch);
      c2.initialize(0.0);

      if (d_decayCoef == constant){
        c2.initialize( d_scalar->c2 );
        proc0cout_cmp( id, 0)
              << "  Coefficient c2: " << d_scalar->c2 << "\n";
      }
      else{
        readTable( patch, level, d_scalar->c2_filename, c2 );
      }

       proc0cout_cmp( id, 0)
              << "  Coefficient c3: " << d_scalar->c3 << "\n"
              << "__________________________________\n";
    }

    //__________________________________
    // Passive Scalar
    CCVariable<double> f;
    new_dw->allocateAndPut(f, d_scalar->Q_CCLabel, indx, patch);
    f.initialize(0);

    //__________________________________
    //  Uniform initialization scalar field in a region
    for(vector<Region*>::iterator iter = d_scalar->regions.begin();
                                  iter != d_scalar->regions.end(); iter++){
      Region* region = *iter;

      if(region->uniformInitialize){

        for(CellIterator iter = patch->getExtraCellIterator(); !iter.done(); iter++){
          IntVector c = *iter;

          Point p = patch->cellPosition(c);
          if(region->piece->inside(p)) {
            f[c] = region->initialScalar;
          }
        } // Over cells
      }

      //__________________________________
      // Sinusoidal & linear initialization
      if(!region->uniformInitialize){
        IntVector freq = region->freq;

        // bulletproofing
        if(region->sinusoidalInitialize && freq.x()==0 && freq.y()==0 && freq.z()==0){
          throw ProblemSetupException("PassiveScalar: you need to specify a <freq> whenever you use sinusoidalInitialize", __FILE__, __LINE__);
        }

        Vector slope = region->slope;
        if((region->linearInitialize || region->triangularInitialize) && slope.x()==0 && slope.y()==0 && slope.z()==0){
          throw ProblemSetupException("PassiveScalar: you need to specify a <slope> whenever you use linearInitialize", __FILE__, __LINE__);
        }

        Vector coeff = region->coeff;
        if( (region->quadraticInitialize || region->exponentialInitialize_1D ||  region->exponentialInitialize_2D)
           && coeff.x()==0 && coeff.y()==0 && coeff.z()==0){
          cerr<<"coeff"<<coeff<<endl;
          throw ProblemSetupException("PassiveScalar: you need to specify a <coeff> for this initialization", __FILE__, __LINE__);
        }

        if(region->exponentialInitialize_1D &&  ( (coeff.x()*coeff.y()!=0) || (coeff.y()*coeff.z()!=0) || (coeff.x()*coeff.z()!=0) )  ) {
          throw ProblemSetupException("PassiveScalar: 1D Exponential Initialize. This profile is designed for 1D problems only. Try exponentialInitialize_2D instead",__FILE__, __LINE__);
        }


        if(region->exponentialInitialize_2D && (coeff.x()!=0) && (coeff.y()!=0) && (coeff.z()!=0) ) {
          throw ProblemSetupException("PassiveScalar: 2D Exponential Initialize. This profile is designed for 2D problems only, one <coeff> must equal zero",__FILE__, __LINE__);
        }

        Point lo = region->piece->getBoundingBox().lower();
        Point hi = region->piece->getBoundingBox().upper();
        Vector dist = hi.asVector() - lo.asVector();

        //__________________________________
        //
        for(CellIterator iter = patch->getExtraCellIterator(); !iter.done(); iter++){

          IntVector c = *iter;

          Point p = patch->cellPosition(c);
          if(region->piece->inside(p)) {
            // normalized distance
            Vector d = (p.asVector() - lo.asVector() )/dist;

            if(region->sinusoidalInitialize){
              f[c] = sin( 2.0 * freq.x() * d.x() * M_PI) +
                     sin( 2.0 * freq.y() * d.y() * M_PI)  +
                     sin( 2.0 * freq.z() * d.z() * M_PI);
            }
            if(region->linearInitialize){  // f[c] = kx + b
              f[c] = (slope.x() * d.x() + slope.y() * d.y() + slope.z() * d.z() ) +  region->initialScalar;
            }
            if(region->triangularInitialize){
              if(d.x() <= 0.5)
                f[c] = slope.x()*d.x();
              else
                f[c] = slope.x()*(1.0-d.x());
            }
            if(region->quadraticInitialize){
              if(d.x() <= 0.5)
                f[c] = pow(d.x(),2) - d.x();
              else{
                f[c] = pow( (1.0 - d.x()),2) - d.x();
              }
            }
            if(region->cubicInitialize){
              if(d.x() <= 0.5)
                f[c] = -1.3333333*pow(d.x(),3)  + pow(d.x(),2);
              else{
                f[c] = -1.3333333*pow( (1.0 - d.x()),3) + pow( (1.0 - d.x()),2);
              }
            }

            // This is a 2-D profile
            if(region->exponentialInitialize_2D) {
              double coeff1 = 0., coeff2 = 0. ,  d1 = 0. , d2= 0.;
              if (coeff.x()==0) {
                coeff1 = coeff.y();
                coeff2 = coeff.z();
                d1 = d.y();
                d2 = d.z();
              }
              else if (coeff.y()==0) {
                coeff1 = coeff.x();
                coeff2 = coeff.z();
                d1 = d.x();
                d2 = d.z();
              }
              else if (coeff.z()==0) {
                coeff1 = coeff.y();
                coeff2 = coeff.x();
                d1 = d.y();
                d2 = d.x();
              }
              f[c] = coeff1 * exp(-1.0/( d1 * ( 1.0 - d1 ) + 1e-100) )
                   * coeff2 * exp(-1.0/( d2 * ( 1.0 - d2 ) + 1e-100) );
            }

            // This is a 1-D profile - Donot use it for 2-D

            if (region->exponentialInitialize_1D ){
              f[c] = coeff.x() * exp(-1.0/( d.x() * ( 1.0 - d.x() ) + 1e-100) )
                   + coeff.y() * exp(-1.0/( d.y() * ( 1.0 - d.y() ) + 1e-100) )
                   + coeff.z() * exp(-1.0/( d.z() * ( 1.0 - d.z() ) + 1e-100) );
            }
          }
        }
      }  // sinusoidal Initialize
    } // regions
    setBC( f, d_scalar->fullName, patch, m_materialManager,indx, new_dw, isNotInitialTimeStep);
  }  // patches
}

//______________________________________________________________________
//
void PassiveScalar::scheduleRestartInitialize(SchedulerP   & sched,
                                                const LevelP & level)
{
  const string schedName = "PassiveScalar::scheduleRestartInitialize("+ d_scalar->fullName+")";
  printSchedule( level, dout_models_ps, schedName );

  // if reinitializing the domain
  if( d_reinitializeDomain ){
    scheduleInitialize( sched, level);
  }
  else if( d_withExpDecayModel ){
    const string taskName = "PassiveScalar::restartInitialize_("+ d_scalar->fullName+")";
    Task* t = scinew Task(taskName, this, &PassiveScalar::restartInitialize);

    t->computes( d_scalar->expDecayCoefLabel );

    sched->addTask(t, level->eachPatch(), d_matl_set);
  }
}

//______________________________________________________________________
void PassiveScalar::restartInitialize(const ProcessorGroup *,
                                      const PatchSubset    * patches,
                                      const MaterialSubset *,
                                      DataWarehouse        * ,
                                      DataWarehouse        * new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    const Level* level = getLevel(patches);

    const string msg = "Doing PassiveScalar::restartInitialize("+ d_scalar->fullName+")";
    printTask(patches, patch, dout_models_ps, msg);

    //__________________________________
    //  Initialize coefficient used in exponential decay model
    if( d_withExpDecayModel ){
      int indx = d_matl->getDWIndex();
      int id   = patch->getID();

      proc0cout_cmp( id, 0)
              << "________________________PassiveScalar\n"
              << "  Coefficient c1: " << d_scalar->c1 << "\n";
              

      CCVariable<double> c2;
      new_dw->allocateAndPut(c2, d_scalar->expDecayCoefLabel, indx, patch);
      c2.initialize(0.0);

      if (d_decayCoef == constant){
        c2.initialize( d_scalar->c2 );
        proc0cout_cmp( id, 0)
              << "  Coefficient c2: " << d_scalar->c2 << "\n";
      }
      else{
        readTable( patch, level, d_scalar->c2_filename, c2 );
      }
      
      proc0cout_cmp( id, 0)
              << "  Coefficient c3: " << d_scalar->c3 << "\n"
              << "__________________________________\n";
    }
  }
}

//______________________________________________________________________
//  Task:     ModifyThermoTransportProperties
//  Purpose:  compute the diffusion coefficient
void PassiveScalar::scheduleModifyThermoTransportProperties(SchedulerP& sched,
                                                            const LevelP& level,
                                                            const MaterialSet* /*ice_matls*/)
{
  const string schedName = "PassiveScalar::scheduleModifyThermoTransportProperties("+ d_scalar->fullName+")";
  printSchedule(level,dout_models_ps,schedName);

  const string taskName = "PassiveScalar::modifyThermoTransportProperties("+ d_scalar->fullName+")";

  Task* t = scinew Task( taskName, this,&PassiveScalar::modifyThermoTransportProperties);

  t->computes( d_scalar->diffusionCoef_CCLabel );
  sched->addTask( t, level->eachPatch(), d_matl_set);
}
//______________________________________________________________________
void PassiveScalar::modifyThermoTransportProperties(const ProcessorGroup*,
                                                    const PatchSubset* patches,
                                                    const MaterialSubset*,
                                                    DataWarehouse* /*old_dw*/,
                                                    DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    const string msg = "Doing PassiveScalar::modifyThermoTransportProperties_("+ d_scalar->fullName+")";
    printTask(patches, patch, dout_models_ps, msg );

    int indx = d_matl->getDWIndex();
    CCVariable<double> diffusionCoeff;
    new_dw->allocateAndPut( diffusionCoeff, d_scalar->diffusionCoef_CCLabel,indx, patch );

    diffusionCoeff.initialize( d_scalar->diff_coeff );
  }
}

//______________________________________________________________________
void PassiveScalar::computeSpecificHeat(CCVariable<double>& ,
                                        const Patch* ,
                                        DataWarehouse* ,
                                        const int )
{
  //none
}

//______________________________________________________________________
void PassiveScalar::scheduleComputeModelSources(SchedulerP& sched,
                                                const LevelP& level)
{
  const string schedName = "PassiveScalar::scheduleComputeModelSources_("+ d_scalar->fullName+")";
  printSchedule(level,dout_models_ps, schedName);

  const string taskName = "PassiveScalar::computeModelSources_("+ d_scalar->fullName+")";
  Task* t = scinew Task( taskName, this,&PassiveScalar::computeModelSources);

  Ghost::GhostType  gac = Ghost::AroundCells;
  Ghost::GhostType  gn  = Ghost::None;

  t->requires( Task::OldDW, Ilb->delTLabel, level.get_rep() );

  t->requires( Task::NewDW, d_scalar->diffusionCoef_CCLabel, gac,1 );
  t->requires( Task::OldDW, d_scalar->Q_CCLabel,             gac,1 );

  if ( d_withExpDecayModel ){
    t->requires( Task::OldDW, d_scalar->expDecayCoefLabel,   gn,0 );
    t->computes( d_scalar->expDecayCoefLabel );
  }

  t->modifies( d_scalar->Q_src_CCLabel );

  sched->addTask(t, level->eachPatch(), d_matl_set);
}

//______________________________________________________________________
void PassiveScalar::computeModelSources(const ProcessorGroup*,
                                        const PatchSubset* patches,
                                        const MaterialSubset* matls,
                                        DataWarehouse* old_dw,
                                        DataWarehouse* new_dw)
{
  const Level* level = getLevel(patches);
  delt_vartype delT;
  old_dw->get(delT, Ilb->delTLabel, level);

  Ghost::GhostType gac = Ghost::AroundCells;
  Ghost::GhostType gn  = Ghost::None;

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    const string msg = "Doing PassiveScalar::computeModelSources_("+ d_scalar->fullName+")";
    printTask(patches, patch, dout_models_ps, msg);

    constCCVariable<double> f_old;
    constCCVariable<double> diff_coeff;
    CCVariable<double>  f_src;

    int indx = d_matl->getDWIndex();
    old_dw->get(f_old,      d_scalar->Q_CCLabel,             indx, patch, gac,1);
    new_dw->get(diff_coeff, d_scalar->diffusionCoef_CCLabel, indx, patch, gac,1);

    new_dw->allocateAndPut(f_src, d_scalar->Q_src_CCLabel,indx, patch);

    f_src.initialize(0.0);

    //__________________________________
    //  scalar diffusion
    double diff_coeff_test = d_scalar->diff_coeff;

    if(diff_coeff_test != 0.0){
      bool use_vol_frac = false; // don't include vol_frac in diffusion calc.
      CCVariable<double> placeHolder;

      scalarDiffusionOperator(new_dw, patch, use_vol_frac, f_old,
                              placeHolder, f_src, diff_coeff, delT);
    }
    //__________________________________
    //  constant decay
    const double decayRate = d_scalar->decayRate;
    if ( decayRate != 0 ){

      for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
        IntVector c = *iter;
        f_src[c] = -delT * decayRate;

#if 0
        if( c == IntVector(3,3,3) ){
          cout << " Linear Decay: " << d_scalar->fullName << endl;
          cout << "\n f_old: " << f_old[c]
               << "\n fsrc:  " << f_src[c] << endl;

        }
#endif
      }
    }

    //__________________________________
    //  exponential decay
    if ( d_withExpDecayModel ){

      constCCVariable<double> c2;
      old_dw->get( c2, d_scalar->expDecayCoefLabel, indx, patch, gn,0);

      new_dw->transferFrom( old_dw, d_scalar->expDecayCoefLabel, patches, matls );

      const double c1 = d_scalar->c1;
      const double c3 = d_scalar->c3;

      for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
        IntVector c = *iter;
        f_src[c]  = f_old[c] * exp(-(c1 * c2[c] + c3)* delT ) - f_old[c];
#if 0
        if( c == IntVector(3,3,3) ){
          cout << " ExponentialDecay: " << d_scalar->fullName <<endl;
          cout << "\n f_old: " << f_old[c]
               << "\n c1:    " << c1
               << "\n c2[c]: " << c2[c]
               << "\n c3:    " << c3
               << "\n exp(): " << exp (-(c1 * c2[c] + c3) * delT )
               << "\n fsrc:  " << f_src[c] << endl;
        }
#endif
      }
    }

    //__________________________________
    //  interior source regions
    for(vector<interiorRegion*>::iterator iter = d_scalar->interiorRegions.begin();
                                          iter != d_scalar->interiorRegions.end(); iter++){
      interiorRegion* region = *iter;

      for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
        IntVector c = *iter;

        Point p = patch->cellPosition(c);

        if(region->piece->inside(p)) {
          f_src[c] = region->value;

          double f_test = f_old[c] + f_src[c];
          double clamp = region->clampValue;

          if (f_test > clamp ){
            f_src[c] = clamp - f_old[c];
          }
        }
      } // Over cells
    }  //interiorRegions

    //__________________________________
    //  Clamp:  a scalar must always be > 0
    for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
      IntVector c = *iter;

      if( (f_old[c] + f_src[c]) < 0.0){
        f_src[c] = -f_old[c];
      }
    }

  }  // patches
}
//__________________________________
void PassiveScalar::scheduleComputeStableTimeStep(SchedulerP&,
                                                  const LevelP&)
{
  // None necessary...
}

//______________________________________________________________________
void PassiveScalar::scheduleTestConservation(SchedulerP& sched,
                                             const PatchSet* patches)
{
  const Level* level = getLevel(patches);
  int L = level->getIndex();

  if(d_runConservationTask && L == 0){

    const string taskName = "PassiveScalar::scheduleTestConservation_("+ d_scalar->fullName+")";
    printSchedule(patches, dout_models_ps, taskName);


    Task* t = scinew Task( taskName, this,&PassiveScalar::testConservation);

    Ghost::GhostType  gn = Ghost::None;
    // compute sum(scalar_f * mass)
    t->requires(Task::OldDW, Ilb->delTLabel, getLevel(patches) );
    t->requires(Task::NewDW, d_scalar->Q_CCLabel,  gn,0);
    t->requires(Task::NewDW, Ilb->rho_CCLabel,     gn,0);
    t->requires(Task::NewDW, Ilb->uvel_FCMELabel,  gn,0);
    t->requires(Task::NewDW, Ilb->vvel_FCMELabel,  gn,0);
    t->requires(Task::NewDW, Ilb->wvel_FCMELabel,  gn,0);

    t->computes(d_scalar->sum_Q_CCLabel);

    sched->addTask(t, patches, d_matl_set);
  }
}

//______________________________________________________________________
void PassiveScalar::testConservation(const ProcessorGroup*,
                                     const PatchSubset* patches,
                                     const MaterialSubset* /*matls*/,
                                     DataWarehouse* old_dw,
                                     DataWarehouse* new_dw)
{
  const Level* level = getLevel(patches);
  delt_vartype delT;
  old_dw->get(delT, Ilb->delTLabel, level);
  Ghost::GhostType gn = Ghost::None;

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    const string msg = "Doing PassiveScalar::testConservation_("+ d_scalar->fullName+")";
    printTask(patches, patch, dout_models_ps, msg);

    //__________________________________
    //  conservation of f test
    constCCVariable<double>   rho_CC;
    constCCVariable<double>   f;
    constSFCXVariable<double> uvel_FC;
    constSFCYVariable<double> vvel_FC;
    constSFCZVariable<double> wvel_FC;
    int indx = d_matl->getDWIndex();

    new_dw->get(f,       d_scalar->Q_CCLabel,  indx, patch, gn,0);
    new_dw->get(rho_CC,  Ilb->rho_CCLabel,     indx, patch, gn,0);
    new_dw->get(uvel_FC, Ilb->uvel_FCMELabel,  indx, patch, gn,0);
    new_dw->get(vvel_FC, Ilb->vvel_FCMELabel,  indx, patch, gn,0);
    new_dw->get(wvel_FC, Ilb->wvel_FCMELabel,  indx, patch, gn,0);

    Vector dx = patch->dCell();
    double cellVol = dx.x()*dx.y()*dx.z();

    CCVariable<double> q_CC;
    new_dw->allocateTemporary(q_CC, patch);

    for(CellIterator iter = patch->getExtraCellIterator(); !iter.done(); iter++) {
      IntVector c = *iter;
      q_CC[c] = rho_CC[c]*cellVol*f[c];
    }

    double sum_mass_f;
    conservationTest<double>(patch, delT, q_CC, uvel_FC, vvel_FC, wvel_FC, sum_mass_f);

    new_dw->put(sum_vartype(sum_mass_f), d_scalar->sum_Q_CCLabel);
  }
}

//______________________________________________________________________
//
void PassiveScalar::scheduleErrorEstimate(const LevelP& coarseLevel,
                                          SchedulerP& sched)
{

  const string taskName = "PassiveScalar::scheduleErrorEstimate_("+ d_scalar->fullName+")";
  printSchedule( coarseLevel,dout_models_ps, taskName );

  Task* t = scinew Task( taskName, this, &PassiveScalar::errorEstimate, false);

  Ghost::GhostType  gac  = Ghost::AroundCells;

  t->requires(Task::NewDW, d_scalar->Q_CCLabel,  d_matl_sub, gac,1);
  t->computes(d_scalar->mag_grad_Q_CCLabel, d_matl_sub);

  t->modifies( m_regridder->getRefineFlagLabel(),      m_regridder->refineFlagMaterials() );
  t->modifies( m_regridder->getRefinePatchFlagLabel(), m_regridder->refineFlagMaterials() );

  // define the material set of 0 and whatever the passive scalar index is
  vector<int> m;
  m.push_back(0);
  m.push_back(d_matl->getDWIndex());

  MaterialSet* matl_set;
  matl_set = scinew MaterialSet();
  matl_set->addAll_unique(m);
  matl_set->addReference();

  sched->addTask(t, coarseLevel->eachPatch(), matl_set);
}
/*_____________________________________________________________________
 Function~  PassiveScalar::errorEstimate--
______________________________________________________________________*/
void PassiveScalar::errorEstimate(const ProcessorGroup*,
                                  const PatchSubset* patches,
                                  const MaterialSubset*,
                                  DataWarehouse*,
                                  DataWarehouse* new_dw,
                                  bool)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    const string msg = "Doing PassiveScalar::errorEstimate_("+ d_scalar->fullName+")";
    printTask(patches, patch, dout_models_ps, msg);

    Ghost::GhostType  gac  = Ghost::AroundCells;
    const VarLabel* refineFlagLabel = m_regridder->getRefineFlagLabel();
    const VarLabel* refinePatchLabel= m_regridder->getRefinePatchFlagLabel();

    CCVariable<int> refineFlag;
    new_dw->getModifiable(refineFlag, refineFlagLabel, 0, patch);

    PerPatch<PatchFlagP> refinePatchFlag;
    new_dw->get(refinePatchFlag, refinePatchLabel, 0, patch);

    int indx = d_matl->getDWIndex();
    constCCVariable<double> f;
    CCVariable<double> mag_grad_f;

    new_dw->get(f,                     d_scalar->Q_CCLabel,          indx, patch,gac,1);
    new_dw->allocateAndPut(mag_grad_f, d_scalar->mag_grad_Q_CCLabel, indx, patch);
    mag_grad_f.initialize(0.0);

    //__________________________________
    // compute gradient
    Vector dx = patch->dCell();

    for(CellIterator iter = patch->getCellIterator();!iter.done();iter++){
      IntVector c = *iter;
      Vector grad_f;

      for(int dir = 0; dir <3; dir ++ ) {
        IntVector r = c;
        IntVector l = c;
        double inv_dx = 0.5 /dx[dir];
        r[dir] += 1;
        l[dir] -= 1;
        grad_f[dir] = (f[r] - f[l])*inv_dx;
      }
      mag_grad_f[c] = grad_f.length();
    }
    //__________________________________
    // set refinement flag
    PatchFlag* refinePatch = refinePatchFlag.get().get_rep();

    for(CellIterator iter = patch->getCellIterator();!iter.done();iter++){
      IntVector c = *iter;

      if( mag_grad_f[c] > d_scalar->refineCriteria){
        refineFlag[c] = true;
        refinePatch->set();
      }
    }
  }  // patches
}
