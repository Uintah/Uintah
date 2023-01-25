#include <CCA/Components/Arches/TransportEqns/DQMOMEqn.h>
#include <CCA/Components/Arches/ParticleModels/ParticleTools.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermFactory.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermBase.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermBase.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Parallel/Parallel.h>
#include <CCA/Components/Arches/ConvectionHelper.h>
#include <Core/Util/Timers/Timers.hpp>

using namespace std;
using namespace Uintah;

//---------------------------------------------------------------------------
// Builder:
DQMOMEqnBuilder::DQMOMEqnBuilder( ArchesLabel* fieldLabels,
                                  ExplicitTimeInt* timeIntegrator,
                                  string eqnName,
                                  string ic_name,
                                  const int quadNode ) :
DQMOMEqnBuilderBase( fieldLabels, timeIntegrator, eqnName ), d_quadNode(quadNode)
{
  d_ic_name = ic_name;
}
DQMOMEqnBuilder::~DQMOMEqnBuilder()
{
}

EqnBase*
DQMOMEqnBuilder::build(){
  return scinew DQMOMEqn(d_fieldLabels, d_timeIntegrator, d_eqnName, d_ic_name, d_quadNode);
}
// End Builder
//---------------------------------------------------------------------------

DQMOMEqn::DQMOMEqn( ArchesLabel* fieldLabels, ExplicitTimeInt* timeIntegrator, string eqnName, string ic_name, const int quadNode )
:
EqnBase( fieldLabels, timeIntegrator, eqnName ), d_quadNode(quadNode)
{
  d_ic_name = ic_name;
  d_weight = false;

  std::string varname = eqnName;
  d_transportVarLabel = VarLabel::create(varname,
            CCVariable<double>::getTypeDescription());
  varname = eqnName+"_Fdiff";
  d_FdiffLabel = VarLabel::create(varname,
            CCVariable<double>::getTypeDescription());
  varname = eqnName+"_Fconv";
  d_FconvLabel = VarLabel::create(varname,
            CCVariable<double>::getTypeDescription());
  varname = eqnName+"_RHS";
  d_RHSLabel = VarLabel::create(varname,
            CCVariable<double>::getTypeDescription());
  varname = eqnName+"_src";
  d_sourceName = varname;

  std::string node;
  std::stringstream sqn;
  sqn << d_quadNode;
  node = sqn.str();
  varname = d_ic_name+"_"+node;
  d_icLabel = VarLabel::create(varname,
            CCVariable<double>::getTypeDescription());
}

DQMOMEqn::~DQMOMEqn()
{
  VarLabel::destroy(d_FdiffLabel);
  VarLabel::destroy(d_FconvLabel);
  VarLabel::destroy(d_RHSLabel);
  VarLabel::destroy(d_transportVarLabel);
  VarLabel::destroy(d_icLabel);
}
//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
void
DQMOMEqn::problemSetup( const ProblemSpecP& inputdb )
{

  // NOTE: some of this may be better off in the EqnBase.cc class
  ProblemSpecP db = inputdb;

  d_boundaryCond->problemSetup( db, d_eqnName );
  unsigned int Nqn = ArchesCore::get_num_env( db, ArchesCore::DQMOM_METHOD );

  ProblemSpecP db_root = db->getRootNode();
  ProblemSpecP dqmom_db = db_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("DQMOM");
  std::string which_dqmom;
  dqmom_db->getAttribute( "type", which_dqmom );
  if ( which_dqmom == "unweightedAbs" )
    d_unweighted = true;
  else
    d_unweighted = false;

  db->getWithDefault("turbulentPrandtlNumber",d_turbPrNo,0.4);

  // save the weight label instead of having to find it every time
  DQMOMEqnFactory& dqmomFactory = DQMOMEqnFactory::self();
  string name = "w_qn";
  string node;
  std::stringstream out;
  out << d_quadNode;
  node = out.str();
  name += node;
  EqnBase& temp_eqn = dqmomFactory.retrieve_scalar_eqn(name);
  DQMOMEqn& eqn = dynamic_cast<DQMOMEqn&>(temp_eqn);
  d_weightLabel = eqn.getTransportEqnLabel();
  d_w_small = eqn.getSmallClipPlusTol();
  if( d_w_small == 0.0 ) {
    d_w_small = 1e-16;
  }

  // Discretization information:
  db->getWithDefault( "conv_scheme", d_convScheme, "upwind");
  db->getWithDefault( "doConv", d_doConv, false);
  db->getWithDefault( "doDiff", d_doDiff, false);
  d_addSources = true;
  d_addExtraSources = false;
  db->getWithDefault( "molecular_diffusivity", d_mol_diff, 0.0);
  if ( !d_weight ){

    if ( db->findBlock("nominal_values") ){
      db->require( "nominal_values", d_nominal );
    } else {
      d_nominal.resize(Nqn);
      for ( auto i = d_nominal.begin(); i != d_nominal.end(); i++ ){
        //putting a random value in here since it wasn't specified.
        //If using birth, then this value should never be used.
        *i = 101010101010.101010101010;
      }
    }
  }

  if ( d_convScheme == "upwind" ){
    d_which_limiter = UPWIND;
  }
  else if ( d_convScheme == "super_bee" ){
    d_which_limiter = SUPERBEE;
  }
  else if ( d_convScheme == "vanleer" ){
    d_which_limiter = VANLEER;
  }
  else if ( d_convScheme == "roe_minmod"){
    d_which_limiter = ROE;
  }
  else {
  throw InvalidValue("Error: Limiter choice not recognized for eqn: "+d_eqnName, __FILE__, __LINE__);
  }

  // Models (source terms):
  for (ProblemSpecP m_db = db->findBlock("model"); m_db != nullptr; m_db = m_db->findNextBlock("model")){
    string model_name;
    string model_type;
    m_db->getAttribute("label", model_name);

    // now tag on the internal coordinate
    std::string model_qn_name = ArchesCore::append_qn_env(model_name, d_quadNode);

    // put it in the list
    d_models.push_back(model_qn_name);

    map<string,string>::iterator icheck = d_type_to_model.find(model_type);
    if ( icheck == d_type_to_model.end() ){
      std::string model_type = ArchesCore::get_model_type( m_db, model_name, ArchesCore::DQMOM_METHOD );
      d_type_to_model[model_type] = model_name;
    }

  }

  // Clipping:
  // defaults:
  clip.activated = false;
  clip.do_low  = false;
  clip.do_high = false;

  ProblemSpecP db_clipping = db->findBlock("Clipping");

  if (db_clipping) {

    clip.activated = true;

    db_clipping->getWithDefault("low", clip.low,  -1.e16);
    db_clipping->getWithDefault("high",clip.high, 1.e16);
    db_clipping->getWithDefault("tol", clip.tol, 1e-10);

    if ( db_clipping->findBlock("low") )
      clip.do_low = true;

    if ( db_clipping->findBlock("high") )
      clip.do_high = true;

    if ( !clip.do_low && !clip.do_high )
      throw InvalidValue("Error: A low or high clipping must be specified if the <Clipping> section is activated.", __FILE__, __LINE__);

  }

  //bullet proofing for weights
  if (d_weight) {

    if (!clip.activated) {

      //By default, set the low value for this weight to 0 and run on low clipping
      clip.activated = true;
      clip.low = 1e-100;
      clip.tol = 0.0;
      clip.do_low = true;

    } else {

      if ( !clip.do_low ){

        //weights always have low clip values!  ie, negative weights not allowed
        clip.do_low = true;
        clip.low = 1e-100;
        clip.tol = 0.0;

      }
    }
  }

  // Scaling information:
  if ( db->findBlock("scaling_const") ){
    db->require("scaling_const", d_scalingConstant );
    if ( Nqn != d_scalingConstant.size() ){
      throw InvalidValue("Error: The number of scaling constants isn't consistent with the number of environments for: "+d_ic_name, __FILE__, __LINE__);
    }
  } else {
    d_scalingConstant.resize(Nqn);
    for ( auto i = d_scalingConstant.begin(); i != d_scalingConstant.end(); i++ ){
      *i = 1.;
    }
  }

  // Extra Source terms (for mms and other tests):
  if (db->findBlock("src")){
    string srcname;
    d_addExtraSources = true;
    for (ProblemSpecP src_db = db->findBlock("src"); src_db != nullptr; src_db = src_db->findNextBlock("src")){
      src_db->getAttribute("label", srcname);
      //which sources are turned on for this equation
      d_sources.push_back( srcname );

    }
  }

  if(d_unweighted == true && d_weight == false){
    string srcname =  d_eqnName + "_unw_src";
    d_addExtraSources = true;
    d_sources.push_back( srcname );
  }

  // There should be some mechanism to make sure that when environment-specific
  // initialization functions are used, ALL environments are specified
  // (Charles)

  // Initialization (new way):
  ProblemSpecP db_initialValue = db->findBlock("initialization");
  if (db_initialValue) {

    db_initialValue->getAttribute("type", d_initFunction);

    // ---------- Constant initialization function ------------------------
    if (d_initFunction == "constant") {

        // Constant: prevent weighted abscissas from being both the same and nonzero
        db_initialValue->require("constant", d_constant_init);
        if( d_weight == false && d_constant_init != 0.0 ) {
          stringstream err_msg;
          err_msg <<  "ERROR: Arches: DQMOMEqn: You can't initialize abscissas of all environments for " + d_eqnName + " to the same non-zero constant value ";
                 err_msg << d_constant_init << " : your A matrix will be singular!  Use 'env_constant' instead of 'constant' for your initialization type.\n";
          throw ProblemSetupException(err_msg.str(),__FILE__,__LINE__);
        } else {
          d_constant_init /= d_scalingConstant[d_quadNode];
        }

    // -------- Environment constant initialization function --------------
    } else if (d_initFunction == "env_constant" ) {

      if ( !db_initialValue->findBlock("env_constant") ) {
        string err_msg = "ERROR: Arches: DQMOMEqn: Could not initialize equation "+d_eqnName+": You specified an 'env_constant' initialization function but did not include any 'env_constant' tags! \n";
        throw ProblemSetupException(err_msg,__FILE__,__LINE__);
      }

      // Environment constant: get the value of the constants
      for( ProblemSpecP db_env_constants = db_initialValue->findBlock("env_constant");
           db_env_constants != nullptr; db_env_constants = db_env_constants->findNextBlock("env_constant") ) {

        string s_tempQuadNode;
        db_env_constants->getAttribute("qn", s_tempQuadNode);
        int i_tempQuadNode = atoi( s_tempQuadNode.c_str() );

        string s_constant;
        db_env_constants->getAttribute("value", s_constant);
        double constantValue = atof( s_constant.c_str() );
        if( i_tempQuadNode == d_quadNode )
          d_constant_init = constantValue / d_scalingConstant[d_quadNode];
      }

    // ------- (Environment & Uniform) Step initialization function ------------
    // NOTE: Right now the only environment-specific attribute of the step function
    //       is the value.  This can be changed later, if someone wants to have
    //       unique step function directions/start/stop for each environment.
    // (Charles)
    } else if (d_initFunction == "step" || d_initFunction == "env_step") {

      // Step functions: prevent uniform steps for abscissa values
      if( d_initFunction == "step" && d_weight == false ) {
        string err_msg = "ERROR: Arches: DQMOMEqn: You can't initialize all quadrature nodes for "+d_eqnName+" to the same step function value, your A matrix will be singular! Use 'env_step' instead of 'step' for your initialization type.\n";
        throw ProblemSetupException(err_msg, __FILE__, __LINE__);
      }

      // Step functions: get step direction
      db_initialValue->require("step_direction", d_step_dir);

      // Step functions: find start/stop location
      if( db_initialValue->findBlock("step_start") ) {
        b_stepUsesPhysicalLocation = true;
        db_initialValue->require("step_start", d_step_start);
        db_initialValue->require("step_end"  , d_step_end);
      } else if ( db_initialValue->findBlock("step_cellstart") ) {
        b_stepUsesCellLocation = true;
        db_initialValue->require("step_cellstart", d_step_cellstart);
        db_initialValue->require("step_cellend", d_step_cellend);
        // swap if out of order
        if(d_step_cellstart > d_step_cellend) {
          int temp = d_step_cellstart;
          d_step_cellstart = d_step_cellend;
          d_step_cellend = temp;
        }
      } else {
        string err_msg = "ERROR: Arches: DQMOMEqn: Could not initialize 'env_step' for equation "+d_eqnName+": You did not specify a starting or stopping point!  Add <step_cellstart> and <step_cellend>, or <step_cellstart> and <step_cellend>! \n";
        throw ProblemSetupException(err_msg,__FILE__,__LINE__);
      }//end start/stop init.

      // Step functions: get step values
      if (d_initFunction == "step") {
        db_initialValue->require("step_value", d_step_value);
        d_step_value /= d_scalingConstant[d_quadNode];

      } else if (d_initFunction == "env_step") {

        if( !(db_initialValue->findBlock("env_step_value")) ) {
          string err_msg = "ERROR: Arches: DQMOMEqn: Could not initialize 'evn_step' for equation "+d_eqnName+": You did not specify any <env_step_value>! \n";
          throw ProblemSetupException(err_msg,__FILE__,__LINE__);
        }
        for( ProblemSpecP db_env_step_value = db_initialValue->findBlock("env_step_value");
             db_env_step_value != nullptr; db_env_step_value = db_env_step_value->findNextBlock("env_step_value") ) {

          string s_tempQuadNode;
          db_env_step_value->getAttribute("qn", s_tempQuadNode);
          int i_tempQuadNode = atoi( s_tempQuadNode.c_str() );

          string s_step_value;
          db_env_step_value->getAttribute("value", s_step_value);
          double step_value = atof( s_step_value.c_str() );

          if( i_tempQuadNode == d_quadNode ) {
            d_step_value = step_value / d_scalingConstant[d_quadNode];
          }
        }
      }//end step_value init.

    } else if (d_initFunction == "mms1") {
      //currently nothing to do here.

    } else if ( d_initFunction == "geometry_fill" ){

      db_initialValue->require("constant_inside", d_constant_in_init);              //fill inside geometry
      db_initialValue->getWithDefault( "constant_outside", d_constant_out_init, 1.e-15); //fill outside geometry

      ProblemSpecP the_geometry = db_initialValue->findBlock("geom_object");
      if (the_geometry) {
        GeometryPieceFactory::create(the_geometry, d_initGeom);
      } else {
        throw ProblemSetupException("You are missing the geometry specification (<geom_object>) for the transport eqn. initialization!", __FILE__, __LINE__);
      }


    // ------------ Other initialization function --------------------
    }
  }

  // need particle info for partMassFlowInlet
  d_partVelNames = std::vector<std::string>(3,"NotSet");
  if (ArchesCore::check_for_particle_method(db, ArchesCore::DQMOM_METHOD )){
    ArchesCore::PARTICLE_ROLE vel_enums[] = {ArchesCore::P_XVEL,
                                             ArchesCore::P_YVEL,
                                             ArchesCore::P_ZVEL};
    for(unsigned int i = 0; i<3; i++) {
      std::string velLabelName =ArchesCore::parse_for_particle_role_to_label(db,vel_enums[i]);
      d_partVelNames[i]=velLabelName+"_qn";

    }
  }
}

//---------------------------------------------------------------------------
// Method: Schedule the evaluation of the transport equation.
//---------------------------------------------------------------------------
void
DQMOMEqn::sched_evalTransportEqn( const LevelP& level,
                                  SchedulerP& sched, int timeSubStep,
                                  EQN_BUILD_PHASE phase )
{

  if (timeSubStep == 0) {
    sched_initializeVariables( level, sched );
  }

  if (d_addExtraSources) {
    sched_computeSources( level, sched, timeSubStep );
  }

  sched_buildTransportEqn( level, sched, timeSubStep );

}

//---------------------------------------------------------------------------
// Method: Add source to RHS and time update the eqn.
//---------------------------------------------------------------------------
void
DQMOMEqn::sched_updateTransportEqn( const LevelP& level,
                                    SchedulerP& sched, int timeSubStep )
{

  sched_addSources( level, sched, timeSubStep );

  sched_solveTransportEqn( level, sched, timeSubStep );

}


//---------------------------------------------------------------------------
// Method: Schedule the intialization of the variables.
//---------------------------------------------------------------------------
void
DQMOMEqn::sched_initializeVariables( const LevelP& level, SchedulerP& sched )
{
  string taskname = "DQMOMEqn::initializeVariables";
  Task* tsk = scinew Task(taskname, this, &DQMOMEqn::initializeVariables);
  Ghost::GhostType gn = Ghost::None;

  //New
  tsk->computes(d_transportVarLabel);
  tsk->computes(d_icLabel);
  tsk->computes(d_RHSLabel);
  tsk->computes(d_FconvLabel);
  tsk->computes(d_FdiffLabel);
  tsk->computes(d_X_flux_label);
  tsk->computes(d_Y_flux_label);
  tsk->computes(d_Z_flux_label);
  tsk->computes(d_X_psi_label);
  tsk->computes(d_Y_psi_label);
  tsk->computes(d_Z_psi_label);

  std::string name = ArchesCore::append_env( "face_pvel_x", d_quadNode );
  d_face_pvel_x = VarLabel::find(name);
  name = ArchesCore::append_env( "face_pvel_y", d_quadNode );
  d_face_pvel_y = VarLabel::find(name);
  name = ArchesCore::append_env( "face_pvel_z", d_quadNode );
  d_face_pvel_z = VarLabel::find(name);

  //Old
  tsk->requires(Task::OldDW, d_transportVarLabel, gn, 0);
  sched->addTask(tsk, level->eachPatch(), d_fieldLabels->d_materialManager->allMaterials( "Arches" ));
}
//---------------------------------------------------------------------------
// Method: Actually initialize the variables.
//---------------------------------------------------------------------------
void DQMOMEqn::initializeVariables( const ProcessorGroup* pc,
                                    const PatchSubset* patches,
                                    const MaterialSubset* matls,
                                    DataWarehouse* old_dw,
                                    DataWarehouse* new_dw )
{

  //patch loop
  for (int p=0; p < patches->size(); p++){

    Ghost::GhostType  gn  = Ghost::None;

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_materialManager->getMaterial( "Arches", archIndex)->getDWIndex();

    CCVariable<double> newVar;
    CCVariable<double> icValue;
    constCCVariable<double> oldVar;
    new_dw->allocateAndPut( newVar, d_transportVarLabel, matlIndex, patch );
    new_dw->allocateAndPut( icValue , d_icLabel, matlIndex, patch );
    old_dw->get(oldVar, d_transportVarLabel, matlIndex, patch, gn, 0);

    newVar.initialize(  0.0);
    icValue.initialize( 0.0);

    // copy old into new
    newVar.copyData(oldVar);

    CCVariable<double> Fdiff;
    CCVariable<double> Fconv;
    CCVariable<double> RHS;

    new_dw->allocateAndPut( Fdiff, d_FdiffLabel, matlIndex, patch );
    new_dw->allocateAndPut( Fconv, d_FconvLabel, matlIndex, patch );
    new_dw->allocateAndPut( RHS, d_RHSLabel, matlIndex, patch );

    Fdiff.initialize(0.0);
    Fconv.initialize(0.0);
    RHS.initialize(0.0);

    SFCXVariable<double> x_flux;
    SFCYVariable<double> y_flux;
    SFCZVariable<double> z_flux;
    new_dw->allocateAndPut( x_flux, d_X_flux_label, matlIndex, patch );
    new_dw->allocateAndPut( y_flux, d_Y_flux_label, matlIndex, patch );
    new_dw->allocateAndPut( z_flux, d_Z_flux_label, matlIndex, patch );
    x_flux.initialize(0.0);
    y_flux.initialize(0.0);
    z_flux.initialize(0.0);

    SFCXVariable<double> x_psi;
    SFCYVariable<double> y_psi;
    SFCZVariable<double> z_psi;
    new_dw->allocateAndPut( x_psi, d_X_psi_label, matlIndex, patch );
    new_dw->allocateAndPut( y_psi, d_Y_psi_label, matlIndex, patch );
    new_dw->allocateAndPut( z_psi, d_Z_psi_label, matlIndex, patch );
    x_psi.initialize(1.0);
    y_psi.initialize(1.0);
    z_psi.initialize(1.0);

  }
}

//---------------------------------------------------------------------------
// Method: Schedule compute for psi
// Probably not needed for DQMOM EQN
//---------------------------------------------------------------------------
void
DQMOMEqn::sched_computePsi( const LevelP& level, SchedulerP& sched )
{

  string taskname = "DQMOMEqn::computePsi";

  Task* tsk = scinew Task( taskname, this, &DQMOMEqn::computePsi );

  tsk->modifies(d_X_psi_label);
  tsk->modifies(d_Y_psi_label);
  tsk->modifies(d_Z_psi_label);

  tsk->requires(Task::NewDW, d_transportVarLabel, Ghost::AroundCells, 2);
  tsk->requires(Task::NewDW, d_face_pvel_x, Ghost::AroundCells, 1);
  tsk->requires(Task::NewDW, d_face_pvel_y, Ghost::AroundCells, 1);
  tsk->requires(Task::NewDW, d_face_pvel_z, Ghost::AroundCells, 1);
  tsk->requires(Task::OldDW, d_fieldLabels->d_areaFractionFXLabel, Ghost::AroundCells, 2);
  tsk->requires(Task::OldDW, d_fieldLabels->d_areaFractionFYLabel, Ghost::AroundCells, 2);
  tsk->requires(Task::OldDW, d_fieldLabels->d_areaFractionFZLabel, Ghost::AroundCells, 2);

  sched->addTask(tsk, level->eachPatch(), d_fieldLabels->d_materialManager->allMaterials( "Arches" ));

}
void
DQMOMEqn::computePsi( const ProcessorGroup* pc,
                      const PatchSubset* patches,
                      const MaterialSubset* matls,
                      DataWarehouse* old_dw,
                      DataWarehouse* new_dw )
{
  //patch loop
  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_materialManager->getMaterial( "Arches", archIndex)->getDWIndex();
    //Vector Dx = patch->dCell();

    Ghost::GhostType  gac = Ghost::AroundCells;

    SFCXVariable<double> psi_x;
    SFCYVariable<double> psi_y;
    SFCZVariable<double> psi_z;

    new_dw->getModifiable(psi_x, d_X_psi_label, matlIndex, patch);
    new_dw->getModifiable(psi_y, d_Y_psi_label, matlIndex, patch);
    new_dw->getModifiable(psi_z, d_Z_psi_label, matlIndex, patch);

    constSFCXVariable<double> u;
    constSFCYVariable<double> v;
    constSFCZVariable<double> w;

    new_dw->get( u, d_face_pvel_x, matlIndex, patch, gac, 1 );
    new_dw->get( v, d_face_pvel_y, matlIndex, patch, gac, 1 );
    new_dw->get( w, d_face_pvel_z, matlIndex, patch, gac, 1 );

    constCCVariable<double> phi;

    new_dw->get( phi, d_transportVarLabel, matlIndex, patch, gac, 2 );

    constSFCXVariable<double> af_x;
    constSFCYVariable<double> af_y;
    constSFCZVariable<double> af_z;

    old_dw->get( af_x, d_fieldLabels->d_areaFractionFXLabel, matlIndex, patch, gac, 2 );
    old_dw->get( af_y, d_fieldLabels->d_areaFractionFYLabel, matlIndex, patch, gac, 2 );
    old_dw->get( af_z, d_fieldLabels->d_areaFractionFZLabel, matlIndex, patch, gac, 2 );

    if ( d_which_limiter == UPWIND ){
      DQMOM_CONV(up,UpwindConvection);
    } else if ( d_which_limiter == SUPERBEE ){
      DQMOM_CONV(sb,SuperBeeConvection);
    } else if ( d_which_limiter == ROE ){
      DQMOM_CONV(roe,RoeConvection);
    } else if ( d_which_limiter == CENTRAL ){
      DQMOM_CONV(central,CentralConvection);
    } else if ( d_which_limiter == VANLEER ){
      DQMOM_CONV(vl,VanLeerConvection);
    }

  }
}


//---------------------------------------------------------------------------
// Method: Schedule compute the sources.
// Probably not needed for DQMOM EQN
//---------------------------------------------------------------------------
void
DQMOMEqn::sched_computeSources( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  // This scheduler only calls other schedulers
  SourceTermFactory& factory = SourceTermFactory::self();
  for (vector<std::string>::iterator iter = d_sources.begin(); iter != d_sources.end(); iter++){

    SourceTermBase& temp_src = factory.retrieve_source_term( *iter );

    temp_src.sched_computeSource( level, sched, timeSubStep );

  }
}

//---------------------------------------------------------------------------
// Method: Schedule build RHS
//---------------------------------------------------------------------------
void
DQMOMEqn::sched_buildRHS( const LevelP& level, SchedulerP& sched )
{

  string taskname = "DQMOMEqn::buildRHS";

  Task* tsk = scinew Task( taskname, this, &DQMOMEqn::buildRHS );

  tsk->modifies(d_RHSLabel);

  tsk->requires(Task::NewDW, d_X_flux_label, Ghost::AroundCells, 1);
  tsk->requires(Task::NewDW, d_Y_flux_label, Ghost::AroundCells, 1);
  tsk->requires(Task::NewDW, d_Z_flux_label, Ghost::AroundCells, 1);

  sched->addTask(tsk, level->eachPatch(), d_fieldLabels->d_materialManager->allMaterials( "Arches" ));

}
void
DQMOMEqn::buildRHS( const ProcessorGroup* pc,
                    const PatchSubset* patches,
                    const MaterialSubset* matls,
                    DataWarehouse* old_dw,
                    DataWarehouse* new_dw )
{
  //patch loop
  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_materialManager->getMaterial( "Arches", archIndex)->getDWIndex();
    Vector Dx = patch->dCell();

    Ghost::GhostType  gac = Ghost::AroundCells;

    constSFCXVariable<double> flux_x;
    constSFCYVariable<double> flux_y;
    constSFCZVariable<double> flux_z;

    new_dw->get( flux_x, d_X_flux_label, matlIndex, patch, gac, 1 );
    new_dw->get( flux_y, d_Y_flux_label, matlIndex, patch, gac, 1 );
    new_dw->get( flux_z, d_Z_flux_label, matlIndex, patch, gac, 1 );

    CCVariable<double> rhs;
    new_dw->getModifiable( rhs, d_RHSLabel, matlIndex, patch );

    Uintah::BlockRange range(patch->getCellLowIndex(), patch->getCellHighIndex());

    IntegrateFlux<CCVariable<double> > integrator( rhs, flux_x, flux_y, flux_z, Dx );
    Uintah::parallel_for( range, integrator );

  }
}

//---------------------------------------------------------------------------
// Method: Schedule build the transport equation.
//---------------------------------------------------------------------------
void
DQMOMEqn::sched_buildTransportEqn( const LevelP& level, SchedulerP& sched, const int timeSubStep )
{

  string taskname = "DQMOMEqn::buildTransportEqn";

  Task* tsk = scinew Task(taskname, this, &DQMOMEqn::buildTransportEqn, timeSubStep);

  Task::WhichDW which_dw;

  ArchesLabel::PartVelMap::iterator pvel_iter = d_fieldLabels->partVel.find(d_quadNode);
  if ( timeSubStep == 0 ){
    which_dw = Task::OldDW;
  } else {
    which_dw = Task::NewDW;
  }

  tsk->requires(which_dw, d_transportVarLabel, Ghost::AroundCells, 2);
  tsk->requires(Task::NewDW, pvel_iter->second, Ghost::AroundCells, 1);
  tsk->requires(Task::NewDW, d_face_pvel_x, Ghost::AroundCells, 1);
  tsk->requires(Task::NewDW, d_face_pvel_y, Ghost::AroundCells, 1);
  tsk->requires(Task::NewDW, d_face_pvel_z, Ghost::AroundCells, 1);

  tsk->modifies(d_X_flux_label);
  tsk->modifies(d_Y_flux_label);
  tsk->modifies(d_Z_flux_label);

  tsk->modifies(d_FdiffLabel);
  tsk->modifies(d_FconvLabel);
  tsk->modifies(d_RHSLabel);

  tsk->requires(Task::OldDW, d_fieldLabels->d_areaFractionLabel, Ghost::AroundCells, 2);
  tsk->requires(Task::OldDW, d_fieldLabels->d_areaFractionFXLabel, Ghost::AroundCells, 2);
  tsk->requires(Task::OldDW, d_fieldLabels->d_areaFractionFYLabel, Ghost::AroundCells, 2);
  tsk->requires(Task::OldDW, d_fieldLabels->d_areaFractionFZLabel, Ghost::AroundCells, 2);
  tsk->requires(Task::OldDW, d_fieldLabels->d_viscosityCTSLabel, Ghost::AroundCells, 1);

  tsk->requires(Task::NewDW, d_X_psi_label, Ghost::None, 0);
  tsk->requires(Task::NewDW, d_Y_psi_label, Ghost::None, 0);
  tsk->requires(Task::NewDW, d_Z_psi_label, Ghost::None, 0);

  if( !d_weight ) {
    tsk->requires(which_dw, d_weightLabel, Ghost::AroundCells, 0);
  }

  // extra srcs
  if (d_addExtraSources) {
    for ( auto iter = d_sources.begin(); iter != d_sources.end(); iter++){
      tsk->requires( Task::NewDW, VarLabel::find(*iter), Ghost::None, 0 );
    }
  }

  sched->addTask(tsk, level->eachPatch(), d_fieldLabels->d_materialManager->allMaterials( "Arches" ));

}
//---------------------------------------------------------------------------
// Method: Actually build the transport equation.
//---------------------------------------------------------------------------
void
DQMOMEqn::buildTransportEqn( const ProcessorGroup* pc,
                             const PatchSubset* patches,
                             const MaterialSubset* matls,
                             DataWarehouse* old_dw,
                             DataWarehouse* new_dw,
                             const int timeSubStep )
{

  //patch loop
  for (int p=0; p < patches->size(); p++){

    Ghost::GhostType  gac = Ghost::AroundCells;
    Ghost::GhostType  gn  = Ghost::None;

    const Patch* patch = patches->get(p);
    const Level* level = patch->getLevel();
    const int ilvl = level->getID();
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_materialManager->getMaterial( "Arches", archIndex)->getDWIndex();
    Vector Dx = patch->dCell();

    DataWarehouse* which_dw;
    if ( timeSubStep == 0 ){
      which_dw = old_dw;
    } else {
      which_dw = new_dw;
    }

    constCCVariable<double> phi;
    constCCVariable<double> mu_t;
    constCCVariable<double> extra_src; // Any additional source (eg, mms or unweighted abscissa src)
    constCCVariable<Vector> partVel;
    constCCVariable<Vector> areaFraction;

    CCVariable<double> Fdiff;
    CCVariable<double> Fconv;
    CCVariable<double> RHS;
    constSFCXVariable<double> af_x;
    constSFCYVariable<double> af_y;
    constSFCZVariable<double> af_z;
    constSFCXVariable<double> uu;
    constSFCYVariable<double> vv;
    constSFCZVariable<double> ww;

    ArchesLabel::PartVelMap::iterator pvel_iter = d_fieldLabels->partVel.find(d_quadNode);
    new_dw->get( partVel, pvel_iter->second, matlIndex, patch, gac, 1 );
    new_dw->get( uu, d_face_pvel_x, matlIndex, patch, gac, 1 );
    new_dw->get( vv, d_face_pvel_y, matlIndex, patch, gac, 1 );
    new_dw->get( ww, d_face_pvel_z, matlIndex, patch, gac, 1 );
    which_dw->get(phi, d_transportVarLabel, matlIndex, patch, gac, 2);

    old_dw->get(mu_t         ,  d_fieldLabels->d_viscosityCTSLabel  , matlIndex , patch , gac , 1);
    old_dw->get(areaFraction ,  d_fieldLabels->d_areaFractionLabel  , matlIndex , patch , gac , 2);
    old_dw->get(af_x, d_fieldLabels->d_areaFractionFXLabel, matlIndex, patch, gac, 2);
    old_dw->get(af_y, d_fieldLabels->d_areaFractionFYLabel, matlIndex, patch, gac, 2);
    old_dw->get(af_z, d_fieldLabels->d_areaFractionFZLabel, matlIndex, patch, gac, 2);

    double vol = Dx.x();
#ifdef YDIM
    vol *= Dx.y();
#endif
#ifdef ZDIM
    vol *= Dx.z();
#endif

    new_dw->getModifiable(Fdiff, d_FdiffLabel, matlIndex, patch);
    new_dw->getModifiable(Fconv, d_FconvLabel, matlIndex, patch);
    new_dw->getModifiable(RHS, d_RHSLabel, matlIndex, patch);
    RHS.initialize(0.0);
    Fdiff.initialize(0.0);
    Fconv.initialize(0.0);

    constCCVariable<double> w;
    if(!d_weight){
      which_dw->get(w, d_weightLabel, matlIndex, patch, gn, 0);
    }

    //----CONVECTION
    if ( d_doConv ){

      d_disc->computeConv( patch, Fconv, phi, partVel, areaFraction, d_convScheme );

      if ( _using_new_intrusion ) {
        _intrusions[ilvl]->addScalarRHS( patch, Dx, d_eqnName, RHS );
      }
    }

    //----DIFFUSION
    if (d_doDiff)
      d_disc->computeDiff( patch, Fdiff, phi, mu_t, d_mol_diff, areaFraction, d_turbPrNo );

    //----SUM UP RHS
    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){

      IntVector c = *iter;

      RHS[c] += Fdiff[c] - Fconv[c];

      if (d_addExtraSources) {

        // Get the factory of source terms
        for ( auto src_iter = d_sources.begin(); src_iter != d_sources.end(); src_iter++){
         //constCCVariable<double> extra_src;
         new_dw->get(extra_src, VarLabel::find( *src_iter ), matlIndex, patch, gn, 0);

         // Add to the RHS
         RHS[c] += extra_src[c]*vol;
        }
      }
    }
  }
}

void
DQMOMEqn::sched_addSources( const LevelP& level, SchedulerP& sched, const int timeSubStep ){

  string taskname = "DQMOMEqn::addSources";

  d_sourceLabel = VarLabel::find(d_sourceName);

  Task* tsk = scinew Task(taskname, this, &DQMOMEqn::addSources, timeSubStep);

  Task::WhichDW which_dw;
  if ( timeSubStep == 0 ){
    which_dw = Task::OldDW;
  } else {
    which_dw = Task::NewDW;
  }
  tsk->modifies( d_RHSLabel );
  tsk->requires( Task::NewDW , d_sourceLabel , Ghost::None , 0 );
  if( !d_weight ) {
    tsk->requires( which_dw, d_weightLabel, Ghost::AroundCells, 0 );
  }

  sched->addTask(tsk, level->eachPatch(), d_fieldLabels->d_materialManager->allMaterials( "Arches" ));

}

void
DQMOMEqn::addSources( const ProcessorGroup* pc,
                      const PatchSubset* patches,
                      const MaterialSubset* matls,
                      DataWarehouse* old_dw,
                      DataWarehouse* new_dw,
                      const int timeSubStep )
{
  //patch loop
  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_materialManager->getMaterial( "Arches", archIndex)->getDWIndex();
    Vector Dx = patch->dCell();
    double vol = Dx.x()* Dx.y()* Dx.z();

    DataWarehouse* which_dw;
    if ( timeSubStep == 0 ){
      which_dw = old_dw;
    } else {
      which_dw = new_dw;
    }

    constCCVariable<double> src;
    constCCVariable<double> weight;
    CCVariable<double> rhs;

    new_dw->get( src, d_sourceLabel, matlIndex, patch, Ghost::None, 0 );
    new_dw->getModifiable( rhs, d_RHSLabel, matlIndex, patch );
    if ( !d_weight ){
      which_dw->get( weight, d_weightLabel, matlIndex, patch, Ghost::None, 0 );
      for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){

        IntVector c = *iter;
        if ( weight[c] > d_w_small )
          rhs[c] += src[c]*vol;

      }
    } else {
      for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){

        IntVector c = *iter;

        rhs[c] += src[c]*vol;

      }
    }
  }
}

//---------------------------------------------------------------------------
// Method: Schedule solve the transport equation.
//---------------------------------------------------------------------------
void
DQMOMEqn::sched_solveTransportEqn( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  string taskname = "DQMOMEqn::solveTransportEqn";

  Task* tsk = scinew Task(taskname, this, &DQMOMEqn::solveTransportEqn, timeSubStep);

  //New
  tsk->modifies(d_transportVarLabel);
  tsk->requires(Task::NewDW, d_RHSLabel, Ghost::None, 0);
  if( !d_weight )
      tsk->requires(Task::NewDW, d_weightLabel, Ghost::None, 0);

  //Old
  tsk->requires(Task::OldDW, d_transportVarLabel, Ghost::None, 0);
  tsk->requires(Task::OldDW, d_fieldLabels->d_delTLabel, Ghost::None, 0 );
  tsk->requires(Task::OldDW, d_fieldLabels->d_volFractionLabel, Ghost::None, 0 );

  sched->addTask(tsk, level->eachPatch(), d_fieldLabels->d_materialManager->allMaterials( "Arches" ));
}

//---------------------------------------------------------------------------
// Method: Actually solve the transport equation.
//---------------------------------------------------------------------------
void
DQMOMEqn::solveTransportEqn( const ProcessorGroup* pc,
                             const PatchSubset* patches,
                             const MaterialSubset* matls,
                             DataWarehouse* old_dw,
                             DataWarehouse* new_dw,
                             int timeSubStep )
{
  //patch loop
  for (int p=0; p < patches->size(); p++){

    Ghost::GhostType  gn  = Ghost::None;

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_materialManager->getMaterial( "Arches", archIndex)->getDWIndex();

    delt_vartype DT;
    old_dw->get(DT, d_fieldLabels->d_delTLabel);
    double dt = DT;

    CCVariable<double> phi;    // phi @ current sub-level
    constCCVariable<double> RHS;
    constCCVariable<double> rk1_phi; // phi @ n for averaging
    constCCVariable<double> vol_fraction;

    new_dw->getModifiable(phi, d_transportVarLabel, matlIndex, patch);
    new_dw->get(RHS, d_RHSLabel, matlIndex, patch, gn, 0);
    old_dw->get(rk1_phi, d_transportVarLabel, matlIndex, patch, gn, 0);
    old_dw->get(vol_fraction, d_fieldLabels->d_volFractionLabel, matlIndex, patch, gn, 0 );

    d_timeIntegrator->singlePatchFEUpdate( patch, phi, RHS, dt, d_eqnName );

    if(d_weight){
        // weights being clipped inside this function call
        d_timeIntegrator->timeAvePhi( patch, phi, rk1_phi, timeSubStep,
            clip.tol, clip.do_low, clip.low, clip.do_high, clip.high, vol_fraction );
    } else {

        constCCVariable<double> w;
        new_dw->get(w, d_weightLabel, matlIndex, patch, gn, 0);
        // weighted abscissa being clipped inside this function call
        d_timeIntegrator->timeAvePhi( patch, phi, rk1_phi, timeSubStep,
            clip.tol, clip.do_low, clip.low, clip.do_high, clip.high, w, vol_fraction);
    }

    //----BOUNDARY CONDITIONS
    // For first time step, bc's have been set in dqmomInit
    computeBCs( patch, d_eqnName, phi );

  }
}

//---------------------------------------------------------------------------
// Method: Schedule the compute of the IC values
//---------------------------------------------------------------------------
void
DQMOMEqn::sched_getUnscaledValues( const LevelP& level, SchedulerP& sched )
{
  string taskname = "DQMOMEqn::getUnscaledValues";

  Task* tsk = scinew Task(taskname, this, &DQMOMEqn::getUnscaledValues);

  //NEW
  tsk->modifies(d_icLabel);
  tsk->requires( Task::NewDW,d_transportVarLabel , Ghost::None, 0 );

  if( !d_weight ) {
    tsk->requires( Task::NewDW,d_weightLabel , Ghost::None, 0 );
  }

  sched->addTask(tsk, level->eachPatch(), d_fieldLabels->d_materialManager->allMaterials( "Arches" ));

}
//---------------------------------------------------------------------------
// Method: Compute the IC vaues by dividing by the weights
//---------------------------------------------------------------------------
void
DQMOMEqn::getUnscaledValues( const ProcessorGroup* pc,
                             const PatchSubset* patches,
                             const MaterialSubset* matls,
                             DataWarehouse* old_dw,
                             DataWarehouse* new_dw )
{

  //patch loop
  for (int p=0; p < patches->size(); p++){

    //Ghost::GhostType  gn  = Ghost::None;

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_materialManager->getMaterial( "Arches", archIndex)->getDWIndex();

    if( d_weight ) {
      constCCVariable<double> w;
      CCVariable<double> w_actual;

      new_dw->getModifiable(w_actual, d_icLabel, matlIndex, patch);
      new_dw->get(w, d_transportVarLabel, matlIndex, patch, Ghost::None,0);

      // now loop over all cells
      for (CellIterator iter=patch->getCellIterator(0); !iter.done(); iter++){
        IntVector c = *iter;

        w_actual[c] = w[c]*d_scalingConstant[d_quadNode];
      }

    } else {

      CCVariable<double> ic;
      new_dw->getModifiable(ic, d_icLabel, matlIndex, patch);

      constCCVariable<double> wa;
      new_dw->get(wa, d_transportVarLabel, matlIndex, patch, Ghost::None,0);

      constCCVariable<double> w;
      new_dw->get(w, d_weightLabel, matlIndex, patch, Ghost::None,0);

      // now loop over all cells
      if ( d_unweighted ) {
        for (CellIterator iter=patch->getCellIterator(0); !iter.done(); iter++){

          IntVector c = *iter;

          ic[c] = wa[c]*d_scalingConstant[d_quadNode];
        }

        // Set BCS - account for using facecentered variable in cell center.
        // low weighted regions are set to zero on boundaries;
        vector<Patch::FaceType> bf;
        patch->getBoundaryFaces(bf);
        Patch::FaceIteratorType PEC = Patch::ExtraPlusEdgeCells;

        for( vector<Patch::FaceType>::const_iterator itr = bf.begin(); itr != bf.end(); ++itr ){
          Patch::FaceType face = *itr;

          IntVector insideCellDir = patch->faceDirection(face);
          for(CellIterator iter=patch->getFaceIterator(face, PEC); !iter.done(); iter++) {

            IntVector c=*iter;

            IntVector bp1(c - insideCellDir);

            if ( ( (w[c]+w[bp1]) / 2.0) > d_w_small){
              ic[c] = wa[c]*2.0*d_scalingConstant[d_quadNode]-ic[bp1];
            }else{
              ic[c] =-ic[bp1];
            }
          }
        }
      } else {
        for (CellIterator iter=patch->getCellIterator(0); !iter.done(); iter++){

          IntVector c = *iter;

          if (w[c] > d_w_small){
            ic[c] = (wa[c]/w[c])*d_scalingConstant[d_quadNode];
          }  else {
            ic[c] = d_nominal[d_quadNode];
          }
        }

        // Set BCS - account for using facecentered variable in cell center.
        // low weighted regions are set to zero on boundaries;
        vector<Patch::FaceType> bf;
        patch->getBoundaryFaces(bf);
        Patch::FaceIteratorType PEC = Patch::ExtraPlusEdgeCells;

        for( vector<Patch::FaceType>::const_iterator itr = bf.begin(); itr != bf.end(); ++itr ){
          Patch::FaceType face = *itr;

          IntVector insideCellDir = patch->faceDirection(face);
          for(CellIterator iter=patch->getFaceIterator(face, PEC); !iter.done(); iter++) {

            IntVector c=*iter;

            IntVector bp1(c - insideCellDir);

            if ( ( (w[c]+w[bp1]) / 2.0) > d_w_small){
              ic[c] = ((wa[c]+wa[bp1])/(w[c]+w[bp1]))*d_scalingConstant[d_quadNode]*2.0-ic[bp1];
            }else{
              ic[c] =-ic[bp1];
            }
          }
        }
      }
    } //end if weight
  }
}
void
DQMOMEqn::sched_advClipping( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
}
