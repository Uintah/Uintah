#include <CCA/Components/Arches/TransportEqns/DQMOMEqn.h>
#include <CCA/Components/Arches/TransportEqns/EqnFactory.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermBase.h>
#include <CCA/Components/Arches/CoalModels/CoalModelFactory.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermFactory.h>
#include <CCA/Components/Arches/Directives.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Parallel/Parallel.h>

using namespace std;
using namespace Uintah;

//---------------------------------------------------------------------------
// Builder:
DQMOMEqnBuilder::DQMOMEqnBuilder( ArchesLabel* fieldLabels, 
                                  ExplicitTimeInt* timeIntegrator,
                                  string eqnName,
                                  int quadNode,
                                  bool isWeight ):
                                  EqnBuilder( fieldLabels, timeIntegrator, eqnName),
                                  d_fieldLabels(fieldLabels),
                                  d_timeIntegrator(timeIntegrator),
                                  d_eqnName(eqnName),
                                  d_quadNode(quadNode),
                                  d_weight(isWeight) {} 

DQMOMEqnBuilder::~DQMOMEqnBuilder(){}
    
EqnBase* DQMOMEqnBuilder::build() {
  return scinew DQMOMEqn(d_fieldLabels, d_timeIntegrator, d_eqnName, d_quadNode, d_weight);
};
// End Builder
//---------------------------------------------------------------------------

DQMOMEqn::DQMOMEqn( ArchesLabel* fieldLabels, ExplicitTimeInt* timeIntegrator, string eqnName, int quadNode, bool isWeight )
: 
EqnBase( fieldLabels, timeIntegrator, eqnName ), d_quadNode(quadNode), d_weight(isWeight)
{
  
  string varname = eqnName+"_Fdiff"; 
  d_FdiffLabel = VarLabel::create(varname, 
            CCVariable<double>::getTypeDescription());
  varname = eqnName+"_Fconv"; 
  d_FconvLabel = VarLabel::create(varname, 
            CCVariable<double>::getTypeDescription());
  varname = eqnName+"_RHS";
  d_RHSLabel = VarLabel::create(varname, 
            CCVariable<double>::getTypeDescription());
  varname = eqnName+"_old";
  d_oldtransportVarLabel = VarLabel::create(varname,
            CCVariable<double>::getTypeDescription());
  varname = eqnName;
  d_transportVarLabel = VarLabel::create(varname,
            CCVariable<double>::getTypeDescription());
  varname = eqnName+"_icv"; // icv = internal coordinate value (this is the unscaled/unweighted value)
  d_icLabel = VarLabel::create(varname, 
            CCVariable<double>::getTypeDescription());
  varname = eqnName+"_src";
  d_sourceLabel = VarLabel::create(varname, 
            CCVariable<double>::getTypeDescription());
}

DQMOMEqn::~DQMOMEqn()
{
  VarLabel::destroy(d_FdiffLabel);
  VarLabel::destroy(d_FconvLabel); 
  VarLabel::destroy(d_RHSLabel);    
  VarLabel::destroy(d_sourceLabel); 
  VarLabel::destroy(d_transportVarLabel);
  VarLabel::destroy(d_oldtransportVarLabel);
  VarLabel::destroy(d_icLabel); 
}
//---------------------------------------------------------------------------
// Method: Problem Setup 
//---------------------------------------------------------------------------
void
DQMOMEqn::problemSetup(const ProblemSpecP& inputdb )
{
  // NOTE: some of this may be better off in the EqnBase.cc class
  ProblemSpecP db = inputdb; 

  db->getWithDefault("turbulentPrandtlNumber",d_turbPrNo,0.4);

  // Discretization information:
  db->getWithDefault( "conv_scheme", d_convScheme, "upwind");
  db->getWithDefault( "doConv", d_doConv, false);
  db->getWithDefault( "doDiff", d_doDiff, false);
  db->getWithDefault( "timestepMultiplier", d_timestepMultiplier, 1.0);
  d_addSources = true; 
  d_addExtraSources = false; 

  // Clipping:
  d_doClipping = false; 
  ProblemSpecP db_clipping = db->findBlock("Clipping");

  if (db_clipping) {
    //This seems like a *safe* number to assume 
    double clip_default = -9999999999.0;
    d_doLowClip = false; 
    d_doHighClip = false; 
    d_doClipping = true;
    
    db_clipping->getWithDefault("low", d_lowClip,  clip_default);
    db_clipping->getWithDefault("high",d_highClip, clip_default);

    if( d_weight ) {
      db_clipping->getWithDefault("small", d_smallClip, TINY);
    }

    if ( d_lowClip != clip_default ) 
      d_doLowClip = true; 

    if ( d_highClip != clip_default ) 
      d_doHighClip = true; 

    if ( !d_doHighClip && !d_doLowClip ) 
      throw ProblemSetupException("A low or high clipping must be specified if the <Clipping> section is activated!", __FILE__, __LINE__);
  } 

  if (d_weight) { 
    if (!d_doClipping) { 
      //By default, set the low value for this weight to 0 and run on low clipping
      d_lowClip = 0;
      d_doClipping = true; 
      d_doLowClip = true;  
      d_smallClip = TINY;
    } else { 
      if (!d_doLowClip){ 
        //weights always have low clip values!  ie, negative weights not allowed
        d_lowClip = 0;
        d_doLowClip = true; 
        d_smallClip = TINY;
      } 
    }
  } 

  // Scaling information:
  db->getWithDefault( "scaling_const", d_scalingConstant, 1.0); 


  // Extra Source terms (for mms and other tests):
  SourceTermFactory& srcFactory = SourceTermFactory::self();
  if (db->findBlock("src")){
    string srcname; 
    d_addExtraSources = true; 
    for (ProblemSpecP src_db = db->findBlock("src"); src_db != 0; src_db = src_db->findNextBlock("src")){
      src_db->getAttribute("label", srcname);
      //which sources are turned on for this equation
      d_sources.push_back( srcFactory.retrieve_source_term(srcname).getSrcLabel() ); 

    }
  }


  // Initialization:
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
          d_constant_init /= d_scalingConstant; 
        }

    // -------- Environment constant initialization function --------------
    } else if (d_initFunction == "env_constant" ) {
      
      if ( !db_initialValue->findBlock("env_constant") ) {
        string err_msg = "ERROR: Arches: DQMOMEqn: Could not initialize equation "+d_eqnName+": You specified an 'env_constant' initialization function but did not include any 'env_constant' tags! \n";
        throw ProblemSetupException(err_msg,__FILE__,__LINE__);
      }

      // Environment constant: get the value of the constants
      for( ProblemSpecP db_env_constants = db_initialValue->findBlock("env_constant");
           db_env_constants != 0; db_env_constants = db_env_constants->findNextBlock("env_constant") ) {
        
        string s_tempQuadNode;
        db_env_constants->getAttribute("qn", s_tempQuadNode);
        int i_tempQuadNode = atoi( s_tempQuadNode.c_str() );

        string s_constant;
        db_env_constants->getAttribute("value", s_constant);
        double constantValue = atof( s_constant.c_str() );
        if( i_tempQuadNode == d_quadNode ) {
          d_constant_init = constantValue / d_scalingConstant;
        }
      }
 

    // ------- (Environment & Uniform) Step initialization function ------------
    // NOTE: Right now the only environment-specific attribute of the step function
    //       is the value.  This can be changed later, if someone wants to have
    //       unique step function directions/start/stop for each environment.
    // - Charles
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
        d_step_value /= d_scalingConstant; 
      
      } else if (d_initFunction == "env_step") {
        
        if( !(db_initialValue->findBlock("env_step_value")) ) {
          string err_msg = "ERROR: Arches: DQMOMEqn: Could not initialize 'evn_step' for equation "+d_eqnName+": You did not specify any <env_step_value>! \n";
          throw ProblemSetupException(err_msg,__FILE__,__LINE__);
        }
        for( ProblemSpecP db_env_step_value = db_initialValue->findBlock("env_step_value");
             db_env_step_value != 0; db_env_step_value = db_env_step_value->findNextBlock("env_step_value") ) {
          
          string s_tempQuadNode;
          db_env_step_value->getAttribute("qn", s_tempQuadNode);
          int i_tempQuadNode = atoi( s_tempQuadNode.c_str() );
        
          string s_step_value;
          db_env_step_value->getAttribute("value", s_step_value);
          double step_value = atof( s_step_value.c_str() );
          if( i_tempQuadNode == d_quadNode ) 
            d_step_value = step_value / d_scalingConstant;
        }
      }//end step_value init.

    } else if (d_initFunction == "mms1") {
      //currently nothing to do here. 

    // ------------ Other initialization function --------------------
    }
  }
}
//---------------------------------------------------------------------------
// Method: Schedule clean up. 
// Probably not needed for DQMOM
//---------------------------------------------------------------------------
void 
DQMOMEqn::sched_cleanUp( const LevelP& level, SchedulerP& sched )
{
  /*
  string taskname = "DQMOMEqn::cleanUp";
  Task* tsk = scinew Task(taskname, this, &DQMOMEqn::cleanUp);

  sched->addTask(tsk, level->eachPatch(), d_fieldLabels->d_sharedState->allArchesMaterials());
  */
}

//---------------------------------------------------------------------------
// Method: Actually clean up. 
//---------------------------------------------------------------------------
void DQMOMEqn::cleanUp( const ProcessorGroup* pc, 
                        const PatchSubset* patches, 
                        const MaterialSubset* matls, 
                        DataWarehouse* old_dw, 
                        DataWarehouse* new_dw )
{}

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
  tsk->computes(d_oldtransportVarLabel); // for rk sub stepping 
  tsk->computes(d_icLabel); 
  tsk->computes(d_RHSLabel); 
  tsk->computes(d_FconvLabel);
  tsk->computes(d_FdiffLabel);

  //Old
  tsk->requires(Task::OldDW, d_transportVarLabel, gn, 0);
  sched->addTask(tsk, level->eachPatch(), d_fieldLabels->d_sharedState->allArchesMaterials());
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
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    CCVariable<double> newVar;
    CCVariable<double> rkoldVar; 
    CCVariable<double> icValue; 
    constCCVariable<double> oldVar; 
    new_dw->allocateAndPut( newVar, d_transportVarLabel, matlIndex, patch );
    new_dw->allocateAndPut( rkoldVar, d_oldtransportVarLabel, matlIndex, patch ); 
    new_dw->allocateAndPut( icValue , d_icLabel, matlIndex, patch );
    old_dw->get(oldVar, d_transportVarLabel, matlIndex, patch, gn, 0);

    newVar.initialize(  0.0);
    rkoldVar.initialize(0.0);
    icValue.initialize( 0.0);

    // copy old into new
    newVar.copyData(oldVar);
    rkoldVar.copyData(oldVar); 

    CCVariable<double> Fdiff; 
    CCVariable<double> Fconv; 
    CCVariable<double> RHS; 

    new_dw->allocateAndPut( Fdiff, d_FdiffLabel, matlIndex, patch );
    new_dw->allocateAndPut( Fconv, d_FconvLabel, matlIndex, patch );
    new_dw->allocateAndPut( RHS, d_RHSLabel, matlIndex, patch ); 
    
    Fdiff.initialize(0.0);
    Fconv.initialize(0.0);
    RHS.initialize(0.0);
    
    curr_time = d_fieldLabels->d_sharedState->getElapsedTime(); 
    curr_ssp_time = curr_time; 
  }
}

#ifdef VERIFY_DQMOM_TRANSPORT
//---------------------------------------------------------------------------
// Method: Schedule compute the sources. 
// Only used for verification of the DQMOM transport equations
//--------------------------------------------------------------------------- 
void 
DQMOMEqn::sched_computeSources( const LevelP& level, 
                                SchedulerP& sched, 
                                int timeSubStep )
{
  // This scheduler only calls other schedulers
  SourceTermFactory& factory = SourceTermFactory::self(); 
  for (vector<std::string>::iterator iter = d_sources.begin(); iter != d_sources.end(); iter++){
    SourceTermBase& temp_src = factory.retrieve_source_term( *iter ); 
    temp_src.sched_computeSource( level, sched, timeSubStep ); 
  }
}
#endif

//---------------------------------------------------------------------------
// Method: Schedule build the transport equation. 
//---------------------------------------------------------------------------
void
DQMOMEqn::sched_buildTransportEqn( const LevelP& level, 
                                   SchedulerP& sched, 
                                   int timeSubStep )
{
  string taskname = "DQMOMEqn::buildTransportEqn"; 

  Task* tsk = scinew Task(taskname, this, &DQMOMEqn::buildTransportEqn);
  
  CoalModelFactory& coalFactory = CoalModelFactory::self();

  //----NEW----
  tsk->modifies(d_transportVarLabel);
  tsk->requires(Task::NewDW, d_oldtransportVarLabel, Ghost::AroundCells, 2);
  tsk->modifies(d_FdiffLabel);
  tsk->modifies(d_FconvLabel);
  tsk->modifies(d_RHSLabel);
  if( coalFactory.useParticleVelocityModel() ) {
    tsk->requires(Task::NewDW, coalFactory.getParticleVelocityLabel(d_quadNode), Ghost::AroundCells, 1);
  } else {
    tsk->requires(Task::OldDW, d_fieldLabels->d_newCCVelocityLabel, Ghost::AroundCells, 1);
  }
 
  //-----OLD-----
  tsk->requires(Task::OldDW, d_fieldLabels->d_areaFractionLabel, Ghost::AroundCells, 1); 
  tsk->requires(Task::OldDW, d_transportVarLabel, Ghost::AroundCells, 2);
  tsk->requires(Task::OldDW, d_fieldLabels->d_viscosityCTSLabel, Ghost::AroundCells, 1);
  tsk->requires(Task::OldDW, d_fieldLabels->d_uVelocitySPBCLabel, Ghost::AroundCells, 1);   

#ifdef YDIM
  tsk->requires(Task::OldDW, d_fieldLabels->d_vVelocitySPBCLabel, Ghost::AroundCells, 1); 
#endif
#ifdef ZDIM
  tsk->requires(Task::OldDW, d_fieldLabels->d_wVelocitySPBCLabel, Ghost::AroundCells, 1); 
#endif

#ifdef VERIFY_DQMOM_TRANSPORT
  // extra srcs
  if (d_addExtraSources) {
    for (vector<std::string>::iterator iter = d_sources.begin(); iter != d_sources.end(); iter++){
      tsk->requires( Task::NewDW, (*iter), Ghost::None, 0 ); 
    }
  }
#endif

  tsk->requires(Task::NewDW, d_sourceLabel, Ghost::None, 0 );

  sched->addTask(tsk, level->eachPatch(), d_fieldLabels->d_sharedState->allArchesMaterials());
}
//---------------------------------------------------------------------------
// Method: Actually build the transport equation. 
//---------------------------------------------------------------------------
void 
DQMOMEqn::buildTransportEqn( const ProcessorGroup* pc, 
                             const PatchSubset* patches, 
                             const MaterialSubset* matls, 
                             DataWarehouse* old_dw, 
                             DataWarehouse* new_dw )
{
  //patch loop
  for (int p=0; p < patches->size(); p++){

    Ghost::GhostType  gac = Ghost::AroundCells;
    Ghost::GhostType  gn  = Ghost::None;

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    Vector Dx = patch->dCell(); 

    constCCVariable<double> oldPhi;
    constCCVariable<double> mu_t;
    constSFCXVariable<double> uVel; 
    constSFCYVariable<double> vVel; 
    constSFCZVariable<double> wVel; 
    constCCVariable<double> src; 
    constCCVariable<Vector> partVel; 
    constCCVariable<Vector> areaFraction; 

    CCVariable<double> phi;
    CCVariable<double> Fdiff; 
    CCVariable<double> Fconv; 
    CCVariable<double> RHS; 

    new_dw->get(oldPhi, d_oldtransportVarLabel, matlIndex, patch, gac, 2);

    new_dw->get(src, d_sourceLabel, matlIndex, patch, gn, 0 );

    old_dw->get(mu_t, d_fieldLabels->d_viscosityCTSLabel, matlIndex, patch, gac, 1); 
    old_dw->get(uVel,   d_fieldLabels->d_uVelocitySPBCLabel, matlIndex, patch, gac, 1); 
    old_dw->get(areaFraction, d_fieldLabels->d_areaFractionLabel, matlIndex, patch, gac, 1); 
    double vol = Dx.x();
#ifdef YDIM
    old_dw->get(vVel,   d_fieldLabels->d_vVelocitySPBCLabel, matlIndex, patch, gac, 1); 
    vol *= Dx.y(); 
#endif
#ifdef ZDIM
    old_dw->get(wVel,   d_fieldLabels->d_wVelocitySPBCLabel, matlIndex, patch, gac, 1); 
    vol *= Dx.z(); 
#endif

    CoalModelFactory& coalFactory = CoalModelFactory::self();

    if( coalFactory.useParticleVelocityModel() ) {
      new_dw->get( partVel, coalFactory.getParticleVelocityLabel(d_quadNode), matlIndex, patch, gac, 1 );
    } else {
      old_dw->get( partVel, d_fieldLabels->d_newCCVelocityLabel, matlIndex, patch, gac, 1 );
    }
    new_dw->getModifiable(phi, d_transportVarLabel, matlIndex, patch);
    new_dw->getModifiable(Fdiff, d_FdiffLabel, matlIndex, patch);
    new_dw->getModifiable(Fconv, d_FconvLabel, matlIndex, patch); 
    new_dw->getModifiable(RHS, d_RHSLabel, matlIndex, patch);
    RHS.initialize(0.0); 
    Fconv.initialize(0.0);
    Fdiff.initialize(0.0); 

    //----BOUNDARY CONDITIONS
    // For first time step, bc's have been set in dqmomInit
    computeBCs( patch, d_eqnName, phi );

    //----CONVECTION
    if (d_doConv) {
      d_disc->computeConv( patch, Fconv, oldPhi, uVel, vVel, wVel, partVel, areaFraction, d_convScheme ); 
    }
  
    //----DIFFUSION
    if (d_doDiff) {
      d_disc->computeDiff( patch, Fdiff, oldPhi, mu_t, areaFraction, d_turbPrNo, matlIndex, d_eqnName );
    }
 
    //----SUM UP RHS
    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
      IntVector c = *iter; 

      RHS[c] += Fdiff[c] - Fconv[c];

      if (d_addSources) {

        RHS[c] += src[c]*vol;           

#ifdef VERIFY_DQMOM_TRANSPORT
        if (d_addExtraSources) { 
          // Going to subtract out the src from the Ax=b solver
          // This assumes that you don't care about the solution (for verification). 
          RHS[c] -= src[c]*vol;

          // Get the factory of source terms
          for (vector<std::string>::iterator src_iter = d_sources.begin(); src_iter != d_sources.end(); src_iter++) {
            new_dw->get(extra_src, (*src_iter), matlIndex, patch, gn, 0);

           // Add to the RHS
           RHS[c] += extra_src[c]*vol; 
          }            
        }
#endif

      }
    }//end cells

#ifdef DEBUG_MODELS
    proc0cout << "Particle velocity = " << partVel[IntVector(1,2,3)] << endl;
    proc0cout << "Gas velocity = [" << uVel[IntVector(1,2,3)] << ", " << vVel[IntVector(1,2,3)] << ", " << wVel[IntVector(1,2,3)] << "]" << endl;
    proc0cout << "RHS = Fconv + Fdiff + src*vol = " << Fconv[IntVector(1,2,3)] << " + " << Fdiff[IntVector(1,2,3)] << " + " << src[IntVector(1,2,3)] << "*" << vol << " = " << RHS[IntVector(1,2,3)] << endl;
#endif

  }//end patches
}


//---------------------------------------------------------------------------
// Method: Schedule solve the transport equation. 
//---------------------------------------------------------------------------
void
DQMOMEqn::sched_solveTransportEqn( const LevelP& level, 
                                   SchedulerP& sched, 
                                   int timeSubStep, 
                                   bool copyOldIntoNew )
{
  string taskname = "DQMOMEqn::solveTransportEqn";

  Task* tsk = scinew Task(taskname, this, &DQMOMEqn::solveTransportEqn, timeSubStep, copyOldIntoNew);

  //New
  tsk->modifies(d_transportVarLabel);
  tsk->modifies(d_oldtransportVarLabel);  
  tsk->requires(Task::NewDW, d_RHSLabel, Ghost::None, 0);

  //Old
  tsk->requires(Task::OldDW, d_transportVarLabel, Ghost::None, 0);
  tsk->requires(Task::OldDW, d_fieldLabels->d_sharedState->get_delt_label(), Ghost::None, 0 );

  //cmr
  //tsk->requires( Task::OldDW, d_fieldLabels->d_MinTimestepLabel );
  //tsk->requires( Task::NewDW, d_fieldLabels->d_MinTimestepLabel );
  //tsk->modifies( d_fieldLabels->d_MinTimestepLabel );

  tsk->requires(Task::OldDW, d_fieldLabels->d_sharedState->get_delt_label(), level.get_rep());

  sched->addTask(tsk, level->eachPatch(), d_fieldLabels->d_sharedState->allArchesMaterials());
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
                             int timeSubStep,
                             bool copyOldIntoNew )
{
  //cmr
  /*
  min_vartype MinTimestep;
  //old_dw->get( MinTimestep, d_fieldLabels->d_MinTimestepLabel );
  new_dw->get( MinTimestep, d_fieldLabels->d_MinTimestepLabel );
  double current_min_delta_t = MinTimestep;
  new_dw->put( min_vartype(current_min_delta_t+1.0), d_fieldLabels->d_MinTimestepLabel );
  //new_dw->put( min_vartype(current_min_delta_t+1.0), d_fieldLabels->d_MinTimestepLabel, getLevel(patches) );
  */

  //patch loop
  for (int p=0; p < patches->size(); p++){

    Ghost::GhostType  gn  = Ghost::None;

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    delt_vartype DT;
    old_dw->get(DT, d_fieldLabels->d_sharedState->get_delt_label());
    double dt = DT; 

    //delt_vartype CurrTimestep;
    //old_dw->get( CurrTimestep, d_fieldLabels->d_sharedState->get_delt_label());
    //double delta_t = CurrTimestep;

    // Here, j is the rk step and n is the time step.  
    CCVariable<double> phi_at_jp1;   // phi^{(j+1)}
    CCVariable<double> phi_at_j;     // phi^{(j)}
    constCCVariable<double> rk1_phi; // phi^{n}
    constCCVariable<double> RHS; 

    new_dw->getModifiable(phi_at_jp1, d_transportVarLabel, matlIndex, patch);
    new_dw->getModifiable(phi_at_j,   d_oldtransportVarLabel, matlIndex, patch); 
    old_dw->get(rk1_phi, d_transportVarLabel, matlIndex, patch, gn, 0);
    new_dw->get(RHS, d_RHSLabel, matlIndex, patch, gn, 0);

#ifdef DEBUG_MODELS
    proc0cout << "Eqn " << d_eqnName << ": forward euler update" << endl;
    proc0cout << "Before: phi = " << phi_at_jp1[IntVector(1,2,3)] << endl;
    Vector dx=patch->dCell();
    double vol= dx.x()*dx.y()*dx.z();
    proc0cout << "RHS*(dt/vol) = " << RHS[IntVector(1,2,3)] << "*(" << dt << "/" << vol << ") = " << (RHS[IntVector(1,2,3)])*(dt/vol) << endl;
#endif

    // update to get phi^{(j+1)}
    d_timeIntegrator->singlePatchFEUpdate( patch, phi_at_jp1, RHS, dt, curr_ssp_time, d_eqnName );

#ifdef DEBUG_MODELS
    proc0cout << "After: phi = " << phi_at_jp1[IntVector(1,2,3)] << endl;
    proc0cout << endl;
#endif
    
    // Compute the current RK time
    double factor = d_timeIntegrator->time_factor[timeSubStep]; 
    curr_ssp_time = curr_time + factor * dt; 

    /// For the RK Averaging procedure, computing the time averaged phi^{time averaged}.
    /// Here, for convenience we assign the time averaged phi to phi_at_jp1, so:
    /// phi^{j+1} = alpha*(phi^{n}) + beta*(phi^{j+1})
    /// @seealso 
    /// Sigal Gottlieb, Chi-Wang Shu and Eitan Tadmor
    /// SIAM Review, Vol. 43, No. 1 (Mar., 2001), pp. 89-112 
    d_timeIntegrator->timeAvePhi( patch, phi_at_jp1, rk1_phi, timeSubStep, curr_ssp_time ); 

    //----BOUNDARY CONDITIONS
    // For first time step, bc's have been set in dqmomInit
    computeBCs( patch, d_eqnName, phi_at_jp1 );
    
    // ---------------------------------
    // Stable timestep calculation

    // copy averaged phi into oldphi, unless it's the last time substep (in which case we need info for time substep jp1 and j)
    if( copyOldIntoNew ) {
      // This is NOT the last time substep
      phi_at_j.copyData(phi_at_jp1); 

    } else {
      // this IS the last time sub-step

      // The calculation procedure looks something like this:
      // 1. Compute the error between the last Runge Kutta time substeps
      // 2. Use this error to estimate a new minimum timestep, and store that minimum timestep
      
      double new_min_delta_t = 1e16;
      for( CellIterator iter = patch->getCellIterator(); !iter.done(); ++iter ) {
        IntVector c = *iter;

        // Step 1: Compute error
        // error = ( phi^{(1)}-phi^{n} )/delta_t - RHS(\phi^{(1)})
        double error = (phi_at_j[c] - rk1_phi[c])/dt - RHS[c];
        double deltat = fabs( phi_at_jp1[c]/(error+TINY) );

        // Step 2: Estimate new min. timestep
        // min_delta_t_stable = phi^{j+1} / error [=] phi/(phi/time) [=] time
        if( fabs(error) > TINY ) {
          new_min_delta_t = min( deltat, new_min_delta_t);
        }

        /*
        //cmr
        if( new_min_delta_t < 0 ) {
          cout << "Equation " << d_eqnName << ": " << endl;
          cout << "Error = [ phi^(j) - phi^(n) ]/dt - RHS  =  [ ";
          cout << phi_at_j[c] << " - " << rk1_phi[c] << " ] / " << dt;
          cout << " - " << RHS[c];
          cout << " = " << error << endl;

          cout << "Delta_t = phi^(j+1) / error = " << phi_at_jp1[c] << "/" << error;
          cout << " = " << deltat << endl;

          int a = 0; ++a;
        }
        */

      }//end cells
      
      new_min_delta_t *= d_timestepMultiplier;
      DQMOMEqnFactory& dqmomFactory  = DQMOMEqnFactory::self(); 
      /*
      //cmr
      cout << "Hi from equation " << d_eqnName << ", about to set minimum timestep var to " << new_min_delta_t << endl;
      */
      dqmomFactory.setMinTimestepVar( d_eqnName, new_min_delta_t );

    }

  }//end patches

}


//---------------------------------------------------------------------------
// Method: Schedule the compute of the IC values
//---------------------------------------------------------------------------
void
DQMOMEqn::sched_getUnscaledValues( const LevelP& level, SchedulerP& sched )
{
  string taskname = "DQMOMEqn::getUnscaledValues"; 

  Task* tsk = scinew Task(taskname, this, &DQMOMEqn::getUnscaledValues);
  
  Ghost::GhostType  gn  = Ghost::None;
  DQMOMEqnFactory& dqmomFactory  = DQMOMEqnFactory::self(); 

  //NEW
  tsk->modifies(d_icLabel);
  tsk->modifies(d_transportVarLabel); 

  if( !d_weight ) {
    string name = "w_qn"; 
    string node; 
    std::stringstream out; 
    out << d_quadNode; 
    node = out.str(); 
    name += node; 

    EqnBase& eqn = dqmomFactory.retrieve_scalar_eqn( name ); 
    
    tsk->requires( Task::NewDW, eqn.getTransportEqnLabel(), gn, 0 ); 
  }
 
  sched->addTask(tsk, level->eachPatch(), d_fieldLabels->d_sharedState->allArchesMaterials());

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

    Ghost::GhostType  gn  = Ghost::None;

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 


    if( d_weight ) {
      CCVariable<double> w;
      CCVariable<double> w_actual;

      new_dw->getModifiable(w_actual, d_icLabel, matlIndex, patch);
      new_dw->getModifiable(w, d_transportVarLabel, matlIndex, patch);
      
      // now loop over all cells
      for (CellIterator iter=patch->getCellIterator(0); !iter.done(); iter++){
        IntVector c = *iter;
        w_actual[c] = w[c]*d_scalingConstant;
      }

    } else {
      constCCVariable<double> w;  
      constCCVariable<double> wa;
      CCVariable<double> ic; 

      new_dw->getModifiable(ic, d_icLabel, matlIndex, patch);
      new_dw->get(wa, d_transportVarLabel, matlIndex, patch, gn, 0 ); 

      DQMOMEqnFactory& dqmomFactory  = DQMOMEqnFactory::self(); 
      string name = "w_qn"; 
      string node;
      std::stringstream out;
      out << d_quadNode;
      node = out.str();
      name += node;
      EqnBase& temp_eqn = dqmomFactory.retrieve_scalar_eqn( name );
      DQMOMEqn& eqn = dynamic_cast<DQMOMEqn&>(temp_eqn);

      const VarLabel* mywLabel = eqn.getTransportEqnLabel();  
      double smallWeight = eqn.getSmallClip(); 

      if ( smallWeight == 0.0 ) {
        smallWeight = TINY;
      }

      new_dw->get(w, mywLabel, matlIndex, patch, gn, 0 ); 

      // now loop over all cells
      for (CellIterator iter=patch->getCellIterator(0); !iter.done(); iter++){
  
        IntVector c = *iter;

        if (w[c] > smallWeight)
          ic[c] = (wa[c]/w[c])*d_scalingConstant;
        else {
          ic[c] = 0.0;
        }
      }

    } //end if weight

  }
}
//---------------------------------------------------------------------------
// Method: Compute the boundary conditions. 
//---------------------------------------------------------------------------
// -- See header file. 


//---------------------------------------------------------------------------
// Method: schedule clipping
//---------------------------------------------------------------------------
void DQMOMEqn::sched_clipPhi( const LevelP& level,
                              SchedulerP& sched )
{
  string taskname = "DQMOMEqn::clipPhi"; 

  Task* tsk = scinew Task(taskname, this, &DQMOMEqn::clipPhi);

  tsk->modifies(d_transportVarLabel);

  tsk->requires(Task::NewDW, d_oldtransportVarLabel, Ghost::None, 0);

  sched->addTask(tsk, level->eachPatch(), d_fieldLabels->d_sharedState->allArchesMaterials());
}

void DQMOMEqn::clipPhi( const ProcessorGroup* pc,
                        const PatchSubset* patches,
                        const MaterialSubset* matls,
                        DataWarehouse* old_dw,
                        DataWarehouse* new_dw )
{

  for (int p=0; p < patches->size(); p++){

    Ghost::GhostType  gn  = Ghost::None;

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    CCVariable<double> phi;
    new_dw->getModifiable( phi, d_transportVarLabel, matlIndex, patch );

    constCCVariable<double> old_phi;
    new_dw->get( old_phi, d_oldtransportVarLabel, matlIndex, patch, gn, 0 );

    if( d_doLowClip ) {
      for( CellIterator iter = patch->getCellIterator(); !iter.done(); ++iter ) {
        IntVector c = *iter;
        
        if( phi[*iter] < d_lowClip+TINY || old_phi[*iter] < d_lowClip+TINY ) {
          phi[*iter] = d_lowClip;
        }
      }//end cells
    }//end if low clip

    if( d_doHighClip ) {
      for( CellIterator iter = patch->getCellIterator(); !iter.done(); ++iter ) {
        if( phi[*iter] > d_highClip || old_phi[*iter] > d_highClip ) {
          phi[*iter] = d_highClip;
        }
      }//end cells
    }//end if high clip

  }//end patches

}


//---------------------------------------------------------------------------
// Method: Schedule dummy initialization
//---------------------------------------------------------------------------
void
DQMOMEqn::sched_dummyInit( const LevelP& level, SchedulerP& sched )
{
  string taskname = "DQMOMEqn::dummyInit"; 

  Task* tsk = scinew Task(taskname, this, &DQMOMEqn::dummyInit);

  Ghost::GhostType  gn = Ghost::None;

  tsk->requires(Task::OldDW, d_transportVarLabel, gn, 0);
  tsk->requires(Task::OldDW, d_FconvLabel, gn, 0);
  tsk->requires(Task::OldDW, d_FdiffLabel, gn, 0);
  tsk->requires(Task::OldDW, d_RHSLabel, gn, 0);
  tsk->requires(Task::OldDW, d_sourceLabel, gn, 0);
  tsk->requires(Task::OldDW, d_sourceLabel,       gn, 0);

  tsk->computes(d_transportVarLabel);
  tsk->computes(d_oldtransportVarLabel); 
  tsk->computes(d_icLabel); 
  tsk->computes(d_FconvLabel);
  tsk->computes(d_FdiffLabel);
  tsk->computes(d_RHSLabel);
  tsk->computes(d_sourceLabel); 

  sched->addTask(tsk, level->eachPatch(), d_fieldLabels->d_sharedState->allArchesMaterials());
}

void 
DQMOMEqn::dummyInit( const ProcessorGroup* pc, 
                     const PatchSubset* patches, 
                     const MaterialSubset* matls, 
                     DataWarehouse* old_dw, 
                     DataWarehouse* new_dw )
{
  //patch loop
  for (int p=0; p < patches->size(); p++){

    Ghost::GhostType  gn = Ghost::None;
    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    CCVariable<double> phi; 
    CCVariable<double> old_phi;
    CCVariable<double> ic; 
    CCVariable<double> src; 
    CCVariable<double> fconv;
    CCVariable<double> fdiff;
    CCVariable<double> rhs;
    constCCVariable<double> phi_oldDW; 

    new_dw->allocateAndPut( phi, d_transportVarLabel, matlIndex, patch ); 
    new_dw->allocateAndPut( old_phi, d_oldtransportVarLabel, matlIndex, patch ); 
    new_dw->allocateAndPut( ic, d_icLabel, matlIndex, patch ); 
    new_dw->allocateAndPut( src, d_sourceLabel, matlIndex, patch ); 
    new_dw->allocateAndPut( fconv, d_FconvLabel, matlIndex, patch );
    new_dw->allocateAndPut( fdiff, d_FdiffLabel, matlIndex, patch );
    new_dw->allocateAndPut( rhs, d_RHSLabel, matlIndex, patch );

    old_dw->get(phi_oldDW, d_transportVarLabel, matlIndex, patch, gn, 0);

    phi.copyData(phi_oldDW);

    old_phi.initialize(0.0);
    ic.initialize(0.0);
    src.initialize(0.0);
    fconv.initialize(0.0);
    fdiff.initialize(0.0);
    rhs.initialize(0.0);

  }
}

