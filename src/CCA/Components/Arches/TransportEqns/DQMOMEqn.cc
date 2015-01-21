#include <CCA/Components/Arches/TransportEqns/DQMOMEqn.h>
#include <CCA/Components/Arches/ParticleModels/ParticleHelper.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermFactory.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermBase.h>
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
                                  string ic_name,
                                  const int quadNode ) : 
DQMOMEqnBuilderBase( fieldLabels, timeIntegrator, eqnName ), d_quadNode(quadNode)
{
  d_ic_name = ic_name; 
}
DQMOMEqnBuilder::~DQMOMEqnBuilder()
{}

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
  d_sourceLabel = VarLabel::create(varname, 
            CCVariable<double>::getTypeDescription());

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
  VarLabel::destroy(d_sourceLabel); 
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
    db->require( "nominal_values", d_nominal ); 
  }

  // Models (source terms):
  for (ProblemSpecP m_db = db->findBlock("model"); m_db !=0; m_db = m_db->findNextBlock("model")){
    string model_name; 
    m_db->getAttribute("label", model_name); 

    // now tag on the internal coordinate
    string node;  
    std::stringstream out; 
    out << d_quadNode; 
    node = out.str(); 
    model_name += "_qn";
    model_name += node; 
    // put it in the list
    d_models.push_back(model_name);
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
  db->require( "scaling_const", d_scalingConstant ); 

  int Nqn = ParticleHelper::get_num_env( db, ParticleHelper::DQMOM ); 

  if ( Nqn != d_scalingConstant.size() ){ 
    throw InvalidValue("Error: The number of scaling constants isn't consistent with the number of environments for: "+d_ic_name, __FILE__, __LINE__);
  }

  // Extra Source terms (for mms and other tests):
  if (db->findBlock("src")){
    string srcname; 
    d_addExtraSources = true; 
    for (ProblemSpecP src_db = db->findBlock("src"); src_db != 0; src_db = src_db->findNextBlock("src")){
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
           db_env_constants != 0; db_env_constants = db_env_constants->findNextBlock("env_constant") ) {
        
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
             db_env_step_value != 0; db_env_step_value = db_env_step_value->findNextBlock("env_step_value") ) {
          
          string s_tempQuadNode;
          db_env_step_value->getAttribute("qn", s_tempQuadNode);
          int i_tempQuadNode = atoi( s_tempQuadNode.c_str() );
        
          string s_step_value;
          db_env_step_value->getAttribute("value", s_step_value);
          double step_value = atof( s_step_value.c_str() );
          if( i_tempQuadNode == d_quadNode ) 
            d_step_value = step_value / d_scalingConstant[d_quadNode];
        }
      }//end step_value init.

    } else if (d_initFunction == "mms1") {
      //currently nothing to do here. 

    // ------------ Other initialization function --------------------
    }
  }
}

//---------------------------------------------------------------------------
// Method: Schedule the evaluation of the transport equation. 
//---------------------------------------------------------------------------
void
DQMOMEqn::sched_evalTransportEqn( const LevelP& level, 
                                  SchedulerP& sched, int timeSubStep )
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

    curr_time = d_fieldLabels->d_sharedState->getElapsedTime(); 
    curr_ssp_time = curr_time; 

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

  tsk->modifies(d_FdiffLabel);
  tsk->modifies(d_FconvLabel);
  tsk->modifies(d_RHSLabel);
 
  tsk->requires(Task::OldDW, d_fieldLabels->d_areaFractionLabel, Ghost::AroundCells, 2); 
  tsk->requires(Task::OldDW, d_fieldLabels->d_viscosityCTSLabel, Ghost::AroundCells, 1);

  if( !d_weight ) {
    tsk->requires(which_dw, d_weightLabel, Ghost::AroundCells, 0);
  }

  // extra srcs
  if (d_addExtraSources) {
    SourceTermFactory& src_factory = SourceTermFactory::self(); 
    for (vector<std::string>::iterator iter = d_sources.begin(); 
         iter != d_sources.end(); iter++){
      SourceTermBase& temp_src = src_factory.retrieve_source_term( *iter ); 
      tsk->requires( Task::NewDW, temp_src.getSrcLabel(), Ghost::None, 0 ); 
    }
  }

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
                             DataWarehouse* new_dw, 
                             const int timeSubStep )
{

  //patch loop
  for (int p=0; p < patches->size(); p++){

    Ghost::GhostType  gac = Ghost::AroundCells;
    Ghost::GhostType  gn  = Ghost::None;

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
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

    ArchesLabel::PartVelMap::iterator pvel_iter = d_fieldLabels->partVel.find(d_quadNode);
    new_dw->get( partVel, pvel_iter->second, matlIndex, patch, gac, 1 ); 
    which_dw->get(phi, d_transportVarLabel, matlIndex, patch, gac, 2);

    old_dw->get(mu_t         , d_fieldLabels->d_viscosityCTSLabel  , matlIndex , patch , gac , 1);
    old_dw->get(areaFraction , d_fieldLabels->d_areaFractionLabel  , matlIndex , patch , gac , 2);

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
    if (d_doConv){

      d_disc->computeConv( patch, Fconv, phi, partVel, areaFraction, d_convScheme ); 

      // look for and add contribution from intrusions.
      if ( _using_new_intrusion ) { 
        _intrusions->addScalarRHS( patch, Dx, d_eqnName, RHS ); 
      }

    }
  
    //----DIFFUSION
    if (d_doDiff)
      d_disc->computeDiff( patch, Fdiff, phi, mu_t, d_mol_diff, areaFraction, d_turbPrNo );
 
    //----SUM UP RHS
    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){

      IntVector c = *iter; 
 
      RHS[c] += Fdiff[c] - Fconv[c];

      //if (d_addSources) {
       
        //// Right now, no source term for weights. The conditional statement prevents errors coming from solving Ax=b,
        //// it should be removed if the source term for weights isn't zero.
        //if(!d_weight){
          //if(w[c] > d_w_small){ 
            //RHS[c] += src[c]*vol;   
          //}
        //}
 
        if (d_addExtraSources) {

          // Get the factory of source terms
          SourceTermFactory& src_factory = SourceTermFactory::self();
          for (vector<std::string>::iterator src_iter = d_sources.begin(); src_iter != d_sources.end(); src_iter++){
           //constCCVariable<double> extra_src; 
           SourceTermBase& temp_src = src_factory.retrieve_source_term( *src_iter );
           new_dw->get(extra_src, temp_src.getSrcLabel(), matlIndex, patch, gn, 0);

           // Add to the RHS
           RHS[c] += extra_src[c]*vol;
          }
        }
     // }
    } 
  }
}

void 
DQMOMEqn::sched_addSources( const LevelP& level, SchedulerP& sched, const int timeSubStep ){ 

  string taskname = "DQMOMEqn::addSources"; 

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
  
  sched->addTask(tsk, level->eachPatch(), d_fieldLabels->d_sharedState->allArchesMaterials());

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
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
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
  tsk->requires(Task::OldDW, d_fieldLabels->d_sharedState->get_delt_label(), Ghost::None, 0 );

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
                             int timeSubStep )
{
  //patch loop
  for (int p=0; p < patches->size(); p++){

    Ghost::GhostType  gn  = Ghost::None;

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    delt_vartype DT;
    old_dw->get(DT, d_fieldLabels->d_sharedState->get_delt_label());
    double dt = DT; 

    CCVariable<double> phi;    // phi @ current sub-level 
    constCCVariable<double> RHS; 
    constCCVariable<double> rk1_phi; // phi @ n for averaging 

    new_dw->getModifiable(phi, d_transportVarLabel, matlIndex, patch);
    new_dw->get(RHS, d_RHSLabel, matlIndex, patch, gn, 0);
    old_dw->get(rk1_phi, d_transportVarLabel, matlIndex, patch, gn, 0);

    d_timeIntegrator->singlePatchFEUpdate( patch, phi, RHS, dt, curr_ssp_time, d_eqnName );

    double factor = d_timeIntegrator->time_factor[timeSubStep]; 
    curr_ssp_time = curr_time + factor * dt; 

    if(d_weight){
        // weights being clipped inside this function call
        d_timeIntegrator->timeAvePhi( patch, phi, rk1_phi, timeSubStep, curr_ssp_time, clip.tol, clip.do_low, clip.low, clip.do_high, clip.high );
    }else{
        constCCVariable<double> w;
        new_dw->get(w, d_weightLabel, matlIndex, patch, gn, 0);
        // weighted abscissa being clipped inside this function call 
        d_timeIntegrator->timeAvePhi( patch, phi, rk1_phi, timeSubStep, curr_ssp_time, clip.tol, clip.do_low, clip.low, clip.do_high, clip.high, w); 
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
  tsk->modifies(d_transportVarLabel); 

  if( !d_weight ) {
    tsk->modifies( d_weightLabel ); 
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

    //Ghost::GhostType  gn  = Ghost::None;

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

        w_actual[c] = w[c]*d_scalingConstant[d_quadNode];
      }

    } else {

      CCVariable<double> ic; 
      new_dw->getModifiable(ic, d_icLabel, matlIndex, patch);
      
      CCVariable<double> wa;
      new_dw->getModifiable(wa, d_transportVarLabel, matlIndex, patch); 

      CCVariable<double> w;  
      new_dw->getModifiable(w, d_weightLabel, matlIndex, patch ); 

      // now loop over all cells
      if ( d_unweighted ) {
        for (CellIterator iter=patch->getCellIterator(0); !iter.done(); iter++){
  
          IntVector c = *iter;
         
          ic[c] = wa[c]*d_scalingConstant[d_quadNode];
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
      }
    } //end if weight
  }
}
void
DQMOMEqn::sched_advClipping( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
}
