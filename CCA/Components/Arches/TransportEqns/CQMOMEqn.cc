#include <CCA/Components/Arches/TransportEqns/CQMOMEqn.h>
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
CQMOMEqnBuilder::CQMOMEqnBuilder( ArchesLabel* fieldLabels,
                                 ExplicitTimeInt* timeIntegrator,
                                 string eqnName ) :
CQMOMEqnBuilderBase( fieldLabels, timeIntegrator, eqnName )
{}
CQMOMEqnBuilder::~CQMOMEqnBuilder(){}

EqnBase*
CQMOMEqnBuilder::build(){
  return scinew CQMOMEqn(d_fieldLabels, d_timeIntegrator, d_eqnName);
}
// End Builder
//---------------------------------------------------------------------------

CQMOMEqn::CQMOMEqn( ArchesLabel* fieldLabels, ExplicitTimeInt* timeIntegrator, string eqnName )
:
EqnBase( fieldLabels, timeIntegrator, eqnName )
{
  
  string varname = eqnName+"_Fdiff";
  d_FdiffLabel = VarLabel::create(varname, CCVariable<double>::getTypeDescription());
  
  varname = eqnName+"_Fconv";
  d_FconvLabel = VarLabel::create(varname, CCVariable<double>::getTypeDescription());
  
  varname = eqnName+"_RHS";
  d_RHSLabel = VarLabel::create(varname, CCVariable<double>::getTypeDescription());
  
  varname = eqnName+"_old";
  d_oldtransportVarLabel = VarLabel::create(varname, CCVariable<double>::getTypeDescription());
  
  varname = eqnName;
  d_transportVarLabel = VarLabel::create(varname, CCVariable<double>::getTypeDescription());

  varname = eqnName+"_src";
  d_sourceLabel = VarLabel::create(varname, CCVariable<double>::getTypeDescription());
  
}

CQMOMEqn::~CQMOMEqn()
{
  VarLabel::destroy(d_FdiffLabel);
  VarLabel::destroy(d_FconvLabel);
  VarLabel::destroy(d_RHSLabel);
  VarLabel::destroy(d_sourceLabel);
  VarLabel::destroy(d_transportVarLabel);
  VarLabel::destroy(d_oldtransportVarLabel);
}
//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
void
CQMOMEqn::problemSetup(const ProblemSpecP& inputdb)
{
  ProblemSpecP db = inputdb;
  
  ProblemSpecP db_root = db->getRootNode();
  ProblemSpecP cqmom_db = db_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("CQMOM");
  std::string which_cqmom;
  cqmom_db->getAttribute( "type", which_cqmom );
  cqmom_db->getAttribute( "partvel", d_usePartVel );
  if ( which_cqmom == "normalized" )
    d_normalized= true;
  else
    d_normalized = false;
  
  db->get("m", momentIndex);
  cqmom_db->getWithDefault("turbulentPrandtlNumber",d_turbPrNo,0.4);
  // Discretization information:
  cqmom_db->getWithDefault( "conv_scheme", d_convScheme, "upwind");
  cqmom_db->getWithDefault( "doConv", d_doConv, false);
  cqmom_db->getWithDefault( "doDiff", d_doDiff, false);

  d_addSources = false;
  d_addExtraSources = false;
  cqmom_db->getWithDefault( "molecular_diffusivity", d_mol_diff, 0.0);
  
  // save the moment eqn label
  CQMOMEqnFactory& cqmomFactory = CQMOMEqnFactory::self();
  cqmom_db->get("NumberInternalCoordinates",M);

  string name = "m_";
  for (int i = 0; i<M ; i++) {
    string node;
    std::stringstream out;
    out << momentIndex[i];
    node = out.str();
    name += node;
  }
  proc0cout << "Problem setup for " << name << endl;
  EqnBase& temp_eqn = cqmomFactory.retrieve_scalar_eqn(name);
  CQMOMEqn& eqn = dynamic_cast<CQMOMEqn&>(temp_eqn);
  d_momentLabel = eqn.getTransportEqnLabel();
  
  d_w_small = eqn.getSmallClip();
  if( d_w_small == 0.0 ) {
    d_w_small = 1e-16;
  }
  
  // Models (source terms):
  for (ProblemSpecP m_db = db->findBlock("model"); m_db !=0; m_db = m_db->findNextBlock("model")){
    //placeholder
  }
  
  // Clipping:
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
  
  // Scaling information:
  //Use this if the moments should be normalized later
  //  cqmom_db->getWithDefault( "scaling_const", d_scalingConstant, 0.0 );
  
  // Extra Source terms (for mms and other tests):
  //make these srouces exist in cqmom block instead of each moment block
  if (cqmom_db->findBlock("src")){
    string srcname;
    d_addExtraSources = true;
    for (ProblemSpecP src_db = db->findBlock("src"); src_db != 0; src_db = src_db->findNextBlock("src")){
      src_db->getAttribute("label", srcname);
      //which sources are turned on for this equation
      d_sources.push_back( srcname );
      
    }
  }


  // Initialization (new way):
  ProblemSpecP db_initialValue = db->findBlock("initialization");
  if (db_initialValue) {
    
    db_initialValue->getAttribute("type", d_initFunction);
    
    // ---------- Constant initialization function ------------------------
    if (d_initFunction == "constant") {
      
      // Constant: normalize if specified
      db_initialValue->get("constant", d_constant_init);
//      if( d_scalingConstant != 0.0 )
//        d_constant_init /= d_scalingConstant;
      
    } else if (d_initFunction == "step" ) {
      
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
//        d_step_value /= d_scalingConstant;
      }
      
    } else if (d_initFunction == "mms1") {
      //currently nothing to do here.
      //placeholder for now
    } else {
      //if no initialization is set, intialize to 0 everywhere
      //might be shortest way(inputfile wise) to specify large problems
      d_initFunction = "constant";
      d_constant_init = 0.0;
    }
  } else {
    //if no initialization is set, intialize to 0 everywhere
    //might be shortest way(inputfile wise) to specify large problems
    d_initFunction = "constant";
    d_constant_init = 0.0;
  }
  
}

//---------------------------------------------------------------------------
// Method: Schedule clean up.
// Probably not needed for DQMOM
//---------------------------------------------------------------------------
void
CQMOMEqn::sched_cleanUp( const LevelP& level, SchedulerP& sched )
{
}
//---------------------------------------------------------------------------
// Method: Schedule the evaluation of the transport equation.
//---------------------------------------------------------------------------
void
CQMOMEqn::sched_evalTransportEqn( const LevelP& level,
                                 SchedulerP& sched, int timeSubStep )
{
  
  if (timeSubStep == 0)
    sched_initializeVariables( level, sched );
  
#ifdef VERIFY_CQMOM_TRANSPORT
  if (d_addExtraSources) {
    proc0cout << endl;
    proc0cout << endl;
    proc0cout << "NOTICE: You have verification turned ON in your DQMOMEqn.h " << endl;
    proc0cout << "Equation " << d_eqnName << " reporting" << endl;
    proc0cout << endl;
    proc0cout << endl;
    
    sched_computeSources( level, sched, timeSubStep );
  }
#endif
  if (d_addExtraSources) {
    sched_computeSources( level, sched, timeSubStep );
  }
  
  sched_buildTransportEqn( level, sched, timeSubStep );
  
  sched_solveTransportEqn( level, sched, timeSubStep );
  
}
//---------------------------------------------------------------------------
// Method: Schedule the intialization of the variables.
//---------------------------------------------------------------------------
void
CQMOMEqn::sched_initializeVariables( const LevelP& level, SchedulerP& sched )
{
  string taskname = "CQMOMEqn::initializeVariables";
  Task* tsk = scinew Task(taskname, this, &CQMOMEqn::initializeVariables);
  Ghost::GhostType gn = Ghost::None;
  //New
  tsk->computes(d_transportVarLabel);
  tsk->computes(d_oldtransportVarLabel); // for rk sub stepping
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
void CQMOMEqn::initializeVariables( const ProcessorGroup* pc,
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
    constCCVariable<double> oldVar;
    new_dw->allocateAndPut( newVar, d_transportVarLabel, matlIndex, patch );
    new_dw->allocateAndPut( rkoldVar, d_oldtransportVarLabel, matlIndex, patch );
    old_dw->get(oldVar, d_transportVarLabel, matlIndex, patch, gn, 0);
    
    newVar.initialize(  0.0);
    rkoldVar.initialize(0.0);
    
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
//---------------------------------------------------------------------------
// Method: Schedule compute the sources.
// Probably not needed for DQMOM EQN
// WILL need this for CQMOM as soruces are independent of linear solve
//---------------------------------------------------------------------------
void
CQMOMEqn::sched_computeSources( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
//  string taskname = "CQMOMEqn::computeSources";
//  Task* tsk = scinew Task(taskname, this, &CQMOMEqn::computeSources);
//  // This scheduler only calls other schedulers
// 
//  SourceTermFactory& factory = SourceTermFactory::self();
//  for (vector<std::string>::iterator iter = d_sources.begin(); iter != d_sources.end(); iter++){
//    
//    SourceTermBase& temp_src = factory.retrieve_source_term( *iter );
//    
//    temp_src.sched_computeSource( level, sched, timeSubStep );
//    
//  }
}

//---------------------------------------------------------------------------
// Method: Actually compute the sources
// Probably not needed for DQMOM EQN
// WILL need this for CQMOM as soruces are independent of linear solve
//---------------------------------------------------------------------------
void
CQMOMEqn::computeSources( const ProcessorGroup* pc,
                          const PatchSubset* patches,
                          const MaterialSubset* matls,
                          DataWarehouse* old_dw,
                          DataWarehouse* new_dw )
{
                          
  
}
//---------------------------------------------------------------------------
// Method: Schedule build the transport equation.
//---------------------------------------------------------------------------
void
CQMOMEqn::sched_buildTransportEqn( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  string taskname = "CQMOMEqn::buildTransportEqn";
  Task* tsk = scinew Task(taskname, this, &CQMOMEqn::buildTransportEqn);
  
  //----NEW----
  tsk->modifies(d_transportVarLabel);
  tsk->requires(Task::NewDW, d_oldtransportVarLabel, Ghost::AroundCells, 2);
  tsk->modifies(d_FdiffLabel);
  tsk->modifies(d_FconvLabel);
  tsk->modifies(d_RHSLabel);
  
  //-----OLD-----
  tsk->requires(Task::OldDW, d_fieldLabels->d_areaFractionLabel, Ghost::AroundCells, 2);
  tsk->requires(Task::OldDW, d_transportVarLabel, Ghost::AroundCells, 2);
  tsk->requires(Task::OldDW, d_fieldLabels->d_viscosityCTSLabel, Ghost::AroundCells, 1);
  tsk->requires(Task::OldDW, d_fieldLabels->d_uVelocitySPBCLabel, Ghost::AroundCells, 1);
#ifdef YDIM
  tsk->requires(Task::OldDW, d_fieldLabels->d_vVelocitySPBCLabel, Ghost::AroundCells, 1);
#endif
#ifdef ZDIM
  tsk->requires(Task::OldDW, d_fieldLabels->d_wVelocitySPBCLabel, Ghost::AroundCells, 1);
#endif
  
  // extra srcs
  if (d_addExtraSources) {
    SourceTermFactory& src_factory = SourceTermFactory::self();
    for (vector<std::string>::iterator iter = d_sources.begin();
         iter != d_sources.end(); iter++){
      SourceTermBase& temp_src = src_factory.retrieve_source_term( *iter );
      tsk->requires( Task::NewDW, temp_src.getSrcLabel(), Ghost::None, 0 );
    }
  }
  
  if (timeSubStep == 0) {
//    tsk->requires(Task::OldDW, d_sourceLabel, Ghost::None, 0);
  } else {
//    tsk->requires(Task::NewDW, d_sourceLabel, Ghost::None, 0);
  }
  
  if ( timeSubStep == 0 ){
    tsk->requires(Task::NewDW, d_fieldLabels->d_densityCPLabel, Ghost::AroundCells, 1);
  } else {
    tsk->requires(Task::NewDW, d_fieldLabels->d_densityCPLabel, Ghost::AroundCells, 1);
  }
  sched->addTask(tsk, level->eachPatch(), d_fieldLabels->d_sharedState->allArchesMaterials());
  
  //NOTE: loop over rquires for weights and abscissas needed for convection term if IC=u,v,w
  if (d_usePartVel) {
    //get a requires on all weights and abscissas
  }
}
//---------------------------------------------------------------------------
// Method: Actually build the transport equation.
//---------------------------------------------------------------------------
void
CQMOMEqn::buildTransportEqn( const ProcessorGroup* pc,
                            const PatchSubset* patches,
                            const MaterialSubset* matls,
                            DataWarehouse* old_dw,
                            DataWarehouse* new_dw )
{
  //patch loop
  for (int p=0; p < patches->size(); p++) {
    
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
    constCCVariable<double> src; //DQMOM_src from Ax=b
    constCCVariable<double> extra_src; // Any additional source (eg, mms or unweighted abscissa src)
    constCCVariable<Vector> areaFraction;
    
    CCVariable<double> phi;
    CCVariable<double> Fdiff;
    CCVariable<double> Fconv;
    CCVariable<double> RHS;
    
    constCCVariable<double> den;
    
    new_dw->get(oldPhi, d_oldtransportVarLabel, matlIndex, patch, gac, 2);
 //   if (new_dw->exists(d_sourceLabel, matlIndex, patch)) {
 //     new_dw->get(src, d_sourceLabel, matlIndex, patch, gn, 0); // only get new_dw value on rkstep > 0
 //   } else {
 //     old_dw->get(src, d_sourceLabel, matlIndex, patch, gn, 0);
 //   }
    
    old_dw->get(mu_t, d_fieldLabels->d_viscosityCTSLabel, matlIndex, patch, gac, 1);
    old_dw->get(uVel,   d_fieldLabels->d_uVelocitySPBCLabel, matlIndex, patch, gac, 1);
    old_dw->get(areaFraction, d_fieldLabels->d_areaFractionLabel, matlIndex, patch, gac, 2);
    double vol = Dx.x();
#ifdef YDIM
    old_dw->get(vVel,   d_fieldLabels->d_vVelocitySPBCLabel, matlIndex, patch, gac, 1);
    vol *= Dx.y();
#endif
#ifdef ZDIM
    old_dw->get(wVel,   d_fieldLabels->d_wVelocitySPBCLabel, matlIndex, patch, gac, 1);
    vol *= Dx.z();
#endif
    
    new_dw->getModifiable(phi, d_transportVarLabel, matlIndex, patch);
    new_dw->getModifiable(Fdiff, d_FdiffLabel, matlIndex, patch);
    new_dw->getModifiable(Fconv, d_FconvLabel, matlIndex, patch);
    new_dw->getModifiable(RHS, d_RHSLabel, matlIndex, patch);
    RHS.initialize(0.0);
    Fconv.initialize(0.0);
    Fdiff.initialize(0.0);
    
    if ( _stage == 0 ){
      new_dw->get(den, d_fieldLabels->d_densityCPLabel, matlIndex, patch, gac, 1);
    } else {
//      new_dw->get(den, VarLabel::find("density_old"), matlIndex, patch, gac, 1);
      new_dw->get(den, d_fieldLabels->d_densityCPLabel, matlIndex, patch, gac, 1);
    }
    
    computeBCs( patch, d_eqnName, phi );
    
    //----CONVECTION
    if (d_doConv) {
//      d_disc->computeConv( patch, Fconv, oldPhi, uVel, vVel, wVel, partVel, areaFraction, d_convScheme );
      //NOTE: use a base scalar transport for inital testing
      //rewrite the convection term later
      if (d_usePartVel) {
        //get weights and abscissas from dw
        //call new convection term with these
      } else {
        d_disc->computeConv( patch, Fconv, oldPhi, uVel, vVel, wVel, den, areaFraction, d_convScheme );
      }
      
      // look for and add contribution from intrusions.
      if ( _using_new_intrusion ) {
        _intrusions->addScalarRHS( patch, Dx, d_eqnName, RHS );
      }
    }
    
    //----DIFFUSION
    if (d_doDiff)
      d_disc->computeDiff( patch, Fdiff, oldPhi, mu_t, d_mol_diff, areaFraction, d_turbPrNo );
    
    //----SUM UP RHS
    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
      IntVector c = *iter;
      
      RHS[c] += Fdiff[c] - Fconv[c];
      
      if (d_addSources)
        RHS[c] += src[c]*vol;


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
    } //cell loop
  
  } //patch loop
  
}
//---------------------------------------------------------------------------
// Method: Schedule solve the transport equation.
//---------------------------------------------------------------------------
void
CQMOMEqn::sched_solveTransportEqn( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  string taskname = "CQMOMEqn::solveTransportEqn";
  Task* tsk = scinew Task(taskname, this, &CQMOMEqn::solveTransportEqn, timeSubStep);
  
  //New
  tsk->modifies(d_transportVarLabel);
  tsk->modifies(d_oldtransportVarLabel);
  tsk->requires(Task::NewDW, d_RHSLabel, Ghost::None, 0);
  
  //Old
  tsk->requires(Task::OldDW, d_transportVarLabel, Ghost::None, 0);
  tsk->requires(Task::OldDW, d_fieldLabels->d_sharedState->get_delt_label(), Ghost::None, 0 );
  
  sched->addTask(tsk, level->eachPatch(), d_fieldLabels->d_sharedState->allArchesMaterials());
}
//---------------------------------------------------------------------------
// Method: Actually solve the transport equation.
//---------------------------------------------------------------------------
void
CQMOMEqn::solveTransportEqn( const ProcessorGroup* pc,
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
    CCVariable<double> oldphi; // phi @ last update for rk substeps
    constCCVariable<double> RHS;
    constCCVariable<double> rk1_phi; // phi @ n for averaging
    
    new_dw->getModifiable(phi, d_transportVarLabel, matlIndex, patch);
    new_dw->getModifiable(oldphi, d_oldtransportVarLabel, matlIndex, patch);
    new_dw->get(RHS, d_RHSLabel, matlIndex, patch, gn, 0);
    old_dw->get(rk1_phi, d_transportVarLabel, matlIndex, patch, gn, 0);
    
    d_timeIntegrator->singlePatchFEUpdate( patch, phi, RHS, dt, curr_ssp_time, d_eqnName );
    
    double factor = d_timeIntegrator->time_factor[timeSubStep];
    curr_ssp_time = curr_time + factor * dt;
    
 //   d_timeIntegrator->timeAvePhi( patch, phi, rk1_phi, new_den, old_den, timeSubStep, curr_ssp_time, clip.tol, clip.do_low, clip.low, clip.do_high, clip.high );
    d_timeIntegrator->timeAvePhi( patch, phi, rk1_phi, timeSubStep, curr_ssp_time, clip.tol, clip.do_low, clip.low, clip.do_high, clip.high);
    
    //----BOUNDARY CONDITIONS
    // For first time step, bc's have been set in dqmomInit
    computeBCs( patch, d_eqnName, phi );
    
    // copy averaged phi into oldphi
    oldphi.copyData(phi);
    
  }
}

void
CQMOMEqn::sched_advClipping( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
}
