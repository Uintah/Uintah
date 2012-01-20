#include <CCA/Components/Arches/TransportEqns/ScalarEqn.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermFactory.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermBase.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Parallel/Parallel.h>

using namespace std;
using namespace Uintah;

//---------------------------------------------------------------------------
// Builder:
//
CCScalarEqnBuilder::CCScalarEqnBuilder( ArchesLabel* fieldLabels, 
                                        ExplicitTimeInt* timeIntegrator,
                                        string eqnName ) : 
EqnBuilder( fieldLabels, timeIntegrator, eqnName )
{}
CCScalarEqnBuilder::~CCScalarEqnBuilder(){}

EqnBase*
CCScalarEqnBuilder::build(){
  return scinew ScalarEqn(d_fieldLabels, d_timeIntegrator, d_eqnName);
}
// End Builder
//---------------------------------------------------------------------------

ScalarEqn::ScalarEqn( ArchesLabel* fieldLabels, ExplicitTimeInt* timeIntegrator, string eqnName )
: 
EqnBase( fieldLabels, timeIntegrator, eqnName )
{
 
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
  varname = eqnName+"_old";
  d_oldtransportVarLabel = VarLabel::create(varname,
            CCVariable<double>::getTypeDescription());
  varname = eqnName+"_scalar_prNo"; 
  d_prNo_label = VarLabel::create( varname, 
            CCVariable<double>::getTypeDescription()); 
}

ScalarEqn::~ScalarEqn()
{
  VarLabel::destroy(d_FdiffLabel);
  VarLabel::destroy(d_FconvLabel); 
  VarLabel::destroy(d_RHSLabel);
  VarLabel::destroy(d_transportVarLabel);
  VarLabel::destroy(d_oldtransportVarLabel);
  VarLabel::destroy(d_prNo_label); 
}
//---------------------------------------------------------------------------
// Method: Problem Setup 
//---------------------------------------------------------------------------
void
ScalarEqn::problemSetup(const ProblemSpecP& inputdb)
{
  ProblemSpecP db = inputdb; 

  d_boundaryCond->problemSetup( db, d_eqnName ); 

  db->getWithDefault("turbulentPrandtlNumber",d_turbPrNo,0.4);

  if (db->findBlock("use_laminar_pr"))
    d_laminar_pr = true; 
  else
    d_laminar_pr = false; 
 
  // Discretization information
  db->getWithDefault( "conv_scheme", d_convScheme, "upwind");
  db->getWithDefault( "doConv", d_doConv, false);
  db->getWithDefault( "doDiff", d_doDiff, false);
  db->getWithDefault( "addSources", d_addSources, true); 
  
  // algorithmic knobs
  //Keep USE_DENSITY_GUESS set to true until the algorithmic details are settled. - Jeremy 
  d_use_density_guess = true; // use the density guess rather than the new density from the table...implies that the equation is updated BEFORE properties are computed. 
  //if (db->findBlock("use_density_guess"))
  //  d_use_density_guess = true; 

  // Source terms:
  if (db->findBlock("src")){
    string srcname; 
    for (ProblemSpecP src_db = db->findBlock("src"); src_db != 0; src_db = src_db->findNextBlock("src")){
      src_db->getAttribute("label", srcname);
      //which sources are turned on for this equation
      d_sources.push_back( srcname ); 

    }
  }

  // Clipping:
  d_doClipping = false; 
  ProblemSpecP db_clipping = db->findBlock("Clipping");

  if (db_clipping) {
    //This seems like a *safe* number to assume 
    double clip_default = -999999;

    d_doLowClip = false; 
    d_doHighClip = false; 
    d_doClipping = true;
    
    db_clipping->getWithDefault("low", d_lowClip,  clip_default);
    db_clipping->getWithDefault("high",d_highClip, clip_default);

    if ( d_lowClip != clip_default ) 
      d_doLowClip = true; 

    if ( d_highClip != clip_default ) 
      d_doHighClip = true; 

    if ( !d_doHighClip && !d_doLowClip ) 
      throw InvalidValue("A low or high clipping must be specified if the <Clipping> section is activated!", __FILE__, __LINE__);
  } 

  // Scaling information:
  db->getWithDefault( "scaling_const", d_scalingConstant, 1.0 ); 

  // Initialization (new way):
  ProblemSpecP db_initialValue = db->findBlock("initialization");
  if (db_initialValue) {

    db_initialValue->getAttribute("type", d_initFunction); 

    if (d_initFunction == "constant") {
      db_initialValue->require("constant", d_constant_init); 

    } else if (d_initFunction == "step") {
      db_initialValue->require("step_direction", d_step_dir); 
      db_initialValue->require("step_value", d_step_value); 

      if( db_initialValue->findBlock("step_start") ) {
        b_stepUsesPhysicalLocation = true;
        db_initialValue->require("step_start", d_step_start); 
        db_initialValue->require("step_end"  , d_step_end); 

      } else if ( db_initialValue->findBlock("step_cellstart") ) {
        b_stepUsesCellLocation = true;
        db_initialValue->require("step_cellstart", d_step_cellstart);
        db_initialValue->require("step_cellend", d_step_cellend);
      }

    } else if (d_initFunction == "mms1") {
      //currently nothing to do here. 
    } else if (d_initFunction == "geometry_fill") {

      db_initialValue->require("constant", d_constant_init); // going to full with this constant 

      ProblemSpecP the_geometry = db_initialValue->findBlock("geom_object"); 
      if (the_geometry) {
        GeometryPieceFactory::create(the_geometry, d_initGeom); 
      } else {
        throw ProblemSetupException("You are missing the geometry specification (<geom_object>) for the transport eqn. initialization!", __FILE__, __LINE__); 
      }
    }
  }
}
//---------------------------------------------------------------------------
// Method: Schedule clean up. 
//---------------------------------------------------------------------------
void 
ScalarEqn::sched_cleanUp( const LevelP& level, SchedulerP& sched )
{
  string taskname = "ScalarEqn::cleanUp";
  Task* tsk = scinew Task(taskname, this, &ScalarEqn::cleanUp);

  sched->addTask(tsk, level->eachPatch(), d_fieldLabels->d_sharedState->allArchesMaterials());
}
//---------------------------------------------------------------------------
// Method: Actually clean up. 
//---------------------------------------------------------------------------
void ScalarEqn::cleanUp( const ProcessorGroup* pc, 
                         const PatchSubset* patches, 
                         const MaterialSubset* matls, 
                         DataWarehouse* old_dw, 
                         DataWarehouse* new_dw )
{

  //Set the initialization flag for the source label to false.
  SourceTermFactory& factory = SourceTermFactory::self(); 
  for (vector<std::string>::iterator iter = d_sources.begin(); iter != d_sources.end(); iter++){
 
    SourceTermBase& temp_src = factory.retrieve_source_term( *iter ); 
  
    temp_src.reinitializeLabel(); 

  }
}
//---------------------------------------------------------------------------
// Method: Schedule the evaluation of the transport equation. 
//---------------------------------------------------------------------------
void
ScalarEqn::sched_evalTransportEqn( const LevelP& level, 
                                   SchedulerP& sched, int timeSubStep )
{

  if (timeSubStep == 0)
    sched_initializeVariables( level, sched );

  if (d_addSources) 
    sched_computeSources( level, sched, timeSubStep ); 

  sched_buildTransportEqn( level, sched, timeSubStep );

  if (d_use_density_guess)
    sched_solveTransportEqn( level, sched, timeSubStep );
  // else we have to do it in ExplicitSolver.cc after properties are updated.

}
//---------------------------------------------------------------------------
// Method: Schedule the intialization of the variables. 
//---------------------------------------------------------------------------
void 
ScalarEqn::sched_initializeVariables( const LevelP& level, SchedulerP& sched )
{
  string taskname = "ScalarEqn::initializeVariables";
  Task* tsk = scinew Task(taskname, this, &ScalarEqn::initializeVariables);
  Ghost::GhostType gn = Ghost::None;

  //New
  tsk->computes(d_transportVarLabel);
  tsk->computes(d_oldtransportVarLabel); // for rk sub stepping 
  tsk->computes(d_RHSLabel); 
  tsk->computes(d_FconvLabel);
  tsk->computes(d_FdiffLabel);
  tsk->computes(d_prNo_label); 

  //Old
  tsk->requires(Task::OldDW, d_transportVarLabel, gn, 0);
  if (d_laminar_pr){
    // This requires that the LaminarPrNo model is activated
    const VarLabel* pr_label = VarLabel::find("laminar_pr"); 
    tsk->requires(Task::OldDW, pr_label, gn, 0); 
  }

  sched->addTask(tsk, level->eachPatch(), d_fieldLabels->d_sharedState->allArchesMaterials());
}
//---------------------------------------------------------------------------
// Method: Actually initialize the variables. 
//---------------------------------------------------------------------------
void ScalarEqn::initializeVariables( const ProcessorGroup* pc, 
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

    newVar.initialize(0.0);
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

    CCVariable<double> pr_no; 
    new_dw->allocateAndPut( pr_no, d_prNo_label, matlIndex, patch ); 

    if ( d_laminar_pr ) { 
      constCCVariable<double> lam_pr_no; 
      const VarLabel* pr_label = VarLabel::find("laminar_pr"); 
      old_dw->get( lam_pr_no, pr_label, matlIndex, patch, gn, 0 ); 

      pr_no.copyData( lam_pr_no ); 

    } else { 

      pr_no.initialize( d_turbPrNo ); 

    }

    curr_time = d_fieldLabels->d_sharedState->getElapsedTime(); 
    curr_ssp_time = curr_time; 

  }
}
//---------------------------------------------------------------------------
// Method: Schedule compute the sources. 
//--------------------------------------------------------------------------- 
void 
ScalarEqn::sched_computeSources( const LevelP& level, SchedulerP& sched, int timeSubStep )
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
ScalarEqn::sched_buildTransportEqn( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  string taskname = "ScalarEqn::buildTransportEqn"; 

  Task* tsk = scinew Task(taskname, this, &ScalarEqn::buildTransportEqn, timeSubStep);

  //----NEW----
  // note that rho and U are copied into new DW in ExplicitSolver::setInitialGuess
  tsk->requires(Task::NewDW, d_fieldLabels->d_densityCPLabel, Ghost::AroundCells, 1); 
  tsk->requires(Task::NewDW, d_fieldLabels->d_viscosityCTSLabel, Ghost::AroundCells, 1);
  tsk->requires(Task::NewDW, d_fieldLabels->d_uVelocitySPBCLabel, Ghost::AroundCells, 1);   
#ifdef YDIM
  tsk->requires(Task::NewDW, d_fieldLabels->d_vVelocitySPBCLabel, Ghost::AroundCells, 1); 
#endif
#ifdef ZDIM
  tsk->requires(Task::NewDW, d_fieldLabels->d_wVelocitySPBCLabel, Ghost::AroundCells, 1); 
#endif
  tsk->modifies(d_FdiffLabel);
  tsk->modifies(d_FconvLabel);
  tsk->modifies(d_RHSLabel);
  tsk->requires(Task::NewDW, d_oldtransportVarLabel, Ghost::AroundCells, 2);
  tsk->requires(Task::NewDW, d_prNo_label, Ghost::None, 0); 

  // srcs
  if (d_addSources) {
    SourceTermFactory& src_factory = SourceTermFactory::self(); 
    for (vector<std::string>::iterator iter = d_sources.begin(); 
         iter != d_sources.end(); iter++){
      SourceTermBase& temp_src = src_factory.retrieve_source_term( *iter ); 
      const VarLabel* temp_varLabel; 
      temp_varLabel = temp_src.getSrcLabel(); 
      tsk->requires( Task::NewDW, temp_src.getSrcLabel(), Ghost::None, 0 ); 
    }
  }
  
  //-----OLD-----
  tsk->requires(Task::OldDW, d_fieldLabels->d_areaFractionLabel, Ghost::AroundCells, 2); 

  sched->addTask(tsk, level->eachPatch(), d_fieldLabels->d_sharedState->allArchesMaterials());
}
//---------------------------------------------------------------------------
// Method: Actually build the transport equation. 
//---------------------------------------------------------------------------
void 
ScalarEqn::buildTransportEqn( const ProcessorGroup* pc, 
                              const PatchSubset* patches, 
                              const MaterialSubset* matls, 
                              DataWarehouse* old_dw, 
                              DataWarehouse* new_dw,
                              int timeSubStep )
{
  //patch loop
  for (int p=0; p < patches->size(); p++){

    //Ghost::GhostType  gaf = Ghost::AroundFaces;
    Ghost::GhostType  gac = Ghost::AroundCells;
    Ghost::GhostType  gn  = Ghost::None;

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    Vector Dx = patch->dCell(); 

    constCCVariable<double> oldPhi;
    constCCVariable<double> den;
    constCCVariable<double> mu_t;
    constSFCXVariable<double> uVel; 
    constSFCYVariable<double> vVel; 
    constSFCZVariable<double> wVel; 
    constCCVariable<Vector> areaFraction; 
    constCCVariable<double> prNo; 

    CCVariable<double> Fdiff; 
    CCVariable<double> Fconv; 
    CCVariable<double> RHS; 

    new_dw->get(oldPhi, d_oldtransportVarLabel, matlIndex, patch, gac, 2);
    new_dw->get(den, d_fieldLabels->d_densityCPLabel, matlIndex, patch, gac, 1); 
    new_dw->get(mu_t,         d_fieldLabels->d_viscosityCTSLabel, matlIndex, patch, gac, 1); 
    new_dw->get(uVel,         d_fieldLabels->d_uVelocitySPBCLabel, matlIndex, patch, gac, 1); 
    new_dw->get(prNo, d_prNo_label, matlIndex, patch, gn, 0); 
    old_dw->get(areaFraction, d_fieldLabels->d_areaFractionLabel, matlIndex, patch, gac, 2); 

    double vol = Dx.x();
#ifdef YDIM
    new_dw->get(vVel,   d_fieldLabels->d_vVelocitySPBCLabel, matlIndex, patch, gac, 1); 
    vol *= Dx.y();
#endif
#ifdef ZDIM
    new_dw->get(wVel,   d_fieldLabels->d_wVelocitySPBCLabel, matlIndex, patch, gac, 1); 
    vol *= Dx.z();
#endif

    new_dw->getModifiable(Fdiff, d_FdiffLabel, matlIndex, patch);
    new_dw->getModifiable(Fconv, d_FconvLabel, matlIndex, patch); 
    new_dw->getModifiable(RHS, d_RHSLabel, matlIndex, patch);
    vector<constCCVarWrapper> sourceVars; 
    if (d_addSources) { 
      SourceTermFactory& src_factory = SourceTermFactory::self(); 
      for (vector<std::string>::iterator src_iter = d_sources.begin(); src_iter != d_sources.end(); src_iter++){
        constCCVarWrapper temp_var;  // Outside of this scope src is no longer available 
        SourceTermBase& temp_src = src_factory.retrieve_source_term( *src_iter ); 
        new_dw->get(temp_var.data, temp_src.getSrcLabel(), matlIndex, patch, gn, 0);
        sourceVars.push_back(temp_var); 
      }
    }
    RHS.initialize(0.0); 
    Fconv.initialize(0.0); 
    Fdiff.initialize(0.0);

    //----CONVECTION
    if (d_doConv) { 
      d_disc->computeConv( patch, Fconv, oldPhi, uVel, vVel, wVel, den, areaFraction, d_convScheme );
      // look for and add contribution from intrusions.
      if ( _using_new_intrusion ) { 
        _intrusions->addScalarRHS( p, Dx, d_eqnName, RHS, den ); 
      }
    }
  
    //----DIFFUSION
    if (d_doDiff)
      d_disc->computeDiff( patch, Fdiff, oldPhi, mu_t, areaFraction, prNo, matlIndex, d_eqnName );
 
    //----SUM UP RHS
    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
      IntVector c = *iter; 

      RHS[c] += Fdiff[c] - Fconv[c];

      //-----ADD SOURCES
      if (d_addSources) {
        for (vector<constCCVarWrapper>::iterator siter = sourceVars.begin(); siter != sourceVars.end(); siter++){
          RHS[c] += (siter->data)[c] * vol; 
        }
      }
    }
  }
}
//---------------------------------------------------------------------------
// Method: Schedule solve the transport equation. 
//---------------------------------------------------------------------------
void
ScalarEqn::sched_solveTransportEqn( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  string taskname = "ScalarEqn::solveTransportEqn";

  Task* tsk = scinew Task(taskname, this, &ScalarEqn::solveTransportEqn, timeSubStep);

  //New
  tsk->modifies(d_transportVarLabel);
  tsk->requires(Task::NewDW, d_RHSLabel, Ghost::None, 0);
  tsk->requires(Task::NewDW, d_fieldLabels->d_densityGuessLabel, Ghost::None, 0);
  tsk->requires(Task::NewDW, d_fieldLabels->d_densityCPLabel, Ghost::None, 0);

  //Old
  tsk->requires(Task::OldDW, d_fieldLabels->d_sharedState->get_delt_label(), Ghost::None, 0);

  sched->addTask(tsk, level->eachPatch(), d_fieldLabels->d_sharedState->allArchesMaterials());
}
//---------------------------------------------------------------------------
// Method: Actually solve the transport equation. 
//---------------------------------------------------------------------------
void 
ScalarEqn::solveTransportEqn( const ProcessorGroup* pc, 
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

    // Here, j is the rk step and n is the time step.  
    //
    CCVariable<double> phi_at_jp1;   // phi^{(j+1)}
    constCCVariable<double> RHS; 
    constCCVariable<double> old_den; 
    constCCVariable<double> new_den; 

    new_dw->getModifiable(phi_at_jp1, d_transportVarLabel, matlIndex, patch);
    new_dw->get(RHS, d_RHSLabel, matlIndex, patch, gn, 0);

    new_dw->get(new_den, d_fieldLabels->d_densityGuessLabel, matlIndex, patch, gn, 0); 
    new_dw->get(old_den, d_fieldLabels->d_densityCPLabel, matlIndex, patch, gn, 0);

    // ----FE UPDATE
    //     to get phi^{(j+1)}
    d_timeIntegrator->singlePatchFEUpdate( patch, phi_at_jp1, old_den, new_den, RHS, dt, curr_ssp_time, d_eqnName);

    if (d_doClipping) 
      clipPhi( patch, phi_at_jp1 ); 

    //----BOUNDARY CONDITIONS
    //    re-compute the boundary conditions after the update.  
    computeBCs( patch, d_eqnName, phi_at_jp1 );

  }
}
//---------------------------------------------------------------------------
// Method: Schedule Time averaging 
//---------------------------------------------------------------------------
void
ScalarEqn::sched_timeAve( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  string taskname = "ScalarEqn::timeAve";

  Task* tsk = scinew Task(taskname, this, &ScalarEqn::timeAve, timeSubStep);

  //New
  tsk->modifies(d_transportVarLabel);
  tsk->modifies(d_oldtransportVarLabel);
  tsk->requires(Task::NewDW, d_fieldLabels->d_densityCPLabel, Ghost::None, 0);

  //Old
  tsk->requires(Task::OldDW, d_transportVarLabel, Ghost::None, 0);
  tsk->requires(Task::OldDW, d_fieldLabels->d_densityCPLabel, Ghost::None, 0); 
  tsk->requires(Task::OldDW, d_fieldLabels->d_sharedState->get_delt_label(), Ghost::None, 0);

  sched->addTask(tsk, level->eachPatch(), d_fieldLabels->d_sharedState->allArchesMaterials());
}
//---------------------------------------------------------------------------
// Method: Time averaging 
//---------------------------------------------------------------------------
void 
ScalarEqn::timeAve( const ProcessorGroup* pc, 
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

    // Compute the current RK time. 
    double factor = d_timeIntegrator->time_factor[timeSubStep]; 
    curr_ssp_time = curr_time + factor * dt;

    CCVariable<double> new_phi; 
    CCVariable<double> last_rk_phi; 
    constCCVariable<double> old_phi;
    constCCVariable<double> new_den; 
    constCCVariable<double> old_den; 

    new_dw->getModifiable( new_phi, d_transportVarLabel, matlIndex, patch ); 
    new_dw->getModifiable( last_rk_phi, d_oldtransportVarLabel, matlIndex, patch ); 
    old_dw->get( old_phi, d_transportVarLabel, matlIndex, patch, gn, 0 ); 
    new_dw->get( new_den, d_fieldLabels->d_densityCPLabel, matlIndex, patch, gn, 0); 
    old_dw->get( old_den, d_fieldLabels->d_densityCPLabel, matlIndex, patch, gn, 0); 

    //----Time averaging done here. 
    d_timeIntegrator->timeAvePhi( patch, new_phi, old_phi, new_den, old_den, timeSubStep, curr_ssp_time ); 

    if (d_doClipping) 
      clipPhi( patch, new_phi ); 

    //----BOUNDARY CONDITIONS
    //    must update BCs for next substep
    computeBCs( patch, d_eqnName, new_phi );

    //----COPY averaged phi into oldphi
    //  I don't think this is needed but keeping it until it is proven...
    last_rk_phi.copyData(new_phi); 

  }
}


//---------------------------------------------------------------------------
// Method: Compute the boundary conditions. 
//---------------------------------------------------------------------------
template<class phiType> void
ScalarEqn::computeBCs( const Patch* patch, 
                       string varName,
                       phiType& phi )
{
  d_boundaryCond->setScalarValueBC( 0, patch, phi, varName ); 
}
//---------------------------------------------------------------------------
// Method: Clip the scalar 
//---------------------------------------------------------------------------
template<class phiType> void
ScalarEqn::clipPhi( const Patch* p, 
                       phiType& phi )
{
  // probably should put these "if"s outside the loop   
  for (CellIterator iter=p->getCellIterator(0); !iter.done(); iter++){

    IntVector c = *iter; 

    if (d_doLowClip) {
      if (phi[c] < d_lowClip) 
        phi[c] = d_lowClip; 
    }

    if (d_doHighClip) { 
      if (phi[c] > d_highClip) 
        phi[c] = d_highClip; 
    } 
  }
}

//---------------------------------------------------------------------------
// Method: Schedule dummy initialization
//---------------------------------------------------------------------------
void
ScalarEqn::sched_dummyInit( const LevelP& level, SchedulerP& sched )
{
  string taskname = "ScalarEqn::dummyInit"; 

  Task* tsk = scinew Task(taskname, this, &ScalarEqn::dummyInit);

  Ghost::GhostType  gn = Ghost::None;

  tsk->requires(Task::OldDW, d_transportVarLabel, gn, 0); 
  tsk->computes(d_transportVarLabel);
  tsk->computes(d_oldtransportVarLabel); 
  tsk->computes(d_FconvLabel); 
  tsk->computes(d_FdiffLabel); 
  tsk->computes(d_RHSLabel); 

  sched->addTask(tsk, level->eachPatch(), d_fieldLabels->d_sharedState->allArchesMaterials());

}
void 
ScalarEqn::dummyInit( const ProcessorGroup* pc, 
                     const PatchSubset* patches, 
                     const MaterialSubset* matls, 
                     DataWarehouse* old_dw, 
                     DataWarehouse* new_dw )
{
  //patch loop
  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    CCVariable<double> phi; 
    CCVariable<double> rkold_phi;
    CCVariable<double> RHS; 
    CCVariable<double> Fconv; 
    CCVariable<double> Fdiff; 
    constCCVariable<double> old_phi; 

    new_dw->allocateAndPut( phi, d_transportVarLabel, matlIndex, patch ); 
    new_dw->allocateAndPut( rkold_phi, d_oldtransportVarLabel, matlIndex, patch ); 
    new_dw->allocateAndPut( RHS, d_RHSLabel, matlIndex, patch); 
    new_dw->allocateAndPut( Fconv, d_FconvLabel, matlIndex, patch); 
    new_dw->allocateAndPut( Fdiff, d_FdiffLabel, matlIndex, patch); 

    old_dw->get( old_phi, d_transportVarLabel, matlIndex, patch, Ghost::None, 0); 

    Fconv.initialize(0.0); 
    Fdiff.initialize(0.0); 
    RHS.initialize(0.0);
    phi.initialize(0.0); 
    rkold_phi.initialize(0.0); 

    phi.copyData(old_phi);

    d_boundaryCond->setScalarValueBC( 0, patch, phi, d_eqnName ); 

  }
}
