/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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

#include <CCA/Components/Arches/Directives.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermBase.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermFactory.h>
#include <CCA/Components/Arches/TransportEqns/ScalarEqn.h>
#include <CCA/Ports/Scheduler.h>

#include <Core/Exceptions/InvalidValue.h>
#include <Core/Grid/DbgOutput.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Parallel/Parallel.h>
#include <Core/ProblemSpec/ProblemSpec.h>

#ifdef DO_TIMINGS
#  include <spatialops/util/TimeLogger.h>
#endif

#include <fstream>

using namespace std;
using namespace Uintah;

static DebugStream dbg("ARCHES", false);

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
  varname = eqnName+"_scalar_prNo";
  d_prNo_label = VarLabel::create( varname,
            CCVariable<double>::getTypeDescription());

  _reinitialize_from_other_var = false;

}

ScalarEqn::~ScalarEqn()
{
  VarLabel::destroy(d_FdiffLabel);
  VarLabel::destroy(d_FconvLabel);
  VarLabel::destroy(d_RHSLabel);
  VarLabel::destroy(d_transportVarLabel);
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

  if (db->findBlock("use_laminar_pr")){
    d_laminar_pr = true;
    db->findBlock("use_laminar_pr")->getAttribute("label",d_pr_label);
  } else {
    d_laminar_pr = false;
  }

  // Discretization information
  // new_dw->get(new_den, d_fieldLabels->density>d_densityCPLabel, matlIndex, patch, gn, 0);
  // new_dw->get(old_den, d_fieldLabels->d_densityTempLabel, matlIndex, patch, gn, 0);
  db->getWithDefault( "conv_scheme", d_convScheme, "upwind");
  db->getWithDefault( "doConv", d_doConv, false);
  db->getWithDefault( "doDiff", d_doDiff, false);

  // algorithmic knobs
  if (db->findBlock("use_density_guess")) {
     _stage = 0;
  } else {
    _stage = 1;
  }

  if ( db->findBlock("determines_properties") ){
    _stage = 0;
  } else {
    _stage = 1;
  }

  //override the stage
  if ( db->findBlock("stage")){
    db->findBlock("stage")->getAttribute("value",_stage);
  }

  if ( db->findBlock("reinitialize_from") ){

    _reinitialize_from_other_var = true;
    db->findBlock("reinitialize_from")->getAttribute("label",_reinit_var_name);

  }

  // Source terms:
  if (db->findBlock("src")){

    string srcname;
    for (ProblemSpecP src_db = db->findBlock("src"); src_db != nullptr; src_db = src_db->findNextBlock("src")){

      SourceContainer this_src;
      double weight;

      src_db->getAttribute("label",  this_src.name   );
      src_db->getWithDefault("weight", weight, 1.0);    // by default, add the source to the RHS

      this_src.weight = weight;

      //which sources are turned on for this equation
      d_sources.push_back( this_src );

    }
  }

  //extra src terms which only require a VarLabel
  if ( db->findBlock("extra_src") ) {
    for (ProblemSpecP src_db = db->findBlock("extra_src"); src_db != nullptr; src_db = src_db->findNextBlock("extra_src")){
      string srcname;
      src_db->getAttribute("label", srcname );

      const VarLabel * tempLabel;
      tempLabel = VarLabel::find( srcname );
      extraSourceLabels.push_back( tempLabel );

      //note: varlabel must be registered prior to problem setup
      if (tempLabel == 0 )
        throw InvalidValue("Error: Cannot find the VarLabel for the source term: " + srcname, __FILE__, __LINE__);
    }
  }
  nExtraSources = extraSourceLabels.size();

  //call common problemSetup
  commonProblemSetup( db );

}
void
ScalarEqn::assign_stage_to_sources(){

  SourceTermFactory& factory = SourceTermFactory::self();
  for (vector<SourceContainer>::iterator iter = d_sources.begin(); iter != d_sources.end(); iter++){
    SourceTermBase& src = factory.retrieve_source_term( iter->name );
    src.set_stage(_stage);
  }

}
//---------------------------------------------------------------------------
// Method: Schedule the evaluation of the transport equation.
//---------------------------------------------------------------------------
void
ScalarEqn::sched_evalTransportEqn( const LevelP& level,
                                   SchedulerP& sched, int timeSubStep )
{

  printSchedule(level,dbg,"ScalarEqn::sched_evalTransportEqn");

  sched_buildTransportEqn( level, sched, timeSubStep );

  sched_solveTransportEqn( level, sched, timeSubStep );

  if ( clip.my_type == ClipInfo::CONSTRAINED )
  {
    sched_advClipping( level, sched, timeSubStep );
  }
}
//---------------------------------------------------------------------------
// Method: Schedule the intialization of the variables.
//---------------------------------------------------------------------------
void
ScalarEqn::sched_initializeVariables( const LevelP& level, SchedulerP& sched )
{
  printSchedule(level,dbg,"ScalarEqn::sched_initializeVariables");

  string taskname = "ScalarEqn::initializeVariables";
  Task* tsk = scinew Task(taskname, this, &ScalarEqn::initializeVariables);
  Ghost::GhostType gn = Ghost::None;

  if (!d_use_constant_D){
    d_mol_D_label = VarLabel::find( d_mol_D_label_name );
  }

  //New
  tsk->computes(d_transportVarLabel);
  tsk->computes(d_RHSLabel);
  tsk->computes(d_FconvLabel);
  tsk->computes(d_FdiffLabel);
  tsk->computes(d_prNo_label);

  //Old
  if ( _reinitialize_from_other_var && d_fieldLabels->recompile_taskgraph ){

    _reinit_var_label = 0;
    _reinit_var_label = VarLabel::find( _reinit_var_name );
    if ( _reinit_var_label == 0 ){
      throw InvalidValue("Error: Cannot find the reinitialization label for the scalar eqn: "+d_eqnName, __FILE__, __LINE__);
    }
    tsk->requires( Task::OldDW, _reinit_var_label, gn, 0 );

  } else {

    tsk->requires(Task::OldDW, d_transportVarLabel, gn, 0);

  }

  if (d_laminar_pr){
    // This requires that the LaminarPrNo model is activated
    const VarLabel* pr_label = VarLabel::find(d_pr_label);
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
    constCCVariable<double> oldVar;
    new_dw->allocateAndPut( newVar, d_transportVarLabel, matlIndex, patch );

    if ( _reinitialize_from_other_var && d_fieldLabels->recompile_taskgraph ){

      old_dw->get(oldVar, _reinit_var_label, matlIndex, patch, gn, 0);

    } else {

      old_dw->get(oldVar, d_transportVarLabel, matlIndex, patch, gn, 0);

    }


    CCVariable<double> Fdiff;
    CCVariable<double> Fconv;
    CCVariable<double> RHS;

    new_dw->allocateAndPut( Fdiff, d_FdiffLabel, matlIndex, patch );
    new_dw->allocateAndPut( Fconv, d_FconvLabel, matlIndex, patch );
    new_dw->allocateAndPut( RHS, d_RHSLabel, matlIndex, patch );

    newVar.initialize(0.0);
    Fdiff.initialize(0.0);
    Fconv.initialize(0.0);
    RHS.initialize(0.0);

    newVar.copyData(oldVar);

    CCVariable<double> pr_no;
    new_dw->allocateAndPut( pr_no, d_prNo_label, matlIndex, patch );

    if ( d_laminar_pr ) {
      constCCVariable<double> lam_pr_no;
      const VarLabel* pr_label = VarLabel::find(d_pr_label);
      old_dw->get( lam_pr_no, pr_label, matlIndex, patch, gn, 0 );

      pr_no.copyData( lam_pr_no );

    } else {

      pr_no.initialize( d_turbPrNo );

    }

  }
}
//---------------------------------------------------------------------------
// Method: Schedule compute the sources.
//---------------------------------------------------------------------------
void
ScalarEqn::sched_computeSources( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
}
//---------------------------------------------------------------------------
// Method: Schedule build the transport equation.
//---------------------------------------------------------------------------
void
ScalarEqn::sched_buildTransportEqn( const LevelP& level, SchedulerP& sched, const int timeSubStep )
{
  string taskname = "ScalarEqn::buildTransportEqn";

  Task* tsk = scinew Task(taskname, this, &ScalarEqn::buildTransportEqn, timeSubStep);
  printSchedule(level,dbg,taskname);

  if ( timeSubStep == 0 ){
    tsk->requires(Task::OldDW, d_transportVarLabel, Ghost::AroundCells, 2);
  } else {
    tsk->requires(Task::NewDW, d_transportVarLabel, Ghost::AroundCells, 2);
  }
  //----NEW----
  // note that rho and U are copied into new DW in ExplicitSolver::setInitialGuess
  if ( _stage == 0 ){
    tsk->requires(Task::NewDW, d_fieldLabels->d_densityCPLabel, Ghost::AroundCells, 1);
  } else {
    tsk->requires(Task::NewDW, VarLabel::find("density_old"), Ghost::AroundCells, 1);
  }
  tsk->requires(Task::NewDW, d_fieldLabels->d_turbViscosLabel, Ghost::AroundCells, 1);
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
  tsk->requires(Task::NewDW, d_prNo_label, Ghost::None, 0);

  // srcs
  for (vector<SourceContainer>::iterator iter = d_sources.begin();
       iter != d_sources.end(); iter++){

    iter->label = VarLabel::find( iter->name );

    if ( iter->label == 0 ){
      throw InvalidValue("Error: Source Label not found: "+iter->name, __FILE__, __LINE__);
    }

    tsk->requires( Task::NewDW, iter->label, Ghost::None, 0 );

  }

  //extra sources
  for ( int i = 0; i < nExtraSources; i++ ) {
    tsk->requires( Task::NewDW, extraSourceLabels[i], Ghost::None, 0 );
  }

  //-----OLD-----
  tsk->requires(Task::OldDW, d_fieldLabels->d_areaFractionLabel, Ghost::AroundCells, 2);

  //---- time substep dependent ---
  if ( !d_use_constant_D ){
    if ( timeSubStep == 0 ){
      tsk->requires( Task::OldDW, d_mol_D_label, Ghost::AroundCells, 1 );
    } else {
      tsk->requires( Task::NewDW, d_mol_D_label, Ghost::AroundCells, 1 );
    }
  }

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
                              const int timeSubStep )
{
  //patch loop
  for (int p=0; p < patches->size(); p++){

    //Ghost::GhostType  gaf = Ghost::AroundFaces;
    Ghost::GhostType  gac = Ghost::AroundCells;
    Ghost::GhostType  gn  = Ghost::None;

    const Patch* patch = patches->get(p);
    const Level* level = patch->getLevel();
    const int ilvl = level->getID();
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex();

    DataWarehouse* which_dw;
    if ( timeSubStep == 0 ){
      which_dw = old_dw;
    } else {
      which_dw = new_dw;
    }

    Vector Dx = patch->dCell();

    constCCVariable<double> phi;
    constCCVariable<double> den;
    constCCVariable<double> mu_t;
    constCCVariable<double> D_mol;
    constSFCXVariable<double> uVel;
    constSFCYVariable<double> vVel;
    constSFCZVariable<double> wVel;
    constCCVariable<Vector> areaFraction;
    constCCVariable<double> prNo;

    CCVariable<double> Fdiff;
    CCVariable<double> Fconv;
    CCVariable<double> RHS;

    which_dw->get(phi, d_transportVarLabel, matlIndex, patch, gac, 2);

    if ( _stage == 0 ){
      new_dw->get(den, d_fieldLabels->d_densityCPLabel, matlIndex, patch, gac, 1);
    } else {
      new_dw->get(den, VarLabel::find("density_old"), matlIndex, patch, gac, 1);
    }
    new_dw->get(mu_t, d_fieldLabels->d_turbViscosLabel, matlIndex, patch, gac, 1);
    new_dw->get(uVel, d_fieldLabels->d_uVelocitySPBCLabel, matlIndex, patch, gac, 1);
    new_dw->get(prNo, d_prNo_label, matlIndex, patch, gn, 0);
    old_dw->get(areaFraction, d_fieldLabels->d_areaFractionLabel, matlIndex, patch, gac, 2);

    if ( !d_use_constant_D ){
      if ( timeSubStep == 0 ){
        old_dw->get(D_mol, d_mol_D_label, matlIndex, patch, gac, 1);
      } else {
        new_dw->get(D_mol, d_mol_D_label, matlIndex, patch, gac, 1);
      }
    }

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
    for (vector<SourceContainer>::iterator src_iter = d_sources.begin(); src_iter != d_sources.end(); src_iter++){
      constCCVarWrapper temp_var;  // Outside of this scope src is no longer available
      new_dw->get(temp_var.data, src_iter->label, matlIndex, patch, gn, 0);
      temp_var.sign = (*src_iter).weight;
      sourceVars.push_back(temp_var);
    }

    //put extra sources into static array
    std::vector <constCCVariable<double> > extraSources (nExtraSources);
    for ( int i = 0; i < nExtraSources; i++ ) {
      const VarLabel* tempLabel = extraSourceLabels[i];
      new_dw->get( extraSources[i], tempLabel, matlIndex, patch, gn, 0);
    }

    RHS.initialize(0.0);
    Fconv.initialize(0.0);
    Fdiff.initialize(0.0);

#ifdef DO_TIMINGS
      SpatialOps::TimeLogger timer("old_scalar_build.out."+d_eqnName);
      timer.start("work");
#endif

    //----CONVECTION
    if (d_doConv) {
      d_disc->computeConv( patch, Fconv, phi, uVel, vVel, wVel, den, areaFraction, d_convScheme );
      // look for and add contribution from intrusions.
      if ( _using_new_intrusion ) {
        _intrusions[ilvl]->addScalarRHS( patch, Dx, d_eqnName, RHS, den );
      }
    }

    //----DIFFUSION
    if ( d_use_constant_D ) {
      if (d_doDiff)
        d_disc->computeDiff( patch, Fdiff, phi, mu_t, d_mol_diff, den, areaFraction, prNo );
    } else {
      if (d_doDiff)
        d_disc->computeDiff( patch, Fdiff, phi, mu_t, D_mol, den, areaFraction, prNo );
    }

    //----SUM UP RHS
    //
    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
      IntVector c = *iter;

      RHS[c] += Fdiff[c] - Fconv[c];

      //-----ADD SOURCES
      for (vector<constCCVarWrapper>::iterator siter = sourceVars.begin(); siter != sourceVars.end(); siter++){
        RHS[c] += siter->sign * (siter->data)[c] * vol;
      }

      //-----ADD Extra Sources
      for ( int i = 0; i< nExtraSources; i++) {
        RHS[c] += extraSources[i][c] * vol;
      }
    }
#ifdef DO_TIMINGS
    timer.stop("work");
#endif
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
  printSchedule(level,dbg, taskname);

  //New
  tsk->modifies(d_transportVarLabel);
  tsk->requires(Task::NewDW, d_RHSLabel, Ghost::None, 0);

  if ( _stage == 0 ){
    tsk->requires(Task::NewDW, d_fieldLabels->d_densityGuessLabel, Ghost::None, 0);
  } else {
    tsk->requires(Task::NewDW, d_fieldLabels->d_densityTempLabel, Ghost::None, 0);
  }
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

    CCVariable<double> phi;
    constCCVariable<double> RHS;
    constCCVariable<double> old_den;
    constCCVariable<double> new_den;

    new_dw->getModifiable(phi, d_transportVarLabel, matlIndex, patch);
    new_dw->get(RHS, d_RHSLabel, matlIndex, patch, gn, 0);

    if ( _stage == 0 ){
      new_dw->get(new_den, d_fieldLabels->d_densityGuessLabel, matlIndex, patch, gn, 0);
      new_dw->get(old_den, d_fieldLabels->d_densityCPLabel, matlIndex, patch, gn, 0);
    } else {
      new_dw->get(new_den, d_fieldLabels->d_densityCPLabel, matlIndex, patch, gn, 0);
      new_dw->get(old_den, d_fieldLabels->d_densityTempLabel, matlIndex, patch, gn, 0);
    }

    // ----FE UPDATE
    //     to get phi^{(j+1)}
    d_timeIntegrator->singlePatchFEUpdate( patch, phi, old_den, new_den, RHS, dt, d_eqnName);

    if ( clip.activated )
      clipPhi( patch, phi );

    //----BOUNDARY CONDITIONS
    // re-compute the boundary conditions after the update.
    // This is needed because of the table lookup before the time
    // averaging occurs
    computeBCs( patch, d_eqnName, phi );

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
  printSchedule(level,dbg, taskname);

  //New
  tsk->modifies(d_transportVarLabel);
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

    CCVariable<double> new_phi;
    constCCVariable<double> old_phi;
    constCCVariable<double> new_den;
    constCCVariable<double> old_den;

    new_dw->getModifiable( new_phi, d_transportVarLabel, matlIndex, patch );
    old_dw->get( old_phi, d_transportVarLabel, matlIndex, patch, gn, 0 );
    new_dw->get( new_den, d_fieldLabels->d_densityCPLabel, matlIndex, patch, gn, 0);
    old_dw->get( old_den, d_fieldLabels->d_densityCPLabel, matlIndex, patch, gn, 0);

    //----Time averaging done here.
    d_timeIntegrator->timeAvePhi( patch, new_phi, old_phi, new_den, old_den, timeSubStep, clip.tol, clip.do_low, clip.low, clip.do_high, clip.high );

    //----BOUNDARY CONDITIONS
    //    must update BCs for next substep
    computeBCs( patch, d_eqnName, new_phi );

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
  if ( clip.do_low || clip.do_high ){
    for (CellIterator iter=p->getCellIterator(0); !iter.done(); iter++){

      IntVector c = *iter;

      if ( clip.do_low ) {
        if ( phi[c] < clip.low+clip.tol )
          phi[c] = clip.low;
      }

      if ( clip.do_high ) {
        if (phi[c] > clip.high-clip.tol)
          phi[c] = clip.high;
      }
    }
  }
}

void
ScalarEqn::sched_advClipping( const LevelP& level, SchedulerP& sched, int timeSubStep )
{

  if ( clip.my_type != ClipInfo::STANDARD ){

    string taskname = "ScalarEqn::advClipping";

    Task* tsk = scinew Task(taskname, this, &ScalarEqn::advClipping, timeSubStep);
    printSchedule(level,dbg, taskname);

    if ( clip.my_type == ClipInfo::CONSTRAINED ){
      const VarLabel* ind_var_label = VarLabel::find( clip.ind_var );
      if ( ind_var_label == 0 ) {
        throw InvalidValue("Error: For Clipping on equation: "+d_eqnName+" -- Cannot find the constraining variable: "+clip.ind_var, __FILE__, __LINE__);
      }
      tsk->requires(Task::NewDW, ind_var_label, Ghost::None, 0);
    }

    tsk->modifies(d_transportVarLabel);

    sched->addTask(tsk, level->eachPatch(), d_fieldLabels->d_sharedState->allArchesMaterials());

  }
}

void
ScalarEqn::advClipping( const ProcessorGroup* pc,
                    const PatchSubset* patches,
                    const MaterialSubset* matls,
                    DataWarehouse* old_dw,
                    DataWarehouse* new_dw,
                    int timeSubStep )
{
  //patch loop
  for (int p=0; p < patches->size(); p++){

    //Ghost::GhostType  gn  = Ghost::None;
    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex();

    int length_clip_dep_low_vec = clip_dep_low_vec.size();
    int length_clip_dep_vec = clip_dep_vec.size();
    int length_clip_ind_vec = clip_ind_vec.size();

    CCVariable<double> scalar;
    new_dw->getModifiable( scalar, d_transportVarLabel, matlIndex, patch );
    // --- clipping as constrained by another variable -- //

    const VarLabel* ind_var_label = VarLabel::find( clip.ind_var );
    constCCVariable<double> ivcon;  //(for "independent scalar")
    new_dw->get( ivcon, ind_var_label, matlIndex, patch, Ghost::None, 0 );

    for (CellIterator iter=patch->getCellIterator(0); !iter.done(); iter++)
    {
      IntVector c = *iter;
      int bounds = 9999;
      int id_pos=bounds;
      // find where independent vector is
      for(int i=0; i < (length_clip_ind_vec-1); i = i + 1)
      {
        if ( (ivcon[c] > clip_ind_vec[i]) &&  (ivcon[c] <= clip_ind_vec[i+1]))
        {
          id_pos=i;
          break;
        }
        else if (ivcon[c]<=clip_ind_vec[0])
        {
          id_pos=0;
          break;
        }
      }
      if ( id_pos == bounds){
        throw InvalidValue("Error: For advanced Clipping on equation: "+d_eqnName+" cannot find the dependent clipping variable.", __FILE__, __LINE__);
      }
      double see_value;
      double see_value_n;
      if ( (ivcon[c]>0) && (ivcon[c]<1) )
      {
        //     std::cout << "MF is between zero and 1" << endl;
        see_value=clip_dep_vec[id_pos]+((clip_dep_vec[id_pos+1]-clip_dep_vec[id_pos])/(clip_ind_vec[id_pos+1]-clip_ind_vec[id_pos]))*(ivcon[c]-clip_ind_vec[id_pos]);
        see_value_n=clip_dep_low_vec[id_pos]+((clip_dep_low_vec[id_pos+1]-clip_dep_low_vec[id_pos])/(clip_ind_vec[id_pos+1]-clip_ind_vec[id_pos]))*(ivcon[c]-clip_ind_vec[id_pos]);
      }
      else if (ivcon[c]<=clip_ind_vec[0])
      {
        //     std::cout << "MF is zero" << endl;
        see_value=clip_dep_vec[0];
        see_value_n=clip_dep_low_vec[0];
      }
      else
      {
        // std::cout<< "MF is 1" << endl;
        see_value=clip_dep_vec[length_clip_dep_vec];
        see_value_n=clip_dep_low_vec[length_clip_dep_low_vec];
      }

      //  double flagga=0;
      if (see_value < scalar[c])
      {
        // std::cout << "before clipping: current PC: " << setprecision(15) << scalar[c] << " current MF: " << setprecision(15) << ivcon[c] << endl;
          scalar[c] = see_value;
        //    flagga++;
      }
      else if (see_value_n> scalar[c])
      {
        //      std::cout << "before clipping: current PC: " << setprecision(15) << scalar[c] << " current MF: " << setprecision(15) << ivcon[c] << endl;
          scalar[c] = see_value_n;
        //    flagga++;
      }

      //  if (flagga>0)
      //  {
      //    std::cout << " clip_low: "  << see_value_n << " clip_high: " << see_value << " given PC1: " << setprecision(15) << scalar[c] << endl;
      //    std::cout << "" << endl;
      //}
    }
    //----BOUNDARY CONDITIONS
    //    must update BCs for next substep
    computeBCs( patch, d_eqnName, scalar );
  }
}
