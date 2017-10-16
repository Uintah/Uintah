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

//----- MomentumSolver.cc ----------------------------------------------

#include <CCA/Components/Arches/Arches.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/ArchesMaterial.h>
#include <CCA/Components/Arches/BoundaryCondition.h>
#include <CCA/Components/Arches/CellInformationP.h>
#include <CCA/Components/Arches/MomentumSolver.h>
#include <CCA/Components/Arches/PhysicalConstants.h>
#include <CCA/Components/Arches/RHSSolver.h>
#include <CCA/Components/Arches/Source.h>
#include <CCA/Components/Arches/ScaleSimilarityModel.h>
#include <CCA/Components/Arches/TimeIntegratorLabel.h>
#include <CCA/Components/MPMArches/MPMArchesLabel.h>
#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Exceptions/VariableNotFoundInGrid.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermFactory.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermBase.h>
#include <CCA/Components/Arches/SourceTerms/ConstSrcTerm.h>
#include <sci_defs/visit_defs.h>

using namespace Uintah;
using namespace std;

//****************************************************************************
// Default constructor for MomentumSolver
//****************************************************************************
MomentumSolver::
MomentumSolver(const ArchesLabel* label,
               const MPMArchesLabel* MAlb,
               TurbulenceModel* turb_model,
               BoundaryCondition* bndry_cond,
               PhysicalConstants* physConst,
               std::map<std::string, std::shared_ptr<TaskFactoryBase> >* task_factory_map
             ) :
               d_lab(label), d_MAlab(MAlb),
               d_turbModel(turb_model),
               d_boundaryCondition(bndry_cond),
               d_physicalConsts(physConst),
               _task_factory_map(task_factory_map)
{
  d_discretize = 0;
  d_source = 0;
  d_rhsSolver = 0;
  _init_type = "none";
  _u_mom = VarLabel::create( "Umom", SFCXVariable<double>::getTypeDescription() );
  _v_mom = VarLabel::create( "Vmom", SFCYVariable<double>::getTypeDescription() );
  _w_mom = VarLabel::create( "Wmom", SFCZVariable<double>::getTypeDescription() );

}

//****************************************************************************
// Destructor
//****************************************************************************
MomentumSolver::~MomentumSolver()
{
  delete d_discretize;
  delete d_source;
  delete d_rhsSolver;
  if ( _init_type != "none" ){
    delete _init_function;
  }

  VarLabel::destroy( _u_mom );
  VarLabel::destroy( _v_mom );
  VarLabel::destroy( _w_mom );

}

//****************************************************************************
// Problem Setup
//****************************************************************************
void
MomentumSolver::problemSetup(const ProblemSpecP& params,
			     SimulationStateP & sharedState)
{
  ProblemSpecP db = params->findBlock("MomentumSolver");

  d_discretize = scinew Discretization(d_physicalConsts);

  string conv_scheme;
  d_central = false;
  db->getWithDefault("convection_scheme",conv_scheme,"upwind");
  if (conv_scheme == "upwind"){
    d_conv_scheme = Discretization::UPWIND;
  }else if (conv_scheme == "central"){
    d_central = true;
    d_conv_scheme = Discretization::CENTRAL;
  } else if (conv_scheme == "wall_upwind") {
    d_conv_scheme = Discretization::WALLUPWIND;
  } else if (conv_scheme == "hybrid") {
    d_conv_scheme = Discretization::HYBRID;
  } else if (conv_scheme == "old_upwind"){
    d_conv_scheme = Discretization::OLD;
  } else if (conv_scheme == "old_central"){
    d_conv_scheme = Discretization::OLD;
    d_central = true;
  }else{
    throw InvalidValue("Convection scheme not supported: " + conv_scheme, __FILE__, __LINE__);
  }

  if ( conv_scheme == "wall_upwind" || conv_scheme == "hybrid" ){
    db->getWithDefault("Re_limit",d_re_limit,2.0);
  }

  // -------------- initialization
  if ( db->findBlock("initialization") ){

    ProblemSpecP db_init = db->findBlock("initialization");
    db->findBlock("initialization")->getAttribute("type", _init_type);

    if ( _init_type == "constant" ){

      _init_function = scinew ConstantVel();

    } else if ( _init_type == "taylor-green" ){

      _init_function = scinew TaylorGreen3D();

    } else if ( _init_type == "almgren" ){

      _init_function = scinew AlmgrenVel();

    } else if ( _init_type == "exponentialvortex" ){

     _init_function = scinew ExponentialVortex();

    } else if ( _init_type == "StABL" ){

      _init_function = scinew StABLVel();

    } else if ( _init_type == "input"){

      _init_function = scinew InputfileInit();

    } else if ( _init_type == "shunn_moin" ){

      _init_function = scinew ShunnMoin();

    } else {

      throw InvalidValue("Initialization type not recognized: " + _init_type, __FILE__, __LINE__);

    }

    _init_function->problemSetup( db_init );

  }

  db->getWithDefault("filter_divergence_constraint",d_filter_divergence_constraint,false);

  d_source = scinew Source(d_physicalConsts);

  // New Source terms (ala the new transport eqn):
  if (db->findBlock("src")){
    string srcname;
    SourceTermFactory& src_factory = SourceTermFactory::self();

    for( ProblemSpecP src_db = db->findBlock( "src" ); src_db != nullptr; src_db = src_db->findNextBlock( "src" ) ) {

      src_db->getAttribute("label", srcname);

      //which sources are turned on for this equation
      d_new_sources.push_back( srcname );

      //Checking in the old source term factory:
      if ( src_factory.source_term_exists( srcname ) ){
        SourceTermBase& a_src = src_factory.retrieve_source_term( srcname );

        ProblemSpecP db_root = db->getRootNode();
        ProblemSpecP db_sources = db_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("TransportEqns")->findBlock("Sources");
        for( ProblemSpecP tmp_src_db = db_sources->findBlock( "src" ); tmp_src_db != nullptr; tmp_src_db = tmp_src_db->findNextBlock( "src" ) ) {
          std::string tempSrcName;
          tmp_src_db->getAttribute("label", tempSrcName);

          if ( tempSrcName == srcname ) {
            //actually call the problem setup on momentum source terms now
            a_src.problemSetup( tmp_src_db );
            break;
          }
        }
      } else {
        // **HACK** This isn't a generic implementation for any source
        //look for it in the turbulence model Factory
        //these should be setup already
        typedef std::map<std::string, std::shared_ptr<TaskFactoryBase> > BFM;
        BFM::iterator turb_fac = _task_factory_map->find("turbulence_model_factory");
        const bool test  = turb_fac->second->has_task(srcname);
        if ( !test ) {
          throw ProblemSetupException("Error: Cannot find a source named: "+srcname+" for the momentum transport", __FILE__, __LINE__ );
        }
      }
    }
  }

  d_rhsSolver = scinew RHSSolver();
  d_mixedModel=d_turbModel->getMixedModel();

  d_wall_closure = "molecular";
  d_wall_const_smag_C = 0.;
  ProblemSpecP db_wall = db->findBlock("wall_closure");
  if ( db_wall != nullptr ){
    db_wall->getAttribute("type", d_wall_closure );

    if ( d_wall_closure == "constant_coefficient" ){
      db_wall->getWithDefault( "wall_csmag", d_wall_const_smag_C, 0.17 );
    }

    if ( d_wall_closure == "constant_coefficient" ||
         d_wall_closure == "dynamic" ){

      db_wall->getWithDefault("standoff_index", d_standoff_index, 1);

    }
  }


#ifdef HAVE_VISIT
  static bool initialized = false;

  // Running with VisIt so add in the variables that the user can
  // modify.
  if( sharedState->getVisIt() && !initialized ) {
    // variable 1 - Must start with the component name and have NO
    // spaces in the var name
    SimulationState::interactiveVar var;
    var.name     = "ARCHES-Re_limit";
    var.type     = Uintah::TypeDescription::double_type;
    var.value    = (void *) &d_re_limit;
    var.range[0]   = 2;
    var.range[1]   = 1e9;
    var.modifiable = true;
    var.recompile  = false;
    var.modified   = false;
    sharedState->d_UPSVars.push_back( var );

    initialized = true;
  }
#endif
}

void MomentumSolver::setInitVelCondition( const Patch* patch,
                                          SFCXVariable<double>& uvel,
                                          SFCYVariable<double>& vvel,
                                          SFCZVariable<double>& wvel,
                                          constCCVariable<double>& rho )
{
  if ( _init_type != "none" ){

    _init_function->setXVel( patch, uvel, rho );
    _init_function->setYVel( patch, vvel, rho );
    _init_function->setZVel( patch, wvel, rho );

  }
}

//****************************************************************************
// Schedule linear momentum solve
//****************************************************************************
void
MomentumSolver::solve(SchedulerP& sched,
                      const PatchSet* patches,
                      const MaterialSet* matls,
                      const TimeIntegratorLabel* timelabels,
                      bool extraProjection)
{
  //computes stencil coefficients and source terms
  // require : pressureCPBC, [u,v,w]VelocityCPBC, densityIN, viscosityIN (new_dw)
  //           [u,v,w]SPBC, densityCP (old_dw)
  // compute : [u,v,w]VelCoefPBLM, [u,v,w]VelConvCoefPBLM
  //           [u,v,w]VelLinSrcPBLM, [u,v,w]VelNonLinSrcPBLM

  sched_buildLinearMatrix(sched, patches, matls, timelabels,
                          extraProjection);


}

//****************************************************************************
// Schedule the build of the linear matrix
//****************************************************************************
void
MomentumSolver::sched_buildLinearMatrix(SchedulerP& sched,
                                        const PatchSet* patches,
                                        const MaterialSet* matls,
                                        const TimeIntegratorLabel* timelabels,
                                        bool extraProjection)
{
  string taskname =  "MomentumSolver::BuildCoeff" +
                     timelabels->integrator_step_name;
  if (extraProjection){
    taskname += "extraProjection";
  }

  Task* tsk = scinew Task(taskname,
                          this, &MomentumSolver::buildLinearMatrix,
                          timelabels, extraProjection);

  Task::WhichDW parent_old_dw;
  if (timelabels->recursion){
    parent_old_dw = Task::ParentOldDW;
  }else {
    parent_old_dw = Task::OldDW;
  }

  Ghost::GhostType  gaf = Ghost::AroundFaces;
  Ghost::GhostType  gac = Ghost::AroundCells;
  tsk->requires(parent_old_dw, d_lab->d_sharedState->get_delt_label());
  tsk->requires(Task::NewDW,   d_lab->d_cellTypeLabel,    gac, 1);
  tsk->requires(Task::NewDW,   d_lab->d_densityCPLabel,   gac, 1);
  tsk->requires(Task::NewDW,   d_lab->d_uVelRhoHatLabel,  gaf, 1);
  tsk->requires(Task::NewDW,   d_lab->d_vVelRhoHatLabel,  gaf, 1);
  tsk->requires(Task::NewDW,   d_lab->d_wVelRhoHatLabel,  gaf, 1);
  tsk->requires(Task::NewDW,   d_lab->d_volFractionLabel, gac, 1);

  tsk->requires(Task::NewDW, d_lab->d_cellInfoLabel, Ghost::None);
  if ( extraProjection ){
    tsk->requires(Task::NewDW, d_lab->d_pressureExtraProjectionLabel,gac, 1);
  }else{
    tsk->requires(Task::NewDW, timelabels->pressure_out,  gac, 1);
  }

  if (d_MAlab) {
    tsk->requires(Task::NewDW, d_lab->d_mmgasVolFracLabel,gac, 1);
  }

  tsk->modifies(d_lab->d_uVelocitySPBCLabel);
  tsk->modifies(d_lab->d_vVelocitySPBCLabel);
  tsk->modifies(d_lab->d_wVelocitySPBCLabel);

  sched->addTask(tsk, patches, matls);
}

//****************************************************************************
// Actual build of the linear matrix
//****************************************************************************
void
MomentumSolver::buildLinearMatrix(const ProcessorGroup* pc,
                                  const PatchSubset* patches,
                                  const MaterialSubset* /*matls*/,
                                  DataWarehouse* old_dw,
                                  DataWarehouse* new_dw,
                                  const TimeIntegratorLabel* timelabels,
                                  bool extraProjection)
{
  DataWarehouse* parent_old_dw;
  if (timelabels->recursion){
    parent_old_dw = new_dw->getOtherDataWarehouse(Task::ParentOldDW);
  }else{
    parent_old_dw = old_dw;
  }
  delt_vartype delT;
  parent_old_dw->get(delT, d_lab->d_sharedState->get_delt_label() );
  double delta_t = delT;
  delta_t *= timelabels->time_multiplier;

  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int indx = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex();
    ArchesVariables velocityVars;
    ArchesConstVariables constVelocityVars;

    // Get the required data
    Ghost::GhostType  gac = Ghost::AroundCells;
    constCCVariable<double> volFraction;
    new_dw->get(constVelocityVars.cellType, d_lab->d_cellTypeLabel,  indx, patch, gac, 1);
    new_dw->get(constVelocityVars.density,  d_lab->d_densityCPLabel, indx, patch, gac, 1);
    new_dw->get(volFraction,                d_lab->d_volFractionLabel, indx, patch, gac, 1);

    if ( extraProjection ){
      new_dw->get(constVelocityVars.pressure, d_lab->d_pressureExtraProjectionLabel, indx, patch, gac, 1);
    }else{
      new_dw->get(constVelocityVars.pressure, timelabels->pressure_out,              indx, patch, gac, 1);
    }

    // Get the PerPatch CellInformation data
    PerPatch<CellInformationP> cellInfoP;
    new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, indx, patch);
    CellInformation* cellinfo = cellInfoP.get().get_rep();

    if (d_MAlab) {
      new_dw->get(constVelocityVars.voidFraction, d_lab->d_mmgasVolFracLabel,indx, patch, gac, 1);
    }

    new_dw->getModifiable(velocityVars.uVelRhoHat, d_lab->d_uVelocitySPBCLabel,indx, patch);
    new_dw->getModifiable(velocityVars.vVelRhoHat, d_lab->d_vVelocitySPBCLabel,indx, patch);
    new_dw->getModifiable(velocityVars.wVelRhoHat, d_lab->d_wVelocitySPBCLabel,indx, patch);
    // Copying the hatted velocity vars into the velocity.  We then directly project these
    // velocities with the pressure.
    new_dw->copyOut(velocityVars.uVelRhoHat,       d_lab->d_uVelRhoHatLabel,   indx, patch);
    new_dw->copyOut(velocityVars.vVelRhoHat,       d_lab->d_vVelRhoHatLabel,   indx, patch);
    new_dw->copyOut(velocityVars.wVelRhoHat,       d_lab->d_wVelRhoHatLabel,   indx, patch);

    if (d_MAlab) {
      d_boundaryCondition->calculateVelocityPred_mm(patch, delta_t,
                                                    cellinfo,&velocityVars, &constVelocityVars);

    }else {
      d_rhsSolver->calculateVelocity(patch, delta_t, cellinfo, &velocityVars,
                                     constVelocityVars.density, constVelocityVars.pressure, volFraction);
    }

    //intrusions:
    d_boundaryCondition->setHattedIntrusionVelocity( patch, velocityVars.uVelRhoHat, velocityVars.vVelRhoHat,
                                                     velocityVars.wVelRhoHat, constVelocityVars.density );

    // boundary condition
    Patch::FaceType mface = Patch::xminus;
    Patch::FaceType pface = Patch::xplus;
    d_boundaryCondition->delPForOutletPressure( patch, indx, delta_t,
                                                 mface, pface,
                                                 velocityVars.uVelRhoHat,
                                                 constVelocityVars.pressure,
                                                 constVelocityVars.density );
    mface = Patch::yminus;
    pface = Patch::yplus;
    d_boundaryCondition->delPForOutletPressure( patch, indx, delta_t,
                                               mface, pface,
                                               velocityVars.vVelRhoHat,
                                               constVelocityVars.pressure,
                                               constVelocityVars.density );
    mface = Patch::zminus;
    pface = Patch::zplus;
    d_boundaryCondition->delPForOutletPressure( patch, indx, delta_t,
                                               mface, pface,
                                               velocityVars.wVelRhoHat,
                                               constVelocityVars.pressure,
                                               constVelocityVars.density );

    d_boundaryCondition->velocityOutletPressureTangentBC(patch,
                                                         &velocityVars, &constVelocityVars);

    double time_shift = 0.0;
    time_shift = delta_t * timelabels->time_position_multiplier_before_average;
    d_boundaryCondition->velRhoHatInletBC(patch,
                                          &velocityVars, &constVelocityVars,
                                          indx,
                                          time_shift);


  }
}

//****************************************************************************
// Schedule calculation of hat velocities
//****************************************************************************
void MomentumSolver::solveVelHat(const LevelP& level,
                                 SchedulerP& sched,
                                 const TimeIntegratorLabel* timelabels, const int curr_level )
{
  const PatchSet* patches = level->eachPatch();
  const MaterialSet* matls = d_lab->d_sharedState->allArchesMaterials();

  IntVector periodic_vector = level->getPeriodicBoundaries();
  d_3d_periodic = (periodic_vector == IntVector(1,1,1));

  int timeSubStep = 0;
  if ( timelabels->integrator_step_number == TimeIntegratorStepNumber::Second )
    timeSubStep = 1;
  else if ( timelabels->integrator_step_number == TimeIntegratorStepNumber::Third )
    timeSubStep = 2;

  // Schedule additional sources for evaluation
  SourceTermFactory& factory = SourceTermFactory::self();
  for ( vector<std::string>::iterator iter = d_new_sources.begin();
        iter != d_new_sources.end(); iter++ ){

    if ( factory.source_term_exists( *iter ) ){

      SourceTermBase& src = factory.retrieve_source_term( *iter );
      src.sched_computeSource( level, sched, timeSubStep );

    }
  }

  sched_buildLinearMatrixVelHat(sched, patches, matls,
                                timelabels, curr_level);

}

// ****************************************************************************
// Schedule build of linear matrix
// ****************************************************************************
void
MomentumSolver::sched_buildLinearMatrixVelHat(SchedulerP& sched,
                                              const PatchSet* patches,
                                              const MaterialSet* matls,
                                              const TimeIntegratorLabel* timelabels,
                                              const int curr_level )
{
  string taskname =  "MomentumSolver::BuildCoeffVelHat" +
                     timelabels->integrator_step_name;
  Task* tsk = scinew Task(taskname,
                          this, &MomentumSolver::buildLinearMatrixVelHat,
                          timelabels, curr_level);


  Task::WhichDW parent_old_dw;
  if (timelabels->recursion){
    parent_old_dw = Task::ParentOldDW;
  }else{
    parent_old_dw = Task::OldDW;
  }

  tsk->requires(parent_old_dw, d_lab->d_sharedState->get_delt_label());

  Task::WhichDW which_dw;

  if ( curr_level == 0 ){
    which_dw = Task::OldDW;
  } else {
    which_dw = Task::NewDW;
  }

  // Requires
  // from old_dw for time integration
  // get old_dw from getTop function

  Ghost::GhostType  gn = Ghost::None;
  Ghost::GhostType  gac = Ghost::AroundCells;
  Ghost::GhostType  gaf = Ghost::AroundFaces;
  Task::MaterialDomainSpec oams = Task::OutOfDomain;  //outside of arches matlSet.

  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel, gac, 2);
  tsk->requires(Task::NewDW, d_lab->d_cellInfoLabel, gn);
  tsk->requires(Task::NewDW, d_lab->d_volFractionLabel, gac, 2);

  if ( d_wall_closure == "constant_coefficient"){
    d_boundaryCondition->sched_wallStressConstSmag( which_dw, tsk );
  }

  if (timelabels->multiple_steps){
    tsk->requires(Task::NewDW, d_lab->d_densityTempLabel,gac, 2);
  }else{
    tsk->requires(Task::OldDW, d_lab->d_densityCPLabel,  gac, 2);
  }

  Task::WhichDW old_values_dw;
  if (timelabels->use_old_values) {
    old_values_dw = parent_old_dw;
    tsk->requires(old_values_dw, d_lab->d_densityCPLabel,   gac, 1);
    tsk->requires(old_values_dw, d_lab->d_uVelocitySPBCLabel, gn, 0);
    tsk->requires(old_values_dw, d_lab->d_vVelocitySPBCLabel, gn, 0);
    tsk->requires(old_values_dw, d_lab->d_wVelocitySPBCLabel, gn, 0);
  }else {
    old_values_dw = Task::NewDW;
    tsk->requires(Task::NewDW,   d_lab->d_densityTempLabel, gac, 1);
  }

  tsk->requires(Task::NewDW, d_lab->d_densityCPLabel,     gac, 1);

  d_denRefArrayLabel = VarLabel::find("denRefArray");
  tsk->requires(Task::OldDW, d_denRefArrayLabel,   gac, 1);

  tsk->requires(Task::NewDW, d_lab->d_viscosityCTSLabel,  gac, 2);
  tsk->requires(Task::NewDW, d_lab->d_uVelocitySPBCLabel, gaf, 2);
  tsk->requires(Task::NewDW, d_lab->d_vVelocitySPBCLabel, gaf, 2);
  tsk->requires(Task::NewDW, d_lab->d_wVelocitySPBCLabel, gaf, 2);
  tsk->requires(Task::OldDW, d_lab->d_pressurePSLabel, gac, 1); 
  tsk->requires(Task::OldDW, d_lab->d_turbViscosLabel, gac, 1); 
  tsk->requires(Task::OldDW, d_lab->d_CCUVelocityLabel, gac, 1); 
  tsk->requires(Task::OldDW, d_lab->d_CCVVelocityLabel, gac, 1); 
  tsk->requires(Task::OldDW, d_lab->d_CCWVelocityLabel, gac, 1); 

//#ifdef divergenceconstraint

  if (d_mixedModel) {
    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First){
      tsk->requires(Task::OldDW, d_lab->d_stressTensorCompLabel,
                                d_lab->d_tensorMatl,  oams,   gac, 1);
   }else {
      tsk->requires(Task::NewDW, d_lab->d_stressTensorCompLabel,
                                d_lab->d_tensorMatl,  oams,   gac, 1);
    }
  }

//#endif // divergenceconstraint
  // for multi-material
    // requires su_drag[x,y,z], sp_drag[x,y,z] for arches

  if (d_MAlab) {
    tsk->requires(Task::NewDW, d_MAlab->d_uVel_mmLinSrcLabel,   gn, 0);
    tsk->requires(Task::NewDW, d_MAlab->d_uVel_mmNonlinSrcLabel,gn, 0);
    tsk->requires(Task::NewDW, d_MAlab->d_vVel_mmLinSrcLabel,   gn, 0);
    tsk->requires(Task::NewDW, d_MAlab->d_vVel_mmNonlinSrcLabel,gn, 0);
    tsk->requires(Task::NewDW, d_MAlab->d_wVel_mmLinSrcLabel,   gn, 0);
    tsk->requires(Task::NewDW, d_MAlab->d_wVel_mmNonlinSrcLabel,gn, 0);
  }

  tsk->modifies(d_lab->d_uVelRhoHatLabel);
  tsk->modifies(d_lab->d_vVelRhoHatLabel);
  tsk->modifies(d_lab->d_wVelRhoHatLabel);
  tsk->modifies(d_lab->d_conv_scheme_x_Label);
  tsk->modifies(d_lab->d_conv_scheme_y_Label);
  tsk->modifies(d_lab->d_conv_scheme_z_Label);

  // Adding new sources from factory:
  for ( vector<std::string>::iterator iter = d_new_sources.begin();
        iter != d_new_sources.end(); iter++ ){

    tsk->requires(Task::NewDW, VarLabel::find( *iter ), gac, 1);

  }

  sched->addTask(tsk, patches, matls);

}

struct sumNonlinearSources{
       sumNonlinearSources(double _vol,
                           SFCXVariable<double> &_uNonlinearSrc,
                           SFCYVariable<double> &_vNonlinearSrc,
                           SFCZVariable<double> &_wNonlinearSrc,
                           constCCVariable<Vector> &_vectorSource)  :
                           vol(_vol),
                           uNonlinearSrc(_uNonlinearSrc),
                           vNonlinearSrc(_vNonlinearSrc),
                           wNonlinearSrc(_wNonlinearSrc),
                           vectorSource(_vectorSource){ }

       void operator()(int i , int j, int k ) const {
         uNonlinearSrc(i,j,k)  += (vectorSource(i,j,k).x()+vectorSource(i-1,j  ,k  ).x())*0.5*vol;
         vNonlinearSrc(i,j,k)  += (vectorSource(i,j,k).y()+vectorSource(i  ,j-1,k  ).y())*0.5*vol;
         wNonlinearSrc(i,j,k)  += (vectorSource(i,j,k).z()+vectorSource(i  ,j  ,k-1).z())*0.5*vol;
       }

  private:
       double vol;
       SFCXVariable<double> &uNonlinearSrc;
       SFCYVariable<double> &vNonlinearSrc;
       SFCZVariable<double> &wNonlinearSrc;
       constCCVariable<Vector> &vectorSource;
};

// ***********************************************************************
// Actual build of linear matrices for momentum components
// ***********************************************************************
void
MomentumSolver::buildLinearMatrixVelHat(const ProcessorGroup* pc,
                                        const PatchSubset* patches,
                                        const MaterialSubset* /*matls*/,
                                        DataWarehouse* old_dw,
                                        DataWarehouse* new_dw,
                                        const TimeIntegratorLabel* timelabels,
                                        const int curr_level )
{
  DataWarehouse* parent_old_dw;
  if (timelabels->recursion){
    parent_old_dw = new_dw->getOtherDataWarehouse(Task::ParentOldDW);
  }else{
    parent_old_dw = old_dw;
  }

  DataWarehouse* which_dw;
  if ( curr_level == 0  ){
    which_dw = old_dw;
  } else {
    which_dw = new_dw;
  }
  delt_vartype delT;
  parent_old_dw->get(delT, d_lab->d_sharedState->get_delt_label() );
  double delta_t = delT;
  delta_t *= timelabels->time_multiplier;

  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int indx = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex();
    ArchesVariables velocityVars;
    ArchesConstVariables constVelocityVars;

    Ghost::GhostType  gn = Ghost::None;
    Ghost::GhostType  gac = Ghost::AroundCells;
    Ghost::GhostType  gaf = Ghost::AroundFaces;

    new_dw->get(constVelocityVars.cellType, d_lab->d_cellTypeLabel, indx, patch, gac, 2);
    constCCVariable<double> volFraction;
    new_dw->get(volFraction, d_lab->d_volFractionLabel, indx, patch, gac, 2);

    //multiple_steps is false on rk step = 0
    if (timelabels->multiple_steps){
      new_dw->get(constVelocityVars.density, d_lab->d_densityTempLabel, indx, patch, gac, 2);
    }else{
      old_dw->get(constVelocityVars.density, d_lab->d_densityCPLabel, indx,  patch, gac, 2);
    }

    DataWarehouse* old_values_dw;
    if (timelabels->use_old_values) {
      old_values_dw = parent_old_dw;
      old_values_dw->get(constVelocityVars.old_density, d_lab->d_densityCPLabel,   indx, patch, gac, 1);
    }
    else {
      old_values_dw = new_dw;
      old_values_dw->get(constVelocityVars.old_density, d_lab->d_densityTempLabel, indx, patch, gac, 1);
    }

    old_values_dw->get(constVelocityVars.old_uVelocity, d_lab->d_uVelocitySPBCLabel, indx, patch, gn, 0);
    old_values_dw->get(constVelocityVars.old_vVelocity, d_lab->d_vVelocitySPBCLabel, indx, patch, gn, 0);
    old_values_dw->get(constVelocityVars.old_wVelocity, d_lab->d_wVelocitySPBCLabel, indx, patch, gn, 0);

    new_dw->get(constVelocityVars.new_density, d_lab->d_densityCPLabel,     indx, patch, gac, 1);
    old_dw->get(constVelocityVars.denRefArray, d_denRefArrayLabel,   indx, patch, gac, 1);

    new_dw->get(constVelocityVars.viscosity,   d_lab->d_viscosityCTSLabel,  indx, patch, gac, 2);
    new_dw->get(constVelocityVars.uVelocity,   d_lab->d_uVelocitySPBCLabel, indx, patch, gaf, 2);
    new_dw->get(constVelocityVars.vVelocity,   d_lab->d_vVelocitySPBCLabel, indx, patch, gaf, 2);
    new_dw->get(constVelocityVars.wVelocity,   d_lab->d_wVelocitySPBCLabel, indx, patch, gaf, 2);
    old_dw->get(constVelocityVars.pressure,       d_lab->d_pressurePSLabel,    indx, patch, gac, 1); 
    old_dw->get(constVelocityVars.turbViscosity,  d_lab->d_turbViscosLabel,    indx, patch, gac, 1); 
    old_dw->get(constVelocityVars.CCUVelocity,    d_lab->d_CCUVelocityLabel,   indx, patch, gac, 1); 
    old_dw->get(constVelocityVars.CCVVelocity,    d_lab->d_CCVVelocityLabel,   indx, patch, gac, 1); 
    old_dw->get(constVelocityVars.CCWVelocity,    d_lab->d_CCWVelocityLabel,   indx, patch, gac, 1); 


    constCCVariable<double> old_divergence;

    PerPatch<CellInformationP> cellInfoP;
    new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, indx, patch);
    CellInformation* cellinfo = cellInfoP.get().get_rep();

    // get multimaterial momentum source terms
    // get velocities for MPMArches with extra ghost cells

    if (d_MAlab) {
      new_dw->get(constVelocityVars.mmuVelSu, d_MAlab->d_uVel_mmNonlinSrcLabel,indx, patch,gn, 0);
      new_dw->get(constVelocityVars.mmvVelSu, d_MAlab->d_vVel_mmNonlinSrcLabel,indx, patch,gn, 0);
      new_dw->get(constVelocityVars.mmwVelSu, d_MAlab->d_wVel_mmNonlinSrcLabel,indx, patch,gn, 0);

      new_dw->get(constVelocityVars.mmuVelSp, d_MAlab->d_uVel_mmLinSrcLabel,   indx, patch,gn, 0);
      new_dw->get(constVelocityVars.mmvVelSp, d_MAlab->d_vVel_mmLinSrcLabel,   indx, patch,gn, 0);
      new_dw->get(constVelocityVars.mmwVelSp, d_MAlab->d_wVel_mmLinSrcLabel,   indx, patch,gn, 0);
    }

    for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++) {
      new_dw->allocateTemporary(velocityVars.uVelocityCoeff[ii],         patch);
      new_dw->allocateTemporary(velocityVars.vVelocityCoeff[ii],         patch);
      new_dw->allocateTemporary(velocityVars.wVelocityCoeff[ii],         patch);
      new_dw->allocateTemporary(velocityVars.uVelocityConvectCoeff[ii],  patch);
      new_dw->allocateTemporary(velocityVars.vVelocityConvectCoeff[ii],  patch);
      new_dw->allocateTemporary(velocityVars.wVelocityConvectCoeff[ii],  patch);
      velocityVars.uVelocityCoeff[ii].initialize(0.0);
      velocityVars.vVelocityCoeff[ii].initialize(0.0);
      velocityVars.wVelocityCoeff[ii].initialize(0.0);
      velocityVars.uVelocityConvectCoeff[ii].initialize(0.0);
      velocityVars.vVelocityConvectCoeff[ii].initialize(0.0);
      velocityVars.wVelocityConvectCoeff[ii].initialize(0.0);
    }

    new_dw->allocateTemporary(velocityVars.uVelLinearSrc,     patch);
    new_dw->allocateTemporary(velocityVars.vVelLinearSrc,     patch);
    new_dw->allocateTemporary(velocityVars.wVelLinearSrc,     patch);

    velocityVars.uVelLinearSrc.initialize(0.0);
    velocityVars.vVelLinearSrc.initialize(0.0);
    velocityVars.wVelLinearSrc.initialize(0.0);

    new_dw->allocateTemporary(velocityVars.uVelNonlinearSrc,  patch);
    new_dw->allocateTemporary(velocityVars.vVelNonlinearSrc,  patch);
    new_dw->allocateTemporary(velocityVars.wVelNonlinearSrc,  patch);

    velocityVars.uVelNonlinearSrc.initialize(0.0);
    velocityVars.vVelNonlinearSrc.initialize(0.0);
    velocityVars.wVelNonlinearSrc.initialize(0.0);

    new_dw->getModifiable(velocityVars.uVelRhoHat, d_lab->d_uVelRhoHatLabel,indx, patch);
    new_dw->getModifiable(velocityVars.vVelRhoHat, d_lab->d_vVelRhoHatLabel,indx, patch);
    new_dw->getModifiable(velocityVars.wVelRhoHat, d_lab->d_wVelRhoHatLabel,indx, patch);

    SFCXVariable<double> conv_scheme_x;
    new_dw->getModifiable(conv_scheme_x, d_lab->d_conv_scheme_x_Label, indx, patch);
    SFCYVariable<double> conv_scheme_y;
    new_dw->getModifiable(conv_scheme_y, d_lab->d_conv_scheme_y_Label, indx, patch);
    SFCZVariable<double> conv_scheme_z;
    new_dw->getModifiable(conv_scheme_z, d_lab->d_conv_scheme_z_Label, indx, patch);

    //This copy is needed for BCs that are reapplied using the
    //extra cell value.
    velocityVars.uVelRhoHat.copy(constVelocityVars.old_uVelocity,
                                 velocityVars.uVelRhoHat.getLowIndex(),
                                 velocityVars.uVelRhoHat.getHighIndex());

    velocityVars.vVelRhoHat.copy(constVelocityVars.old_vVelocity,
                                 velocityVars.vVelRhoHat.getLowIndex(),
                                 velocityVars.vVelRhoHat.getHighIndex());

    velocityVars.wVelRhoHat.copy(constVelocityVars.old_wVelocity,
                                 velocityVars.wVelRhoHat.getLowIndex(),
                                 velocityVars.wVelRhoHat.getHighIndex());

    //__________________________________
    //  compute coefficients and vel src
    d_discretize->calculateVelocityCoeff( patch,
                                          delta_t, d_central,
                                          cellinfo, &velocityVars,
                                          &constVelocityVars, &volFraction, &conv_scheme_x, &conv_scheme_y, &conv_scheme_z, d_conv_scheme,
                                          d_re_limit );

    //  //__________________________________
    //  //  Compute the sources
    //  d_source->calculateVelocitySource( patch,
    //                                     delta_t,
    //                                     cellinfo, &velocityVars,
    //                                     &constVelocityVars);

    //----------------------------------
    // If not doing MPMArches, then need to
    // take of the wall shear stress
    // This adds the contribution to the nonLinearSrc of each
    // direction depending on the boundary conditions.
    // if ( !d_MAlab ){
    //   d_boundaryCondition->wallStress( patch, &velocityVars, &constVelocityVars, volFraction, IsImag );
    // }

    if ( d_wall_closure == "constant_coefficient" ){

      d_boundaryCondition->wallStressConstSmag(
                                                patch,
                                                which_dw,
                                                d_wall_const_smag_C,
                                                d_standoff_index,
                                                constVelocityVars.uVelocity,
                                                constVelocityVars.vVelocity,
                                                constVelocityVars.wVelocity,
                                                velocityVars.uVelNonlinearSrc,
                                                velocityVars.vVelNonlinearSrc,
                                                velocityVars.wVelNonlinearSrc,
                                                constVelocityVars.density,
                                                volFraction
                                              );

    } else if ( d_wall_closure == "dynamic" ){

      d_boundaryCondition->wallStressDynSmag(
                                                patch,
                                                d_standoff_index,
                                                constVelocityVars.viscosity,
                                                constVelocityVars.uVelocity,
                                                constVelocityVars.vVelocity,
                                                constVelocityVars.wVelocity,
                                                velocityVars.uVelNonlinearSrc,
                                                velocityVars.vVelNonlinearSrc,
                                                velocityVars.wVelNonlinearSrc,
                                                constVelocityVars.density,
                                                volFraction
                                              );

    } else if ( d_wall_closure == "molecular" ){

      d_boundaryCondition->wallStressMolecular(
                                                patch,
                                                constVelocityVars.uVelocity,
                                                constVelocityVars.vVelocity,
                                                constVelocityVars.wVelocity,
                                                velocityVars.uVelNonlinearSrc,
                                                velocityVars.vVelNonlinearSrc,
                                                velocityVars.wVelNonlinearSrc,
                                                volFraction
                                              );

    } else if ( d_wall_closure == "log" ){

      d_boundaryCondition->wallStressLog(
                                                patch,
                                                &velocityVars, 
                                                &constVelocityVars,
                                                volFraction
                                              );
    }

    //__________________________________
    // ---- This needs to get moved somewhere else ----
    // for scalesimilarity model add stress tensor to the source of velocity eqn.
    if (d_mixedModel) {
      StencilMatrix<constCCVariable<double> > stressTensor; //9 point tensor
      if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First){
        for (int ii = 0; ii < d_lab->d_tensorMatl->size(); ii++) {
          old_dw->get(stressTensor[ii],
                      d_lab->d_stressTensorCompLabel, ii, patch,
                      gac, 1);
        }
      }else{
        for (int ii = 0; ii < d_lab->d_tensorMatl->size(); ii++) {
          new_dw->get(stressTensor[ii],
                      d_lab->d_stressTensorCompLabel, ii, patch,
                      gac, 1);
        }
      }


      double sue, suw, sun, sus, sut, sub;
      //__________________________________
      //      u velocity
      for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++) {
        IntVector c = *iter;
        int colX = c.x();
        int colY = c.y();
        int colZ = c.z();
        IntVector prevXCell(colX-1, colY, colZ);

        sue = cellinfo->sns[colY]*cellinfo->stb[colZ]*
                     (stressTensor[0])[c];
        suw = cellinfo->sns[colY]*cellinfo->stb[colZ]*
                     (stressTensor[0])[prevXCell];
        sun = 0.25*cellinfo->sew[colX]*cellinfo->stb[colZ]*
                     ((stressTensor[1])[c]+
                      (stressTensor[1])[prevXCell]+
                      (stressTensor[1])[IntVector(colX,colY+1,colZ)]+
                      (stressTensor[1])[IntVector(colX-1,colY+1,colZ)]);
        sus =  0.25*cellinfo->sew[colX]*cellinfo->stb[colZ]*
                     ((stressTensor[1])[c]+
                      (stressTensor[1])[prevXCell]+
                      (stressTensor[1])[IntVector(colX,colY-1,colZ)]+
                      (stressTensor[1])[IntVector(colX-1,colY-1,colZ)]);
        sut = 0.25*cellinfo->sns[colY]*cellinfo->sew[colX]*
                     ((stressTensor[2])[c]+
                      (stressTensor[2])[prevXCell]+
                      (stressTensor[2])[IntVector(colX,colY,colZ+1)]+
                      (stressTensor[2])[IntVector(colX-1,colY,colZ+1)]);
        sub =  0.25*cellinfo->sns[colY]*cellinfo->sew[colX]*
                     ((stressTensor[2])[c]+
                      (stressTensor[2])[prevXCell]+
                      (stressTensor[2])[IntVector(colX,colY,colZ-1)]+
                      (stressTensor[2])[IntVector(colX-1,colY,colZ-1)]);
        velocityVars.uVelNonlinearSrc[c] += suw-sue+sus-sun+sub-sut;
      }

      //__________________________________
      //      v velocity
      for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++) {
        IntVector c = *iter;
        int colX = c.x();
        int colY = c.y();
        int colZ = c.z();
        IntVector prevYCell(colX, colY-1, colZ);

        sue = 0.25*cellinfo->sns[colY]*cellinfo->stb[colZ]*
                   ((stressTensor[3])[c]+
                    (stressTensor[3])[prevYCell]+
                    (stressTensor[3])[IntVector(colX+1,colY,colZ)]+
                    (stressTensor[3])[IntVector(colX+1,colY-1,colZ)]);
        suw =  0.25*cellinfo->sns[colY]*cellinfo->stb[colZ]*
                    ((stressTensor[3])[c]+
                     (stressTensor[3])[prevYCell]+
                     (stressTensor[3])[IntVector(colX-1,colY,colZ)]+
                     (stressTensor[3])[IntVector(colX-1,colY-1,colZ)]);
        sun = cellinfo->sew[colX]*cellinfo->stb[colZ]*
                     (stressTensor[4])[c];
        sus = cellinfo->sew[colX]*cellinfo->stb[colZ]*
                     (stressTensor[4])[prevYCell];
        sut = 0.25*cellinfo->sns[colY]*cellinfo->sew[colX]*
                    ((stressTensor[5])[c]+
                     (stressTensor[5])[prevYCell]+
                     (stressTensor[5])[IntVector(colX,colY,colZ+1)]+
                     (stressTensor[5])[IntVector(colX,colY-1,colZ+1)]);
        sub =  0.25*cellinfo->sns[colY]*cellinfo->sew[colX]*
                    ((stressTensor[5])[c]+
                     (stressTensor[5])[prevYCell]+
                     (stressTensor[5])[IntVector(colX,colY,colZ-1)]+
                     (stressTensor[5])[IntVector(colX,colY-1,colZ-1)]);
        velocityVars.vVelNonlinearSrc[c] += suw-sue+sus-sun+sub-sut;
      }

      //__________________________________
      //       w velocity
      for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++) {
        IntVector c = *iter;
        int colX = c.x();
        int colY = c.y();
        int colZ = c.z();
        IntVector prevZCell(colX, colY, colZ-1);

        sue = 0.25*cellinfo->sns[colY]*cellinfo->stb[colZ]*
                     ((stressTensor[6])[c]+
                      (stressTensor[6])[prevZCell]+
                      (stressTensor[6])[IntVector(colX+1,colY,colZ)]+
                      (stressTensor[6])[IntVector(colX+1,colY,colZ-1)]);
        suw =  0.25*cellinfo->sns[colY]*cellinfo->stb[colZ]*
                     ((stressTensor[6])[c]+
                      (stressTensor[6])[prevZCell]+
                      (stressTensor[6])[IntVector(colX-1,colY,colZ)]+
                      (stressTensor[6])[IntVector(colX-1,colY,colZ-1)]);
        sun = 0.25*cellinfo->sew[colX]*cellinfo->stb[colZ]*
                     ((stressTensor[7])[c]+
                      (stressTensor[7])[prevZCell]+
                      (stressTensor[7])[IntVector(colX,colY+1,colZ)]+
                      (stressTensor[7])[IntVector(colX,colY+1,colZ-1)]);
        sus =  0.25*cellinfo->sew[colX]*cellinfo->stb[colZ]*
                     ((stressTensor[7])[c]+
                      (stressTensor[7])[prevZCell]+
                      (stressTensor[7])[IntVector(colX,colY-1,colZ)]+
                      (stressTensor[7])[IntVector(colX,colY-1,colZ-1)]);
        sut = cellinfo->sew[colX]*cellinfo->sns[colY]*
                     (stressTensor[8])[c];
        sub = cellinfo->sew[colX]*cellinfo->sns[colY]*
                     (stressTensor[8])[prevZCell];
        velocityVars.wVelNonlinearSrc[c] += suw-sue+sus-sun+sub-sut;
      }
    }
    //__________________________________

    // add multimaterial momentum source term
    if (d_MAlab){
      d_source->computemmMomentumSource(pc, patch, cellinfo,
                                        &velocityVars, &constVelocityVars);
    }

    // Adding new sources from factory:
    SourceTermFactory& factory = SourceTermFactory::self();
    for ( auto iter = d_new_sources.begin(); iter != d_new_sources.end(); iter++){

      if ( factory.source_term_exists( *iter ) ){
        SourceTermBase& src = factory.retrieve_source_term( *iter );
        SourceTermBase::MY_GRID_TYPE src_type = src.getSourceGridType();

        switch (src_type) {
          case SourceTermBase::CCVECTOR_SRC:
            { new_dw->get( velocityVars.otherVectorSource, VarLabel::find( *iter ), indx, patch, Ghost::AroundCells, 1);
            Vector Dx  = patch->dCell();
            double vol = Dx.x()*Dx.y()*Dx.z();

            Uintah::BlockRange range(patch->getCellLowIndex(),patch->getCellHighIndex());
            sumNonlinearSources doSumSrc(vol,velocityVars.uVelNonlinearSrc,
                                             velocityVars.vVelNonlinearSrc,
                                             velocityVars.wVelNonlinearSrc,
                                             velocityVars.otherVectorSource);
            Uintah::parallel_for( range, doSumSrc);
            }
            break;
          case SourceTermBase::FX_SRC:
            { new_dw->get( velocityVars.otherFxSource, VarLabel::find( *iter ), indx, patch, Ghost::None, 0);
            Vector Dx  = patch->dCell();
            double vol = Dx.x()*Dx.y()*Dx.z();
            for (CellIterator iter=patch->getSFCXIterator(); !iter.done(); iter++){
              IntVector c = *iter;
              velocityVars.uVelNonlinearSrc[c]  += velocityVars.otherFxSource[c]*vol;
            }}
            break;
          case SourceTermBase::FY_SRC:
            { new_dw->get( velocityVars.otherFySource, VarLabel::find( *iter ), indx, patch, Ghost::None, 0);
            Vector Dx  = patch->dCell();
            double vol = Dx.x()*Dx.y()*Dx.z();
            for (CellIterator iter=patch->getSFCYIterator(); !iter.done(); iter++){
              IntVector c = *iter;
              velocityVars.vVelNonlinearSrc[c]  += velocityVars.otherFySource[c]*vol;
            }}
            break;
          case SourceTermBase::FZ_SRC:
            { new_dw->get( velocityVars.otherFzSource, VarLabel::find( *iter ), indx, patch, Ghost::None, 0);
            Vector Dx  = patch->dCell();
            double vol = Dx.x()*Dx.y()*Dx.z();
            for (CellIterator iter=patch->getSFCZIterator(); !iter.done(); iter++){
              IntVector c = *iter;
              velocityVars.wVelNonlinearSrc[c]  += velocityVars.otherFzSource[c]*vol;
            }}
            break;
          default:
            proc0cout << "For source term of type: " << src_type << endl;
            throw InvalidValue("Error: Trying to add a source term to momentum equation with incompatible type",__FILE__, __LINE__);

        }
      } else {

        // **HACK** Look for it in the turbulence model factory.
        // Error checking for non-existant sources was performed in problemSetup.
        // !!!!! HARD CODED FOR CCVARIABLE<VECTOR> !!!!!
        new_dw->get( velocityVars.otherVectorSource, VarLabel::find( *iter ), indx, patch, Ghost::AroundCells, 1);
        Vector Dx  = patch->dCell();
        double vol = Dx.x()*Dx.y()*Dx.z();

        Uintah::BlockRange range(patch->getCellLowIndex(),patch->getCellHighIndex());
        sumNonlinearSources doSumSrc(vol,velocityVars.uVelNonlinearSrc,
                                         velocityVars.vVelNonlinearSrc,
                                         velocityVars.wVelNonlinearSrc,
                                         velocityVars.otherVectorSource);
        Uintah::parallel_for( range, doSumSrc);

      }
    }

     // sets coefs in the direction of the wall to zero
    //  d_boundaryCondition->wallVelocityBC(patch, cellinfo,
    //                                    &velocityVars, &constVelocityVars);

    //  d_source->modifyVelMassSource(patch, volFraction,
    //                                &velocityVars, &constVelocityVars);

    d_discretize->calculateVelDiagonal(patch,&velocityVars);

    if (d_MAlab) {
      d_boundaryCondition->calculateVelRhoHat_mm(patch, delta_t,
                                                 cellinfo, &velocityVars,
                                                 &constVelocityVars);
    } else {
      d_rhsSolver->calculateHatVelocity(patch, delta_t,
                                       cellinfo, &velocityVars, &constVelocityVars, volFraction);
    }

    d_boundaryCondition->setHattedIntrusionVelocity( patch, velocityVars.uVelRhoHat,
                                                    velocityVars.vVelRhoHat, velocityVars.wVelRhoHat, constVelocityVars.new_density );



    double time_shift = 0.0;
    time_shift = delta_t * timelabels->time_position_multiplier_before_average;
    d_boundaryCondition->velRhoHatInletBC(patch,
                                          &velocityVars, &constVelocityVars,
                                          indx,
                                          time_shift);

    d_boundaryCondition->velocityOutletPressureBC( patch,
                                                        indx,
                                                        velocityVars.uVelRhoHat,
                                                        velocityVars.vVelRhoHat,
                                                        velocityVars.wVelRhoHat,
                                                        constVelocityVars.old_uVelocity,
                                                        constVelocityVars.old_vVelocity,
                                                        constVelocityVars.old_wVelocity );

  }
}

//****************************************************************************
// Schedule the averaging of hat velocities for Runge-Kutta step
//****************************************************************************
void
MomentumSolver::sched_averageRKHatVelocities(SchedulerP& sched,
                                             const PatchSet* patches,
                                             const MaterialSet* matls,
                                             const TimeIntegratorLabel* timelabels)
{
  string taskname =  "MomentumSolver::averageRKHatVelocities" +
                     timelabels->integrator_step_name;
  Task* tsk = scinew Task(taskname, this,
                          &MomentumSolver::averageRKHatVelocities,
                          timelabels);

  Ghost::GhostType  gn = Ghost::None;
  Ghost::GhostType  gac = Ghost::AroundCells;

  tsk->requires(Task::OldDW , d_lab->d_uVelocitySPBCLabel , gn  , 0);
  tsk->requires(Task::OldDW , d_lab->d_vVelocitySPBCLabel , gn  , 0);
  tsk->requires(Task::OldDW , d_lab->d_wVelocitySPBCLabel , gn  , 0);

  tsk->requires(Task::OldDW , d_lab->d_densityCPLabel     , gac , 1);
  tsk->requires(Task::NewDW , d_lab->d_cellTypeLabel      , gac , 1);

  tsk->requires(Task::NewDW , d_lab->d_densityTempLabel   , gac , 1);
  tsk->requires(Task::NewDW , d_lab->d_densityCPLabel     , gac , 1);

  tsk->modifies(d_lab->d_uVelRhoHatLabel);
  tsk->modifies(d_lab->d_vVelRhoHatLabel);
  tsk->modifies(d_lab->d_wVelRhoHatLabel);

  sched->addTask(tsk, patches, matls);
}


//****************************************************************************
// Actually average the Runge-Kutta hat velocities here
//****************************************************************************
void
MomentumSolver::averageRKHatVelocities(const ProcessorGroup*,
                                       const PatchSubset* patches,
                                       const MaterialSubset*,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw,
                                       const TimeIntegratorLabel* timelabels)
{
  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int indx = d_lab->d_sharedState->
                     getArchesMaterial(archIndex)->getDWIndex();

    constCCVariable<double> old_density;
    constCCVariable<double> temp_density;
    constCCVariable<double> new_density;
    constCCVariable<int> cellType;
    constSFCXVariable<double> old_uvel;
    constSFCYVariable<double> old_vvel;
    constSFCZVariable<double> old_wvel;
    SFCXVariable<double> new_uvel;
    SFCYVariable<double> new_vvel;
    SFCZVariable<double> new_wvel;

    Ghost::GhostType  gn = Ghost::None;
    Ghost::GhostType  gac = Ghost::AroundCells;

    old_dw->get(old_density, d_lab->d_densityCPLabel, indx, patch, gac, 1);
    new_dw->get(cellType,    d_lab->d_cellTypeLabel,  indx, patch, gac, 1);

    old_dw->get(old_uvel, d_lab->d_uVelocitySPBCLabel, indx, patch, gn, 0);
    old_dw->get(old_vvel, d_lab->d_vVelocitySPBCLabel, indx, patch, gn, 0);
    old_dw->get(old_wvel, d_lab->d_wVelocitySPBCLabel, indx, patch, gn, 0);

    new_dw->get(temp_density, d_lab->d_densityTempLabel, indx, patch, gac,1);
    new_dw->get(new_density, d_lab->d_densityCPLabel,    indx, patch, gac,1);

    new_dw->getModifiable(new_uvel, d_lab->d_uVelRhoHatLabel, indx, patch);
    new_dw->getModifiable(new_vvel, d_lab->d_vVelRhoHatLabel, indx, patch);
    new_dw->getModifiable(new_wvel, d_lab->d_wVelRhoHatLabel, indx, patch);

    double factor_old, factor_new, factor_divide;
    factor_old = timelabels->factor_old;
    factor_new = timelabels->factor_new;
    factor_divide = timelabels->factor_divide;

    IntVector x_offset(1,0,0);
    IntVector y_offset(0,1,0);
    IntVector z_offset(0,0,1);

    //At one point we tried using vol fraction here but that
    //causes problems with inlets because the volume fraction there
    //is 1.0.  We don't want the inlet velocities to be changed
    //otherwise we lose the boundary condition information.

    //__________________________________
    //  X  (This includes the extra cells)
    CellIterator SFCX_iter = patch->getSFCXIterator();

    const int flow = -1;

    for(; !SFCX_iter.done(); SFCX_iter++) {
      IntVector c = *SFCX_iter;
      IntVector L = c - x_offset;
      if ( cellType[c] == flow && cellType[L] == flow ){
        new_uvel[c] = (factor_old * old_uvel[c] * (old_density[c]  + old_density[L])
                    +  factor_new * new_uvel[c] * (temp_density[c] + temp_density[L]))/
                       (factor_divide * (new_density[c] + new_density[L]));
      }
    }
    //__________________________________
    // Y  (This includes the extra cells)
    CellIterator SFCY_iter = patch->getSFCZIterator();

    for(; !SFCY_iter.done(); SFCY_iter++) {
      IntVector c = *SFCY_iter;
      IntVector L = c - y_offset;
      if ( cellType[c] == flow && cellType[L] == flow ){
        new_vvel[c] = (factor_old * old_vvel[c] * (old_density[c]  + old_density[L])
                    +  factor_new * new_vvel[c] * (temp_density[c] + temp_density[L]))/
                       (factor_divide * (new_density[c] + new_density[L]));
      }
    }
    //__________________________________
    // Z  (This includes the extra cells)
    CellIterator SFCZ_iter = patch->getSFCZIterator();

    for(; !SFCZ_iter.done(); SFCZ_iter++) {
      IntVector c = *SFCZ_iter;
      IntVector L = c - z_offset;
      if ( cellType[c] == flow && cellType[L] == flow ){
        new_wvel[c] = (factor_old * old_wvel[c] * (old_density[c]  + old_density[L])
                    +  factor_new * new_wvel[c] * (temp_density[c] + temp_density[L]))/
                       (factor_divide * (new_density[c] + new_density[L]));
      }
    }

    //__________________________________
    // Apply boundary conditions
    d_boundaryCondition->velocityOutletPressureBC( patch,
                                                        indx,
                                                        new_uvel, new_vvel, new_wvel,
                                                        old_uvel, old_vvel, old_wvel );

    d_boundaryCondition->setHattedIntrusionVelocity( patch, new_uvel, new_vvel, new_wvel, new_density );

  }  // patches
}

//
//------------------------------------------------------------------------------------------------

void
MomentumSolver::sched_computeMomentum( const LevelP& level,
                                         SchedulerP& sched,
                                         const int timesubstep,
                                         const bool isInitialization )
{

  Task* tsk = scinew Task( "MomentumSolver::computeMomentum", this,
                            &MomentumSolver::computeMomentum, timesubstep, isInitialization );

  Task::WhichDW which_dw;
  if ( timesubstep == 0 ){
    tsk->computes( _u_mom );
    tsk->computes( _v_mom );
    tsk->computes( _w_mom );
    which_dw = Task::NewDW;
    //which_dw = isInitialization ? Task::NewDW : Task::OldDW;
  } else {
    tsk->modifies( _u_mom );
    tsk->modifies( _v_mom );
    tsk->modifies( _w_mom );
    which_dw = Task::NewDW;
  }

  tsk->requires( which_dw, d_lab->d_uVelocitySPBCLabel, Ghost::AroundFaces, 1 );
  tsk->requires( which_dw, d_lab->d_vVelocitySPBCLabel, Ghost::AroundFaces, 1 );
  tsk->requires( which_dw, d_lab->d_wVelocitySPBCLabel, Ghost::AroundFaces, 1 );
  tsk->requires( which_dw, d_lab->d_densityCPLabel, Ghost::AroundCells, 1 );

  sched->addTask( tsk, level->eachPatch(), d_lab->d_sharedState->allArchesMaterials() );

}

//
//------------------------------------------------------------------------------------------------

void MomentumSolver::computeMomentum( const ProcessorGroup* pc,
                                        const PatchSubset* patches,
                                        const MaterialSubset*,
                                        DataWarehouse* old_dw,
                                        DataWarehouse* new_dw,
                                        const int timesubstep,
                                        const bool isInitialization )
{
  //patch loop
  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex();

    DataWarehouse* which_dw;

    SFCXVariable<double> u_mom;
    SFCYVariable<double> v_mom;
    SFCZVariable<double> w_mom;

    constCCVariable<double> rho;
    constSFCXVariable<double> u;
    constSFCYVariable<double> v;
    constSFCZVariable<double> w;

    if ( timesubstep == 0 ){
      //which_dw = isInitialization ? new_dw : old_dw;
      which_dw = new_dw;
      new_dw->allocateAndPut( u_mom, _u_mom, matlIndex, patch );
      new_dw->allocateAndPut( v_mom, _v_mom, matlIndex, patch );
      new_dw->allocateAndPut( w_mom, _w_mom, matlIndex, patch );
      // CAREFUL HERE - WE MAY NEED TO INITIALIZE!
      u_mom.initialize(0.0);
      v_mom.initialize(0.0);
      w_mom.initialize(0.0);
    } else {
      which_dw = new_dw;
      new_dw->getModifiable( u_mom, _u_mom, matlIndex, patch );
      new_dw->getModifiable( v_mom, _v_mom, matlIndex, patch );
      new_dw->getModifiable( w_mom, _w_mom, matlIndex, patch );
    }

    which_dw->get( u, d_lab->d_uVelocitySPBCLabel, matlIndex, patch, Ghost::AroundFaces, 1 );
    which_dw->get( v, d_lab->d_vVelocitySPBCLabel, matlIndex, patch, Ghost::AroundFaces, 1 );
    which_dw->get( w, d_lab->d_wVelocitySPBCLabel, matlIndex, patch, Ghost::AroundFaces, 1 );
    which_dw->get( rho, d_lab->d_densityCPLabel,   matlIndex, patch, Ghost::AroundCells, 1 );

    IntVector lo = patch->getExtraCellLowIndex();
    IntVector hi = patch->getExtraCellHighIndex();

    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){

      IntVector c = *iter;

      IntVector west = c - IntVector(1,0,0);
      IntVector south = c - IntVector(0,1,0);
      IntVector bottom = c - IntVector(0,0,1);
      u_mom[c] = 0.5 * (rho[c] + rho[west]) * u[c];
      v_mom[c] = 0.5 * (rho[c] + rho[south]) * v[c];
      w_mom[c] = 0.5 * (rho[c] + rho[bottom]) * w[c];
    }


    //__________________________________
    // set values in extra cells
    vector<Patch::FaceType> b_face;
    patch->getBoundaryFaces(b_face);
    vector<Patch::FaceType>::const_iterator itr;

    // Loop over boundary faces
    for( itr = b_face.begin(); itr != b_face.end(); ++itr ){
      Patch::FaceType face = *itr;

      IntVector unitNormal = patch->getFaceDirection(face);

      Patch::FaceIteratorType MEC = Patch::ExtraMinusEdgeCells;
      CellIterator iter=patch->getFaceIterator(face, MEC);

      switch (face) {
        case Patch::xminus:
          for(;!iter.done();iter++){
            IntVector c = *iter;
            IntVector shiftX = c;
            IntVector shiftY = c - IntVector(0,1,0);
            IntVector shiftZ = c - IntVector(0,0,1);

            u_mom[c] = rho[c] * u[c];
            v_mom[c] = 0.5 * (rho[c] + rho[shiftY]) * v[c];
            w_mom[c] = 0.5 * (rho[c] + rho[shiftZ]) * w[c];
          }
          break;
        case Patch::xplus:
          for(;!iter.done();iter++){
            IntVector c = *iter;
            IntVector shiftX = c + IntVector(1,0,0);
            IntVector shiftY = c + IntVector(0,1,0);
            IntVector shiftZ = c + IntVector(0,0,1);

            u_mom[c]      = 0.5 * (rho[c] + rho[c - IntVector(1,0,0)]) * u[c];
            u_mom[shiftX] = rho[c] * u[c];
            v_mom[c] = 0.5 * (rho[c] + rho[shiftY]) * v[c];
            w_mom[c] = 0.5 * (rho[c] + rho[shiftZ]) * w[c];
          }
          break;
        case Patch::yminus:
          for(;!iter.done();iter++){
            IntVector c = *iter;

            IntVector shiftX = c - IntVector(1,0,0);
            IntVector shiftY = c;
            IntVector shiftZ = c - IntVector(0,0,1);

            u_mom[c] = 0.5 * (rho[c] + rho[shiftX]) * u[c];
            v_mom[c] = rho[c] * v[c];
            w_mom[c] = 0.5 * (rho[c] + rho[shiftZ]) * w[c];
          }
          break;
        case Patch::yplus:
          for(;!iter.done();iter++){
            IntVector c = *iter;

            IntVector shiftX = c + IntVector(1,0,0);
            IntVector shiftY = c + IntVector(0,1,0);
            IntVector shiftZ = c + IntVector(0,0,1);

            u_mom[c] = 0.5 * (rho[c] + rho[shiftX]) * u[c];
            v_mom[c] = 0.5 * (rho[c] + rho[c - IntVector(0,1,0)]) * v[c];
            v_mom[shiftY] = rho[c] * v[c];
            w_mom[c] = 0.5 * (rho[c] + rho[shiftZ]) * w[c];
          }
          break;

        case Patch::zminus:
          for(;!iter.done();iter++){
            IntVector c = *iter;

            IntVector shiftX = c - IntVector(1,0,0);
            IntVector shiftY = c - IntVector(0,1,0);
            IntVector shiftZ = c;

            u_mom[c] = 0.5 * (rho[c] + rho[shiftX]) * u[c];
            v_mom[c] = 0.5 * (rho[c] + rho[shiftY]) * v[c];
            w_mom[c] = rho[c] * w[c];
          }
          break;
        case Patch::zplus:
          for(;!iter.done();iter++){
            IntVector c = *iter;

            IntVector shiftX = c + IntVector(1,0,0);
            IntVector shiftY = c + IntVector(0,1,0);
            IntVector shiftZ = c + IntVector(0,0,1);

            u_mom[c] = 0.5 * (rho[c] + rho[shiftX]) * u[c];
            v_mom[c] = 0.5 * (rho[c] + rho[shiftY]) * v[c];
            w_mom[c] = 0.5 * (rho[c] + rho[c - IntVector(0,0,1)]) * w[c];
            w_mom[shiftZ] = rho[c] * w[c];
          }
          break;


        default:
          break;
      }
    }

  }
}

//
//------------------------------------------------------------------------------
