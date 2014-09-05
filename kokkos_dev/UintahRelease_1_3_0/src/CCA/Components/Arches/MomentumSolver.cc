/*

The MIT License

Copyright (c) 1997-2010 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


//----- MomentumSolver.cc ----------------------------------------------

#include <TauProfilerForSCIRun.h>
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
#include <CCA/Components/Arches/OdtClosure.h>
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
               PhysicalConstants* physConst) : 
                                   d_lab(label), d_MAlab(MAlb),
                                   d_turbModel(turb_model), 
                                   d_boundaryCondition(bndry_cond),
                                   d_physicalConsts(physConst)
{
  d_discretize = 0;
  d_source = 0;
  d_rhsSolver = 0;
}

//****************************************************************************
// Destructor
//****************************************************************************
MomentumSolver::~MomentumSolver()
{
  delete d_discretize;
  delete d_source;
  delete d_rhsSolver;
}

//****************************************************************************
// Problem Setup 
//****************************************************************************
void 
MomentumSolver::problemSetup(const ProblemSpecP& params)
{
  ProblemSpecP db = params->findBlock("MomentumSolver");

  d_discretize = scinew Discretization();

  string conv_scheme;
  db->getWithDefault("convection_scheme",conv_scheme,"upwind");
  if (conv_scheme == "upwind"){
    d_central = false;
  }else if (conv_scheme == "central"){
    d_central = true;     
  }else{
    throw InvalidValue("Convection scheme not supported: " + conv_scheme, __FILE__, __LINE__);
  }
  
  db->getWithDefault("pressure_correction",         d_pressure_correction,false);
  db->getWithDefault("filter_divergence_constraint",d_filter_divergence_constraint,false);

  d_source = scinew Source(d_physicalConsts);
  
  d_discretize->setMMS(d_doMMS);
  if (d_doMMS){
    d_source->problemSetup(db);
  }
// ++ jeremy ++
  d_source->setBoundary(d_boundaryCondition);
// -- jeremy --            

  // New Source terms (ala the new transport eqn):
  if (db->findBlock("src")){
    string srcname; 
    for (ProblemSpecP src_db = db->findBlock("src"); src_db != 0; src_db = src_db->findNextBlock("src")){
      src_db->getAttribute("label", srcname);
      //which sources are turned on for this equation
      d_new_sources.push_back( srcname ); 

    }
  }

  d_rhsSolver = scinew RHSSolver();
  d_rhsSolver->setMMS(d_doMMS);

  d_mixedModel=d_turbModel->getMixedModel();
}

//****************************************************************************
// Schedule linear momentum solve
//****************************************************************************
void 
MomentumSolver::solve(SchedulerP& sched,
                      const PatchSet* patches,
                      const MaterialSet* matls,
                      const TimeIntegratorLabel* timelabels,
                      bool extraProjection,
                      bool doing_EKT_now)
{
  //computes stencil coefficients and source terms
  // require : pressureCPBC, [u,v,w]VelocityCPBC, densityIN, viscosityIN (new_dw)
  //           [u,v,w]SPBC, densityCP (old_dw)
  // compute : [u,v,w]VelCoefPBLM, [u,v,w]VelConvCoefPBLM
  //           [u,v,w]VelLinSrcPBLM, [u,v,w]VelNonLinSrcPBLM

  sched_buildLinearMatrix(sched, patches, matls, timelabels,
                          extraProjection, doing_EKT_now);
    

}

//****************************************************************************
// Schedule the build of the linear matrix
//****************************************************************************
void 
MomentumSolver::sched_buildLinearMatrix(SchedulerP& sched,
                                        const PatchSet* patches,
                                        const MaterialSet* matls,
                                        const TimeIntegratorLabel* timelabels,
                                        bool extraProjection,
                                        bool doing_EKT_now)
{
  string taskname =  "MomentumSolver::BuildCoeff" +
                     timelabels->integrator_step_name;
  if (extraProjection){
    taskname += "extraProjection";
  }
  if (doing_EKT_now){
    taskname += "EKTnow";
  }
  Task* tsk = scinew Task(taskname,
                          this, &MomentumSolver::buildLinearMatrix,
                          timelabels, extraProjection, doing_EKT_now);

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
  
  tsk->requires(Task::NewDW, d_lab->d_cellInfoLabel, Ghost::None);
  if ((extraProjection)||(doing_EKT_now)){
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
  
  if ((doing_EKT_now)&&(timelabels->integrator_step_name == "Predictor")){
    tsk->computes(d_lab->d_uVelocityEKTLabel);
    tsk->computes(d_lab->d_vVelocityEKTLabel);
    tsk->computes(d_lab->d_wVelocityEKTLabel);
  }

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
                                  bool extraProjection,
                                  bool doing_EKT_now)
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
    new_dw->get(constVelocityVars.cellType, d_lab->d_cellTypeLabel,  indx, patch, gac, 1);
    new_dw->get(constVelocityVars.density,  d_lab->d_densityCPLabel, indx, patch, gac, 1);

    if ((extraProjection)||(doing_EKT_now)){
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
    // Huh?  -Todd
    new_dw->copyOut(velocityVars.uVelRhoHat,       d_lab->d_uVelRhoHatLabel,   indx, patch);  
    new_dw->copyOut(velocityVars.vVelRhoHat,       d_lab->d_vVelRhoHatLabel,   indx, patch);  
    new_dw->copyOut(velocityVars.wVelRhoHat,       d_lab->d_wVelRhoHatLabel,   indx, patch);  

    if (d_MAlab) {
      d_boundaryCondition->calculateVelocityPred_mm(patch, delta_t, 
                                                    cellinfo,&velocityVars, &constVelocityVars);

    }else {
      d_rhsSolver->calculateVelocity(patch, delta_t, cellinfo, &velocityVars,
                                     constVelocityVars.density, constVelocityVars.pressure);

      /*if (d_boundaryCondition->getIntrusionBC())
        d_boundaryCondition->calculateIntrusionVel(pc, patch,
                                                   index, cellinfo,
                                                   &velocityVars,
                                                   &constVelocityVars);*/
    }
    
    // boundary condition
    if ((d_boundaryCondition->getOutletBC())||(d_boundaryCondition->getPressureBC())){
      d_boundaryCondition->addPresGradVelocityOutletPressureBC(patch, cellinfo,
                                                                delta_t, &velocityVars,
                                                                &constVelocityVars);
                                                                
      d_boundaryCondition->velocityOutletPressureTangentBC(patch, 
                                            &velocityVars, &constVelocityVars);
    }

    SFCXVariable<double> uVel_EKT;
    SFCYVariable<double> vVel_EKT;
    SFCZVariable<double> wVel_EKT;
    if ((doing_EKT_now)&&(timelabels->integrator_step_name == "Predictor")){
      new_dw->allocateAndPut(uVel_EKT, d_lab->d_uVelocityEKTLabel, indx, patch);
      new_dw->allocateAndPut(vVel_EKT, d_lab->d_vVelocityEKTLabel, indx, patch);
      new_dw->allocateAndPut(wVel_EKT, d_lab->d_wVelocityEKTLabel, indx, patch);
      uVel_EKT.copyData(velocityVars.uVelRhoHat);
      vVel_EKT.copyData(velocityVars.vVelRhoHat);
      wVel_EKT.copyData(velocityVars.wVelRhoHat);
    }
  }
}

//****************************************************************************
// Schedule calculation of hat velocities
//****************************************************************************
void MomentumSolver::solveVelHat(const LevelP& level,
                                 SchedulerP& sched,
                                 const TimeIntegratorLabel* timelabels,
                                 bool d_EKTCorrection)
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
  for (vector<std::string>::iterator iter = d_new_sources.begin(); 
      iter != d_new_sources.end(); iter++){
    SourceTermBase& src = factory.retrieve_source_term( *iter ); 
    src.sched_computeSource( level, sched, timeSubStep ); 
  }

  sched_buildLinearMatrixVelHat(sched, patches, matls, 
                                timelabels, d_EKTCorrection);

}



// ****************************************************************************
// Schedule build of linear matrix
// ****************************************************************************
void 
MomentumSolver::sched_buildLinearMatrixVelHat(SchedulerP& sched,
                                              const PatchSet* patches,
                                              const MaterialSet* matls,
                                              const TimeIntegratorLabel* timelabels,
                                              bool d_EKTCorrection)
{
  string taskname =  "MomentumSolver::BuildCoeffVelHat" +
                     timelabels->integrator_step_name;
  Task* tsk = scinew Task(taskname, 
                          this, &MomentumSolver::buildLinearMatrixVelHat,
                          timelabels, d_EKTCorrection);

  
  Task::WhichDW parent_old_dw;
  if (timelabels->recursion){
    parent_old_dw = Task::ParentOldDW;
  }else{
    parent_old_dw = Task::OldDW;
  }
  
  tsk->requires(parent_old_dw, d_lab->d_sharedState->get_delt_label());
    
  // Requires
  // from old_dw for time integration
  // get old_dw from getTop function

  Ghost::GhostType  gn = Ghost::None;
  Ghost::GhostType  gac = Ghost::AroundCells;
  Ghost::GhostType  gaf = Ghost::AroundFaces;
  Task::DomainSpec oams = Task::OutOfDomain;  //outside of arches matlSet.
  
  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel, gac, 2);
  tsk->requires(Task::NewDW, d_lab->d_cellInfoLabel, gn);

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

  if (d_EKTCorrection){
    old_values_dw = Task::NewDW;
  }

  tsk->requires(Task::NewDW, d_lab->d_densityCPLabel,     gac, 1);
  tsk->requires(Task::NewDW, d_lab->d_denRefArrayLabel,   gac, 1);
  tsk->requires(Task::NewDW, d_lab->d_viscosityCTSLabel,  gac, 2);
  tsk->requires(Task::NewDW, d_lab->d_uVelocitySPBCLabel, gaf, 2);
  tsk->requires(Task::NewDW, d_lab->d_vVelocitySPBCLabel, gaf, 2);
  tsk->requires(Task::NewDW, d_lab->d_wVelocitySPBCLabel, gaf, 2);

  if (d_pressure_correction){
    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First){
      tsk->requires(Task::OldDW, timelabels->pressure_guess, gac, 1);
    }else{
      tsk->requires(Task::NewDW, timelabels->pressure_guess, gac, 1);
    }
  }
  
  // required for computing div constraint
//#ifdef divergenceconstraint
  if (timelabels->multiple_steps){
    tsk->requires(Task::NewDW, d_lab->d_scalarTempLabel,    gac, 1);
  }else{
    tsk->requires(Task::OldDW, d_lab->d_scalarSPLabel,      gac, 1);
  }
  tsk->requires(Task::OldDW, d_lab->d_divConstraintLabel,   gn, 0);
  tsk->requires(Task::NewDW, d_lab->d_drhodfCPLabel,        gn, 0);
  tsk->requires(Task::NewDW, d_lab->d_scalDiffCoefLabel, 
                             d_lab->d_stencilMatl, oams,    gn, 0);
  tsk->requires(Task::NewDW, d_lab->d_scalDiffCoefSrcLabel, gn, 0);
//#endif

  if ((dynamic_cast<const OdtClosure*>(d_turbModel))||d_mixedModel) {
    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First){
      tsk->requires(Task::OldDW, d_lab->d_stressTensorCompLabel,
                                d_lab->d_tensorMatl,  oams,   gac, 1);
   }else {
      tsk->requires(Task::NewDW, d_lab->d_stressTensorCompLabel,
                                d_lab->d_tensorMatl,  oams,   gac, 1);
    }
  }

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
    
//#ifdef divergenceconstraint
  if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First){
    tsk->computes(d_lab->d_divConstraintLabel);
  }else{
    tsk->modifies(d_lab->d_divConstraintLabel);
  }
//#endif
 // build linear matrix vel hat 
  tsk->modifies(d_lab->d_umomBoundarySrcLabel);
  tsk->modifies(d_lab->d_vmomBoundarySrcLabel);
  tsk->modifies(d_lab->d_wmomBoundarySrcLabel);

  if (d_doMMS) {
    tsk->modifies(d_lab->d_uFmmsLabel);
    tsk->modifies(d_lab->d_vFmmsLabel);
    tsk->modifies(d_lab->d_wFmmsLabel);
  }

  // Adding new sources from factory:
  SourceTermFactory& factor = SourceTermFactory::self(); 
  for (vector<std::string>::iterator iter = d_new_sources.begin(); 
      iter != d_new_sources.end(); iter++){

    SourceTermBase& src = factor.retrieve_source_term( *iter ); 
    const VarLabel* srcLabel = src.getSrcLabel(); 
    tsk->requires(Task::NewDW, srcLabel, gn, 0); 

  }

  sched->addTask(tsk, patches, matls);
}

 


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
                                        bool d_EKTCorrection)
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
  double time=d_lab->d_sharedState->getElapsedTime();
          
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

    if (d_EKTCorrection){
      old_values_dw = new_dw;
    }
    old_values_dw->get(constVelocityVars.old_uVelocity, d_lab->d_uVelocitySPBCLabel, indx, patch, gn, 0);
    old_values_dw->get(constVelocityVars.old_vVelocity, d_lab->d_vVelocitySPBCLabel, indx, patch, gn, 0);
    old_values_dw->get(constVelocityVars.old_wVelocity, d_lab->d_wVelocitySPBCLabel, indx, patch, gn, 0);

    new_dw->get(constVelocityVars.new_density, d_lab->d_densityCPLabel,     indx, patch, gac, 1);
    new_dw->get(constVelocityVars.denRefArray, d_lab->d_denRefArrayLabel,   indx, patch, gac, 1);
    new_dw->get(constVelocityVars.viscosity,   d_lab->d_viscosityCTSLabel,  indx, patch, gac, 2);
    new_dw->get(constVelocityVars.uVelocity,   d_lab->d_uVelocitySPBCLabel, indx, patch, gaf, 2);
    new_dw->get(constVelocityVars.vVelocity,   d_lab->d_vVelocitySPBCLabel, indx, patch, gaf, 2);
    new_dw->get(constVelocityVars.wVelocity,   d_lab->d_wVelocitySPBCLabel, indx, patch, gaf, 2);

    if (d_pressure_correction){
      if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First){
        old_dw->get(constVelocityVars.pressure, timelabels->pressure_guess, indx, patch, gac, 1);
      }else{
        new_dw->get(constVelocityVars.pressure, timelabels->pressure_guess, indx, patch, gac, 1);
      }
    }

//#ifdef divergenceconstraint
    if (timelabels->multiple_steps){
      new_dw->get(constVelocityVars.scalar, d_lab->d_scalarTempLabel,indx, patch, gac, 1);
    }else{
      old_dw->get(constVelocityVars.scalar, d_lab->d_scalarSPLabel,  indx, patch, gac, 1);
    }
    constCCVariable<double> old_divergence;
    old_dw->get(old_divergence,             d_lab->d_divConstraintLabel,indx, patch, gn, 0);
    new_dw->get(constVelocityVars.drhodf,   d_lab->d_drhodfCPLabel,     indx, patch, gn, 0);
    
    for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++){
      new_dw->get(constVelocityVars.scalarDiffusionCoeff[ii],
                  d_lab->d_scalDiffCoefLabel,ii, patch, gn, 0);
    }
                  
    new_dw->get(constVelocityVars.scalarDiffNonlinearSrc, 
                d_lab->d_scalDiffCoefSrcLabel, indx, patch,gn, 0);
                
    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First){
      new_dw->allocateAndPut(velocityVars.divergence,
                             d_lab->d_divConstraintLabel, indx, patch);
    }else{
      new_dw->getModifiable(velocityVars.divergence,
                             d_lab->d_divConstraintLabel, indx, patch);
    }
    velocityVars.divergence.initialize(0.0);
    //#endif    

 // boundary source terms 
    new_dw->getModifiable(velocityVars.umomBoundarySrc,
                                             d_lab->d_umomBoundarySrcLabel, indx, patch);
    new_dw->getModifiable(velocityVars.vmomBoundarySrc,
                                             d_lab->d_vmomBoundarySrcLabel, indx, patch);
    new_dw->getModifiable(velocityVars.wmomBoundarySrc,
                                             d_lab->d_wmomBoundarySrcLabel, indx, patch);

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
    }
   
    new_dw->allocateTemporary(velocityVars.uVelLinearSrc,     patch);
    new_dw->allocateTemporary(velocityVars.vVelLinearSrc,     patch);
    new_dw->allocateTemporary(velocityVars.wVelLinearSrc,     patch);
    
    new_dw->allocateTemporary(velocityVars.uVelNonlinearSrc,  patch);
    new_dw->allocateTemporary(velocityVars.vVelNonlinearSrc,  patch);
    new_dw->allocateTemporary(velocityVars.wVelNonlinearSrc,  patch);
     
    new_dw->getModifiable(velocityVars.uVelRhoHat, d_lab->d_uVelRhoHatLabel,indx, patch);
    new_dw->getModifiable(velocityVars.vVelRhoHat, d_lab->d_vVelRhoHatLabel,indx, patch);
    new_dw->getModifiable(velocityVars.wVelRhoHat, d_lab->d_wVelRhoHatLabel,indx, patch);
    
    velocityVars.uVelRhoHat.copy(constVelocityVars.old_uVelocity,
                                 velocityVars.uVelRhoHat.getLowIndex(),
                                 velocityVars.uVelRhoHat.getHighIndex());
                                 
    velocityVars.vVelRhoHat.copy(constVelocityVars.old_vVelocity,
                                 velocityVars.vVelRhoHat.getLowIndex(),
                                 velocityVars.vVelRhoHat.getHighIndex());
                                 
    velocityVars.wVelRhoHat.copy(constVelocityVars.old_wVelocity,
                                 velocityVars.wVelRhoHat.getLowIndex(),
                                 velocityVars.wVelRhoHat.getHighIndex());

    if (d_doMMS){
      new_dw->getModifiable(velocityVars.uFmms, d_lab->d_uFmmsLabel, indx, patch);
      new_dw->getModifiable(velocityVars.vFmms, d_lab->d_vFmmsLabel, indx, patch);
      new_dw->getModifiable(velocityVars.wFmms, d_lab->d_wFmmsLabel, indx, patch);
      velocityVars.uFmms.initialize(0.0);
      velocityVars.vFmms.initialize(0.0);
      velocityVars.wFmms.initialize(0.0);
    }


    //__________________________________
    //  compute coefficients
    d_discretize->calculateVelocityCoeff(patch, 
                                     delta_t, d_central, 
                                     cellinfo, &velocityVars,
                                     &constVelocityVars);
                                         
    //__________________________________
    //  Compute the sources
    d_source->calculateVelocitySource(patch, 
                                      delta_t,
                                      cellinfo, &velocityVars,
                                      &constVelocityVars);
    if (d_doMMS){
      d_source->calculateVelMMSSource(pc, patch, 
                                    delta_t, time,
                                    cellinfo, &velocityVars,
                                    &constVelocityVars);
    }


    //__________________________________
    // for scalesimilarity model add stress tensor to the source of velocity eqn.
    if ((dynamic_cast<const OdtClosure*>(d_turbModel))||d_mixedModel) {
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
    SourceTermFactory& factor = SourceTermFactory::self(); 
    for (vector<std::string>::iterator iter = d_new_sources.begin(); 
       iter != d_new_sources.end(); iter++){

      SourceTermBase& src = factor.retrieve_source_term( *iter ); 
      const VarLabel* srcLabel = src.getSrcLabel(); 
      SourceTermBase::MY_TYPE src_type = src.getSourceType(); 

      switch (src_type) {
        case SourceTermBase::CCVECTOR_SRC: 
          { new_dw->get( velocityVars.otherVectorSource, srcLabel, indx, patch, Ghost::None, 0); 
          Vector Dx  = patch->dCell();
          double vol = Dx.x()*Dx.y()*Dx.z();
          for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
            IntVector c = *iter;
            velocityVars.uVelNonlinearSrc[c]  += velocityVars.otherVectorSource[c].x()*vol;
            velocityVars.vVelNonlinearSrc[c]  += velocityVars.otherVectorSource[c].y()*vol;
            velocityVars.wVelNonlinearSrc[c]  += velocityVars.otherVectorSource[c].z()*vol;
          }} 
          break; 
        case SourceTermBase::FX_SRC:
          { new_dw->get( velocityVars.otherFxSource, srcLabel, indx, patch, Ghost::None, 0);
          Vector Dx  = patch->dCell();
          double vol = Dx.x()*Dx.y()*Dx.z();
          for (CellIterator iter=patch->getSFCXIterator(); !iter.done(); iter++){
            IntVector c = *iter;
            velocityVars.uVelNonlinearSrc[c]  += velocityVars.otherFxSource[c]*vol;
          }} 
          break; 
        case SourceTermBase::FY_SRC: 
          { new_dw->get( velocityVars.otherFySource, srcLabel, indx, patch, Ghost::None, 0);
          Vector Dx  = patch->dCell();
          double vol = Dx.x()*Dx.y()*Dx.z();
          for (CellIterator iter=patch->getSFCYIterator(); !iter.done(); iter++){
            IntVector c = *iter;
            velocityVars.vVelNonlinearSrc[c]  += velocityVars.otherFySource[c]*vol;
          }} 
          break; 
        case SourceTermBase::FZ_SRC:
          { new_dw->get( velocityVars.otherFzSource, srcLabel, indx, patch, Ghost::None, 0);
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
    }

    // Calculate the Velocity BCS
    //  inputs : densityCP, [u,v,w]VelocitySIVBC, [u,v,w]VelCoefPBLM
    //           [u,v,w]VelLinSrcPBLM, [u,v,w]VelNonLinSrcPBLM
    //  outputs: [u,v,w]VelCoefPBLM, [u,v,w]VelLinSrcPBLM, 
    //           [u,v,w]VelNonLinSrcPBLM

    if (d_boundaryCondition->anyArchesPhysicalBC()) {

      if (!d_doMMS) {
        d_boundaryCondition->velocityBC(patch,
                                      cellinfo, &velocityVars,
                                      &constVelocityVars);
      }

     /*if (d_boundaryCondition->getIntrusionBC()) {
        // if 0'ing stuff below for zero friction drag
#if 0
        d_boundaryCondition->intrusionMomExchangeBC(pc, patch, index,
                                                    cellinfo, &velocityVars,
                                                    &constVelocityVars);
#endif
      }*/
    }
    // apply multimaterial velocity bc
    // treats multimaterial wall as intrusion
    if (d_MAlab){
      d_boundaryCondition->mmvelocityBC(patch, cellinfo,
                                        &velocityVars, &constVelocityVars);

    }
    // Modify Velocity Mass Source
    //  inputs : [u,v,w]VelocitySIVBC, [u,v,w]VelCoefPBLM, 
    //           [u,v,w]VelConvCoefPBLM, [u,v,w]VelLinSrcPBLM, 
    //           [u,v,w]VelNonLinSrcPBLM
    //  outputs: [u,v,w]VelLinSrcPBLM, [u,v,w]VelNonLinSrcPBLM

    d_source->modifyVelMassSource(patch,
                                  &velocityVars, &constVelocityVars);



    // Calculate Velocity diagonal
    //  inputs : [u,v,w]VelCoefPBLM, [u,v,w]VelLinSrcPBLM
    //  outputs: [u,v,w]VelCoefPBLM

    d_discretize->calculateVelDiagonal(patch,&velocityVars);

    if (d_MAlab) {
      d_boundaryCondition->calculateVelRhoHat_mm(patch, delta_t,
                                                 cellinfo, &velocityVars,
                                                 &constVelocityVars);
    }else {
      d_rhsSolver->calculateHatVelocity(patch, delta_t,
                                       cellinfo, &velocityVars, &constVelocityVars);
    }


    //MMS boundary conditions ~Setting the uncorrected velocities~
    if (d_doMMS) { 
      double time_shiftmms = 0.0;
      time_shiftmms = delta_t * timelabels->time_position_multiplier_before_average;

      d_boundaryCondition->mmsvelocityBC(patch, cellinfo, 
                                         &velocityVars, &constVelocityVars, 
                                         time_shiftmms, 
                                         delta_t);
    }


    if (d_pressure_correction) {
      d_rhsSolver->calculateVelocity(patch, delta_t, cellinfo, &velocityVars,
                                     constVelocityVars.new_density,constVelocityVars.pressure);
    }
    
    //__________________________________
    //
    double time_shift = 0.0;
    if (d_boundaryCondition->getInletBC()) {
    time_shift = delta_t * timelabels->time_position_multiplier_before_average;
    d_boundaryCondition->velRhoHatInletBC(patch,
                                          &velocityVars, &constVelocityVars,
                                          time_shift);
    }
    if ((d_boundaryCondition->getOutletBC())||
        (d_boundaryCondition->getPressureBC()))
      d_boundaryCondition->velRhoHatOutletPressureBC(patch,
                                           &velocityVars, &constVelocityVars);

    /*
  if (d_pressure_correction) {
  int outlet_celltypeval = d_boundaryCondition->outletCellType();
  if (!(outlet_celltypeval==-10)) {
  IntVector idxLo = patch->getFortranCellLowIndex();
  IntVector idxHi = patch->getFortranCellHighIndex();


  bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
  bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
  bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
  bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
  bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
  bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;

  if (xminus) {
    int colX = idxLo.x();
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector xminusCell(colX-1, colY, colZ);
        if (constVelocityVars.cellType[xminusCell] == outlet_celltypeval) {
           double avdenlow = 0.5 * (constVelocityVars.new_density[currCell] +
                                    constVelocityVars.new_density[xminusCell]);

           velocityVars.uVelRhoHat[currCell] -= 2.0*delta_t*
                                 constVelocityVars.pressure[currCell]/
                                (cellinfo->sew[colX] * avdenlow);

           velocityVars.uVelRhoHat[xminusCell] = velocityVars.uVelRhoHat[currCell];

        }
      }
    }
  }
  if (xplus) {
    int colX = idxHi.x();
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector xplusCell(colX+1, colY, colZ);
        IntVector xplusplusCell(colX+2, colY, colZ);
        if (constVelocityVars.cellType[xplusCell] == outlet_celltypeval) {
           double avden = 0.5 * (constVelocityVars.new_density[xplusCell] +
                                 constVelocityVars.new_density[currCell]);

           velocityVars.uVelRhoHat[xplusCell] += 2.0*delta_t*
                                constVelocityVars.pressure[currCell]/
                                (cellinfo->sew[colX] * avden);

           velocityVars.uVelRhoHat[xplusplusCell] = velocityVars.uVelRhoHat[xplusCell];
        }
      }
    }
  }
  if (yminus) {
    int colY = idxLo.y();
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector yminusCell(colX, colY-1, colZ);
        if (constVelocityVars.cellType[yminusCell] == outlet_celltypeval) {
           double avdenlow = 0.5 * (constVelocityVars.new_density[currCell] +
                                    constVelocityVars.new_density[yminusCell]);

           velocityVars.vVelRhoHat[currCell] -= 2.0*delta_t*
                                 constVelocityVars.pressure[currCell]/
                                (cellinfo->sns[colY] * avdenlow);

           velocityVars.vVelRhoHat[yminusCell] = velocityVars.vVelRhoHat[currCell];

        }
      }
    }
  }
  if (yplus) {
    int colY = idxHi.y();
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector yplusCell(colX, colY+1, colZ);
        IntVector yplusplusCell(colX, colY+2, colZ);
        if (constVelocityVars.cellType[yplusCell] == outlet_celltypeval) {
           double avden = 0.5 * (constVelocityVars.new_density[yplusCell] +
                                 constVelocityVars.new_density[currCell]);

           velocityVars.vVelRhoHat[yplusCell] += 2.0*delta_t*
                                 constVelocityVars.pressure[currCell]/
                                (cellinfo->sns[colY] * avden);

           velocityVars.vVelRhoHat[yplusplusCell] = velocityVars.vVelRhoHat[yplusCell];

        }
      }
    }
  }
  if (zminus) {
    int colZ = idxLo.z();
    for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector zminusCell(colX, colY, colZ-1);
        if (constVelocityVars.cellType[zminusCell] == outlet_celltypeval) {
           double avdenlow = 0.5 * (constVelocityVars.new_density[currCell] +
                                    constVelocityVars.new_density[zminusCell]);

           velocityVars.wVelRhoHat[currCell] -= 2.0*delta_t*
                                 constVelocityVars.pressure[currCell]/
                                (cellinfo->stb[colZ] * avdenlow);

           velocityVars.wVelRhoHat[zminusCell] = velocityVars.wVelRhoHat[currCell];

        }
      }
    }
  }
  if (zplus) {
    int colZ = idxHi.z();
    for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector zplusCell(colX, colY, colZ+1);
        IntVector zplusplusCell(colX, colY, colZ+2);
        if (constVelocityVars.cellType[zplusCell] == outlet_celltypeval) {
           double avden = 0.5 * (constVelocityVars.new_density[zplusCell] +
                                 constVelocityVars.new_density[currCell]);

           velocityVars.wVelRhoHat[zplusCell] += 2.0*delta_t*
                                 constVelocityVars.pressure[currCell]/
                                (cellinfo->stb[colZ] * avden);

           velocityVars.wVelRhoHat[zplusplusCell] = velocityVars.wVelRhoHat[zplusCell];

        }
      }
    }
  }
  }
  }*/

//#ifdef divergenceconstraint    
    // compute divergence constraint to use in pressure equation
    d_discretize->computeDivergence(pc, patch, new_dw, &velocityVars, &constVelocityVars,
                    d_filter_divergence_constraint,d_3d_periodic);


    //__________________________________
    //  Jeremy,  should this be in computeDivergence?
    double factor_old = timelabels->factor_old;
    double factor_new = timelabels->factor_new;
    double factor_divide = timelabels->factor_divide;
    
    for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++) {
      const IntVector c = *iter;                                                                 
      velocityVars.divergence[c] = (factor_old*old_divergence[c]+                              
                                    factor_new*velocityVars.divergence[c])/factor_divide;      
    }
//#endif

  }
}

//****************************************************************************
// Schedule the averaging of hat velocities for Runge-Kutta step
//****************************************************************************
void 
MomentumSolver::sched_averageRKHatVelocities(SchedulerP& sched,
                                             const PatchSet* patches,
                                             const MaterialSet* matls,
                                             const TimeIntegratorLabel* timelabels,
                                             bool d_EKTCorrection)
{
  string taskname =  "MomentumSolver::averageRKHatVelocities" +
                     timelabels->integrator_step_name;
  Task* tsk = scinew Task(taskname, this,
                          &MomentumSolver::averageRKHatVelocities,
                          timelabels, d_EKTCorrection);
  
  Ghost::GhostType  gn = Ghost::None;
  Ghost::GhostType  gac = Ghost::AroundCells;
  
  if (d_EKTCorrection) {
    tsk->requires(Task::NewDW, d_lab->d_uVelocityEKTLabel,gn, 0);
    tsk->requires(Task::NewDW, d_lab->d_vVelocityEKTLabel,gn, 0);
    tsk->requires(Task::NewDW, d_lab->d_wVelocityEKTLabel,gn, 0);
  }
  else {
    tsk->requires(Task::OldDW, d_lab->d_uVelocitySPBCLabel,gn, 0);
    tsk->requires(Task::OldDW, d_lab->d_vVelocitySPBCLabel,gn, 0);
    tsk->requires(Task::OldDW, d_lab->d_wVelocitySPBCLabel,gn, 0);
  }
  tsk->requires(Task::OldDW,   d_lab->d_densityCPLabel,    gac,1);
  tsk->requires(Task::NewDW,   d_lab->d_cellTypeLabel,     gn, 0);

  tsk->requires(Task::NewDW,   d_lab->d_densityTempLabel,  gac,1);
  tsk->requires(Task::NewDW,   d_lab->d_densityCPLabel,    gac,1);

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
                                       const TimeIntegratorLabel* timelabels,
                                       bool d_EKTCorrection)
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
    new_dw->get(cellType,    d_lab->d_cellTypeLabel,  indx, patch, gn, 0);
    if (d_EKTCorrection) {
      new_dw->get(old_uvel, d_lab->d_uVelocityEKTLabel, indx, patch, gn, 0);
      new_dw->get(old_vvel, d_lab->d_vVelocityEKTLabel, indx, patch, gn, 0);
      new_dw->get(old_wvel, d_lab->d_wVelocityEKTLabel, indx, patch, gn, 0);
    }
    else {
      old_dw->get(old_uvel, d_lab->d_uVelocitySPBCLabel, indx, patch, gn, 0);
      old_dw->get(old_vvel, d_lab->d_vVelocitySPBCLabel, indx, patch, gn, 0);
      old_dw->get(old_wvel, d_lab->d_wVelocitySPBCLabel, indx, patch, gn, 0);
    }

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
    
    //__________________________________
    //  X  (This includes the extra cells)
    CellIterator SFCX_iter = patch->getSFCXIterator();
    
    for(; !SFCX_iter.done(); SFCX_iter++) {
      IntVector c = *SFCX_iter;
      IntVector L = c - x_offset;
      if (new_density[c]<=1.0e-12 || new_density[L]<=1.0e-12){     // CLAMP
        new_uvel[c] = 0.0;
      }else{
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
      if (new_density[c]<=1.0e-12 || new_density[L]<=1.0e-12){     // CLAMP
        new_vvel[c] = 0.0;
      }else{
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
      if (new_density[c]<=1.0e-12 || new_density[L]<=1.0e-12){     // CLAMP
        new_wvel[c] = 0.0;
      }else{
        new_wvel[c] = (factor_old * old_wvel[c] * (old_density[c]  + old_density[L])
                    +  factor_new * new_wvel[c] * (temp_density[c] + temp_density[L]))/
                       (factor_divide * (new_density[c] + new_density[L]));
      }
    }
    
//__________________________________
//  Apply boundary conditions
// Tangential bc's are not needed to be set for hat velocities
// Commented them out to avoid confusion
    if (d_boundaryCondition->anyArchesPhysicalBC()) {
      int outlet_celltypeval = d_boundaryCondition->outletCellType();
      int pressure_celltypeval = d_boundaryCondition->pressureCellType();
      IntVector idxLo = patch->getFortranCellLowIndex();
      IntVector idxHi = patch->getFortranCellHighIndex();
      bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
      bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
      bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
      bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
      bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
      bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;

      int sign = 0;

      if (xminus) {
        int colX = idxLo.x();
        for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
          for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
            IntVector currCell(colX, colY, colZ);
            IntVector xminusCell(colX-1, colY, colZ);
            IntVector xplusCell(colX+1, colY, colZ);

            if ((cellType[xminusCell] == outlet_celltypeval)||
                (cellType[xminusCell] == pressure_celltypeval)) {
              if ((zminus && (colZ == idxLo.z()))||(zplus && (colZ == idxHi.z()))
                  ||(yminus && (colY == idxLo.y()))||(yplus && (colY == idxHi.y())))
                new_uvel[currCell] = 0.0;
              else {
              if (cellType[xminusCell] == outlet_celltypeval)
                sign = 1;
              else
                sign = -1;
              if (sign * old_uvel[currCell] < -1.0e-10)
                new_uvel[currCell] = new_uvel[xplusCell];
              else
                new_uvel[currCell] = 0.0;
              }
              new_uvel[xminusCell] = new_uvel[currCell];

            }
          }
        }
      }
      if (xplus) {
        int colX = idxHi.x();
        for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
          for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
            IntVector currCell(colX, colY, colZ);
            IntVector xplusCell(colX+1, colY, colZ);
            IntVector xplusplusCell(colX+2, colY, colZ);

            if ((cellType[xplusCell] == outlet_celltypeval)||
                (cellType[xplusCell] == pressure_celltypeval)) {
              if ((zminus && (colZ == idxLo.z()))||(zplus && (colZ == idxHi.z()))
                  ||(yminus && (colY == idxLo.y()))||(yplus && (colY == idxHi.y())))
                new_uvel[xplusCell] = 0.0;
              else {
              if (cellType[xplusCell] == outlet_celltypeval)
                sign = 1;
              else
                sign = -1;
              if (sign * old_uvel[xplusCell] > 1.0e-10)
                new_uvel[xplusCell] = new_uvel[currCell];
              else
                new_uvel[xplusCell] = 0.0;
              }
              new_uvel[xplusplusCell] = new_uvel[xplusCell];

            }
          }
        }
      }
      if (yminus) {
        int colY = idxLo.y();
        for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
          for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
            IntVector currCell(colX, colY, colZ);
            IntVector yminusCell(colX, colY-1, colZ);
            IntVector yplusCell(colX, colY+1, colZ);

            if ((cellType[yminusCell] == outlet_celltypeval)||
                (cellType[yminusCell] == pressure_celltypeval)) {
              if ((zminus && (colZ == idxLo.z()))||(zplus && (colZ == idxHi.z()))
                  ||(xminus && (colX == idxLo.x()))||(xplus && (colX == idxHi.x())))
                new_vvel[currCell] = 0.0;
              else {
              if (cellType[yminusCell] == outlet_celltypeval)
                sign = 1;
              else
                sign = -1;
              if (sign * old_vvel[currCell] < -1.0e-10)
                new_vvel[currCell] = new_vvel[yplusCell];
              else
                new_vvel[currCell] = 0.0;
              }
              new_vvel[yminusCell] = new_vvel[currCell];

            }
          }
        }
      }
      if (yplus) {
        int colY = idxHi.y();
        for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
          for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
            IntVector currCell(colX, colY, colZ);
            IntVector yplusCell(colX, colY+1, colZ);
            IntVector yplusplusCell(colX, colY+2, colZ);

            if ((cellType[yplusCell] == outlet_celltypeval)||
                (cellType[yplusCell] == pressure_celltypeval)) {
              if ((zminus && (colZ == idxLo.z()))||(zplus && (colZ == idxHi.z()))
                  ||(xminus && (colX == idxLo.x()))||(xplus && (colX == idxHi.x())))
                new_vvel[yplusCell] = 0.0;
              else {
              if (cellType[yplusCell] == outlet_celltypeval)
                sign = 1;
              else
                sign = -1;
              if (sign * old_vvel[yplusCell] > 1.0e-10)
                new_vvel[yplusCell] = new_vvel[currCell];
              else
                new_vvel[yplusCell] = 0.0;
              }
              new_vvel[yplusplusCell] = new_vvel[yplusCell];

            }
          }
        }
      }
      if (zminus) {
        int colZ = idxLo.z();
        for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
          for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
            IntVector currCell(colX, colY, colZ);
            IntVector zminusCell(colX, colY, colZ-1);
            IntVector zplusCell(colX, colY, colZ+1);

            if ((cellType[zminusCell] == outlet_celltypeval)||
                (cellType[zminusCell] == pressure_celltypeval)) {
              if ((yminus && (colY == idxLo.y()))||(yplus && (colY == idxHi.y()))
                  ||(xminus && (colX == idxLo.x()))||(xplus && (colX == idxHi.x())))
                new_wvel[currCell] = 0.0;
              else {
              if (cellType[zminusCell] == outlet_celltypeval)
                sign = 1;
              else
                sign = -1;
              if (sign * old_wvel[currCell] < -1.0e-10)
                new_wvel[currCell] = new_wvel[zplusCell];
              else
                new_wvel[currCell] = 0.0;
              }
              new_wvel[zminusCell] = new_wvel[currCell];

            }
          }
        }
      }
      if (zplus) {
        int colZ = idxHi.z();
        for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
          for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
            IntVector currCell(colX, colY, colZ);
            IntVector zplusCell(colX, colY, colZ+1);
            IntVector zplusplusCell(colX, colY, colZ+2);

            if ((cellType[zplusCell] == outlet_celltypeval)||
                (cellType[zplusCell] == pressure_celltypeval)) {
              if ((yminus && (colY == idxLo.y()))||(yplus && (colY == idxHi.y()))
                  ||(xminus && (colX == idxLo.x()))||(xplus && (colX == idxHi.x())))
                new_wvel[zplusCell] = 0.0;
              else {
              if (cellType[zplusCell] == outlet_celltypeval)
                sign = 1;
              else
                sign = -1;
              if (sign * old_wvel[zplusCell] > 1.0e-10)
                new_wvel[zplusCell] = new_wvel[currCell];
              else
                new_wvel[zplusCell] = 0.0;
              }
              new_wvel[zplusplusCell] = new_wvel[zplusCell];

            }
          }
        }
      }
    }  // any physical BC
  }  // patches
}
// ****************************************************************************
// Schedule preparation for extra projection
// ****************************************************************************
void 
MomentumSolver::sched_prepareExtraProjection(SchedulerP& sched,
                                             const PatchSet* patches,
                                             const MaterialSet* matls,
                                             const TimeIntegratorLabel* timelabels,
                                             bool set_BC)
{
  string taskname =  "MomentumSolver::prepareExtraProjection" +
                     timelabels->integrator_step_name;
  Task* tsk = scinew Task(taskname, 
                          this, &MomentumSolver::prepareExtraProjection,
                          timelabels, set_BC);

  Task::WhichDW parent_old_dw;
  if (timelabels->recursion){ 
    parent_old_dw = Task::ParentOldDW;
  }else{
    parent_old_dw = Task::OldDW;
  }
  
  tsk->requires(parent_old_dw, d_lab->d_sharedState->get_delt_label());
   
  Ghost::GhostType  gn = Ghost::None;
  tsk->requires(Task::NewDW, d_lab->d_uVelocitySPBCLabel,gn, 0);
  tsk->requires(Task::NewDW, d_lab->d_vVelocitySPBCLabel,gn, 0);
  tsk->requires(Task::NewDW, d_lab->d_wVelocitySPBCLabel,gn, 0);
  if (set_BC) {
    tsk->requires(Task::NewDW, d_lab->d_densityCPLabel, gn, 0);
    tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel,  gn, 0);
  }
  tsk->modifies(d_lab->d_uVelRhoHatLabel);
  tsk->modifies(d_lab->d_vVelRhoHatLabel);
  tsk->modifies(d_lab->d_wVelRhoHatLabel);
    
  sched->addTask(tsk, patches, matls);
}

// ***********************************************************************
// Actual preparation of extra projection
// ***********************************************************************
void 
MomentumSolver::prepareExtraProjection(const ProcessorGroup* pc,
                                       const PatchSubset* patches,
                                       const MaterialSubset* /*matls*/,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw,
                                       const TimeIntegratorLabel* timelabels,
                                       bool set_BC)
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
    
    Ghost::GhostType  gn = Ghost::None;
    new_dw->getModifiable(velocityVars.uVelRhoHat, d_lab->d_uVelRhoHatLabel,   indx, patch);
    new_dw->copyOut(velocityVars.uVelRhoHat,       d_lab->d_uVelocitySPBCLabel,indx, patch);

    new_dw->getModifiable(velocityVars.vVelRhoHat, d_lab->d_vVelRhoHatLabel,   indx, patch);
    new_dw->copyOut(velocityVars.vVelRhoHat,       d_lab->d_vVelocitySPBCLabel,indx, patch);

    new_dw->getModifiable(velocityVars.wVelRhoHat, d_lab->d_wVelRhoHatLabel,   indx, patch);
    new_dw->copyOut(velocityVars.wVelRhoHat, d_lab->d_wVelocitySPBCLabel,      indx, patch);
    
    if (set_BC) {
      new_dw->get(constVelocityVars.old_uVelocity, d_lab->d_uVelocitySPBCLabel,indx, patch, gn, 0);
      new_dw->get(constVelocityVars.old_vVelocity, d_lab->d_vVelocitySPBCLabel,indx, patch, gn, 0);
      new_dw->get(constVelocityVars.old_wVelocity, d_lab->d_wVelocitySPBCLabel,indx, patch, gn, 0);
      new_dw->get(constVelocityVars.new_density,   d_lab->d_densityCPLabel,    indx, patch, gn, 0);
      new_dw->get(constVelocityVars.cellType,      d_lab->d_cellTypeLabel,     indx, patch, gn, 0);
      
      double time_shift = 0.0;
      if (d_boundaryCondition->getInletBC()) {
        time_shift = delta_t * timelabels->time_position_multiplier_before_average;
        d_boundaryCondition->velRhoHatInletBC(patch,
                                              &velocityVars, &constVelocityVars,
                                              time_shift);
      }
      if ((d_boundaryCondition->getOutletBC())||
          (d_boundaryCondition->getPressureBC()))
        d_boundaryCondition->velRhoHatOutletPressureBC(patch,
                                           &velocityVars, &constVelocityVars);
    }
  }
}
