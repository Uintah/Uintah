/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

//----- PicardNonlinearSolver.cc ----------------------------------------------

#include <CCA/Components/Arches/PicardNonlinearSolver.h>
#include <Core/Containers/StaticArray.h>
#include <CCA/Components/Arches/Arches.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/ArchesMaterial.h>
#include <CCA/Components/Arches/BoundaryCondition.h>
#include <CCA/Components/Arches/CellInformationP.h>
#include <CCA/Components/Arches/EnthalpySolver.h>
#include <CCA/Components/Arches/MomentumSolver.h>
#include <CCA/Components/Arches/PhysicalConstants.h>
#include <CCA/Components/Arches/PressureSolverV2.h>
#include <CCA/Components/Arches/Properties.h>
#include <CCA/Components/Arches/ScalarSolver.h>
#include <CCA/Components/Arches/ScaleSimilarityModel.h>
#include <CCA/Components/Arches/TimeIntegratorLabel.h>
#include <CCA/Components/MPMArches/MPMArchesLabel.h>
#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/LoadBalancer.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Exceptions/VariableNotFoundInGrid.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Parallel/ProcessorGroup.h>
#ifdef PetscFilter
#include <CCA/Components/Arches/Filter.h>
#endif

#include <cmath>

#include <iostream>
using namespace std;

using namespace Uintah;

// ****************************************************************************
// Default constructor for PicardNonlinearSolver
// ****************************************************************************
PicardNonlinearSolver::
PicardNonlinearSolver(ArchesLabel* label, 
                      const MPMArchesLabel* MAlb,
                      Properties* props, 
                      BoundaryCondition* bc,
                      TurbulenceModel* turbModel,
                      PhysicalConstants* physConst,
                      bool calc_Scalar,
                      bool calc_enthalpy,
                      bool calc_variance,
                      const ProcessorGroup* myworld,
                      SolverInterface* hypreSolver):
                      NonlinearSolver(myworld),
                      d_lab(label), d_MAlab(MAlb), d_props(props), 
                      d_boundaryCondition(bc), d_turbModel(turbModel),
                      d_calScalar(calc_Scalar),
                      d_enthalpySolve(calc_enthalpy),
                      d_calcVariance(calc_variance),
                      d_physicalConsts(physConst),
                      d_hypreSolver(hypreSolver)
{
  d_perproc_patches = 0;
  d_pressSolver = 0;
  d_momSolver = 0;
  d_scalarSolver = 0;
  d_enthalpySolver = 0;
  nosolve_timelabels_allocated = false;
}

// ****************************************************************************
// Destructor
// ****************************************************************************
PicardNonlinearSolver::~PicardNonlinearSolver()
{
  delete d_pressSolver;
  delete d_momSolver;
  delete d_scalarSolver;
  delete d_enthalpySolver;
  for (int curr_level = 0; curr_level < numTimeIntegratorLevels; curr_level ++)
    delete d_timeIntegratorLabels[curr_level];
  if (nosolve_timelabels_allocated)
    delete nosolve_timelabels;
  if(d_perproc_patches && d_perproc_patches->removeReference())
    delete d_perproc_patches;
}

// ****************************************************************************
// Problem Setup 
// ****************************************************************************
void 
PicardNonlinearSolver::problemSetup(const ProblemSpecP& params)
  // MultiMaterialInterface* mmInterface
{
  ProblemSpecP db = params->findBlock("PicardSolver");
  db->getWithDefault("max_iter", d_nonlinear_its, 10);
  db->getWithDefault("res_tol", d_resTol, 1.0e-5);
  
  db->require("probe_data", d_probe_data);
  if (d_probe_data) {
    IntVector prbPoint;
    for (ProblemSpecP probe_db = db->findBlock("ProbePoints");
         probe_db != 0;
         probe_db = probe_db->findNextBlock("ProbePoints")) {
      probe_db->require("probe_point", prbPoint);
      d_probePoints.push_back(prbPoint);
    }
  }

  d_pressSolver = scinew PressureSolver(d_lab, d_MAlab,
                                        d_boundaryCondition,
                                        d_physicalConsts, d_myworld,
                                        d_hypreSolver);
  d_pressSolver->problemSetup(db);

  d_momSolver = scinew MomentumSolver(d_lab, d_MAlab,
                                      d_turbModel, d_boundaryCondition,
                                      d_physicalConsts);
  d_momSolver->problemSetup(db);

  if (d_calScalar) {
    d_scalarSolver = scinew ScalarSolver(d_lab, d_MAlab,
                                         d_turbModel, d_boundaryCondition,
                                         d_physicalConsts);
    d_scalarSolver->problemSetup(db);
  }
  d_radiationCalc = false;
  d_DORadiationCalc = false;
  
  if (d_enthalpySolve) {
    d_enthalpySolver = scinew EnthalpySolver(d_lab, d_MAlab,
                                             d_turbModel, d_boundaryCondition,
                                             d_physicalConsts, d_myworld);
    d_enthalpySolver->problemSetup(db);
    d_radiationCalc   = d_enthalpySolver->checkRadiation();
    d_DORadiationCalc = d_enthalpySolver->checkDORadiation();
  }
  db->getWithDefault("timeIntegratorType",d_timeIntegratorType,"BE");
    
  if (d_timeIntegratorType == "BE") {
    d_timeIntegratorLabels.push_back(scinew TimeIntegratorLabel(d_lab,
                                     TimeIntegratorStepType::BE));
    numTimeIntegratorLevels = 1;
  }
  else {
    throw ProblemSetupException("Integrator type is not defined "+d_timeIntegratorType,
                                __FILE__, __LINE__);
  }
  double d_underrelax;
  db->getWithDefault("underrelax",d_underrelax,1.0);
  d_timeIntegratorLabels[0]->factor_new = d_underrelax;
  d_timeIntegratorLabels[0]->factor_old = 1.0 - d_underrelax;

  db->getWithDefault("kineticEnergy_fromFC",d_KE_fromFC,false);
#ifdef PetscFilter
    d_props->setFilter(d_turbModel->getFilter());
//#ifdef divergenceconstraint
    d_momSolver->setDiscretizationFilter(d_turbModel->getFilter());
//#endif
#endif
  d_dynScalarModel = d_turbModel->getDynScalarModel();
  if (d_enthalpySolve) {
    d_H_air = d_props->getAdiabaticAirEnthalpy();
    d_enthalpySolver->setAdiabaticAirEnthalpy(d_H_air);
  }
}

// ****************************************************************************
// Schedule non linear solve and carry out some actual operations
// ****************************************************************************
int PicardNonlinearSolver::nonlinearSolve(const LevelP& level,
                                          SchedulerP& sched)
{
  if(d_perproc_patches && d_perproc_patches->removeReference())
    delete d_perproc_patches;

  LoadBalancer* lb = sched->getLoadBalancer();
  d_perproc_patches = lb->getPerProcessorPatchSet(level);
  d_perproc_patches->addReference();

  Task* tsk = scinew Task("PicardNonlinearSolver::recursiveSolver",
                           this, &PicardNonlinearSolver::recursiveSolver,
                           level, sched.get_rep());
  tsk->hasSubScheduler();
  
  
  Ghost::GhostType  gn = Ghost::None;
  Ghost::GhostType  gac = Ghost::AroundCells;
  Task::MaterialDomainSpec oams = Task::OutOfDomain;  //outside of arches matlSet.
  
  tsk->requires(Task::OldDW, d_lab->d_sharedState->get_delt_label());
  if (dynamic_cast<const ScaleSimilarityModel*>(d_turbModel)) {
    tsk->requires(Task::OldDW, d_lab->d_scalarFluxCompLabel,
                  d_lab->d_vectorMatl, oams, gn,  0);
                  
    tsk->requires(Task::OldDW, d_lab->d_stressTensorCompLabel,
                  d_lab->d_tensorMatl, oams, gac, 1);
  }
  
  tsk->requires(Task::OldDW, d_lab->d_pressurePSLabel,   gn, 0);
  tsk->requires(Task::OldDW, d_lab->d_uVelocitySPBCLabel,gn, 0);
  tsk->requires(Task::OldDW, d_lab->d_vVelocitySPBCLabel,gn, 0);
  tsk->requires(Task::OldDW, d_lab->d_wVelocitySPBCLabel,gn, 0);
  tsk->requires(Task::OldDW, d_lab->d_divConstraintLabel,gn, 0);
  tsk->requires(Task::OldDW, d_lab->d_scalarSPLabel,     gn, 0);
  tsk->requires(Task::OldDW, d_lab->d_densityCPLabel,    gac, 1);
  tsk->requires(Task::OldDW, d_lab->d_densityOldOldLabel,gn,  0);
  tsk->requires(Task::OldDW, d_lab->d_oldDeltaTLabel);
  tsk->requires(Task::OldDW, d_lab->d_viscosityCTSLabel, gn,  0);
  
  //__________________________________
  if (d_MAlab){ 
    tsk->requires(Task::NewDW, d_lab->d_mmcellTypeLabel,  gn, 0);
    tsk->requires(Task::OldDW, d_lab->d_densityMicroLabel,gn, 0);
  }else{
    tsk->requires(Task::OldDW, d_lab->d_cellTypeLabel,    gn, 0);
  }
  
  //__________________________________
  if (d_enthalpySolve) {
    tsk->requires(Task::OldDW, d_lab->d_enthalpySPLabel, gn,  0);
    tsk->requires(Task::OldDW, d_lab->d_tempINLabel,     gac, 1);
    tsk->requires(Task::OldDW, d_lab->d_cpINLabel,       gac, 1);
    
    if (d_radiationCalc) {
      tsk->requires(Task::OldDW, d_lab->d_absorpINLabel,gn, 0);
      if (d_DORadiationCalc) {
        tsk->requires(Task::OldDW, d_lab->d_co2INLabel,           gn, 0);
        tsk->requires(Task::OldDW, d_lab->d_h2oINLabel,           gn, 0);
        tsk->requires(Task::OldDW, d_lab->d_sootFVINLabel,        gn, 0);
        tsk->requires(Task::OldDW, d_lab->d_radiationSRCINLabel,  gn, 0);
        tsk->requires(Task::OldDW, d_lab->d_radiationFluxEINLabel,gn, 0);
        tsk->requires(Task::OldDW, d_lab->d_radiationFluxWINLabel,gn, 0);
        tsk->requires(Task::OldDW, d_lab->d_radiationFluxNINLabel,gn, 0);
        tsk->requires(Task::OldDW, d_lab->d_radiationFluxSINLabel,gn, 0);
        tsk->requires(Task::OldDW, d_lab->d_radiationFluxTINLabel,gn, 0);
        tsk->requires(Task::OldDW, d_lab->d_radiationFluxBINLabel,gn, 0);
        tsk->requires(Task::OldDW, d_lab->d_abskgINLabel,         gn, 0);
      }
    } 
  }
  
  //__________________________________
  if (d_dynScalarModel) {
    if (d_calScalar){
      tsk->requires(Task::OldDW, d_lab->d_scalarDiffusivityLabel,     gn, 0);
    }
    if (d_enthalpySolve){
      tsk->requires(Task::OldDW, d_lab->d_enthalpyDiffusivityLabel,   gn, 0);
    }
  }
  
  //__________________________________
  //
  tsk->computes(d_lab->d_cellTypeLabel);
  tsk->computes(d_lab->d_pressurePSLabel);
  tsk->computes(d_lab->d_uVelocitySPBCLabel);
  tsk->computes(d_lab->d_vVelocitySPBCLabel);
  tsk->computes(d_lab->d_wVelocitySPBCLabel);
  tsk->computes(d_lab->d_uVelRhoHatLabel);
  tsk->computes(d_lab->d_vVelRhoHatLabel);
  tsk->computes(d_lab->d_wVelRhoHatLabel);
  tsk->computes(d_lab->d_filterdrhodtLabel);
  tsk->computes(d_lab->d_drhodfCPLabel);
  tsk->computes(d_lab->d_velocityDivergenceLabel);
  tsk->computes(d_lab->d_velDivResidualLabel);
  tsk->computes(d_lab->d_continuityResidualLabel);
  tsk->computes(d_lab->d_divConstraintLabel);
  tsk->computes(d_lab->d_densityCPLabel);
  tsk->computes(d_lab->d_viscosityCTSLabel);
  tsk->computes(d_lab->d_scalarSPLabel);
  tsk->computes(d_lab->d_totalKineticEnergyLabel);
  tsk->computes(d_lab->d_newCCVelocityLabel);
  tsk->computes(d_lab->d_newCCUVelocityLabel);
  tsk->computes(d_lab->d_newCCVVelocityLabel);
  tsk->computes(d_lab->d_newCCWVelocityLabel);
  tsk->computes(d_lab->d_oldDeltaTLabel);
  tsk->computes(d_lab->d_densityOldOldLabel);
  
  // warning **only works for one scalarVar
  if (d_calcVariance) {
    tsk->computes(d_lab->d_scalarVarSPLabel);
  }
  
  //__________________________________
  if (d_enthalpySolve) {
    tsk->computes(d_lab->d_enthalpySPLabel);
    tsk->computes(d_lab->d_totalRadSrcLabel);
    tsk->computes(d_lab->d_tempINLabel);
    tsk->computes(d_lab->d_cpINLabel);
    
    if (d_radiationCalc) {
      tsk->computes(d_lab->d_absorpINLabel);
      
      if (d_DORadiationCalc) {
        tsk->computes(d_lab->d_co2INLabel);
        tsk->computes(d_lab->d_h2oINLabel);
        tsk->computes(d_lab->d_sootFVINLabel);
        tsk->computes(d_lab->d_abskgINLabel);
        tsk->computes(d_lab->d_radiationSRCINLabel);
        tsk->computes(d_lab->d_radiationFluxEINLabel);
        tsk->computes(d_lab->d_radiationFluxWINLabel);
        tsk->computes(d_lab->d_radiationFluxNINLabel);
        tsk->computes(d_lab->d_radiationFluxSINLabel);
        tsk->computes(d_lab->d_radiationFluxTINLabel);
        tsk->computes(d_lab->d_radiationFluxBINLabel);
      }
    }
  }

//  tsk->computes(d_lab->d_CsLabel);
  //__________________________________
  if (d_dynScalarModel) {
    if (d_calScalar) {
      tsk->computes(d_lab->d_scalarDiffusivityLabel);
      tsk->computes(d_lab->d_ShFLabel);
    }
    if (d_enthalpySolve) {
      tsk->computes(d_lab->d_enthalpyDiffusivityLabel);
      tsk->computes(d_lab->d_ShELabel);
    }
  }
  //__________________________________
  if (d_MAlab){
    tsk->computes(d_lab->d_densityMicroLabel);
  }

  //__________________________________
  if (d_boundaryCondition->anyArchesPhysicalBC()) {
    tsk->computes(d_lab->d_uvwoutLabel);
    tsk->computes(d_lab->d_totalflowINLabel);
    tsk->computes(d_lab->d_totalflowOUTLabel);
    tsk->computes(d_lab->d_denAccumLabel);
    tsk->computes(d_lab->d_netflowOUTBCLabel);
    tsk->computes(d_lab->d_scalarFlowRateLabel);
    
    if (d_boundaryCondition->getCarbonBalance()) {
      tsk->computes(d_lab->d_CO2FlowRateLabel);
    }
    if (d_enthalpySolve) {
      tsk->computes(d_lab->d_enthalpyFlowRateLabel);
    }
  }

  
  if (dynamic_cast<const ScaleSimilarityModel*>(d_turbModel)) {
    tsk->computes(d_lab->d_scalarFluxCompLabel,  d_lab->d_vectorMatl, oams);
    tsk->computes(d_lab->d_stressTensorCompLabel,d_lab->d_tensorMatl, oams);
  }

  sched->addTask(tsk, d_perproc_patches, d_lab->d_sharedState->allArchesMaterials());

  const PatchSet* patches = level->eachPatch();
  const MaterialSet* matls = d_lab->d_sharedState->allArchesMaterials();
  if (d_boundaryCondition->anyArchesPhysicalBC()) {
    d_boundaryCondition->sched_getScalarEfficiency(sched, patches, matls);
  }
  return(0);
}

//______________________________________________________________________
//
void 
PicardNonlinearSolver::recursiveSolver(const ProcessorGroup* pg,
                                       const PatchSubset* patches,
                                       const MaterialSubset* matls,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw,
                                       LevelP level, 
                                       Scheduler* sched)
{
  DataWarehouse::ScrubMode ParentOldDW_scrubmode =
                           old_dw->setScrubbing(DataWarehouse::ScrubNone);
  DataWarehouse::ScrubMode ParentNewDW_scrubmode =
                           new_dw->setScrubbing(DataWarehouse::ScrubNone);
                           
  SchedulerP subsched = sched->createSubScheduler();
  subsched->initialize(3, 1);
  subsched->setParentDWs(old_dw, new_dw);
  subsched->clearMappings();
  subsched->mapDataWarehouse(Task::ParentOldDW, 0);
  subsched->mapDataWarehouse(Task::ParentNewDW, 1);
  subsched->mapDataWarehouse(Task::OldDW, 2);
  subsched->mapDataWarehouse(Task::NewDW, 3);

  GridP grid = level->getGrid();
  subsched->advanceDataWarehouse(grid);

  const PatchSet* local_patches  = level->eachPatch();
  const MaterialSet* local_matls = d_lab->d_sharedState->allArchesMaterials();
  
  IntVector periodic_vector = level->getPeriodicBoundaries();
  d_3d_periodic = (periodic_vector == IntVector(1,1,1));
  d_turbModel->set3dPeriodic(d_3d_periodic);
  d_props->set3dPeriodic(d_3d_periodic);


  //__________________________________
  sched_setInitialGuess(                      subsched, local_patches, local_matls);

  // Start the iterations

#ifdef PetscFilter
  if (d_turbModel->getFilter()) {
    // if the matrix is not initialized
    if (!d_turbModel->getFilter()->isInitialized()){ 
      d_turbModel->sched_initFilterMatrix(level, 
                                             subsched, local_patches, local_matls);
    }
  }
#endif

  int curr_level = 0;

  sched_getDensityGuess(                     subsched, local_patches, local_matls,
                                             d_timeIntegratorLabels[curr_level]);
                                         
  sched_checkDensityGuess(                   subsched, local_patches, local_matls,
                                             d_timeIntegratorLabels[curr_level]);

  
  d_scalarSolver->solve(                     subsched, local_patches, local_matls, 
                                             d_timeIntegratorLabels[curr_level] );

  if (d_enthalpySolve){
    d_enthalpySolver->solve(          level, subsched, local_patches, local_matls,
                                             d_timeIntegratorLabels[curr_level]);
  }

  if (d_calcVariance) {
    d_turbModel->sched_computeScalarVariance(subsched, local_patches, local_matls,
                                             d_timeIntegratorLabels[curr_level]);
                                                                    
    d_turbModel->sched_computeScalarDissipation(
                                              subsched, local_patches, local_matls,
                                              d_timeIntegratorLabels[curr_level]);
  }

  d_props->sched_reComputeProps(             subsched, local_patches, local_matls,
                                             d_timeIntegratorLabels[curr_level],
                                                       true, false);
 
  d_props->sched_computeDenRefArray(         subsched, local_patches, local_matls,
                                             d_timeIntegratorLabels[curr_level]);
                                            
    //sched_syncRhoF(subsched, local_patches, local_matls,
//                   d_timeIntegratorLabels[curr_level]);

    // linearizes and solves pressure eqn
    // first computes, hatted velocities and then computes
    // the pressure poisson equation
  d_momSolver->solveVelHat(         level, subsched,
                                           d_timeIntegratorLabels[curr_level] );
                             
  // using RKSSP averaging to perform underrelaxation
  if (d_timeIntegratorLabels[curr_level]->factor_new < 1.0) {
     sched_saveFECopies(                     subsched, local_patches, local_matls,
                                             d_timeIntegratorLabels[curr_level]);
                                             
    d_timeIntegratorLabels[curr_level]->integrator_step_number = TimeIntegratorStepNumber::Second;
    
    d_props->sched_averageRKProps(           subsched, local_patches, local_matls,
                                             d_timeIntegratorLabels[curr_level]);
                                             
    d_props->sched_saveTempDensity(          subsched, local_patches, local_matls,
                                             d_timeIntegratorLabels[curr_level]);
    if (d_calcVariance) {
      d_turbModel->sched_computeScalarVariance(
                                             subsched, local_patches, local_matls,
                                             d_timeIntegratorLabels[curr_level]);
                                                                    
      d_turbModel->sched_computeScalarDissipation(
                                             subsched, local_patches, local_matls,
                                             d_timeIntegratorLabels[curr_level]);
    }
    d_props->sched_reComputeProps(           subsched, local_patches, local_matls,
                                             d_timeIntegratorLabels[curr_level],
                                                      false, false);

    d_momSolver->sched_averageRKHatVelocities(subsched, local_patches, local_matls,
                                              d_timeIntegratorLabels[curr_level] );
                                                                            
    d_timeIntegratorLabels[curr_level]->integrator_step_number = TimeIntegratorStepNumber::First;
  }

  d_props->sched_computeDrhodt(              subsched, local_patches, local_matls,
                                             d_timeIntegratorLabels[curr_level]);

  d_pressSolver->sched_solve(                level, subsched, 
                                             d_timeIntegratorLabels[curr_level],
                                                             false);
  
  // project velocities using the projection step
    d_momSolver->solve(                      subsched, local_patches, local_matls,
                                             d_timeIntegratorLabels[curr_level],
                                                                    false);

  if (d_boundaryCondition->anyArchesPhysicalBC()) {
    d_boundaryCondition->sched_getFlowINOUT( subsched, local_patches, local_matls,
                                             d_timeIntegratorLabels[curr_level]);
                                          
    d_boundaryCondition->sched_correctVelocityOutletBC(
                                              subsched, local_patches, local_matls,
                                              d_timeIntegratorLabels[curr_level]);
  }
  if (d_boundaryCondition->anyArchesPhysicalBC()) {
    d_boundaryCondition->sched_getScalarFlowRate(
                                              subsched, local_patches, local_matls);
  }
 
  sched_interpolateFromFCToCC(                subsched, local_patches, local_matls,
                                              d_timeIntegratorLabels[curr_level]);
                                              
  d_turbModel->sched_reComputeTurbSubmodel(   subsched, local_patches, local_matls,
                                              d_timeIntegratorLabels[curr_level]);
                                            
//sched_updateDensityGuess(                   subsched, local_patches, local_matls,
//                                            d_timeIntegratorLabels[curr_level]);
//d_timeIntegratorLabels[curr_level]->integrator_step_number= TimeIntegratorStepNumber::Second;
//d_scalarSolver->solve(                      subsched, local_patches, local_matls, 
//                                            d_timeIntegratorLabels[curr_level], index);
//d_timeIntegratorLabels[curr_level]->integrator_step_number= TimeIntegratorStepNumber::First;

  sched_printTotalKE(                         subsched, local_patches, local_matls,
                                              d_timeIntegratorLabels[curr_level]);
  //______________________________________________________________________
  // print information at probes provided in input file
  // WARNING: have no clue how to print probe data only for last iteration
  if (d_probe_data){
    sched_probeData(subsched, local_patches, local_matls);
  }
  subsched->compile();
  int nlIterations = 0;
  double scalar_clipped = 0.0;
  double reactscalar_clipped = 0.0;
  double norm;
  double init_norm = 0.0;
  int num_procs = d_myworld->size();
  
  if (dynamic_cast<const ScaleSimilarityModel*>(d_turbModel)) {
    subsched->get_dw(3)->transferFrom(old_dw, d_lab->d_scalarFluxCompLabel, patches, matls); 
    subsched->get_dw(3)->transferFrom(old_dw, d_lab->d_stressTensorCompLabel, patches, matls); 
  }
  
  subsched->get_dw(3)->transferFrom(old_dw, d_lab->d_cellTypeLabel,      patches, matls); 
  subsched->get_dw(3)->transferFrom(old_dw, d_lab->d_pressurePSLabel,    patches, matls); 
  subsched->get_dw(3)->transferFrom(old_dw, d_lab->d_uVelocitySPBCLabel, patches, matls); 
  subsched->get_dw(3)->transferFrom(old_dw, d_lab->d_vVelocitySPBCLabel, patches, matls); 
  subsched->get_dw(3)->transferFrom(old_dw, d_lab->d_wVelocitySPBCLabel, patches, matls); 
  subsched->get_dw(3)->transferFrom(old_dw, d_lab->d_divConstraintLabel, patches, matls); 
  subsched->get_dw(3)->transferFrom(old_dw, d_lab->d_scalarSPLabel,      patches, matls);
  
  if (d_enthalpySolve) {
    subsched->get_dw(3)->transferFrom(old_dw, d_lab->d_enthalpySPLabel, patches, matls); 
    subsched->get_dw(3)->transferFrom(old_dw, d_lab->d_tempINLabel,     patches, matls); 
    subsched->get_dw(3)->transferFrom(old_dw, d_lab->d_cpINLabel,       patches, matls); 
    
    if (d_radiationCalc) {
      subsched->get_dw(3)->transferFrom(old_dw, d_lab->d_absorpINLabel, patches, matls);
      if (d_DORadiationCalc) {
        subsched->get_dw(3)->transferFrom(old_dw, d_lab->d_co2INLabel,            patches, matls);
        subsched->get_dw(3)->transferFrom(old_dw, d_lab->d_h2oINLabel,            patches, matls);
        subsched->get_dw(3)->transferFrom(old_dw, d_lab->d_sootFVINLabel,         patches, matls);
        subsched->get_dw(3)->transferFrom(old_dw, d_lab->d_radiationSRCINLabel,   patches, matls);
        subsched->get_dw(3)->transferFrom(old_dw, d_lab->d_radiationFluxEINLabel, patches, matls);
        subsched->get_dw(3)->transferFrom(old_dw, d_lab->d_radiationFluxWINLabel, patches, matls);
        subsched->get_dw(3)->transferFrom(old_dw, d_lab->d_radiationFluxNINLabel, patches, matls);
        subsched->get_dw(3)->transferFrom(old_dw, d_lab->d_radiationFluxSINLabel, patches, matls);
        subsched->get_dw(3)->transferFrom(old_dw, d_lab->d_radiationFluxTINLabel, patches, matls);
        subsched->get_dw(3)->transferFrom(old_dw, d_lab->d_radiationFluxBINLabel, patches, matls);
        subsched->get_dw(3)->transferFrom(old_dw, d_lab->d_abskgINLabel,          patches, matls);
      }
    }
  }
  
  subsched->get_dw(3)->transferFrom(old_dw, d_lab->d_densityCPLabel,    patches, matls); 
  subsched->get_dw(3)->transferFrom(old_dw, d_lab->d_viscosityCTSLabel, patches, matls); 
  
  if (d_dynScalarModel) {
    if (d_calScalar)
      subsched->get_dw(3)->transferFrom(old_dw, d_lab->d_scalarDiffusivityLabel,      patches, matls);
    if (d_enthalpySolve)
      subsched->get_dw(3)->transferFrom(old_dw, d_lab->d_enthalpyDiffusivityLabel,    patches, matls);
  }
  do{
    subsched->advanceDataWarehouse(grid);
    subsched->get_dw(2)->setScrubbing(DataWarehouse::ScrubComplete);
    subsched->get_dw(3)->setScrubbing(DataWarehouse::ScrubNone);
    if (d_enthalpySolve) {
      d_enthalpySolver->set_iteration_number(nlIterations);
    }
    subsched->execute();    
    
    delt_vartype delT;
    old_dw->get(delT, d_lab->d_sharedState->get_delt_label() );
    
    double delta_t = delT;
    delta_t *= d_timeIntegratorLabels[curr_level]->time_multiplier;
    max_vartype nm;
    subsched->get_dw(3)->get(nm, d_lab->d_InitNormLabel);
    double current_norm = nm;
    norm = delta_t*current_norm+d_rho_norm+d_u_norm+d_v_norm+d_w_norm;
    
    if (nlIterations == 0){
      init_norm = norm;
    }
    
    if(pg->myrank() == 0){
     cout << "PicardSolver init norm: " << init_norm << " current norm: " << norm << endl;
    }
    max_vartype sc;
    max_vartype rsc;
    subsched->get_dw(3)->get(sc, d_lab->d_ScalarClippedLabel);
    scalar_clipped = sc;

    ++nlIterations;
  
  }while ((nlIterations < d_nonlinear_its)&&
          (norm > d_resTol)&&
          (scalar_clipped == 0.0)&&
          (reactscalar_clipped == 0.0));

  if ((nlIterations == d_nonlinear_its)&&(norm > d_resTol)){
    if(pg->myrank() == 0)
      cout << "Maximum allowed number of iterations reached" << endl;
  }    
     
  if (norm/(init_norm+1.0e-10) > 1) {
    if(pg->myrank() == 0){
       cout << "WARNING! Iterations diverge! Restarting timestep." << endl;
      new_dw->abortTimestep();
      new_dw->restartTimestep();
    }
  }
  
  if ((scalar_clipped > 0.0)||(reactscalar_clipped > 0.0)) {
    if(pg->myrank() == 0){
       cout << "WARNING! Scalars got clipped! Restarting timestep." << endl;
      new_dw->abortTimestep();
      new_dw->restartTimestep();
    }
  }


  new_dw->transferFrom(subsched->get_dw(3), d_lab->d_cellTypeLabel,           patches, matls);
  new_dw->transferFrom(subsched->get_dw(3), d_lab->d_pressurePSLabel,         patches, matls);
  new_dw->transferFrom(subsched->get_dw(3), d_lab->d_uVelocitySPBCLabel,      patches, matls);
  new_dw->transferFrom(subsched->get_dw(3), d_lab->d_vVelocitySPBCLabel,      patches, matls);
  new_dw->transferFrom(subsched->get_dw(3), d_lab->d_wVelocitySPBCLabel,      patches, matls);
  new_dw->transferFrom(subsched->get_dw(3), d_lab->d_uVelRhoHatLabel,         patches, matls);
  new_dw->transferFrom(subsched->get_dw(3), d_lab->d_vVelRhoHatLabel,         patches, matls);
  new_dw->transferFrom(subsched->get_dw(3), d_lab->d_wVelRhoHatLabel,         patches, matls);
  new_dw->transferFrom(subsched->get_dw(3), d_lab->d_filterdrhodtLabel,       patches, matls);
  new_dw->transferFrom(subsched->get_dw(3), d_lab->d_drhodfCPLabel,           patches, matls);
  new_dw->transferFrom(subsched->get_dw(3), d_lab->d_velocityDivergenceLabel, patches, matls); 
  new_dw->transferFrom(subsched->get_dw(3), d_lab->d_velDivResidualLabel,     patches, matls); 
  new_dw->transferFrom(subsched->get_dw(3), d_lab->d_continuityResidualLabel, patches, matls); 
  new_dw->transferFrom(subsched->get_dw(3), d_lab->d_divConstraintLabel,      patches, matls); 
  new_dw->transferFrom(subsched->get_dw(3), d_lab->d_scalarSPLabel,           patches, matls); 
  
  if (d_calcVariance){
    new_dw->transferFrom(subsched->get_dw(3), d_lab->d_scalarVarSPLabel,      patches, matls); 
  }
  
  
  //__________________________________
  if (d_enthalpySolve) {
     
    sum_vartype totalradsrc;
    subsched->get_dw(3)->get(totalradsrc, d_lab->d_totalRadSrcLabel);
    double trs = totalradsrc;
    trs /= num_procs;
    new_dw->put(sum_vartype(trs), d_lab->d_totalRadSrcLabel);
    
    new_dw->transferFrom(subsched->get_dw(3), d_lab->d_tempINLabel, patches, matls); 
    new_dw->transferFrom(subsched->get_dw(3), d_lab->d_cpINLabel,   patches, matls);
    new_dw->transferFrom(subsched->get_dw(3), d_lab->d_enthalpySPLabel, patches, matls); 
    
    if (d_radiationCalc) {
      new_dw->transferFrom(subsched->get_dw(3), d_lab->d_absorpINLabel, patches, matls);
      if (d_DORadiationCalc) {
        new_dw->transferFrom(subsched->get_dw(3), d_lab->d_co2INLabel,            patches, matls);
        new_dw->transferFrom(subsched->get_dw(3), d_lab->d_h2oINLabel,            patches, matls);
        new_dw->transferFrom(subsched->get_dw(3), d_lab->d_sootFVINLabel,         patches, matls);
        new_dw->transferFrom(subsched->get_dw(3), d_lab->d_abskgINLabel,          patches, matls);
        new_dw->transferFrom(subsched->get_dw(3), d_lab->d_radiationSRCINLabel,   patches, matls);
        new_dw->transferFrom(subsched->get_dw(3), d_lab->d_radiationFluxEINLabel, patches, matls);
        new_dw->transferFrom(subsched->get_dw(3), d_lab->d_radiationFluxWINLabel, patches, matls);
        new_dw->transferFrom(subsched->get_dw(3), d_lab->d_radiationFluxNINLabel, patches, matls);
        new_dw->transferFrom(subsched->get_dw(3), d_lab->d_radiationFluxSINLabel, patches, matls);
        new_dw->transferFrom(subsched->get_dw(3), d_lab->d_radiationFluxTINLabel, patches, matls);
        new_dw->transferFrom(subsched->get_dw(3), d_lab->d_radiationFluxBINLabel, patches, matls);
      }
    }
  }
  new_dw->transferFrom(subsched->get_dw(3), d_lab->d_densityCPLabel,    patches, matls); 
  new_dw->transferFrom(subsched->get_dw(3), d_lab->d_viscosityCTSLabel, patches, matls); 
//new_dw->transferFrom(subsched->get_dw(3), d_lab->d_CsLabel,           patches, matls); 
  
  //__________________________________
  if (d_dynScalarModel) {
    if (d_calScalar) {
      new_dw->transferFrom(subsched->get_dw(3), d_lab->d_scalarDiffusivityLabel,      patches, matls);
      new_dw->transferFrom(subsched->get_dw(3), d_lab->d_ShFLabel,                    patches, matls);
    }
    if (d_enthalpySolve) {
      new_dw->transferFrom(subsched->get_dw(3), d_lab->d_enthalpyDiffusivityLabel,    patches, matls);
      new_dw->transferFrom(subsched->get_dw(3), d_lab->d_ShELabel,                    patches, matls);
    }
  }
  
  delt_vartype uvwout;
  sum_vartype flowin;
  sum_vartype flowout;
  sum_vartype denaccum;
  sum_vartype netflowoutbc;
  sum_vartype scalarfr;
  double fin, fout, da, nbc, sfr;
  
  //__________________________________
  if (d_boundaryCondition->anyArchesPhysicalBC()) {
    subsched->get_dw(3)->get(uvwout,       d_lab->d_uvwoutLabel);
    subsched->get_dw(3)->get(flowin,       d_lab->d_totalflowINLabel);
    subsched->get_dw(3)->get(flowout,      d_lab->d_totalflowOUTLabel);
    subsched->get_dw(3)->get(denaccum,     d_lab->d_denAccumLabel);
    subsched->get_dw(3)->get(netflowoutbc, d_lab->d_netflowOUTBCLabel);
    subsched->get_dw(3)->get(scalarfr,     d_lab->d_scalarFlowRateLabel);
    
    fin   = flowin;
    fout  = flowout;
    da    = denaccum;
    nbc   = netflowoutbc;
    sfr   = scalarfr;
    fin   /= num_procs;
    fout  /= num_procs;
    da    /= num_procs;
    nbc   /= num_procs;
    sfr   /= num_procs;
    
    new_dw->put(uvwout,             d_lab->d_uvwoutLabel);
    new_dw->put(sum_vartype(fin),   d_lab->d_totalflowINLabel);
    new_dw->put(sum_vartype(fout),  d_lab->d_totalflowOUTLabel);
    new_dw->put(sum_vartype(da),    d_lab->d_denAccumLabel);
    new_dw->put(sum_vartype(nbc),   d_lab->d_netflowOUTBCLabel);
    new_dw->put(sum_vartype(sfr),   d_lab->d_scalarFlowRateLabel);
    
    if (d_boundaryCondition->getCarbonBalance()) {
      subsched->get_dw(3)->get(scalarfr, d_lab->d_CO2FlowRateLabel);
      sfr  = scalarfr;
      sfr /= num_procs;
      new_dw->put(sum_vartype(sfr), d_lab->d_CO2FlowRateLabel);
    }
    
    if (d_enthalpySolve) {
      subsched->get_dw(3)->get(scalarfr, d_lab->d_enthalpyFlowRateLabel);
      sfr  = scalarfr;
      sfr /= num_procs;
      new_dw->put(sum_vartype(sfr), d_lab->d_enthalpyFlowRateLabel);
    }
  }
  
  sum_vartype totalkineticenergy;
  subsched->get_dw(3)->get(totalkineticenergy, d_lab->d_totalKineticEnergyLabel);
  double tke = totalkineticenergy;
  tke /= num_procs;
  new_dw->put(sum_vartype(tke), d_lab->d_totalKineticEnergyLabel);
  
  new_dw->transferFrom(subsched->get_dw(3), d_lab->d_newCCVelocityLabel,  patches, matls); 
  new_dw->transferFrom(subsched->get_dw(3), d_lab->d_newCCUVelocityLabel, patches, matls); 
  new_dw->transferFrom(subsched->get_dw(3), d_lab->d_newCCVVelocityLabel, patches, matls); 
  new_dw->transferFrom(subsched->get_dw(3), d_lab->d_newCCWVelocityLabel, patches, matls); 
  
  delt_vartype old_delta_t;
  subsched->get_dw(3)->get(old_delta_t, d_lab->d_oldDeltaTLabel);
  new_dw->put(old_delta_t, d_lab->d_oldDeltaTLabel);
  new_dw->transferFrom(subsched->get_dw(3), d_lab->d_densityOldOldLabel, patches, matls); 
  
  if (dynamic_cast<const ScaleSimilarityModel*>(d_turbModel))  {
    new_dw->transferFrom(subsched->get_dw(3), d_lab->d_scalarFluxCompLabel,   patches, matls);
    new_dw->transferFrom(subsched->get_dw(3), d_lab->d_stressTensorCompLabel, patches, matls);
  }

  old_dw->setScrubbing(ParentOldDW_scrubmode);
  new_dw->setScrubbing(ParentNewDW_scrubmode);
}

// ****************************************************************************
// No Solve option (used to skip first time step calculation
// so that further time steps will have correct initial condition)
// ****************************************************************************
int PicardNonlinearSolver::noSolve(const LevelP& level,
                                   SchedulerP& sched)
{
  const PatchSet* patches = level->eachPatch();
  const MaterialSet* matls = d_lab->d_sharedState->allArchesMaterials();

  // use BE timelabels for nosolve
  nosolve_timelabels = scinew TimeIntegratorLabel(d_lab,
                                            TimeIntegratorStepType::BE);
  nosolve_timelabels_allocated = true;

  //initializes and allocates vars for new_dw
  // set initial guess
  // require : old_dw -> pressureSPBC, [u,v,w]velocitySPBC, scalarSP, 
  // densityCP, viscosityCTS
  // compute : new_dw -> pressureIN, [u,v,w]velocityIN, scalarIN, densityIN,
  //                     viscosityIN

  sched_setInitialGuess(sched, patches, matls);

  // check if filter is defined...
#ifdef PetscFilter
  if (d_turbModel->getFilter()) {
    // if the matrix is not initialized
    if (!d_turbModel->getFilter()->isInitialized()) 
      d_turbModel->sched_initFilterMatrix(   level, sched, patches, matls);
  }
#endif

  if (d_calcVariance) {
    d_turbModel->sched_computeScalarVariance(       sched, patches, matls,
                                                    nosolve_timelabels);
                                              
    d_turbModel->sched_computeScalarDissipation(    sched, patches, matls,
                                                    nosolve_timelabels);
  }

  d_props->sched_computePropsFirst_mm(              sched, patches, matls);

  d_props->sched_computeDrhodt(                     sched, patches, matls,
                                                    nosolve_timelabels);

  d_boundaryCondition->sched_setInletFlowRates(     sched, patches, matls);

  sched_dummySolve(                                 sched, patches, matls);


  // Schedule an interpolation of the face centered velocity data 
  // to a cell centered vector for used by the viz tools
  sched_interpolateFromFCToCC(                      sched, patches, matls, 
                                                    nosolve_timelabels);
  
  d_turbModel->sched_reComputeTurbSubmodel(         sched, patches, matls,
                                                    nosolve_timelabels);
  
  d_pressSolver->sched_addHydrostaticTermtoPressure(sched, patches, matls,
                                                    nosolve_timelabels);
 
  // print information at probes provided in input file
  if (d_probe_data){
    sched_probeData(sched, patches, matls);
  }
  return(0);
}

// ****************************************************************************
// Schedule initialize 
// ****************************************************************************
void 
PicardNonlinearSolver::sched_setInitialGuess(SchedulerP& sched, 
                                             const PatchSet* patches,
                                             const MaterialSet* matls)
{
  //copies old db to new_db and then uses non-linear
  //solver to compute new values
  Task* tsk = scinew Task( "PicardNonlinearSolver::setInitialGuess",
                          this, &PicardNonlinearSolver::setInitialGuess);
                          
  Ghost::GhostType  gn = Ghost::None;
  if (d_MAlab){
    tsk->requires(Task::NewDW, d_lab->d_mmcellTypeLabel,   gn, 0);
    tsk->requires(Task::OldDW, d_lab->d_densityMicroLabel, gn, 0);
  }else{
    tsk->requires(Task::OldDW, d_lab->d_cellTypeLabel,      gn, 0);
  }
  tsk->requires(Task::OldDW, d_lab->d_uVelocitySPBCLabel,   gn, 0);
  tsk->requires(Task::OldDW, d_lab->d_vVelocitySPBCLabel,   gn, 0);
  tsk->requires(Task::OldDW, d_lab->d_wVelocitySPBCLabel,   gn, 0);
  tsk->requires(Task::OldDW, d_lab->d_scalarSPLabel,        gn, 0);
  tsk->requires(Task::OldDW, d_lab->d_densityCPLabel,       gn, 0);
  tsk->requires(Task::OldDW, d_lab->d_viscosityCTSLabel,    gn, 0);

  if (d_enthalpySolve){
    tsk->requires(Task::OldDW, d_lab->d_enthalpySPLabel,    gn, 0);
  }
  
  if (d_dynScalarModel) {
    if (d_calScalar){
      tsk->requires(Task::OldDW, d_lab->d_scalarDiffusivityLabel,     gn, 0);
    }
    if (d_enthalpySolve){
      tsk->requires(Task::OldDW, d_lab->d_enthalpyDiffusivityLabel,   gn, 0);
    }
  }
  
  //__________________________________
  tsk->computes(d_lab->d_cellTypeLabel);
  tsk->computes(d_lab->d_uVelocitySPBCLabel);
  tsk->computes(d_lab->d_vVelocitySPBCLabel);
  tsk->computes(d_lab->d_wVelocitySPBCLabel);
  tsk->computes(d_lab->d_uVelRhoHatLabel);
  tsk->computes(d_lab->d_vVelRhoHatLabel);
  tsk->computes(d_lab->d_wVelRhoHatLabel);
  tsk->computes(d_lab->d_densityCPLabel);
  tsk->computes(d_lab->d_scalarSPLabel);
  tsk->computes(d_lab->d_viscosityCTSLabel);
    
  if (d_timeIntegratorLabels[0]->multiple_steps)
    tsk->computes(d_lab->d_scalarTempLabel);

  if (d_enthalpySolve) {
    tsk->computes(d_lab->d_enthalpySPLabel);
    if (d_timeIntegratorLabels[0]->multiple_steps){
      tsk->computes(d_lab->d_enthalpyTempLabel);
    }
  }
  
  
  //__________________________________
  if ((d_timeIntegratorLabels[0]->multiple_steps)||
      (d_timeIntegratorLabels[0]->factor_new < 1.0))
    tsk->computes(d_lab->d_densityTempLabel);

  
  //__________________________________
  if (d_dynScalarModel) {
    if (d_calScalar){
      tsk->computes(d_lab->d_scalarDiffusivityLabel);
    }
    if (d_enthalpySolve){
      tsk->computes(d_lab->d_enthalpyDiffusivityLabel);
    }
    
  }  
  if (d_MAlab){
    tsk->computes(d_lab->d_densityMicroINLabel);
  }
  sched->addTask(tsk, patches, matls);
}

// ****************************************************************************
// Schedule data copy for first time step of Multimaterial algorithm
// ****************************************************************************
void
PicardNonlinearSolver::sched_dummySolve(SchedulerP& sched,
                                        const PatchSet* patches,
                                        const MaterialSet* matls)
{
  Task* tsk = scinew Task( "PicardNonlinearSolver::dataCopy",
                           this, &PicardNonlinearSolver::dummySolve);

  tsk->requires(Task::OldDW, d_lab->d_pressurePSLabel, Ghost::None, 0);

  if (d_calcVariance) {
    tsk->requires(Task::OldDW, d_lab->d_scalarVarSPLabel, Ghost::None, 0);
  }

  tsk->computes(d_lab->d_pressurePSLabel);
  tsk->computes(d_lab->d_presNonLinSrcPBLMLabel);

  // warning **only works for one scalar

  tsk->computes(d_lab->d_uvwoutLabel);
  tsk->computes(d_lab->d_totalflowINLabel);
  tsk->computes(d_lab->d_totalflowOUTLabel);
  tsk->computes(d_lab->d_netflowOUTBCLabel);
  tsk->computes(d_lab->d_denAccumLabel);
  tsk->computes(d_lab->d_scalarEfficiencyLabel);
  tsk->computes(d_lab->d_carbonEfficiencyLabel);

  if (d_calcVariance) {
    tsk->computes(d_lab->d_scalarVarSPLabel);
  }

  sched->addTask(tsk, patches, matls);  
  
}

// ****************************************************************************
// Schedule Interpolate from SFCX, SFCY, SFCZ to CC<Vector>
// ****************************************************************************
void 
PicardNonlinearSolver::sched_interpolateFromFCToCC(SchedulerP& sched, 
                                                   const PatchSet* patches,
                                                   const MaterialSet* matls,
                                                   const TimeIntegratorLabel* timelabels)
{
  string taskname =  "PicardNonlinearSolver::interpFCToCC" +
                     timelabels->integrator_step_name;
  Task* tsk = scinew Task(taskname, this, 
                         &PicardNonlinearSolver::interpolateFromFCToCC, timelabels);

  Ghost::GhostType  gn = Ghost::None;
  Ghost::GhostType  gac = Ghost::AroundCells;
  Ghost::GhostType  gaf = Ghost::AroundFaces;
  
  if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First) {
    tsk->requires(Task::OldDW, d_lab->d_uVelocitySPBCLabel,gaf, 1);
    tsk->requires(Task::OldDW, d_lab->d_vVelocitySPBCLabel,gaf, 1);
    tsk->requires(Task::OldDW, d_lab->d_wVelocitySPBCLabel,gaf, 1);
    tsk->requires(Task::OldDW, d_lab->d_densityCPLabel,    gn, 0);
// hat velocities are only interpolated for first substep, since they are
// not really needed anyway
    tsk->requires(Task::NewDW, d_lab->d_uVelRhoHatLabel,   gaf, 1);
    tsk->requires(Task::NewDW, d_lab->d_vVelRhoHatLabel,   gaf, 1);
    tsk->requires(Task::NewDW, d_lab->d_wVelRhoHatLabel,   gaf, 1);

    tsk->computes(d_lab->d_oldCCVelocityLabel);
    tsk->computes(d_lab->d_uVelRhoHat_CCLabel);
    tsk->computes(d_lab->d_vVelRhoHat_CCLabel);
    tsk->computes(d_lab->d_wVelRhoHat_CCLabel);
  }


  tsk->requires(Task::NewDW, d_lab->d_uVelocitySPBCLabel, gaf, 1);
  tsk->requires(Task::NewDW, d_lab->d_vVelocitySPBCLabel, gaf, 1);
  tsk->requires(Task::NewDW, d_lab->d_wVelocitySPBCLabel, gaf, 1);
  tsk->requires(Task::NewDW, d_lab->d_filterdrhodtLabel,  gn,  0);
  tsk->requires(Task::NewDW, d_lab->d_densityCPLabel,     gac, 1);
  tsk->requires(Task::NewDW, d_lab->d_divConstraintLabel, gn,  0);

  if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First) {
    tsk->computes(d_lab->d_newCCVelocityLabel);
    tsk->computes(d_lab->d_newCCVelMagLabel);
    tsk->computes(d_lab->d_newCCUVelocityLabel);
    tsk->computes(d_lab->d_newCCVVelocityLabel);
    tsk->computes(d_lab->d_newCCWVelocityLabel);
    tsk->computes(d_lab->d_kineticEnergyLabel);
    tsk->computes(d_lab->d_velocityDivergenceLabel);
    tsk->computes(d_lab->d_velDivResidualLabel);
    tsk->computes(d_lab->d_continuityResidualLabel);
    tsk->computes(d_lab->d_uVelNormLabel);
    tsk->computes(d_lab->d_vVelNormLabel);
    tsk->computes(d_lab->d_wVelNormLabel);
    tsk->computes(d_lab->d_rhoNormLabel);
  }
  else {
    tsk->modifies(d_lab->d_newCCVelocityLabel);
    tsk->modifies(d_lab->d_newCCVelMagLabel);
    tsk->modifies(d_lab->d_newCCUVelocityLabel);
    tsk->modifies(d_lab->d_newCCVVelocityLabel);
    tsk->modifies(d_lab->d_newCCWVelocityLabel);
    tsk->modifies(d_lab->d_kineticEnergyLabel);
    tsk->modifies(d_lab->d_velocityDivergenceLabel);
    tsk->modifies(d_lab->d_velDivResidualLabel);
    tsk->modifies(d_lab->d_continuityResidualLabel);
  }
  tsk->computes(timelabels->tke_out);
      
  sched->addTask(tsk, patches, matls);  
}
// ****************************************************************************
// Actual interpolation from FC to CC Variable of type Vector 
// ** WARNING ** For multiple patches we need ghost information for
//               interpolation
// ****************************************************************************
void 
PicardNonlinearSolver::interpolateFromFCToCC(const ProcessorGroup* ,
                                             const PatchSubset* patches,
                                             const MaterialSubset*,
                                             DataWarehouse* old_dw,
                                             DataWarehouse* new_dw,
                                             const TimeIntegratorLabel* timelabels)
{
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int indx = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    constSFCXVariable<double> oldUVel;
    constSFCYVariable<double> oldVVel;
    constSFCZVariable<double> oldWVel;
    constSFCXVariable<double> uHatVel_FCX;
    constSFCYVariable<double> vHatVel_FCY;
    constSFCZVariable<double> wHatVel_FCZ;
    CCVariable<Vector> oldCCVel;
    CCVariable<double> uHatVel_CC;
    CCVariable<double> vHatVel_CC;
    CCVariable<double> wHatVel_CC;
    CCVariable<double> divergence;
    CCVariable<double> div_residual;
    CCVariable<double> residual;
    constCCVariable<double> density;
    constCCVariable<double> old_density;
    constCCVariable<double> drhodt;
    constCCVariable<double> div_constraint;

    constSFCXVariable<double> newUVel;
    constSFCYVariable<double> newVVel;
    constSFCZVariable<double> newWVel;
    CCVariable<Vector> newCCVel;
    CCVariable<double> newCCVelMag;
    CCVariable<double> newCCUVel;
    CCVariable<double> newCCVVel;
    CCVariable<double> newCCWVel;
    CCVariable<double> kineticEnergy;

    bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
    bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
    bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
    bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
    bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
    bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;
    
    IntVector idxLo = patch->getFortranCellLowIndex();
    IntVector idxHi = patch->getFortranCellHighIndex();

    // Get the PerPatch CellInformation data
    PerPatch<CellInformationP> cellInfoP;
    new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, indx, patch);
    CellInformation* cellinfo = cellInfoP.get().get_rep();


    Ghost::GhostType  gn = Ghost::None;
    Ghost::GhostType  gac = Ghost::AroundCells;
    Ghost::GhostType  gaf = Ghost::AroundFaces;
    
    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First) {
      old_dw->get(oldUVel,     d_lab->d_uVelocitySPBCLabel, indx, patch,gaf, 1);
      old_dw->get(oldVVel,     d_lab->d_vVelocitySPBCLabel, indx, patch,gaf, 1);
      old_dw->get(oldWVel,     d_lab->d_wVelocitySPBCLabel, indx, patch,gaf, 1);
      new_dw->get(uHatVel_FCX, d_lab->d_uVelRhoHatLabel,    indx, patch,gaf, 1);
      new_dw->get(vHatVel_FCY, d_lab->d_vVelRhoHatLabel,    indx, patch,gaf, 1);
      new_dw->get(wHatVel_FCZ, d_lab->d_wVelRhoHatLabel,    indx, patch,gaf, 1);
      old_dw->get(old_density, d_lab->d_densityCPLabel,     indx, patch,gn,  0);
                  
      new_dw->allocateAndPut(oldCCVel, d_lab->d_oldCCVelocityLabel,   indx, patch);
      new_dw->allocateAndPut(uHatVel_CC, d_lab->d_uVelRhoHat_CCLabel, indx, patch);
      new_dw->allocateAndPut(vHatVel_CC, d_lab->d_vVelRhoHat_CCLabel, indx, patch);
      new_dw->allocateAndPut(wHatVel_CC, d_lab->d_wVelRhoHat_CCLabel, indx, patch);
                             
      oldCCVel.initialize(Vector(0.0,0.0,0.0));
      uHatVel_CC.initialize(0.0);
      vHatVel_CC.initialize(0.0);
      wHatVel_CC.initialize(0.0);
      for (int kk = idxLo.z(); kk <= idxHi.z(); ++kk) {
        for (int jj = idxLo.y(); jj <= idxHi.y(); ++jj) {
          for (int ii = idxLo.x(); ii <= idxHi.x(); ++ii) {
 
            IntVector idx(ii,jj,kk);
            IntVector idxU(ii+1,jj,kk);
            IntVector idxV(ii,jj+1,kk);
            IntVector idxW(ii,jj,kk+1);
 
            double old_u = cellinfo->wfac[ii] * oldUVel[idx] +
                           cellinfo->efac[ii] * oldUVel[idxU];
            double uhat  = cellinfo->wfac[ii] * uHatVel_FCX[idx] +
                           cellinfo->efac[ii] * uHatVel_FCX[idxU];
                          
            double old_v = cellinfo->sfac[jj] * oldVVel[idx] +
                           cellinfo->nfac[jj] * oldVVel[idxV];
            double vhat  = cellinfo->sfac[jj] * vHatVel_FCY[idx] +
                           cellinfo->nfac[jj] * vHatVel_FCY[idxV];
                          
            double old_w = cellinfo->bfac[kk] * oldWVel[idx] +
                           cellinfo->tfac[kk] * oldWVel[idxW];
            double what  = cellinfo->bfac[kk] * wHatVel_FCZ[idx] +
                           cellinfo->tfac[kk] * wHatVel_FCZ[idxW];
 
            oldCCVel[idx] = Vector(old_u,old_v,old_w);
            uHatVel_CC[idx] = uhat;
            vHatVel_CC[idx] = vhat;
            wHatVel_CC[idx] = what;
          }
        }
      }
      // boundary conditions not to compute erroneous values in the case of ramping
      if (xminus) {
        int ii = idxLo.x()-1;
        for (int kk = idxLo.z(); kk <=  idxHi.z(); kk ++) {
          for (int jj = idxLo.y(); jj <=  idxHi.y(); jj ++) {
            IntVector idx(ii,jj,kk);
            IntVector idxU(ii+1,jj,kk);
            IntVector idxV(ii,jj+1,kk);
            IntVector idxW(ii,jj,kk+1);
 
            double old_u = oldUVel[idxU];
            double uhat  = uHatVel_FCX[idxU];
            double old_v = cellinfo->sfac[jj] * oldVVel[idx] +
                           cellinfo->nfac[jj] * oldVVel[idxV];
            double vhat  = cellinfo->sfac[jj] * vHatVel_FCY[idx] +
                           cellinfo->nfac[jj] * vHatVel_FCY[idxV];
            double old_w = cellinfo->bfac[kk] * oldWVel[idx] +
                           cellinfo->tfac[kk] * oldWVel[idxW];
            double what  = cellinfo->bfac[kk] * wHatVel_FCZ[idx] +
                           cellinfo->tfac[kk] * wHatVel_FCZ[idxW];
 
            oldCCVel[idx] = Vector(old_u,old_v,old_w);
            uHatVel_CC[idx] = uhat;
            vHatVel_CC[idx] = vhat;
            wHatVel_CC[idx] = what;
          }
        }
      }
      if (xplus) {
        int ii =  idxHi.x()+1;
        for (int kk = idxLo.z(); kk <=  idxHi.z(); kk ++) {
          for (int jj = idxLo.y(); jj <=  idxHi.y(); jj ++) {
            IntVector idx(ii,jj,kk);
            IntVector idxU(ii+1,jj,kk);
            IntVector idxV(ii,jj+1,kk);
            IntVector idxW(ii,jj,kk+1);
 
            double old_u = oldUVel[idx];
            double uhat = uHatVel_FCX[idx];
            double old_v = cellinfo->sfac[jj] * oldVVel[idx] +
                           cellinfo->nfac[jj] * oldVVel[idxV];
            double vhat = cellinfo->sfac[jj] * vHatVel_FCY[idx] +
                          cellinfo->nfac[jj] * vHatVel_FCY[idxV];
            double old_w = cellinfo->bfac[kk] * oldWVel[idx] +
                           cellinfo->tfac[kk] * oldWVel[idxW];
            double what = cellinfo->bfac[kk] * wHatVel_FCZ[idx] +
                          cellinfo->tfac[kk] * wHatVel_FCZ[idxW];
 
            oldCCVel[idx] = Vector(old_u,old_v,old_w);
            uHatVel_CC[idx] = uhat;
            vHatVel_CC[idx] = vhat;
            wHatVel_CC[idx] = what;
          }
        }
      }
      if (yminus) {
        int jj = idxLo.y()-1;
        for (int kk = idxLo.z(); kk <=  idxHi.z(); kk ++) {
          for (int ii = idxLo.x(); ii <=  idxHi.x(); ii ++) {
            IntVector idx(ii,jj,kk);
            IntVector idxU(ii+1,jj,kk);
            IntVector idxV(ii,jj+1,kk);
            IntVector idxW(ii,jj,kk+1);
 
            double old_u = cellinfo->wfac[ii] * oldUVel[idx] +
                           cellinfo->efac[ii] * oldUVel[idxU];
            double uhat = cellinfo->wfac[ii] * uHatVel_FCX[idx] +
                          cellinfo->efac[ii] * uHatVel_FCX[idxU];
            double old_v = oldVVel[idxV];
            double vhat = vHatVel_FCY[idxV];
            double old_w = cellinfo->bfac[kk] * oldWVel[idx] +
                           cellinfo->tfac[kk] * oldWVel[idxW];
            double what = cellinfo->bfac[kk] * wHatVel_FCZ[idx] +
                          cellinfo->tfac[kk] * wHatVel_FCZ[idxW];
 
            oldCCVel[idx] = Vector(old_u,old_v,old_w);
            uHatVel_CC[idx] = uhat;
            vHatVel_CC[idx] = vhat;
            wHatVel_CC[idx] = what;
          }
        }
      }
      if (yplus) {
        int jj =  idxHi.y()+1;
        for (int kk = idxLo.z(); kk <=  idxHi.z(); kk ++) {
          for (int ii = idxLo.x(); ii <=  idxHi.x(); ii ++) {
            IntVector idx(ii,jj,kk);
            IntVector idxU(ii+1,jj,kk);
            IntVector idxV(ii,jj+1,kk);
            IntVector idxW(ii,jj,kk+1);
 
            double old_u = cellinfo->wfac[ii] * oldUVel[idx] +
                           cellinfo->efac[ii] * oldUVel[idxU];
            double uhat = cellinfo->wfac[ii] * uHatVel_FCX[idx] +
                          cellinfo->efac[ii] * uHatVel_FCX[idxU];
            double old_v = oldVVel[idx];
            double vhat = vHatVel_FCY[idx];
            double old_w = cellinfo->bfac[kk] * oldWVel[idx] +
                           cellinfo->tfac[kk] * oldWVel[idxW];
            double what = cellinfo->bfac[kk] * wHatVel_FCZ[idx] +
                          cellinfo->tfac[kk] * wHatVel_FCZ[idxW];
 
            oldCCVel[idx] = Vector(old_u,old_v,old_w);
            uHatVel_CC[idx] = uhat;
            vHatVel_CC[idx] = vhat;
            wHatVel_CC[idx] = what;
          }
        }
      }
      if (zminus) {
        int kk = idxLo.z()-1;
        for (int jj = idxLo.y(); jj <=  idxHi.y(); jj ++) {
          for (int ii = idxLo.x(); ii <=  idxHi.x(); ii ++) {
            IntVector idx(ii,jj,kk);
            IntVector idxU(ii+1,jj,kk);
            IntVector idxV(ii,jj+1,kk);
            IntVector idxW(ii,jj,kk+1);
 
            double old_u = cellinfo->wfac[ii] * oldUVel[idx] +
                           cellinfo->efac[ii] * oldUVel[idxU];
            double uhat  = cellinfo->wfac[ii] * uHatVel_FCX[idx] +
                           cellinfo->efac[ii] * uHatVel_FCX[idxU];
            double old_v = cellinfo->sfac[jj] * oldVVel[idx] +
                           cellinfo->nfac[jj] * oldVVel[idxV];
            double vhat  = cellinfo->sfac[jj] * vHatVel_FCY[idx] +
                           cellinfo->nfac[jj] * vHatVel_FCY[idxV];
            double old_w = oldWVel[idxW];
            double what = wHatVel_FCZ[idxW];
 
            oldCCVel[idx] = Vector(old_u,old_v,old_w);
            uHatVel_CC[idx] = uhat;
            vHatVel_CC[idx] = vhat;
            wHatVel_CC[idx] = what;
          }
        }
      }
      if (zplus) {
        int kk =  idxHi.z()+1;
        for (int jj = idxLo.y(); jj <=  idxHi.y(); jj ++) {
          for (int ii = idxLo.x(); ii <=  idxHi.x(); ii ++) {
            IntVector idx(ii,jj,kk);
            IntVector idxU(ii+1,jj,kk);
            IntVector idxV(ii,jj+1,kk);
            IntVector idxW(ii,jj,kk+1);
 
            double old_u = cellinfo->wfac[ii] * oldUVel[idx] +
                           cellinfo->efac[ii] * oldUVel[idxU];
            double uhat  = cellinfo->wfac[ii] * uHatVel_FCX[idx] +
                           cellinfo->efac[ii] * uHatVel_FCX[idxU];
            double old_v = cellinfo->sfac[jj] * oldVVel[idx] +
                           cellinfo->nfac[jj] * oldVVel[idxV];
            double vhat  = cellinfo->sfac[jj] * vHatVel_FCY[idx] +
                           cellinfo->nfac[jj] * vHatVel_FCY[idxV];
            double old_w = oldWVel[idx];
            double what = wHatVel_FCZ[idx];
 
            oldCCVel[idx] = Vector(old_u,old_v,old_w);
            uHatVel_CC[idx] = uhat;
            vHatVel_CC[idx] = vhat;
            wHatVel_CC[idx] = what;
          }
        }
      }
    } 

    new_dw->get(newUVel,        d_lab->d_uVelocitySPBCLabel, indx, patch, gaf, 1);
    new_dw->get(newVVel,        d_lab->d_vVelocitySPBCLabel, indx, patch, gaf, 1);
    new_dw->get(newWVel,        d_lab->d_wVelocitySPBCLabel, indx, patch, gaf, 1);
    new_dw->get(drhodt,         d_lab->d_filterdrhodtLabel,  indx, patch, gn,  0);
    new_dw->get(density,        d_lab->d_densityCPLabel,     indx, patch, gac, 1);
    new_dw->get(div_constraint, d_lab->d_divConstraintLabel, indx, patch, gn, 0);
    
    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First) {
      new_dw->allocateAndPut(newCCVel,      d_lab->d_newCCVelocityLabel,     indx, patch);
      new_dw->allocateAndPut(newCCVelMag,   d_lab->d_newCCVelMagLabel,       indx, patch);
      new_dw->allocateAndPut(newCCUVel,     d_lab->d_newCCUVelocityLabel,    indx, patch);
      new_dw->allocateAndPut(newCCVVel,     d_lab->d_newCCVVelocityLabel,    indx, patch);
      new_dw->allocateAndPut(newCCWVel,     d_lab->d_newCCWVelocityLabel,    indx, patch);
      new_dw->allocateAndPut(kineticEnergy, d_lab->d_kineticEnergyLabel,     indx, patch);
      new_dw->allocateAndPut(divergence,    d_lab->d_velocityDivergenceLabel,indx, patch);
      new_dw->allocateAndPut(div_residual,  d_lab->d_velDivResidualLabel,    indx, patch);
      new_dw->allocateAndPut(residual,      d_lab->d_continuityResidualLabel,indx, patch);
    }else {
      new_dw->getModifiable(newCCVel,     d_lab->d_newCCVelocityLabel,      indx, patch);
      new_dw->getModifiable(newCCVelMag,  d_lab->d_newCCVelMagLabel,        indx, patch);
      new_dw->getModifiable(newCCUVel,    d_lab->d_newCCUVelocityLabel,     indx, patch);
      new_dw->getModifiable(newCCVVel,    d_lab->d_newCCVVelocityLabel,     indx, patch);
      new_dw->getModifiable(newCCWVel,    d_lab->d_newCCWVelocityLabel,     indx, patch);
      new_dw->getModifiable(kineticEnergy, d_lab->d_kineticEnergyLabel,     indx, patch);
      new_dw->getModifiable(divergence,   d_lab->d_velocityDivergenceLabel, indx, patch);
      new_dw->getModifiable(div_residual, d_lab->d_velDivResidualLabel,     indx, patch);
      new_dw->getModifiable(residual,     d_lab->d_continuityResidualLabel, indx, patch);
    }
    newCCVel.initialize(Vector(0.0,0.0,0.0));
    newCCUVel.initialize(0.0);
    newCCVVel.initialize(0.0);
    newCCWVel.initialize(0.0);
    kineticEnergy.initialize(0.0);
    divergence.initialize(0.0);
    div_residual.initialize(0.0);
    residual.initialize(0.0);


    double total_kin_energy = 0.0;
    double u_norm = 0.0;
    double v_norm = 0.0;
    double w_norm = 0.0;
    double rho_norm = 0.0;

    for (int kk = idxLo.z(); kk <= idxHi.z(); ++kk) {
      for (int jj = idxLo.y(); jj <= idxHi.y(); ++jj) {
        for (int ii = idxLo.x(); ii <= idxHi.x(); ++ii) {
          
          IntVector idx(ii,jj,kk);
          IntVector idxU(ii+1,jj,kk);
          IntVector idxV(ii,jj+1,kk);
          IntVector idxW(ii,jj,kk+1);
          
          double new_u = cellinfo->wfac[ii] * newUVel[idx] +
                         cellinfo->efac[ii] * newUVel[idxU];
          double new_v = cellinfo->sfac[jj] * newVVel[idx] +
                         cellinfo->nfac[jj] * newVVel[idxV];
          double new_w = cellinfo->bfac[kk] * newWVel[idx] +
                         cellinfo->tfac[kk] * newWVel[idxW];
          
          newCCVel[idx] = Vector(new_u,new_v,new_w);
          newCCUVel[idx] = new_u;
          newCCVVel[idx] = new_v;
          newCCWVel[idx] = new_w;
          newCCVelMag[idx] = sqrt(new_u*new_u+new_v*new_v+new_w*new_w);
          if (!d_KE_fromFC){
            kineticEnergy[idx] = (new_u*new_u+new_v*new_v+new_w*new_w)/2.0;
          }else{
            kineticEnergy[idx] = (newUVel[idx]*newUVel[idx]+
                                  newVVel[idx]*newVVel[idx]+
                                  newWVel[idx]*newWVel[idx])/2.0;
          }
          total_kin_energy += kineticEnergy[idx];
          u_norm += (newUVel[idx]-oldUVel[idx])*(newUVel[idx]-oldUVel[idx]);
          v_norm += (newVVel[idx]-oldVVel[idx])*(newVVel[idx]-oldVVel[idx]);
          w_norm += (newWVel[idx]-oldWVel[idx])*(newWVel[idx]-oldWVel[idx]);
          rho_norm += (density[idx]-old_density[idx])*(density[idx]-old_density[idx]);
        }
      }
    }
    // boundary conditions not to compute erroneous values in the case of ramping
    if (xminus) {
      int ii = idxLo.x()-1;
      for (int kk = idxLo.z(); kk <=  idxHi.z(); kk ++) {
        for (int jj = idxLo.y(); jj <=  idxHi.y(); jj ++) {
          IntVector idx(ii,jj,kk);
          IntVector idxU(ii+1,jj,kk);
          IntVector idxV(ii,jj+1,kk);
          IntVector idxW(ii,jj,kk+1);
          
          double new_u = newUVel[idxU];
          double new_v = cellinfo->sfac[jj] * newVVel[idx] +
                         cellinfo->nfac[jj] * newVVel[idxV];
          double new_w = cellinfo->bfac[kk] * newWVel[idx] +
                         cellinfo->tfac[kk] * newWVel[idxW];
          
          newCCVel[idx] = Vector(new_u,new_v,new_w);
          newCCUVel[idx] = new_u;
          newCCVVel[idx] = new_v;
          newCCWVel[idx] = new_w;
          newCCVelMag[idx] = sqrt(new_u*new_u+new_v*new_v+new_w*new_w);
          if (!d_KE_fromFC){
            kineticEnergy[idx] = (new_u*new_u+new_v*new_v+new_w*new_w)/2.0;
          }else{
            kineticEnergy[idx] = (newUVel[idxU]*newUVel[idxU]+
                                  newVVel[idx]*newVVel[idx]+
                                  newWVel[idx]*newWVel[idx])/2.0;
          }
          total_kin_energy += kineticEnergy[idx];
        }
      }
    }
    if (xplus) {
      int ii =  idxHi.x()+1;
      for (int kk = idxLo.z(); kk <=  idxHi.z(); kk ++) {
        for (int jj = idxLo.y(); jj <=  idxHi.y(); jj ++) {
          IntVector idx(ii,jj,kk);
          IntVector idxU(ii+1,jj,kk);
          IntVector idxV(ii,jj+1,kk);
          IntVector idxW(ii,jj,kk+1);
          
          double new_u = newUVel[idx];
          double new_v = cellinfo->sfac[jj] * newVVel[idx] +
                         cellinfo->nfac[jj] * newVVel[idxV];
          double new_w = cellinfo->bfac[kk] * newWVel[idx] +
                         cellinfo->tfac[kk] * newWVel[idxW];
          
          newCCVel[idx] = Vector(new_u,new_v,new_w);
          newCCUVel[idx] = new_u;
          newCCVVel[idx] = new_v;
          newCCWVel[idx] = new_w;
          newCCVelMag[idx] = sqrt(new_u*new_u+new_v*new_v+new_w*new_w);
          if (!d_KE_fromFC){
            kineticEnergy[idx] = (new_u*new_u+new_v*new_v+new_w*new_w)/2.0;
          }else{
            kineticEnergy[idx] = (newUVel[idx]*newUVel[idx]+
                                  newVVel[idx]*newVVel[idx]+
                                  newWVel[idx]*newWVel[idx])/2.0;
          }
          total_kin_energy += kineticEnergy[idx];
          u_norm += (newUVel[idx]-oldUVel[idx])*(newUVel[idx]-oldUVel[idx]);
        }
      }
    }
    if (yminus) {
      int jj = idxLo.y()-1;
      for (int kk = idxLo.z(); kk <=  idxHi.z(); kk ++) {
        for (int ii = idxLo.x(); ii <=  idxHi.x(); ii ++) {
          IntVector idx(ii,jj,kk);
          IntVector idxU(ii+1,jj,kk);
          IntVector idxV(ii,jj+1,kk);
          IntVector idxW(ii,jj,kk+1);
          
          double new_u = cellinfo->wfac[ii] * newUVel[idx] +
                         cellinfo->efac[ii] * newUVel[idxU];
          double new_v = newVVel[idxV];
          double new_w = cellinfo->bfac[kk] * newWVel[idx] +
                         cellinfo->tfac[kk] * newWVel[idxW];
          
          newCCVel[idx] = Vector(new_u,new_v,new_w);
          newCCUVel[idx] = new_u;
          newCCVVel[idx] = new_v;
          newCCWVel[idx] = new_w;
          newCCVelMag[idx] = sqrt(new_u*new_u+new_v*new_v+new_w*new_w);
          if (!d_KE_fromFC){
            kineticEnergy[idx] = (new_u*new_u+new_v*new_v+new_w*new_w)/2.0;
          }else{
            kineticEnergy[idx] = (newUVel[idx]*newUVel[idx]+
                                  newVVel[idxV]*newVVel[idxV]+
                                  newWVel[idx]*newWVel[idx])/2.0;
          }
          total_kin_energy += kineticEnergy[idx];
        }
      }
    }
    if (yplus) {
      int jj =  idxHi.y()+1;
      for (int kk = idxLo.z(); kk <=  idxHi.z(); kk ++) {
        for (int ii = idxLo.x(); ii <=  idxHi.x(); ii ++) {
          IntVector idx(ii,jj,kk);
          IntVector idxU(ii+1,jj,kk);
          IntVector idxV(ii,jj+1,kk);
          IntVector idxW(ii,jj,kk+1);
          
          double new_u = cellinfo->wfac[ii] * newUVel[idx] +
                         cellinfo->efac[ii] * newUVel[idxU];
          double new_v = newVVel[idx];
          double new_w = cellinfo->bfac[kk] * newWVel[idx] +
                         cellinfo->tfac[kk] * newWVel[idxW];
          
          newCCVel[idx] = Vector(new_u,new_v,new_w);
          newCCUVel[idx] = new_u;
          newCCVVel[idx] = new_v;
          newCCWVel[idx] = new_w;
          newCCVelMag[idx] = sqrt(new_u*new_u+new_v*new_v+new_w*new_w);
          if (!d_KE_fromFC){
            kineticEnergy[idx] = (new_u*new_u+new_v*new_v+new_w*new_w)/2.0;
          }else{
            kineticEnergy[idx] = (newUVel[idx]*newUVel[idx]+
                                  newVVel[idx]*newVVel[idx]+
                                  newWVel[idx]*newWVel[idx])/2.0;
          }
          total_kin_energy += kineticEnergy[idx];
          v_norm += (newVVel[idx]-oldVVel[idx])*(newVVel[idx]-oldVVel[idx]);
        }
      }
    }
    if (zminus) {
      int kk = idxLo.z()-1;
      for (int jj = idxLo.y(); jj <=  idxHi.y(); jj ++) {
        for (int ii = idxLo.x(); ii <=  idxHi.x(); ii ++) {
          IntVector idx(ii,jj,kk);
          IntVector idxU(ii+1,jj,kk);
          IntVector idxV(ii,jj+1,kk);
          IntVector idxW(ii,jj,kk+1);
          
          double new_u = cellinfo->wfac[ii] * newUVel[idx] +
                         cellinfo->efac[ii] * newUVel[idxU];
          double new_v = cellinfo->sfac[jj] * newVVel[idx] +
                         cellinfo->nfac[jj] * newVVel[idxV];
          double new_w = newWVel[idxW];
          
          newCCVel[idx] = Vector(new_u,new_v,new_w);
          newCCUVel[idx] = new_u;
          newCCVVel[idx] = new_v;
          newCCWVel[idx] = new_w;
          newCCVelMag[idx] = sqrt(new_u*new_u+new_v*new_v+new_w*new_w);
          if (!d_KE_fromFC){
            kineticEnergy[idx] = (new_u*new_u+new_v*new_v+new_w*new_w)/2.0;
          }else{
            kineticEnergy[idx] = (newUVel[idx]*newUVel[idx]+
                                  newVVel[idx]*newVVel[idx]+
                                  newWVel[idxW]*newWVel[idxW])/2.0;
          }
          total_kin_energy += kineticEnergy[idx];
        }
      }
    }
    if (zplus) {
      int kk =  idxHi.z()+1;
      for (int jj = idxLo.y(); jj <=  idxHi.y(); jj ++) {
        for (int ii = idxLo.x(); ii <=  idxHi.x(); ii ++) {
          IntVector idx(ii,jj,kk);
          IntVector idxU(ii+1,jj,kk);
          IntVector idxV(ii,jj+1,kk);
          IntVector idxW(ii,jj,kk+1);
          
          double new_u = cellinfo->wfac[ii] * newUVel[idx] +
                         cellinfo->efac[ii] * newUVel[idxU];
          double new_v = cellinfo->sfac[jj] * newVVel[idx] +
                         cellinfo->nfac[jj] * newVVel[idxV];
          double new_w = newWVel[idx];
          
          newCCVel[idx] = Vector(new_u,new_v,new_w);
          newCCUVel[idx] = new_u;
          newCCVVel[idx] = new_v;
          newCCWVel[idx] = new_w;
          newCCVelMag[idx] = sqrt(new_u*new_u+new_v*new_v+new_w*new_w);
          if (!d_KE_fromFC){
            kineticEnergy[idx] = (new_u*new_u+new_v*new_v+new_w*new_w)/2.0;
          }else{
            kineticEnergy[idx] = (newUVel[idx]*newUVel[idx]+
                                  newVVel[idx]*newVVel[idx]+
                                  newWVel[idxW]*newWVel[idxW])/2.0;
          }
          total_kin_energy += kineticEnergy[idx];
          w_norm += (newWVel[idx]-oldWVel[idx])*(newWVel[idx]-oldWVel[idx]);
        }
      }
    }

    for (int kk = idxLo.z(); kk <= idxHi.z(); ++kk) {
      for (int jj = idxLo.y(); jj <= idxHi.y(); ++jj) {
        for (int ii = idxLo.x(); ii <= idxHi.x(); ++ii) {

          IntVector idx(ii,jj,kk);
          IntVector idxU(ii+1,jj,kk);
          IntVector idxV(ii,jj+1,kk);
          IntVector idxW(ii,jj,kk+1);
          IntVector idxxminus(ii-1,jj,kk);
          IntVector idxyminus(ii,jj-1,kk);
          IntVector idxzminus(ii,jj,kk-1);
          double vol =cellinfo->sns[jj]*cellinfo->stb[kk]*cellinfo->sew[ii];
          
          divergence[idx] = (newUVel[idxU]-newUVel[idx])/cellinfo->sew[ii]+
                            (newVVel[idxV]-newVVel[idx])/cellinfo->sns[jj]+
                            (newWVel[idxW]-newWVel[idx])/cellinfo->stb[kk];

          div_residual[idx] = divergence[idx]-div_constraint[idx]/vol;

          residual[idx] = (0.5*(density[idxU]+density[idx])*newUVel[idxU]-
                           0.5*(density[idx]+density[idxxminus])*newUVel[idx])/cellinfo->sew[ii]+
                          (0.5*(density[idxV]+density[idx])*newVVel[idxV]-
                           0.5*(density[idx]+density[idxyminus])*newVVel[idx])/cellinfo->sns[jj]+
                          (0.5*(density[idxW]+density[idx])*newWVel[idxW]-
                           0.5*(density[idx]+density[idxzminus])*newWVel[idx])/cellinfo->stb[kk]+
                          drhodt[idx]/vol;
        }
      }
    }
    new_dw->put(sum_vartype(total_kin_energy), timelabels->tke_out); 
    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First) {
      new_dw->put(sum_vartype(u_norm),  d_lab->d_uVelNormLabel); 
      new_dw->put(sum_vartype(v_norm),  d_lab->d_vVelNormLabel); 
      new_dw->put(sum_vartype(w_norm),  d_lab->d_wVelNormLabel); 
      new_dw->put(sum_vartype(rho_norm), d_lab->d_rhoNormLabel); 
    }
  }
}

// ****************************************************************************
// Schedule probe data
// ****************************************************************************
void 
PicardNonlinearSolver::sched_probeData(SchedulerP& sched, 
                                       const PatchSet* patches,
                                       const MaterialSet* matls)
{
  Task* tsk = scinew Task( "PicardNonlinearSolver::probeData",
                     this, &PicardNonlinearSolver::probeData);
                     
  Ghost::GhostType  gn = Ghost::None;
  tsk->requires(Task::NewDW, d_lab->d_uVelocitySPBCLabel, gn, 0);
  tsk->requires(Task::NewDW, d_lab->d_vVelocitySPBCLabel, gn, 0);
  tsk->requires(Task::NewDW, d_lab->d_wVelocitySPBCLabel, gn, 0);
  tsk->requires(Task::NewDW, d_lab->d_densityCPLabel,     gn, 0);
  tsk->requires(Task::NewDW, d_lab->d_pressurePSLabel,    gn, 0);
  tsk->requires(Task::NewDW, d_lab->d_viscosityCTSLabel,  gn, 0);
  tsk->requires(Task::NewDW, d_lab->d_scalarSPLabel,      gn, 0);
  tsk->requires(Task::NewDW, d_lab->d_newCCUVelocityLabel,gn, 0);
  tsk->requires(Task::NewDW, d_lab->d_newCCVVelocityLabel,gn, 0);
  tsk->requires(Task::NewDW, d_lab->d_newCCWVelocityLabel,gn, 0);
  tsk->requires(Task::NewDW, d_lab->d_kineticEnergyLabel, gn, 0);

  if (d_calcVariance) {
    tsk->requires(Task::NewDW, d_lab->d_scalarVarSPLabel, gn, 0);
  }

  if (d_enthalpySolve){
    tsk->requires(Task::NewDW, d_lab->d_tempINLabel,      gn, 0);
  }
  
  if (d_MAlab) {
    tsk->requires(Task::NewDW, d_lab->d_mmgasVolFracLabel,gn, 0);
    tsk->requires(Task::NewDW, d_MAlab->integTemp_CCLabel,gn, 0);
    
    tsk->requires(Task::NewDW, d_MAlab->totHT_CCLabel,    gn, 0);

    tsk->requires(Task::NewDW, d_MAlab->totHT_FCXLabel,   gn, 0);
    tsk->requires(Task::NewDW, d_MAlab->totHT_FCYLabel,   gn, 0);
    tsk->requires(Task::NewDW, d_MAlab->totHT_FCZLabel,   gn, 0);

    tsk->requires(Task::NewDW, d_MAlab->totHtFluxXLabel,  gn, 0);
    tsk->requires(Task::NewDW, d_MAlab->totHtFluxYLabel,  gn, 0);
    tsk->requires(Task::NewDW, d_MAlab->totHtFluxZLabel,  gn, 0);
  }

  sched->addTask(tsk, patches, matls);
  
}
// ****************************************************************************
// Actual probe data
// ****************************************************************************
void 
PicardNonlinearSolver::probeData(const ProcessorGroup* ,
                                 const PatchSubset* patches,
                                 const MaterialSubset*,
                                 DataWarehouse*,
                                 DataWarehouse* new_dw)
{

  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int indx = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    double time = d_lab->d_sharedState->getElapsedTime();

    constSFCXVariable<double> newUVel;
    constSFCYVariable<double> newVVel;
    constSFCZVariable<double> newWVel;
    
    Ghost::GhostType  gn = Ghost::None;
    new_dw->get(newUVel, d_lab->d_uVelocitySPBCLabel, indx, patch, gn, 0);
    new_dw->get(newVVel, d_lab->d_vVelocitySPBCLabel, indx, patch, gn, 0);
    new_dw->get(newWVel, d_lab->d_wVelocitySPBCLabel, indx, patch, gn, 0);
    
    constCCVariable<double> newintUVel;
    constCCVariable<double> newintVVel;
    constCCVariable<double> newintWVel;
    new_dw->get(newintUVel, d_lab->d_newCCUVelocityLabel, indx, patch, gn, 0);
    new_dw->get(newintVVel, d_lab->d_newCCVVelocityLabel, indx, patch, gn, 0);
    new_dw->get(newintWVel, d_lab->d_newCCWVelocityLabel, indx, patch, gn, 0);
    
    constCCVariable<double> density;
    constCCVariable<double> viscosity;
    constCCVariable<double> pressure;
    constCCVariable<double> mixtureFraction;
    constCCVariable<double> kineticEnergy;
    
    new_dw->get(density,          d_lab->d_densityCPLabel,    indx, patch, gn, 0);
    new_dw->get(viscosity,        d_lab->d_viscosityCTSLabel, indx, patch, gn, 0);
    new_dw->get(pressure,         d_lab->d_pressurePSLabel,   indx, patch, gn, 0);
    new_dw->get(mixtureFraction,  d_lab->d_scalarSPLabel,     indx, patch, gn, 0);
    new_dw->get(kineticEnergy,    d_lab->d_kineticEnergyLabel,indx, patch, gn, 0);
    
    constCCVariable<double> mixFracVariance;
    if (d_calcVariance) {
      new_dw->get(mixFracVariance, d_lab->d_scalarVarSPLabel, indx, patch, gn, 0);
    }
    
    constCCVariable<double> gasfraction;
    constCCVariable<double> tempSolid;
    constCCVariable<double> totalHT;
    constSFCXVariable<double> totalHT_FCX;
    constSFCYVariable<double> totalHT_FCY;
    constSFCZVariable<double> totalHT_FCZ;
    constSFCXVariable<double> totHtFluxX;
    constSFCYVariable<double> totHtFluxY;
    constSFCZVariable<double> totHtFluxZ;
    if (d_MAlab) {
      new_dw->get(gasfraction, d_lab->d_mmgasVolFracLabel, indx, patch, gn, 0);
      new_dw->get(tempSolid,   d_MAlab->integTemp_CCLabel, indx, patch, gn, 0);

      new_dw->get(totalHT,     d_MAlab->totHT_CCLabel,     indx, patch, gn, 0);
      new_dw->get(totalHT_FCX, d_MAlab->totHT_FCXLabel,    indx, patch, gn, 0);
      new_dw->get(totalHT_FCY, d_MAlab->totHT_FCYLabel,    indx, patch, gn, 0);
      new_dw->get(totalHT_FCZ, d_MAlab->totHT_FCZLabel,    indx, patch, gn, 0);

      new_dw->get(totHtFluxX, d_MAlab->totHtFluxXLabel,    indx, patch, gn, 0);
      new_dw->get(totHtFluxY, d_MAlab->totHtFluxYLabel,    indx, patch, gn, 0);
      new_dw->get(totHtFluxZ, d_MAlab->totHtFluxZLabel,    indx, patch, gn, 0);
    }

    constCCVariable<double> temperature;
    if (d_enthalpySolve){
      new_dw->get(temperature, d_lab->d_tempINLabel, indx, patch, gn, 0);
    }
    for (vector<IntVector>::const_iterator iter = d_probePoints.begin();
         iter != d_probePoints.end(); iter++) {

      if (patch->containsCell(*iter)) {
        cerr.precision(10);
        cerr << "for Intvector: " << *iter << endl;
        cerr << "Density: " << density[*iter] << endl;
        cerr << "Viscosity: " << viscosity[*iter] << endl;
        cerr << "Pressure: " << pressure[*iter] << endl;
        cerr << "MixtureFraction: " << mixtureFraction[*iter] << endl;
        if (d_enthalpySolve)
          cerr<<"Gas Temperature: " << temperature[*iter] << endl;
        cerr << "UVelocity: " << newUVel[*iter] << endl;
        cerr << "VVelocity: " << newVVel[*iter] << endl;
        cerr << "WVelocity: " << newWVel[*iter] << endl;
        cerr << "CCUVelocity: " << newintUVel[*iter] << endl;
        cerr << "CCVVelocity: " << newintVVel[*iter] << endl;
        cerr << "CCWVelocity: " << newintWVel[*iter] << endl;
        cerr << "KineticEnergy: " << kineticEnergy[*iter] << endl;
        
        if (d_calcVariance) {
          cerr << "MixFracVariance: " << mixFracVariance[*iter] << endl;
        }
        
        if (d_MAlab) {
          cerr.precision(16);
          cerr << "gas vol fraction: " << gasfraction[*iter] << endl;
          cerr << " Solid Temperature at Location " << *iter << " At time " << time << ","<< tempSolid[*iter] << endl;
          cerr << " Total Heat Rate at Location " << *iter << " At time " << time << ","<< totalHT[*iter] << endl;
          cerr << " Total X-Dir Heat Rate at Location " << *iter << " At time " << time << ","<< totalHT_FCX[*iter] << endl;
          cerr << " Total Y-Dir Heat Rate at Location " << *iter << " At time " << time << ","<< totalHT_FCY[*iter] << endl;
          cerr << " Total Z-Dir Heat Rate at Location " << *iter << " At time " << time << ","<< totalHT_FCZ[*iter] << endl;
          cerr << " Total X-Dir Heat Flux at Location " << *iter << " At time " << time << ","<< totHtFluxX[*iter] << endl;
          cerr << " Total Y-Dir Heat Flux at Location " << *iter << " At time " << time << ","<< totHtFluxY[*iter] << endl;
          cerr << " Total Z-Dir Heat Flux at Location " << *iter << " At time " << time << ","<< totHtFluxZ[*iter] << endl;
        }

      }
    }
  }
}

// ****************************************************************************
// Actual initialize 
// ****************************************************************************
void 
PicardNonlinearSolver::setInitialGuess(const ProcessorGroup* ,
                                       const PatchSubset* patches,
                                       const MaterialSubset*,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw)
{
  // Get the pressure, velocity, scalars, density and viscosity from the
  // old datawarehouse
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int indx = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    
    constCCVariable<double> denMicro;
    CCVariable<double> denMicro_new;
    
    Ghost::GhostType  gn = Ghost::None;
    if (d_MAlab) {
      old_dw->get(denMicro, d_lab->d_densityMicroLabel, indx, patch, gn, 0);
      new_dw->allocateAndPut(denMicro_new, d_lab->d_densityMicroINLabel, indx, patch);
      denMicro_new.copyData(denMicro);
    }
    
    constCCVariable<int> cellType;
    if (d_MAlab){
      new_dw->get(cellType, d_lab->d_mmcellTypeLabel, indx, patch,gn, 0);
    }else{
      old_dw->get(cellType, d_lab->d_cellTypeLabel,   indx, patch,gn, 0);
    }
    constSFCXVariable<double> uVelocity;
    constSFCYVariable<double> vVelocity;
    constSFCZVariable<double> wVelocity;
    constCCVariable<double> scalar;
    constCCVariable<double> enthalpy;
    constCCVariable<double> scalardiff;
    constCCVariable<double> enthalpydiff;
    constCCVariable<double> reactscalardiff;
    constCCVariable<double> density;
    constCCVariable<double> viscosity;
    
    old_dw->get(uVelocity, d_lab->d_uVelocitySPBCLabel, indx, patch, gn, 0);
    old_dw->get(vVelocity, d_lab->d_vVelocitySPBCLabel, indx, patch, gn, 0);
    old_dw->get(wVelocity, d_lab->d_wVelocitySPBCLabel, indx, patch, gn, 0);
    
    old_dw->get(scalar,    d_lab->d_scalarSPLabel,      indx, patch, gn, 0);
    old_dw->get(density,   d_lab->d_densityCPLabel,     indx, patch, gn, 0);
    old_dw->get(viscosity, d_lab->d_viscosityCTSLabel,  indx, patch, gn, 0);
    
    if (d_enthalpySolve){
      old_dw->get(enthalpy, d_lab->d_enthalpySPLabel, indx, patch, gn, 0);
    }
    
    //__________________________________
    if (d_dynScalarModel) {
      if (d_calScalar){
       old_dw->get(scalardiff,      d_lab->d_scalarDiffusivityLabel,    indx, patch, gn, 0);
      }
      if (d_enthalpySolve){
       old_dw->get(enthalpydiff,    d_lab->d_enthalpyDiffusivityLabel,  indx, patch, gn, 0);
      }
    }


  // Create vars for new_dw ***warning changed new_dw to old_dw...check
    CCVariable<int> cellType_new;
    new_dw->allocateAndPut(cellType_new, d_lab->d_cellTypeLabel, indx, patch);
    cellType_new.copyData(cellType);

    // Get the PerPatch CellInformation data from oldDW, initialize it if it is
    // not there
    if (!(d_MAlab)) {
      PerPatch<CellInformationP> cellInfoP;
      if (new_dw->exists(d_lab->d_cellInfoLabel, indx, patch)){ 
        throw InvalidValue("cellInformation should not be initialized yet",
                           __FILE__, __LINE__);
      }
      
      if (old_dw->exists(d_lab->d_cellInfoLabel, indx, patch)){ 
        old_dw->get(cellInfoP, d_lab->d_cellInfoLabel, indx, patch);
      }else {
        cellInfoP.setData(scinew CellInformation(patch));
      }
      new_dw->put(cellInfoP, d_lab->d_cellInfoLabel, indx, patch);
    }

    SFCXVariable<double> uVelocity_new;
    new_dw->allocateAndPut(uVelocity_new, d_lab->d_uVelocitySPBCLabel, indx, patch);
    uVelocity_new.copyData(uVelocity); 
    
    SFCYVariable<double> vVelocity_new;
    new_dw->allocateAndPut(vVelocity_new, d_lab->d_vVelocitySPBCLabel, indx, patch);
    vVelocity_new.copyData(vVelocity); 
    
    SFCZVariable<double> wVelocity_new;
    new_dw->allocateAndPut(wVelocity_new, d_lab->d_wVelocitySPBCLabel, indx, patch);
    wVelocity_new.copyData(wVelocity); 
    
    SFCXVariable<double> uVelRhoHat_new;
    new_dw->allocateAndPut(uVelRhoHat_new, d_lab->d_uVelRhoHatLabel,   indx, patch);
    uVelRhoHat_new.initialize(0.0); 
    
    SFCYVariable<double> vVelRhoHat_new;
    new_dw->allocateAndPut(vVelRhoHat_new, d_lab->d_vVelRhoHatLabel,   indx, patch);
    vVelRhoHat_new.initialize(0.0); 
    
    SFCZVariable<double> wVelRhoHat_new;
    new_dw->allocateAndPut(wVelRhoHat_new, d_lab->d_wVelRhoHatLabel,   indx, patch);
    wVelRhoHat_new.initialize(0.0); 

    CCVariable<double> scalar_new;
    CCVariable<double> scalar_temp;
    new_dw->allocateAndPut(scalar_new,     d_lab->d_scalarSPLabel,      indx, patch);
    scalar_new.copyData(scalar); // copy old into new
    
    
    if (d_timeIntegratorLabels[0]->multiple_steps) {
      new_dw->allocateAndPut(scalar_temp, d_lab->d_scalarTempLabel,     indx, patch);
      scalar_temp.copyData(scalar); // copy old into new
    }

    constCCVariable<double> reactscalar;
    CCVariable<double> new_reactscalar;
    CCVariable<double> temp_reactscalar;


    CCVariable<double> new_enthalpy;
    CCVariable<double> temp_enthalpy;
    if (d_enthalpySolve) {
      new_dw->allocateAndPut(new_enthalpy, d_lab->d_enthalpySPLabel, indx, patch);
      new_enthalpy.copyData(enthalpy);
      if (d_timeIntegratorLabels[0]->multiple_steps) {
        new_dw->allocateAndPut(temp_enthalpy, d_lab->d_enthalpyTempLabel, indx, patch);
        temp_enthalpy.copyData(enthalpy);
      }
    }
    CCVariable<double> density_new;
    new_dw->allocateAndPut(density_new, d_lab->d_densityCPLabel, indx, patch);
    density_new.copyData(density); // copy old into new
    
    if ((d_timeIntegratorLabels[0]->multiple_steps)||
        (d_timeIntegratorLabels[0]->factor_new < 1.0)) {
      CCVariable<double> density_temp;
      new_dw->allocateAndPut(density_temp, d_lab->d_densityTempLabel, indx, patch);
      density_temp.copyData(density); // copy old into new
    }

    CCVariable<double> viscosity_new;
    new_dw->allocateAndPut(viscosity_new, d_lab->d_viscosityCTSLabel, indx, patch);
    viscosity_new.copyData(viscosity); // copy old into new
    CCVariable<double> scalardiff_new;
    CCVariable<double> enthalpydiff_new;
    CCVariable<double> reactscalardiff_new;
    
    if (d_dynScalarModel) {
      if (d_calScalar) {
        new_dw->allocateAndPut(scalardiff_new, d_lab->d_scalarDiffusivityLabel,indx, patch);
        scalardiff_new.copyData(scalardiff); // copy old into new
      }
      if (d_enthalpySolve) {
        new_dw->allocateAndPut(enthalpydiff_new,
                        d_lab->d_enthalpyDiffusivityLabel, indx, patch);
        enthalpydiff_new.copyData(enthalpydiff); // copy old into new
      }
    }
  }
}


// ****************************************************************************
// Actual Data Copy for first time step of MPMArches
// ****************************************************************************

void 
PicardNonlinearSolver::dummySolve(const ProcessorGroup* ,
                                  const PatchSubset* patches,
                                  const MaterialSubset*,
                                  DataWarehouse* old_dw,
                                  DataWarehouse* new_dw)
{
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int indx = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    constCCVariable<double> scalarVar;
    CCVariable<double> scalarVar_new;
    constCCVariable<double> pressure;
    CCVariable<double> pressure_new;
    CCVariable<double> pressureNLSource;
    
    Ghost::GhostType  gn = Ghost::None;
    old_dw->get(pressure,                d_lab->d_pressurePSLabel, indx, patch, gn, 0);
    new_dw->allocateAndPut(pressure_new, d_lab->d_pressurePSLabel, indx, patch);
    new_dw->allocateAndPut(pressureNLSource, d_lab->d_presNonLinSrcPBLMLabel, indx, patch);
    
    pressureNLSource.initialize(0.0);
    pressure_new.copyData(pressure);     // copy old into new

    
    if (d_calcVariance) {
      old_dw->get(scalarVar,                d_lab->d_scalarVarSPLabel, indx, patch, gn, 0);
      new_dw->allocateAndPut(scalarVar_new, d_lab->d_scalarVarSPLabel, indx, patch);
      scalarVar_new.copyData(scalarVar); // copy old into new
    }

    cout << "PicardNonlinearSolver.cc: DOING DUMMY SOLVE " << endl;
    double uvwout       = 0.0;
    double flowIN       = 0.0;
    double flowOUT      = 0.0;
    double flowOUToutbc = 0.0;
    double denAccum     = 0.0;
    double carbon_efficiency = 0.0;
    double scalar_efficiency = 0.0;


    new_dw->put(delt_vartype(uvwout),         d_lab->d_uvwoutLabel);
    new_dw->put(delt_vartype(flowIN),         d_lab->d_totalflowINLabel);
    new_dw->put(delt_vartype(flowOUT),        d_lab->d_totalflowOUTLabel);
    new_dw->put(delt_vartype(flowOUToutbc),   d_lab->d_netflowOUTBCLabel);
    new_dw->put(delt_vartype(denAccum),       d_lab->d_denAccumLabel);
    new_dw->put(delt_vartype(carbon_efficiency), d_lab->d_carbonEfficiencyLabel);
    new_dw->put(delt_vartype(scalar_efficiency), d_lab->d_scalarEfficiencyLabel);

  }
}

//______________________________________________________________________
//
void 
PicardNonlinearSolver::sched_printTotalKE(SchedulerP& sched, 
                                          const PatchSet* patches,
                                          const MaterialSet* matls,
                                          const TimeIntegratorLabel* timelabels)
{
  string taskname =  "PicardNonlinearSolver::printTotalKE" +
                     timelabels->integrator_step_name;
  Task* tsk = scinew Task(taskname,
                          this, &PicardNonlinearSolver::printTotalKE,
                          timelabels);
  
  tsk->requires(Task::NewDW, timelabels->tke_out);
  if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First) {
    tsk->requires(Task::NewDW, d_lab->d_uVelNormLabel);
    tsk->requires(Task::NewDW, d_lab->d_vVelNormLabel);
    tsk->requires(Task::NewDW, d_lab->d_wVelNormLabel);
    tsk->requires(Task::NewDW, d_lab->d_rhoNormLabel);
  }

  sched->addTask(tsk, patches, matls);
}
//______________________________________________________________________
//
void 
PicardNonlinearSolver::printTotalKE(const ProcessorGroup* ,
                                    const PatchSubset* ,
                                    const MaterialSubset*,
                                    DataWarehouse*,
                                    DataWarehouse* new_dw,
                                    const TimeIntegratorLabel* timelabels)
{
  sum_vartype tke;
  new_dw->get(tke, timelabels->tke_out);
  sum_vartype un,vn,wn,rhon;
  
  if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First) {
    new_dw->get(un,   d_lab->d_uVelNormLabel);
    new_dw->get(vn,   d_lab->d_vVelNormLabel);
    new_dw->get(wn,   d_lab->d_wVelNormLabel);
    new_dw->get(rhon, d_lab->d_rhoNormLabel);
  }
  double total_kin_energy = tke;
  int me = d_myworld->myrank();
  if (me == 0){
     cerr << "Total kinetic energy " <<  total_kin_energy << endl;
  }
  
  if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First) {
    d_u_norm=sqrt(un);
    d_v_norm=sqrt(vn);
    d_w_norm=sqrt(wn);
    d_rho_norm=sqrt(rhon);
    if (me == 0) {
      cerr << "U norm " <<  d_u_norm << endl;
      cerr << "V norm " <<  d_v_norm << endl;
      cerr << "W norm " <<  d_w_norm << endl;
      cerr << "Rho norm " <<  d_rho_norm << endl;
    }
  }
}

//****************************************************************************
// Schedule saving of temp copies of variables
//****************************************************************************
void 
PicardNonlinearSolver::sched_saveTempCopies(SchedulerP& sched, 
                                            const PatchSet* patches,
                                            const MaterialSet* matls,
                                            const TimeIntegratorLabel* timelabels)
{
  string taskname =  "PicardNonlinearSolver::saveTempCopies" +
                     timelabels->integrator_step_name;
  Task* tsk = scinew Task(taskname, this,
                          &PicardNonlinearSolver::saveTempCopies,
                          timelabels);
                          
  Ghost::GhostType  gn = Ghost::None;
  tsk->requires(Task::NewDW, d_lab->d_densityCPLabel, gn, 0);
  tsk->requires(Task::NewDW, d_lab->d_scalarSPLabel,  gn, 0);
  
  if (d_enthalpySolve){
    tsk->requires(Task::NewDW, d_lab->d_enthalpySPLabel,    gn, 0);
  }
  
  tsk->modifies(d_lab->d_densityTempLabel);
  tsk->modifies(d_lab->d_scalarTempLabel);
  
  if (d_enthalpySolve){
    tsk->modifies(d_lab->d_enthalpyTempLabel);
  }
  
  sched->addTask(tsk, patches, matls);
}
//****************************************************************************
// Actually save temp copies here
//****************************************************************************
void 
PicardNonlinearSolver::saveTempCopies(const ProcessorGroup*,
                                      const PatchSubset* patches,
                                      const MaterialSubset*,
                                      DataWarehouse*,
                                      DataWarehouse* new_dw,
                                      const TimeIntegratorLabel*)
{
  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int indx = d_lab->d_sharedState->
                     getArchesMaterial(archIndex)->getDWIndex(); 

    CCVariable<double> temp_density;
    CCVariable<double> temp_scalar;
    CCVariable<double> temp_reactscalar;
    CCVariable<double> temp_enthalpy;

    new_dw->getModifiable(temp_density, d_lab->d_densityTempLabel,  indx, patch);
    new_dw->getModifiable(temp_scalar,  d_lab->d_scalarTempLabel,   indx, patch);
    new_dw->copyOut(temp_density,       d_lab->d_densityCPLabel,    indx, patch);
    new_dw->copyOut(temp_scalar,        d_lab->d_scalarSPLabel,     indx, patch);
    
    if (d_enthalpySolve) {
      new_dw->getModifiable(temp_enthalpy, d_lab->d_enthalpyTempLabel,indx, patch);
      new_dw->copyOut(temp_enthalpy,       d_lab->d_enthalpySPLabel,  indx, patch);
    }
  }
}
//****************************************************************************
// Schedule computation of density guess from the continuity equation
//****************************************************************************
void 
PicardNonlinearSolver::sched_getDensityGuess(SchedulerP& sched,
                                             const PatchSet* patches,
                                             const MaterialSet* matls,
                                             const TimeIntegratorLabel* timelabels)
{
  string taskname =  "PicardNonlinearSolver::getDensityGuess" +
                     timelabels->integrator_step_name;
  Task* tsk = scinew Task(taskname, this,
                          &PicardNonlinearSolver::getDensityGuess,
                          timelabels);

  Task::WhichDW parent_old_dw;
  if (timelabels->recursion){ 
    parent_old_dw = Task::ParentOldDW;
  }else{
    parent_old_dw = Task::OldDW;
  }
  

  Task::WhichDW old_values_dw;
  if (timelabels->use_old_values){
    old_values_dw = parent_old_dw;
  }else{ 
    old_values_dw = Task::NewDW;
  }
  
  Ghost::GhostType  gn = Ghost::None;
  Ghost::GhostType  gac = Ghost::AroundCells;
  Ghost::GhostType  gaf = Ghost::AroundFaces;
    
  tsk->requires(parent_old_dw, d_lab->d_sharedState->get_delt_label());
  tsk->requires(old_values_dw, d_lab->d_densityCPLabel,     gn, 0);
  tsk->requires(Task::NewDW,   d_lab->d_densityCPLabel,     gac, 1);
  tsk->requires(Task::NewDW,   d_lab->d_uVelocitySPBCLabel, gaf, 1);
  tsk->requires(Task::NewDW,   d_lab->d_vVelocitySPBCLabel, gaf, 1);
  tsk->requires(Task::NewDW,   d_lab->d_wVelocitySPBCLabel, gaf, 1);
  tsk->requires(Task::NewDW,   d_lab->d_cellTypeLabel,      gn, 0);


  if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First){
    tsk->computes(d_lab->d_densityGuessLabel);
  }else{
    tsk->modifies(d_lab->d_densityGuessLabel);
  }
  tsk->computes(timelabels->negativeDensityGuess);

  sched->addTask(tsk, patches, matls);
}
//****************************************************************************
// Actually compute density guess from the continuity equation
//****************************************************************************
void 
PicardNonlinearSolver::getDensityGuess(const ProcessorGroup*,
                                       const PatchSubset* patches,
                                       const MaterialSubset*,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw,
                                       const TimeIntegratorLabel* timelabels)
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

  double negativeDensityGuess = 0.0;


  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int indx = d_lab->d_sharedState->
                     getArchesMaterial(archIndex)->getDWIndex(); 

    CCVariable<double> densityGuess;
    constCCVariable<double> density;
    constSFCXVariable<double> uVelocity;
    constSFCYVariable<double> vVelocity;
    constSFCZVariable<double> wVelocity;
    constCCVariable<int> cellType;
    
    Ghost::GhostType  gn = Ghost::None;
    Ghost::GhostType  gac = Ghost::AroundCells;
    Ghost::GhostType  gaf = Ghost::AroundFaces;
    new_dw->get(density,   d_lab->d_densityCPLabel,     indx,patch, gac, 1);
    new_dw->get(uVelocity, d_lab->d_uVelocitySPBCLabel, indx,patch, gaf, 1);
    new_dw->get(vVelocity, d_lab->d_vVelocitySPBCLabel, indx,patch, gaf, 1);
    new_dw->get(wVelocity, d_lab->d_wVelocitySPBCLabel, indx,patch, gaf, 1);
    new_dw->get(cellType,  d_lab->d_cellTypeLabel,      indx,patch, gn, 0);
    
    PerPatch<CellInformationP> cellInfoP;
    new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, indx, patch);
    CellInformation* cellinfo = cellInfoP.get().get_rep();

    DataWarehouse* old_values_dw;
    if (timelabels->use_old_values){
      old_values_dw = parent_old_dw;
    }else{
      old_values_dw = new_dw;
    }
    
    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First){
      new_dw->allocateAndPut(densityGuess, d_lab->d_densityGuessLabel,indx, patch);
    }else{
      new_dw->getModifiable(densityGuess,  d_lab->d_densityGuessLabel,indx, patch);
    }
    old_values_dw->copyOut(densityGuess, d_lab->d_densityCPLabel,indx, patch);

    
// Need to skip first timestep since we start with unprojected velocities
//    int currentTimeStep=d_lab->d_sharedState->getCurrentTopLevelTimeStep();
//    if (currentTimeStep > 1) {
      IntVector idxLo = patch->getFortranCellLowIndex();
      IntVector idxHi = patch->getFortranCellHighIndex();
      for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
        for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
          for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
            IntVector currCell(colX, colY, colZ);
            IntVector xplusCell(colX+1, colY, colZ);
            IntVector xminusCell(colX-1, colY, colZ);
            IntVector yplusCell(colX, colY+1, colZ);
            IntVector yminusCell(colX, colY-1, colZ);
            IntVector zplusCell(colX, colY, colZ+1);
            IntVector zminusCell(colX, colY, colZ-1);
          

            densityGuess[currCell] -= delta_t * 0.5* (
            ((density[currCell]+density[xplusCell])*uVelocity[xplusCell] -
             (density[currCell]+density[xminusCell])*uVelocity[currCell]) /
            cellinfo->sew[colX] +
            ((density[currCell]+density[yplusCell])*vVelocity[yplusCell] -
             (density[currCell]+density[yminusCell])*vVelocity[currCell]) /
            cellinfo->sns[colY] +
            ((density[currCell]+density[zplusCell])*wVelocity[zplusCell] -
             (density[currCell]+density[zminusCell])*wVelocity[currCell]) /
            cellinfo->stb[colZ]);
            if (densityGuess[currCell] < 0.0) {
              cout << "got negative density guess at " << currCell << " , density guess value was " << densityGuess[currCell] << endl;
              negativeDensityGuess = 1.0;
            }
          }
        }
      } 
      
      //__________________________________
      if (d_boundaryCondition->anyArchesPhysicalBC()) {
        bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
        bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
        bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
        bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
        bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
        bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;
        int outlet_celltypeval = d_boundaryCondition->outletCellType();
        int pressure_celltypeval = d_boundaryCondition->pressureCellType();
        if (xminus) {
          int colX = idxLo.x();
          for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
            for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
              IntVector currCell(colX, colY, colZ);
              IntVector xminusCell(colX-1, colY, colZ);
        
               if ((cellType[xminusCell] == outlet_celltypeval)||
                  (cellType[xminusCell] == pressure_celltypeval)) {
                densityGuess[xminusCell] = densityGuess[currCell];
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

               if ((cellType[xplusCell] == outlet_celltypeval)||
                  (cellType[xplusCell] == pressure_celltypeval)) {
                densityGuess[xplusCell] = densityGuess[currCell];
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
        
               if ((cellType[yminusCell] == outlet_celltypeval)||
                  (cellType[yminusCell] == pressure_celltypeval)) {
                densityGuess[yminusCell] = densityGuess[currCell];
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

               if ((cellType[yplusCell] == outlet_celltypeval)||
                  (cellType[yplusCell] == pressure_celltypeval)) {
                densityGuess[yplusCell] = densityGuess[currCell];
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

               if ((cellType[zminusCell] == outlet_celltypeval)||
                  (cellType[zminusCell] == pressure_celltypeval)) {
                densityGuess[zminusCell] = densityGuess[currCell];
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

               if ((cellType[zplusCell] == outlet_celltypeval)||
                  (cellType[zplusCell] == pressure_celltypeval)) {
                densityGuess[zplusCell] = densityGuess[currCell];
              }
            }
          }
        }
      }
   // }
      new_dw->put(sum_vartype(negativeDensityGuess), timelabels->negativeDensityGuess);
  }
}
//****************************************************************************
// Schedule check for negative density guess
//****************************************************************************
void 
PicardNonlinearSolver::sched_checkDensityGuess(SchedulerP& sched,
                                               const PatchSet* patches,
                                               const MaterialSet* matls,
                                               const TimeIntegratorLabel* timelabels)
{
  string taskname =  "PicardNonlinearSolver::checkDensityGuess" +
                     timelabels->integrator_step_name;
  Task* tsk = scinew Task(taskname, this,
                          &PicardNonlinearSolver::checkDensityGuess,
                          timelabels);

  Task::WhichDW parent_old_dw;
  if (timelabels->recursion){
    parent_old_dw = Task::ParentOldDW;
  }else{
    parent_old_dw = Task::OldDW;
  }
  
  Task::WhichDW old_values_dw;
  if (timelabels->use_old_values){
    old_values_dw = parent_old_dw;
  }else{ 
    old_values_dw = Task::NewDW;
  }
  tsk->requires(old_values_dw, d_lab->d_densityCPLabel,Ghost::None, 0);
  tsk->requires(Task::NewDW, timelabels->negativeDensityGuess);


  tsk->modifies(d_lab->d_densityGuessLabel);

  sched->addTask(tsk, patches, matls);
}
//****************************************************************************
// Actually check for negative density guess
//****************************************************************************
void 
PicardNonlinearSolver::checkDensityGuess(const ProcessorGroup* pc,
                                         const PatchSubset* patches,
                                         const MaterialSubset*,
                                         DataWarehouse* old_dw,
                                         DataWarehouse* new_dw,
                                         const TimeIntegratorLabel* timelabels)
{
  DataWarehouse* parent_old_dw;
  if (timelabels->recursion){
    parent_old_dw = new_dw->getOtherDataWarehouse(Task::ParentOldDW);
  }else{ 
    parent_old_dw = old_dw;
  }
  double negativeDensityGuess = 0.0;
  sum_vartype nDG;
  new_dw->get(nDG, timelabels->negativeDensityGuess);
  negativeDensityGuess = nDG;

  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int indx = d_lab->d_sharedState->
                     getArchesMaterial(archIndex)->getDWIndex(); 

    CCVariable<double> densityGuess;


    DataWarehouse* old_values_dw;
    if (timelabels->use_old_values){
      old_values_dw = parent_old_dw;
    }else{
      old_values_dw = new_dw;
    }
    new_dw->getModifiable(densityGuess, d_lab->d_densityGuessLabel,indx, patch);
    if (negativeDensityGuess > 0.0) {
      if (pc->myrank() == 0)
        cout << "WARNING: got negative density guess. Reverting to old density" << endl;
      old_values_dw->copyOut(densityGuess, d_lab->d_densityCPLabel,indx, patch);
    }   
  }
}
//****************************************************************************
// Schedule update of density guess
//****************************************************************************
void 
PicardNonlinearSolver::sched_updateDensityGuess(SchedulerP& sched,
                                                const PatchSet* patches,
                                                const MaterialSet* matls,
                                                const TimeIntegratorLabel* timelabels)
{
  string taskname =  "PicardNonlinearSolver::updateDensityGuess" +
                     timelabels->integrator_step_name;
  Task* tsk = scinew Task(taskname, this,
                          &PicardNonlinearSolver::updateDensityGuess,
                          timelabels);

  tsk->requires(Task::NewDW, d_lab->d_densityCPLabel, Ghost::AroundCells,1);
  tsk->modifies(d_lab->d_densityGuessLabel);

  sched->addTask(tsk, patches, matls);
}
//****************************************************************************
// Actually compute density guess from the continuity equation
//****************************************************************************
void 
PicardNonlinearSolver::updateDensityGuess(const ProcessorGroup*,
                                          const PatchSubset* patches,
                                          const MaterialSubset*,
                                          DataWarehouse*,
                                          DataWarehouse* new_dw,
                                          const TimeIntegratorLabel*)
{
  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int indx = d_lab->d_sharedState->
                     getArchesMaterial(archIndex)->getDWIndex(); 

    CCVariable<double> densityGuess;
    constCCVariable<double> density;

    new_dw->getModifiable(densityGuess, d_lab->d_densityGuessLabel,indx, patch);
    new_dw->copyOut(densityGuess,       d_lab->d_densityCPLabel,   indx, patch);
  }
}
//****************************************************************************
// Schedule syncronizing of rho*f with new density
//****************************************************************************
void 
PicardNonlinearSolver::sched_syncRhoF(SchedulerP& sched,
                                      const PatchSet* patches,
                                      const MaterialSet* matls,
                                      const TimeIntegratorLabel* timelabels)
{
  string taskname =  "PicardNonlinearSolver::syncRhoF" +
                     timelabels->integrator_step_name;
  Task* tsk = scinew Task(taskname, this,
                          &PicardNonlinearSolver::syncRhoF, timelabels);
                          
  Ghost::GhostType  gn = Ghost::None;
  tsk->requires(Task::NewDW, d_lab->d_densityGuessLabel,gn,0);
  tsk->requires(Task::NewDW, d_lab->d_densityCPLabel,   gn,0);
  tsk->modifies(d_lab->d_scalarSPLabel);
  
  if (d_enthalpySolve){
    tsk->modifies(d_lab->d_enthalpySPLabel);
  }
  sched->addTask(tsk, patches, matls);
}
//****************************************************************************
// Actually syncronize of rho*f with new density
//****************************************************************************
void 
PicardNonlinearSolver::syncRhoF(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset*,
                                DataWarehouse*,
                                DataWarehouse* new_dw,
                                const TimeIntegratorLabel*)
{
  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int indx = d_lab->d_sharedState->
                     getArchesMaterial(archIndex)->getDWIndex(); 

    constCCVariable<double> densityGuess;
    constCCVariable<double> density;
    CCVariable<double> scalar;
    CCVariable<double> reactscalar;
    CCVariable<double> enthalpy;
    
    Ghost::GhostType  gn = Ghost::None;
    new_dw->get(densityGuess,     d_lab->d_densityGuessLabel, indx, patch, gn, 0);
    new_dw->get(density,          d_lab->d_densityCPLabel,    indx, patch, gn, 0);
    new_dw->getModifiable(scalar, d_lab->d_scalarSPLabel,     indx, patch);
    
    if (d_enthalpySolve){
      new_dw->getModifiable(enthalpy,    d_lab->d_enthalpySPLabel,   indx, patch);
    }
    
    IntVector idxLo = patch->getExtraCellLowIndex();
    IntVector idxHi = patch->getExtraCellHighIndex();
    for (int colZ = idxLo.z(); colZ < idxHi.z(); colZ ++) {
      for (int colY = idxLo.y(); colY < idxHi.y(); colY ++) {
        for (int colX = idxLo.x(); colX < idxHi.x(); colX ++) {
          IntVector currCell(colX, colY, colZ);

          if (density[currCell] > 0.0) {
            scalar[currCell] = scalar[currCell] * densityGuess[currCell] /density[currCell];
            
            if (scalar[currCell] > 1.0){
              scalar[currCell] = 1.0;
            }else if (scalar[currCell] < 0.0){
              scalar[currCell] = 0.0;
            }
            
            if (d_enthalpySolve){
              enthalpy[currCell] = enthalpy[currCell] * densityGuess[currCell] /
                                 density[currCell];
            }
          }  // rho > 0
        }  // x
      }  // y
    }  // z
  }  // patches
}
//****************************************************************************
// Schedule saving of FE copies of variables
//****************************************************************************
void 
PicardNonlinearSolver::sched_saveFECopies(SchedulerP& sched, 
                                          const PatchSet* patches,
                                          const MaterialSet* matls,
                                          const TimeIntegratorLabel* timelabels)
{
  string taskname =  "PicardNonlinearSolver::saveFECopies" +
                     timelabels->integrator_step_name;
  Task* tsk = scinew Task(taskname, this,
                          &PicardNonlinearSolver::saveFECopies,
                          timelabels);

  Ghost::GhostType  gn = Ghost::None;
  tsk->requires(Task::NewDW, d_lab->d_scalarSPLabel, gn, 0);
  if (d_enthalpySolve){
    tsk->requires(Task::NewDW, d_lab->d_enthalpySPLabel,    gn, 0);
  }
 
  tsk->computes(d_lab->d_scalarFELabel);
  if (d_enthalpySolve){
    tsk->computes(d_lab->d_enthalpyFELabel);
  }

  sched->addTask(tsk, patches, matls);
}
//****************************************************************************
// Actually save temp copies here
//****************************************************************************
void 
PicardNonlinearSolver::saveFECopies(const ProcessorGroup*,
                                    const PatchSubset* patches,
                                    const MaterialSubset*,
                                    DataWarehouse*,
                                    DataWarehouse* new_dw,
                                    const TimeIntegratorLabel*)
{
  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int indx = d_lab->d_sharedState->
                     getArchesMaterial(archIndex)->getDWIndex(); 

    CCVariable<double> temp_scalar;
    CCVariable<double> temp_reactscalar;
    CCVariable<double> temp_enthalpy;

    new_dw->allocateAndPut(temp_scalar, d_lab->d_scalarFELabel,indx, patch);
    new_dw->copyOut(temp_scalar,        d_lab->d_scalarSPLabel,indx, patch);
    
    if (d_enthalpySolve) {
      new_dw->allocateAndPut(temp_enthalpy, d_lab->d_enthalpyFELabel,indx, patch);
      new_dw->copyOut(temp_enthalpy,        d_lab->d_enthalpySPLabel,indx, patch);
    }
  }
}
