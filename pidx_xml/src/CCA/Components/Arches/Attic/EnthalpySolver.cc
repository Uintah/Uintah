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

//----- EnthalpySolver.cc ----------------------------------------------

#include <CCA/Components/Arches/Arches.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/ArchesMaterial.h>
#include <CCA/Components/Arches/BoundaryCondition.h>
#include <CCA/Components/Arches/CellInformationP.h>
#include <CCA/Components/Arches/Discretization.h>
#include <CCA/Components/Arches/EnthalpySolver.h>
#include <CCA/Components/Arches/PhysicalConstants.h>
#include <CCA/Components/Arches/RHSSolver.h>
#include <CCA/Components/Arches/Radiation/DORadiationModel.h>
#include <CCA/Components/Arches/Radiation/RadPetscSolver.h>
#include <CCA/Components/Arches/Source.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermBase.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermFactory.h>
#include <CCA/Components/Arches/TimeIntegratorLabel.h>
#include <CCA/Components/Arches/TurbulenceModel.h>
#include <CCA/Components/MPMArches/MPMArchesLabel.h>

#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Parallel/ProcessorGroup.h>


using namespace Uintah;
using namespace std;

//****************************************************************************
// Default constructor for PressureSolver
//****************************************************************************
EnthalpySolver::EnthalpySolver(const ArchesLabel* label,
                               const MPMArchesLabel* MAlb,
                               TurbulenceModel* turb_model,
                               BoundaryCondition* bndry_cond,
                               PhysicalConstants* physConst,
                               const ProcessorGroup* myworld) :
                                 d_lab(label), d_MAlab(MAlb),
                                 d_turbModel(turb_model), 
                                 d_boundaryCondition(bndry_cond),
                                 d_physicalConsts(physConst),
                                 d_myworld(myworld)

{
  d_perproc_patches = 0;
  d_discretize = 0;
  d_source = 0;
  d_rhsSolver = 0;
  d_DORadiation = 0;
  d_DORadiationCalc = false;
  d_radiationCalc = false;
}

//****************************************************************************
// Destructor
//****************************************************************************
EnthalpySolver::~EnthalpySolver()
{
  delete d_discretize;
  delete d_source;
  delete d_rhsSolver;
  delete d_DORadiation;
  if(d_perproc_patches && d_perproc_patches->removeReference())
    delete d_perproc_patches;
  //VarLabel::destroy(d_abskpLabel);
}

//****************************************************************************
// Problem Setup
//****************************************************************************
void 
EnthalpySolver::problemSetup(const ProblemSpecP& params)
{
  ProblemSpecP db = params->findBlock("EnthalpySolver");
  
  //__________________________________
  //  Radiation
  if (db->findBlock("DORadiationModel")) {
    d_DORadiationCalc = true;
    d_radiationCalc   = true; 
    d_DORadiation = scinew DORadiationModel( d_lab, d_MAlab, d_boundaryCondition, d_myworld);
    d_DORadiation->problemSetup(db, false);
  }

  //__________________________________
  //  Convection scheme
  d_discretize = scinew Discretization();

  string conv_scheme;
  db->getWithDefault("convection_scheme",conv_scheme,"central-upwind");
  
  if (conv_scheme == "central-upwind"){ 
    d_conv_scheme = 0;
  }else if (conv_scheme == "flux_limited"){ 
    d_conv_scheme = 1;
  }else{ 
    throw InvalidValue("Convection scheme not supported: " + conv_scheme, __FILE__, __LINE__);
  }
        
  string limiter_type;
  if (d_conv_scheme == 1) {
    db->getWithDefault("limiter_type",limiter_type,"superbee");
    if (limiter_type == "minmod"){
      d_limiter_type = -1;
    } else if (limiter_type == "superbee"){ 
      d_limiter_type = 0;
    }else if (limiter_type == "vanLeer"){ 
      d_limiter_type = 1;
    }else if (limiter_type == "none") {
      d_limiter_type = 2;
      cout << "WARNING! Running central scheme for scalar," << endl;
      cout << "which can be unstable." << endl;
    }
    else if (limiter_type == "central-upwind"){
      d_limiter_type = 3;
    }else if (limiter_type == "upwind"){
      d_limiter_type = 4;
    }else{
      throw InvalidValue("Flux limiter type not supported: " + limiter_type, __FILE__, __LINE__);
    }
    
    string boundary_limiter_type;
    d_boundary_limiter_type = 3;
    if (d_limiter_type < 3) {
      db->getWithDefault("boundary_limiter_type",boundary_limiter_type,"central-upwind");
      if (boundary_limiter_type == "none") {
        d_boundary_limiter_type = 2;
        cout << "WARNING! Running central scheme for scalar on the boundaries," << endl;
        cout << "which can be unstable." << endl;
      }else if (boundary_limiter_type == "central-upwind"){ 
        d_boundary_limiter_type = 3;
      }else if (boundary_limiter_type == "upwind"){
        d_boundary_limiter_type = 4;
      }else{
       throw InvalidValue("Flux limiter type on the boundary"
                                    "not supported: " + boundary_limiter_type, __FILE__, __LINE__);
      }
      d_central_limiter = false;
      if (d_limiter_type < 2){
        db->getWithDefault("central_limiter",d_central_limiter,false);
      }
    }
  }

  // make source and boundary_condition objects
  d_source = scinew Source(d_physicalConsts);
  
  if (d_doMMS){
    d_source->problemSetup(db);
  }
  d_rhsSolver = scinew RHSSolver();

  d_dynScalarModel = d_turbModel->getDynScalarModel();
  double model_turbPrNo;
  model_turbPrNo = d_turbModel->getTurbulentPrandtlNumber();

  // see if Prandtl number gets overridden here
  d_turbPrNo = 0.0;
  if (!(d_dynScalarModel)) {
    if (db->findBlock("turbulentPrandtlNumber"))
      db->getWithDefault("turbulentPrandtlNumber",d_turbPrNo,0.4);

    // if it is not set in both places
    if ((d_turbPrNo == 0.0)&&(model_turbPrNo == 0.0)){
      throw InvalidValue("Turbulent Prandtl number is not specified for"
                             "mixture fraction ", __FILE__, __LINE__);
    // if it is set in turbulence model
    }else if (d_turbPrNo == 0.0){
      d_turbPrNo = model_turbPrNo;
    }
  }

  d_discretize->setTurbulentPrandtlNumber(d_turbPrNo);



  // New Source terms (ala the new transport eqn):
  if (db->findBlock("src")){
    string srcname; 
    for (ProblemSpecP src_db = db->findBlock("src"); src_db != 0; src_db = src_db->findNextBlock("src")){
      src_db->getAttribute("label", srcname);
      //which sources are turned on for this equation
      d_new_sources.push_back( srcname ); 

    }
  }
}

//****************************************************************************
// Schedule solve of linearized enthalpy equation
//****************************************************************************
void 
EnthalpySolver::solve(const LevelP& level,
                      SchedulerP& sched,
                      const PatchSet* patches,
                      const MaterialSet* matls,
                      const TimeIntegratorLabel* timelabels)
{
  int timeSubStep = 0;
  if ( timelabels->integrator_step_number == TimeIntegratorStepNumber::Second )
    timeSubStep = 1;
  else if ( timelabels->integrator_step_number == TimeIntegratorStepNumber::Third )
    timeSubStep = 2;

  bool isFirstIntegrationStep = ( timelabels->integrator_step_number == TimeIntegratorStepNumber::First );
  
  
  //__________________________________
  // Schedule additional sources for evaluation
  SourceTermFactory& factory = SourceTermFactory::self();
  for (vector<std::string>::iterator iter = d_new_sources.begin();
      iter != d_new_sources.end(); iter++){
    SourceTermBase& src = factory.retrieve_source_term( *iter );
    src.sched_computeSource( level, sched, timeSubStep );
  }

  // Discrete Ordanences
  if(d_DORadiationCalc){
    d_DORadiation->sched_computeSource(level, sched, matls, timelabels, isFirstIntegrationStep);
  }
  
  
  //__________________________________
  ////computes stencil coefficients and source terms
  // requires : enthalpyIN, [u,v,w]VelocitySPBC, densityIN, viscosityIN
  // computes : scalCoefSBLM, scalLinSrcSBLM, scalNonLinSrcSBLM
  sched_buildLinearMatrix(level, sched, patches, matls, timelabels );
  
  // Schedule the enthalpy solve
  // require : enthalpyIN, scalCoefSBLM, scalNonLinSrcSBLM
  sched_enthalpyLinearSolve(sched, patches, matls, timelabels );
}

//****************************************************************************
// Schedule build of linear matrix
//****************************************************************************
void 
EnthalpySolver::sched_buildLinearMatrix(const LevelP& level,
                                        SchedulerP& sched,
                                        const PatchSet*,
                                        const MaterialSet* matls,
                                        const TimeIntegratorLabel* timelabels)
{
  if(d_perproc_patches && d_perproc_patches->removeReference())
    delete d_perproc_patches;

  LoadBalancer* lb = sched->getLoadBalancer();
  d_perproc_patches = lb->getPerProcessorPatchSet(level);
  d_perproc_patches->addReference();
  //  const MaterialSet* matls = d_lab->d_sharedState->allArchesMaterials();


  string taskname =  "EnthalpySolver::BuildCoeff" +
                     timelabels->integrator_step_name;
  
  Task* tsk = scinew Task(taskname, this,
                          &EnthalpySolver::buildLinearMatrix,
                          timelabels );

  Task::WhichDW parent_old_dw;
  if (timelabels->recursion){ 
    parent_old_dw = Task::ParentOldDW;
  }else{ 
    parent_old_dw = Task::OldDW;
  }

  tsk->requires(parent_old_dw, d_lab->d_sharedState->get_delt_label());
  
  // This task requires enthalpy and density from old time step for transient
  // calculation
  
  Ghost::GhostType  gac = Ghost::AroundCells;
  Ghost::GhostType  gaf = Ghost::AroundFaces;
  Ghost::GhostType  gn = Ghost::None;
  Task::MaterialDomainSpec oams = Task::OutOfDomain;  //outside of arches matlSet.
  
  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel,    gac, 1);
  tsk->requires(Task::NewDW, d_lab->d_enthalpySPLabel,  gac, 2);
  tsk->requires(Task::NewDW, d_lab->d_densityCPLabel,   gac, 2);

  Task::WhichDW old_values_dw;
  if (timelabels->use_old_values){
    old_values_dw = parent_old_dw;
  }else{ 
    old_values_dw = Task::NewDW;
  }
  tsk->requires(old_values_dw, d_lab->d_enthalpySPLabel,  gn, 0);
  tsk->requires(old_values_dw, d_lab->d_densityCPLabel,   gn, 0);
  
  tsk->requires(Task::NewDW, d_lab->d_cellInfoLabel, gn);


  if (d_dynScalarModel){
    tsk->requires(Task::NewDW, d_lab->d_enthalpyDiffusivityLabel, gac, 2);
  }else{
    tsk->requires(Task::NewDW, d_lab->d_viscosityCTSLabel,        gac, 2);
  }
  
  tsk->requires(Task::NewDW, d_lab->d_uVelocitySPBCLabel, gaf, 1);
  tsk->requires(Task::NewDW, d_lab->d_vVelocitySPBCLabel, gaf, 1);
  tsk->requires(Task::NewDW, d_lab->d_wVelocitySPBCLabel, gaf, 1);

  Task::WhichDW which_dw = Task::NewDW;
  if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First) {
    which_dw = Task::OldDW;
  }

  tsk->requires(which_dw, d_lab->d_tempINLabel,  gac, 1);
  tsk->requires(which_dw, d_lab->d_cpINLabel,    gac, 1);
  

  if (d_radiationCalc) {
    tsk->requires(Task::NewDW, d_lab->d_radiationSRCINLabel, gn, 0);
  } 

  if (d_MAlab && d_boundaryCondition->getIfCalcEnergyExchange()) {
    tsk->requires(Task::NewDW, d_MAlab->d_enth_mmLinSrc_CCLabel,   gn, 0);
    tsk->requires(Task::NewDW, d_MAlab->d_enth_mmNonLinSrc_tmp_CCLabel,gn, 0);
  }

  if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First ) {
    tsk->computes(d_lab->d_enthCoefSBLMLabel, d_lab->d_stencilMatl, oams);
    tsk->computes(d_lab->d_enthDiffCoefLabel, d_lab->d_stencilMatl, oams);
    tsk->computes(d_lab->d_enthNonLinSrcSBLMLabel);
  }
  else {
    tsk->modifies(d_lab->d_enthCoefSBLMLabel, d_lab->d_stencilMatl, oams);
    tsk->modifies(d_lab->d_enthDiffCoefLabel, d_lab->d_stencilMatl, oams);
    tsk->modifies(d_lab->d_enthNonLinSrcSBLMLabel);
  }
  
   //__________________________________
   // Adding new sources from factory:
   SourceTermFactory& factory = SourceTermFactory::self(); 
   for (vector<std::string>::iterator iter = d_new_sources.begin(); 
       iter != d_new_sources.end(); iter++){

     SourceTermBase& src = factory.retrieve_source_term( *iter ); 
     const VarLabel* srcLabel = src.getSrcLabel(); 
     tsk->requires(Task::NewDW, srcLabel, gn, 0); 
  }

  tsk->requires(Task::NewDW, timelabels->negativeDensityGuess);

  tsk->modifies(d_lab->d_enthalpyBoundarySrcLabel);                  

  //  sched->addTask(tsk, patches, matls);
  sched->addTask(tsk, d_perproc_patches, matls);


}

      
//****************************************************************************
// Actually build linear matrix
//****************************************************************************
void EnthalpySolver::buildLinearMatrix(const ProcessorGroup* pc,
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

  constCCVariable<double> solidTemp;

  double negativeDensityGuess = 0.0;
  sum_vartype nDG;
  
  new_dw->get(nDG, timelabels->negativeDensityGuess);
  negativeDensityGuess = nDG;
  
  Ghost::GhostType  gac = Ghost::AroundCells;
  Ghost::GhostType  gaf = Ghost::AroundFaces;
  Ghost::GhostType  gn = Ghost::None;      

  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int indx = d_lab->d_sharedState->
                    getArchesMaterial(archIndex)->getDWIndex(); 
    ArchesVariables enthalpyVars;
    ArchesConstVariables constEnthalpyVars;
    
    // Get the PerPatch CellInformation data
    PerPatch<CellInformationP> cellInfoP;
    new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, indx, patch);
    CellInformation* cellinfo = cellInfoP.get().get_rep();

    new_dw->get(constEnthalpyVars.cellType, d_lab->d_cellTypeLabel, indx, patch, gac, 1);

    DataWarehouse* old_values_dw;
    if (timelabels->use_old_values){
      old_values_dw = parent_old_dw;
    }else{ 
      old_values_dw = new_dw;
    }
    old_values_dw->get(constEnthalpyVars.old_enthalpy, d_lab->d_enthalpySPLabel, indx, patch, gn, 0);
    old_values_dw->get(constEnthalpyVars.old_density,  d_lab->d_densityCPLabel,  indx, patch, gn, 0);
    new_dw->get(       constEnthalpyVars.enthalpy,     d_lab->d_enthalpySPLabel, indx, patch, gn, 0);
   
    new_dw->get(constEnthalpyVars.density, d_lab->d_densityCPLabel, indx, patch,  gac, 2);

    //__________________________________
    //
    if (d_dynScalarModel){
      new_dw->get(constEnthalpyVars.viscosity, d_lab->d_enthalpyDiffusivityLabel, indx, patch, gac, 2);
    }else {
      new_dw->get(constEnthalpyVars.viscosity, d_lab->d_viscosityCTSLabel,        indx, patch,  gac, 2);
    }
    
    if (d_conv_scheme > 0) {
      new_dw->get(constEnthalpyVars.scalar, d_lab->d_enthalpySPLabel, indx, patch,  gac, 2);
    }
    
    //__________________________________
    //
    new_dw->get(constEnthalpyVars.uVelocity, d_lab->d_uVelocitySPBCLabel,  indx, patch, gaf, 1);
    new_dw->get(constEnthalpyVars.vVelocity, d_lab->d_vVelocitySPBCLabel,  indx, patch, gaf, 1);
    new_dw->get(constEnthalpyVars.wVelocity, d_lab->d_wVelocitySPBCLabel,  indx, patch, gaf, 1);
    
    //__________________________________
    //
    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First){
      old_dw->getCopy(enthalpyVars.temperature, d_lab->d_tempINLabel, indx, patch, gac, 1);
      old_dw->get(constEnthalpyVars.cp,         d_lab->d_cpINLabel,   indx, patch, gac, 1);
    }
    else {
      new_dw->getCopy(enthalpyVars.temperature, d_lab->d_tempINLabel, indx, patch, gac, 1);
      new_dw->get(constEnthalpyVars.cp,         d_lab->d_cpINLabel,   indx, patch, gac, 1);
    }

    //__________________________________
    // allocate matrix coeffs
    if ( timelabels->integrator_step_number == TimeIntegratorStepNumber::First ) {
      for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++) {
        new_dw->allocateAndPut( enthalpyVars.scalarCoeff[ii],         d_lab->d_enthCoefSBLMLabel, ii, patch);
        new_dw->allocateAndPut( enthalpyVars.scalarDiffusionCoeff[ii],d_lab->d_enthDiffCoefLabel, ii, patch);

        enthalpyVars.scalarCoeff[ii].initialize(0.0);
        enthalpyVars.scalarDiffusionCoeff[ii].initialize(0.0);
      }
      new_dw->allocateAndPut( enthalpyVars.scalarNonlinearSrc,        d_lab->d_enthNonLinSrcSBLMLabel, indx, patch);
      enthalpyVars.scalarNonlinearSrc.initialize(0.0);
    }
    else {
      for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++) {
        new_dw->getModifiable(enthalpyVars.scalarCoeff[ii],           d_lab->d_enthCoefSBLMLabel, ii, patch);
        new_dw->getModifiable(enthalpyVars.scalarDiffusionCoeff[ii],  d_lab->d_enthDiffCoefLabel, ii, patch);

        enthalpyVars.scalarCoeff[ii].initialize(0.0);
        enthalpyVars.scalarDiffusionCoeff[ii].initialize(0.0);
      }
      new_dw->getModifiable(enthalpyVars.scalarNonlinearSrc,       d_lab->d_enthNonLinSrcSBLMLabel, indx, patch);
      enthalpyVars.scalarNonlinearSrc.initialize(0.0);
    }


    //__________________________________
    // Adding new sources from factory:
    SourceTermFactory& factory = SourceTermFactory::self(); 
    for (vector<std::string>::iterator iter = d_new_sources.begin(); 
       iter != d_new_sources.end(); iter++){

      SourceTermBase& src = factory.retrieve_source_term( *iter ); 
      const VarLabel* srcLabel = src.getSrcLabel(); 
      new_dw->get( enthalpyVars.otherSource, srcLabel, indx, patch, Ghost::None, 0); 
    }
    
    //__________________________________
    //boundary source terms
    new_dw->getModifiable(enthalpyVars.enthalpyBoundarySrc, 
                          d_lab->d_enthalpyBoundarySrcLabel, indx, patch);

    for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++) {
      new_dw->allocateTemporary(enthalpyVars.scalarConvectCoeff[ii],  patch);
      enthalpyVars.scalarConvectCoeff[ii].initialize(0.0);
    }
    
    new_dw->allocateTemporary(enthalpyVars.scalarLinearSrc,  patch);
    enthalpyVars.scalarLinearSrc.initialize(0.0);


    if (d_MAlab && d_boundaryCondition->getIfCalcEnergyExchange()) {
      new_dw->get(constEnthalpyVars.mmEnthSu, d_MAlab->d_enth_mmNonLinSrc_tmp_CCLabel,indx, patch, gn, 0);
      new_dw->get(constEnthalpyVars.mmEnthSp, d_MAlab->d_enth_mmLinSrc_CCLabel,   indx, patch, gn, 0);
    }

    // compute ith component of enthalpy stencil coefficients
    // inputs : enthalpySP, [u,v,w]VelocityMS, densityCP, viscosityCTS
    // outputs: scalCoefSBLM
    d_discretize->calculateScalarCoeff(patch,
                                       cellinfo, 
                                       &enthalpyVars, &constEnthalpyVars,
                                       d_conv_scheme);

    // Calculate enthalpy source terms
    // inputs : [u,v,w]VelocityMS, enthalpySP, densityCP, viscosityCTS
    // outputs: scalLinSrcSBLM, scalNonLinSrcSBLM
    d_source->calculateEnthalpySource(pc, patch,
                                    delta_t, cellinfo, 
                                    &enthalpyVars, &constEnthalpyVars);

    // Add enthalpy source terms due to multimaterial intrusions
    if (d_MAlab && d_boundaryCondition->getIfCalcEnergyExchange()) {

      d_source->addMMEnthalpySource(pc, patch, cellinfo,
                                    &enthalpyVars, &constEnthalpyVars);

    }

    if (d_new_sources.size() > 0) {
      d_source->addOtherScalarSource(pc, patch, cellinfo, &enthalpyVars); 
    }

    if (d_conv_scheme > 0) {
      int wall_celltypeval = d_boundaryCondition->wallCellType();
      d_discretize->calculateScalarFluxLimitedConvection
                                                  (patch, cellinfo,
                                                  &enthalpyVars,
                                                  &constEnthalpyVars,
                                                  wall_celltypeval, 
                                                  d_limiter_type, 
                                                  d_boundary_limiter_type,
                                                  d_central_limiter); 
    }
// Adiabatic enthalpy fix for first timestep only
//    int currentTimeStep=d_lab->d_sharedState->getCurrentTopLevelTimeStep();
//    if (currentTimeStep == 1) {
    if (negativeDensityGuess > 0.0) {
      if (pc->myrank() == 0)
        cout << "Applying old density fix for enthalpy" << endl;
        
      IntVector idxLo = patch->getFortranCellLowIndex();
      IntVector idxHi = patch->getFortranCellHighIndex();
      double areaew, areans, areatb, div;
      for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
        for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
          for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {

              IntVector currCell(colX, colY, colZ);
              IntVector xplusCell(colX+1, colY, colZ);
              IntVector yplusCell(colX, colY+1, colZ);
              IntVector zplusCell(colX, colY, colZ+1);
              IntVector xminusCell(colX-1, colY, colZ);
              IntVector yminusCell(colX, colY-1, colZ);
              IntVector zminusCell(colX, colY, colZ-1);
              areaew = cellinfo->sns[colY] * cellinfo->stb[colZ];
              areans = cellinfo->sew[colX] * cellinfo->stb[colZ];
              areatb = cellinfo->sew[colX] * cellinfo->sns[colY];

              div = (constEnthalpyVars.uVelocity[xplusCell] * 0.5 *
                     (constEnthalpyVars.density[currCell] +
                      constEnthalpyVars.density[xplusCell]) -
                     constEnthalpyVars.uVelocity[currCell] * 0.5 *
                     (constEnthalpyVars.density[currCell] +
                      constEnthalpyVars.density[xminusCell])) * areaew +
                    (constEnthalpyVars.vVelocity[yplusCell] * 0.5 *
                     (constEnthalpyVars.density[currCell] +
                      constEnthalpyVars.density[yplusCell]) -
                     constEnthalpyVars.vVelocity[currCell] * 0.5 *
                     (constEnthalpyVars.density[currCell] +
                      constEnthalpyVars.density[yminusCell])) * areans +
                    (constEnthalpyVars.wVelocity[zplusCell] * 0.5 *
                     (constEnthalpyVars.density[currCell] +
                      constEnthalpyVars.density[zplusCell]) -
                     constEnthalpyVars.wVelocity[currCell] * 0.5 *
                     (constEnthalpyVars.density[currCell] +
                      constEnthalpyVars.density[zminusCell])) * areatb;

              enthalpyVars.scalarNonlinearSrc[currCell] += div * d_H_air;
          }
        }
      }
    }

    //__________________________________
    //  Radiation contribution
    if (d_radiationCalc) {
      constCCVariable<double> radSrc;
      new_dw->get(radSrc,    d_lab->d_radiationSRCINLabel,  indx, patch, gn, 0);

      for(CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
        IntVector c = *iter;
        int i = c.x();
        int j = c.y();
        int k = c.z();
        double vol=cellinfo->sew[i]*cellinfo->sns[j]*cellinfo->stb[k];
        enthalpyVars.scalarNonlinearSrc[c] += vol * radSrc[c];
      }  
    }

    //__________________________________
    // Calculate the enthalpy boundary conditions
    // inputs : enthalpySP, scalCoefSBLM
    // outputs: scalCoefSBLM
    if (d_boundaryCondition->anyArchesPhysicalBC()) {
      d_boundaryCondition->scalarBC(patch, &enthalpyVars, &constEnthalpyVars);
    }

    if (d_boundaryCondition->isUsingNewBC()) {
      d_boundaryCondition->scalarBC(patch, &enthalpyVars, &constEnthalpyVars);

    }

    // apply multimaterial intrusion wallbc
    if (d_MAlab)
      d_boundaryCondition->mmscalarWallBC( patch, cellinfo,
                                          &enthalpyVars, &constEnthalpyVars);

    // similar to mascal
    // inputs :
    // outputs:
    d_source->modifyEnthalpyMassSource(pc, patch, delta_t,
                                       &enthalpyVars, &constEnthalpyVars,
                                       d_conv_scheme);
    
    // Calculate the enthalpy diagonal terms
    // inputs : scalCoefSBLM, scalLinSrcSBLM
    // outputs: scalCoefSBLM
    d_discretize->calculateScalarDiagonal(patch, &enthalpyVars);

  }
}


//****************************************************************************
// Schedule linear solve of enthalpy
//****************************************************************************
void
EnthalpySolver::sched_enthalpyLinearSolve(SchedulerP& sched,
                                          const PatchSet* patches,
                                          const MaterialSet* matls,
                                          const TimeIntegratorLabel* timelabels )
{
  string taskname =  "EnthalpySolver::enthalpyLinearSolve" + 
                     timelabels->integrator_step_name;

  Task* tsk = scinew Task(taskname, this,
                          &EnthalpySolver::enthalpyLinearSolve,
                          timelabels );
  
  Task::WhichDW parent_old_dw;
  if (timelabels->recursion){
    parent_old_dw = Task::ParentOldDW;
  }else{ 
    parent_old_dw = Task::OldDW;
  }
  
  Ghost::GhostType  gac = Ghost::AroundCells;
  Ghost::GhostType  gn = Ghost::None;
  Task::MaterialDomainSpec oams = Task::OutOfDomain;  //outside of arches matlSet.
  
  tsk->requires(parent_old_dw, d_lab->d_sharedState->get_delt_label());
  
  tsk->requires(Task::NewDW,   d_lab->d_densityGuessLabel, gn, 0);
  tsk->requires(Task::NewDW,   d_lab->d_cellInfoLabel,     gn, 0);
  tsk->requires(Task::NewDW,   d_lab->d_cellTypeLabel,     gac, 1);
  

  if (timelabels->multiple_steps){
    tsk->requires(Task::NewDW, d_lab->d_enthalpyTempLabel, gac, 1);
  }else{
    tsk->requires(Task::OldDW, d_lab->d_enthalpySPLabel,   gac, 1);
  }
  
  tsk->requires(Task::NewDW,   d_lab->d_enthCoefSBLMLabel, 
                               d_lab->d_stencilMatl, oams, gn, 0);
                               
  tsk->requires(Task::NewDW,   d_lab->d_enthNonLinSrcSBLMLabel, gn, 0);

  if (d_MAlab) {
    tsk->requires(Task::NewDW, d_lab->d_mmgasVolFracLabel,gn, 0);
  } 
 
  tsk->modifies(d_lab->d_enthalpySPLabel);
  
  sched->addTask(tsk, patches, matls);
}
//****************************************************************************
// Actual enthalpy solve .. may be changed after recursive tasks are added
//****************************************************************************
void 
EnthalpySolver::enthalpyLinearSolve(const ProcessorGroup* pc,
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
  Ghost::GhostType  gac = Ghost::AroundCells;
  Ghost::GhostType  gn = Ghost::None;


  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int indx = d_lab->d_sharedState->
                    getArchesMaterial(archIndex)->getDWIndex(); 
    ArchesVariables enthalpyVars;
    ArchesConstVariables constEnthalpyVars;

    PerPatch<CellInformationP> cellInfoP;
    new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, indx, patch);
    CellInformation* cellinfo = cellInfoP.get().get_rep();

    new_dw->get(constEnthalpyVars.cellType,      d_lab->d_cellTypeLabel,     indx, patch, gac, 1);
    new_dw->get(constEnthalpyVars.density_guess, d_lab->d_densityGuessLabel, indx, patch, gn, 0);

    if (timelabels->multiple_steps)
      new_dw->get(constEnthalpyVars.old_enthalpy, d_lab->d_enthalpyTempLabel,indx, patch, gac, 1);
    else
      old_dw->get(constEnthalpyVars.old_enthalpy, d_lab->d_enthalpySPLabel,  indx, patch, gac, 1);

    // for explicit calculation
    new_dw->getModifiable(enthalpyVars.enthalpy,  d_lab->d_enthalpySPLabel,  indx, patch);
     new_dw->getModifiable(enthalpyVars.scalar,   d_lab->d_enthalpySPLabel,  indx, patch);

    for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++){
      new_dw->get(constEnthalpyVars.scalarCoeff[ii],
                  d_lab->d_enthCoefSBLMLabel, 
                  ii, patch, gn, 0);
    }
                  
    new_dw->get(constEnthalpyVars.scalarNonlinearSrc,
                d_lab->d_enthNonLinSrcSBLMLabel,indx, patch, gn, 0);


    if (d_MAlab) {
      new_dw->get(constEnthalpyVars.voidFraction, d_lab->d_mmgasVolFracLabel,indx, patch, gn, 0);
    }

    // make it a separate task later
    if (d_MAlab){ 
      d_boundaryCondition->enthalpyLisolve_mm(patch, delta_t, 
                                              &enthalpyVars, 
                                              &constEnthalpyVars, 
                                              cellinfo);
    }
    else{
      d_rhsSolver->enthalpyLisolve(pc, patch, delta_t, 
                                      &enthalpyVars, &constEnthalpyVars,
                                      cellinfo);
    }


// Outlet bc is done here not to change old enthalpy
    if ((d_boundaryCondition->getOutletBC())||
        (d_boundaryCondition->getPressureBC())) {
      d_boundaryCondition->scalarOutletPressureBC(patch,
                                            &enthalpyVars, &constEnthalpyVars);
    }

    if ( d_boundaryCondition->isUsingNewBC() ){ 
      d_boundaryCondition->scalarOutletPressureBC(patch,
                                            &enthalpyVars, &constEnthalpyVars);
    }
  }
}
