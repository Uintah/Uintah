/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
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


//----- ScalarSolver.cc ----------------------------------------------

#include <CCA/Components/Arches/ScalarSolver.h>
#include <CCA/Components/Arches/Arches.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/ArchesMaterial.h>
#include <CCA/Components/Arches/BoundaryCondition.h>
#include <CCA/Components/Arches/CellInformationP.h>
#include <CCA/Components/Arches/Discretization.h>
#include <CCA/Components/Arches/PetscSolver.h>
#include <CCA/Components/Arches/PhysicalConstants.h>
#include <CCA/Components/Arches/RHSSolver.h>
#include <CCA/Components/Arches/Source.h>
#include <CCA/Components/Arches/ScaleSimilarityModel.h>
#include <CCA/Components/Arches/TimeIntegratorLabel.h>
#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Exceptions/VariableNotFoundInGrid.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermFactory.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermBase.h>

using namespace Uintah;
using namespace std;

//****************************************************************************
// Default constructor for ScalarSolver
//****************************************************************************
ScalarSolver::ScalarSolver(const ArchesLabel* label,
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
ScalarSolver::~ScalarSolver()
{
  delete d_discretize;
  delete d_source;
  delete d_rhsSolver;
}

//****************************************************************************
// Problem Setup
//****************************************************************************
void 
ScalarSolver::problemSetup(const ProblemSpecP& params)
{
  ProblemSpecP db = params->findBlock("MixtureFractionSolver");

  d_discretize = scinew Discretization();

  string conv_scheme;
  db->getWithDefault("convection_scheme",conv_scheme,"central-upwind");
  
  if (conv_scheme == "central-upwind"){
    d_conv_scheme = 0;
  }else if (conv_scheme == "flux_limited"){
    d_conv_scheme = 1;
  }else {
    throw InvalidValue("Convection scheme not supported: " + conv_scheme, __FILE__, __LINE__);
  }
  
  string limiter_type;
  if (d_conv_scheme == 1) {
    db->getWithDefault("limiter_type",limiter_type,"superbee");
    if (limiter_type == "minmod"){
      d_limiter_type = -1;
    }else if (limiter_type == "superbee"){
      d_limiter_type = 0;
    }else if (limiter_type == "vanLeer"){
      d_limiter_type = 1;
    }else if (limiter_type == "none") {
      d_limiter_type = 2;
      cout << "WARNING! Running central scheme for scalar," << endl;
      cout << "which can be unstable." << endl;
    }else if (limiter_type == "central-upwind"){
      d_limiter_type = 3;
    }else if (limiter_type == "upwind"){
      d_limiter_type = 4;
    }else{ 
      throw InvalidValue("Flux limiter type "
                                           "not supported: " + limiter_type, __FILE__, __LINE__);
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
    if (db->findBlock("turbulentPrandtlNumber")){
      db->getWithDefault("turbulentPrandtlNumber",d_turbPrNo,0.4);
    }
    // if it is not set in both places
    if ((d_turbPrNo == 0.0)&&(model_turbPrNo == 0.0)){
          throw InvalidValue("Turbulent Prandtl number is not specified for"
                             "mixture fraction ", __FILE__, __LINE__);
    // if it is set in turbulence model
    }else if (d_turbPrNo == 0.0){
      d_turbPrNo = model_turbPrNo;
    }

    // if it is set here or set in both places, 
    // we only need to set mixture fraction Pr number in turbulence model
    if (!(model_turbPrNo == d_turbPrNo)) {
      cout << "Turbulent Prandtl number for mixture fraction is set to "
      << d_turbPrNo << endl;
      d_turbModel->setTurbulentPrandtlNumber(d_turbPrNo);
    }
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

  d_discretize->setTurbulentPrandtlNumber(d_turbPrNo);
}

//****************************************************************************
// Schedule solve of linearized scalar equation
//****************************************************************************
void 
ScalarSolver::solve(SchedulerP& sched,
                    const PatchSet* patches,
                    const MaterialSet* matls,
                    const TimeIntegratorLabel* timelabels)
{
  //computes stencil coefficients and source terms
  // requires : scalarIN, [u,v,w]VelocitySPBC, densityIN, viscosityIN
  // computes : scalCoefSBLM, scalLinSrcSBLM, scalNonLinSrcSBLM
  sched_buildLinearMatrix(sched, patches, matls, timelabels);
  
  // Schedule the scalar solve
  // require : scalarIN, scalCoefSBLM, scalNonLinSrcSBLM
  sched_scalarLinearSolve(sched, patches, matls, timelabels);
}

//****************************************************************************
// Schedule build of linear matrix
//****************************************************************************
void 
ScalarSolver::sched_buildLinearMatrix(SchedulerP& sched,
                                      const PatchSet* patches,
                                      const MaterialSet* matls,
                                      const TimeIntegratorLabel* timelabels)
{
  string taskname =  "ScalarSolver::BuildCoeff" +
                     timelabels->integrator_step_name;

  Task* tsk = scinew Task(taskname, this,
                          &ScalarSolver::buildLinearMatrix,
                          timelabels);


  Task::WhichDW parent_old_dw;
  if (timelabels->recursion){
    parent_old_dw = Task::ParentOldDW;
  }else{
    parent_old_dw = Task::OldDW;
  }
  
  tsk->requires(parent_old_dw, d_lab->d_sharedState->get_delt_label());
  
  // This task requires scalar and density from old time step for transient
  // calculation
  //DataWarehouseP old_dw = new_dw->getTop();
  Ghost::GhostType  gac = Ghost::AroundCells;
  Ghost::GhostType  gaf = Ghost::AroundFaces;
  Ghost::GhostType  gn = Ghost::None;  
  Task::DomainSpec oams = Task::OutOfDomain;  //outside of arches matlSet.
  
  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel,  gac, 1);
  tsk->requires(Task::NewDW, d_lab->d_scalarSPLabel,  gac, 2);
  tsk->requires(Task::NewDW, d_lab->d_densityCPLabel, gac, 2);

  Task::WhichDW old_values_dw;
  if (timelabels->use_old_values){
    old_values_dw = parent_old_dw;
  }else{ 
    old_values_dw = Task::NewDW;
  }
  tsk->requires(old_values_dw, d_lab->d_scalarSPLabel,  gn, 0);
  tsk->requires(old_values_dw, d_lab->d_densityCPLabel, gn, 0);
  
  tsk->requires(Task::NewDW, d_lab->d_cellInfoLabel, gn);
  
  if (d_dynScalarModel){
    tsk->requires(Task::NewDW, d_lab->d_scalarDiffusivityLabel,gac, 2);
  }else{
    tsk->requires(Task::NewDW, d_lab->d_viscosityCTSLabel,     gac, 2);
  }
  tsk->requires(Task::NewDW, d_lab->d_uVelocitySPBCLabel, gaf, 1);
  tsk->requires(Task::NewDW, d_lab->d_vVelocitySPBCLabel, gaf, 1);
  tsk->requires(Task::NewDW, d_lab->d_wVelocitySPBCLabel, gaf, 1);

  if (dynamic_cast<const ScaleSimilarityModel*>(d_turbModel)) {
    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First){
      tsk->requires(Task::OldDW, d_lab->d_scalarFluxCompLabel,
          d_lab->d_vectorMatl, oams, gac, 1);
    }else{
      tsk->requires(Task::NewDW, d_lab->d_scalarFluxCompLabel,
          d_lab->d_vectorMatl, oams,gac, 1);
    }
  }

      // added one more argument of index to specify scalar component
  if ( timelabels->integrator_step_number == TimeIntegratorStepNumber::First ) {

    // -------- New Coefficient Stuff -------------
    tsk->computes(d_lab->d_scalarTotCoefLabel); 

    // --------------------------------------------

    tsk->computes(d_lab->d_scalCoefSBLMLabel, d_lab->d_stencilMatl, oams);
    tsk->computes(d_lab->d_scalDiffCoefLabel, d_lab->d_stencilMatl, oams);
    tsk->computes(d_lab->d_scalNonLinSrcSBLMLabel);
//#ifdef divergenceconstraint
    tsk->computes(d_lab->d_scalDiffCoefSrcLabel);
//#endif
    tsk->modifies(d_lab->d_scalarBoundarySrcLabel);

    // Adding new sources from factory:
    SourceTermFactory& factory = SourceTermFactory::self(); 
    for (vector<std::string>::iterator iter = d_new_sources.begin(); 
        iter != d_new_sources.end(); iter++){

      SourceTermBase& src = factory.retrieve_source_term( *iter ); 
      const VarLabel* srcLabel = src.getSrcLabel(); 
      tsk->requires(Task::OldDW, srcLabel, gn, 0); 

    }
  }else {

    // -------- New Coefficient Stuff -------------
    tsk->modifies(d_lab->d_scalarTotCoefLabel); 

    // --------------------------------------------

    tsk->modifies(d_lab->d_scalCoefSBLMLabel, d_lab->d_stencilMatl, oams);
    tsk->modifies(d_lab->d_scalDiffCoefLabel, d_lab->d_stencilMatl, oams);
    tsk->modifies(d_lab->d_scalNonLinSrcSBLMLabel);
//#ifdef divergenceconstraint
    tsk->modifies(d_lab->d_scalDiffCoefSrcLabel);
//#endif
    tsk->modifies(d_lab->d_scalarBoundarySrcLabel);

    // Adding new sources from factory:
    SourceTermFactory& factory = SourceTermFactory::self(); 
    for (vector<std::string>::iterator iter = d_new_sources.begin(); 
        iter != d_new_sources.end(); iter++){

      SourceTermBase& src = factory.retrieve_source_term( *iter ); 
      const VarLabel* srcLabel = src.getSrcLabel(); 
      tsk->requires(Task::NewDW, srcLabel, gn, 0); 

    }
  }       
  sched->addTask(tsk, patches, matls);
}

      
//****************************************************************************
// Actually build linear matrix
//****************************************************************************
void ScalarSolver::buildLinearMatrix(const ProcessorGroup* pc,
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

  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int indx = d_lab->d_sharedState->
                    getArchesMaterial(archIndex)->getDWIndex(); 
    ArchesVariables scalarVars;
    ArchesConstVariables constScalarVars;
    
    // Get the PerPatch CellInformation data
    PerPatch<CellInformationP> cellInfoP;
    new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, indx, patch);


    DataWarehouse* old_values_dw;
    if (timelabels->use_old_values){
      old_values_dw = parent_old_dw;
    }else{ 
      old_values_dw = new_dw;
    }
    
    //__________________________________
    Ghost::GhostType  gac = Ghost::AroundCells;
    Ghost::GhostType  gaf = Ghost::AroundFaces;
    Ghost::GhostType  gn = Ghost::None;

    CellInformation* cellinfo = cellInfoP.get().get_rep();
    new_dw->get(constScalarVars.cellType,           d_lab->d_cellTypeLabel, indx, patch, gac, 1);
    
    old_values_dw->get(constScalarVars.old_scalar,  d_lab->d_scalarSPLabel,  indx, patch, gn, 0);
    old_values_dw->get(constScalarVars.old_density, d_lab->d_densityCPLabel, indx, patch, gn, 0);
    new_dw->get(       constScalarVars.density,     d_lab->d_densityCPLabel, indx, patch, gac, 2);

    if (d_dynScalarModel){
      new_dw->get(constScalarVars.viscosity, d_lab->d_scalarDiffusivityLabel,indx, patch, gac, 2);
    }else{
      new_dw->get(constScalarVars.viscosity, d_lab->d_viscosityCTSLabel,     indx, patch, gac, 2);
    }
    new_dw->get(constScalarVars.scalar,    d_lab->d_scalarSPLabel,      indx, patch, gac, 2);
    // for explicit get old values
    new_dw->get(constScalarVars.uVelocity, d_lab->d_uVelocitySPBCLabel, indx, patch, gaf, 1);
    new_dw->get(constScalarVars.vVelocity, d_lab->d_vVelocitySPBCLabel, indx, patch, gaf, 1);
    new_dw->get(constScalarVars.wVelocity, d_lab->d_wVelocitySPBCLabel, indx, patch, gaf, 1);

   //__________________________________
   // allocate matrix coeffs
  if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First) {
 
    //New coefficients:----------------
    new_dw->allocateAndPut(scalarVars.scalarTotCoef, 
                           d_lab->d_scalarTotCoefLabel, indx, patch, gn, 0);

    //---------------------------------   

    for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++) {
      new_dw->allocateAndPut(scalarVars.scalarCoeff[ii],
                             d_lab->d_scalCoefSBLMLabel, ii, patch);
      scalarVars.scalarCoeff[ii].initialize(0.0);
      
      new_dw->allocateAndPut(scalarVars.scalarDiffusionCoeff[ii],
                             d_lab->d_scalDiffCoefLabel, ii, patch);
      scalarVars.scalarDiffusionCoeff[ii].initialize(0.0);
    }
    new_dw->allocateAndPut(scalarVars.scalarNonlinearSrc,
                           d_lab->d_scalNonLinSrcSBLMLabel, indx, patch);
    scalarVars.scalarNonlinearSrc.initialize(0.0);
//#ifdef divergenceconstraint
    new_dw->allocateAndPut(scalarVars.scalarDiffNonlinearSrc,
                           d_lab->d_scalDiffCoefSrcLabel, indx, patch);
    scalarVars.scalarDiffNonlinearSrc.initialize(0.0);
//#endif
    new_dw->getModifiable(scalarVars.scalarBoundarySrc,
                                d_lab->d_scalarBoundarySrcLabel, indx, patch);

    // Adding new sources from factory:
    SourceTermFactory& factory = SourceTermFactory::self(); 
    for (vector<std::string>::iterator iter = d_new_sources.begin(); 
       iter != d_new_sources.end(); iter++){

      SourceTermBase& src = factory.retrieve_source_term( *iter ); 
      const VarLabel* srcLabel = src.getSrcLabel(); 
      // here we have made the assumption that the momentum source is always a vector... 
      // and that we only have one.  probably want to fix this. 
      old_dw->get( scalarVars.otherSource, srcLabel, indx, patch, Ghost::None, 0); 

    }

  }else {
    for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++) {
      new_dw->getModifiable(scalarVars.scalarCoeff[ii],
                            d_lab->d_scalCoefSBLMLabel, ii, patch);
      scalarVars.scalarCoeff[ii].initialize(0.0);
      
      new_dw->getModifiable(scalarVars.scalarDiffusionCoeff[ii],
                            d_lab->d_scalDiffCoefLabel, ii, patch);
      scalarVars.scalarDiffusionCoeff[ii].initialize(0.0);
    }
    //New coefficients:----------------
    new_dw->getModifiable(scalarVars.scalarTotCoef, 
                          d_lab->d_scalarTotCoefLabel, indx, patch);
    //---------------------------------   
 
    new_dw->getModifiable(scalarVars.scalarNonlinearSrc,
                          d_lab->d_scalNonLinSrcSBLMLabel, indx, patch);
    scalarVars.scalarNonlinearSrc.initialize(0.0);
//#ifdef divergenceconstraint
    new_dw->getModifiable(scalarVars.scalarDiffNonlinearSrc,
                          d_lab->d_scalDiffCoefSrcLabel, indx, patch);
    scalarVars.scalarDiffNonlinearSrc.initialize(0.0);
//#endif
    new_dw->getModifiable(scalarVars.scalarBoundarySrc,
                          d_lab->d_scalarBoundarySrcLabel, indx, patch);
 
    // Adding new sources from factory:
    SourceTermFactory& factory = SourceTermFactory::self(); 
    for (vector<std::string>::iterator iter = d_new_sources.begin(); 
       iter != d_new_sources.end(); iter++){

      SourceTermBase& src = factory.retrieve_source_term( *iter ); 
      const VarLabel* srcLabel = src.getSrcLabel(); 
      // here we have made the assumption that the momentum source is always a vector... 
      // and that we only have one.  probably want to fix this. 
      new_dw->get( scalarVars.otherSource, srcLabel, indx, patch, Ghost::None, 0); 

    }
  }

  for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++) {
    new_dw->allocateTemporary(scalarVars.scalarConvectCoeff[ii],  patch);
    scalarVars.scalarConvectCoeff[ii].initialize(0.0);
  }
  new_dw->allocateTemporary(scalarVars.scalarLinearSrc,  patch);
  scalarVars.scalarLinearSrc.initialize(0.0);
  // compute ith component of scalar stencil coefficients
  // inputs : scalarSP, [u,v,w]VelocityMS, densityCP, viscosityCTS
  // outputs: scalCoefSBLM
  d_discretize->calculateScalarCoeff(patch,
                                     cellinfo, 
                                     &scalarVars, &constScalarVars,
                                     d_conv_scheme);

  //new stuff-----------------------**DONE**
  new_dw->allocateTemporary(scalarVars.scalarConvCoef, patch); 
  new_dw->allocateTemporary(scalarVars.scalarDiffCoef, patch); 
  //calculateScalarCoeff__new(pc, patch,
  //                                   delta_t, cellinfo, 
  //                                   &scalarVars, &constScalarVars,
  //                                   d_conv_scheme);


   // Calculate scalar source terms
   // inputs : [u,v,w]VelocityMS, scalarSP, densityCP, viscosityCTS
   // outputs: scalLinSrcSBLM, scalNonLinSrcSBLM
   d_source->calculateScalarSource(pc, patch,
                                  delta_t, cellinfo, 
                                  &scalarVars, &constScalarVars);
   if (d_new_sources.size() > 0) {
     d_source->addOtherScalarSource(pc, patch, cellinfo, &scalarVars); 
   }

   //---NEW Source term calculation **DONE**
   //d_source->calculateScalarSource__new(pc, patch,
   //                               delta_t, cellinfo, 
   //                               &scalarVars, &constScalarVars);

   if (d_doMMS){
    d_source->calculateScalarMMSSource(pc, patch,
                                    delta_t, cellinfo, 
                                    &scalarVars, &constScalarVars);
    }
    if (d_conv_scheme > 0) {
      int wall_celltypeval = d_boundaryCondition->wallCellType();
      d_discretize->calculateScalarFluxLimitedConvection
                                                  (patch,  cellinfo,
                                                    &scalarVars, &constScalarVars,
                                                  wall_celltypeval, 
                                                  d_limiter_type,
                                                  d_boundary_limiter_type,
                                                  d_central_limiter); 
    } 

    // for scalesimilarity model add scalarflux to the source of scalar eqn.
    if (dynamic_cast<const ScaleSimilarityModel*>(d_turbModel)) {
      StencilMatrix<constCCVariable<double> > scalarFlux; //3 point stencil
      
      if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First){
        for (int ii = 0; ii < d_lab->d_vectorMatl->size(); ii++) {
          old_dw->get(scalarFlux[ii], d_lab->d_scalarFluxCompLabel, ii, patch,gac, 1);
        }
      }else{
        for (int ii = 0; ii < d_lab->d_vectorMatl->size(); ii++) {
          new_dw->get(scalarFlux[ii], d_lab->d_scalarFluxCompLabel, ii, patch, gac, 1);
        }
      }
      
      double sue, suw, sun, sus, sut, sub;
      for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
        IntVector c = *iter;
        int i = c.x();
        int j = c.y();
        int k = c.z();
        
        IntVector E(i+1, j, k);   IntVector W(i-1, j, k);
        IntVector N(i, j+1, k);   IntVector S(i, j-1, k);
        IntVector T(i, j, k+1);   IntVector B(i, j, k-1);
        
        sue = 0.5*cellinfo->sns[j]*cellinfo->stb[k]* (scalarFlux[0][c] + scalarFlux[0][E]);
        suw = 0.5*cellinfo->sns[j]*cellinfo->stb[k]* (scalarFlux[0][W] + scalarFlux[0][c]);
        sun = 0.5*cellinfo->sew[i]*cellinfo->stb[k]* (scalarFlux[1][c] + scalarFlux[1][N]);
        sus = 0.5*cellinfo->sew[i]*cellinfo->stb[k]* (scalarFlux[1][c] + scalarFlux[1][S]);
        sut = 0.5*cellinfo->sns[j]*cellinfo->sew[i]* (scalarFlux[2][c] + scalarFlux[2][T]);
        sub = 0.5*cellinfo->sns[j]*cellinfo->sew[i]* (scalarFlux[2][c] + scalarFlux[2][B]);
#if 1
        scalarVars.scalarNonlinearSrc[c] += suw-sue+sus-sun+sub-sut;
#ifdef divergenceconstraint
        scalarVars.scalarDiffNonlinearSrc[c] = suw-sue+sus-sun+sub-sut;
#endif
#endif
      }
    }
    // Calculate the scalar boundary conditions
    // inputs : scalarSP, scalCoefSBLM
    // outputs: scalCoefSBLM
    
    
    if (d_boundaryCondition->anyArchesPhysicalBC()) {
      d_boundaryCondition->scalarBC(patch,
                                    &scalarVars, &constScalarVars);
      //d_boundaryCondition->scalarBC__new(pc, patch, 
      //                              &scalarVars, &constScalarVars);
    }
    // apply multimaterial intrusion wallbc ...
    // NOTE: Why not do this in scalarBC?
    if (d_MAlab){
      d_boundaryCondition->mmscalarWallBC(patch, cellinfo,
                                          &scalarVars, &constScalarVars);
    }
    // similar to mascal
    d_source->modifyScalarMassSource(pc, patch, delta_t,
                                     &scalarVars, &constScalarVars,
                                     d_conv_scheme);
    // ----New modifyScalarMassSource --- **DONE**
    //d_source->modifyScalarMassSource__new(pc, patch, delta_t,
    //                                      &scalarVars, &constScalarVars,
    //                                      d_conv_scheme);

    
    // Calculate the scalar diagonal terms
    // inputs : scalCoefSBLM, scalLinSrcSBLM
    // outputs: scalCoefSBLM
    d_discretize->calculateScalarDiagonal(patch, &scalarVars);
    //d_discretize->calculateScalarDiagonal__new(patch, &scalarVars);

  }
}


//****************************************************************************
// Schedule linear solve of scalar
//****************************************************************************
void
ScalarSolver::sched_scalarLinearSolve(SchedulerP& sched,
                                      const PatchSet* patches,
                                      const MaterialSet* matls,
                                      const TimeIntegratorLabel* timelabels)
{
  string taskname =  "ScalarSolver::ScalarLinearSolve" + 
                     timelabels->integrator_step_name;

  Task* tsk = scinew Task(taskname, this,
                          &ScalarSolver::scalarLinearSolve,
                          timelabels);
  
  Task::WhichDW parent_old_dw;
  if (timelabels->recursion){
    parent_old_dw = Task::ParentOldDW;
  }else{ 
    parent_old_dw = Task::OldDW;
  }
  
  Ghost::GhostType  gac = Ghost::AroundCells;
  Ghost::GhostType  gn  = Ghost::None;
  Task::DomainSpec oams = Task::OutOfDomain;  //outside of arches matlSet.
  
  tsk->requires(parent_old_dw, d_lab->d_sharedState->get_delt_label());
  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel,     gac, 1);
  tsk->requires(Task::NewDW, d_lab->d_densityGuessLabel, gn,  0);
  
  tsk->requires(Task::NewDW, d_lab->d_cellInfoLabel, gn);
  
  if (timelabels->multiple_steps){
    tsk->requires(Task::NewDW, d_lab->d_scalarTempLabel, gac, 1);
  }else{
    tsk->requires(Task::OldDW, d_lab->d_scalarSPLabel,   gac, 1);
  }
  tsk->requires(Task::NewDW, d_lab->d_scalCoefSBLMLabel,  
                             d_lab->d_stencilMatl, oams, gn, 0);
                             
  tsk->requires(Task::NewDW, d_lab->d_scalNonLinSrcSBLMLabel, gn, 0);

  tsk->requires(Task::NewDW, d_lab->d_scalarTotCoefLabel, gn, 0);
  //tsk->modifies(d_lab->d_scalarTotCoefLabel);

  if (d_MAlab) {
    tsk->requires(Task::NewDW, d_lab->d_mmgasVolFracLabel, gn, 0);
  }    
 
  tsk->modifies(d_lab->d_scalarSPLabel);

  if (timelabels->recursion){
    tsk->computes(d_lab->d_ScalarClippedLabel);
  }
  
  sched->addTask(tsk, patches, matls);
}
//****************************************************************************
// Actual scalar solve .. may be changed after recursive tasks are added
//****************************************************************************
void 
ScalarSolver::scalarLinearSolve(const ProcessorGroup* pc,
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

  
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int indx = d_lab->d_sharedState->
                    getArchesMaterial(archIndex)->getDWIndex(); 
    ArchesVariables scalarVars;
    ArchesConstVariables constScalarVars;

    // Get the PerPatch CellInformation data
    PerPatch<CellInformationP> cellInfoP;
    new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, indx, patch);
    
    //__________________________________
    Ghost::GhostType  gac = Ghost::AroundCells;
    Ghost::GhostType  gn = Ghost::None;
    CellInformation* cellinfo = cellInfoP.get().get_rep();

    new_dw->get(constScalarVars.density_guess, d_lab->d_densityGuessLabel, indx, patch, gn, 0);

    if (timelabels->multiple_steps){
      new_dw->get(constScalarVars.old_scalar, d_lab->d_scalarTempLabel, indx, patch, gac, 1);
    }else{
      old_dw->get(constScalarVars.old_scalar, d_lab->d_scalarSPLabel,   indx, patch, gac, 1);
    }
    
    // for explicit calculation
    new_dw->getModifiable(scalarVars.scalar, d_lab->d_scalarSPLabel,  indx, patch);
    
    for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++){
      new_dw->get(constScalarVars.scalarCoeff[ii], 
                                                 d_lab->d_scalCoefSBLMLabel, ii, patch, gn, 0);
    }
    //new_dw->getModifiable(scalarVars.scalarTotCoef, 
    //                      d_lab->d_scalarTotCoefLabel, indx, patch);
    new_dw->get(constScalarVars.scalarTotCoef, d_lab->d_scalarTotCoefLabel, indx, patch, gn, 0);

    new_dw->get(constScalarVars.scalarNonlinearSrc,
                                                d_lab->d_scalNonLinSrcSBLMLabel, indx, patch, gn, 0);

    new_dw->get(constScalarVars.cellType,       d_lab->d_cellTypeLabel,     indx, patch, gac, 1);
    if (d_MAlab) {
      new_dw->get(constScalarVars.voidFraction, d_lab->d_mmgasVolFracLabel, indx, patch, gn, 0);
    }

    // make it a separate task later
    //-----new Explicit solver interface:
    // same interface for Arches with or without intrusions
    //int intrusionVal = d_boundaryCondition->getMMWallId();
    //bool doingMM = d_MAlab;
    //d_rhsSolver->scalarExplicitUpdate(pc, patch, delta_t, &scalarVars, &constScalarVars, cellinfo, doingMM, intrusionVal);

    if (d_MAlab){
      d_boundaryCondition->scalarLisolve_mm(patch, delta_t, 
                                            &scalarVars, &constScalarVars,
                                            cellinfo);
    }else{
 
      d_rhsSolver->scalarLisolve(pc, patch, delta_t, 
                                 &scalarVars, &constScalarVars,
                                 cellinfo);
    }
    
    //__________________________________
    //  CLAMP
    double scalar_clipped = 0.0;
    bool did_clipping_low = false;
    bool did_clipping_high = false;  

    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
      IntVector c = *iter;

      if (scalarVars.scalar[c] > 1.0) {

        scalarVars.scalar[c] = 1.0; 
        scalar_clipped = 1.0;
        did_clipping_low = true; 

        // uncomment this if you want to debug your clipping:
        //cout << "scalar got clipped to 1 at " << c
        //<< " , scalar value was " << scalarVars.scalar[c] 
        //<< " , density guess was " 
        //<< constScalarVars.density_guess[c] << endl;

      }  
      else if (scalarVars.scalar[c] < 0.0) {

        scalar_clipped = 1.0;
        scalarVars.scalar[c] = 0.0; 
        did_clipping_high = true; 

        // uncomment this if you want to debug your clipping: 
        //cout << "scalar got clipped to 0 at " << c
        //<< " , scalar value was " << scalarVars.scalar[c]
        //<< " , density guess was " 
        //<< constScalarVars.density_guess[c] << endl;
        //cout << "Try setting <scalarUnderflowCheck>true</scalarUnderflowCheck> "
        //<< "in the <ARCHES> section of the input file, "
        //<<"but it would only help for first time substep if RKSSP is used" << endl;
      }
    }

    IntVector low = patch->getCellLowIndex(); 
    IntVector high = patch->getCellHighIndex(); 
    if ( did_clipping_low ) {
      cout << "NOTICE: This patch had scalar UNDERflow that required clipping!" << endl;
      cout << "Patch bounds: " << low << " to " << high << endl;
    } 
    if ( did_clipping_high ) {
      cout << "NOTICE: This patch had scalar OVERflow that required clipping!" << endl;
      cout << "Patch bounds: " << low << " to " << high << endl;
    }
    
    if (timelabels->recursion){
      new_dw->put(max_vartype(scalar_clipped), d_lab->d_ScalarClippedLabel);
    }
    
    // Outlet bc is done here not to change old scalar
    if ((d_boundaryCondition->getOutletBC())||
        (d_boundaryCondition->getPressureBC())){
      d_boundaryCondition->scalarOutletPressureBC(patch, &scalarVars, &constScalarVars);
    }
  }  // patches
}
//---------------------------------------------------------------
// New scalar coefficient method
void 
ScalarSolver::calculateScalarCoeff__new( const ProcessorGroup*,
                                        const Patch* patch,
                                        double,
                                        CellInformation* cellinfo,
                                        ArchesVariables* vars,
                                        ArchesConstVariables* constvars,
                                        int conv_scheme)
{
  for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
    IntVector curr = *iter; 
    double tiny = 1.0E-20; //needed?

    // cell face areas and cell volume 
    double areaew = cellinfo->sns[curr.y()]*cellinfo->stb[curr.z()];  
    double areans = cellinfo->sew[curr.x()]*cellinfo->stb[curr.z()];
    double areatb = cellinfo->sew[curr.x()]*cellinfo->sns[curr.y()];
    double vol = cellinfo->sew[curr.x()]*cellinfo->sns[curr.y()]*cellinfo->stb[curr.z()];

    // -- convection coefficients -- 
    double ceo = 0.5 * (constvars->density[curr] + constvars->density[curr + IntVector(1,0,0)])
                  * constvars->uVelocity[curr + IntVector(1,0,0)] * areaew;
    double cwo = 0.5 * (constvars->density[curr] + constvars->density[curr - IntVector(1,0,0)])
                  * constvars->uVelocity[curr] * areaew; 
    double cno = 0.5 * (constvars->density[curr] + constvars->density[curr + IntVector(0,1,0)])
                  * constvars->vVelocity[curr + IntVector(0,1,0)] * areans; 
    double cso = 0.5 * (constvars->density[curr] + constvars->density[curr - IntVector(0,1,0)])
                  * constvars->vVelocity[curr] * areans; 
    double cto = 0.5 * (constvars->density[curr] + constvars->density[curr + IntVector(0,0,1)])
                  * constvars->wVelocity[curr + IntVector(0,0,1)] * areatb; 
    double cbo = 0.5 * (constvars->density[curr] + constvars->density[curr - IntVector(0,0,1)])
                  * constvars->wVelocity[curr] * areatb; 

    // not sure what this does, but it was labeled "new differencing stuff" in the fortran
    // we will also store the convection coefficient here too 
    vars->scalarConvCoef[curr].e = cellinfo->cee[curr.x()]*ceo + cellinfo->cwe[curr.x()]*cwo; 
    vars->scalarConvCoef[curr].w = cellinfo->cww[curr.x()]*cwo - cellinfo->cwe[curr.x()]*ceo;
    vars->scalarConvCoef[curr].n = cellinfo->cnn[curr.y()]*cno + cellinfo->csn[curr.y()]*cso; 
    vars->scalarConvCoef[curr].s = cellinfo->css[curr.y()]*cso - cellinfo->csn[curr.y()]*cno;
    vars->scalarConvCoef[curr].t = cellinfo->ctt[curr.z()]*cto + cellinfo->cbt[curr.z()]*cbo;  
    vars->scalarConvCoef[curr].b = cellinfo->cbb[curr.z()]*cbo - cellinfo->cbt[curr.z()]*cto;
    // Clamp these to zero to eliminate "noise"
    // Need a more elegant way to take care of this.
    // This may not even be needed since tiny = 1E-20
    vars->scalarConvCoef[curr].e = abs(vars->scalarConvCoef[curr].e) < tiny ? 0:vars->scalarConvCoef[curr].e;
    vars->scalarConvCoef[curr].w = abs(vars->scalarConvCoef[curr].w) < tiny ? 0:vars->scalarConvCoef[curr].w;
    vars->scalarConvCoef[curr].n = abs(vars->scalarConvCoef[curr].n) < tiny ? 0:vars->scalarConvCoef[curr].n;
    vars->scalarConvCoef[curr].s = abs(vars->scalarConvCoef[curr].s) < tiny ? 0:vars->scalarConvCoef[curr].s;
    vars->scalarConvCoef[curr].t = abs(vars->scalarConvCoef[curr].t) < tiny ? 0:vars->scalarConvCoef[curr].t;
    vars->scalarConvCoef[curr].b = abs(vars->scalarConvCoef[curr].b) < tiny ? 0:vars->scalarConvCoef[curr].b;

    // -- diffusion coefficients -- 
    // **NOTE removed warning about negative diffusion coefs related to stretching, need warning?
    vars->scalarDiffCoef[curr].e = 
                  (cellinfo->fac1ew[curr.x()]*constvars->viscosity[curr + IntVector(1,0,0)] +
                   cellinfo->fac2ew[curr.x()]*constvars->viscosity[curr + IntVector(cellinfo->e_shift[curr.x()],0,0)])/d_turbPrNo;
    vars->scalarDiffCoef[curr].w =
                  (cellinfo->fac3ew[curr.x()]*constvars->viscosity[curr - IntVector(1,0,0)] + 
                   cellinfo->fac4ew[curr.x()]*constvars->viscosity[curr + IntVector(cellinfo->w_shift[curr.x()],0,0)])/d_turbPrNo;
    vars->scalarDiffCoef[curr].n = 
                  (cellinfo->fac1ns[curr.y()]*constvars->viscosity[curr + IntVector(0,1,0)] + 
                   cellinfo->fac2ns[curr.y()]*constvars->viscosity[curr + IntVector(0,cellinfo->n_shift[curr.y()],0)])/d_turbPrNo;
    vars->scalarDiffCoef[curr].s = 
                  (cellinfo->fac3ns[curr.y()]*constvars->viscosity[curr - IntVector(0,1,0)] + 
                   cellinfo->fac4ns[curr.y()]*constvars->viscosity[curr + IntVector(0,cellinfo->s_shift[curr.y()],0)])/d_turbPrNo;
    vars->scalarDiffCoef[curr].t = 
                  (cellinfo->fac1tb[curr.z()]*constvars->viscosity[curr + IntVector(0,0,1)] + 
                   cellinfo->fac2tb[curr.z()]*constvars->viscosity[curr + IntVector(0,0,cellinfo->t_shift[curr.z()])])/d_turbPrNo;
    vars->scalarDiffCoef[curr].b = 
                  (cellinfo->fac3tb[curr.z()]*constvars->viscosity[curr - IntVector(0,0,1)] + 
                   cellinfo->fac4tb[curr.z()]*constvars->viscosity[curr + IntVector(0,0,cellinfo->b_shift[curr.z()])])/d_turbPrNo; 

    vars->scalarDiffCoef[curr].e *= areaew / cellinfo->dxep[curr.x()];
    vars->scalarDiffCoef[curr].w *= areaew / cellinfo->dxpw[curr.x()];
    vars->scalarDiffCoef[curr].n *= areans / cellinfo->dynp[curr.y()];
    vars->scalarDiffCoef[curr].s *= areans / cellinfo->dyps[curr.y()];
    vars->scalarDiffCoef[curr].t *= areatb / cellinfo->dztp[curr.z()];
    vars->scalarDiffCoef[curr].b *= areatb / cellinfo->dzpb[curr.z()];


    // -- choose a scheme and compute total coefficient --  
    // **NOTE might want to replace (int conv_scheme) with an enum?
    // **NOTE central differencing was turned off in fortran..turn back on?
    if (conv_scheme == 0) {
      //L2UP
      double coefE = vars->scalarDiffCoef[curr].e - 0.5*abs(vars->scalarConvCoef[curr].e);
      double coefW = vars->scalarDiffCoef[curr].w - 0.5*abs(vars->scalarConvCoef[curr].w);
      double coefN = vars->scalarDiffCoef[curr].n - 0.5*abs(vars->scalarConvCoef[curr].n);
      double coefS = vars->scalarDiffCoef[curr].s - 0.5*abs(vars->scalarConvCoef[curr].s);
      double coefT = vars->scalarDiffCoef[curr].t - 0.5*abs(vars->scalarConvCoef[curr].t);
      double coefB = vars->scalarDiffCoef[curr].b - 0.5*abs(vars->scalarConvCoef[curr].b);

      double signTest = coefE < 0 ? -1:1;
      vars->scalarTotCoef[curr].e = vars->scalarDiffCoef[curr].e*(1.0-max(0.0,signTest))
                                    + max(0.0,coefE) + max(0.0,-vars->scalarConvCoef[curr].e);
      signTest = coefW < 0 ? -1:1; 
      vars->scalarTotCoef[curr].w = vars->scalarDiffCoef[curr].w*(1.0-max(0.0,signTest))
                                    + max(0.0,coefW) + max(0.0,vars->scalarConvCoef[curr].w);
      signTest = coefN < 0 ? -1:1;
      vars->scalarTotCoef[curr].n = vars->scalarDiffCoef[curr].n*(1.0-max(0.0,signTest))
                                    + max(0.0,coefN) + max(0.0,-vars->scalarConvCoef[curr].n);
      signTest = coefS < 0 ? -1:1;
      vars->scalarTotCoef[curr].s = vars->scalarDiffCoef[curr].s*(1.0-max(0.0,signTest))
                                    + max(0.0,coefS) + max(0.0,vars->scalarConvCoef[curr].s); 
      signTest = coefT < 0 ? -1:1;
      vars->scalarTotCoef[curr].t = vars->scalarDiffCoef[curr].t*(1.0-max(0.0,signTest))
                                    + max(0.0,coefT) + max(0.0,-vars->scalarConvCoef[curr].t);
      signTest = coefB < 0 ? -1:1;
      vars->scalarTotCoef[curr].b = vars->scalarDiffCoef[curr].b*(1.0-max(0.0,signTest))
                                    + max(0.0,coefB) + max(0.0,vars->scalarConvCoef[curr].b);

      // This is making me shudder a little bit but it is needed for this scheme.  I am doing this 
      // here which removes it from the do loop in mascal_scalar
      vars->scalarConvCoef[curr].e = vars->scalarTotCoef[curr].e < 0.0 ? 0:vars->scalarConvCoef[curr].e;
      vars->scalarConvCoef[curr].w = vars->scalarTotCoef[curr].w < 0.0 ? 0:vars->scalarConvCoef[curr].w;
      vars->scalarConvCoef[curr].n = vars->scalarTotCoef[curr].n < 0.0 ? 0:vars->scalarConvCoef[curr].n;
      vars->scalarConvCoef[curr].s = vars->scalarTotCoef[curr].s < 0.0 ? 0:vars->scalarConvCoef[curr].s;
      vars->scalarConvCoef[curr].t = vars->scalarTotCoef[curr].t < 0.0 ? 0:vars->scalarConvCoef[curr].t;
      vars->scalarConvCoef[curr].b = vars->scalarTotCoef[curr].b < 0.0 ? 0:vars->scalarConvCoef[curr].b;
    }
    else if (conv_scheme == 1) {
      //LENO 
      //**Note: this should change to include the flux limiter stuff here and not put it in another place.
      //**Note: I don't think I need to bother in zeroing out the convection coef like the fortran code did 
      //        since I believe it is never used.
      vars->scalarTotCoef[curr].e = vars->scalarDiffCoef[curr].e;
      vars->scalarTotCoef[curr].w = vars->scalarDiffCoef[curr].w;
      vars->scalarTotCoef[curr].n = vars->scalarDiffCoef[curr].n;
      vars->scalarTotCoef[curr].s = vars->scalarDiffCoef[curr].s;
      vars->scalarTotCoef[curr].t = vars->scalarDiffCoef[curr].t;
      vars->scalarTotCoef[curr].b = vars->scalarDiffCoef[curr].b;

    }  
    else { 
      //UPWIND (default)
      double coefE = vars->scalarDiffCoef[curr].e - 0.5*abs(vars->scalarConvCoef[curr].e);
      double coefW = vars->scalarDiffCoef[curr].w - 0.5*abs(vars->scalarConvCoef[curr].w);
      double coefN = vars->scalarDiffCoef[curr].n - 0.5*abs(vars->scalarConvCoef[curr].n);
      double coefS = vars->scalarDiffCoef[curr].s - 0.5*abs(vars->scalarConvCoef[curr].s);
      double coefT = vars->scalarDiffCoef[curr].t - 0.5*abs(vars->scalarConvCoef[curr].t);
      double coefB = vars->scalarDiffCoef[curr].b - 0.5*abs(vars->scalarConvCoef[curr].b);
      
      double tew = ( coefE < 0 || coefW < 0 ) ? 0.0 : 1.0;
      double tns = ( coefN < 0 || coefS < 0 ) ? 0.0 : 1.0;
      double ttb = ( coefT < 0 || coefB < 0 ) ? 0.0 : 1.0;

      double cpe = constvars->density[curr]*
                  (cellinfo->efac[curr.x()]*constvars->uVelocity[curr+IntVector(1,0,0)] +
                   cellinfo->wfac[curr.x()]*constvars->uVelocity[curr])*vol/cellinfo->dxep[curr.x()]; 
      double cpw = constvars->density[curr]*
                  (cellinfo->efac[curr.x()]*constvars->uVelocity[curr+IntVector(1,0,0)] +
                   cellinfo->wfac[curr.x()]*constvars->uVelocity[curr])*vol/cellinfo->dxpw[curr.x()];
      double cpn = constvars->density[curr]*
                  (cellinfo->nfac[curr.y()]*constvars->vVelocity[curr+IntVector(0,1,0)] + 
                   cellinfo->sfac[curr.y()]*constvars->vVelocity[curr])*vol/cellinfo->dynp[curr.y()];
      double cps = constvars->density[curr]*
                  (cellinfo->nfac[curr.y()]*constvars->vVelocity[curr+IntVector(0,1,0)] + 
                   cellinfo->sfac[curr.y()]*constvars->vVelocity[curr])*vol/cellinfo->dyps[curr.y()];
      double cpt = constvars->density[curr]*
                  (cellinfo->tfac[curr.z()]*constvars->wVelocity[curr+IntVector(0,0,1)] + 
                   cellinfo->bfac[curr.z()]*constvars->wVelocity[curr])*vol/cellinfo->dztp[curr.z()];
      double cpb = constvars->density[curr]*
                  (cellinfo->tfac[curr.z()]*constvars->wVelocity[curr+IntVector(0,0,1)] + 
                   cellinfo->bfac[curr.z()]*constvars->wVelocity[curr])*vol/cellinfo->dzpb[curr.z()];

      double aec = -0.5*vars->scalarConvCoef[curr].e*tew + max(0.0,-cpe)*(1.0-tew);
      double awc =  0.5*vars->scalarConvCoef[curr].w*tew + max(0.0, cpw)*(1.0-tew);
      double anc = -0.5*vars->scalarConvCoef[curr].n*tns + max(0.0,-cpn)*(1.0-tns);
      double asc =  0.5*vars->scalarConvCoef[curr].s*tns + max(0.0, cps)*(1.0-tns);
      double atc = -0.5*vars->scalarConvCoef[curr].t*ttb + max(0.0,-cpt)*(1.0-ttb);
      double abc =  0.5*vars->scalarConvCoef[curr].b*ttb + max(0.0, cpb)*(1.0-ttb);

      vars->scalarTotCoef[curr].e = aec + vars->scalarDiffCoef[curr].e;
      vars->scalarTotCoef[curr].w = awc + vars->scalarDiffCoef[curr].w;
      vars->scalarTotCoef[curr].n = anc + vars->scalarDiffCoef[curr].n;
      vars->scalarTotCoef[curr].s = asc + vars->scalarDiffCoef[curr].s;
      vars->scalarTotCoef[curr].t = atc + vars->scalarDiffCoef[curr].t;
      vars->scalarTotCoef[curr].b = abc + vars->scalarDiffCoef[curr].b;

    }
  }
}

