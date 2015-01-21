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
#include <CCA/Components/Arches/TurbulenceModel.h>
#include <CCA/Components/Arches/ScaleSimilarityModel.h>
#include <CCA/Components/Arches/TimeIntegratorLabel.h>
#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Exceptions/VariableNotFoundInGrid.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/ProblemSpec/ProblemSpec.h>


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
    if (conv_scheme == "central-upwind") d_conv_scheme = 0;
      else if (conv_scheme == "flux_limited") d_conv_scheme = 1;
	else throw InvalidValue("Convection scheme not supported: " + conv_scheme, __FILE__, __LINE__);

  string limiter_type;
  if (d_conv_scheme == 1) {
    db->getWithDefault("limiter_type",limiter_type,"superbee");
    if (limiter_type == "superbee") d_limiter_type = 0;
      else if (limiter_type == "vanLeer") d_limiter_type = 1;
        else if (limiter_type == "none") {
	  d_limiter_type = 2;
	  cout << "WARNING! Running central scheme for scalar," << endl;
	  cout << "which can be unstable." << endl;
	}
          else if (limiter_type == "central-upwind") d_limiter_type = 3;
            else if (limiter_type == "upwind") d_limiter_type = 4;
	      else throw InvalidValue("Flux limiter type "
		                           "not supported: " + limiter_type, __FILE__, __LINE__);
  string boundary_limiter_type;
  d_boundary_limiter_type = 3;
  if (d_limiter_type < 3) {
    db->getWithDefault("boundary_limiter_type",boundary_limiter_type,"central-upwind");
    if (boundary_limiter_type == "none") {
	  d_boundary_limiter_type = 2;
	  cout << "WARNING! Running central scheme for scalar on the boundaries," << endl;
	  cout << "which can be unstable." << endl;
    }
      else if (boundary_limiter_type == "central-upwind") d_boundary_limiter_type = 3;
        else if (boundary_limiter_type == "upwind") d_boundary_limiter_type = 4;
	  else throw InvalidValue("Flux limiter type on the boundary"
		                  "not supported: " + boundary_limiter_type, __FILE__, __LINE__);
    d_central_limiter = false;
    if (d_limiter_type < 2)
      db->getWithDefault("central_limiter",d_central_limiter,false);
  }
  }

  // make source and boundary_condition objects
  d_source = scinew Source(d_physicalConsts);
  
  if (d_doMMS)
	  d_source->problemSetup(db);

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
    if ((d_turbPrNo == 0.0)&&(model_turbPrNo == 0.0))
	  throw InvalidValue("Turbulent Prandtl number is not specified for"
		             "mixture fraction ", __FILE__, __LINE__);
    // if it is set in turbulence model
    else if (d_turbPrNo == 0.0)
      d_turbPrNo = model_turbPrNo;

    // if it is set here or set in both places, 
    // we only need to set mixture fraction Pr number in turbulence model
    if (!(model_turbPrNo == d_turbPrNo)) {
      cout << "Turbulent Prandtl number for mixture fraction is set to "
      << d_turbPrNo << endl;
      d_turbModel->setTurbulentPrandtlNumber(d_turbPrNo);
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
		    const TimeIntegratorLabel* timelabels,
                    bool d_EKTCorrection,
                    bool doing_EKT_now)
{
  //computes stencil coefficients and source terms
  // requires : scalarIN, [u,v,w]VelocitySPBC, densityIN, viscosityIN
  // computes : scalCoefSBLM, scalLinSrcSBLM, scalNonLinSrcSBLM
  sched_buildLinearMatrix(sched, patches, matls, timelabels, d_EKTCorrection,
                          doing_EKT_now);
  
  // Schedule the scalar solve
  // require : scalarIN, scalCoefSBLM, scalNonLinSrcSBLM
  sched_scalarLinearSolve(sched, patches, matls, timelabels, d_EKTCorrection,
                          doing_EKT_now);
}

//****************************************************************************
// Schedule build of linear matrix
//****************************************************************************
void 
ScalarSolver::sched_buildLinearMatrix(SchedulerP& sched,
				      const PatchSet* patches,
				      const MaterialSet* matls,
				      const TimeIntegratorLabel* timelabels,
                                      bool d_EKTCorrection,
                                      bool doing_EKT_now)
{
  string taskname =  "ScalarSolver::BuildCoeff" +
		     timelabels->integrator_step_name;
  if (doing_EKT_now) taskname += "EKTnow";
  Task* tsk = scinew Task(taskname, this,
			  &ScalarSolver::buildLinearMatrix,
			  timelabels, d_EKTCorrection, doing_EKT_now);


  Task::WhichDW parent_old_dw;
  if (timelabels->recursion) parent_old_dw = Task::ParentOldDW;
  else parent_old_dw = Task::OldDW;

  tsk->requires(parent_old_dw, d_lab->d_sharedState->get_delt_label());
  
  // This task requires scalar and density from old time step for transient
  // calculation
  //DataWarehouseP old_dw = new_dw->getTop();  
  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);

  tsk->requires(Task::NewDW, d_lab->d_scalarSPLabel,
		Ghost::AroundCells, Arches::TWOGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_densityCPLabel, 
		Ghost::AroundCells, Arches::TWOGHOSTCELLS);

  Task::WhichDW old_values_dw;
  if (timelabels->use_old_values) old_values_dw = parent_old_dw;
  else old_values_dw = Task::NewDW;

   tsk->requires(old_values_dw, d_lab->d_scalarSPLabel,
		 Ghost::None, Arches::ZEROGHOSTCELLS);
   tsk->requires(old_values_dw, d_lab->d_densityCPLabel, 
		 Ghost::None, Arches::ZEROGHOSTCELLS);

  if (d_dynScalarModel)
    tsk->requires(Task::NewDW, d_lab->d_scalarDiffusivityLabel,
		  Ghost::AroundCells, Arches::TWOGHOSTCELLS);
  else
    tsk->requires(Task::NewDW, d_lab->d_viscosityCTSLabel,
		  Ghost::AroundCells, Arches::TWOGHOSTCELLS);

  tsk->requires(Task::NewDW, d_lab->d_uVelocitySPBCLabel,
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_vVelocitySPBCLabel,
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_wVelocitySPBCLabel,
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);

  if (dynamic_cast<const ScaleSimilarityModel*>(d_turbModel)) 
  if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First)
    tsk->requires(Task::OldDW, d_lab->d_scalarFluxCompLabel,
		  d_lab->d_vectorMatl, Task::OutOfDomain,
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);
  else
    tsk->requires(Task::NewDW, d_lab->d_scalarFluxCompLabel,
		  d_lab->d_vectorMatl, Task::OutOfDomain,
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);


      // added one more argument of index to specify scalar component
  if ((timelabels->integrator_step_number == TimeIntegratorStepNumber::First)
      &&((!(d_EKTCorrection))||((d_EKTCorrection)&&(doing_EKT_now)))) {
    tsk->computes(d_lab->d_scalCoefSBLMLabel, d_lab->d_stencilMatl,
		  Task::OutOfDomain);
    tsk->computes(d_lab->d_scalDiffCoefLabel, d_lab->d_stencilMatl,
		  Task::OutOfDomain);
    tsk->computes(d_lab->d_scalNonLinSrcSBLMLabel);
//#ifdef divergenceconstraint
    tsk->computes(d_lab->d_scalDiffCoefSrcLabel);
//#endif
  }
  else {
    tsk->modifies(d_lab->d_scalCoefSBLMLabel, d_lab->d_stencilMatl,
		  Task::OutOfDomain);
    tsk->modifies(d_lab->d_scalDiffCoefLabel, d_lab->d_stencilMatl,
		  Task::OutOfDomain);
    tsk->modifies(d_lab->d_scalNonLinSrcSBLMLabel);
//#ifdef divergenceconstraint
    tsk->modifies(d_lab->d_scalDiffCoefSrcLabel);
//#endif
  }
  if (doing_EKT_now)
    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First)
      tsk->computes(d_lab->d_scalarEKTLabel);
    else
      tsk->modifies(d_lab->d_scalarEKTLabel);

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
				     const TimeIntegratorLabel* timelabels,
                                     bool d_EKTCorrection,
                                     bool doing_EKT_now)
{

  DataWarehouse* parent_old_dw;
  if (timelabels->recursion) parent_old_dw = new_dw->getOtherDataWarehouse(Task::ParentOldDW);
  else parent_old_dw = old_dw;

  delt_vartype delT;
  parent_old_dw->get(delT, d_lab->d_sharedState->get_delt_label() );
  double delta_t = delT;
  delta_t *= timelabels->time_multiplier;

  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->
                    getArchesMaterial(archIndex)->getDWIndex(); 
    ArchesVariables scalarVars;
    ArchesConstVariables constScalarVars;
    
    // Get the PerPatch CellInformation data
    PerPatch<CellInformationP> cellInfoP;
    if (new_dw->exists(d_lab->d_cellInfoLabel, matlIndex, patch)) 
      new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    else 
      throw VariableNotFoundInGrid("cellInformation"," ", __FILE__, __LINE__);
    CellInformation* cellinfo = cellInfoP.get().get_rep();

    // from old_dw get PCELL, DENO, FO
    new_dw->get(constScalarVars.cellType, d_lab->d_cellTypeLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);

    DataWarehouse* old_values_dw;
    if (timelabels->use_old_values) old_values_dw = parent_old_dw;
    else old_values_dw = new_dw;
    
    old_values_dw->get(constScalarVars.old_scalar, d_lab->d_scalarSPLabel, 
		       matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    old_values_dw->get(constScalarVars.old_density, d_lab->d_densityCPLabel, 
		       matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
  
    // from new_dw get DEN, VIS, F, U, V, W
    new_dw->get(constScalarVars.density, d_lab->d_densityCPLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::TWOGHOSTCELLS);

    if (d_dynScalarModel)
      new_dw->get(constScalarVars.viscosity, d_lab->d_scalarDiffusivityLabel, 
		  matlIndex, patch, Ghost::AroundCells, Arches::TWOGHOSTCELLS);
    else
      new_dw->get(constScalarVars.viscosity, d_lab->d_viscosityCTSLabel, 
		  matlIndex, patch, Ghost::AroundCells, Arches::TWOGHOSTCELLS);

    new_dw->get(constScalarVars.scalar, d_lab->d_scalarSPLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::TWOGHOSTCELLS);
    // for explicit get old values
    new_dw->get(constScalarVars.uVelocity, d_lab->d_uVelocitySPBCLabel, 
		matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->get(constScalarVars.vVelocity, d_lab->d_vVelocitySPBCLabel, 
		matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->get(constScalarVars.wVelocity, d_lab->d_wVelocitySPBCLabel, 
		matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);

  // allocate matrix coeffs
  if ((timelabels->integrator_step_number == TimeIntegratorStepNumber::First)
      &&((!(d_EKTCorrection))||((d_EKTCorrection)&&(doing_EKT_now)))) {
    for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++) {
      new_dw->allocateAndPut(scalarVars.scalarCoeff[ii],
			     d_lab->d_scalCoefSBLMLabel, ii, patch);
      scalarVars.scalarCoeff[ii].initialize(0.0);
      new_dw->allocateAndPut(scalarVars.scalarDiffusionCoeff[ii],
			     d_lab->d_scalDiffCoefLabel, ii, patch);
      scalarVars.scalarDiffusionCoeff[ii].initialize(0.0);
    }
    new_dw->allocateAndPut(scalarVars.scalarNonlinearSrc,
			   d_lab->d_scalNonLinSrcSBLMLabel, matlIndex, patch);
    scalarVars.scalarNonlinearSrc.initialize(0.0);
//#ifdef divergenceconstraint
    new_dw->allocateAndPut(scalarVars.scalarDiffNonlinearSrc,
			   d_lab->d_scalDiffCoefSrcLabel, matlIndex, patch);
    scalarVars.scalarDiffNonlinearSrc.initialize(0.0);
//#endif
  }
  else {
    for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++) {
      new_dw->getModifiable(scalarVars.scalarCoeff[ii],
			    d_lab->d_scalCoefSBLMLabel, ii, patch);
      scalarVars.scalarCoeff[ii].initialize(0.0);
      new_dw->getModifiable(scalarVars.scalarDiffusionCoeff[ii],
			    d_lab->d_scalDiffCoefLabel, ii, patch);
      scalarVars.scalarDiffusionCoeff[ii].initialize(0.0);
    }
    new_dw->getModifiable(scalarVars.scalarNonlinearSrc,
			  d_lab->d_scalNonLinSrcSBLMLabel, matlIndex, patch);
    scalarVars.scalarNonlinearSrc.initialize(0.0);
//#ifdef divergenceconstraint
    new_dw->getModifiable(scalarVars.scalarDiffNonlinearSrc,
			  d_lab->d_scalDiffCoefSrcLabel, matlIndex, patch);
    scalarVars.scalarDiffNonlinearSrc.initialize(0.0);
//#endif
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
    d_discretize->calculateScalarCoeff(pc, patch,
				       delta_t, cellinfo, 
				       &scalarVars, &constScalarVars,
				       d_conv_scheme);

    // Calculate scalar source terms
    // inputs : [u,v,w]VelocityMS, scalarSP, densityCP, viscosityCTS
    // outputs: scalLinSrcSBLM, scalNonLinSrcSBLM
    d_source->calculateScalarSource(pc, patch,
				    delta_t, cellinfo, 
				    &scalarVars, &constScalarVars);
    if (d_doMMS)
    d_source->calculateScalarMMSSource(pc, patch,
				    delta_t, cellinfo, 
				    &scalarVars, &constScalarVars);
    if (d_conv_scheme > 0) {
      int wall_celltypeval = d_boundaryCondition->wallCellType();
      d_discretize->calculateScalarFluxLimitedConvection
		                                  (pc, patch,  cellinfo,
				  	          &scalarVars, &constScalarVars,
					          wall_celltypeval, 
						  d_limiter_type,
						  d_boundary_limiter_type,
						  d_central_limiter); 
    } 

    // for scalesimilarity model add scalarflux to the source of scalar eqn.
    if (dynamic_cast<const ScaleSimilarityModel*>(d_turbModel)) {
      StencilMatrix<constCCVariable<double> > scalarFlux; //3 point stencil
      if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First)
      for (int ii = 0; ii < d_lab->d_vectorMatl->size(); ii++) {
	old_dw->get(scalarFlux[ii], 
			d_lab->d_scalarFluxCompLabel, ii, patch,
			Ghost::AroundCells, Arches::ONEGHOSTCELL);
      }
      else
      for (int ii = 0; ii < d_lab->d_vectorMatl->size(); ii++) {
	new_dw->get(scalarFlux[ii], 
			d_lab->d_scalarFluxCompLabel, ii, patch,
			Ghost::AroundCells, Arches::ONEGHOSTCELL);
      }
      IntVector indexLow = patch->getCellFORTLowIndex();
      IntVector indexHigh = patch->getCellFORTHighIndex();
      
      // set density for the whole domain
      
      
      // Store current cell
      double sue, suw, sun, sus, sut, sub;
      for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
	for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
	  for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {
	    IntVector currCell(colX, colY, colZ);
	    IntVector prevXCell(colX-1, colY, colZ);
	    IntVector prevYCell(colX, colY-1, colZ);
	    IntVector prevZCell(colX, colY, colZ-1);
	    IntVector nextXCell(colX+1, colY, colZ);
	    IntVector nextYCell(colX, colY+1, colZ);
	    IntVector nextZCell(colX, colY, colZ+1);
	    
	    sue = 0.5*cellinfo->sns[colY]*cellinfo->stb[colZ]*
	      ((scalarFlux[0])[currCell]+(scalarFlux[0])[nextXCell]);
	    suw = 0.5*cellinfo->sns[colY]*cellinfo->stb[colZ]*
	      ((scalarFlux[0])[prevXCell]+(scalarFlux[0])[currCell]);
	    sun = 0.5*cellinfo->sew[colX]*cellinfo->stb[colZ]*
	      ((scalarFlux[1])[currCell]+ (scalarFlux[1])[nextYCell]);
	    sus = 0.5*cellinfo->sew[colX]*cellinfo->stb[colZ]*
	      ((scalarFlux[1])[currCell]+(scalarFlux[1])[prevYCell]);
	    sut = 0.5*cellinfo->sns[colY]*cellinfo->sew[colX]*
	      ((scalarFlux[2])[currCell]+ (scalarFlux[2])[nextZCell]);
	    sub = 0.5*cellinfo->sns[colY]*cellinfo->sew[colX]*
	      ((scalarFlux[2])[currCell]+ (scalarFlux[2])[prevZCell]);
#if 1
	    scalarVars.scalarNonlinearSrc[currCell] += suw-sue+sus-sun+sub-sut;
#ifdef divergenceconstraint
	    scalarVars.scalarDiffNonlinearSrc[currCell] = suw-sue+sus-sun+sub-sut;
#endif
#endif
	  }
	}
      }
    }
    // Calculate the scalar boundary conditions
    // inputs : scalarSP, scalCoefSBLM
    // outputs: scalCoefSBLM
    
    
    if (d_boundaryCondition->anyArchesPhysicalBC()) {
      d_boundaryCondition->scalarBC(pc, patch,
				    &scalarVars, &constScalarVars);
      /*if (d_boundaryCondition->getIntrusionBC())
        d_boundaryCondition->intrusionScalarBC(pc, patch, cellinfo,
					       &scalarVars, &constScalarVars);*/
    }
    // apply multimaterial intrusion wallbc
    if (d_MAlab)
      d_boundaryCondition->mmscalarWallBC(pc, patch, cellinfo,
					  &scalarVars, &constScalarVars);
    
    // similar to mascal
    // inputs :
    // outputs:
    d_source->modifyScalarMassSource(pc, patch, delta_t,
				     &scalarVars, &constScalarVars,
				     d_conv_scheme);
    
    // Calculate the scalar diagonal terms
    // inputs : scalCoefSBLM, scalLinSrcSBLM
    // outputs: scalCoefSBLM
    d_discretize->calculateScalarDiagonal(pc, patch, &scalarVars);

    CCVariable<double> scalar;
    if (doing_EKT_now) {
      if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First)
        new_dw->allocateAndPut(scalar, d_lab->d_scalarEKTLabel, 
                  matlIndex, patch);
      else
        new_dw->getModifiable(scalar, d_lab->d_scalarEKTLabel, 
                  matlIndex, patch);

        new_dw->copyOut(scalar, d_lab->d_scalarSPLabel,
		  matlIndex, patch);
    }

  }
}


//****************************************************************************
// Schedule linear solve of scalar
//****************************************************************************
void
ScalarSolver::sched_scalarLinearSolve(SchedulerP& sched,
				      const PatchSet* patches,
				      const MaterialSet* matls,
				      const TimeIntegratorLabel* timelabels,
                                      bool d_EKTCorrection,
                                      bool doing_EKT_now)
{
  string taskname =  "ScalarSolver::ScalarLinearSolve" + 
		     timelabels->integrator_step_name;
  if (doing_EKT_now) taskname += "EKTnow";
  Task* tsk = scinew Task(taskname, this,
			  &ScalarSolver::scalarLinearSolve,
			  timelabels, d_EKTCorrection, doing_EKT_now);
  
  Task::WhichDW parent_old_dw;
  if (timelabels->recursion) parent_old_dw = Task::ParentOldDW;
  else parent_old_dw = Task::OldDW;

  tsk->requires(parent_old_dw, d_lab->d_sharedState->get_delt_label());

  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);

  tsk->requires(Task::NewDW, d_lab->d_densityGuessLabel, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
  
  if (timelabels->multiple_steps)
    tsk->requires(Task::NewDW, d_lab->d_scalarTempLabel, 
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);
  else
    tsk->requires(Task::OldDW, d_lab->d_scalarSPLabel, 
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);

  tsk->requires(Task::NewDW, d_lab->d_scalCoefSBLMLabel, 
		d_lab->d_stencilMatl, Task::OutOfDomain,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_scalNonLinSrcSBLMLabel, 
		Ghost::None, Arches::ZEROGHOSTCELLS);


  if (d_MAlab) {
    tsk->requires(Task::NewDW, d_lab->d_mmgasVolFracLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
  }    

  if (doing_EKT_now)
    tsk->modifies(d_lab->d_scalarEKTLabel);
  else 
    tsk->modifies(d_lab->d_scalarSPLabel);
  if (timelabels->recursion)
    tsk->computes(d_lab->d_ScalarClippedLabel);
  
  
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
				const TimeIntegratorLabel* timelabels,
                                bool d_EKTCorrection,
                                bool doing_EKT_now)
{
  DataWarehouse* parent_old_dw;
  if (timelabels->recursion) parent_old_dw = new_dw->getOtherDataWarehouse(Task::ParentOldDW);
  else parent_old_dw = old_dw;

  delt_vartype delT;
  parent_old_dw->get(delT, d_lab->d_sharedState->get_delt_label() );
  double delta_t = delT;
  delta_t *= timelabels->time_multiplier;

  
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->
                    getArchesMaterial(archIndex)->getDWIndex(); 
    ArchesVariables scalarVars;
    ArchesConstVariables constScalarVars;

    // Get the PerPatch CellInformation data
    PerPatch<CellInformationP> cellInfoP;
    if (new_dw->exists(d_lab->d_cellInfoLabel, matlIndex, patch)) 
      new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    else 
      throw VariableNotFoundInGrid("cellInformation"," ", __FILE__, __LINE__);
    CellInformation* cellinfo = cellInfoP.get().get_rep();

    new_dw->get(constScalarVars.density_guess, d_lab->d_densityGuessLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);

    if (timelabels->multiple_steps)
      new_dw->get(constScalarVars.old_scalar, d_lab->d_scalarTempLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    else
      old_dw->get(constScalarVars.old_scalar, d_lab->d_scalarSPLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);

    // for explicit calculation
    if (doing_EKT_now)
      new_dw->getModifiable(scalarVars.scalar, d_lab->d_scalarEKTLabel, 
                  matlIndex, patch);
    else
      new_dw->getModifiable(scalarVars.scalar, d_lab->d_scalarSPLabel, 
                  matlIndex, patch);

    for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++)
      new_dw->get(constScalarVars.scalarCoeff[ii], d_lab->d_scalCoefSBLMLabel, 
		  ii, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(constScalarVars.scalarNonlinearSrc,
		d_lab->d_scalNonLinSrcSBLMLabel, matlIndex, patch,
		Ghost::None, Arches::ZEROGHOSTCELLS);

    new_dw->get(constScalarVars.cellType, d_lab->d_cellTypeLabel,
		    matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    if (d_MAlab) {
      new_dw->get(constScalarVars.voidFraction, d_lab->d_mmgasVolFracLabel,
		      matlIndex, patch, 
		      Ghost::None, Arches::ZEROGHOSTCELLS);
    }

    // make it a separate task later

    if (d_MAlab)
      d_boundaryCondition->scalarLisolve_mm(pc, patch, delta_t, 
					    &scalarVars, &constScalarVars,
					    cellinfo);
    else
      d_rhsSolver->scalarLisolve(pc, patch, delta_t, 
				    &scalarVars, &constScalarVars,
				    cellinfo);

  double scalar_clipped = 0.0;
  double epsilon = 1.0e-15;
  // Get the patch bounds and the variable bounds
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();
  for (int ii = idxLo.x(); ii <= idxHi.x(); ii++) {
    for (int jj = idxLo.y(); jj <= idxHi.y(); jj++) {
      for (int kk = idxLo.z(); kk <= idxHi.z(); kk++) {
	IntVector currCell(ii,jj,kk);
	if (scalarVars.scalar[currCell] > 1.0) {
          if (scalarVars.scalar[currCell] > 1.0 + epsilon) {
	    scalar_clipped = 1.0;
	    cout << "scalar got clipped to 1 at " << currCell
            << " , scalar value was " << scalarVars.scalar[currCell] 
	    << " , density guess was " 
	    << constScalarVars.density_guess[currCell] << endl;
          }
	  scalarVars.scalar[currCell] = 1.0;
	}  
	else if (scalarVars.scalar[currCell] < 0.0) {
          if (scalarVars.scalar[currCell] < - epsilon) {
	    scalar_clipped = 1.0;
	    cout << "scalar got clipped to 0 at " << currCell
            << " , scalar value was " << scalarVars.scalar[currCell]
	    << " , density guess was " 
	    << constScalarVars.density_guess[currCell] << endl;
	    cout << "Try setting <scalarUnderflowCheck>true</scalarUnderflowCheck> "
	    << "in the <ARCHES> section of the input file, "
	    <<"but it would only help for first time substep if RKSSP is used" << endl;
          }
	  scalarVars.scalar[currCell] = 0.0;
	}
      }
    }
  }
  if (timelabels->recursion)
    new_dw->put(max_vartype(scalar_clipped), d_lab->d_ScalarClippedLabel);

// Outlet bc is done here not to change old scalar
    if ((d_boundaryCondition->getOutletBC())||
        (d_boundaryCondition->getPressureBC()))
    d_boundaryCondition->scalarOutletPressureBC(pc, patch,
				        &scalarVars, &constScalarVars);

  }
}

