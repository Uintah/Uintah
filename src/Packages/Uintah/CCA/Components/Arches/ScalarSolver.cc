//----- ScalarSolver.cc ----------------------------------------------

#include <Packages/Uintah/CCA/Components/Arches/ScalarSolver.h>
#include <Packages/Uintah/CCA/Components/Arches/Arches.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesLabel.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesMaterial.h>
#include <Packages/Uintah/CCA/Components/Arches/BoundaryCondition.h>
#include <Packages/Uintah/CCA/Components/Arches/CellInformationP.h>
#include <Packages/Uintah/CCA/Components/Arches/Discretization.h>
#include <Packages/Uintah/CCA/Components/Arches/PetscSolver.h>
#include <Packages/Uintah/CCA/Components/Arches/PhysicalConstants.h>
#include <Packages/Uintah/CCA/Components/Arches/RBGSSolver.h>
#include <Packages/Uintah/CCA/Components/Arches/Source.h>
#include <Packages/Uintah/CCA/Components/Arches/TurbulenceModel.h>
#include <Packages/Uintah/CCA/Components/Arches/ScaleSimilarityModel.h>
#include <Packages/Uintah/CCA/Components/Arches/TimeIntegratorLabel.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/Core/Exceptions/InvalidValue.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/PerPatch.h>
#include <Packages/Uintah/Core/Grid/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/SFCZVariable.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>


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
  d_linearSolver = 0;
}

//****************************************************************************
// Destructor
//****************************************************************************
ScalarSolver::~ScalarSolver()
{
  delete d_discretize;
  delete d_source;
  delete d_linearSolver;
}

//****************************************************************************
// Problem Setup
//****************************************************************************
void 
ScalarSolver::problemSetup(const ProblemSpecP& params)
{
  ProblemSpecP db = params->findBlock("MixtureFractionSolver");
  string finite_diff;
  db->require("finite_difference", finite_diff);
  if (finite_diff == "second") 
    d_discretize = scinew Discretization();
  else {
    throw InvalidValue("Finite Differencing scheme "
		       "not supported: " + finite_diff);
    //throw InvalidValue("Finite Differencing scheme "
	//	       "not supported: " + finite_diff, db);
  }
  string conv_scheme;
  db->getWithDefault("convection_scheme",conv_scheme,"l2up");
//  if (db->findBlock("convection_scheme")) {
//    db->require("convection_scheme",conv_scheme);
    if (conv_scheme == "l2up") d_conv_scheme = 0;
    else if (conv_scheme == "eno") d_conv_scheme = 1;
         else if (conv_scheme == "weno") d_conv_scheme = 2;
	      else throw InvalidValue("Convection scheme "
		       "not supported: " + conv_scheme);
//  } else
//    d_conv_scheme = 0;
  // make source and boundary_condition objects
  d_source = scinew Source(d_turbModel, d_physicalConsts);
  string linear_sol;
  db->require("linear_solver", linear_sol);
  if (linear_sol == "linegs")
    d_linearSolver = scinew RBGSSolver();
  else if (linear_sol == "petsc")
     d_linearSolver = scinew PetscSolver(0); // CHEAT - steve d_myworld);
  else {
    throw InvalidValue("linear solver option"
		       " not supported" + linear_sol);
    //throw InvalidValue("linear solver option"
	//	       " not supported" + linear_sol, db);
  }
  d_linearSolver->problemSetup(db);
}

//****************************************************************************
// Schedule solve of linearized scalar equation
//****************************************************************************
void 
ScalarSolver::solve(SchedulerP& sched,
		    const PatchSet* patches,
		    const MaterialSet* matls,
		    const TimeIntegratorLabel* timelabels,
		    int index)
{
  //computes stencil coefficients and source terms
  // requires : scalarIN, [u,v,w]VelocitySPBC, densityIN, viscosityIN
  // computes : scalCoefSBLM, scalLinSrcSBLM, scalNonLinSrcSBLM
  sched_buildLinearMatrix(sched, patches, matls, timelabels, index);
  
  // Schedule the scalar solve
  // require : scalarIN, scalCoefSBLM, scalNonLinSrcSBLM
  // compute : scalResidualSS, scalCoefSS, scalNonLinSrcSS, scalarSP
  //d_linearSolver->sched_scalarSolve(level, sched, new_dw, matrix_dw, index);
  sched_scalarLinearSolve(sched, patches, matls, timelabels, index);
}

//****************************************************************************
// Schedule build of linear matrix
//****************************************************************************
void 
ScalarSolver::sched_buildLinearMatrix(SchedulerP& sched,
				      const PatchSet* patches,
				      const MaterialSet* matls,
				      const TimeIntegratorLabel* timelabels,
				      int index)
{
  string taskname =  "ScalarSolver::BuildCoeff" +
		     timelabels->integrator_step_name;
  Task* tsk = scinew Task(taskname, this,
			  &ScalarSolver::buildLinearMatrix,
			  timelabels, index);


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

  tsk->requires(Task::NewDW, d_lab->d_viscosityCTSLabel,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_uVelocitySPBCLabel,
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_vVelocitySPBCLabel,
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_wVelocitySPBCLabel,
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);

  if (dynamic_cast<const ScaleSimilarityModel*>(d_turbModel)) 
  if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First)
    tsk->requires(Task::OldDW, d_lab->d_scalarFluxCompLabel,
		  d_lab->d_scalarFluxMatl, Task::OutOfDomain,
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);
  else
    tsk->requires(Task::NewDW, d_lab->d_scalarFluxCompLabel,
		  d_lab->d_scalarFluxMatl, Task::OutOfDomain,
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);


  if (d_conv_scheme > 0) {
    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First) {
      tsk->requires(Task::OldDW, timelabels->maxabsu_in);
      tsk->requires(Task::OldDW, timelabels->maxabsv_in);
      tsk->requires(Task::OldDW, timelabels->maxabsw_in);
    }
    else {
      tsk->requires(Task::NewDW, timelabels->maxabsu_in);
      tsk->requires(Task::NewDW, timelabels->maxabsv_in);
      tsk->requires(Task::NewDW, timelabels->maxabsw_in);
    }
  }

      // added one more argument of index to specify scalar component
  if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First) {
    tsk->computes(d_lab->d_scalCoefSBLMLabel, d_lab->d_stencilMatl,
		  Task::OutOfDomain);
    tsk->computes(d_lab->d_scalDiffCoefLabel, d_lab->d_stencilMatl,
		  Task::OutOfDomain);
    tsk->computes(d_lab->d_scalNonLinSrcSBLMLabel);
#ifdef divergenceconstraint
    tsk->computes(d_lab->d_scalDiffCoefSrcLabel);
#endif
  }
  else {
    tsk->modifies(d_lab->d_scalCoefSBLMLabel, d_lab->d_stencilMatl,
		  Task::OutOfDomain);
    tsk->modifies(d_lab->d_scalDiffCoefLabel, d_lab->d_stencilMatl,
		  Task::OutOfDomain);
    tsk->modifies(d_lab->d_scalNonLinSrcSBLMLabel);
#ifdef divergenceconstraint
    tsk->modifies(d_lab->d_scalDiffCoefSrcLabel);
#endif
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
				     const TimeIntegratorLabel* timelabels,
				     int index)
{

  DataWarehouse* parent_old_dw;
  if (timelabels->recursion) parent_old_dw = new_dw->getOtherDataWarehouse(Task::ParentOldDW);
  else parent_old_dw = old_dw;

  delt_vartype delT;
  parent_old_dw->get(delT, d_lab->d_sharedState->get_delt_label() );
  double delta_t = delT;
  delta_t *= timelabels->time_multiplier;

  double maxAbsU = 0.0;
  double maxAbsV = 0.0;
  double maxAbsW = 0.0;
  if (d_conv_scheme > 0) {
    max_vartype mxAbsU;
    max_vartype mxAbsV;
    max_vartype mxAbsW;
    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First) {
      old_dw->get(mxAbsU, timelabels->maxabsu_in);
      old_dw->get(mxAbsV, timelabels->maxabsv_in);
      old_dw->get(mxAbsW, timelabels->maxabsw_in);
    }
    else {
      new_dw->get(mxAbsU, timelabels->maxabsu_in);
      new_dw->get(mxAbsV, timelabels->maxabsv_in);
      new_dw->get(mxAbsW, timelabels->maxabsw_in);
    }
    maxAbsU = mxAbsU;
    maxAbsV = mxAbsV;
    maxAbsW = mxAbsW;
  }

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
    else {
      cellInfoP.setData(scinew CellInformation(patch));
      new_dw->put(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    }
    CellInformation* cellinfo = cellInfoP.get().get_rep();

    // from old_dw get PCELL, DENO, FO(index)
    new_dw->get(constScalarVars.cellType, d_lab->d_cellTypeLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);

    DataWarehouse* old_values_dw;
    if (timelabels->use_old_values) old_values_dw = parent_old_dw;
    else old_values_dw = new_dw;
    
    old_values_dw->get(constScalarVars.old_scalar, d_lab->d_scalarSPLabel, 
		       matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    old_values_dw->get(constScalarVars.old_density, d_lab->d_densityCPLabel, 
		       matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
  
    // from new_dw get DEN, VIS, F(index), U, V, W
    new_dw->get(constScalarVars.density, d_lab->d_densityCPLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::TWOGHOSTCELLS);
    new_dw->get(constScalarVars.viscosity, d_lab->d_viscosityCTSLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
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
  if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First) {
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
#ifdef divergenceconstraint
    new_dw->allocateAndPut(scalarVars.scalarDiffNonlinearSrc,
			   d_lab->d_scalDiffCoefSrcLabel, matlIndex, patch);
    scalarVars.scalarDiffNonlinearSrc.initialize(0.0);
#endif
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
#ifdef divergenceconstraint
    new_dw->getModifiable(scalarVars.scalarDiffNonlinearSrc,
			  d_lab->d_scalDiffCoefSrcLabel, matlIndex, patch);
    scalarVars.scalarDiffNonlinearSrc.initialize(0.0);
#endif
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
				       delta_t, index, cellinfo, 
				       &scalarVars, &constScalarVars,
				       d_conv_scheme);

    // Calculate scalar source terms
    // inputs : [u,v,w]VelocityMS, scalarSP, densityCP, viscosityCTS
    // outputs: scalLinSrcSBLM, scalNonLinSrcSBLM
    d_source->calculateScalarSource(pc, patch,
				    delta_t, index, cellinfo, 
				    &scalarVars, &constScalarVars);
    if (d_conv_scheme > 0) {
      int wallID = d_boundaryCondition->wallCellType();
      if (d_conv_scheme == 2)
        d_discretize->calculateScalarWENOscheme(pc, patch,  index, cellinfo,
					        maxAbsU, maxAbsV, maxAbsW, 
				  	        &scalarVars, &constScalarVars,
						wallID);
      else
        d_discretize->calculateScalarENOscheme(pc, patch,  index, cellinfo,
					       maxAbsU, maxAbsV, maxAbsW, 
				  	       &scalarVars, &constScalarVars,
					       wallID); 
    } 

    // for scalesimilarity model add scalarflux to the source of scalar eqn.
    if (dynamic_cast<const ScaleSimilarityModel*>(d_turbModel)) {
      StencilMatrix<constCCVariable<double> > scalarFlux; //3 point stencil
      if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First)
      for (int ii = 0; ii < d_lab->d_scalarFluxMatl->size(); ii++) {
	old_dw->get(scalarFlux[ii], 
			d_lab->d_scalarFluxCompLabel, ii, patch,
			Ghost::AroundCells, Arches::ONEGHOSTCELL);
      }
      else
      for (int ii = 0; ii < d_lab->d_scalarFluxMatl->size(); ii++) {
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
    
    
    d_boundaryCondition->scalarBC(pc, patch,  index, cellinfo, 
				  &scalarVars, &constScalarVars);
    if (d_boundaryCondition->getIntrusionBC())
      d_boundaryCondition->intrusionScalarBC(pc, patch, cellinfo,
					     &scalarVars, &constScalarVars);
    // apply multimaterial intrusion wallbc
    if (d_MAlab)
      d_boundaryCondition->mmscalarWallBC(pc, patch, cellinfo,
					  &scalarVars, &constScalarVars);
    
    // similar to mascal
    // inputs :
    // outputs:
    d_source->modifyScalarMassSource(pc, patch, delta_t, index,
				     &scalarVars, &constScalarVars,
				     d_conv_scheme);
    
    // Calculate the scalar diagonal terms
    // inputs : scalCoefSBLM, scalLinSrcSBLM
    // outputs: scalCoefSBLM
    d_discretize->calculateScalarDiagonal(pc, patch, index, &scalarVars);

    // apply underelax to eqn
    d_linearSolver->computeScalarUnderrelax(pc, patch, index, 
					    &scalarVars, &constScalarVars);

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
				      int index)
{
  string taskname =  "ScalarSolver::ScalarLinearSolve" + 
		     timelabels->integrator_step_name;
  Task* tsk = scinew Task(taskname, this,
			  &ScalarSolver::scalarLinearSolve,
			  timelabels, index);
  
  Task::WhichDW parent_old_dw;
  if (timelabels->recursion) parent_old_dw = Task::ParentOldDW;
  else parent_old_dw = Task::OldDW;

  tsk->requires(parent_old_dw, d_lab->d_sharedState->get_delt_label());

  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_densityCPLabel, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_densityGuessLabel, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
  
  if (timelabels->multiple_steps)
    tsk->requires(Task::NewDW, d_lab->d_scalarTempLabel, 
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);
  else
    tsk->requires(Task::OldDW, d_lab->d_scalarSPLabel, 
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);

  Task::WhichDW old_values_dw;
  if (timelabels->use_old_values) old_values_dw = parent_old_dw;
  else old_values_dw = Task::NewDW;

   tsk->requires(old_values_dw, d_lab->d_scalarSPLabel,
		 Ghost::None, Arches::ZEROGHOSTCELLS);
   tsk->requires(old_values_dw, d_lab->d_densityCPLabel, 
		 Ghost::None, Arches::ZEROGHOSTCELLS);

  tsk->requires(Task::NewDW, d_lab->d_scalCoefSBLMLabel, 
		d_lab->d_stencilMatl, Task::OutOfDomain,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_scalNonLinSrcSBLMLabel, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_uVelocitySPBCLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_vVelocitySPBCLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_wVelocitySPBCLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);

  if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First) {
    tsk->requires(Task::OldDW, timelabels->maxabsu_in);
    tsk->requires(Task::OldDW, timelabels->maxabsv_in);
    tsk->requires(Task::OldDW, timelabels->maxabsw_in);
  }
  else {
    tsk->requires(Task::NewDW, timelabels->maxabsu_in);
    tsk->requires(Task::NewDW, timelabels->maxabsv_in);
    tsk->requires(Task::NewDW, timelabels->maxabsw_in);
  }

  if (d_MAlab) {
    tsk->requires(Task::NewDW, d_lab->d_mmgasVolFracLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
  }    

  tsk->modifies(d_lab->d_scalarSPLabel);
  
  
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
				int index)
{
  DataWarehouse* parent_old_dw;
  if (timelabels->recursion) parent_old_dw = new_dw->getOtherDataWarehouse(Task::ParentOldDW);
  else parent_old_dw = old_dw;

  delt_vartype delT;
  parent_old_dw->get(delT, d_lab->d_sharedState->get_delt_label() );
  double delta_t = delT;
  delta_t *= timelabels->time_multiplier;

  double maxAbsU;
  double maxAbsV;
  double maxAbsW;
  max_vartype mxAbsU;
  max_vartype mxAbsV;
  max_vartype mxAbsW;
  if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First) {
    old_dw->get(mxAbsU, timelabels->maxabsu_in);
    old_dw->get(mxAbsV, timelabels->maxabsv_in);
    old_dw->get(mxAbsW, timelabels->maxabsw_in);
  }
  else {
    new_dw->get(mxAbsU, timelabels->maxabsu_in);
    new_dw->get(mxAbsV, timelabels->maxabsv_in);
    new_dw->get(mxAbsW, timelabels->maxabsw_in);
  }
  maxAbsU = mxAbsU;
  maxAbsV = mxAbsV;
  maxAbsW = mxAbsW;
  
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
    else {
      cellInfoP.setData(scinew CellInformation(patch));
      new_dw->put(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    }
    CellInformation* cellinfo = cellInfoP.get().get_rep();

    new_dw->get(constScalarVars.old_density, d_lab->d_densityCPLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(constScalarVars.density_guess, d_lab->d_densityGuessLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);

    if (timelabels->multiple_steps)
      new_dw->get(constScalarVars.old_scalar, d_lab->d_scalarTempLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    else
      old_dw->get(constScalarVars.old_scalar, d_lab->d_scalarSPLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);

    DataWarehouse* old_values_dw;
    if (timelabels->use_old_values) old_values_dw = parent_old_dw;
    else old_values_dw = new_dw;
    
    old_values_dw->get(constScalarVars.old_old_scalar, d_lab->d_scalarSPLabel, 
		       matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    old_values_dw->get(constScalarVars.old_old_density, d_lab->d_densityCPLabel, 
		       matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);

    // for explicit calculation
    new_dw->getModifiable(scalarVars.scalar, d_lab->d_scalarSPLabel, 
                matlIndex, patch);

    new_dw->get(constScalarVars.uVelocity, d_lab->d_uVelocitySPBCLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(constScalarVars.vVelocity, d_lab->d_vVelocitySPBCLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(constScalarVars.wVelocity, d_lab->d_wVelocitySPBCLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);

    for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++)
      new_dw->get(constScalarVars.scalarCoeff[ii], d_lab->d_scalCoefSBLMLabel, 
		  ii, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(constScalarVars.scalarNonlinearSrc,
		d_lab->d_scalNonLinSrcSBLMLabel, matlIndex, patch,
		Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->allocateTemporary(scalarVars.residualScalar,  patch);

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
      d_linearSolver->scalarLisolve(pc, patch, index, delta_t, 
				    &scalarVars, &constScalarVars,
				    cellinfo);

// Outlet bc is done here not to change old scalar
    int out_celltypeval = d_boundaryCondition->outletCellType();
    if (!(out_celltypeval==-10))
    d_boundaryCondition->scalarOutletBC(pc, patch,  index, cellinfo, 
				        &scalarVars, &constScalarVars, delta_t,
					maxAbsU, maxAbsV, maxAbsW);
    
    d_boundaryCondition->scalarPressureBC(pc, patch,  index, cellinfo, 
				  	  &scalarVars, &constScalarVars, delta_t);

  }
}

