//----- PressureSolver.cc ----------------------------------------------

#include <sci_defs.h>

#include <Packages/Uintah/CCA/Components/Arches/debug.h>
#include <Packages/Uintah/CCA/Components/Arches/PressureSolver.h>
#include <Packages/Uintah/CCA/Components/Arches/Arches.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesLabel.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesMaterial.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesVariables.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesConstVariables.h>
#include <Packages/Uintah/CCA/Components/Arches/BoundaryCondition.h>
#include <Packages/Uintah/CCA/Components/Arches/CellInformationP.h>
#include <Packages/Uintah/CCA/Components/Arches/Discretization.h>
#include <Packages/Uintah/CCA/Components/Arches/LinearSolver.h>
#include <Packages/Uintah/CCA/Components/Arches/PetscSolver.h>
#ifdef HAVE_HYPRE
#include <Packages/Uintah/CCA/Components/Arches/HypreSolver.h>
#endif
#include <Packages/Uintah/CCA/Components/Arches/PhysicalConstants.h>
#include <Packages/Uintah/CCA/Components/Arches/RBGSSolver.h>
#include <Packages/Uintah/CCA/Components/Arches/Source.h>
#include <Packages/Uintah/CCA/Components/Arches/TurbulenceModel.h>
#include <Packages/Uintah/CCA/Components/Arches/ScaleSimilarityModel.h>
#include <Packages/Uintah/CCA/Components/Arches/TimeIntegratorLabel.h>
#include <Packages/Uintah/CCA/Components/MPMArches/MPMArchesLabel.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Ports/LoadBalancer.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/Core/Exceptions/InvalidValue.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/PerPatch.h>
#include <Packages/Uintah/Core/Grid/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/SFCZVariable.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>

using namespace Uintah;
using namespace std;

#include <Packages/Uintah/CCA/Components/Arches/fortran/add_hydrostatic_term_topressure_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/normpress_fort.h>
//#include <Packages/Uintah/CCA/Components/Arches/fortran/pressrcpred_fort.h>

// ****************************************************************************
// Default constructor for PressureSolver
// ****************************************************************************
PressureSolver::PressureSolver(const ArchesLabel* label,
			       const MPMArchesLabel* MAlb,
			       TurbulenceModel* turb_model,
			       BoundaryCondition* bndry_cond,
			       PhysicalConstants* physConst,
			       const ProcessorGroup* myworld):
                                     d_lab(label), d_MAlab(MAlb),
                                     d_turbModel(turb_model), 
                                     d_boundaryCondition(bndry_cond),
				     d_physicalConsts(physConst),
				     d_myworld(myworld)
{
  d_perproc_patches=0;
  d_discretize = 0;
  d_source = 0;
  d_linearSolver = 0; 
}

// ****************************************************************************
// Destructor
// ****************************************************************************
PressureSolver::~PressureSolver()
{
  if(d_perproc_patches && d_perproc_patches->removeReference())
    delete d_perproc_patches;
  delete d_discretize;
  delete d_source;
  delete d_linearSolver;
}

// ****************************************************************************
// Problem Setup
// ****************************************************************************
void 
PressureSolver::problemSetup(const ProblemSpecP& params)
{
  ProblemSpecP db = params->findBlock("PressureSolver");
  db->require("ref_point", d_pressRef);
  string finite_diff;
  db->require("finite_difference", finite_diff);
  if (finite_diff == "second") d_discretize = scinew Discretization();
  else {
    throw InvalidValue("Finite Differencing scheme "
		       "not supported: " + finite_diff);
  }

  // make source and boundary_condition objects
  d_source = scinew Source(d_turbModel, d_physicalConsts);

  string linear_sol;
  db->require("linear_solver", linear_sol);
  if (linear_sol == "linegs") d_linearSolver = scinew RBGSSolver();
  else if (linear_sol == "petsc") d_linearSolver = scinew PetscSolver(d_myworld);
#ifdef HAVE_HYPRE
  else if (linear_sol == "hypre") d_linearSolver = scinew HypreSolver(d_myworld);
#endif
  else {
    throw InvalidValue("Linear solver option"
		       " not supported" + linear_sol);
  }
  d_linearSolver->problemSetup(db);
}
// ****************************************************************************
// Schedule solve of linearized pressure equation
// ****************************************************************************
void PressureSolver::solve(const LevelP& level,
			   SchedulerP& sched,
		 	   const TimeIntegratorLabel* timelabels)
{
  const PatchSet* patches = level->eachPatch();
  const MaterialSet* matls = d_lab->d_sharedState->allArchesMaterials();

  sched_buildLinearMatrix(sched, patches, matls, timelabels);

  sched_pressureLinearSolve(level, sched, timelabels);

  if (d_MAlab) {
    if (timelabels->integrator_step_name == "FE")
      sched_addHydrostaticTermtoPressure(sched, patches, matls);
  }

}

// ****************************************************************************
// Schedule build of linear matrix
// ****************************************************************************
void 
PressureSolver::sched_buildLinearMatrix(SchedulerP& sched,
					const PatchSet* patches,
					const MaterialSet* matls,
		 	                const TimeIntegratorLabel* timelabels)
{

  //  build pressure equation coefficients and source
  string taskname =  "PressureSolver::buildLinearMatrix" +
		     timelabels->integrator_step_name;
  Task* tsk = scinew Task(taskname, this,
			  &PressureSolver::buildLinearMatrix,
			  timelabels);
    

  Task::WhichDW parent_old_dw;
  if (timelabels->recursion) parent_old_dw = Task::ParentOldDW;
  else parent_old_dw = Task::OldDW;

  tsk->requires(parent_old_dw, d_lab->d_sharedState->get_delt_label());
  
  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel, 
		Ghost::AroundCells, Arches::ONEGHOSTCELL);

  if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First)
  tsk->requires(Task::OldDW, timelabels->pressure_guess,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  else
  tsk->requires(Task::NewDW, timelabels->pressure_guess,
		Ghost::None, Arches::ZEROGHOSTCELLS);

  tsk->requires(Task::NewDW, d_lab->d_densityCPLabel, 
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_uVelRhoHatLabel,
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_vVelRhoHatLabel,
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_wVelRhoHatLabel,
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
  // get drhodt that goes in the rhs of the pressure equation
  tsk->requires(Task::NewDW, d_lab->d_filterdrhodtLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
#ifdef divergenceconstraint
  tsk->requires(Task::NewDW, d_lab->d_divConstraintLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
#endif
  if (d_MAlab) {
    tsk->requires(Task::NewDW, d_lab->d_mmgasVolFracLabel, 
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);
  }

  if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First) {
    tsk->computes(d_lab->d_presCoefPBLMLabel, d_lab->d_stencilMatl,
		  Task::OutOfDomain);
    tsk->computes(d_lab->d_presNonLinSrcPBLMLabel);
  }
  else {
    tsk->modifies(d_lab->d_presCoefPBLMLabel, d_lab->d_stencilMatl,
		  Task::OutOfDomain);
    tsk->modifies(d_lab->d_presNonLinSrcPBLMLabel);
  }
  

  sched->addTask(tsk, patches, matls);
}

// ****************************************************************************
// Actually build of linear matrix for pressure equation
// ****************************************************************************
void 
PressureSolver::buildLinearMatrix(const ProcessorGroup* pc,
				  const PatchSubset* patches,
				  const MaterialSubset* /* matls */,
				  DataWarehouse* old_dw,
				  DataWarehouse* new_dw,
		 	          const TimeIntegratorLabel* timelabels)
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
    ArchesVariables pressureVars;
    ArchesConstVariables constPressureVars;
    int nofStencils = 7;

    new_dw->get(constPressureVars.cellType, d_lab->d_cellTypeLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);

    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First)
    old_dw->get(constPressureVars.pressure, timelabels->pressure_guess, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    else
    new_dw->get(constPressureVars.pressure, timelabels->pressure_guess, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);

    new_dw->get(constPressureVars.density, d_lab->d_densityCPLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->get(constPressureVars.uVelRhoHat, d_lab->d_uVelRhoHatLabel, 
		matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->get(constPressureVars.vVelRhoHat, d_lab->d_vVelRhoHatLabel, 
		matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->get(constPressureVars.wVelRhoHat, d_lab->d_wVelRhoHatLabel, 
		matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->get(constPressureVars.filterdrhodt, d_lab->d_filterdrhodtLabel,
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
#ifdef divergenceconstraint
    new_dw->get(constPressureVars.divergence, d_lab->d_divConstraintLabel,
		    matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
#endif


    PerPatch<CellInformationP> cellInfoP;
    if (new_dw->exists(d_lab->d_cellInfoLabel, matlIndex, patch)) 
      new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    else {
      cellInfoP.setData(scinew CellInformation(patch));
      new_dw->put(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    }
    CellInformation* cellinfo = cellInfoP.get().get_rep();

  
    // Calculate Pressure Coeffs
    for (int ii = 0; ii < nofStencils; ii++) {
      if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First)
        new_dw->allocateAndPut(pressureVars.pressCoeff[ii],
			       d_lab->d_presCoefPBLMLabel, ii, patch);
      else
        new_dw->getModifiable(pressureVars.pressCoeff[ii],
			      d_lab->d_presCoefPBLMLabel, ii, patch);
      pressureVars.pressCoeff[ii].initialize(0.0);
    }

    d_discretize->calculatePressureCoeff(pc, patch, old_dw, new_dw, 
					 delta_t, cellinfo,
					 &pressureVars, &constPressureVars);

    // Modify pressure coefficients for multimaterial formulation

    if (d_MAlab) {

      new_dw->get(constPressureVars.voidFraction,
		  d_lab->d_mmgasVolFracLabel, matlIndex, patch,
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);

      d_discretize->mmModifyPressureCoeffs(pc, patch, &pressureVars,
					   &constPressureVars);

    }

    // Calculate Pressure Source
    //  inputs : pressureSPBC, [u,v,w]VelocitySIVBC, densityCP,
    //           [u,v,w]VelCoefPBLM, [u,v,w]VelNonLinSrcPBLM
    //  outputs: presLinSrcPBLM, presNonLinSrcPBLM
    // Allocate space

    new_dw->allocateTemporary(pressureVars.pressLinearSrc,  patch);
    pressureVars.pressLinearSrc.initialize(0.0);
    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First)
      new_dw->allocateAndPut(pressureVars.pressNonlinearSrc,
			     d_lab->d_presNonLinSrcPBLMLabel, matlIndex, patch);
    else
      new_dw->getModifiable(pressureVars.pressNonlinearSrc,
			    d_lab->d_presNonLinSrcPBLMLabel, matlIndex, patch);
    pressureVars.pressNonlinearSrc.initialize(0.0);

    d_source->calculatePressureSourcePred(pc, patch, delta_t,
    					  cellinfo, &pressureVars,
					  &constPressureVars);

    // Calculate Pressure BC
    //  inputs : pressureIN, presCoefPBLM
    //  outputs: presCoefPBLM


    // do multimaterial bc; this is done before 
    // calculatePressDiagonal because unlike the outlet
    // boundaries in the explicit projection, we want to 
    // show the effect of AE, etc. in AP for the 
    // intrusion boundaries
    if (d_boundaryCondition->anyArchesPhysicalBC())
      if (d_boundaryCondition->getIntrusionBC())
        d_boundaryCondition->intrusionPressureBC(pc, patch, cellinfo,
					         &pressureVars,&constPressureVars);
    
    if (d_MAlab)
      d_boundaryCondition->mmpressureBC(pc, patch, cellinfo,
					&pressureVars, &constPressureVars);

    // Calculate Pressure Diagonal
    d_discretize->calculatePressDiagonal(pc, patch, old_dw, new_dw, 
					 &pressureVars);

    if (d_boundaryCondition->anyArchesPhysicalBC())
      d_boundaryCondition->pressureBC(pc, patch, old_dw, new_dw, 
				      cellinfo, &pressureVars,&constPressureVars);
    // apply underelaxation to eqn
// Pressure underrelaxation is turned off since it breaks continuity!!!
/*    if (!(d_pressure_correction))
    d_linearSolver->computePressUnderrelax(pc, patch,
					   &pressureVars, &constPressureVars);*/

  }
}

// ****************************************************************************
// Schedule solver for linear matrix
// ****************************************************************************
void 
PressureSolver::sched_pressureLinearSolve(const LevelP& level,
					  SchedulerP& sched,
		 	          	  const TimeIntegratorLabel* timelabels)
{
  if(d_perproc_patches && d_perproc_patches->removeReference())
    delete d_perproc_patches;

  LoadBalancer* lb = sched->getLoadBalancer();
  d_perproc_patches = lb->createPerProcessorPatchSet(level, d_myworld);
  d_perproc_patches->addReference();
  const MaterialSet* matls = d_lab->d_sharedState->allArchesMaterials();

  string taskname =  "PressureSolver::PressLinearSolve_all" + 
		     timelabels->integrator_step_name;
  Task* tsk = scinew Task(taskname, this,
			  &PressureSolver::pressureLinearSolve_all,
			  timelabels);

  // Requires
  // coefficient for the variable for which solve is invoked

  if (!(d_pressure_correction))
    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First)
    tsk->requires(Task::OldDW, timelabels->pressure_guess, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);
    else
    tsk->requires(Task::NewDW, timelabels->pressure_guess, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);

  tsk->requires(Task::NewDW, d_lab->d_presCoefPBLMLabel, d_lab->d_stencilMatl,
		Task::OutOfDomain, Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_presNonLinSrcPBLMLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);

#ifdef compute_residual
  // computes global residual
  tsk->computes(d_lab->d_presResidPSLabel);
  tsk->computes(d_lab->d_presTruncPSLabel);
#endif

  tsk->computes(timelabels->pressure_out);
  tsk->computes(d_lab->d_InitNormLabel);

#ifdef ARCHES_PRES_DEBUG
  cerr << "Adding computes on patch: " << patch->getID() << '\n';
#endif

  sched->addTask(tsk, d_perproc_patches, matls);

  const Patch* d_pressRefPatch = level->selectPatchForCellIndex(d_pressRef);
  if(!d_pressRefPatch){

    for(Level::const_patchIterator iter=level->patchesBegin();
	iter != level->patchesEnd(); iter++){

      const Patch* patch=*iter;
      if(patch->containsCell(d_pressRef))
	d_pressRefPatch = patch;
    }

    if(!d_pressRefPatch)
      throw InternalError("Patch containing pressure reference point was not found");
  }

  d_pressRefProc = lb->getPatchwiseProcessorAssignment(d_pressRefPatch,
						       d_myworld);
}


void 
PressureSolver::pressureLinearSolve_all(const ProcessorGroup* pg,
					const PatchSubset* patches,
					const MaterialSubset*,
					DataWarehouse* old_dw,
					DataWarehouse* new_dw,
		 	          	const TimeIntegratorLabel* timelabels)
{
  int archIndex = 0; // only one arches material
  int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
  ArchesVariables pressureVars;
  int me = pg->myrank();
  // initializeMatrix...
  d_linearSolver->matrixCreate(d_perproc_patches, patches);
  for (int p = 0; p < patches->size(); p++) {
    const Patch *patch = patches->get(p);
    // Underrelax...
    // This calls fillRows on linear(petsc) solver
    pressureLinearSolve(pg, patch, matlIndex, old_dw, new_dw, pressureVars,
			timelabels);
  }
  bool converged =  d_linearSolver->pressLinearSolve();
  if (converged) {
    for (int p = 0; p < patches->size(); p++) {
      const Patch *patch = patches->get(p);
      //	  unpack from linear solver.
      d_linearSolver->copyPressSoln(patch, &pressureVars);
    }
  } else {
    if (pg->myrank() == 0){
      cerr << "pressure solver not converged, using old values" << endl;
    }
    throw InternalError("pressure solver is diverging");
  }
  double init_norm = d_linearSolver->getInitNorm();
  new_dw->put(max_vartype(init_norm), d_lab->d_InitNormLabel);
  if(d_pressRefProc == me){
    CCVariable<double> pressure;
    pressure.copyPointer(pressureVars.pressure);
    pressureVars.press_ref = pressure[d_pressRef];
    cerr << "press_ref for norm: " << pressureVars.press_ref << " " <<
      d_pressRefProc << endl;
  }
  MPI_Bcast(&pressureVars.press_ref, 1, MPI_DOUBLE, d_pressRefProc, pg->getComm());
  for (int p = 0; p < patches->size(); p++) {
    const Patch *patch = patches->get(p);
    normPressure(pg, patch, &pressureVars);
    //    updatePressure(pg, patch, &pressureVars);
    // put back the results
  }

  // destroy matrix
  d_linearSolver->destroyMatrix();
}

// Actual linear solve
void 
PressureSolver::pressureLinearSolve(const ProcessorGroup* pc,
				    const Patch* patch,
				    const int matlIndex,
				    DataWarehouse* old_dw,
				    DataWarehouse* new_dw,
				    ArchesVariables& pressureVars,
		 	            const TimeIntegratorLabel* timelabels)
{
  ArchesConstVariables constPressureVars;
  // Get the required data
  new_dw->allocateAndPut(pressureVars.pressure, timelabels->pressure_out,
			 matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
  if (!(d_pressure_correction))
    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First)
    old_dw->copyOut(pressureVars.pressure, timelabels->pressure_guess, 
	            matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    else
    new_dw->copyOut(pressureVars.pressure, timelabels->pressure_guess, 
	            matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
  else
    pressureVars.pressure.initialize(0.0);

  for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++) 
    new_dw->get(constPressureVars.pressCoeff[ii], d_lab->d_presCoefPBLMLabel, 
		   ii, patch, Ghost::None, Arches::ZEROGHOSTCELLS);

  new_dw->get(constPressureVars.pressNonlinearSrc, 
		  d_lab->d_presNonLinSrcPBLMLabel, 
		  matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);

#ifdef compute_residual
  // compute eqn residual, L1 norm
  new_dw->allocate(pressureVars.residualPressure, d_lab->d_pressureRes,
			  matlIndex, patch);
  d_linearSolver->computePressResidual(pc, patch, old_dw, new_dw, 
				       &pressureVars);
  new_dw->put(sum_vartype(pressureVars.residPress), d_lab->d_presResidPSLabel);
  new_dw->put(sum_vartype(pressureVars.truncPress), d_lab->d_presTruncPSLabel);
#else
  pressureVars.residPress=pressureVars.truncPress=0;
#endif

  // for parallel code lisolve will become a recursive task and 
  // will make the following subroutine separate
  // get patch numer ***warning****
  // sets matrix
  d_linearSolver->setPressMatrix(pc, patch,
				 &pressureVars, &constPressureVars, d_lab);
}

// ************************************************************************
// Schedule addition of hydrostatic term to relative pressure calculated
// in pressure solve
// ************************************************************************

void
PressureSolver::sched_addHydrostaticTermtoPressure(SchedulerP& sched, 
						   const PatchSet* patches,
						   const MaterialSet* matls)

{
  Task* tsk = scinew Task("Psolve:addhydrostaticterm",
			  this, &PressureSolver::addHydrostaticTermtoPressure);

  
  tsk->requires(Task::NewDW, d_lab->d_pressurePSLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::OldDW, d_lab->d_densityMicroLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);

  tsk->computes(d_lab->d_pressPlusHydroLabel);
  
  sched->addTask(tsk, patches, matls);

}

// ****************************************************************************
// Actual addition of hydrostatic term to relative pressure
// ****************************************************************************

void 
PressureSolver::addHydrostaticTermtoPressure(const ProcessorGroup*,
					     const PatchSubset* patches,
					     const MaterialSubset*,
					     DataWarehouse* old_dw,
					     DataWarehouse* new_dw)

{
  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);

    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    constCCVariable<double> prel;
    CCVariable<double> pPlusHydro;
    constCCVariable<double> denMicro;
    constCCVariable<int> cellType;

    double gx = d_physicalConsts->getGravity(1);
    double gy = d_physicalConsts->getGravity(2);
    double gz = d_physicalConsts->getGravity(3);


    // Get the PerPatch CellInformation data

    PerPatch<CellInformationP> cellInfoP;

    if (new_dw->exists(d_lab->d_cellInfoLabel, matlIndex, patch)) 

      new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);

    else {

      cellInfoP.setData(scinew CellInformation(patch));
      new_dw->put(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);

    }
    CellInformation* cellinfo = cellInfoP.get().get_rep();

    new_dw->get(prel, d_lab->d_pressurePSLabel,
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    old_dw->get(denMicro, d_lab->d_densityMicroLabel,
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(cellType, d_lab->d_cellTypeLabel,
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);

    new_dw->allocateAndPut(pPlusHydro, d_lab->d_pressPlusHydroLabel,
		     matlIndex, patch);

    IntVector valid_lo = patch->getCellFORTLowIndex();
    IntVector valid_hi = patch->getCellFORTHighIndex();

    int mmwallid = d_boundaryCondition->getMMWallId();

    pPlusHydro.initialize(0.0);

    fort_add_hydrostatic_term_topressure(pPlusHydro, prel, denMicro,
					 gx, gy, gz, cellinfo->xx,
					 cellinfo->yy, cellinfo->zz,
					 valid_lo, valid_hi,
					 cellType, mmwallid);
		
    // allocateAndPut instead:
    /* new_dw->put(pPlusHydro, d_lab->d_pressPlusHydroLabel,
		matlIndex, patch); */;

  }
}


// ****************************************************************************
// normalize the pressure solution
// ****************************************************************************
void 
PressureSolver::normPressure(const ProcessorGroup*,
			     const Patch* patch,
			     ArchesVariables* vars)
{
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();
  double pressref = vars->press_ref;
  pressref = 0.0;
  fort_normpress(idxLo, idxHi, vars->pressure, pressref);

#ifdef ARCHES_PRES_DEBUG
  cerr << " After Pressure Normalization : " << endl;
  for (int ii = idxLo.x(); ii <= idxHi.x(); ii++) {
    cerr << "pressure for ii = " << ii << endl;
    for (int jj = idxLo.y(); jj <= idxHi.y(); jj++) {
      for (int kk = idxLo.z(); kk <= idxHi.z(); kk++) {
	cerr.width(10);
	cerr << vars->pressure[IntVector(ii,jj,kk)] << " " ; 
      }
      cerr << endl;
    }
  }
#endif
}  

void 
PressureSolver::updatePressure(const ProcessorGroup*,
			     const Patch* patch,
			     ArchesVariables* vars)
{
  IntVector idxLo = patch->getCellLowIndex();
  IntVector idxHi = patch->getCellHighIndex();
  for (int ii = idxLo.x(); ii < idxHi.x(); ii++) {
    for (int jj = idxLo.y(); jj < idxHi.y(); jj++) {
      for (int kk = idxLo.z(); kk < idxHi.z(); kk++) {
	IntVector currCell(ii,jj,kk);
	vars->pressureNew[currCell] = vars->pressure[currCell];
      }
    }
  }
}  

