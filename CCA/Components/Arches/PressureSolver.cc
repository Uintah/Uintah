//----- PressureSolver.cc ----------------------------------------------

#include <Packages/Uintah/CCA/Components/Arches/debug.h>
#include <Packages/Uintah/CCA/Components/Arches/PressureSolver.h>
#include <Packages/Uintah/CCA/Components/Arches/Arches.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesLabel.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesLabel.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesMaterial.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesVariables.h>
#include <Packages/Uintah/CCA/Components/Arches/BoundaryCondition.h>
#include <Packages/Uintah/CCA/Components/Arches/CellInformationP.h>
#include <Packages/Uintah/CCA/Components/Arches/Discretization.h>
#include <Packages/Uintah/CCA/Components/Arches/PetscSolver.h>
#include <Packages/Uintah/CCA/Components/Arches/PhysicalConstants.h>
#include <Packages/Uintah/CCA/Components/Arches/RBGSSolver.h>
#include <Packages/Uintah/CCA/Components/Arches/Source.h>
#include <Packages/Uintah/CCA/Components/Arches/TurbulenceModel.h>
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
			   SchedulerP& sched)
{
  const PatchSet* patches = level->eachPatch();
  const MaterialSet* matls = d_lab->d_sharedState->allArchesMaterials();

  //computes stencil coefficients and source terms
  // require : old_dw -> pressureSPBC, densityCP, viscosityCTS, [u,v,w]VelocitySPBC
  //           new_dw -> pressureSPBC, densityCP, viscosityCTS, [u,v,w]VelocitySIVBC
  // compute : uVelConvCoefPBLM, vVelConvCoefPBLM, wVelConvCoefPBLM
  //           uVelCoefPBLM, vVelCoefPBLM, wVelCoefPBLM, uVelLinSrcPBLM
  //           vVelLinSrcPBLM, wVelLinSrcPBLM, uVelNonLinSrcPBLM 
  //           vVelNonLinSrcPBLM, wVelNonLinSrcPBLM, presCoefPBLM 
  //           presLinSrcPBLM, presNonLinSrcPBLM
  //sched_buildLinearMatrix(level, sched, new_dw, matrix_dw, delta_t);
  // build the structure and get all the old variables

  sched_buildLinearMatrix(sched, patches, matls);

  //residual at the start of linear solve
  // this can be part of linear solver
#if 0
  calculateResidual(level, sched, old_dw, new_dw);
  calculateOrderMagnitude(level, sched, old_dw, new_dw);
#endif

  // Schedule the pressure solve
  // require : pressureIN, presCoefPBLM, presNonLinSrcPBLM
  // compute : presResidualPS, presCoefPS, presNonLinSrcPS, pressurePS
  //d_linearSolver->sched_pressureSolve(level, sched, new_dw, matrix_dw);

  sched_pressureLinearSolve(level, sched);

  // Schedule Calculation of pressure norm
  // require :
  // compute :
  //sched_normPressure(level, sched, new_dw, matrix_dw);

  // schedule addition of hydrostatic term to pressure

  if (d_MAlab) {

    sched_addHydrostaticTermtoPressure(sched, patches, matls);

  }

}

// ****************************************************************************
// Schedule build of linear matrix
// ****************************************************************************
void 
PressureSolver::sched_buildLinearMatrix(SchedulerP& sched, const PatchSet* patches,
					const MaterialSet* matls)
{

  // Build momentum equation coefficients and sources that are needed 
  // to later build pressure equation coefficients and source

  {
    Task* tsk = scinew Task( "Psolve::BuildCoeff", 
			     this, &PressureSolver::buildLinearMatrix);

    int numGhostCells = 1;
    int zeroGhostCells = 0;

    tsk->requires(Task::OldDW, d_lab->d_sharedState->get_delt_label());
    
  // Requires
  // from old_dw for time integration
  // get old_dw from getTop function
    tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel,
		  Ghost::AroundCells, numGhostCells);
    tsk->requires(Task::OldDW, d_lab->d_densityINLabel, 
		  Ghost::AroundCells, numGhostCells);
    tsk->requires(Task::OldDW, d_lab->d_uVelocitySPBCLabel,
		  Ghost::None, zeroGhostCells);
    tsk->requires(Task::OldDW, d_lab->d_vVelocitySPBCLabel,
		  Ghost::None, zeroGhostCells);
    tsk->requires(Task::OldDW, d_lab->d_wVelocitySPBCLabel,
		  Ghost::None, zeroGhostCells);

  // from new_dw
  // for new task graph to work

    tsk->requires(Task::NewDW, d_lab->d_pressurePSLabel, 
		  Ghost::AroundCells, numGhostCells);
    tsk->requires(Task::NewDW, d_lab->d_densityINLabel,
		  Ghost::AroundCells, numGhostCells+1);
    tsk->requires(Task::NewDW, d_lab->d_viscosityINLabel,
		  Ghost::AroundCells, numGhostCells);
    tsk->requires(Task::NewDW, d_lab->d_uVelocityOUTBCLabel,
		  Ghost::AroundFaces, numGhostCells+1);
    tsk->requires(Task::NewDW, d_lab->d_vVelocityOUTBCLabel,
		  Ghost::AroundFaces, numGhostCells+1);
    tsk->requires(Task::NewDW, d_lab->d_wVelocityOUTBCLabel,
		  Ghost::AroundFaces, numGhostCells+1);
    tsk->requires(Task::NewDW, d_lab->d_denRefArrayLabel,
    		  Ghost::AroundCells, numGhostCells);

    // for multi-material
    // requires su_drag[x,y,z], sp_drag[x,y,z] for arches

    if (d_MAlab) {

      tsk->requires(Task::NewDW, d_MAlab->d_uVel_mmLinSrcLabel,
		    Ghost::None, zeroGhostCells);
      tsk->requires(Task::NewDW, d_MAlab->d_uVel_mmNonlinSrcLabel,
		    Ghost::None, zeroGhostCells);
      tsk->requires(Task::NewDW, d_MAlab->d_vVel_mmLinSrcLabel, 
		    Ghost::None, zeroGhostCells);
      tsk->requires(Task::NewDW, d_MAlab->d_vVel_mmNonlinSrcLabel,
		    Ghost::None, zeroGhostCells);
      tsk->requires(Task::NewDW, d_MAlab->d_wVel_mmLinSrcLabel,
		    Ghost::None, zeroGhostCells);
      tsk->requires(Task::NewDW, d_MAlab->d_wVel_mmNonlinSrcLabel,
		    Ghost::None, zeroGhostCells);

    }
    
    
    // requires convection coeff because of the nodal
    // differencing
    // computes all the components of velocity

    tsk->computes(d_lab->d_uVelCoefPBLMLabel,
		  d_lab->d_stencilMatl, Task::OutOfDomain);
    tsk->computes(d_lab->d_vVelCoefPBLMLabel,
		  d_lab->d_stencilMatl, Task::OutOfDomain);
    tsk->computes(d_lab->d_wVelCoefPBLMLabel,
		  d_lab->d_stencilMatl, Task::OutOfDomain);
    
    tsk->computes(d_lab->d_uVelLinSrcPBLMLabel);
    tsk->computes(d_lab->d_vVelLinSrcPBLMLabel);
    tsk->computes(d_lab->d_wVelLinSrcPBLMLabel);
    tsk->computes(d_lab->d_uVelNonLinSrcPBLMLabel);
    tsk->computes(d_lab->d_vVelNonLinSrcPBLMLabel);
    tsk->computes(d_lab->d_wVelNonLinSrcPBLMLabel);
    
    sched->addTask(tsk, patches, matls);
  }

  // Now build pressure equation coefficients from momentum equation 
  // coefficients

  {
    Task* tsk = scinew Task( "Psolve::BuildCoeffP",
			     this,
			     &PressureSolver::buildLinearMatrixPress);
    
    int numGhostCells = 1;
    int zeroGhostCells = 0;

    tsk->requires(Task::OldDW, d_lab->d_sharedState->get_delt_label());
    
    // int matlIndex = 0;
    // Requires
    // from old_dw for time integration
    // get old_dw from getTop function

    tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel, 
		  Ghost::AroundCells, numGhostCells);

    // fix it

    tsk->requires(Task::OldDW, d_lab->d_densityINLabel,
		  Ghost::AroundCells, numGhostCells);

    // from new_dw

    tsk->requires(Task::NewDW, d_lab->d_pressurePSLabel,
		  Ghost::AroundCells, numGhostCells);
    tsk->requires(Task::NewDW, d_lab->d_densityINLabel, 
		  Ghost::AroundCells, numGhostCells+1);
    tsk->requires(Task::NewDW, d_lab->d_viscosityINLabel,
		  Ghost::AroundCells, numGhostCells);
    tsk->requires(Task::NewDW, d_lab->d_uVelocityOUTBCLabel,
		  Ghost::AroundFaces, numGhostCells+1);
    tsk->requires(Task::NewDW, d_lab->d_vVelocityOUTBCLabel,
		  Ghost::AroundFaces, numGhostCells+1);
    tsk->requires(Task::NewDW, d_lab->d_wVelocityOUTBCLabel,
		  Ghost::AroundFaces, numGhostCells+1);
    
    /// requires convection coeff because of the nodal
    // differencing
    // computes all the components of velocity

    tsk->computes(d_lab->d_presCoefPBLMLabel, d_lab->d_stencilMatl,
		   Task::OutOfDomain);

    tsk->computes(d_lab->d_presNonLinSrcPBLMLabel);

    if (d_MAlab) {
      tsk->requires(Task::NewDW, d_lab->d_mmgasVolFracLabel, 
		    Ghost::AroundCells, numGhostCells);
    }

    sched->addTask(tsk, patches, matls);
  }

}


// ****************************************************************************
// Schedule solver for linear matrix
// ****************************************************************************
void 
PressureSolver::sched_pressureLinearSolve(const LevelP& level,
					  SchedulerP& sched)
{
  if(d_perproc_patches && d_perproc_patches->removeReference())
    delete d_perproc_patches;

  LoadBalancer* lb = sched->getLoadBalancer();
  d_perproc_patches = lb->createPerProcessorPatchSet(level, d_myworld);
  d_perproc_patches->addReference();
  const MaterialSet* matls = d_lab->d_sharedState->allArchesMaterials();

  Task* tsk = scinew Task("PressureSolver::PressLinearSolve",
			  this,
			  &PressureSolver::pressureLinearSolve_all);
  int numGhostCells = 1;
  int zeroGhostCells = 0;

  // Requires
  // coefficient for the variable for which solve is invoked

  tsk->requires(Task::NewDW, d_lab->d_pressurePSLabel, 
		Ghost::AroundCells, numGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_presCoefPBLMLabel, d_lab->d_stencilMatl,
		Task::OutOfDomain, Ghost::None, zeroGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_presNonLinSrcPBLMLabel,
		Ghost::None, zeroGhostCells);

  // computes global residual
  tsk->computes(d_lab->d_presResidPSLabel);
  tsk->computes(d_lab->d_presTruncPSLabel);

  tsk->computes(d_lab->d_pressureSPBCLabel);

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


// ***********************************************************************
// Actual build of linear matrices for momentum components
// ***********************************************************************

void 
PressureSolver::buildLinearMatrix(const ProcessorGroup* pc,
				  const PatchSubset* patches,
				  const MaterialSubset* /*matls*/,
				  DataWarehouse* old_dw,
				  DataWarehouse* new_dw)
{
  delt_vartype delT;
  old_dw->get(delT, d_lab->d_sharedState->get_delt_label() );
  double delta_t = delT;
  
  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    ArchesVariables pressureVars;

    // compute all three componenets of velocity stencil coefficients

    int numGhostCells = 1;
    int zeroGhostCells = 0;

    // Get the reference density
    // Get the required data

    new_dw->getCopy(pressureVars.density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells+1);
    new_dw->getCopy(pressureVars.viscosity, d_lab->d_viscosityINLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->getCopy(pressureVars.pressure, d_lab->d_pressurePSLabel, 
		matlIndex, patch, Ghost::None, zeroGhostCells);
    new_dw->getCopy(pressureVars.denRefArray, d_lab->d_denRefArrayLabel,
    		matlIndex, patch, Ghost::AroundCells, numGhostCells);

    PerPatch<CellInformationP> cellInfoP;

    //  old_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    // checkpointing
    //  new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    //  old_dw->get(cellInfoP, d_cellInfoLabel, matlIndex, patch);

    if (new_dw->exists(d_lab->d_cellInfoLabel, matlIndex, patch)) 

      new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);

    else {

      cellInfoP.setData(scinew CellInformation(patch));
      new_dw->put(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);

    }

    CellInformation* cellinfo = cellInfoP.get().get_rep();

    new_dw->getCopy(pressureVars.uVelocity, d_lab->d_uVelocityOUTBCLabel, 
		matlIndex, patch, Ghost::AroundFaces, numGhostCells);
    new_dw->getCopy(pressureVars.vVelocity, d_lab->d_vVelocityOUTBCLabel, 
		matlIndex, patch, Ghost::AroundFaces, numGhostCells);
    new_dw->getCopy(pressureVars.wVelocity, d_lab->d_wVelocityOUTBCLabel, 
		matlIndex, patch, Ghost::AroundFaces, numGhostCells);
#if 0
    old_dw->getCopy(pressureVars.old_uVelocity, d_lab->d_uVelocitySPBCLabel, 
		matlIndex, patch, Ghost::None, zeroGhostCells);
    old_dw->getCopy(pressureVars.old_vVelocity, d_lab->d_vVelocitySPBCLabel, 
		matlIndex, patch, Ghost::None, zeroGhostCells);
    old_dw->getCopy(pressureVars.old_wVelocity, d_lab->d_wVelocitySPBCLabel, 
		matlIndex, patch, Ghost::None, zeroGhostCells);
#endif
    new_dw->getCopy(pressureVars.old_uVelocity, d_lab->d_uVelocityOUTBCLabel, 
		matlIndex, patch, Ghost::None, zeroGhostCells);
    new_dw->getCopy(pressureVars.old_vVelocity, d_lab->d_vVelocityOUTBCLabel, 
		matlIndex, patch, Ghost::None, zeroGhostCells);
    new_dw->getCopy(pressureVars.old_wVelocity, d_lab->d_wVelocityOUTBCLabel, 
		matlIndex, patch, Ghost::None, zeroGhostCells);

    // modified - June 20th
    new_dw->getCopy(pressureVars.old_density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->getCopy(pressureVars.cellType, d_lab->d_cellTypeLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
    
    for(int index = 1; index <= Arches::NDIM; ++index) {

    // get multimaterial momentum source terms

      if (d_MAlab) {
	switch (index) {
	
	case Arches::XDIR:

	  new_dw->getCopy(pressureVars.mmuVelSu, d_MAlab->d_uVel_mmNonlinSrcLabel,
		      matlIndex, patch,
		      Ghost::None, zeroGhostCells);
	  new_dw->getCopy(pressureVars.mmuVelSp, d_MAlab->d_uVel_mmLinSrcLabel,
		      matlIndex, patch,
		      Ghost::None, zeroGhostCells);
	  break;

	case Arches::YDIR:

	  new_dw->getCopy(pressureVars.mmvVelSu, d_MAlab->d_vVel_mmNonlinSrcLabel,
		      matlIndex, patch,
		      Ghost::None, zeroGhostCells);
	  new_dw->getCopy(pressureVars.mmvVelSp, d_MAlab->d_vVel_mmLinSrcLabel,
		      matlIndex, patch,
		      Ghost::None, zeroGhostCells);
	  break;
	case Arches::ZDIR:

	  new_dw->getCopy(pressureVars.mmwVelSu, d_MAlab->d_wVel_mmNonlinSrcLabel,
		      matlIndex, patch,
		      Ghost::None, zeroGhostCells);
	  new_dw->getCopy(pressureVars.mmwVelSp, d_MAlab->d_wVel_mmLinSrcLabel,
		      matlIndex, patch,
		      Ghost::None, zeroGhostCells);
	  break;
	}
      }
      
      for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++) {

	switch(index) {

	case Arches::XDIR:

	  new_dw->allocate(pressureVars.uVelocityCoeff[ii], 
			   d_lab->d_uVelCoefPBLMLabel, ii, patch);
	  new_dw->allocate(pressureVars.uVelocityConvectCoeff[ii], 
			   d_lab->d_uVelConvCoefPBLMLabel, ii, patch);
	  break;
	case Arches::YDIR:
	  new_dw->allocate(pressureVars.vVelocityCoeff[ii], 
			   d_lab->d_vVelCoefPBLMLabel, ii, patch);
	  new_dw->allocate(pressureVars.vVelocityConvectCoeff[ii],
			   d_lab->d_vVelConvCoefPBLMLabel, ii, patch);
	  break;
	case Arches::ZDIR:
	  new_dw->allocate(pressureVars.wVelocityCoeff[ii], 
			   d_lab->d_wVelCoefPBLMLabel, ii, patch);
	  new_dw->allocate(pressureVars.wVelocityConvectCoeff[ii], 
			   d_lab->d_wVelConvCoefPBLMLabel, ii, patch);
	  break;
	default:
	  throw InvalidValue("invalid index for velocity in PressureSolver");
	}
      }

      // Calculate Velocity Coeffs :
      //  inputs : [u,v,w]VelocitySIVBC, densityIN, viscosityIN
      //  outputs: [u,v,w]VelCoefPBLM, [u,v,w]VelConvCoefPBLM 

      d_discretize->calculateVelocityCoeff(pc, patch, 
					   delta_t, index, 
					   cellinfo, &pressureVars);

      // Calculate Velocity source
      //  inputs : [u,v,w]VelocitySIVBC, densityIN, viscosityIN
      //  outputs: [u,v,w]VelLinSrcPBLM, [u,v,w]VelNonLinSrcPBLM
      // get data
      // allocate
      
      switch(index) {

      case Arches::XDIR:

	new_dw->allocate(pressureVars.uVelLinearSrc, 
			 d_lab->d_uVelLinSrcPBLMLabel, matlIndex, patch);
	new_dw->allocate(pressureVars.uVelNonlinearSrc, 
			 d_lab->d_uVelNonLinSrcPBLMLabel,
			 matlIndex, patch);
	break;

      case Arches::YDIR:

	new_dw->allocate(pressureVars.vVelLinearSrc, 
			 d_lab->d_vVelLinSrcPBLMLabel, matlIndex, patch);
	new_dw->allocate(pressureVars.vVelNonlinearSrc, 
			 d_lab->d_vVelNonLinSrcPBLMLabel, matlIndex, patch);
	break;

      case Arches::ZDIR:

	new_dw->allocate(pressureVars.wVelLinearSrc, 
			 d_lab->d_wVelLinSrcPBLMLabel, matlIndex, patch);
	new_dw->allocate(pressureVars.wVelNonlinearSrc,
			 d_lab->d_wVelNonLinSrcPBLMLabel, matlIndex, patch);
	break;

      default:
	throw InvalidValue("Invalid index in PressureSolver for calcVelSrc");

      }

      d_source->calculateVelocitySource(pc, patch, 
					delta_t, index,
					cellinfo, &pressureVars);

      // add multimaterial momentum source term

      if (d_MAlab)
	d_source->computemmMomentumSource(pc, patch, index, cellinfo,
					  &pressureVars);

      // Calculate the Velocity BCS
      //  inputs : densityCP, [u,v,w]VelocitySIVBC, [u,v,w]VelCoefPBLM
      //           [u,v,w]VelLinSrcPBLM, [u,v,w]VelNonLinSrcPBLM
      //  outputs: [u,v,w]VelCoefPBLM, [u,v,w]VelLinSrcPBLM, 
      //           [u,v,w]VelNonLinSrcPBLM
      
    d_boundaryCondition->velocityBC(pc, patch, 
				    index,
				    cellinfo, &pressureVars);
    // apply multimaterial velocity bc
    // treats multimaterial wall as intrusion

    if (d_MAlab)
      d_boundaryCondition->mmvelocityBC(pc, patch, index, cellinfo, &pressureVars);
    
    // Modify Velocity Mass Source
    //  inputs : [u,v,w]VelocitySIVBC, [u,v,w]VelCoefPBLM, 
    //           [u,v,w]VelConvCoefPBLM, [u,v,w]VelLinSrcPBLM, 
    //           [u,v,w]VelNonLinSrcPBLM
    //  outputs: [u,v,w]VelLinSrcPBLM, [u,v,w]VelNonLinSrcPBLM

    d_source->modifyVelMassSource(pc, patch, delta_t, index,
				  &pressureVars);

    // Calculate Velocity diagonal
    //  inputs : [u,v,w]VelCoefPBLM, [u,v,w]VelLinSrcPBLM
    //  outputs: [u,v,w]VelCoefPBLM

    d_discretize->calculateVelDiagonal(pc, patch,
				       index,
				       &pressureVars);
#ifdef ARCHES_PRES_DEBUG
    std::cerr << "Done building matrix for press coeff" << endl;
#endif
    
    }

  // put required vars

    for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++) {
      new_dw->put(pressureVars.uVelocityCoeff[ii], d_lab->d_uVelCoefPBLMLabel, 
		  ii, patch);
      new_dw->put(pressureVars.vVelocityCoeff[ii], d_lab->d_vVelCoefPBLMLabel, 
		  ii, patch);
      new_dw->put(pressureVars.wVelocityCoeff[ii], d_lab->d_wVelCoefPBLMLabel, 
		  ii, patch);
    }
    new_dw->put(pressureVars.uVelNonlinearSrc, 
		d_lab->d_uVelNonLinSrcPBLMLabel, matlIndex, patch);
    new_dw->put(pressureVars.uVelLinearSrc, 
		d_lab->d_uVelLinSrcPBLMLabel, matlIndex, patch);
    new_dw->put(pressureVars.vVelNonlinearSrc, 
		d_lab->d_vVelNonLinSrcPBLMLabel, matlIndex, patch);
    new_dw->put(pressureVars.vVelLinearSrc, 
		d_lab->d_vVelLinSrcPBLMLabel, matlIndex, patch);
    new_dw->put(pressureVars.wVelNonlinearSrc, 
		d_lab->d_wVelNonLinSrcPBLMLabel, matlIndex, patch);
    new_dw->put(pressureVars.wVelLinearSrc, 
		d_lab->d_wVelLinSrcPBLMLabel, matlIndex, patch);

#ifdef ARCHES_PRES_DEBUG
    std::cerr << "Done building matrix for vel coeff for pressure" << endl;
#endif
    
  }
}
// ****************************************************************************
// Actually build of linear matrix for pressure equation
// ****************************************************************************
void 
PressureSolver::buildLinearMatrixPress(const ProcessorGroup* pc,
				       const PatchSubset* patches,
				       const MaterialSubset*/* matls */,
				       DataWarehouse* old_dw,
				       DataWarehouse* new_dw)
{
  delt_vartype delT;
  old_dw->get(delT, d_lab->d_sharedState->get_delt_label() );
  double delta_t = delT;
  
  // Get the pressure, velocity, scalars, density and viscosity from the
  // old datawarehouse
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    ArchesVariables pressureVars;
  // compute all three componenets of velocity stencil coefficients
    int numGhostCells = 1;
    int zeroGhostCells = 0;
    int nofStencils = 7;
    // Get the reference density
    // Get the required data
    new_dw->getCopy(pressureVars.density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->getCopy(pressureVars.viscosity, d_lab->d_viscosityINLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->getCopy(pressureVars.pressure, d_lab->d_pressurePSLabel, 
		matlIndex, patch, Ghost::None, zeroGhostCells);
    // Get the PerPatch CellInformation data
    PerPatch<CellInformationP> cellInfoP;
    // *** warning..checkpointing
    //  old_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    //  new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    //  old_dw->get(cellInfoP, d_cellInfoLabel, matlIndex, patch);
    if (new_dw->exists(d_lab->d_cellInfoLabel, matlIndex, patch)) 
      new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    else {
      cellInfoP.setData(scinew CellInformation(patch));
      new_dw->put(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    }
    CellInformation* cellinfo = cellInfoP.get().get_rep();
    new_dw->getCopy(pressureVars.uVelocity, d_lab->d_uVelocityOUTBCLabel, 
		matlIndex, patch, Ghost::AroundFaces, numGhostCells+1);
    new_dw->getCopy(pressureVars.vVelocity, d_lab->d_vVelocityOUTBCLabel, 
		matlIndex, patch, Ghost::AroundFaces, numGhostCells+1);
    new_dw->getCopy(pressureVars.wVelocity, d_lab->d_wVelocityOUTBCLabel, 
		matlIndex, patch, Ghost::AroundFaces, numGhostCells+1);
    // *** warning fix it
    old_dw->getCopy(pressureVars.old_density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::None, zeroGhostCells);
    new_dw->getCopy(pressureVars.cellType, d_lab->d_cellTypeLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
  
    for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++) {
      new_dw->getCopy(pressureVars.uVelocityCoeff[ii], 
		  d_lab->d_uVelCoefPBLMLabel, ii, patch,
		  Ghost::AroundFaces, numGhostCells);
      new_dw->getCopy(pressureVars.vVelocityCoeff[ii], 
		  d_lab->d_vVelCoefPBLMLabel, ii, patch,
		  Ghost::AroundFaces, numGhostCells);
      new_dw->getCopy(pressureVars.wVelocityCoeff[ii], 
		  d_lab->d_wVelCoefPBLMLabel, ii, patch,
		  Ghost::AroundFaces, numGhostCells);
    }
    new_dw->getCopy(pressureVars.uVelNonlinearSrc, 
		d_lab->d_uVelNonLinSrcPBLMLabel,
		matlIndex, patch,
		Ghost::AroundFaces, numGhostCells);
    new_dw->getCopy(pressureVars.vVelNonlinearSrc, 
		d_lab->d_vVelNonLinSrcPBLMLabel, matlIndex, patch,
		Ghost::AroundFaces, numGhostCells);
    new_dw->getCopy(pressureVars.wVelNonlinearSrc,
		d_lab->d_wVelNonLinSrcPBLMLabel, matlIndex, patch,
		Ghost::AroundFaces, numGhostCells);
    
    // Calculate Pressure Coeffs
    //  inputs : densityIN, pressureIN, [u,v,w]VelCoefPBLM[Arches::AP]
    //  outputs: presCoefPBLM[Arches::AE..AB] 
    for (int ii = 0; ii < nofStencils; ii++)
      new_dw->allocate(pressureVars.pressCoeff[ii], 
		       d_lab->d_presCoefPBLMLabel, ii, patch);
    d_discretize->calculatePressureCoeff(pc, patch, old_dw, new_dw, 
					 delta_t, cellinfo, &pressureVars);

    // Modify pressure coefficients for multimaterial formulation

    if (d_MAlab) {

      new_dw->getCopy(pressureVars.voidFraction,
		  d_lab->d_mmgasVolFracLabel, matlIndex, patch,
		  Ghost::AroundCells, numGhostCells);

      d_discretize->mmModifyPressureCoeffs(pc, patch, &pressureVars);

    }

    // Calculate Pressure Source
    //  inputs : pressureSPBC, [u,v,w]VelocitySIVBC, densityCP,
    //           [u,v,w]VelCoefPBLM, [u,v,w]VelNonLinSrcPBLM
    //  outputs: presLinSrcPBLM, presNonLinSrcPBLM
    // Allocate space

    new_dw->allocate(pressureVars.pressLinearSrc, 
		     d_lab->d_presLinSrcPBLMLabel, matlIndex, patch);
    new_dw->allocate(pressureVars.pressNonlinearSrc, 
		     d_lab->d_presNonLinSrcPBLMLabel, matlIndex, patch);

    d_source->calculatePressureSource(pc, patch, delta_t,
				      cellinfo, &pressureVars);

    // Calculate Pressure BC
    //  inputs : pressureIN, presCoefPBLM
    //  outputs: presCoefPBLM

    d_boundaryCondition->pressureBC(pc, patch, old_dw, new_dw, 
				    cellinfo, &pressureVars);
    // do multimaterial bc

    if (d_MAlab)
      d_boundaryCondition->mmpressureBC(pc, patch, cellinfo, &pressureVars);

    // Calculate Pressure Diagonal
    //  inputs : presCoefPBLM, presLinSrcPBLM
    //  outputs: presCoefPBLM 

    d_discretize->calculatePressDiagonal(pc, patch, old_dw, new_dw, 
					 &pressureVars);
  
    // put required vars

    for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++) {
      new_dw->put(pressureVars.pressCoeff[ii], d_lab->d_presCoefPBLMLabel, 
		  ii, patch);
    }
    new_dw->put(pressureVars.pressNonlinearSrc, 
		d_lab->d_presNonLinSrcPBLMLabel, matlIndex, patch);

#ifdef ARCHES_PRES_DEBUG
  std::cerr << "Done building matrix for press coeff" << endl;
#endif

  }
}

void 
PressureSolver::pressureLinearSolve_all (const ProcessorGroup* pg,
					 const PatchSubset* patches,
					 const MaterialSubset*,
					 DataWarehouse* old_dw,
					 DataWarehouse* new_dw)
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
    pressureLinearSolve(pg, patch, matlIndex, old_dw, new_dw, pressureVars);
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
    // put back the results
    new_dw->put(pressureVars.pressure, d_lab->d_pressureSPBCLabel, 
		matlIndex, patch);
  }

  // destroy matrix
  d_linearSolver->destroyMatrix();
}

// Actual linear solve
void 
PressureSolver::pressureLinearSolve (const ProcessorGroup* pc,
				     const Patch* patch,
				     const int matlIndex,
				     DataWarehouse* /*old_dw*/,
				     DataWarehouse* new_dw,
				     ArchesVariables& pressureVars)
{
  int numGhostCells = 1;
  int zeroGhostCells = 0;
  // Get the required data
  {
  new_dw->allocate(pressureVars.pressure, d_lab->d_pressurePredLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
  new_dw->copyOut(pressureVars.pressure, d_lab->d_pressurePSLabel, 
	      matlIndex, patch, Ghost::AroundCells, numGhostCells);
  }
  for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++) 
    new_dw->getCopy(pressureVars.pressCoeff[ii], d_lab->d_presCoefPBLMLabel, 
		   ii, patch, Ghost::None, zeroGhostCells);

  {
  new_dw->allocate(pressureVars.pressNonlinearSrc, 
		d_lab->d_presNonLinSrcPBLMLabel, matlIndex, patch, Ghost::None, zeroGhostCells);
  new_dw->copyOut(pressureVars.pressNonlinearSrc, 
		 d_lab->d_presNonLinSrcPBLMLabel, 
		 matlIndex, patch, Ghost::None, zeroGhostCells);
  }

#if 0
  // compute eqn residual, L1 norm
  new_dw->allocate(pressureVars.residualPressure, d_lab->d_pressureRes,
			  matlIndex, patch);
  d_linearSolver->computePressResidual(pc, patch, old_dw, new_dw, 
				       &pressureVars);
#else
  pressureVars.residPress=pressureVars.truncPress=0;
#endif
  new_dw->put(sum_vartype(pressureVars.residPress), d_lab->d_presResidPSLabel);
  new_dw->put(sum_vartype(pressureVars.truncPress), d_lab->d_presTruncPSLabel);
  // apply underelaxation to eqn
  d_linearSolver->computePressUnderrelax(pc, patch, 
					 &pressureVars);
  // put back computed matrix coeffs and nonlinear source terms 
  // modified as a result of underrelaxation 
  // into the matrix datawarehouse

  // for parallel code lisolve will become a recursive task and 
  // will make the following subroutine separate
  // get patch numer ***warning****
  // sets matrix
  d_linearSolver->setPressMatrix(pc, patch, &pressureVars, d_lab);
  //  d_linearSolver->pressLinearSolve();
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

  int zeroGhostCells = 0;
  
  tsk->requires(Task::NewDW, d_lab->d_pressurePSLabel,
		Ghost::None, zeroGhostCells);
  tsk->requires(Task::OldDW, d_lab->d_densityMicroLabel,
		Ghost::None, zeroGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_mmcellTypeLabel,
		Ghost::None, zeroGhostCells);

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

    int zeroGhostCells = 0;

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
		matlIndex, patch, Ghost::None, zeroGhostCells);
    old_dw->get(denMicro, d_lab->d_densityMicroLabel,
		matlIndex, patch, Ghost::None, zeroGhostCells);
    new_dw->get(cellType, d_lab->d_mmcellTypeLabel,
		matlIndex, patch, Ghost::None, zeroGhostCells);

    new_dw->allocate(pPlusHydro, d_lab->d_pressPlusHydroLabel,
		     matlIndex, patch);

    IntVector valid_lo = patch->getCellFORTLowIndex();
    IntVector valid_hi = patch->getCellFORTHighIndex();

    int mmwallid = d_boundaryCondition->getMMWallId();

    fort_add_hydrostatic_term_topressure(pPlusHydro, prel, denMicro,
					 gx, gy, gz, cellinfo->xx,
					 cellinfo->yy, cellinfo->zz,
					 valid_lo, valid_hi,
					 cellType, mmwallid);
		
    new_dw->put(pPlusHydro, d_lab->d_pressPlusHydroLabel,
		matlIndex, patch);

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



// ****************************************************************************
// Schedule solve of linearized pressure equation
// ****************************************************************************
void PressureSolver::solvePred(const LevelP& level,
			       SchedulerP& sched)
{
  const PatchSet* patches = level->eachPatch();
  const MaterialSet* matls = d_lab->d_sharedState->allArchesMaterials();

  //computes stencil coefficients and source terms
  // require : old_dw -> pressureSPBC, densityCP, viscosityCTS, [u,v,w]VelocitySPBC
  //           new_dw -> pressureSPBC, densityCP, viscosityCTS, [u,v,w]VelocitySIVBC
  // compute : uVelConvCoefPBLM, vVelConvCoefPBLM, wVelConvCoefPBLM
  //           uVelCoefPBLM, vVelCoefPBLM, wVelCoefPBLM, uVelLinSrcPBLM
  //           vVelLinSrcPBLM, wVelLinSrcPBLM, uVelNonLinSrcPBLM 
  //           vVelNonLinSrcPBLM, wVelNonLinSrcPBLM, presCoefPBLM 
  //           presLinSrcPBLM, presNonLinSrcPBLM
  //sched_buildLinearMatrix(level, sched, new_dw, matrix_dw, delta_t);
  // build the structure and get all the old variables

  sched_buildLinearMatrixPred(sched, patches, matls);

  // Schedule the pressure solve
  // require : pressureIN, presCoefPBLM, presNonLinSrcPBLM
  // compute : presResidualPS, presCoefPS, presNonLinSrcPS, pressurePS
  //d_linearSolver->sched_pressureSolve(level, sched, new_dw, matrix_dw);

  sched_pressureLinearSolvePred(level, sched);

  if (d_MAlab) {

    // currently coded only for predictor-only case...

#ifdef correctorstep
#else
    sched_addHydrostaticTermtoPressure(sched, patches, matls);
#endif

  }

}

// ****************************************************************************
// Schedule build of linear matrix
// ****************************************************************************
void 
PressureSolver::sched_buildLinearMatrixPred(SchedulerP& sched,
					    const PatchSet* patches,
					    const MaterialSet* matls)
{

  // Build momentum equation coefficients and sources that are needed 
  // to later build pressure equation coefficients and source

  {
    Task* tsk = scinew Task( "Psolve::BuildCoeffPred", 
			     this, &PressureSolver::buildLinearMatrixPred);

    int numGhostCells = 1;
    int zeroGhostCells = 0;

    tsk->requires(Task::OldDW, d_lab->d_sharedState->get_delt_label());
    
  // Requires
  // from old_dw for time integration
  // get old_dw from getTop function
    tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel,
		  Ghost::AroundCells, numGhostCells);
    //    tsk->requires(Task::NewDW, d_lab->d_densityINLabel, 
    //		  Ghost::None, zeroGhostCells);
  // from new_dw
  // for new task graph to work

    tsk->requires(Task::NewDW, d_lab->d_pressurePSLabel, 
		  Ghost::AroundCells, numGhostCells);
    tsk->requires(Task::NewDW, d_lab->d_densityINLabel,
		  Ghost::AroundCells, numGhostCells+1);
#ifdef correctorstep
    tsk->requires(Task::NewDW, d_lab->d_densityPredLabel, 
		  Ghost::AroundCells, numGhostCells+1);
#else
    tsk->requires(Task::NewDW, d_lab->d_densityCPLabel, 
		  Ghost::AroundCells, numGhostCells+1);
#endif
    tsk->requires(Task::NewDW, d_lab->d_viscosityINLabel,
		  Ghost::AroundCells, numGhostCells);
    tsk->requires(Task::NewDW, d_lab->d_uVelocityOUTBCLabel,
		  Ghost::AroundFaces, numGhostCells+1);
    tsk->requires(Task::NewDW, d_lab->d_vVelocityOUTBCLabel,
		  Ghost::AroundFaces, numGhostCells+1);
    tsk->requires(Task::NewDW, d_lab->d_wVelocityOUTBCLabel,
		  Ghost::AroundFaces, numGhostCells+1);
    tsk->requires(Task::NewDW, d_lab->d_denRefArrayLabel,
    		  Ghost::AroundCells, numGhostCells);

    // for multi-material
    // requires su_drag[x,y,z], sp_drag[x,y,z] for arches

    if (d_MAlab) {

      tsk->requires(Task::NewDW, d_MAlab->d_uVel_mmLinSrcLabel,
		    Ghost::None, zeroGhostCells);
      tsk->requires(Task::NewDW, d_MAlab->d_uVel_mmNonlinSrcLabel,
		    Ghost::None, zeroGhostCells);
      tsk->requires(Task::NewDW, d_MAlab->d_vVel_mmLinSrcLabel, 
		    Ghost::None, zeroGhostCells);
      tsk->requires(Task::NewDW, d_MAlab->d_vVel_mmNonlinSrcLabel,
		    Ghost::None, zeroGhostCells);
      tsk->requires(Task::NewDW, d_MAlab->d_wVel_mmLinSrcLabel,
		    Ghost::None, zeroGhostCells);
      tsk->requires(Task::NewDW, d_MAlab->d_wVel_mmNonlinSrcLabel,
		    Ghost::None, zeroGhostCells);

    }
    
    
    // requires convection coeff because of the nodal
    // differencing
    // computes all the components of velocity

    tsk->computes(d_lab->d_uVelRhoHatLabel);
    tsk->computes(d_lab->d_vVelRhoHatLabel);
    tsk->computes(d_lab->d_wVelRhoHatLabel);
    
    sched->addTask(tsk, patches, matls);
  }

  // Now build pressure equation coefficients from momentum equation 
  // coefficients

  {
    Task* tsk = scinew Task( "Psolve::BuildCoeffPPred",
			     this,
			     &PressureSolver::buildLinearMatrixPressPred);
    
    int numGhostCells = 1;
    int zeroGhostCells = 0;

    tsk->requires(Task::OldDW, d_lab->d_sharedState->get_delt_label());

    // int matlIndex = 0;
    // Requires
    // from old_dw for time integration
    // get old_dw from getTop function

    tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel, 
		  Ghost::AroundCells, numGhostCells);

    // fix it
    // ***warning*June 20th
    tsk->requires(Task::OldDW, d_lab->d_densityINLabel,
		  Ghost::None, zeroGhostCells);
    tsk->requires(Task::NewDW, d_lab->d_densityINLabel,
		  Ghost::AroundCells, numGhostCells);

    // from new_dw

    tsk->requires(Task::NewDW, d_lab->d_pressurePSLabel,
		  Ghost::AroundCells, numGhostCells);
#ifdef correctorstep
    tsk->requires(Task::NewDW, d_lab->d_densityPredLabel, 
		  Ghost::AroundCells, numGhostCells);
#else
    tsk->requires(Task::NewDW, d_lab->d_densityCPLabel, 
		  Ghost::AroundCells, numGhostCells);
#endif
    tsk->requires(Task::NewDW, d_lab->d_viscosityINLabel,
		  Ghost::AroundCells, numGhostCells);
    tsk->requires(Task::NewDW, d_lab->d_uVelocityOUTBCLabel,
		  Ghost::AroundFaces, numGhostCells+1);
    tsk->requires(Task::NewDW, d_lab->d_vVelocityOUTBCLabel,
		  Ghost::AroundFaces, numGhostCells+1);
    tsk->requires(Task::NewDW, d_lab->d_wVelocityOUTBCLabel,
		  Ghost::AroundFaces, numGhostCells+1);
    
    /// requires convection coeff because of the nodal
    // differencing
    // computes all the components of velocity

    tsk->computes(d_lab->d_presCoefPBLMLabel, d_lab->d_stencilMatl,
		   Task::OutOfDomain);

    tsk->requires(Task::NewDW, d_lab->d_uVelRhoHatLabel,
		  Ghost::AroundFaces, numGhostCells);
    tsk->requires(Task::NewDW, d_lab->d_vVelRhoHatLabel,
		  Ghost::AroundFaces, numGhostCells);
    tsk->requires(Task::NewDW, d_lab->d_wVelRhoHatLabel,
		  Ghost::AroundFaces, numGhostCells);
    tsk->computes(d_lab->d_presNonLinSrcPBLMLabel);

    if (d_MAlab) {
      tsk->requires(Task::NewDW, d_lab->d_mmgasVolFracLabel, 
		    Ghost::AroundCells, numGhostCells);
    }

    sched->addTask(tsk, patches, matls);
  }

}


// ***********************************************************************
// Actual build of linear matrices for momentum components
// ***********************************************************************

void 
PressureSolver::buildLinearMatrixPred(const ProcessorGroup* pc,
				      const PatchSubset* patches,
				      const MaterialSubset* /*matls*/,
				      DataWarehouse* old_dw,
				      DataWarehouse* new_dw)
{
  delt_vartype delT;
  old_dw->get(delT, d_lab->d_sharedState->get_delt_label() );
  double delta_t = delT;
#ifdef correctorstep
  delta_t /= 2.0;
#endif
  
  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    ArchesVariables pressureVars;

    // compute all three componenets of velocity stencil coefficients

    int numGhostCells = 1;
    int zeroGhostCells = 0;

    // Get the reference density
    // Get the required data

#ifdef correctorstep
    new_dw->getCopy(pressureVars.density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells+1);
    new_dw->getCopy(pressureVars.new_density, d_lab->d_densityPredLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
#else
    new_dw->getCopy(pressureVars.density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells+1);
    new_dw->getCopy(pressureVars.new_density, d_lab->d_densityCPLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
#endif
    new_dw->getCopy(pressureVars.viscosity, d_lab->d_viscosityINLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->getCopy(pressureVars.pressure, d_lab->d_pressurePSLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->getCopy(pressureVars.denRefArray, d_lab->d_denRefArrayLabel,
    		matlIndex, patch, Ghost::AroundCells, numGhostCells);

    PerPatch<CellInformationP> cellInfoP;

    //  old_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    // checkpointing
    //  new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    //  old_dw->get(cellInfoP, d_cellInfoLabel, matlIndex, patch);

    if (new_dw->exists(d_lab->d_cellInfoLabel, matlIndex, patch)) 

      new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);

    else {

      cellInfoP.setData(scinew CellInformation(patch));
      new_dw->put(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);

    }

    CellInformation* cellinfo = cellInfoP.get().get_rep();

    new_dw->getCopy(pressureVars.uVelocity, d_lab->d_uVelocityOUTBCLabel, 
		matlIndex, patch, Ghost::AroundFaces, numGhostCells);
    new_dw->getCopy(pressureVars.vVelocity, d_lab->d_vVelocityOUTBCLabel, 
		matlIndex, patch, Ghost::AroundFaces, numGhostCells);
    new_dw->getCopy(pressureVars.wVelocity, d_lab->d_wVelocityOUTBCLabel, 
		matlIndex, patch, Ghost::AroundFaces, numGhostCells);
#if 0
    old_dw->getCopy(pressureVars.old_uVelocity, d_lab->d_uVelocitySPBCLabel, 
		matlIndex, patch, Ghost::AroundFaces, numGhostCells);
    old_dw->getCopy(pressureVars.old_vVelocity, d_lab->d_vVelocitySPBCLabel, 
		matlIndex, patch, Ghost::AroundFaces, numGhostCells);
    old_dw->getCopy(pressureVars.old_wVelocity, d_lab->d_wVelocitySPBCLabel, 
		matlIndex, patch, Ghost::AroundFaces, numGhostCells);
#endif
    new_dw->getCopy(pressureVars.old_uVelocity, d_lab->d_uVelocityOUTBCLabel, 
		matlIndex, patch, Ghost::None, zeroGhostCells);
    new_dw->getCopy(pressureVars.old_vVelocity, d_lab->d_vVelocityOUTBCLabel, 
		matlIndex, patch, Ghost::None, zeroGhostCells);
    new_dw->getCopy(pressureVars.old_wVelocity, d_lab->d_wVelocityOUTBCLabel, 
		matlIndex, patch, Ghost::None, zeroGhostCells);

    new_dw->getCopy(pressureVars.old_density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->getCopy(pressureVars.cellType, d_lab->d_cellTypeLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
    
    for(int index = 1; index <= Arches::NDIM; ++index) {

    // get multimaterial momentum source terms

      if (d_MAlab) {
	switch (index) {
	
	case Arches::XDIR:

	  new_dw->getCopy(pressureVars.mmuVelSu, d_MAlab->d_uVel_mmNonlinSrcLabel,
		      matlIndex, patch,
		      Ghost::None, zeroGhostCells);
	  new_dw->getCopy(pressureVars.mmuVelSp, d_MAlab->d_uVel_mmLinSrcLabel,
		      matlIndex, patch,
		      Ghost::None, zeroGhostCells);
	  break;

	case Arches::YDIR:

	  new_dw->getCopy(pressureVars.mmvVelSu, d_MAlab->d_vVel_mmNonlinSrcLabel,
		      matlIndex, patch,
		      Ghost::None, zeroGhostCells);
	  new_dw->getCopy(pressureVars.mmvVelSp, d_MAlab->d_vVel_mmLinSrcLabel,
		      matlIndex, patch,
		      Ghost::None, zeroGhostCells);
	  break;
	case Arches::ZDIR:

	  new_dw->getCopy(pressureVars.mmwVelSu, d_MAlab->d_wVel_mmNonlinSrcLabel,
		      matlIndex, patch,
		      Ghost::None, zeroGhostCells);
	  new_dw->getCopy(pressureVars.mmwVelSp, d_MAlab->d_wVel_mmLinSrcLabel,
		      matlIndex, patch,
		      Ghost::None, zeroGhostCells);
	  break;
	}
      }
      
      for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++) {

	switch(index) {

	case Arches::XDIR:

	  new_dw->allocate(pressureVars.uVelocityCoeff[ii], 
			   d_lab->d_uVelCoefPBLMLabel, ii, patch);
	  new_dw->allocate(pressureVars.uVelocityConvectCoeff[ii], 
			   d_lab->d_uVelConvCoefPBLMLabel, ii, patch);
	  break;
	case Arches::YDIR:
	  new_dw->allocate(pressureVars.vVelocityCoeff[ii], 
			   d_lab->d_vVelCoefPBLMLabel, ii, patch);
	  new_dw->allocate(pressureVars.vVelocityConvectCoeff[ii],
			   d_lab->d_vVelConvCoefPBLMLabel, ii, patch);
	  break;
	case Arches::ZDIR:
	  new_dw->allocate(pressureVars.wVelocityCoeff[ii], 
			   d_lab->d_wVelCoefPBLMLabel, ii, patch);
	  new_dw->allocate(pressureVars.wVelocityConvectCoeff[ii], 
			   d_lab->d_wVelConvCoefPBLMLabel, ii, patch);
	  break;
	default:
	  throw InvalidValue("invalid index for velocity in PressureSolver");
	}
      }

      // Calculate Velocity Coeffs :
      //  inputs : [u,v,w]VelocitySIVBC, densityIN, viscosityIN
      //  outputs: [u,v,w]VelCoefPBLM, [u,v,w]VelConvCoefPBLM 

      d_discretize->calculateVelocityCoeff(pc, patch, 
					   delta_t, index, 
					   cellinfo, &pressureVars);

      // Calculate Velocity source
      //  inputs : [u,v,w]VelocitySIVBC, densityIN, viscosityIN
      //  outputs: [u,v,w]VelLinSrcPBLM, [u,v,w]VelNonLinSrcPBLM
      // get data
      // allocate
      
      switch(index) {

      case Arches::XDIR:

	new_dw->allocate(pressureVars.uVelLinearSrc, 
			 d_lab->d_uVelLinSrcPBLMLabel, matlIndex, patch);
	new_dw->allocate(pressureVars.uVelNonlinearSrc, 
			 d_lab->d_uVelNonLinSrcPBLMLabel,
			 matlIndex, patch);
	new_dw->allocate(pressureVars.uVelRhoHat, 
			 d_lab->d_uVelRhoHatLabel,
			 matlIndex, patch, Ghost::AroundFaces, numGhostCells);
	pressureVars.uVelRhoHat.copy(pressureVars.uVelocity,
				     pressureVars.uVelRhoHat.getLowIndex(),
				     pressureVars.uVelRhoHat.getHighIndex());

	break;

      case Arches::YDIR:

	new_dw->allocate(pressureVars.vVelLinearSrc, 
			 d_lab->d_vVelLinSrcPBLMLabel, matlIndex, patch);
	new_dw->allocate(pressureVars.vVelNonlinearSrc, 
			 d_lab->d_vVelNonLinSrcPBLMLabel, matlIndex, patch);
	new_dw->allocate(pressureVars.vVelRhoHat, 
			 d_lab->d_vVelRhoHatLabel,
			 matlIndex, patch, Ghost::AroundFaces, numGhostCells);
	pressureVars.vVelRhoHat.copy(pressureVars.vVelocity,
				     pressureVars.vVelRhoHat.getLowIndex(),
				     pressureVars.vVelRhoHat.getHighIndex());

	break;

      case Arches::ZDIR:

	new_dw->allocate(pressureVars.wVelLinearSrc, 
			 d_lab->d_wVelLinSrcPBLMLabel, matlIndex, patch);
	new_dw->allocate(pressureVars.wVelNonlinearSrc,
			 d_lab->d_wVelNonLinSrcPBLMLabel, matlIndex, patch);
	new_dw->allocate(pressureVars.wVelRhoHat, 
			 d_lab->d_wVelRhoHatLabel,
			 matlIndex, patch, Ghost::AroundFaces, numGhostCells);
	pressureVars.wVelRhoHat.copy(pressureVars.wVelocity,
				     pressureVars.wVelRhoHat.getLowIndex(),
				     pressureVars.wVelRhoHat.getHighIndex());

	break;

      default:
	throw InvalidValue("Invalid index in PressureSolver for calcVelSrc");

      }

      d_source->calculateVelocitySource(pc, patch, 
					delta_t, index,
					cellinfo, &pressureVars);
      //      d_source->addPressureSource(pc, patch, delta_t, index,
      //				  cellinfo, &pressureVars);
      // add multimaterial momentum source term

      if (d_MAlab)
	d_source->computemmMomentumSource(pc, patch, index, cellinfo,
					  &pressureVars);

      // Calculate the Velocity BCS
      //  inputs : densityCP, [u,v,w]VelocitySIVBC, [u,v,w]VelCoefPBLM
      //           [u,v,w]VelLinSrcPBLM, [u,v,w]VelNonLinSrcPBLM
      //  outputs: [u,v,w]VelCoefPBLM, [u,v,w]VelLinSrcPBLM, 
      //           [u,v,w]VelNonLinSrcPBLM
      
    d_boundaryCondition->velocityBC(pc, patch, 
				    index,
				    cellinfo, &pressureVars);
    // apply multimaterial velocity bc
    // treats multimaterial wall as intrusion

    if (d_MAlab)
      d_boundaryCondition->mmvelocityBC(pc, patch, index, cellinfo, &pressureVars);
    
    // Modify Velocity Mass Source
    //  inputs : [u,v,w]VelocitySIVBC, [u,v,w]VelCoefPBLM, 
    //           [u,v,w]VelConvCoefPBLM, [u,v,w]VelLinSrcPBLM, 
    //           [u,v,w]VelNonLinSrcPBLM
    //  outputs: [u,v,w]VelLinSrcPBLM, [u,v,w]VelNonLinSrcPBLM

    d_source->modifyVelMassSource(pc, patch, delta_t, index,
				  &pressureVars);

    // Calculate Velocity diagonal
    //  inputs : [u,v,w]VelCoefPBLM, [u,v,w]VelLinSrcPBLM
    //  outputs: [u,v,w]VelCoefPBLM

    d_discretize->calculateVelDiagonal(pc, patch,
				       index,
				       &pressureVars);
    d_discretize->calculateVelRhoHat(pc, patch, index, delta_t,
				     cellinfo, &pressureVars);

#ifdef ARCHES_PRES_DEBUG
    std::cerr << "Done building matrix for press coeff" << endl;
#endif
    
    }
    d_boundaryCondition->newrecomputePressureBC(pc, patch,
						cellinfo, &pressureVars); 

  // put required vars

    new_dw->put(pressureVars.uVelRhoHat, d_lab->d_uVelRhoHatLabel, 
		matlIndex, patch);
    new_dw->put(pressureVars.vVelRhoHat, d_lab->d_vVelRhoHatLabel, 
		matlIndex, patch);
    new_dw->put(pressureVars.wVelRhoHat, d_lab->d_wVelRhoHatLabel, 
		matlIndex, patch);

#ifdef ARCHES_PRES_DEBUG
    std::cerr << "Done building matrix for vel coeff for pressure" << endl;
#endif
    
  }
}

// ****************************************************************************
// Actually build of linear matrix for pressure equation
// ****************************************************************************
void 
PressureSolver::buildLinearMatrixPressPred(const ProcessorGroup* pc,
					   const PatchSubset* patches,
					   const MaterialSubset*/* matls */,
					   DataWarehouse* old_dw,
					   DataWarehouse* new_dw)
{
  delt_vartype delT;
  old_dw->get(delT, d_lab->d_sharedState->get_delt_label() );
  double delta_t = delT;
#ifdef correctorstep
  delta_t /= 2.0;
#endif
  
  // Get the pressure, velocity, scalars, density and viscosity from the
  // old datawarehouse
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->
                    getArchesMaterial(archIndex)->getDWIndex(); 
    ArchesVariables pressureVars;
  // compute all three componenets of velocity stencil coefficients
    int numGhostCells = 1;
    int zeroGhostCells = 0;
    int nofStencils = 7;
    // Get the reference density
    // Get the required data
#ifdef correctorstep
    new_dw->getCopy(pressureVars.density, d_lab->d_densityPredLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
#else
    new_dw->getCopy(pressureVars.density, d_lab->d_densityCPLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
#endif
    new_dw->getCopy(pressureVars.viscosity, d_lab->d_viscosityINLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->getCopy(pressureVars.pressure, d_lab->d_pressurePSLabel, 
		matlIndex, patch, Ghost::None, zeroGhostCells);
    // Get the PerPatch CellInformation data
    PerPatch<CellInformationP> cellInfoP;
    // *** warning..checkpointing
    //  old_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    //  new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    //  old_dw->get(cellInfoP, d_cellInfoLabel, matlIndex, patch);
    if (new_dw->exists(d_lab->d_cellInfoLabel, matlIndex, patch)) 
      new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    else {
      cellInfoP.setData(scinew CellInformation(patch));
      new_dw->put(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    }
    CellInformation* cellinfo = cellInfoP.get().get_rep();
    new_dw->getCopy(pressureVars.uVelocity, d_lab->d_uVelocityOUTBCLabel, 
		matlIndex, patch, Ghost::AroundFaces, numGhostCells+1);
    new_dw->getCopy(pressureVars.vVelocity, d_lab->d_vVelocityOUTBCLabel, 
		matlIndex, patch, Ghost::AroundFaces, numGhostCells+1);
    new_dw->getCopy(pressureVars.wVelocity, d_lab->d_wVelocityOUTBCLabel, 
		matlIndex, patch, Ghost::AroundFaces, numGhostCells+1);
    new_dw->getCopy(pressureVars.uVelRhoHat, d_lab->d_uVelRhoHatLabel, 
		matlIndex, patch, Ghost::AroundFaces, numGhostCells);
    new_dw->getCopy(pressureVars.vVelRhoHat, d_lab->d_vVelRhoHatLabel, 
		matlIndex, patch, Ghost::AroundFaces, numGhostCells);
    new_dw->getCopy(pressureVars.wVelRhoHat, d_lab->d_wVelRhoHatLabel, 
		matlIndex, patch, Ghost::AroundFaces, numGhostCells);
    //**warning
    new_dw->getCopy(pressureVars.old_density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
    old_dw->getCopy(pressureVars.old_old_density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::None, zeroGhostCells);
    new_dw->getCopy(pressureVars.cellType, d_lab->d_cellTypeLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
  
    // Calculate Pressure Coeffs
    //  inputs : densityIN, pressureIN, [u,v,w]VelCoefPBLM[Arches::AP]
    //  outputs: presCoefPBLM[Arches::AE..AB] 
    for (int ii = 0; ii < nofStencils; ii++)
      new_dw->allocate(pressureVars.pressCoeff[ii], 
		       d_lab->d_presCoefPBLMLabel, ii, patch);
    d_discretize->calculatePressureCoeff(pc, patch, old_dw, new_dw, 
					 delta_t, cellinfo, &pressureVars);

    // Modify pressure coefficients for multimaterial formulation

    if (d_MAlab) {

      new_dw->getCopy(pressureVars.voidFraction,
		  d_lab->d_mmgasVolFracLabel, matlIndex, patch,
		  Ghost::AroundCells, numGhostCells);

      d_discretize->mmModifyPressureCoeffs(pc, patch, &pressureVars);

    }

    // Calculate Pressure Source
    //  inputs : pressureSPBC, [u,v,w]VelocitySIVBC, densityCP,
    //           [u,v,w]VelCoefPBLM, [u,v,w]VelNonLinSrcPBLM
    //  outputs: presLinSrcPBLM, presNonLinSrcPBLM
    // Allocate space

    new_dw->allocate(pressureVars.pressLinearSrc, 
		     d_lab->d_presLinSrcPBLMLabel, matlIndex, patch);
    pressureVars.pressLinearSrc.initialize(0.0);
    new_dw->allocate(pressureVars.pressNonlinearSrc, 
		     d_lab->d_presNonLinSrcPBLMLabel, matlIndex, patch);
    pressureVars.pressNonlinearSrc.initialize(0.0);
    //    d_source->calculatePressureSource(pc, patch, delta_t,
    //				      cellinfo, &pressureVars);
    d_source->calculatePressureSourcePred(pc, patch, delta_t,
    					  cellinfo, &pressureVars);

    // Calculate Pressure BC
    //  inputs : pressureIN, presCoefPBLM
    //  outputs: presCoefPBLM
    d_discretize->calculatePressDiagonal(pc, patch, old_dw, new_dw, 
					 &pressureVars);

    d_boundaryCondition->pressureBC(pc, patch, old_dw, new_dw, 
				    cellinfo, &pressureVars);
    // do multimaterial bc

    if (d_MAlab)
      d_boundaryCondition->mmpressureBC(pc, patch, cellinfo, &pressureVars);

    // Calculate Pressure Diagonal
    //  inputs : presCoefPBLM, presLinSrcPBLM
    //  outputs: presCoefPBLM 

  
    // put required vars

    for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++) {
      new_dw->put(pressureVars.pressCoeff[ii], d_lab->d_presCoefPBLMLabel, 
		  ii, patch);
    }
    new_dw->put(pressureVars.pressNonlinearSrc, 
		d_lab->d_presNonLinSrcPBLMLabel, matlIndex, patch);

#ifdef ARCHES_PRES_DEBUG
  std::cerr << "Done building matrix for press coeff" << endl;
#endif

  }
}

// ****************************************************************************
// Schedule solver for linear matrix
// ****************************************************************************
void 
PressureSolver::sched_pressureLinearSolvePred(const LevelP& level,
					      SchedulerP& sched)
{
  if(d_perproc_patches && d_perproc_patches->removeReference())
    delete d_perproc_patches;

  LoadBalancer* lb = sched->getLoadBalancer();
  d_perproc_patches = lb->createPerProcessorPatchSet(level, d_myworld);
  d_perproc_patches->addReference();
  const MaterialSet* matls = d_lab->d_sharedState->allArchesMaterials();

  Task* tsk = scinew Task("PressureSolver::PressLinearSolvePred",
			  this,
			  &PressureSolver::pressureLinearSolvePred_all);
  int numGhostCells = 1;
  int zeroGhostCells = 0;

  // Requires
  // coefficient for the variable for which solve is invoked

  tsk->requires(Task::NewDW, d_lab->d_pressurePSLabel, 
		Ghost::AroundCells, numGhostCells);
  //  tsk->requires(Task::OldDW, d_lab->d_pressureCorrSPBCLabel, 
  //	Ghost::AroundCells, numGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_presCoefPBLMLabel, d_lab->d_stencilMatl,
		Task::OutOfDomain, Ghost::None, zeroGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_presNonLinSrcPBLMLabel,
		Ghost::None, zeroGhostCells);

  // computes global residual
  tsk->computes(d_lab->d_presResidPSLabel);
  tsk->computes(d_lab->d_presTruncPSLabel);
#ifdef correctorstep
  tsk->computes(d_lab->d_pressurePredLabel);
#else
  tsk->computes(d_lab->d_pressureSPBCLabel);
#endif
  //  tsk->computes(d_lab->d_pressureCorrSPBCLabel);
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
PressureSolver::pressureLinearSolvePred_all (const ProcessorGroup* pg,
					     const PatchSubset* patches,
					     const MaterialSubset*,
					     DataWarehouse* old_dw,
					     DataWarehouse* new_dw)
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
    pressureLinearSolvePred(pg, patch, matlIndex, old_dw, new_dw, pressureVars);
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
#ifdef correctorstep
    new_dw->put(pressureVars.pressure, d_lab->d_pressurePredLabel, 
		matlIndex, patch);
#else
    new_dw->put(pressureVars.pressure, d_lab->d_pressureSPBCLabel, 
		matlIndex, patch);
    //    new_dw->put(pressureVars.pressure, d_lab->d_pressureCorrSPBCLabel, 
    //		matlIndex, patch);
#endif
  }

  // destroy matrix
  d_linearSolver->destroyMatrix();
}

// Actual linear solve
void 
PressureSolver::pressureLinearSolvePred (const ProcessorGroup* pc,
					 const Patch* patch,
					 const int matlIndex,
					 DataWarehouse* old_dw,
					 DataWarehouse* new_dw,
					 ArchesVariables& pressureVars)
{
  int numGhostCells = 1;
  int zeroGhostCells = 0;
  // Get the required data
  {
#ifdef correctorstep
  new_dw->allocate(pressureVars.pressure, d_lab->d_pressurePredLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
#else
  new_dw->allocate(pressureVars.pressure, d_lab->d_pressureSPBCLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
#endif
  new_dw->copyOut(pressureVars.pressure, d_lab->d_pressurePSLabel, 
	      matlIndex, patch, Ghost::AroundCells, numGhostCells);
#if 0
  new_dw->allocate(pressureVars.pressure, d_lab->d_pressureCorrSPBCLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
  new_dw->copyOut(pressureVars.pressure, d_lab->d_pressurePSLabel, 
	      matlIndex, patch, Ghost::AroundCells, numGhostCells);
#endif
  }
  for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++) 
    new_dw->getCopy(pressureVars.pressCoeff[ii], d_lab->d_presCoefPBLMLabel, 
		   ii, patch, Ghost::None, zeroGhostCells);

  new_dw->getCopy(pressureVars.pressNonlinearSrc, 
		  d_lab->d_presNonLinSrcPBLMLabel, 
		  matlIndex, patch, Ghost::None, zeroGhostCells);

#if 0
  // compute eqn residual, L1 norm
  new_dw->allocate(pressureVars.residualPressure, d_lab->d_pressureRes,
			  matlIndex, patch);
  d_linearSolver->computePressResidual(pc, patch, old_dw, new_dw, 
				       &pressureVars);
#else
  pressureVars.residPress=pressureVars.truncPress=0;
#endif
  new_dw->put(sum_vartype(pressureVars.residPress), d_lab->d_presResidPSLabel);
  new_dw->put(sum_vartype(pressureVars.truncPress), d_lab->d_presTruncPSLabel);
  // apply underelaxation to eqn
  d_linearSolver->computePressUnderrelax(pc, patch, 
					 &pressureVars);
  // put back computed matrix coeffs and nonlinear source terms 
  // modified as a result of underrelaxation 
  // into the matrix datawarehouse

  // for parallel code lisolve will become a recursive task and 
  // will make the following subroutine separate
  // get patch numer ***warning****
  // sets matrix
  d_linearSolver->setPressMatrix(pc, patch, &pressureVars, d_lab);
  //  d_linearSolver->pressLinearSolve();
}

// ************************************************************************
// Schedule addition of hydrostatic term to relative pressure calculated
// in pressure solve
// ************************************************************************


// ****************************************************************************
// Schedule solve of linearized pressure equation, corrector step
// ****************************************************************************
void PressureSolver::solveCorr(const LevelP& level,
			       SchedulerP& sched)
{
  const PatchSet* patches = level->eachPatch();
  const MaterialSet* matls = d_lab->d_sharedState->allArchesMaterials();

  //computes stencil coefficients and source terms
  // require : old_dw -> pressureSPBC, densityCP, viscosityCTS, [u,v,w]VelocitySPBC
  //           new_dw -> pressureSPBC, densityCP, viscosityCTS, [u,v,w]VelocitySIVBC
  // compute : uVelConvCoefPBLM, vVelConvCoefPBLM, wVelConvCoefPBLM
  //           uVelCoefPBLM, vVelCoefPBLM, wVelCoefPBLM, uVelLinSrcPBLM
  //           vVelLinSrcPBLM, wVelLinSrcPBLM, uVelNonLinSrcPBLM 
  //           vVelNonLinSrcPBLM, wVelNonLinSrcPBLM, presCoefPBLM 
  //           presLinSrcPBLM, presNonLinSrcPBLM
  //sched_buildLinearMatrix(level, sched, new_dw, matrix_dw, delta_t);
  // build the structure and get all the old variables

  sched_buildLinearMatrixCorr(sched, patches, matls);

  // Schedule the pressure solve
  // require : pressureIN, presCoefPBLM, presNonLinSrcPBLM
  // compute : presResidualPS, presCoefPS, presNonLinSrcPS, pressurePS
  //d_linearSolver->sched_pressureSolve(level, sched, new_dw, matrix_dw);

  sched_pressureLinearSolveCorr(level, sched);

}

// ****************************************************************************
// Schedule build of linear matrix
// ****************************************************************************
void 
PressureSolver::sched_buildLinearMatrixCorr(SchedulerP& sched,
					    const PatchSet* patches,
					    const MaterialSet* matls)
{
  // Build momentum equation coefficients and sources that are needed 
  // to later build pressure equation coefficients and source

  {
    Task* tsk = scinew Task( "Psolve::BuildCoeffCorr", 
			     this, &PressureSolver::buildLinearMatrixCorr);

    int numGhostCells = 1;
    int zeroGhostCells = 0;

    tsk->requires(Task::OldDW, d_lab->d_sharedState->get_delt_label());
    
  // Requires
  // from old_dw for time integration
  // get old_dw from getTop function
    tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel,
		  Ghost::AroundCells, numGhostCells);
    //    tsk->requires(Task::NewDW, d_lab->d_densityINLabel, 
    //		  Ghost::None, zeroGhostCells);
    tsk->requires(Task::NewDW, d_lab->d_uVelocityOUTBCLabel,
		  Ghost::AroundFaces, numGhostCells+1);
    tsk->requires(Task::NewDW, d_lab->d_vVelocityOUTBCLabel,
		  Ghost::AroundFaces, numGhostCells+1);
    tsk->requires(Task::NewDW, d_lab->d_wVelocityOUTBCLabel,
		  Ghost::AroundFaces, numGhostCells+1);

  // from new_dw
  // for new task graph to work

    tsk->requires(Task::NewDW, d_lab->d_pressurePredLabel, 
		  Ghost::AroundCells, numGhostCells);
    tsk->requires(Task::NewDW, d_lab->d_densityINLabel,
		  Ghost::AroundCells, numGhostCells+1);
    tsk->requires(Task::NewDW, d_lab->d_densityCPLabel, 
		  Ghost::AroundCells, numGhostCells);
    tsk->requires(Task::NewDW, d_lab->d_densityPredLabel, 
		  Ghost::AroundCells, numGhostCells+1);
    tsk->requires(Task::NewDW, d_lab->d_viscosityINLabel,
		  Ghost::AroundCells, numGhostCells);
    tsk->requires(Task::NewDW, d_lab->d_uVelocityPredLabel,
		  Ghost::AroundFaces, numGhostCells+1);
    tsk->requires(Task::NewDW, d_lab->d_vVelocityPredLabel,
		  Ghost::AroundFaces, numGhostCells+1);
    tsk->requires(Task::NewDW, d_lab->d_wVelocityPredLabel,
		  Ghost::AroundFaces, numGhostCells+1);
    tsk->requires(Task::NewDW, d_lab->d_denRefArrayLabel,
    		  Ghost::AroundCells, numGhostCells);

    // for multi-material
    // requires su_drag[x,y,z], sp_drag[x,y,z] for arches

    if (d_MAlab) {

      tsk->requires(Task::NewDW, d_MAlab->d_uVel_mmLinSrcLabel,
		    Ghost::None, zeroGhostCells);
      tsk->requires(Task::NewDW, d_MAlab->d_uVel_mmNonlinSrcLabel,
		    Ghost::None, zeroGhostCells);
      tsk->requires(Task::NewDW, d_MAlab->d_vVel_mmLinSrcLabel, 
		    Ghost::None, zeroGhostCells);
      tsk->requires(Task::NewDW, d_MAlab->d_vVel_mmNonlinSrcLabel,
		    Ghost::None, zeroGhostCells);
      tsk->requires(Task::NewDW, d_MAlab->d_wVel_mmLinSrcLabel,
		    Ghost::None, zeroGhostCells);
      tsk->requires(Task::NewDW, d_MAlab->d_wVel_mmNonlinSrcLabel,
		    Ghost::None, zeroGhostCells);

    }
    
    
    // requires convection coeff because of the nodal
    // differencing
    // computes all the components of velocity

    tsk->computes(d_lab->d_uVelRhoHatCorrLabel);
    tsk->computes(d_lab->d_vVelRhoHatCorrLabel);
    tsk->computes(d_lab->d_wVelRhoHatCorrLabel);
    
    sched->addTask(tsk, patches, matls);
  }
  // Now build pressure equation coefficients from momentum equation 
  // coefficients

  {
    Task* tsk = scinew Task( "Psolve::BuildCoeffPCorr",
			     this,
			     &PressureSolver::buildLinearMatrixPressCorr);
    
    int numGhostCells = 1;
    int zeroGhostCells = 0;

    tsk->requires(Task::OldDW, d_lab->d_sharedState->get_delt_label());
    
    // int matlIndex = 0;
    // Requires
    // from old_dw for time integration
    // get old_dw from getTop function

    tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel, 
		  Ghost::AroundCells, numGhostCells);

    tsk->requires(Task::OldDW, d_lab->d_densityINLabel,
		  Ghost::None, zeroGhostCells);
    tsk->requires(Task::NewDW, d_lab->d_densityINLabel,
		  Ghost::AroundCells, numGhostCells);

    // from new_dw

    tsk->requires(Task::NewDW, d_lab->d_pressurePredLabel,
		  Ghost::AroundCells, numGhostCells);
    tsk->requires(Task::NewDW, d_lab->d_densityCPLabel, 
		  Ghost::AroundCells, numGhostCells+1);
    tsk->requires(Task::NewDW, d_lab->d_viscosityINLabel,
		  Ghost::AroundCells, numGhostCells);
    tsk->requires(Task::NewDW, d_lab->d_uVelRhoHatCorrLabel,
		  Ghost::AroundFaces, numGhostCells);
    tsk->requires(Task::NewDW, d_lab->d_vVelRhoHatCorrLabel,
		  Ghost::AroundFaces, numGhostCells);
    tsk->requires(Task::NewDW, d_lab->d_wVelRhoHatCorrLabel,
		  Ghost::AroundFaces, numGhostCells);

    tsk->computes(d_lab->d_presCoefCorrLabel, d_lab->d_stencilMatl,
		   Task::OutOfDomain);
    tsk->computes(d_lab->d_presNonLinSrcCorrLabel);

    if (d_MAlab) {
      tsk->requires(Task::NewDW, d_lab->d_mmgasVolFracLabel, 
		    Ghost::AroundCells, numGhostCells);
    }

    sched->addTask(tsk, patches, matls);
  }

}



void 
PressureSolver::buildLinearMatrixCorr(const ProcessorGroup* pc,
				      const PatchSubset* patches,
				      const MaterialSubset* /*matls*/,
				      DataWarehouse* old_dw,
				      DataWarehouse* new_dw)
{
  delt_vartype delT;
  old_dw->get(delT, d_lab->d_sharedState->get_delt_label() );
  double delta_t = delT;
  
  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    ArchesVariables pressureVars;

    // compute all three componenets of velocity stencil coefficients

    int numGhostCells = 1;
    int zeroGhostCells = 0;

    // Get the reference density
    // Get the required data

    new_dw->getCopy(pressureVars.density, d_lab->d_densityPredLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells+1);
    new_dw->getCopy(pressureVars.new_density, d_lab->d_densityCPLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->getCopy(pressureVars.viscosity, d_lab->d_viscosityINLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->getCopy(pressureVars.pressure, d_lab->d_pressurePredLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->getCopy(pressureVars.denRefArray, d_lab->d_denRefArrayLabel,
    		matlIndex, patch, Ghost::AroundCells, numGhostCells);

    PerPatch<CellInformationP> cellInfoP;

    //  old_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    // checkpointing
    //  new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    //  old_dw->get(cellInfoP, d_cellInfoLabel, matlIndex, patch);

    if (new_dw->exists(d_lab->d_cellInfoLabel, matlIndex, patch)) 

      new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);

    else {

      cellInfoP.setData(scinew CellInformation(patch));
      new_dw->put(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);

    }

    CellInformation* cellinfo = cellInfoP.get().get_rep();

    new_dw->getCopy(pressureVars.uVelocity, d_lab->d_uVelocityPredLabel, 
		matlIndex, patch, Ghost::AroundFaces, numGhostCells);
    new_dw->getCopy(pressureVars.vVelocity, d_lab->d_vVelocityPredLabel, 
		matlIndex, patch, Ghost::AroundFaces, numGhostCells);
    new_dw->getCopy(pressureVars.wVelocity, d_lab->d_wVelocityPredLabel, 
		matlIndex, patch, Ghost::AroundFaces, numGhostCells);
    new_dw->getCopy(pressureVars.old_uVelocity, d_lab->d_uVelocityOUTBCLabel, 
		matlIndex, patch, Ghost::None, zeroGhostCells);
    new_dw->getCopy(pressureVars.old_vVelocity, d_lab->d_vVelocityOUTBCLabel, 
		matlIndex, patch, Ghost::None, zeroGhostCells);
    new_dw->getCopy(pressureVars.old_wVelocity, d_lab->d_wVelocityOUTBCLabel, 
		matlIndex, patch, Ghost::None, zeroGhostCells);

    new_dw->getCopy(pressureVars.old_density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->getCopy(pressureVars.cellType, d_lab->d_cellTypeLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
    
    for(int index = 1; index <= Arches::NDIM; ++index) {

    // get multimaterial momentum source terms

      if (d_MAlab) {
	switch (index) {
	
	case Arches::XDIR:

	  new_dw->getCopy(pressureVars.mmuVelSu, d_MAlab->d_uVel_mmNonlinSrcLabel,
		      matlIndex, patch,
		      Ghost::None, zeroGhostCells);
	  new_dw->getCopy(pressureVars.mmuVelSp, d_MAlab->d_uVel_mmLinSrcLabel,
		      matlIndex, patch,
		      Ghost::None, zeroGhostCells);
	  break;

	case Arches::YDIR:

	  new_dw->getCopy(pressureVars.mmvVelSu, d_MAlab->d_vVel_mmNonlinSrcLabel,
		      matlIndex, patch,
		      Ghost::None, zeroGhostCells);
	  new_dw->getCopy(pressureVars.mmvVelSp, d_MAlab->d_vVel_mmLinSrcLabel,
		      matlIndex, patch,
		      Ghost::None, zeroGhostCells);
	  break;
	case Arches::ZDIR:

	  new_dw->getCopy(pressureVars.mmwVelSu, d_MAlab->d_wVel_mmNonlinSrcLabel,
		      matlIndex, patch,
		      Ghost::None, zeroGhostCells);
	  new_dw->getCopy(pressureVars.mmwVelSp, d_MAlab->d_wVel_mmLinSrcLabel,
		      matlIndex, patch,
		      Ghost::None, zeroGhostCells);
	  break;
	}
      }
      
      for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++) {

	switch(index) {

	case Arches::XDIR:

	  new_dw->allocate(pressureVars.uVelocityCoeff[ii], 
			   d_lab->d_uVelCoefPBLMCorrLabel, ii, patch);
	  new_dw->allocate(pressureVars.uVelocityConvectCoeff[ii], 
			   d_lab->d_uVelConvCoefPBLMCorrLabel, ii, patch);
	  break;
	case Arches::YDIR:
	  new_dw->allocate(pressureVars.vVelocityCoeff[ii], 
			   d_lab->d_vVelCoefPBLMCorrLabel, ii, patch);
	  new_dw->allocate(pressureVars.vVelocityConvectCoeff[ii],
			   d_lab->d_vVelConvCoefPBLMCorrLabel, ii, patch);
	  break;
	case Arches::ZDIR:
	  new_dw->allocate(pressureVars.wVelocityCoeff[ii], 
			   d_lab->d_wVelCoefPBLMCorrLabel, ii, patch);
	  new_dw->allocate(pressureVars.wVelocityConvectCoeff[ii], 
			   d_lab->d_wVelConvCoefPBLMCorrLabel, ii, patch);
	  break;
	default:
	  throw InvalidValue("invalid index for velocity in PressureSolver");
	}
      }

      // Calculate Velocity Coeffs :
      //  inputs : [u,v,w]VelocitySIVBC, densityIN, viscosityIN
      //  outputs: [u,v,w]VelCoefPBLM, [u,v,w]VelConvCoefPBLM 

      d_discretize->calculateVelocityCoeff(pc, patch, 
					   delta_t, index, 
					   cellinfo, &pressureVars);

      // Calculate Velocity source
      //  inputs : [u,v,w]VelocitySIVBC, densityIN, viscosityIN
      //  outputs: [u,v,w]VelLinSrcPBLM, [u,v,w]VelNonLinSrcPBLM
      // get data
      // allocate
      
      switch(index) {

      case Arches::XDIR:

	new_dw->allocate(pressureVars.uVelLinearSrc, 
			 d_lab->d_uVelLinSrcPBLMCorrLabel, matlIndex, patch);
	new_dw->allocate(pressureVars.uVelNonlinearSrc, 
			 d_lab->d_uVelNonLinSrcPBLMCorrLabel,
			 matlIndex, patch);
	new_dw->allocate(pressureVars.uVelRhoHat, 
			 d_lab->d_uVelRhoHatCorrLabel,
			 matlIndex, patch, Ghost::AroundFaces, numGhostCells);
	pressureVars.uVelRhoHat.copy(pressureVars.uVelocity,
				     pressureVars.uVelRhoHat.getLowIndex(),
				     pressureVars.uVelRhoHat.getHighIndex());

	break;

      case Arches::YDIR:

	new_dw->allocate(pressureVars.vVelLinearSrc, 
			 d_lab->d_vVelLinSrcPBLMCorrLabel, matlIndex, patch);
	new_dw->allocate(pressureVars.vVelNonlinearSrc, 
			 d_lab->d_vVelNonLinSrcPBLMCorrLabel, matlIndex, patch);
	new_dw->allocate(pressureVars.vVelRhoHat, 
			 d_lab->d_vVelRhoHatCorrLabel,
			 matlIndex, patch, Ghost::AroundFaces, numGhostCells);
	pressureVars.vVelRhoHat.copy(pressureVars.vVelocity,
				     pressureVars.vVelRhoHat.getLowIndex(),
				     pressureVars.vVelRhoHat.getHighIndex());

	break;

      case Arches::ZDIR:

	new_dw->allocate(pressureVars.wVelLinearSrc, 
			 d_lab->d_wVelLinSrcPBLMCorrLabel, matlIndex, patch);
	new_dw->allocate(pressureVars.wVelNonlinearSrc,
			 d_lab->d_wVelNonLinSrcPBLMCorrLabel, matlIndex, patch);
	new_dw->allocate(pressureVars.wVelRhoHat, 
			 d_lab->d_wVelRhoHatCorrLabel,
			 matlIndex, patch, Ghost::AroundFaces, numGhostCells);
	pressureVars.wVelRhoHat.copy(pressureVars.wVelocity,
				     pressureVars.wVelRhoHat.getLowIndex(),
				     pressureVars.wVelRhoHat.getHighIndex());

	break;

      default:
	throw InvalidValue("Invalid index in PressureSolver for calcVelSrc");

      }

      d_source->calculateVelocitySource(pc, patch, 
					delta_t, index,
					cellinfo, &pressureVars);
      //      d_source->addPressureSource(pc, patch, delta_t, index,
      //				  cellinfo, &pressureVars);
      // add multimaterial momentum source term

      if (d_MAlab)
	d_source->computemmMomentumSource(pc, patch, index, cellinfo,
					  &pressureVars);

      // Calculate the Velocity BCS
      //  inputs : densityCP, [u,v,w]VelocitySIVBC, [u,v,w]VelCoefPBLM
      //           [u,v,w]VelLinSrcPBLM, [u,v,w]VelNonLinSrcPBLM
      //  outputs: [u,v,w]VelCoefPBLM, [u,v,w]VelLinSrcPBLM, 
      //           [u,v,w]VelNonLinSrcPBLM
      
    d_boundaryCondition->velocityBC(pc, patch, 
				    index,
				    cellinfo, &pressureVars);
    // apply multimaterial velocity bc
    // treats multimaterial wall as intrusion

    if (d_MAlab)
      d_boundaryCondition->mmvelocityBC(pc, patch, index, cellinfo, &pressureVars);
    
    // Modify Velocity Mass Source
    //  inputs : [u,v,w]VelocitySIVBC, [u,v,w]VelCoefPBLM, 
    //           [u,v,w]VelConvCoefPBLM, [u,v,w]VelLinSrcPBLM, 
    //           [u,v,w]VelNonLinSrcPBLM
    //  outputs: [u,v,w]VelLinSrcPBLM, [u,v,w]VelNonLinSrcPBLM

    d_source->modifyVelMassSource(pc, patch, delta_t, index,
				  &pressureVars);

    // Calculate Velocity diagonal
    //  inputs : [u,v,w]VelCoefPBLM, [u,v,w]VelLinSrcPBLM
    //  outputs: [u,v,w]VelCoefPBLM

    d_discretize->calculateVelDiagonal(pc, patch,
				       index,
				       &pressureVars);
    d_discretize->calculateVelRhoHat(pc, patch, index, delta_t,
				     cellinfo, &pressureVars);

#ifdef ARCHES_PRES_DEBUG
    std::cerr << "Done building matrix for press coeff" << endl;
#endif
    
    }
    d_boundaryCondition->newrecomputePressureBC(pc, patch,
						cellinfo, &pressureVars); 

  // put required vars

    new_dw->put(pressureVars.uVelRhoHat, d_lab->d_uVelRhoHatCorrLabel, 
		matlIndex, patch);
    new_dw->put(pressureVars.vVelRhoHat, d_lab->d_vVelRhoHatCorrLabel, 
		matlIndex, patch);
    new_dw->put(pressureVars.wVelRhoHat, d_lab->d_wVelRhoHatCorrLabel, 
		matlIndex, patch);

#ifdef ARCHES_PRES_DEBUG
    std::cerr << "Done building matrix for vel coeff for pressure" << endl;
#endif
    
  }
}



// ****************************************************************************
// Actually build of linear matrix for pressure equation
// ****************************************************************************
void 
PressureSolver::buildLinearMatrixPressCorr(const ProcessorGroup* pc,
					   const PatchSubset* patches,
					   const MaterialSubset*/* matls */,
					   DataWarehouse* old_dw,
					   DataWarehouse* new_dw)
{
  delt_vartype delT;
  old_dw->get(delT, d_lab->d_sharedState->get_delt_label() );
  double delta_t = delT;
  
  // Get the pressure, velocity, scalars, density and viscosity from the
  // old datawarehouse
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->
                    getArchesMaterial(archIndex)->getDWIndex(); 
    ArchesVariables pressureVars;
  // compute all three componenets of velocity stencil coefficients
    int numGhostCells = 1;
    int zeroGhostCells = 0;
    int nofStencils = 7;
    // Get the reference density
    // Get the required data
    new_dw->getCopy(pressureVars.density, d_lab->d_densityCPLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->getCopy(pressureVars.viscosity, d_lab->d_viscosityINLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->getCopy(pressureVars.pressure, d_lab->d_pressurePredLabel, 
		matlIndex, patch, Ghost::None, zeroGhostCells);
    // Get the PerPatch CellInformation data
    PerPatch<CellInformationP> cellInfoP;
    // *** warning..checkpointing
    //  old_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    //  new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    //  old_dw->get(cellInfoP, d_cellInfoLabel, matlIndex, patch);
    if (new_dw->exists(d_lab->d_cellInfoLabel, matlIndex, patch)) 
      new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    else {
      cellInfoP.setData(scinew CellInformation(patch));
      new_dw->put(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    }
    CellInformation* cellinfo = cellInfoP.get().get_rep();
    new_dw->getCopy(pressureVars.old_density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
    old_dw->getCopy(pressureVars.old_old_density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::None, zeroGhostCells);
    new_dw->getCopy(pressureVars.pred_density, d_lab->d_densityCPLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->getCopy(pressureVars.cellType, d_lab->d_cellTypeLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
  
    new_dw->getCopy(pressureVars.uVelRhoHat, d_lab->d_uVelRhoHatCorrLabel, 
		matlIndex, patch, Ghost::AroundFaces, numGhostCells);
    new_dw->getCopy(pressureVars.vVelRhoHat, d_lab->d_vVelRhoHatCorrLabel, 
		matlIndex, patch, Ghost::AroundFaces, numGhostCells);
    new_dw->getCopy(pressureVars.wVelRhoHat, d_lab->d_wVelRhoHatCorrLabel, 
		matlIndex, patch, Ghost::AroundFaces, numGhostCells);
    
    // Calculate Pressure Coeffs
    //  inputs : densityIN, pressureIN, [u,v,w]VelCoefPBLM[Arches::AP]
    //  outputs: presCoefPBLM[Arches::AE..AB] 
    for (int ii = 0; ii < nofStencils; ii++) {
      new_dw->allocate(pressureVars.pressCoeff[ii], 
		       d_lab->d_presCoefCorrLabel, ii, patch);
      pressureVars.pressCoeff[ii].initialize(0.0);
    }
    d_discretize->calculatePressureCoeff(pc, patch, old_dw, new_dw, 
					 delta_t, cellinfo, &pressureVars);

    // Modify pressure coefficients for multimaterial formulation

    if (d_MAlab) {

      new_dw->getCopy(pressureVars.voidFraction,
		  d_lab->d_mmgasVolFracLabel, matlIndex, patch,
		  Ghost::AroundCells, numGhostCells);

      d_discretize->mmModifyPressureCoeffs(pc, patch, &pressureVars);

    }

    // Calculate Pressure Source
    //  inputs : pressureSPBC, [u,v,w]VelocitySIVBC, densityCP,
    //           [u,v,w]VelCoefPBLM, [u,v,w]VelNonLinSrcPBLM
    //  outputs: presLinSrcPBLM, presNonLinSrcPBLM
    // Allocate space

    new_dw->allocate(pressureVars.pressLinearSrc, 
		     d_lab->d_presLinSrcCorrLabel, matlIndex, patch);
    new_dw->allocate(pressureVars.pressNonlinearSrc, 
		     d_lab->d_presNonLinSrcCorrLabel, matlIndex, patch);
    pressureVars.pressLinearSrc.initialize(0.0);
    pressureVars.pressNonlinearSrc.initialize(0.0);
    d_source->calculatePressureSourcePred(pc, patch, delta_t,
				      cellinfo, &pressureVars);

    // Calculate Pressure BC
    //  inputs : pressureIN, presCoefPBLM
    //  outputs: presCoefPBLM
    d_discretize->calculatePressDiagonal(pc, patch, old_dw, new_dw, 
					 &pressureVars);
    d_boundaryCondition->pressureBC(pc, patch, old_dw, new_dw, 
				    cellinfo, &pressureVars);
    // do multimaterial bc

    if (d_MAlab)
      d_boundaryCondition->mmpressureBC(pc, patch, cellinfo, &pressureVars);

    // Calculate Pressure Diagonal
    //  inputs : presCoefPBLM, presLinSrcPBLM
    //  outputs: presCoefPBLM 

  
    // put required vars

    for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++) {
      new_dw->put(pressureVars.pressCoeff[ii], d_lab->d_presCoefCorrLabel, 
		  ii, patch);
    }
    new_dw->put(pressureVars.pressNonlinearSrc, 
		d_lab->d_presNonLinSrcCorrLabel, matlIndex, patch);

#ifdef ARCHES_PRES_DEBUG
  std::cerr << "Done building matrix for press coeff" << endl;
#endif

  }
}

// ****************************************************************************
// Schedule solver for linear matrix
// ****************************************************************************
void 
PressureSolver::sched_pressureLinearSolveCorr(const LevelP& level,
					      SchedulerP& sched)
{
  if(d_perproc_patches && d_perproc_patches->removeReference())
    delete d_perproc_patches;

  LoadBalancer* lb = sched->getLoadBalancer();
  d_perproc_patches = lb->createPerProcessorPatchSet(level, d_myworld);
  d_perproc_patches->addReference();
  const MaterialSet* matls = d_lab->d_sharedState->allArchesMaterials();

  Task* tsk = scinew Task("PressureSolver::PressLinearSolveCorr",
			  this,
			  &PressureSolver::pressureLinearSolveCorr_all);
  int numGhostCells = 1;
  int zeroGhostCells = 0;

  // Requires
  // coefficient for the variable for which solve is invoked

  tsk->requires(Task::NewDW, d_lab->d_pressurePredLabel, 
		Ghost::AroundCells, numGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_presCoefCorrLabel, d_lab->d_stencilMatl,
		Task::OutOfDomain, Ghost::None, zeroGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_presNonLinSrcCorrLabel,
		Ghost::None, zeroGhostCells);

  // computes global residual
  tsk->computes(d_lab->d_presResidPSLabel);
  tsk->computes(d_lab->d_presTruncPSLabel);

  tsk->computes(d_lab->d_pressureSPBCLabel);

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
PressureSolver::pressureLinearSolveCorr_all (const ProcessorGroup* pg,
					     const PatchSubset* patches,
					     const MaterialSubset*,
					     DataWarehouse* old_dw,
					     DataWarehouse* new_dw)
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
    pressureLinearSolveCorr(pg, patch, matlIndex, old_dw, new_dw, pressureVars);
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
    // put back the results
    new_dw->put(pressureVars.pressure, d_lab->d_pressureSPBCLabel, 
		matlIndex, patch);
  }

  // destroy matrix
  d_linearSolver->destroyMatrix();
}

// Actual linear solve
void 
PressureSolver::pressureLinearSolveCorr (const ProcessorGroup* pc,
					 const Patch* patch,
					 const int matlIndex,
					 DataWarehouse* /*old_dw*/,
					 DataWarehouse* new_dw,
					 ArchesVariables& pressureVars)
{
  int numGhostCells = 1;
  int zeroGhostCells = 0;
  // Get the required data
  new_dw->allocate(pressureVars.pressure, d_lab->d_pressureSPBCLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
  new_dw->copyOut(pressureVars.pressure, d_lab->d_pressurePredLabel, 
	      matlIndex, patch, Ghost::AroundCells, numGhostCells);
  for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++) 
    new_dw->getCopy(pressureVars.pressCoeff[ii], d_lab->d_presCoefCorrLabel, 
		   ii, patch, Ghost::None, zeroGhostCells);

  new_dw->getCopy(pressureVars.pressNonlinearSrc, 
		 d_lab->d_presNonLinSrcCorrLabel, 
		 matlIndex, patch, Ghost::None, zeroGhostCells);

#if 0
  // compute eqn residual, L1 norm
  new_dw->allocate(pressureVars.residualPressure, d_lab->d_pressureRes,
			  matlIndex, patch);
  d_linearSolver->computePressResidual(pc, patch, old_dw, new_dw, 
				       &pressureVars);
#else
  pressureVars.residPress=pressureVars.truncPress=0;
#endif
  new_dw->put(sum_vartype(pressureVars.residPress), d_lab->d_presResidPSLabel);
  new_dw->put(sum_vartype(pressureVars.truncPress), d_lab->d_presTruncPSLabel);
  // apply underelaxation to eqn
  d_linearSolver->computePressUnderrelax(pc, patch, 
					 &pressureVars);
  // put back computed matrix coeffs and nonlinear source terms 
  // modified as a result of underrelaxation 
  // into the matrix datawarehouse

  // for parallel code lisolve will become a recursive task and 
  // will make the following subroutine separate
  // get patch numer ***warning****
  // sets matrix
  d_linearSolver->setPressMatrix(pc, patch, &pressureVars, d_lab);
  //  d_linearSolver->pressLinearSolve();
}



