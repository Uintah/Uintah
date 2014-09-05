//----- PressureSolver.cc ----------------------------------------------

#include <Packages/Uintah/CCA/Components/Arches/debug.h>
#include <Packages/Uintah/CCA/Components/Arches/PressureSolver.h>
#include <Packages/Uintah/CCA/Components/Arches/Arches.h>
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
#include <Packages/Uintah/CCA/Components/Arches/ScaleSimilarityModel.h>
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


    tsk->requires(Task::OldDW, d_lab->d_sharedState->get_delt_label());
    
  // Requires
  // from old_dw for time integration
  // get old_dw from getTop function
    tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel,
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);
    tsk->requires(Task::OldDW, d_lab->d_densityINLabel, 
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);
#if 0
    tsk->requires(Task::OldDW, d_lab->d_uVelocitySPBCLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
    tsk->requires(Task::OldDW, d_lab->d_vVelocitySPBCLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
    tsk->requires(Task::OldDW, d_lab->d_wVelocitySPBCLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
#endif

  // from new_dw
  // for new task graph to work

    tsk->requires(Task::NewDW, d_lab->d_pressurePSLabel, 
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);
    tsk->requires(Task::NewDW, d_lab->d_densityINLabel,
		  Ghost::AroundCells, Arches::TWOGHOSTCELLS);
    tsk->requires(Task::NewDW, d_lab->d_viscosityINLabel,
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);
    tsk->requires(Task::NewDW, d_lab->d_uVelocityOUTBCLabel,
		  Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    tsk->requires(Task::NewDW, d_lab->d_vVelocityOUTBCLabel,
		  Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    tsk->requires(Task::NewDW, d_lab->d_wVelocityOUTBCLabel,
		  Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    tsk->requires(Task::NewDW, d_lab->d_denRefArrayLabel,
    		  Ghost::AroundCells, Arches::ONEGHOSTCELL);

    // for multi-material
    // requires su_drag[x,y,z], sp_drag[x,y,z] for arches

    if (d_MAlab) {

      tsk->requires(Task::NewDW, d_MAlab->d_uVel_mmLinSrcLabel,
		    Ghost::None, Arches::ZEROGHOSTCELLS);
      tsk->requires(Task::NewDW, d_MAlab->d_uVel_mmNonlinSrcLabel,
		    Ghost::None, Arches::ZEROGHOSTCELLS);
      tsk->requires(Task::NewDW, d_MAlab->d_vVel_mmLinSrcLabel, 
		    Ghost::None, Arches::ZEROGHOSTCELLS);
      tsk->requires(Task::NewDW, d_MAlab->d_vVel_mmNonlinSrcLabel,
		    Ghost::None, Arches::ZEROGHOSTCELLS);
      tsk->requires(Task::NewDW, d_MAlab->d_wVel_mmLinSrcLabel,
		    Ghost::None, Arches::ZEROGHOSTCELLS);
      tsk->requires(Task::NewDW, d_MAlab->d_wVel_mmNonlinSrcLabel,
		    Ghost::None, Arches::ZEROGHOSTCELLS);

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
    

    tsk->requires(Task::OldDW, d_lab->d_sharedState->get_delt_label());
    
    // int matlIndex = 0;
    // Requires
    // from old_dw for time integration
    // get old_dw from getTop function

    tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel, 
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);

    // fix it

    tsk->requires(Task::OldDW, d_lab->d_densityINLabel,
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);

    // from new_dw

    tsk->requires(Task::NewDW, d_lab->d_pressurePSLabel,
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);
    tsk->requires(Task::NewDW, d_lab->d_densityINLabel, 
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);
    tsk->requires(Task::NewDW, d_lab->d_viscosityINLabel,
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);
    tsk->requires(Task::NewDW, d_lab->d_uVelocityOUTBCLabel,
		  Ghost::AroundFaces, Arches::TWOGHOSTCELLS);
    tsk->requires(Task::NewDW, d_lab->d_vVelocityOUTBCLabel,
		  Ghost::AroundFaces, Arches::TWOGHOSTCELLS);
    tsk->requires(Task::NewDW, d_lab->d_wVelocityOUTBCLabel,
		  Ghost::AroundFaces, Arches::TWOGHOSTCELLS);
    
    /// requires convection coeff because of the nodal
    // differencing
    // computes all the components of velocity

    tsk->computes(d_lab->d_presCoefPBLMLabel, d_lab->d_stencilMatl,
		   Task::OutOfDomain);

    tsk->computes(d_lab->d_presNonLinSrcPBLMLabel);

    if (d_MAlab) {
      tsk->requires(Task::NewDW, d_lab->d_mmgasVolFracLabel, 
		    Ghost::AroundCells, Arches::ONEGHOSTCELL);
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

  // Requires
  // coefficient for the variable for which solve is invoked

  tsk->requires(Task::NewDW, d_lab->d_pressurePSLabel, 
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_presCoefPBLMLabel, d_lab->d_stencilMatl,
		Task::OutOfDomain, Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_presNonLinSrcPBLMLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);

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


    // Get the reference density
    // Get the required data

    new_dw->getCopy(pressureVars.density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::TWOGHOSTCELLS);
    new_dw->getCopy(pressureVars.viscosity, d_lab->d_viscosityINLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->getCopy(pressureVars.pressure, d_lab->d_pressurePSLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->getCopy(pressureVars.denRefArray, d_lab->d_denRefArrayLabel,
    		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);

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
		matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->getCopy(pressureVars.vVelocity, d_lab->d_vVelocityOUTBCLabel, 
		matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->getCopy(pressureVars.wVelocity, d_lab->d_wVelocityOUTBCLabel, 
		matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
#if 0
    old_dw->getCopy(pressureVars.old_uVelocity, d_lab->d_uVelocitySPBCLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    old_dw->getCopy(pressureVars.old_vVelocity, d_lab->d_vVelocitySPBCLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    old_dw->getCopy(pressureVars.old_wVelocity, d_lab->d_wVelocitySPBCLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
#endif
    new_dw->getCopy(pressureVars.old_uVelocity, d_lab->d_uVelocityOUTBCLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->getCopy(pressureVars.old_vVelocity, d_lab->d_vVelocityOUTBCLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->getCopy(pressureVars.old_wVelocity, d_lab->d_wVelocityOUTBCLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);

    // modified - June 20th
    new_dw->getCopy(pressureVars.old_density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->getCopy(pressureVars.cellType, d_lab->d_cellTypeLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    
    for(int index = 1; index <= Arches::NDIM; ++index) {

    // get multimaterial momentum source terms

      if (d_MAlab) {
	switch (index) {
	
	case Arches::XDIR:

	  new_dw->getCopy(pressureVars.mmuVelSu, d_MAlab->d_uVel_mmNonlinSrcLabel,
		      matlIndex, patch,
		      Ghost::None, Arches::ZEROGHOSTCELLS);
	  new_dw->getCopy(pressureVars.mmuVelSp, d_MAlab->d_uVel_mmLinSrcLabel,
		      matlIndex, patch,
		      Ghost::None, Arches::ZEROGHOSTCELLS);
	  break;

	case Arches::YDIR:

	  new_dw->getCopy(pressureVars.mmvVelSu, d_MAlab->d_vVel_mmNonlinSrcLabel,
		      matlIndex, patch,
		      Ghost::None, Arches::ZEROGHOSTCELLS);
	  new_dw->getCopy(pressureVars.mmvVelSp, d_MAlab->d_vVel_mmLinSrcLabel,
		      matlIndex, patch,
		      Ghost::None, Arches::ZEROGHOSTCELLS);
	  break;
	case Arches::ZDIR:

	  new_dw->getCopy(pressureVars.mmwVelSu, d_MAlab->d_wVel_mmNonlinSrcLabel,
		      matlIndex, patch,
		      Ghost::None, Arches::ZEROGHOSTCELLS);
	  new_dw->getCopy(pressureVars.mmwVelSp, d_MAlab->d_wVel_mmLinSrcLabel,
		      matlIndex, patch,
		      Ghost::None, Arches::ZEROGHOSTCELLS);
	  break;
	}
      }
      
      for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++) {

	switch(index) {

	case Arches::XDIR:

	  new_dw->allocateAndPut(pressureVars.uVelocityCoeff[ii], d_lab->d_uVelCoefPBLMLabel, ii, patch);
	  new_dw->allocateTemporary(pressureVars.uVelocityConvectCoeff[ii],  patch);
	  break;
	case Arches::YDIR:
	  new_dw->allocateAndPut(pressureVars.vVelocityCoeff[ii], d_lab->d_vVelCoefPBLMLabel, ii, patch);
	  new_dw->allocateTemporary(pressureVars.vVelocityConvectCoeff[ii],  patch);
	  break;
	case Arches::ZDIR:
	  new_dw->allocateAndPut(pressureVars.wVelocityCoeff[ii], d_lab->d_wVelCoefPBLMLabel, ii, patch);
	  new_dw->allocateTemporary(pressureVars.wVelocityConvectCoeff[ii],  patch);
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

	new_dw->allocateAndPut(pressureVars.uVelLinearSrc, d_lab->d_uVelLinSrcPBLMLabel, matlIndex, patch);
	new_dw->allocateAndPut(pressureVars.uVelNonlinearSrc, d_lab->d_uVelNonLinSrcPBLMLabel,
			 matlIndex, patch);
	break;

      case Arches::YDIR:

	new_dw->allocateAndPut(pressureVars.vVelLinearSrc, d_lab->d_vVelLinSrcPBLMLabel, matlIndex, patch);
	new_dw->allocateAndPut(pressureVars.vVelNonlinearSrc, d_lab->d_vVelNonLinSrcPBLMLabel, matlIndex, patch);
	break;

      case Arches::ZDIR:

	new_dw->allocateAndPut(pressureVars.wVelLinearSrc, d_lab->d_wVelLinSrcPBLMLabel, matlIndex, patch);
	new_dw->allocateAndPut(pressureVars.wVelNonlinearSrc, d_lab->d_wVelNonLinSrcPBLMLabel, matlIndex, patch);
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
      // allocateAndPut instead:
      /* new_dw->put(pressureVars.uVelocityCoeff[ii], d_lab->d_uVelCoefPBLMLabel, 
		  ii, patch); */;
      // allocateAndPut instead:
      /* new_dw->put(pressureVars.vVelocityCoeff[ii], d_lab->d_vVelCoefPBLMLabel, 
		  ii, patch); */;
      // allocateAndPut instead:
      /* new_dw->put(pressureVars.wVelocityCoeff[ii], d_lab->d_wVelCoefPBLMLabel, 
		  ii, patch); */;
    }
    // allocateAndPut instead:
    /* new_dw->put(pressureVars.uVelNonlinearSrc, 
		d_lab->d_uVelNonLinSrcPBLMLabel, matlIndex, patch); */;
    // allocateAndPut instead:
    /* new_dw->put(pressureVars.uVelLinearSrc, 
		d_lab->d_uVelLinSrcPBLMLabel, matlIndex, patch); */;
    // allocateAndPut instead:
    /* new_dw->put(pressureVars.vVelNonlinearSrc, 
		d_lab->d_vVelNonLinSrcPBLMLabel, matlIndex, patch); */;
    // allocateAndPut instead:
    /* new_dw->put(pressureVars.vVelLinearSrc, 
		d_lab->d_vVelLinSrcPBLMLabel, matlIndex, patch); */;
    // allocateAndPut instead:
    /* new_dw->put(pressureVars.wVelNonlinearSrc, 
		d_lab->d_wVelNonLinSrcPBLMLabel, matlIndex, patch); */;
    // allocateAndPut instead:
    /* new_dw->put(pressureVars.wVelLinearSrc, 
		d_lab->d_wVelLinSrcPBLMLabel, matlIndex, patch); */;

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
				       const MaterialSubset* /* matls */,
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
    int nofStencils = 7;
    // Get the reference density
    // Get the required data
    new_dw->getCopy(pressureVars.density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->getCopy(pressureVars.viscosity, d_lab->d_viscosityINLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->getCopy(pressureVars.pressure, d_lab->d_pressurePSLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
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
		matlIndex, patch, Ghost::AroundFaces, Arches::TWOGHOSTCELLS);
    new_dw->getCopy(pressureVars.vVelocity, d_lab->d_vVelocityOUTBCLabel, 
		matlIndex, patch, Ghost::AroundFaces, Arches::TWOGHOSTCELLS);
    new_dw->getCopy(pressureVars.wVelocity, d_lab->d_wVelocityOUTBCLabel, 
		matlIndex, patch, Ghost::AroundFaces, Arches::TWOGHOSTCELLS);
    // *** warning fix it
    old_dw->getCopy(pressureVars.old_density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->getCopy(pressureVars.cellType, d_lab->d_cellTypeLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
  
    for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++) {
      new_dw->getCopy(pressureVars.uVelocityCoeff[ii], 
		  d_lab->d_uVelCoefPBLMLabel, ii, patch,
		  Ghost::AroundFaces, Arches::ONEGHOSTCELL);
      new_dw->getCopy(pressureVars.vVelocityCoeff[ii], 
		  d_lab->d_vVelCoefPBLMLabel, ii, patch,
		  Ghost::AroundFaces, Arches::ONEGHOSTCELL);
      new_dw->getCopy(pressureVars.wVelocityCoeff[ii], 
		  d_lab->d_wVelCoefPBLMLabel, ii, patch,
		  Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    }
    new_dw->getCopy(pressureVars.uVelNonlinearSrc, 
		d_lab->d_uVelNonLinSrcPBLMLabel,
		matlIndex, patch,
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->getCopy(pressureVars.vVelNonlinearSrc, 
		d_lab->d_vVelNonLinSrcPBLMLabel, matlIndex, patch,
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->getCopy(pressureVars.wVelNonlinearSrc,
		d_lab->d_wVelNonLinSrcPBLMLabel, matlIndex, patch,
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    
    // Calculate Pressure Coeffs
    //  inputs : densityIN, pressureIN, [u,v,w]VelCoefPBLM[Arches::AP]
    //  outputs: presCoefPBLM[Arches::AE..AB] 
    for (int ii = 0; ii < nofStencils; ii++)
      new_dw->allocateAndPut(pressureVars.pressCoeff[ii], d_lab->d_presCoefPBLMLabel, ii, patch);
    d_discretize->calculatePressureCoeff(pc, patch, old_dw, new_dw, 
					 delta_t, cellinfo, &pressureVars);

    // Modify pressure coefficients for multimaterial formulation

    if (d_MAlab) {

      new_dw->getCopy(pressureVars.voidFraction,
		  d_lab->d_mmgasVolFracLabel, matlIndex, patch,
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);

      d_discretize->mmModifyPressureCoeffs(pc, patch, &pressureVars);

    }

    // Calculate Pressure Source
    //  inputs : pressureSPBC, [u,v,w]VelocitySIVBC, densityCP,
    //           [u,v,w]VelCoefPBLM, [u,v,w]VelNonLinSrcPBLM
    //  outputs: presLinSrcPBLM, presNonLinSrcPBLM
    // Allocate space

    new_dw->allocateTemporary(pressureVars.pressLinearSrc,  patch);
    new_dw->allocateAndPut(pressureVars.pressNonlinearSrc, d_lab->d_presNonLinSrcPBLMLabel, matlIndex, patch);

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
      // allocateAndPut instead:
      /* new_dw->put(pressureVars.pressCoeff[ii], d_lab->d_presCoefPBLMLabel, 
		  ii, patch); */;
    }
    // allocateAndPut instead:
    /* new_dw->put(pressureVars.pressNonlinearSrc, 
		d_lab->d_presNonLinSrcPBLMLabel, matlIndex, patch); */;

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
  // Get the required data
  {
  new_dw->allocateTemporary(pressureVars.pressure,  patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
  new_dw->copyOut(pressureVars.pressure, d_lab->d_pressurePSLabel, 
	      matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
  }
  for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++) 
    new_dw->getCopy(pressureVars.pressCoeff[ii], d_lab->d_presCoefPBLMLabel, 
		   ii, patch, Ghost::None, Arches::ZEROGHOSTCELLS);

  {
  new_dw->allocateTemporary(pressureVars.pressNonlinearSrc,  patch, Ghost::None, Arches::ZEROGHOSTCELLS);
  new_dw->copyOut(pressureVars.pressNonlinearSrc, 
		 d_lab->d_presNonLinSrcPBLMLabel, 
		 matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
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

  
  tsk->requires(Task::NewDW, d_lab->d_pressurePSLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::OldDW, d_lab->d_densityMicroLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_mmcellTypeLabel,
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
    new_dw->get(cellType, d_lab->d_mmcellTypeLabel,
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);

    new_dw->allocateAndPut(pPlusHydro, d_lab->d_pressPlusHydroLabel,
		     matlIndex, patch);

    IntVector valid_lo = patch->getCellFORTLowIndex();
    IntVector valid_hi = patch->getCellFORTHighIndex();

    int mmwallid = d_boundaryCondition->getMMWallId();

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


    tsk->requires(Task::OldDW, d_lab->d_sharedState->get_delt_label());
    
  // Requires
  // from old_dw for time integration
  // get old_dw from getTop function
    tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel,
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);
    //    tsk->requires(Task::NewDW, d_lab->d_densityINLabel, 
    //		  Ghost::None, Arches::ZEROGHOSTCELLS);
  // from new_dw
  // for new task graph to work

    tsk->requires(Task::NewDW, d_lab->d_pressurePSLabel, 
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);
    tsk->requires(Task::NewDW, d_lab->d_densityINLabel,
		  Ghost::AroundCells, Arches::TWOGHOSTCELLS);

    if (dynamic_cast<const ScaleSimilarityModel*>(d_turbModel)) 
      tsk->requires(Task::OldDW, d_lab->d_stressTensorCompLabel, d_lab->d_stressTensorMatl,
		    Task::OutOfDomain, Ghost::AroundCells, Arches::ONEGHOSTCELL);

#ifdef correctorstep
    tsk->requires(Task::NewDW, d_lab->d_densityPredLabel, 
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);
    tsk->requires(Task::NewDW, d_lab->d_denRefArrayPredLabel,
    		  Ghost::AroundCells, Arches::ONEGHOSTCELL);
#else
    tsk->requires(Task::NewDW, d_lab->d_densityCPLabel, 
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);
    tsk->requires(Task::NewDW, d_lab->d_denRefArrayLabel,
    		  Ghost::AroundCells, Arches::ONEGHOSTCELL);
#endif
    tsk->requires(Task::NewDW, d_lab->d_viscosityINLabel,
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);
    tsk->requires(Task::NewDW, d_lab->d_uVelocityOUTBCLabel,
		  Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    tsk->requires(Task::NewDW, d_lab->d_vVelocityOUTBCLabel,
		  Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    tsk->requires(Task::NewDW, d_lab->d_wVelocityOUTBCLabel,
		  Ghost::AroundFaces, Arches::ONEGHOSTCELL);
#if 0
    tsk->requires(Task::OldDW, d_lab->d_uVelocitySPBCLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
    tsk->requires(Task::OldDW, d_lab->d_vVelocitySPBCLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
    tsk->requires(Task::OldDW, d_lab->d_wVelocitySPBCLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
#endif
    // required for computing div constraint
#ifdef divergenceconstraint
#ifdef correctorstep
    tsk->requires(Task::NewDW, d_lab->d_scalarPredLabel, 
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);
    tsk->requires(Task::NewDW, d_lab->d_drhodfPredLabel, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);
#else
    tsk->requires(Task::NewDW, d_lab->d_scalarSPLabel, 
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);
    tsk->requires(Task::NewDW, d_lab->d_drhodfCPLabel, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);
#endif
    tsk->requires(Task::NewDW, d_lab->d_scalDiffCoefPredLabel, 
		  d_lab->d_stencilMatl, Task::OutOfDomain,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
    tsk->requires(Task::NewDW, d_lab->d_scalDiffCoefSrcPredLabel, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);
#endif

    // for multi-material
    // requires su_drag[x,y,z], sp_drag[x,y,z] for arches

    if (d_MAlab) {

      tsk->requires(Task::NewDW, d_MAlab->d_uVel_mmLinSrcLabel,
		    Ghost::None, Arches::ZEROGHOSTCELLS);
      tsk->requires(Task::NewDW, d_MAlab->d_uVel_mmNonlinSrcLabel,
		    Ghost::None, Arches::ZEROGHOSTCELLS);
      tsk->requires(Task::NewDW, d_MAlab->d_vVel_mmLinSrcLabel, 
		    Ghost::None, Arches::ZEROGHOSTCELLS);
      tsk->requires(Task::NewDW, d_MAlab->d_vVel_mmNonlinSrcLabel,
		    Ghost::None, Arches::ZEROGHOSTCELLS);
      tsk->requires(Task::NewDW, d_MAlab->d_wVel_mmLinSrcLabel,
		    Ghost::None, Arches::ZEROGHOSTCELLS);
      tsk->requires(Task::NewDW, d_MAlab->d_wVel_mmNonlinSrcLabel,
		    Ghost::None, Arches::ZEROGHOSTCELLS);

    }
    
    
    // requires convection coeff because of the nodal
    // differencing
    // computes all the components of velocity

    tsk->computes(d_lab->d_uVelRhoHatLabel);
    tsk->computes(d_lab->d_vVelRhoHatLabel);
    tsk->computes(d_lab->d_wVelRhoHatLabel);
#ifdef Runge_Kutta_3d
#ifndef Runge_Kutta_3d_ssp
    tsk->computes(d_lab->d_uVelTempLabel);
    tsk->computes(d_lab->d_vVelTempLabel);
    tsk->computes(d_lab->d_wVelTempLabel);
#endif
#endif
//  tsk->computes(d_lab->d_velocityDivergenceLabel);
//  tsk->computes(d_lab->d_velocityDivergenceBCLabel);
    
#ifdef divergenceconstraint
    tsk->computes(d_lab->d_divConstraintLabel);
#endif
    sched->addTask(tsk, patches, matls);
  }

  // Now build pressure equation coefficients from momentum equation 
  // coefficients

  {
    Task* tsk = scinew Task( "Psolve::BuildCoeffPPred",
			     this,
			     &PressureSolver::buildLinearMatrixPressPred);
    

    tsk->requires(Task::OldDW, d_lab->d_sharedState->get_delt_label());

    // int matlIndex = 0;
    // Requires
    // from old_dw for time integration
    // get old_dw from getTop function

    tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel, 
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);

    // fix it
    // ***warning*June 20th
    tsk->requires(Task::OldDW, d_lab->d_densityINLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
    tsk->requires(Task::NewDW, d_lab->d_densityINLabel,
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);

    // from new_dw

    tsk->requires(Task::NewDW, d_lab->d_pressurePSLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);	
#ifdef correctorstep
    tsk->requires(Task::NewDW, d_lab->d_densityPredLabel, 
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);
#else
    tsk->requires(Task::NewDW, d_lab->d_densityCPLabel, 
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);
#endif
    tsk->requires(Task::NewDW, d_lab->d_viscosityINLabel,
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);
    tsk->requires(Task::NewDW, d_lab->d_uVelocityOUTBCLabel,
		  Ghost::AroundFaces, Arches::TWOGHOSTCELLS);
    tsk->requires(Task::NewDW, d_lab->d_vVelocityOUTBCLabel,
		  Ghost::AroundFaces, Arches::TWOGHOSTCELLS);
    tsk->requires(Task::NewDW, d_lab->d_wVelocityOUTBCLabel,
		  Ghost::AroundFaces, Arches::TWOGHOSTCELLS);
    
    /// requires convection coeff because of the nodal
    // differencing
    // computes all the components of velocity

    tsk->computes(d_lab->d_presCoefPBLMLabel, d_lab->d_stencilMatl,
		   Task::OutOfDomain);

    tsk->requires(Task::NewDW, d_lab->d_uVelRhoHatLabel,
		  Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    tsk->requires(Task::NewDW, d_lab->d_vVelRhoHatLabel,
		  Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    tsk->requires(Task::NewDW, d_lab->d_wVelRhoHatLabel,
		  Ghost::AroundFaces, Arches::ONEGHOSTCELL);
#ifdef divergenceconstraint
    tsk->requires(Task::NewDW, d_lab->d_divConstraintLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
#endif
    tsk->computes(d_lab->d_presNonLinSrcPBLMLabel);

    if (d_MAlab) {
      tsk->requires(Task::NewDW, d_lab->d_mmgasVolFracLabel, 
		    Ghost::AroundCells, Arches::ONEGHOSTCELL);
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
#ifndef Runge_Kutta_2nd
#ifndef Runge_Kutta_3d
#ifndef Runge_Kutta_3d_ssp
  delta_t /= 2.0;
#endif
#endif
#endif
#endif

#ifdef Runge_Kutta_3d
#ifndef Runge_Kutta_3d_ssp
  double gamma_1 = 8.0/15.0;
  delta_t *= gamma_1; 
#endif
#endif
  
  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    ArchesVariables pressureVars;

    // compute all three componenets of velocity stencil coefficients


    // Get the reference density
    // Get the required data

    new_dw->getCopy(pressureVars.density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::TWOGHOSTCELLS);
#ifdef correctorstep
    new_dw->getCopy(pressureVars.new_density, d_lab->d_densityPredLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->getCopy(pressureVars.denRefArray, d_lab->d_denRefArrayPredLabel,
    		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
#else
    new_dw->getCopy(pressureVars.new_density, d_lab->d_densityCPLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->getCopy(pressureVars.denRefArray, d_lab->d_denRefArrayLabel,
    		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
#endif
    new_dw->getCopy(pressureVars.viscosity, d_lab->d_viscosityINLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->getCopy(pressureVars.pressure, d_lab->d_pressurePSLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);

#ifdef divergenceconstraint
#ifdef correctorstep
    new_dw->getCopy(pressureVars.scalar, d_lab->d_scalarPredLabel,
		    matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->getCopy(pressureVars.drhodf, d_lab->d_drhodfPredLabel,
		    matlIndex, patch);
#else
    new_dw->getCopy(pressureVars.scalar, d_lab->d_scalarSPLabel,
		    matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->getCopy(pressureVars.drhodf, d_lab->d_drhodfCPLabel,
		    matlIndex, patch);
#endif
    for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++)
      new_dw->getCopy(pressureVars.scalarDiffusionCoeff[ii], 
		      d_lab->d_scalDiffCoefPredLabel, 
		      ii, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->getCopy(pressureVars.scalarDiffNonlinearSrc, d_lab->d_scalDiffCoefSrcPredLabel,
		    matlIndex, patch);
    new_dw->allocateAndPut(pressureVars.divergence, d_lab->d_divConstraintLabel,
		     matlIndex, patch);
    pressureVars.divergence.initialize(0.0);
#endif

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
		matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->getCopy(pressureVars.vVelocity, d_lab->d_vVelocityOUTBCLabel, 
		matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->getCopy(pressureVars.wVelocity, d_lab->d_wVelocityOUTBCLabel, 
		matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
#if 0
    old_dw->getCopy(pressureVars.old_uVelocity, d_lab->d_uVelocitySPBCLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    old_dw->getCopy(pressureVars.old_vVelocity, d_lab->d_vVelocitySPBCLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    old_dw->getCopy(pressureVars.old_wVelocity, d_lab->d_wVelocitySPBCLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
#else
    new_dw->getCopy(pressureVars.old_uVelocity, d_lab->d_uVelocityOUTBCLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->getCopy(pressureVars.old_vVelocity, d_lab->d_vVelocityOUTBCLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->getCopy(pressureVars.old_wVelocity, d_lab->d_wVelocityOUTBCLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
#endif
    new_dw->getCopy(pressureVars.old_density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->getCopy(pressureVars.cellType, d_lab->d_cellTypeLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    
    for(int index = 1; index <= Arches::NDIM; ++index) {

    // get multimaterial momentum source terms

      if (d_MAlab) {
	switch (index) {
	
	case Arches::XDIR:

	  new_dw->getCopy(pressureVars.mmuVelSu, d_MAlab->d_uVel_mmNonlinSrcLabel,
		      matlIndex, patch,
		      Ghost::None, Arches::ZEROGHOSTCELLS);
	  new_dw->getCopy(pressureVars.mmuVelSp, d_MAlab->d_uVel_mmLinSrcLabel,
		      matlIndex, patch,
		      Ghost::None, Arches::ZEROGHOSTCELLS);
	  break;

	case Arches::YDIR:

	  new_dw->getCopy(pressureVars.mmvVelSu, d_MAlab->d_vVel_mmNonlinSrcLabel,
		      matlIndex, patch,
		      Ghost::None, Arches::ZEROGHOSTCELLS);
	  new_dw->getCopy(pressureVars.mmvVelSp, d_MAlab->d_vVel_mmLinSrcLabel,
		      matlIndex, patch,
		      Ghost::None, Arches::ZEROGHOSTCELLS);
	  break;
	case Arches::ZDIR:

	  new_dw->getCopy(pressureVars.mmwVelSu, d_MAlab->d_wVel_mmNonlinSrcLabel,
		      matlIndex, patch,
		      Ghost::None, Arches::ZEROGHOSTCELLS);
	  new_dw->getCopy(pressureVars.mmwVelSp, d_MAlab->d_wVel_mmLinSrcLabel,
		      matlIndex, patch,
		      Ghost::None, Arches::ZEROGHOSTCELLS);
	  break;
	}
      }
      
      for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++) {

	switch(index) {

	case Arches::XDIR:

	  new_dw->allocateTemporary(pressureVars.uVelocityCoeff[ii],  patch);
	  new_dw->allocateTemporary(pressureVars.uVelocityConvectCoeff[ii],  patch);
	  break;
	case Arches::YDIR:
	  new_dw->allocateTemporary(pressureVars.vVelocityCoeff[ii],  patch);
	  new_dw->allocateTemporary(pressureVars.vVelocityConvectCoeff[ii],  patch);
	  break;
	case Arches::ZDIR:
	  new_dw->allocateTemporary(pressureVars.wVelocityCoeff[ii],  patch);
	  new_dw->allocateTemporary(pressureVars.wVelocityConvectCoeff[ii],  patch);
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

	new_dw->allocateTemporary(pressureVars.uVelLinearSrc,  patch);
	new_dw->allocateTemporary(pressureVars.uVelNonlinearSrc,  patch);
	new_dw->allocateAndPut(pressureVars.uVelRhoHat, d_lab->d_uVelRhoHatLabel,
			 matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
	pressureVars.uVelRhoHat.copy(pressureVars.uVelocity,
				     pressureVars.uVelRhoHat.getLowIndex(),
				     pressureVars.uVelRhoHat.getHighIndex());

	break;

      case Arches::YDIR:

	new_dw->allocateTemporary(pressureVars.vVelLinearSrc,  patch);
	new_dw->allocateTemporary(pressureVars.vVelNonlinearSrc,  patch);
	new_dw->allocateAndPut(pressureVars.vVelRhoHat, d_lab->d_vVelRhoHatLabel,
			 matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
	pressureVars.vVelRhoHat.copy(pressureVars.vVelocity,
				     pressureVars.vVelRhoHat.getLowIndex(),
				     pressureVars.vVelRhoHat.getHighIndex());

	break;

      case Arches::ZDIR:

	new_dw->allocateTemporary(pressureVars.wVelLinearSrc,  patch);
	new_dw->allocateTemporary(pressureVars.wVelNonlinearSrc,  patch);
	new_dw->allocateAndPut(pressureVars.wVelRhoHat, d_lab->d_wVelRhoHatLabel,
			 matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
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
      // for scalesimilarity model add stress tensor to the source of velocity eqn.
      if (dynamic_cast<const ScaleSimilarityModel*>(d_turbModel)) {
	StencilMatrix<CCVariable<double> > stressTensor; //9 point tensor
	for (int ii = 0; ii < d_lab->d_stressTensorMatl->size(); ii++) {
	  old_dw->getCopy(stressTensor[ii], 
			  d_lab->d_stressTensorCompLabel, ii, patch,
			  Ghost::AroundCells, Arches::ONEGHOSTCELL);
	}

	IntVector indexLow = patch->getCellFORTLowIndex();
	IntVector indexHigh = patch->getCellFORTHighIndex();
	
	// set density for the whole domain


	      // Store current cell
	double sue, suw, sun, sus, sut, sub;
	switch (index) {
	case Arches::XDIR:
	  for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
	    for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
	      for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {
		IntVector currCell(colX, colY, colZ);
		IntVector prevXCell(colX-1, colY, colZ);
		IntVector prevYCell(colX, colY-1, colZ);
		IntVector prevZCell(colX, colY, colZ-1);

		sue = cellinfo->sns[colY]*cellinfo->stb[colZ]*
		             (stressTensor[0])[currCell];
		suw = cellinfo->sns[colY]*cellinfo->stb[colZ]*
		             (stressTensor[0])[prevXCell];
		sun = 0.25*cellinfo->sew[colX]*cellinfo->stb[colZ]*
		             ((stressTensor[1])[currCell]+
			      (stressTensor[1])[prevXCell]+
			      (stressTensor[1])[IntVector(colX,colY+1,colZ)]+
			      (stressTensor[1])[IntVector(colX-1,colY+1,colZ)]);
		sus =  0.25*cellinfo->sew[colX]*cellinfo->stb[colZ]*
		             ((stressTensor[1])[currCell]+
			      (stressTensor[1])[prevXCell]+
			      (stressTensor[1])[IntVector(colX,colY-1,colZ)]+
			      (stressTensor[1])[IntVector(colX-1,colY-1,colZ)]);
		sut = 0.25*cellinfo->sns[colY]*cellinfo->sew[colX]*
		             ((stressTensor[2])[currCell]+
			      (stressTensor[2])[prevXCell]+
			      (stressTensor[2])[IntVector(colX,colY,colZ+1)]+
			      (stressTensor[2])[IntVector(colX-1,colY,colZ+1)]);
		sub =  0.25*cellinfo->sns[colY]*cellinfo->sew[colX]*
		             ((stressTensor[2])[currCell]+
			      (stressTensor[2])[prevXCell]+
			      (stressTensor[2])[IntVector(colX,colY,colZ-1)]+
			      (stressTensor[2])[IntVector(colX-1,colY,colZ-1)]);
		pressureVars.uVelNonlinearSrc[currCell] += suw-sue+sus-sun+sub-sut;
	      }
	    }
	  }
	  break;
	case Arches::YDIR:
	  for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
	    for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
	      for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {
		IntVector currCell(colX, colY, colZ);
		IntVector prevXCell(colX-1, colY, colZ);
		IntVector prevYCell(colX, colY-1, colZ);
		IntVector prevZCell(colX, colY, colZ-1);

		sue = 0.25*cellinfo->sns[colY]*cellinfo->stb[colZ]*
		  ((stressTensor[3])[currCell]+
		   (stressTensor[3])[prevYCell]+
		   (stressTensor[3])[IntVector(colX+1,colY,colZ)]+
		   (stressTensor[3])[IntVector(colX+1,colY-1,colZ)]);
		suw =  0.25*cellinfo->sns[colY]*cellinfo->stb[colZ]*
		  ((stressTensor[3])[currCell]+
		   (stressTensor[3])[prevYCell]+
		   (stressTensor[3])[IntVector(colX-1,colY,colZ)]+
		   (stressTensor[3])[IntVector(colX-1,colY-1,colZ)]);
		sun = cellinfo->sew[colX]*cellinfo->stb[colZ]*
		  (stressTensor[4])[currCell];
		sus = cellinfo->sew[colX]*cellinfo->stb[colZ]*
		  (stressTensor[4])[prevYCell];
		sut = 0.25*cellinfo->sns[colY]*cellinfo->sew[colX]*
		  ((stressTensor[5])[currCell]+
		   (stressTensor[5])[prevYCell]+
		   (stressTensor[5])[IntVector(colX,colY,colZ+1)]+
		   (stressTensor[5])[IntVector(colX,colY-1,colZ+1)]);
		sub =  0.25*cellinfo->sns[colY]*cellinfo->sew[colX]*
		  ((stressTensor[5])[currCell]+
		   (stressTensor[5])[prevYCell]+
		   (stressTensor[5])[IntVector(colX,colY,colZ-1)]+
		   (stressTensor[5])[IntVector(colX,colY-1,colZ-1)]);
		pressureVars.vVelNonlinearSrc[currCell] += suw-sue+sus-sun+sub-sut;
	      }
	    }
	  }
	  break;
	case Arches::ZDIR:
	  for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
	    for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
	      for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {
		IntVector currCell(colX, colY, colZ);
		IntVector prevXCell(colX-1, colY, colZ);
		IntVector prevYCell(colX, colY-1, colZ);
		IntVector prevZCell(colX, colY, colZ-1);

		sue = 0.25*cellinfo->sns[colY]*cellinfo->stb[colZ]*
		             ((stressTensor[6])[currCell]+
			      (stressTensor[6])[prevZCell]+
			      (stressTensor[6])[IntVector(colX+1,colY,colZ)]+
			      (stressTensor[6])[IntVector(colX+1,colY,colZ-1)]);
		suw =  0.25*cellinfo->sns[colY]*cellinfo->stb[colZ]*
		             ((stressTensor[6])[currCell]+
			      (stressTensor[6])[prevZCell]+
			      (stressTensor[6])[IntVector(colX-1,colY,colZ)]+
			      (stressTensor[6])[IntVector(colX-1,colY,colZ-1)]);
		sun = 0.25*cellinfo->sew[colX]*cellinfo->stb[colZ]*
		             ((stressTensor[7])[currCell]+
			      (stressTensor[7])[prevZCell]+
			      (stressTensor[7])[IntVector(colX,colY+1,colZ)]+
			      (stressTensor[7])[IntVector(colX,colY+1,colZ-1)]);
		sus =  0.25*cellinfo->sew[colX]*cellinfo->stb[colZ]*
		             ((stressTensor[7])[currCell]+
			      (stressTensor[7])[prevZCell]+
			      (stressTensor[7])[IntVector(colX,colY-1,colZ)]+
			      (stressTensor[7])[IntVector(colX,colY-1,colZ-1)]);
		sut = cellinfo->sew[colX]*cellinfo->sns[colY]*
		             (stressTensor[8])[currCell];
		sub = cellinfo->sew[colX]*cellinfo->sns[colY]*
		             (stressTensor[8])[prevZCell];
		pressureVars.wVelNonlinearSrc[currCell] += suw-sue+sus-sun+sub-sut;
	      }
	    }
	  }
	  break;
	default:
	  throw InvalidValue("Invalid index in PressureSolver::BuildVelCoeff");
	}
      }
	
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
#if 0
      d_discretize->calculateVelDiagonal(pc, patch,
					 index,
					 &pressureVars);
      d_source->modifyVelMassSource(pc, patch, delta_t, index,
				    &pressureVars);
#endif
      d_source->modifyVelMassSource(pc, patch, delta_t, index,
				    &pressureVars);
      d_discretize->calculateVelDiagonal(pc, patch,
					 index,
					 &pressureVars);


      // Calculate Velocity diagonal
      //  inputs : [u,v,w]VelCoefPBLM, [u,v,w]VelLinSrcPBLM
      //  outputs: [u,v,w]VelCoefPBLM

      d_discretize->calculateVelRhoHat(pc, patch, index, delta_t,
				     cellinfo, &pressureVars);

#ifdef Runge_Kutta_3d 
#ifndef Runge_Kutta_3d_ssp
    SFCXVariable<double> temp_uVel;
    SFCYVariable<double> temp_vVel;
    SFCZVariable<double> temp_wVel;
    constCCVariable<double> old_density;
    constCCVariable<double> new_density;
    IntVector indexLow;
    IntVector indexHigh;

    new_dw->get(old_density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->get(new_density, d_lab->d_densityPredLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);

    switch(index) {

    case Arches::XDIR:

      new_dw->allocateAndPut(temp_uVel, d_lab->d_uVelTempLabel, matlIndex, patch);
      temp_uVel.initialize(0.0);
    
      indexLow = patch->getSFCXFORTLowIndex();
      indexHigh = patch->getSFCXFORTHighIndex();
    
      for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
        for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
          for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {

              IntVector currCell(colX, colY, colZ);
              IntVector xshiftedCell(colX-1, colY, colZ);

              temp_uVel[currCell] = 0.5*(
              (new_density[currCell]+new_density[xshiftedCell])*
	      pressureVars.uVelRhoHat[currCell]-
              (old_density[currCell]+old_density[xshiftedCell])*
              pressureVars.uVelocity[currCell])/gamma_1;
          }
        }
      }
      // allocateAndPut instead:
      /* new_dw->put(temp_uVel, d_lab->d_uVelTempLabel, matlIndex, patch); */;

    break;

    case Arches::YDIR:

      new_dw->allocateAndPut(temp_vVel, d_lab->d_vVelTempLabel, matlIndex, patch);
      temp_vVel.initialize(0.0);
    
      indexLow = patch->getSFCYFORTLowIndex();
      indexHigh = patch->getSFCYFORTHighIndex();
    
      for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
        for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
          for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {

              IntVector currCell(colX, colY, colZ);
              IntVector yshiftedCell(colX, colY-1, colZ);

              temp_vVel[currCell] = 0.5*(
              (new_density[currCell]+new_density[yshiftedCell])*
	      pressureVars.vVelRhoHat[currCell]-
              (old_density[currCell]+old_density[yshiftedCell])*
              pressureVars.vVelocity[currCell])/gamma_1;
          }
        }
      }
      // allocateAndPut instead:
      /* new_dw->put(temp_vVel, d_lab->d_vVelTempLabel, matlIndex, patch); */;
    
    break;

    case Arches::ZDIR:

      new_dw->allocateAndPut(temp_wVel, d_lab->d_wVelTempLabel, matlIndex, patch);
      temp_wVel.initialize(0.0);

      indexLow = patch->getSFCZFORTLowIndex();
      indexHigh = patch->getSFCZFORTHighIndex();
    
      for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
        for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
          for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {

              IntVector currCell(colX, colY, colZ);
              IntVector zshiftedCell(colX, colY, colZ-1);

              temp_wVel[currCell] = 0.5*(
              (new_density[currCell]+new_density[zshiftedCell])*
	      pressureVars.wVelRhoHat[currCell]-
              (old_density[currCell]+old_density[zshiftedCell])*
              pressureVars.wVelocity[currCell])/gamma_1;
          }
        }
      }
      // allocateAndPut instead:
      /* new_dw->put(temp_wVel, d_lab->d_wVelTempLabel, matlIndex, patch); */;

    break;

    default:
      throw InvalidValue("Invalid index in Pred PressureSolver for RK3");
    }
#endif
#endif

#ifdef ARCHES_PRES_DEBUG
    std::cerr << "Done building matrix for press coeff" << endl;
#endif
    
    }
    d_boundaryCondition->newrecomputePressureBC(pc, patch, cellinfo,
						&pressureVars);

#ifdef divergenceconstraint    
    // compute divergence constraint to use in pressure equation
    d_discretize->computeDivergence(pc, patch, &pressureVars);
    // allocateAndPut instead:
    /* new_dw->put(pressureVars.divergence, d_lab->d_divConstraintLabel,
		matlIndex, patch); */;
#endif
  // put required vars

    // allocateAndPut instead:
    /* new_dw->put(pressureVars.uVelRhoHat, d_lab->d_uVelRhoHatLabel, 
		matlIndex, patch); */;
    // allocateAndPut instead:
    /* new_dw->put(pressureVars.vVelRhoHat, d_lab->d_vVelRhoHatLabel, 
		matlIndex, patch); */;
    // allocateAndPut instead:
    /* new_dw->put(pressureVars.wVelRhoHat, d_lab->d_wVelRhoHatLabel, 
		matlIndex, patch); */;

/*  CCVariable<double> divergence;
  divergence.allocate(pressureVars.old_density.getLowIndex(),
                                   pressureVars.old_density.getHighIndex());
  divergence.initialize(0.0);
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

  delta_t = 1.0;
  fort_pressrcpred(idxLo, idxHi, divergence, pressureVars.old_density,
                   pressureVars.old_density, pressureVars.old_density, 
                   pressureVars.uVelocity, pressureVars.vVelocity, 
                   pressureVars.wVelocity, delta_t,
                   cellinfo->sew, cellinfo->sns, cellinfo->stb);

  new_dw->put(divergence, d_lab->d_velocityDivergenceBCLabel, matlIndex, patch);
  
  divergence.initialize(0.0);
  pressureVars.uVelocity.initialize(0.0);
  pressureVars.vVelocity.initialize(0.0);
  pressureVars.wVelocity.initialize(0.0);
  old_dw->copyOut(pressureVars.uVelocity, d_lab->d_uVelocitySPBCLabel, 
		matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
  old_dw->copyOut(pressureVars.vVelocity, d_lab->d_vVelocitySPBCLabel, 
		matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
  old_dw->copyOut(pressureVars.wVelocity, d_lab->d_wVelocitySPBCLabel, 
		matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);

  delta_t = 1.0;
  fort_pressrcpred(idxLo, idxHi, divergence, pressureVars.old_density,
                   pressureVars.old_density, pressureVars.old_density, 
                   pressureVars.uVelocity, pressureVars.vVelocity, 
                   pressureVars.wVelocity, delta_t,
                   cellinfo->sew, cellinfo->sns, cellinfo->stb);

  new_dw->put(divergence, d_lab->d_velocityDivergenceLabel, matlIndex, patch);
*/

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
					   const MaterialSubset* /* matls */,
					   DataWarehouse* old_dw,
					   DataWarehouse* new_dw)
{
  delt_vartype delT;
  old_dw->get(delT, d_lab->d_sharedState->get_delt_label() );
  double delta_t = delT;

#ifdef correctorstep
#ifndef Runge_Kutta_2nd
#ifndef Runge_Kutta_3d
#ifndef Runge_Kutta_3d_ssp
  delta_t /= 2.0;
#endif
#endif
#endif
#endif

#ifdef Runge_Kutta_3d
#ifndef Runge_Kutta_3d_ssp
  double gamma_1 = 8.0/15.0;
  delta_t *= gamma_1; 
#endif
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
    int nofStencils = 7;
    // Get the reference density
    // Get the required data
#ifdef correctorstep
    new_dw->getCopy(pressureVars.density, d_lab->d_densityPredLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
#else
    new_dw->getCopy(pressureVars.density, d_lab->d_densityCPLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
#endif
    new_dw->getCopy(pressureVars.viscosity, d_lab->d_viscosityINLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->getCopy(pressureVars.pressure, d_lab->d_pressurePSLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
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
		matlIndex, patch, Ghost::AroundFaces, Arches::TWOGHOSTCELLS);
    new_dw->getCopy(pressureVars.vVelocity, d_lab->d_vVelocityOUTBCLabel, 
		matlIndex, patch, Ghost::AroundFaces, Arches::TWOGHOSTCELLS);
    new_dw->getCopy(pressureVars.wVelocity, d_lab->d_wVelocityOUTBCLabel, 
		matlIndex, patch, Ghost::AroundFaces, Arches::TWOGHOSTCELLS);
    new_dw->getCopy(pressureVars.uVelRhoHat, d_lab->d_uVelRhoHatLabel, 
		matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->getCopy(pressureVars.vVelRhoHat, d_lab->d_vVelRhoHatLabel, 
		matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->getCopy(pressureVars.wVelRhoHat, d_lab->d_wVelRhoHatLabel, 
		matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    //**warning
#ifdef divergenceconstraint
    new_dw->getCopy(pressureVars.divergence, d_lab->d_divConstraintLabel,
		    matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
#endif
    new_dw->getCopy(pressureVars.old_density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    // 2nd order density time derivative needs to be worked out for 3d order R.-K. and for Rajesh's RK2
    old_dw->getCopy(pressureVars.old_old_density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->getCopy(pressureVars.cellType, d_lab->d_cellTypeLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
  
    // Calculate Pressure Coeffs
    //  inputs : densityIN, pressureIN, [u,v,w]VelCoefPBLM[Arches::AP]
    //  outputs: presCoefPBLM[Arches::AE..AB] 
    for (int ii = 0; ii < nofStencils; ii++)
      new_dw->allocateAndPut(pressureVars.pressCoeff[ii], d_lab->d_presCoefPBLMLabel, ii, patch);
    d_discretize->calculatePressureCoeff(pc, patch, old_dw, new_dw, 
					 delta_t, cellinfo, &pressureVars);

    // Modify pressure coefficients for multimaterial formulation

    if (d_MAlab) {

      new_dw->getCopy(pressureVars.voidFraction,
		  d_lab->d_mmgasVolFracLabel, matlIndex, patch,
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);

      d_discretize->mmModifyPressureCoeffs(pc, patch, &pressureVars);

    }

    // Calculate Pressure Source
    //  inputs : pressureSPBC, [u,v,w]VelocitySIVBC, densityCP,
    //           [u,v,w]VelCoefPBLM, [u,v,w]VelNonLinSrcPBLM
    //  outputs: presLinSrcPBLM, presNonLinSrcPBLM
    // Allocate space

    new_dw->allocateTemporary(pressureVars.pressLinearSrc,  patch);
    pressureVars.pressLinearSrc.initialize(0.0);
    new_dw->allocateAndPut(pressureVars.pressNonlinearSrc, d_lab->d_presNonLinSrcPBLMLabel, matlIndex, patch);
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
      // allocateAndPut instead:
      /* new_dw->put(pressureVars.pressCoeff[ii], d_lab->d_presCoefPBLMLabel, 
		  ii, patch); */;
    }
    // allocateAndPut instead:
    /* new_dw->put(pressureVars.pressNonlinearSrc, 
		d_lab->d_presNonLinSrcPBLMLabel, matlIndex, patch); */;

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

  // Requires
  // coefficient for the variable for which solve is invoked

  tsk->requires(Task::NewDW, d_lab->d_pressurePSLabel, 
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
  //  tsk->requires(Task::OldDW, d_lab->d_pressureCorrSPBCLabel, 
  //	Ghost::AroundCells, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_presCoefPBLMLabel, d_lab->d_stencilMatl,
		Task::OutOfDomain, Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_presNonLinSrcPBLMLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);

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
					 DataWarehouse* /*old_dw*/,
					 DataWarehouse* new_dw,
					 ArchesVariables& pressureVars)
{
  // Get the required data
  {
#ifdef correctorstep
  new_dw->allocateTemporary(pressureVars.pressure,  patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
#else
  new_dw->allocateTemporary(pressureVars.pressure,  patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
#endif
  new_dw->copyOut(pressureVars.pressure, d_lab->d_pressurePSLabel, 
	      matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
#if 0
  new_dw->allocate(pressureVars.pressure, d_lab->d_pressureCorrSPBCLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
  new_dw->copyOut(pressureVars.pressure, d_lab->d_pressurePSLabel, 
	      matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
#endif
  }
  for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++) 
    new_dw->getCopy(pressureVars.pressCoeff[ii], d_lab->d_presCoefPBLMLabel, 
		   ii, patch, Ghost::None, Arches::ZEROGHOSTCELLS);

  new_dw->getCopy(pressureVars.pressNonlinearSrc, 
		  d_lab->d_presNonLinSrcPBLMLabel, 
		  matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);

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


    tsk->requires(Task::OldDW, d_lab->d_sharedState->get_delt_label());
    
  // Requires
  // from old_dw for time integration
  // get old_dw from getTop function
    tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel,
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);
    //    tsk->requires(Task::NewDW, d_lab->d_densityINLabel, 
    //		  Ghost::None, Arches::ZEROGHOSTCELLS);

  #ifdef Runge_Kutta_3d
    tsk->requires(Task::NewDW, d_lab->d_densityIntermLabel,
		  Ghost::AroundCells, Arches::TWOGHOSTCELLS);

    tsk->requires(Task::NewDW, d_lab->d_pressureIntermLabel, 
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);
    tsk->requires(Task::NewDW, d_lab->d_viscosityIntermLabel,
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);
    tsk->requires(Task::NewDW, d_lab->d_uVelocityIntermLabel,
		  Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    tsk->requires(Task::NewDW, d_lab->d_vVelocityIntermLabel,
		  Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    tsk->requires(Task::NewDW, d_lab->d_wVelocityIntermLabel,
		  Ghost::AroundFaces, Arches::ONEGHOSTCELL);

#ifndef Runge_Kutta_3d_ssp
    tsk->requires(Task::NewDW, d_lab->d_uVelTempLabel,
                  Ghost::None, Arches::ZEROGHOSTCELLS);
    tsk->requires(Task::NewDW, d_lab->d_vVelTempLabel,
                  Ghost::None, Arches::ZEROGHOSTCELLS);
    tsk->requires(Task::NewDW, d_lab->d_wVelTempLabel,
                  Ghost::None, Arches::ZEROGHOSTCELLS);
#endif
   #else
  #ifndef Runge_Kutta_2nd
    tsk->requires(Task::NewDW, d_lab->d_densityINLabel,
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);
#if 1
    tsk->requires(Task::NewDW, d_lab->d_uVelocityOUTBCLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
    tsk->requires(Task::NewDW, d_lab->d_vVelocityOUTBCLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
    tsk->requires(Task::NewDW, d_lab->d_wVelocityOUTBCLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);

#else
    tsk->requires(Task::OldDW, d_lab->d_uVelocitySPBCLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
    tsk->requires(Task::OldDW, d_lab->d_vVelocitySPBCLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
    tsk->requires(Task::OldDW, d_lab->d_wVelocitySPBCLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
#endif
   #endif
  // from new_dw
  // for new task graph to work

    tsk->requires(Task::NewDW, d_lab->d_densityPredLabel,
		  Ghost::AroundCells, Arches::TWOGHOSTCELLS);
    tsk->requires(Task::NewDW, d_lab->d_pressurePredLabel, 
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);
    tsk->requires(Task::NewDW, d_lab->d_viscosityPredLabel,
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);
    tsk->requires(Task::NewDW, d_lab->d_uVelocityPredLabel,
		  Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    tsk->requires(Task::NewDW, d_lab->d_vVelocityPredLabel,
		  Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    tsk->requires(Task::NewDW, d_lab->d_wVelocityPredLabel,
		  Ghost::AroundFaces, Arches::ONEGHOSTCELL);
   #endif
    tsk->requires(Task::NewDW, d_lab->d_densityCPLabel, 
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);
    tsk->requires(Task::NewDW, d_lab->d_denRefArrayLabel,
    		  Ghost::AroundCells, Arches::ONEGHOSTCELL);
#ifdef divergenceconstraint
    tsk->requires(Task::NewDW, d_lab->d_scalarSPLabel, 
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);
    tsk->requires(Task::NewDW, d_lab->d_drhodfCPLabel, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);
    tsk->requires(Task::NewDW, d_lab->d_scalDiffCoefCorrLabel, 
		  d_lab->d_stencilMatl, Task::OutOfDomain,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
#endif

    // for multi-material
    // requires su_drag[x,y,z], sp_drag[x,y,z] for arches

    if (d_MAlab) {

      tsk->requires(Task::NewDW, d_MAlab->d_uVel_mmLinSrcLabel,
		    Ghost::None, Arches::ZEROGHOSTCELLS);
      tsk->requires(Task::NewDW, d_MAlab->d_uVel_mmNonlinSrcLabel,
		    Ghost::None, Arches::ZEROGHOSTCELLS);
      tsk->requires(Task::NewDW, d_MAlab->d_vVel_mmLinSrcLabel, 
		    Ghost::None, Arches::ZEROGHOSTCELLS);
      tsk->requires(Task::NewDW, d_MAlab->d_vVel_mmNonlinSrcLabel,
		    Ghost::None, Arches::ZEROGHOSTCELLS);
      tsk->requires(Task::NewDW, d_MAlab->d_wVel_mmLinSrcLabel,
		    Ghost::None, Arches::ZEROGHOSTCELLS);
      tsk->requires(Task::NewDW, d_MAlab->d_wVel_mmNonlinSrcLabel,
		    Ghost::None, Arches::ZEROGHOSTCELLS);

    }
    
    
    // requires convection coeff because of the nodal
    // differencing
    // computes all the components of velocity

    tsk->computes(d_lab->d_uVelRhoHatCorrLabel);
    tsk->computes(d_lab->d_vVelRhoHatCorrLabel);
    tsk->computes(d_lab->d_wVelRhoHatCorrLabel);
#ifdef divergenceconstraint
    tsk->modifies(d_lab->d_divConstraintLabel);
#endif
    
    sched->addTask(tsk, patches, matls);
  }
  // Now build pressure equation coefficients from momentum equation 
  // coefficients

  {
    Task* tsk = scinew Task( "Psolve::BuildCoeffPCorr",
			     this,
			     &PressureSolver::buildLinearMatrixPressCorr);
    

    tsk->requires(Task::OldDW, d_lab->d_sharedState->get_delt_label());
    
    // int matlIndex = 0;
    // Requires
    // from old_dw for time integration
    // get old_dw from getTop function

    tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel, 
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);

  #ifdef Runge_Kutta_3d
    tsk->requires(Task::NewDW, d_lab->d_densityIntermLabel,
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);
    tsk->requires(Task::OldDW, d_lab->d_densityINLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);

    tsk->requires(Task::NewDW, d_lab->d_pressureIntermLabel,
		  Ghost::AroundCells, Arches::ZEROGHOSTCELLS);
    tsk->requires(Task::NewDW, d_lab->d_viscosityIntermLabel,
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);
  #else
  #ifdef Runge_Kutta_2nd
    tsk->requires(Task::NewDW, d_lab->d_densityPredLabel,
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);
    // the following is for old_old_density (2nd order time differencing)
    tsk->requires(Task::NewDW, d_lab->d_densityINLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
  #else
    tsk->requires(Task::NewDW, d_lab->d_densityINLabel,
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);
    // the following is for old_old_density (2nd order time differencing)
    tsk->requires(Task::OldDW, d_lab->d_densityINLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
  #endif
    // from new_dw
    tsk->requires(Task::NewDW, d_lab->d_pressurePredLabel,
		  Ghost::AroundCells, Arches::ZEROGHOSTCELLS);
    tsk->requires(Task::NewDW, d_lab->d_viscosityPredLabel,
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);
  #endif
    tsk->requires(Task::NewDW, d_lab->d_densityCPLabel, 
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);
    tsk->requires(Task::NewDW, d_lab->d_uVelRhoHatCorrLabel,
		  Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    tsk->requires(Task::NewDW, d_lab->d_vVelRhoHatCorrLabel,
		  Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    tsk->requires(Task::NewDW, d_lab->d_wVelRhoHatCorrLabel,
		  Ghost::AroundFaces, Arches::ONEGHOSTCELL);
#ifdef divergenceconstraint
    tsk->requires(Task::NewDW, d_lab->d_divConstraintLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
#endif

    tsk->computes(d_lab->d_presCoefCorrLabel, d_lab->d_stencilMatl,
		   Task::OutOfDomain);
    tsk->computes(d_lab->d_presNonLinSrcCorrLabel);

    if (d_MAlab) {
      tsk->requires(Task::NewDW, d_lab->d_mmgasVolFracLabel, 
		    Ghost::AroundCells, Arches::ONEGHOSTCELL);
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

#ifdef Runge_Kutta_3d
#ifndef Runge_Kutta_3d_ssp
  double gamma_3 = 3.0/4.0;
  double zeta_2 = -5.0/12.0;
  delta_t *= gamma_3;
#endif
#endif
  
  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    ArchesVariables pressureVars;

    // compute all three componenets of velocity stencil coefficients


    // Get the reference density
    // Get the required data

    new_dw->getCopy(pressureVars.new_density, d_lab->d_densityCPLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
  #ifdef Runge_Kutta_3d
    new_dw->getCopy(pressureVars.density, d_lab->d_densityIntermLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::TWOGHOSTCELLS);
    new_dw->getCopy(pressureVars.pressure, d_lab->d_pressureIntermLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->getCopy(pressureVars.viscosity, d_lab->d_viscosityIntermLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
  #else
    new_dw->getCopy(pressureVars.density, d_lab->d_densityPredLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::TWOGHOSTCELLS);
    new_dw->getCopy(pressureVars.pressure, d_lab->d_pressurePredLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->getCopy(pressureVars.viscosity, d_lab->d_viscosityPredLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
  #endif
    new_dw->getCopy(pressureVars.denRefArray, d_lab->d_denRefArrayLabel,
    		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);

#ifdef divergenceconstraint
    new_dw->getCopy(pressureVars.scalar, d_lab->d_scalarSPLabel,
		    matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->getCopy(pressureVars.drhodf, d_lab->d_drhodfCPLabel,
		    matlIndex, patch);
    for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++)
      new_dw->getCopy(pressureVars.scalarDiffusionCoeff[ii], 
		      d_lab->d_scalDiffCoefCorrLabel, 
		      ii, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->getModifiable(pressureVars.divergence, d_lab->d_divConstraintLabel,
		          matlIndex, patch);
    pressureVars.divergence.initialize(0.0);
#endif

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

  #ifdef Runge_Kutta_3d
    new_dw->getCopy(pressureVars.uVelocity, d_lab->d_uVelocityIntermLabel, 
		matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->getCopy(pressureVars.vVelocity, d_lab->d_vVelocityIntermLabel, 
		matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->getCopy(pressureVars.wVelocity, d_lab->d_wVelocityIntermLabel, 
		matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);

    new_dw->getCopy(pressureVars.old_uVelocity, d_lab->d_uVelocityIntermLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->getCopy(pressureVars.old_vVelocity, d_lab->d_vVelocityIntermLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->getCopy(pressureVars.old_wVelocity, d_lab->d_wVelocityIntermLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);

    new_dw->getCopy(pressureVars.old_density, d_lab->d_densityIntermLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
  #else
    new_dw->getCopy(pressureVars.uVelocity, d_lab->d_uVelocityPredLabel, 
		matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->getCopy(pressureVars.vVelocity, d_lab->d_vVelocityPredLabel, 
		matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->getCopy(pressureVars.wVelocity, d_lab->d_wVelocityPredLabel, 
		matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
  #ifdef Runge_Kutta_2nd
    new_dw->getCopy(pressureVars.old_uVelocity, d_lab->d_uVelocityPredLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->getCopy(pressureVars.old_vVelocity, d_lab->d_vVelocityPredLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->getCopy(pressureVars.old_wVelocity, d_lab->d_wVelocityPredLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);

    new_dw->getCopy(pressureVars.old_density, d_lab->d_densityPredLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
  #else
#if 1
    new_dw->getCopy(pressureVars.old_uVelocity, d_lab->d_uVelocityOUTBCLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->getCopy(pressureVars.old_vVelocity, d_lab->d_vVelocityOUTBCLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->getCopy(pressureVars.old_wVelocity, d_lab->d_wVelocityOUTBCLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
#else
    old_dw->getCopy(pressureVars.old_uVelocity, d_lab->d_uVelocitySPBCLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    old_dw->getCopy(pressureVars.old_vVelocity, d_lab->d_vVelocitySPBCLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    old_dw->getCopy(pressureVars.old_wVelocity, d_lab->d_wVelocitySPBCLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
#endif

    new_dw->getCopy(pressureVars.old_density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
  #endif
  #endif
    new_dw->getCopy(pressureVars.cellType, d_lab->d_cellTypeLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    
    for(int index = 1; index <= Arches::NDIM; ++index) {

    // get multimaterial momentum source terms

      if (d_MAlab) {
	switch (index) {
	
	case Arches::XDIR:

	  new_dw->getCopy(pressureVars.mmuVelSu, d_MAlab->d_uVel_mmNonlinSrcLabel,
		      matlIndex, patch,
		      Ghost::None, Arches::ZEROGHOSTCELLS);
	  new_dw->getCopy(pressureVars.mmuVelSp, d_MAlab->d_uVel_mmLinSrcLabel,
		      matlIndex, patch,
		      Ghost::None, Arches::ZEROGHOSTCELLS);
	  break;

	case Arches::YDIR:

	  new_dw->getCopy(pressureVars.mmvVelSu, d_MAlab->d_vVel_mmNonlinSrcLabel,
		      matlIndex, patch,
		      Ghost::None, Arches::ZEROGHOSTCELLS);
	  new_dw->getCopy(pressureVars.mmvVelSp, d_MAlab->d_vVel_mmLinSrcLabel,
		      matlIndex, patch,
		      Ghost::None, Arches::ZEROGHOSTCELLS);
	  break;
	case Arches::ZDIR:

	  new_dw->getCopy(pressureVars.mmwVelSu, d_MAlab->d_wVel_mmNonlinSrcLabel,
		      matlIndex, patch,
		      Ghost::None, Arches::ZEROGHOSTCELLS);
	  new_dw->getCopy(pressureVars.mmwVelSp, d_MAlab->d_wVel_mmLinSrcLabel,
		      matlIndex, patch,
		      Ghost::None, Arches::ZEROGHOSTCELLS);
	  break;
	}
      }
      
      for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++) {

	switch(index) {

	case Arches::XDIR:

	  new_dw->allocateTemporary(pressureVars.uVelocityCoeff[ii],  patch);
	  new_dw->allocateTemporary(pressureVars.uVelocityConvectCoeff[ii],  patch);
	  break;
	case Arches::YDIR:
	  new_dw->allocateTemporary(pressureVars.vVelocityCoeff[ii],  patch);
	  new_dw->allocateTemporary(pressureVars.vVelocityConvectCoeff[ii],  patch);
	  break;
	case Arches::ZDIR:
	  new_dw->allocateTemporary(pressureVars.wVelocityCoeff[ii],  patch);
	  new_dw->allocateTemporary(pressureVars.wVelocityConvectCoeff[ii],  patch);
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

	new_dw->allocateTemporary(pressureVars.uVelLinearSrc,  patch);
	new_dw->allocateTemporary(pressureVars.uVelNonlinearSrc,  patch);
	new_dw->allocateAndPut(pressureVars.uVelRhoHat, d_lab->d_uVelRhoHatCorrLabel,
			 matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
	pressureVars.uVelRhoHat.copy(pressureVars.uVelocity,
				     pressureVars.uVelRhoHat.getLowIndex(),
				     pressureVars.uVelRhoHat.getHighIndex());

	break;

      case Arches::YDIR:

	new_dw->allocateTemporary(pressureVars.vVelLinearSrc,  patch);
	new_dw->allocateTemporary(pressureVars.vVelNonlinearSrc,  patch);
	new_dw->allocateAndPut(pressureVars.vVelRhoHat, d_lab->d_vVelRhoHatCorrLabel,
			 matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
	pressureVars.vVelRhoHat.copy(pressureVars.vVelocity,
				     pressureVars.vVelRhoHat.getLowIndex(),
				     pressureVars.vVelRhoHat.getHighIndex());

	break;

      case Arches::ZDIR:

	new_dw->allocateTemporary(pressureVars.wVelLinearSrc,  patch);
	new_dw->allocateTemporary(pressureVars.wVelNonlinearSrc,  patch);
	new_dw->allocateAndPut(pressureVars.wVelRhoHat, d_lab->d_wVelRhoHatCorrLabel,
			 matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
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
#if 0
      d_discretize->calculateVelDiagonal(pc, patch,
					 index,
					 &pressureVars);

      d_source->modifyVelMassSource(pc, patch, delta_t, index,
				    &pressureVars);
#endif
      d_source->modifyVelMassSource(pc, patch, delta_t, index,
				    &pressureVars);
      d_discretize->calculateVelDiagonal(pc, patch,
					 index,
					 &pressureVars);


    // Calculate Velocity diagonal
    //  inputs : [u,v,w]VelCoefPBLM, [u,v,w]VelLinSrcPBLM
    //  outputs: [u,v,w]VelCoefPBLM

    d_discretize->calculateVelRhoHat(pc, patch, index, delta_t,
				     cellinfo, &pressureVars);

#ifdef Runge_Kutta_3d
#ifndef Runge_Kutta_3d_ssp
    constSFCXVariable<double> temp_uVel;
    constSFCYVariable<double> temp_vVel;
    constSFCZVariable<double> temp_wVel;
    constCCVariable<double> new_density;
    IntVector indexLow;
    IntVector indexHigh;

    new_dw->get(new_density, d_lab->d_densityCPLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);

    switch(index) {

    case Arches::XDIR:

      new_dw->get(temp_uVel, d_lab->d_uVelTempLabel,
		 matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    
      indexLow = patch->getSFCXFORTLowIndex();
      indexHigh = patch->getSFCXFORTHighIndex();
    
      for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
        for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
          for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {

              IntVector currCell(colX, colY, colZ);
              IntVector xshiftedCell(colX-1, colY, colZ);

              pressureVars.uVelRhoHat[currCell] += zeta_2*temp_uVel[currCell]/
              (0.5*(new_density[currCell]+new_density[xshiftedCell]));
          }
        }
      }
    
    break;

    case Arches::YDIR:

      new_dw->get(temp_vVel, d_lab->d_vVelTempLabel,
		 matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);

      indexLow = patch->getSFCYFORTLowIndex();
      indexHigh = patch->getSFCYFORTHighIndex();
    
      for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
        for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
          for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {

              IntVector currCell(colX, colY, colZ);
              IntVector yshiftedCell(colX, colY-1, colZ);

              pressureVars.vVelRhoHat[currCell] += zeta_2*temp_vVel[currCell]/
              (0.5*(new_density[currCell]+new_density[yshiftedCell]));
          }
        }
      }
    
    break;

    case Arches::ZDIR:

      new_dw->get(temp_wVel, d_lab->d_wVelTempLabel,
		 matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);

      indexLow = patch->getSFCZFORTLowIndex();
      indexHigh = patch->getSFCZFORTHighIndex();
    
      for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
        for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
          for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {

              IntVector currCell(colX, colY, colZ);
              IntVector zshiftedCell(colX, colY, colZ-1);

              pressureVars.wVelRhoHat[currCell] += zeta_2*temp_wVel[currCell]/
              (0.5*(new_density[currCell]+new_density[zshiftedCell]));
          }
        }
      }
    break;

    default:
      throw InvalidValue("Invalid index in Corr PressureSolver for RK3");
    }
#endif
#endif

#ifdef ARCHES_PRES_DEBUG
    std::cerr << "Done building matrix for press coeff" << endl;
#endif
    
    }
    d_boundaryCondition->newrecomputePressureBC(pc, patch,
						cellinfo, &pressureVars); 
#ifdef divergenceconstraint    
    // compute divergence constraint to use in pressure equation
    d_discretize->computeDivergence(pc, patch, &pressureVars);
#endif


  // put required vars

    // allocateAndPut instead:
    /* new_dw->put(pressureVars.uVelRhoHat, d_lab->d_uVelRhoHatCorrLabel, 
		matlIndex, patch); */;
    // allocateAndPut instead:
    /* new_dw->put(pressureVars.vVelRhoHat, d_lab->d_vVelRhoHatCorrLabel, 
		matlIndex, patch); */;
    // allocateAndPut instead:
    /* new_dw->put(pressureVars.wVelRhoHat, d_lab->d_wVelRhoHatCorrLabel, 
		matlIndex, patch); */;

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
					   const MaterialSubset* /* matls */,
					   DataWarehouse* old_dw,
					   DataWarehouse* new_dw)
{
  delt_vartype delT;
  old_dw->get(delT, d_lab->d_sharedState->get_delt_label() );
  double delta_t = delT;

#ifdef Runge_Kutta_3d
#ifndef Runge_Kutta_3d_ssp
  double gamma_3 = 3.0/4.0;
  delta_t *= gamma_3;
#endif
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
    int nofStencils = 7;
    // Get the reference density
    // Get the required data
    new_dw->getCopy(pressureVars.density, d_lab->d_densityCPLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
  #ifdef Runge_Kutta_3d
    new_dw->getCopy(pressureVars.pressure, d_lab->d_pressureIntermLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->getCopy(pressureVars.viscosity, d_lab->d_viscosityIntermLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
  #else
    new_dw->getCopy(pressureVars.pressure, d_lab->d_pressurePredLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->getCopy(pressureVars.viscosity, d_lab->d_viscosityPredLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
  #endif
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
  #ifdef Runge_Kutta_3d
    new_dw->getCopy(pressureVars.old_density, d_lab->d_densityIntermLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    // 2nd order density time derivative needs to be worked out for 3d order R.-K.
    old_dw->getCopy(pressureVars.old_old_density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
  #else
  #ifdef Runge_Kutta_2nd
    new_dw->getCopy(pressureVars.old_density, d_lab->d_densityPredLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->getCopy(pressureVars.old_old_density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
  #else
    new_dw->getCopy(pressureVars.old_density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    old_dw->getCopy(pressureVars.old_old_density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
  #endif
  #endif
    //new_dw->getCopy(pressureVars.pred_density, d_lab->d_densityCPLabel, 
    //		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->getCopy(pressureVars.cellType, d_lab->d_cellTypeLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
  
    new_dw->getCopy(pressureVars.uVelRhoHat, d_lab->d_uVelRhoHatCorrLabel, 
		matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->getCopy(pressureVars.vVelRhoHat, d_lab->d_vVelRhoHatCorrLabel, 
		matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->getCopy(pressureVars.wVelRhoHat, d_lab->d_wVelRhoHatCorrLabel, 
		matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
#ifdef divergenceconstraint
    new_dw->getCopy(pressureVars.divergence, d_lab->d_divConstraintLabel,
		    matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
#endif
    
    // Calculate Pressure Coeffs
    //  inputs : densityIN, pressureIN, [u,v,w]VelCoefPBLM[Arches::AP]
    //  outputs: presCoefPBLM[Arches::AE..AB] 
    for (int ii = 0; ii < nofStencils; ii++) {
      new_dw->allocateAndPut(pressureVars.pressCoeff[ii], d_lab->d_presCoefCorrLabel, ii, patch);
      pressureVars.pressCoeff[ii].initialize(0.0);
    }
    d_discretize->calculatePressureCoeff(pc, patch, old_dw, new_dw, 
					 delta_t, cellinfo, &pressureVars);

    // Modify pressure coefficients for multimaterial formulation

    if (d_MAlab) {

      new_dw->getCopy(pressureVars.voidFraction,
		  d_lab->d_mmgasVolFracLabel, matlIndex, patch,
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);

      d_discretize->mmModifyPressureCoeffs(pc, patch, &pressureVars);

    }

    // Calculate Pressure Source
    //  inputs : pressureSPBC, [u,v,w]VelocitySIVBC, densityCP,
    //           [u,v,w]VelCoefPBLM, [u,v,w]VelNonLinSrcPBLM
    //  outputs: presLinSrcPBLM, presNonLinSrcPBLM
    // Allocate space

    new_dw->allocateTemporary(pressureVars.pressLinearSrc,  patch);
    new_dw->allocateAndPut(pressureVars.pressNonlinearSrc, d_lab->d_presNonLinSrcCorrLabel, matlIndex, patch);
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
      // allocateAndPut instead:
      /* new_dw->put(pressureVars.pressCoeff[ii], d_lab->d_presCoefCorrLabel, 
		  ii, patch); */;
    }
    // allocateAndPut instead:
    /* new_dw->put(pressureVars.pressNonlinearSrc, 
		d_lab->d_presNonLinSrcCorrLabel, matlIndex, patch); */;

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

  // Requires
  // coefficient for the variable for which solve is invoked

  #ifdef Runge_Kutta_3d
  tsk->requires(Task::NewDW, d_lab->d_pressureIntermLabel, 
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
  #else
  tsk->requires(Task::NewDW, d_lab->d_pressurePredLabel, 
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
  #endif
  tsk->requires(Task::NewDW, d_lab->d_presCoefCorrLabel, d_lab->d_stencilMatl,
		Task::OutOfDomain, Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_presNonLinSrcCorrLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);

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
  // Get the required data
  new_dw->allocateTemporary(pressureVars.pressure,  patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
  #ifdef Runge_Kutta_3d
  new_dw->copyOut(pressureVars.pressure, d_lab->d_pressureIntermLabel, 
	      matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
  #else
  new_dw->copyOut(pressureVars.pressure, d_lab->d_pressurePredLabel, 
	      matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
  #endif
  for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++) 
    new_dw->getCopy(pressureVars.pressCoeff[ii], d_lab->d_presCoefCorrLabel, 
		   ii, patch, Ghost::None, Arches::ZEROGHOSTCELLS);

  new_dw->getCopy(pressureVars.pressNonlinearSrc, 
		 d_lab->d_presNonLinSrcCorrLabel, 
		 matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);

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

// ****************************************************************************
// Schedule solve of linearized pressure equation, intermediate step
// ****************************************************************************
void PressureSolver::solveInterm(const LevelP& level,
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

  sched_buildLinearMatrixInterm(sched, patches, matls);

  // Schedule the pressure solve
  // require : pressureIN, presCoefPBLM, presNonLinSrcPBLM
  // compute : presResidualPS, presCoefPS, presNonLinSrcPS, pressurePS
  //d_linearSolver->sched_pressureSolve(level, sched, new_dw, matrix_dw);

  sched_pressureLinearSolveInterm(level, sched);

}

// ****************************************************************************
// Schedule build of linear matrix
// ****************************************************************************
void 
PressureSolver::sched_buildLinearMatrixInterm(SchedulerP& sched,
					    const PatchSet* patches,
					    const MaterialSet* matls)
{
  // Build momentum equation coefficients and sources that are needed 
  // to later build pressure equation coefficients and source

  {
    Task* tsk = scinew Task( "Psolve::BuildCoeffInterm", 
			     this, &PressureSolver::buildLinearMatrixInterm);


    tsk->requires(Task::OldDW, d_lab->d_sharedState->get_delt_label());
    
  // Requires
  // from old_dw for time integration
  // get old_dw from getTop function
    tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel,
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);
    //    tsk->requires(Task::NewDW, d_lab->d_densityINLabel, 
    //		  Ghost::None, Arches::ZEROGHOSTCELLS);

  // from new_dw
  // for new task graph to work

    tsk->requires(Task::NewDW, d_lab->d_densityIntermLabel, 
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);

    tsk->requires(Task::NewDW, d_lab->d_pressurePredLabel, 
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);
    tsk->requires(Task::NewDW, d_lab->d_densityPredLabel,
		  Ghost::AroundCells, Arches::TWOGHOSTCELLS);
    tsk->requires(Task::NewDW, d_lab->d_uVelocityPredLabel,
		  Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    tsk->requires(Task::NewDW, d_lab->d_vVelocityPredLabel,
		  Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    tsk->requires(Task::NewDW, d_lab->d_wVelocityPredLabel,
		  Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    tsk->requires(Task::NewDW, d_lab->d_viscosityPredLabel,
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);

    tsk->requires(Task::NewDW, d_lab->d_denRefArrayIntermLabel,
    		  Ghost::AroundCells, Arches::ONEGHOSTCELL);

#ifdef divergenceconstraint
    tsk->requires(Task::NewDW, d_lab->d_scalarIntermLabel, 
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);
    tsk->requires(Task::NewDW, d_lab->d_drhodfIntermLabel, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);
    tsk->requires(Task::NewDW, d_lab->d_scalDiffCoefIntermLabel, 
		  d_lab->d_stencilMatl, Task::OutOfDomain,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
#endif
    // for multi-material
    // requires su_drag[x,y,z], sp_drag[x,y,z] for arches

    if (d_MAlab) {

      tsk->requires(Task::NewDW, d_MAlab->d_uVel_mmLinSrcLabel,
		    Ghost::None, Arches::ZEROGHOSTCELLS);
      tsk->requires(Task::NewDW, d_MAlab->d_uVel_mmNonlinSrcLabel,
		    Ghost::None, Arches::ZEROGHOSTCELLS);
      tsk->requires(Task::NewDW, d_MAlab->d_vVel_mmLinSrcLabel, 
		    Ghost::None, Arches::ZEROGHOSTCELLS);
      tsk->requires(Task::NewDW, d_MAlab->d_vVel_mmNonlinSrcLabel,
		    Ghost::None, Arches::ZEROGHOSTCELLS);
      tsk->requires(Task::NewDW, d_MAlab->d_wVel_mmLinSrcLabel,
		    Ghost::None, Arches::ZEROGHOSTCELLS);
      tsk->requires(Task::NewDW, d_MAlab->d_wVel_mmNonlinSrcLabel,
		    Ghost::None, Arches::ZEROGHOSTCELLS);

    }
    
    
    // requires convection coeff because of the nodal
    // differencing
    // computes all the components of velocity

#ifndef Runge_Kutta_3d_ssp
    tsk->modifies(d_lab->d_uVelTempLabel);
    tsk->modifies(d_lab->d_vVelTempLabel);
    tsk->modifies(d_lab->d_wVelTempLabel);
#endif
    tsk->computes(d_lab->d_uVelRhoHatIntermLabel);
    tsk->computes(d_lab->d_vVelRhoHatIntermLabel);
    tsk->computes(d_lab->d_wVelRhoHatIntermLabel);
#ifdef divergenceconstraint
    tsk->modifies(d_lab->d_divConstraintLabel);
#endif
    
    sched->addTask(tsk, patches, matls);
  }
  // Now build pressure equation coefficients from momentum equation 
  // coefficients

  {
    Task* tsk = scinew Task( "Psolve::BuildCoeffPInterm",
			     this,
			     &PressureSolver::buildLinearMatrixPressInterm);
    

    tsk->requires(Task::OldDW, d_lab->d_sharedState->get_delt_label());
    
    // int matlIndex = 0;
    // Requires
    // from old_dw for time integration
    // get old_dw from getTop function

    tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel, 
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);

    tsk->requires(Task::NewDW, d_lab->d_densityIntermLabel, 
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);
    tsk->requires(Task::NewDW, d_lab->d_densityPredLabel,
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);
    tsk->requires(Task::OldDW, d_lab->d_densityINLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);

    tsk->requires(Task::NewDW, d_lab->d_pressurePredLabel,
		  Ghost::AroundCells, Arches::ZEROGHOSTCELLS);
    tsk->requires(Task::NewDW, d_lab->d_viscosityPredLabel,
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);

    tsk->requires(Task::NewDW, d_lab->d_uVelRhoHatIntermLabel,
		  Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    tsk->requires(Task::NewDW, d_lab->d_vVelRhoHatIntermLabel,
		  Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    tsk->requires(Task::NewDW, d_lab->d_wVelRhoHatIntermLabel,
		  Ghost::AroundFaces, Arches::ONEGHOSTCELL);
#ifdef divergenceconstraint
    tsk->requires(Task::NewDW, d_lab->d_divConstraintLabel,
		  Ghost::None, Arches::ZEROGHOSTCELLS);
#endif

    tsk->computes(d_lab->d_presCoefIntermLabel, d_lab->d_stencilMatl,
		   Task::OutOfDomain);
    tsk->computes(d_lab->d_presNonLinSrcIntermLabel);

    if (d_MAlab) {
      tsk->requires(Task::NewDW, d_lab->d_mmgasVolFracLabel, 
		    Ghost::AroundCells, Arches::ONEGHOSTCELL);
    }

    sched->addTask(tsk, patches, matls);
  }

}



void 
PressureSolver::buildLinearMatrixInterm(const ProcessorGroup* pc,
				      const PatchSubset* patches,
				      const MaterialSubset* /*matls*/,
				      DataWarehouse* old_dw,
				      DataWarehouse* new_dw)
{
  delt_vartype delT;
  old_dw->get(delT, d_lab->d_sharedState->get_delt_label() );
  double delta_t = delT;

#ifndef Runge_Kutta_3d_ssp
  double gamma_2 = 5.0/12.0;
  double zeta_1 = -17.0/60.0;
  delta_t *= gamma_2; 
#endif
  
  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    ArchesVariables pressureVars;

    // compute all three componenets of velocity stencil coefficients


    // Get the reference density
    // Get the required data

    new_dw->getCopy(pressureVars.density, d_lab->d_densityPredLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::TWOGHOSTCELLS);
    new_dw->getCopy(pressureVars.new_density, d_lab->d_densityIntermLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->getCopy(pressureVars.pressure, d_lab->d_pressurePredLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->getCopy(pressureVars.viscosity, d_lab->d_viscosityPredLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->getCopy(pressureVars.denRefArray, d_lab->d_denRefArrayIntermLabel,
    		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
#ifdef divergenceconstraint
    new_dw->getCopy(pressureVars.scalar, d_lab->d_scalarIntermLabel,
		    matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->getCopy(pressureVars.drhodf, d_lab->d_drhodfIntermLabel,
		    matlIndex, patch);
    for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++)
      new_dw->getCopy(pressureVars.scalarDiffusionCoeff[ii], 
		      d_lab->d_scalDiffCoefIntermLabel, 
		      ii, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->getModifiable(pressureVars.divergence, d_lab->d_divConstraintLabel,
		          matlIndex, patch);
    pressureVars.divergence.initialize(0.0);
#endif

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
		matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->getCopy(pressureVars.vVelocity, d_lab->d_vVelocityPredLabel, 
		matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->getCopy(pressureVars.wVelocity, d_lab->d_wVelocityPredLabel, 
		matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);

    new_dw->getCopy(pressureVars.old_uVelocity, d_lab->d_uVelocityPredLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->getCopy(pressureVars.old_vVelocity, d_lab->d_vVelocityPredLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->getCopy(pressureVars.old_wVelocity, d_lab->d_wVelocityPredLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);

    new_dw->getCopy(pressureVars.old_density, d_lab->d_densityPredLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->getCopy(pressureVars.cellType, d_lab->d_cellTypeLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    
    for(int index = 1; index <= Arches::NDIM; ++index) {

    // get multimaterial momentum source terms

      if (d_MAlab) {
	switch (index) {
	
	case Arches::XDIR:

	  new_dw->getCopy(pressureVars.mmuVelSu, d_MAlab->d_uVel_mmNonlinSrcLabel,
		      matlIndex, patch,
		      Ghost::None, Arches::ZEROGHOSTCELLS);
	  new_dw->getCopy(pressureVars.mmuVelSp, d_MAlab->d_uVel_mmLinSrcLabel,
		      matlIndex, patch,
		      Ghost::None, Arches::ZEROGHOSTCELLS);
	  break;

	case Arches::YDIR:

	  new_dw->getCopy(pressureVars.mmvVelSu, d_MAlab->d_vVel_mmNonlinSrcLabel,
		      matlIndex, patch,
		      Ghost::None, Arches::ZEROGHOSTCELLS);
	  new_dw->getCopy(pressureVars.mmvVelSp, d_MAlab->d_vVel_mmLinSrcLabel,
		      matlIndex, patch,
		      Ghost::None, Arches::ZEROGHOSTCELLS);
	  break;
	case Arches::ZDIR:

	  new_dw->getCopy(pressureVars.mmwVelSu, d_MAlab->d_wVel_mmNonlinSrcLabel,
		      matlIndex, patch,
		      Ghost::None, Arches::ZEROGHOSTCELLS);
	  new_dw->getCopy(pressureVars.mmwVelSp, d_MAlab->d_wVel_mmLinSrcLabel,
		      matlIndex, patch,
		      Ghost::None, Arches::ZEROGHOSTCELLS);
	  break;
	}
      }
      
      for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++) {

	switch(index) {

	case Arches::XDIR:

	  new_dw->allocateTemporary(pressureVars.uVelocityCoeff[ii],  patch);
	  new_dw->allocateTemporary(pressureVars.uVelocityConvectCoeff[ii],  patch);
	  break;
	case Arches::YDIR:
	  new_dw->allocateTemporary(pressureVars.vVelocityCoeff[ii],  patch);
	  new_dw->allocateTemporary(pressureVars.vVelocityConvectCoeff[ii],  patch);
	  break;
	case Arches::ZDIR:
	  new_dw->allocateTemporary(pressureVars.wVelocityCoeff[ii],  patch);
	  new_dw->allocateTemporary(pressureVars.wVelocityConvectCoeff[ii],  patch);
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

	new_dw->allocateTemporary(pressureVars.uVelLinearSrc,  patch);
	new_dw->allocateTemporary(pressureVars.uVelNonlinearSrc,  patch);
	new_dw->allocateAndPut(pressureVars.uVelRhoHat, d_lab->d_uVelRhoHatIntermLabel,
			 matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
	pressureVars.uVelRhoHat.copy(pressureVars.uVelocity,
				     pressureVars.uVelRhoHat.getLowIndex(),
				     pressureVars.uVelRhoHat.getHighIndex());

	break;

      case Arches::YDIR:

	new_dw->allocateTemporary(pressureVars.vVelLinearSrc,  patch);
	new_dw->allocateTemporary(pressureVars.vVelNonlinearSrc,  patch);
	new_dw->allocateAndPut(pressureVars.vVelRhoHat, d_lab->d_vVelRhoHatIntermLabel,
			 matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
	pressureVars.vVelRhoHat.copy(pressureVars.vVelocity,
				     pressureVars.vVelRhoHat.getLowIndex(),
				     pressureVars.vVelRhoHat.getHighIndex());

	break;

      case Arches::ZDIR:

	new_dw->allocateTemporary(pressureVars.wVelLinearSrc,  patch);
	new_dw->allocateTemporary(pressureVars.wVelNonlinearSrc,  patch);
	new_dw->allocateAndPut(pressureVars.wVelRhoHat, d_lab->d_wVelRhoHatIntermLabel,
			 matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
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
      d_boundaryCondition->mmvelocityBC(pc, patch, index, cellinfo, 
					&pressureVars);
    
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

      
#ifndef Runge_Kutta_3d_ssp
    SFCXVariable<double> temp_uVel;
    SFCYVariable<double> temp_vVel;
    SFCZVariable<double> temp_wVel;
    constCCVariable<double> old_density;
    constCCVariable<double> new_density;
    IntVector indexLow;
    IntVector indexHigh;

    new_dw->get(old_density, d_lab->d_densityPredLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->get(new_density, d_lab->d_densityIntermLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    
    switch(index) {

    case Arches::XDIR:

      new_dw->getModifiable(temp_uVel, d_lab->d_uVelTempLabel,
                  matlIndex, patch);
    
      indexLow = patch->getSFCXFORTLowIndex();
      indexHigh = patch->getSFCXFORTHighIndex();
    
      for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
        for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
          for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {

              IntVector currCell(colX, colY, colZ);
              IntVector xshiftedCell(colX-1, colY, colZ);


              pressureVars.uVelRhoHat[currCell] += zeta_1*temp_uVel[currCell]/
              (0.5*(new_density[currCell]+new_density[xshiftedCell]));
              temp_uVel[currCell] = 0.5*(
              (new_density[currCell]+new_density[xshiftedCell])*
	      pressureVars.uVelRhoHat[currCell]-
              (old_density[currCell]+old_density[xshiftedCell])*
              pressureVars.uVelocity[currCell])/
              gamma_2-zeta_1*temp_uVel[currCell]/gamma_2;
          }
        }
      }
//    new_dw->put(temp_wVel, d_lab->d_wVelTempLabel, matlIndex, patch);
    
    break;

    case Arches::YDIR:

      new_dw->getModifiable(temp_vVel, d_lab->d_vVelTempLabel,
                  matlIndex, patch);

      indexLow = patch->getSFCYFORTLowIndex();
      indexHigh = patch->getSFCYFORTHighIndex();
    
      for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
        for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
          for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {

              IntVector currCell(colX, colY, colZ);
              IntVector yshiftedCell(colX, colY-1, colZ);

              pressureVars.vVelRhoHat[currCell] += zeta_1*temp_vVel[currCell]/
              (0.5*(new_density[currCell]+new_density[yshiftedCell]));
              temp_vVel[currCell] = 0.5*(
              (new_density[currCell]+new_density[yshiftedCell])*
	      pressureVars.vVelRhoHat[currCell]-
              (old_density[currCell]+old_density[yshiftedCell])*
              pressureVars.vVelocity[currCell])/
              gamma_2-zeta_1*temp_vVel[currCell]/gamma_2;
          }
        }
      }
//    new_dw->put(temp_uVel, d_lab->d_uVelTempLabel, matlIndex, patch);
    
    break;

    case Arches::ZDIR:

      new_dw->getModifiable(temp_wVel, d_lab->d_wVelTempLabel,
                  matlIndex, patch);

      indexLow = patch->getSFCZFORTLowIndex();
      indexHigh = patch->getSFCZFORTHighIndex();
    
      for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
        for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
          for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {

              IntVector currCell(colX, colY, colZ);
              IntVector zshiftedCell(colX, colY, colZ-1);

              pressureVars.wVelRhoHat[currCell] += zeta_1*temp_wVel[currCell]/
              (0.5*(new_density[currCell]+new_density[zshiftedCell]));
              temp_wVel[currCell] = 0.5*(
              (new_density[currCell]+new_density[zshiftedCell])*
              pressureVars.wVelRhoHat[currCell]-
              (old_density[currCell]+old_density[zshiftedCell])*
              pressureVars.wVelocity[currCell])/
              gamma_2-zeta_1*temp_wVel[currCell]/gamma_2;
          }
        }
      }
//    new_dw->put(temp_wVel, d_lab->d_wVelTempLabel, matlIndex, patch);

    break;

    default:
      throw InvalidValue("Invalid index in Interm PressureSolver for RK3");
    }
#endif
    
#ifdef ARCHES_PRES_DEBUG
    std::cerr << "Done building matrix for press coeff" << endl;
#endif
    
    }
    d_boundaryCondition->newrecomputePressureBC(pc, patch,
    						cellinfo, &pressureVars); 
#ifdef divergenceconstraint    
    // compute divergence constraint to use in pressure equation
    d_discretize->computeDivergence(pc, patch, &pressureVars);
#endif


  // put required vars

    // allocateAndPut instead:
    /* new_dw->put(pressureVars.uVelRhoHat, d_lab->d_uVelRhoHatIntermLabel, 
		matlIndex, patch); */;
    // allocateAndPut instead:
    /* new_dw->put(pressureVars.vVelRhoHat, d_lab->d_vVelRhoHatIntermLabel, 
		matlIndex, patch); */;
    // allocateAndPut instead:
    /* new_dw->put(pressureVars.wVelRhoHat, d_lab->d_wVelRhoHatIntermLabel, 
		matlIndex, patch); */;

#ifdef ARCHES_PRES_DEBUG
    std::cerr << "Done building matrix for vel coeff for pressure" << endl;
#endif
    
  }
}



// ****************************************************************************
// Actually build of linear matrix for pressure equation
// ****************************************************************************
void 
PressureSolver::buildLinearMatrixPressInterm(const ProcessorGroup* pc,
					   const PatchSubset* patches,
					   const MaterialSubset* /* matls */,
					   DataWarehouse* old_dw,
					   DataWarehouse* new_dw)
{
  delt_vartype delT;
  old_dw->get(delT, d_lab->d_sharedState->get_delt_label() );
  double delta_t = delT;

#ifndef Runge_Kutta_3d_ssp
  double gamma_2 = 5.0/12.0;
  delta_t *= gamma_2; 
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
    int nofStencils = 7;
    // Get the reference density
    // Get the required data
    new_dw->getCopy(pressureVars.density, d_lab->d_densityIntermLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->getCopy(pressureVars.pressure, d_lab->d_pressurePredLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->getCopy(pressureVars.viscosity, d_lab->d_viscosityPredLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
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
    new_dw->getCopy(pressureVars.old_density, d_lab->d_densityPredLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    old_dw->getCopy(pressureVars.old_old_density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
    //new_dw->getCopy(pressureVars.pred_density, d_lab->d_densityIntermLabel, 
    //		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->getCopy(pressureVars.cellType, d_lab->d_cellTypeLabel, 
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
  
    new_dw->getCopy(pressureVars.uVelRhoHat, d_lab->d_uVelRhoHatIntermLabel, 
		matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->getCopy(pressureVars.vVelRhoHat, d_lab->d_vVelRhoHatIntermLabel, 
		matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->getCopy(pressureVars.wVelRhoHat, d_lab->d_wVelRhoHatIntermLabel, 
		matlIndex, patch, Ghost::AroundFaces, Arches::ONEGHOSTCELL);
#ifdef divergenceconstraint
    new_dw->getCopy(pressureVars.divergence, d_lab->d_divConstraintLabel,
		    matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);
#endif
    
    // Calculate Pressure Coeffs
    //  inputs : densityIN, pressureIN, [u,v,w]VelCoefPBLM[Arches::AP]
    //  outputs: presCoefPBLM[Arches::AE..AB] 
    for (int ii = 0; ii < nofStencils; ii++) {
      new_dw->allocateAndPut(pressureVars.pressCoeff[ii], d_lab->d_presCoefIntermLabel, ii, patch);
      pressureVars.pressCoeff[ii].initialize(0.0);
    }
    d_discretize->calculatePressureCoeff(pc, patch, old_dw, new_dw, 
					 delta_t, cellinfo, &pressureVars);

    // Modify pressure coefficients for multimaterial formulation

    if (d_MAlab) {

      new_dw->getCopy(pressureVars.voidFraction,
		  d_lab->d_mmgasVolFracLabel, matlIndex, patch,
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);

      d_discretize->mmModifyPressureCoeffs(pc, patch, &pressureVars);

    }

    // Calculate Pressure Source
    //  inputs : pressureSPBC, [u,v,w]VelocitySIVBC, densityCP,
    //           [u,v,w]VelCoefPBLM, [u,v,w]VelNonLinSrcPBLM
    //  outputs: presLinSrcPBLM, presNonLinSrcPBLM
    // Allocate space

    new_dw->allocateTemporary(pressureVars.pressLinearSrc,  patch);
    new_dw->allocateAndPut(pressureVars.pressNonlinearSrc, d_lab->d_presNonLinSrcIntermLabel, matlIndex, patch);
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
      // allocateAndPut instead:
      /* new_dw->put(pressureVars.pressCoeff[ii], d_lab->d_presCoefIntermLabel, 
		  ii, patch); */;
    }
    // allocateAndPut instead:
    /* new_dw->put(pressureVars.pressNonlinearSrc, 
		d_lab->d_presNonLinSrcIntermLabel, matlIndex, patch); */;

#ifdef ARCHES_PRES_DEBUG
  std::cerr << "Done building matrix for press coeff" << endl;
#endif

  }
}

// ****************************************************************************
// Schedule solver for linear matrix
// ****************************************************************************
void 
PressureSolver::sched_pressureLinearSolveInterm(const LevelP& level,
					      SchedulerP& sched)
{
  if(d_perproc_patches && d_perproc_patches->removeReference())
    delete d_perproc_patches;

  LoadBalancer* lb = sched->getLoadBalancer();
  d_perproc_patches = lb->createPerProcessorPatchSet(level, d_myworld);
  d_perproc_patches->addReference();
  const MaterialSet* matls = d_lab->d_sharedState->allArchesMaterials();

  Task* tsk = scinew Task("PressureSolver::PressLinearSolveInterm",
			  this,
			  &PressureSolver::pressureLinearSolveInterm_all);

  // Requires
  // coefficient for the variable for which solve is invoked

  tsk->requires(Task::NewDW, d_lab->d_pressurePredLabel, 
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_presCoefIntermLabel, d_lab->d_stencilMatl,
		Task::OutOfDomain, Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_presNonLinSrcIntermLabel,
		Ghost::None, Arches::ZEROGHOSTCELLS);

  // computes global residual
  tsk->computes(d_lab->d_presResidPSLabel);
  tsk->computes(d_lab->d_presTruncPSLabel);

  tsk->computes(d_lab->d_pressureIntermLabel);

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
PressureSolver::pressureLinearSolveInterm_all (const ProcessorGroup* pg,
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
    pressureLinearSolveInterm(pg, patch, matlIndex, old_dw, new_dw, pressureVars);
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
    new_dw->put(pressureVars.pressure, d_lab->d_pressureIntermLabel, 
		matlIndex, patch);
  }

  // destroy matrix
  d_linearSolver->destroyMatrix();
}

// Actual linear solve
void 
PressureSolver::pressureLinearSolveInterm (const ProcessorGroup* pc,
					 const Patch* patch,
					 const int matlIndex,
					 DataWarehouse* /*old_dw*/,
					 DataWarehouse* new_dw,
					 ArchesVariables& pressureVars)
{
  // Get the required data
  new_dw->allocateTemporary(pressureVars.pressure,  patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
  new_dw->copyOut(pressureVars.pressure, d_lab->d_pressurePredLabel, 
	      matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
  for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++) 
    new_dw->getCopy(pressureVars.pressCoeff[ii], d_lab->d_presCoefIntermLabel, 
		   ii, patch, Ghost::None, Arches::ZEROGHOSTCELLS);

  new_dw->getCopy(pressureVars.pressNonlinearSrc, 
		 d_lab->d_presNonLinSrcIntermLabel, 
		 matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);

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

