//----- PressureSolver.cc ----------------------------------------------

#include <Packages/Uintah/CCA/Components/Arches/PressureSolver.h>
#include <Packages/Uintah/CCA/Components/Arches/CellInformationP.h>
#include <Packages/Uintah/CCA/Components/Arches/Discretization.h>
#include <Packages/Uintah/CCA/Components/Arches/Source.h>
#include <Packages/Uintah/CCA/Components/Arches/BoundaryCondition.h>
#include <Packages/Uintah/CCA/Components/Arches/TurbulenceModel.h>
#include <Packages/Uintah/CCA/Components/Arches/PhysicalConstants.h>
#include <Packages/Uintah/CCA/Components/Arches/RBGSSolver.h>
#include <Packages/Uintah/CCA/Components/Arches/PetscSolver.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesVariables.h>
#include <Packages/Uintah/Core/Exceptions/InvalidValue.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/SFCZVariable.h>
#include <Packages/Uintah/CCA/Ports/LoadBalancer.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/CCA/Components/Arches/Arches.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesFort.h>
#include <Core/Util/NotFinished.h>

using namespace Uintah;
using namespace std;

//****************************************************************************
// Default constructor for PressureSolver
//****************************************************************************
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
}

//****************************************************************************
// Destructor
//****************************************************************************
PressureSolver::~PressureSolver()
{
}

//****************************************************************************
// Problem Setup
//****************************************************************************
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

//****************************************************************************
// Schedule solve of linearized pressure equation
//****************************************************************************
void PressureSolver::solve(const LevelP& level,
			   SchedulerP& sched,
			   DataWarehouseP& old_dw,
			   DataWarehouseP& new_dw,
			   double time, double delta_t)
{
#if 0
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
  sched_buildLinearMatrix(level, sched, old_dw, new_dw, delta_t);

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
  sched_pressureLinearSolve(level, sched, old_dw, new_dw);

  // Schedule Calculation of pressure norm
  // require :
  // compute :
  //sched_normPressure(level, sched, new_dw, matrix_dw);
  
#else
  NOT_FINISHED("new task stuff");
#endif
}

//****************************************************************************
// Schedule build of linear matrix
//****************************************************************************
void 
PressureSolver::sched_buildLinearMatrix(SchedulerP& sched, const PatchSet* patches,
					const MaterialSet* matls,
					double delta_t)
{
#if 0
  for(Level::const_patchIterator iter=level->patchesBegin();
      iter != level->patchesEnd(); iter++){
    const Patch* patch=*iter;
    {
      Task* tsk = scinew Task("PressureSolver::BuildCoeff",
			   patch, old_dw, new_dw, this,
			   &PressureSolver::buildLinearMatrix, delta_t);

      int numGhostCells = 1;
      int zeroGhostCells = 0;
      int matlIndex = 0;
      int nofStencils = 7;
      // Requires
      // from old_dw for time integration
      // get old_dw from getTop function
      tsk->requires(old_dw, d_lab->d_cellTypeLabel, matlIndex, patch, 
		    Ghost::AroundCells, numGhostCells);
      // checkpointing
      //      tsk->requires(new_dw, d_lab->d_cellInfoLabel, matlIndex, patch,
      //		    Ghost::None, numGhostCells);
      tsk->requires(old_dw, d_lab->d_refDensity_label);
      tsk->requires(old_dw, d_lab->d_densityINLabel, matlIndex, patch, 
		    Ghost::None, zeroGhostCells);
      tsk->requires(old_dw, d_lab->d_uVelocitySPBCLabel, matlIndex, patch, 
		    Ghost::None, zeroGhostCells);
      tsk->requires(old_dw, d_lab->d_vVelocitySPBCLabel, matlIndex, patch, 
		    Ghost::None, zeroGhostCells);
      tsk->requires(old_dw, d_lab->d_wVelocitySPBCLabel, matlIndex, patch, 
		    Ghost::None, zeroGhostCells);
      // from new_dw
      // for new task graph to work
      tsk->requires(new_dw, d_lab->d_pressureINLabel, matlIndex, patch, 
		    Ghost::AroundCells, numGhostCells);
      tsk->requires(new_dw, d_lab->d_densityINLabel, matlIndex, patch, 
		    Ghost::AroundCells, numGhostCells+1);
      tsk->requires(new_dw, d_lab->d_viscosityINLabel, matlIndex, patch, 
		    Ghost::AroundCells, numGhostCells);
      tsk->requires(new_dw, d_lab->d_uVelocitySIVBCLabel, matlIndex, patch, 
		    Ghost::AroundCells, numGhostCells+1);
      tsk->requires(new_dw, d_lab->d_vVelocitySIVBCLabel, matlIndex, patch, 
		    Ghost::AroundCells, numGhostCells+1);
      tsk->requires(new_dw, d_lab->d_wVelocitySIVBCLabel, matlIndex, patch, 
		    Ghost::AroundCells, numGhostCells+1);
      // for multi-material
    // requires su_drag[x,y,z], sp_drag[x,y,z] for arches
      if (d_MAlab) {
	tsk->requires(new_dw, d_MAlab->d_uVel_mmLinSrcLabel, matlIndex, patch,
		      Ghost::None, zeroGhostCells);
	tsk->requires(new_dw, d_MAlab->d_uVel_mmNonlinSrcLabel, matlIndex, patch,
		      Ghost::None, zeroGhostCells);
	tsk->requires(new_dw, d_MAlab->d_vVel_mmLinSrcLabel, matlIndex, patch,
		      Ghost::None, zeroGhostCells);
	tsk->requires(new_dw, d_MAlab->d_vVel_mmNonlinSrcLabel, matlIndex, patch,
		      Ghost::None, zeroGhostCells);
	tsk->requires(new_dw, d_MAlab->d_wVel_mmLinSrcLabel, matlIndex, patch,
		      Ghost::None, zeroGhostCells);
	tsk->requires(new_dw, d_MAlab->d_wVel_mmNonlinSrcLabel, matlIndex, patch,
		      Ghost::None, zeroGhostCells);
      }
 

      /// requires convection coeff because of the nodal
      // differencing
      // computes all the components of velocity
      for (int ii = 0; ii < nofStencils; ii++) {
	tsk->computes(new_dw, d_lab->d_uVelCoefPBLMLabel, ii, patch);
	tsk->computes(new_dw, d_lab->d_vVelCoefPBLMLabel, ii, patch);
	tsk->computes(new_dw, d_lab->d_wVelCoefPBLMLabel, ii, patch);
      }
      tsk->computes(new_dw, d_lab->d_uVelLinSrcPBLMLabel, matlIndex, patch);
      tsk->computes(new_dw, d_lab->d_vVelLinSrcPBLMLabel, matlIndex, patch);
      tsk->computes(new_dw, d_lab->d_wVelLinSrcPBLMLabel, matlIndex, patch);
      tsk->computes(new_dw, d_lab->d_uVelNonLinSrcPBLMLabel, matlIndex, patch);
      tsk->computes(new_dw, d_lab->d_vVelNonLinSrcPBLMLabel, matlIndex, patch);
      tsk->computes(new_dw, d_lab->d_wVelNonLinSrcPBLMLabel, matlIndex, patch);
     
      sched->addTask(tsk);
    }
  }
  for(Level::const_patchIterator iter=level->patchesBegin();
      iter != level->patchesEnd(); iter++){
    const Patch* patch=*iter;
    {
      Task* tsk = scinew Task("PressureSolver::BuildCoeffPress",
			   patch, old_dw, new_dw, this,
			   &PressureSolver::buildLinearMatrixPress, delta_t);

      int numGhostCells = 1;
      int zeroGhostCells = 0;
      int matlIndex = 0;
      int nofStencils = 7;
      // Requires
      // from old_dw for time integration
      // get old_dw from getTop function
      tsk->requires(old_dw, d_lab->d_cellTypeLabel, matlIndex, patch, 
		    Ghost::AroundCells, numGhostCells);
      // fix it
      tsk->requires(old_dw, d_lab->d_densityINLabel, matlIndex, patch, 
		    Ghost::None, zeroGhostCells);
      tsk->requires(old_dw, d_lab->d_uVelocitySPBCLabel, matlIndex, patch, 
		    Ghost::None, zeroGhostCells);
      tsk->requires(old_dw, d_lab->d_vVelocitySPBCLabel, matlIndex, patch, 
		    Ghost::None, zeroGhostCells);
      tsk->requires(old_dw, d_lab->d_wVelocitySPBCLabel, matlIndex, patch, 
		    Ghost::None, zeroGhostCells);
      // from new_dw
      tsk->requires(new_dw, d_lab->d_pressureINLabel, matlIndex, patch, 
		    Ghost::AroundCells, numGhostCells);
      tsk->requires(new_dw, d_lab->d_densityINLabel, matlIndex, patch, 
		    Ghost::AroundCells, numGhostCells+1);
      tsk->requires(new_dw, d_lab->d_viscosityINLabel, matlIndex, patch, 
		    Ghost::AroundCells, numGhostCells);
      tsk->requires(new_dw, d_lab->d_uVelocitySIVBCLabel, matlIndex, patch, 
		    Ghost::AroundCells, numGhostCells+1);
      tsk->requires(new_dw, d_lab->d_vVelocitySIVBCLabel, matlIndex, patch, 
		    Ghost::AroundCells, numGhostCells+1);
      tsk->requires(new_dw, d_lab->d_wVelocitySIVBCLabel, matlIndex, patch, 
		    Ghost::AroundCells, numGhostCells+1);

      /// requires convection coeff because of the nodal
      // differencing
      // computes all the components of velocity
      for (int ii = 0; ii < nofStencils; ii++) {
	tsk->requires(new_dw, d_lab->d_uVelCoefPBLMLabel, ii, patch,
		      Ghost::AroundCells, numGhostCells);
	tsk->requires(new_dw, d_lab->d_vVelCoefPBLMLabel, ii, patch,
		      Ghost::AroundCells, numGhostCells);
	tsk->requires(new_dw, d_lab->d_wVelCoefPBLMLabel, ii, patch,
		      Ghost::AroundCells, numGhostCells);
	tsk->computes(new_dw, d_lab->d_presCoefPBLMLabel, ii, patch);
      }
      tsk->requires(new_dw, d_lab->d_uVelLinSrcPBLMLabel, matlIndex, patch,
		    Ghost::AroundCells, numGhostCells);
      tsk->requires(new_dw, d_lab->d_vVelLinSrcPBLMLabel, matlIndex, patch,
		    Ghost::AroundCells, numGhostCells);
      tsk->requires(new_dw, d_lab->d_wVelLinSrcPBLMLabel, matlIndex, patch,
		    Ghost::AroundCells, numGhostCells);
      tsk->requires(new_dw, d_lab->d_uVelNonLinSrcPBLMLabel, matlIndex, patch,
		    Ghost::AroundCells, numGhostCells);
      tsk->requires(new_dw, d_lab->d_vVelNonLinSrcPBLMLabel, matlIndex, patch,
		    Ghost::AroundCells, numGhostCells);
      tsk->requires(new_dw, d_lab->d_wVelNonLinSrcPBLMLabel, matlIndex, patch,
		    Ghost::AroundCells, numGhostCells);
      tsk->computes(new_dw, d_lab->d_presNonLinSrcPBLMLabel, matlIndex, patch);
     
      sched->addTask(tsk);
    }

  }
#else
  NOT_FINISHED("new task stuff");
#endif
  }


//****************************************************************************
// Schedule solver for linear matrix
//****************************************************************************
void 
PressureSolver::sched_pressureLinearSolve(SchedulerP& sched, const PatchSet* patches,
					  const MaterialSet* matls)
{
#if 0
#if 0
  for(Level::const_patchIterator iter=level->patchesBegin();
      iter != level->patchesEnd(); iter++){
    const Patch* patch=*iter;
    {
      Task* tsk = scinew Task("PressureSolver::PressLinearSolve",
			   patch, old_dw, new_dw, this,
			   &PressureSolver::pressureLinearSolve);

      int numGhostCells = 0;
      int matlIndex = 0;
      int nofStencils = 7;
      // Requires
      // coefficient for the variable for which solve is invoked
      tsk->requires(new_dw, d_lab->d_pressureINLabel, matlIndex, patch, 
		    Ghost::None, numGhostCells);
      for (int ii = 0; ii < nofStencils; ii++)
	tsk->requires(new_dw, d_lab->d_presCoefPBLMLabel, ii, patch, 
		      Ghost::None, numGhostCells);
      tsk->requires(new_dw, d_lab->d_presNonLinSrcPBLMLabel, matlIndex, patch, 
		    Ghost::None, numGhostCells);
      // computes global residual
      //      tsk->computes(new_dw, d_lab->d_presResidPSLabel, matlIndex, patch);
      //      tsk->computes(new_dw, d_lab->d_presTruncPSLabel, matlIndex, patch);
      tsk->computes(new_dw, d_lab->d_pressurePSLabel, matlIndex, patch);

      sched->addTask(tsk);
    }
  }
#else
  int numProcessors = d_myworld->size();
  vector<Task*> tasks(numProcessors, (Task*)0);
  LoadBalancer* lb = sched->getLoadBalancer();

  //cerr << "In sched_PressureLinearSolve\n";
  for(Level::const_patchIterator iter=level->patchesBegin();
      iter != level->patchesEnd(); iter++){
    const Patch* patch=*iter;
    {
       int proc = lb->getPatchwiseProcessorAssignment(patch, d_myworld);
       Task* tsk = tasks[proc];
       if(!tsk){
	  tsk = scinew Task("PressureSolver::PressLinearSolve",
			    patch, old_dw, new_dw, this,
			    &PressureSolver::pressureLinearSolve_all,
			    level, sched);
	  tasks[proc]=tsk;
       }

       int numGhostCells = 1;
       int zeroGhostCells = 0;
       int matlIndex = 0;
       int nofStencils = 7;
       // Requires
       // coefficient for the variable for which solve is invoked
       tsk->requires(new_dw, d_lab->d_pressureINLabel, matlIndex, patch, 
		     Ghost::AroundCells, numGhostCells);
       for (int ii = 0; ii < nofStencils; ii++)
	  tsk->requires(new_dw, d_lab->d_presCoefPBLMLabel, ii, patch, 
			Ghost::None, zeroGhostCells);
       tsk->requires(new_dw, d_lab->d_presNonLinSrcPBLMLabel, matlIndex, patch, 
		     Ghost::None, zeroGhostCells);
       // computes global residual
       //      tsk->computes(new_dw, d_lab->d_presResidPSLabel, matlIndex, patch);
       //      tsk->computes(new_dw, d_lab->d_presTruncPSLabel, matlIndex, patch);
       tsk->computes(new_dw, d_lab->d_pressurePSLabel, matlIndex, patch);
#ifdef ARCHES_PRES_DEBUG
       cerr << "Adding computes on patch: " << patch->getID() << '\n';
#endif

    }
  }
  for(int i=0;i<tasks.size();i++)
     if(tasks[i]){
	sched->addTask(tasks[i]);
#ifdef ARCHES_PRES_DEBUG
	cerr << "Adding task: " << *tasks[i] << '\n';
#endif
     }
  sched->releaseLoadBalancer();
#endif
#else
  NOT_FINISHED("new task stuff");
#endif
}


//****************************************************************************
// Actually build of linear matrix
//****************************************************************************
void 
PressureSolver::buildLinearMatrix(const ProcessorGroup* pc,
				  const Patch* patch,
				  DataWarehouseP& old_dw,
				  DataWarehouseP& new_dw,
				  double delta_t)
{
  ArchesVariables pressureVars;
  // compute all three componenets of velocity stencil coefficients
  int matlIndex = 0;
  int numGhostCells = 1;
  int zeroGhostCells = 0;
  int nofStencils = 7;
  // Get the reference density
  // Get the required data
  new_dw->get(pressureVars.density, d_lab->d_densityINLabel, 
	      matlIndex, patch, Ghost::AroundCells, numGhostCells+1);
  new_dw->get(pressureVars.viscosity, d_lab->d_viscosityINLabel, 
	      matlIndex, patch, Ghost::AroundCells, numGhostCells);
  new_dw->get(pressureVars.pressure, d_lab->d_pressureINLabel, 
	      matlIndex, patch, Ghost::None, zeroGhostCells);
  sum_vartype den_ref_var;
  old_dw->get(den_ref_var, d_lab->d_refDensity_label);
  pressureVars.den_Ref = den_ref_var;
  //cerr << "getdensity_ref " << pressureVars.den_Ref << endl;
  // Get the PerPatch CellInformation data
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
  new_dw->get(pressureVars.uVelocity, d_lab->d_uVelocitySIVBCLabel, 
	      matlIndex, patch, Ghost::AroundCells, numGhostCells);
  new_dw->get(pressureVars.vVelocity, d_lab->d_vVelocitySIVBCLabel, 
	      matlIndex, patch, Ghost::AroundCells, numGhostCells);
  new_dw->get(pressureVars.wVelocity, d_lab->d_wVelocitySIVBCLabel, 
	      matlIndex, patch, Ghost::AroundCells, numGhostCells);
  old_dw->get(pressureVars.old_uVelocity, d_lab->d_uVelocitySPBCLabel, 
		matlIndex, patch, Ghost::None, zeroGhostCells);
  old_dw->get(pressureVars.old_vVelocity, d_lab->d_vVelocitySPBCLabel, 
		matlIndex, patch, Ghost::None, zeroGhostCells);
  old_dw->get(pressureVars.old_wVelocity, d_lab->d_wVelocitySPBCLabel, 
		matlIndex, patch, Ghost::None, zeroGhostCells);
  old_dw->get(pressureVars.old_density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::None, zeroGhostCells);
  old_dw->get(pressureVars.cellType, d_lab->d_cellTypeLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
  
  for(int index = 1; index <= Arches::NDIM; ++index) {
    // get multimaterial momentum source terms
    if (d_MAlab) {
      switch (index) {
	
      case Arches::XDIR:
	new_dw->get(pressureVars.mmuVelSu, d_MAlab->d_uVel_mmNonlinSrcLabel,
		    matlIndex, patch,
		    Ghost::None, zeroGhostCells);
	new_dw->get(pressureVars.mmuVelSp, d_MAlab->d_uVel_mmLinSrcLabel,
		    matlIndex, patch,
		    Ghost::None, zeroGhostCells);
	break;
      case Arches::YDIR:
	new_dw->get(pressureVars.mmvVelSu, d_MAlab->d_vVel_mmNonlinSrcLabel,
		    matlIndex, patch,
		    Ghost::None, zeroGhostCells);
	new_dw->get(pressureVars.mmvVelSp, d_MAlab->d_vVel_mmLinSrcLabel,
		    matlIndex, patch,
		    Ghost::None, zeroGhostCells);
	break;
      case Arches::ZDIR:
	new_dw->get(pressureVars.mmwVelSu, d_MAlab->d_wVel_mmNonlinSrcLabel,
		    matlIndex, patch,
		    Ghost::None, zeroGhostCells);
	new_dw->get(pressureVars.mmwVelSp, d_MAlab->d_wVel_mmLinSrcLabel,
		    matlIndex, patch,
		    Ghost::None, zeroGhostCells);
	break;
      }
    }
      

    for (int ii = 0; ii < nofStencils; ii++) {
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
    d_discretize->calculateVelocityCoeff(pc, patch, old_dw, new_dw, 
					 delta_t, index, 
					 Arches::PRESSURE, 
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
    d_source->calculateVelocitySource(pc, patch, old_dw, new_dw, 
				      delta_t, index,
				      Arches::PRESSURE,
				      cellinfo, &pressureVars);
    // add multimaterial momentum source term
    if (d_MAlab)
      d_source->computemmMomentumSource(pc, patch, index, cellinfo,
					&pressureVars);
#ifdef multimaterialform
    if (d_mmInterface) {
      MultiMaterialVars* mmVars = d_mmInterface->getMMVars();
      d_mmSGSModel->computeMomentumSource(patch, index, cellinfo,
					  mmVars, &pressureVars);
    }
#endif
    // Calculate the Velocity BCS
    //  inputs : densityCP, [u,v,w]VelocitySIVBC, [u,v,w]VelCoefPBLM
    //           [u,v,w]VelLinSrcPBLM, [u,v,w]VelNonLinSrcPBLM
    //  outputs: [u,v,w]VelCoefPBLM, [u,v,w]VelLinSrcPBLM, 
    //           [u,v,w]VelNonLinSrcPBLM
    
    d_boundaryCondition->velocityBC(pc, patch, old_dw, new_dw, 
				    index,
				    Arches::PRESSURE, cellinfo, &pressureVars);
    // apply multimaterial velocity bc
    // treats multimaterial wall as intrusion
    if (d_MAlab)
      d_boundaryCondition->mmvelocityBC(pc, patch, index, cellinfo, &pressureVars);

    // Modify Velocity Mass Source
    //  inputs : [u,v,w]VelocitySIVBC, [u,v,w]VelCoefPBLM, 
    //           [u,v,w]VelConvCoefPBLM, [u,v,w]VelLinSrcPBLM, 
    //           [u,v,w]VelNonLinSrcPBLM
    //  outputs: [u,v,w]VelLinSrcPBLM, [u,v,w]VelNonLinSrcPBLM
    d_source->modifyVelMassSource(pc, patch, old_dw, new_dw, delta_t, index,
				  Arches::PRESSURE, &pressureVars);

    // Calculate Velocity diagonal
    //  inputs : [u,v,w]VelCoefPBLM, [u,v,w]VelLinSrcPBLM
    //  outputs: [u,v,w]VelCoefPBLM
    d_discretize->calculateVelDiagonal(pc, patch, old_dw, new_dw, 
				       index,
				       Arches::PRESSURE, &pressureVars);
#ifdef ARCHES_PRES_DEBUG
    std::cerr << "Done building matrix for press coeff" << endl;
#endif

  }
  // put required vars
  for (int ii = 0; ii < nofStencils; ii++) {
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

//****************************************************************************
// Actually build of linear matrix
//****************************************************************************
void 
PressureSolver::buildLinearMatrixPress(const ProcessorGroup* pc,
				       const Patch* patch,
				       DataWarehouseP& old_dw,
				       DataWarehouseP& new_dw,
				       double delta_t)
{
   ArchesVariables pressureVars;
  // compute all three componenets of velocity stencil coefficients
  int matlIndex = 0;
  int numGhostCells = 1;
  int zeroGhostCells = 0;
  int nofStencils = 7;
  // Get the reference density
  // Get the required data
  new_dw->get(pressureVars.density, d_lab->d_densityINLabel, 
	      matlIndex, patch, Ghost::AroundCells, numGhostCells);
  new_dw->get(pressureVars.viscosity, d_lab->d_viscosityINLabel, 
	      matlIndex, patch, Ghost::AroundCells, numGhostCells);
  new_dw->get(pressureVars.pressure, d_lab->d_pressureINLabel, 
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
  new_dw->get(pressureVars.uVelocity, d_lab->d_uVelocitySIVBCLabel, 
	      matlIndex, patch, Ghost::AroundCells, numGhostCells+1);
  new_dw->get(pressureVars.vVelocity, d_lab->d_vVelocitySIVBCLabel, 
	      matlIndex, patch, Ghost::AroundCells, numGhostCells+1);
  new_dw->get(pressureVars.wVelocity, d_lab->d_wVelocitySIVBCLabel, 
	      matlIndex, patch, Ghost::AroundCells, numGhostCells+1);
  // *** warning fix it
  old_dw->get(pressureVars.old_density, d_lab->d_densityINLabel, 
		matlIndex, patch, Ghost::None, zeroGhostCells);
  old_dw->get(pressureVars.cellType, d_lab->d_cellTypeLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
  
  for (int ii = 0; ii < nofStencils; ii++) {
    new_dw->get(pressureVars.uVelocityCoeff[ii], 
		d_lab->d_uVelCoefPBLMLabel, ii, patch,
		Ghost::AroundCells, numGhostCells);
    new_dw->get(pressureVars.vVelocityCoeff[ii], 
		d_lab->d_vVelCoefPBLMLabel, ii, patch,
		Ghost::AroundCells, numGhostCells);
    new_dw->get(pressureVars.wVelocityCoeff[ii], 
		d_lab->d_wVelCoefPBLMLabel, ii, patch,
		Ghost::AroundCells, numGhostCells);
  }
  new_dw->get(pressureVars.uVelNonlinearSrc, 
	      d_lab->d_uVelNonLinSrcPBLMLabel,
	      matlIndex, patch,
	      Ghost::AroundCells, numGhostCells);
  new_dw->get(pressureVars.vVelNonlinearSrc, 
	      d_lab->d_vVelNonLinSrcPBLMLabel, matlIndex, patch,
	      Ghost::AroundCells, numGhostCells);
  new_dw->get(pressureVars.wVelNonlinearSrc,
	      d_lab->d_wVelNonLinSrcPBLMLabel, matlIndex, patch,
	      Ghost::AroundCells, numGhostCells);
 
  // Calculate Pressure Coeffs
  //  inputs : densityIN, pressureIN, [u,v,w]VelCoefPBLM[Arches::AP]
  //  outputs: presCoefPBLM[Arches::AE..AB] 
  for (int ii = 0; ii < nofStencils; ii++)
    new_dw->allocate(pressureVars.pressCoeff[ii], 
			d_lab->d_presCoefPBLMLabel, ii, patch);
  d_discretize->calculatePressureCoeff(pc, patch, old_dw, new_dw, 
				       delta_t, cellinfo, &pressureVars);

  // Calculate Pressure Source
  //  inputs : pressureSPBC, [u,v,w]VelocitySIVBC, densityCP,
  //           [u,v,w]VelCoefPBLM, [u,v,w]VelNonLinSrcPBLM
  //  outputs: presLinSrcPBLM, presNonLinSrcPBLM
  // Allocate space
  new_dw->allocate(pressureVars.pressLinearSrc, 
		      d_lab->d_presLinSrcPBLMLabel, matlIndex, patch);
  new_dw->allocate(pressureVars.pressNonlinearSrc, 
		      d_lab->d_presNonLinSrcPBLMLabel, matlIndex, patch);

  d_source->calculatePressureSource(pc, patch, old_dw, new_dw, delta_t,
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
  for (int ii = 0; ii < nofStencils; ii++) {
    new_dw->put(pressureVars.pressCoeff[ii], d_lab->d_presCoefPBLMLabel, 
		   ii, patch);
  }
  new_dw->put(pressureVars.pressNonlinearSrc, 
	      d_lab->d_presNonLinSrcPBLMLabel, matlIndex, patch);
#ifdef ARCHES_PRES_DEBUG
  std::cerr << "Done building matrix for press coeff" << endl;
#endif

}


void 
PressureSolver::pressureLinearSolve_all (const ProcessorGroup* pg,
					 const Patch*,
					 DataWarehouseP& old_dw,
					 DataWarehouseP& new_dw,
					 LevelP level, SchedulerP sched)
{
  ArchesVariables pressureVars;
  int me = pg->myrank();
  LoadBalancer* lb = sched->getLoadBalancer();

  // initializeMatrix...
  d_linearSolver->matrixCreate(level, lb);
#ifdef ARCHES_PRES_DEBUG
  cerr << "Finished creating petsc matrix\n";
#endif

  for(Level::const_patchIterator iter=level->patchesBegin();
      iter != level->patchesEnd(); iter++){
    const Patch* patch=*iter;
    {
       int proc = lb->getPatchwiseProcessorAssignment(patch, d_myworld);
       if(proc == me){
	  // Underrelax...

	  // This calls fillRows on linear(petsc) solver
#ifdef ARCHES_PRES_DEBUG
	  cerr << "Calling pressureLinearSolve for patch: " << patch->getID() << '\n';
#endif
	  pressureLinearSolve(pg, patch, old_dw, new_dw, pressureVars);
#ifdef ARCHES_PRES_DEBUG
	  cerr << "Done with pressureLinearSolve for patch: " << patch->getID() << '\n';
#endif
       }
    }
  }
  // MPI_Reduce();
  // solve
#ifdef ARCHES_PRES_DEBUG
  cerr << "Calling pressLinearSolve\n";
#endif
  bool converged =  d_linearSolver->pressLinearSolve();
#ifdef ARCHES_PRES_DEBUG
  cerr << "Done with pressLinearSolve\n";
#endif
  int pressRefProc = -1;
  for(Level::const_patchIterator iter=level->patchesBegin();
      iter != level->patchesEnd(); iter++){
    const Patch* patch=*iter;
    {
      int proc = lb->getPatchwiseProcessorAssignment(patch, d_myworld);
      //int proc = find_processor_assignment(patch);
      if (converged) {
	if(proc == me){
	  //	  unpack from linear solver.
	  d_linearSolver->copyPressSoln(patch, &pressureVars);
#ifdef ARCHES_PRES_DEBUG
	  cerr << "Calling normPressure for patch: " << patch->getID() << '\n';
#endif
	}
      }
      else {
	if ((proc == me)&&(me==0))
	  cerr << "pressure solver not converged, using old values" << endl;
      }
      if (patch->containsCell(d_pressRef)) {
	pressRefProc = proc;
	if(pressRefProc == me){
	  pressureVars.press_ref = pressureVars.pressure[d_pressRef];
	  cerr << "press_ref for norm: " << pressureVars.press_ref << " " <<
	    pressRefProc << endl;
	}
      }
    }
  }
  if(pressRefProc == -1)
    throw InternalError("Patch containing pressure reference point was not found");
  
  MPI_Bcast(&pressureVars.press_ref, 1, MPI_DOUBLE, pressRefProc, pg->getComm());
  for(Level::const_patchIterator iter=level->patchesBegin();
      iter != level->patchesEnd(); iter++){
    const Patch* patch=*iter;
    {
#ifdef ARCHES_PRES_DEBUG
	 cerr << "After presssoln" << endl;
#endif
	 int proc = lb->getPatchwiseProcessorAssignment(patch, d_myworld);
	 //int proc = find_processor_assignment(patch);
	 if(proc == me){
#ifdef ARCHES_PRES_DEBUG
	   for(CellIterator iter = patch->getCellIterator();
	       !iter.done(); iter++){
	     cerr.width(10);
	     cerr << "press"<<*iter << ": " << pressureVars.pressure[*iter] << "\n" ; 
	   }
#endif
	   normPressure(pg, patch, &pressureVars);
#ifdef ARCHES_PRES_DEBUG
	   cerr << "Done with normPressure for patch: " 
		<< patch->getID() << '\n';
#endif
	 // put back the results
	   int matlIndex = 0;
	   new_dw->put(pressureVars.pressure, d_lab->d_pressurePSLabel, 
		       matlIndex, patch);
	 }
    }
  }

  // destroy matrix
  d_linearSolver->destroyMatrix();
}

// Actual linear solve
void 
PressureSolver::pressureLinearSolve (const ProcessorGroup* pc,
				     const Patch* patch,
				     DataWarehouseP& old_dw,
				     DataWarehouseP& new_dw,
				     ArchesVariables& pressureVars)
{
  int matlIndex = 0;
  int numGhostCells = 1;
  int zeroGhostCells = 0;
  int nofStencils = 7;
  // Get the required data
  new_dw->get(pressureVars.pressure, d_lab->d_pressureINLabel, 
	      matlIndex, patch, Ghost::AroundCells, numGhostCells);
  for (int ii = 0; ii < nofStencils; ii++) 
    new_dw->get(pressureVars.pressCoeff[ii], d_lab->d_presCoefPBLMLabel, 
		   ii, patch, Ghost::None, zeroGhostCells);

  new_dw->get(pressureVars.pressNonlinearSrc, 
		 d_lab->d_presNonLinSrcPBLMLabel, 
		 matlIndex, patch, Ghost::None, zeroGhostCells);

  // compute eqn residual, L1 norm
  new_dw->allocate(pressureVars.residualPressure, d_lab->d_pressureRes,
			  matlIndex, patch);
#if 0
  d_linearSolver->computePressResidual(pc, patch, old_dw, new_dw, 
				       &pressureVars);
#else
  pressureVars.residPress=pressureVars.truncPress=0;
#endif
  new_dw->put(sum_vartype(pressureVars.residPress), d_lab->d_presResidPSLabel);
  new_dw->put(sum_vartype(pressureVars.truncPress), d_lab->d_presTruncPSLabel);
  // apply underelaxation to eqn
  d_linearSolver->computePressUnderrelax(pc, patch, old_dw, new_dw,
					 &pressureVars);
  // put back computed matrix coeffs and nonlinear source terms 
  // modified as a result of underrelaxation 
  // into the matrix datawarehouse
#if 0
  for (int ii = 0; ii < nofStencils; ii++) {
    new_dw->put(pressureVars.pressCoeff[ii], d_lab->d_presCoefPSLabel, ii, patch);
  }
  new_dw->put(pressureVars.pressNonLinSrc, d_lab->d_presNonLinSrcPSLabel, 
	      matlIndex, patch);
#endif

  // for parallel code lisolve will become a recursive task and 
  // will make the following subroutine separate
  // get patch numer ***warning****
  // sets matrix
  d_linearSolver->setPressMatrix(pc, patch, old_dw, new_dw, &pressureVars, d_lab);
  //  d_linearSolver->pressLinearSolve();

}
  
  

//****************************************************************************
// normalize the pressure solution
//****************************************************************************
void 
PressureSolver::normPressure(const ProcessorGroup*,
			     const Patch* patch,
			     ArchesVariables* vars)
{
  IntVector domLo = vars->pressure.getFortLowIndex();
  IntVector domHi = vars->pressure.getFortHighIndex();
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();
  double pressref = vars->press_ref;
  //  double pressref = 0.0;
  FORT_NORMPRESS(domLo.get_pointer(),domHi.get_pointer(),
		idxLo.get_pointer(), idxHi.get_pointer(),
		vars->pressure.getPointer(), 
		&pressref);

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

