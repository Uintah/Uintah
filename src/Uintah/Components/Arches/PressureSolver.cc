//----- PressureSolver.cc ----------------------------------------------

/* REFERENCED */
static char *id="@(#) $Id$";

#include <Uintah/Components/Arches/PressureSolver.h>
#include <Uintah/Components/Arches/Discretization.h>
#include <Uintah/Components/Arches/Source.h>
#include <Uintah/Components/Arches/BoundaryCondition.h>
#include <Uintah/Components/Arches/TurbulenceModel.h>
#include <Uintah/Components/Arches/PhysicalConstants.h>
#include <Uintah/Components/Arches/RBGSSolver.h>
#include <Uintah/Components/Arches/ArchesVariables.h>
#include <Uintah/Exceptions/InvalidValue.h>
#include <Uintah/Interface/Scheduler.h>
#include <Uintah/Interface/DataWarehouse.h>
#include <Uintah/Interface/ProblemSpec.h>
#include <Uintah/Grid/Level.h>
#include <Uintah/Grid/Task.h>
#include <Uintah/Grid/CCVariable.h>
#include <Uintah/Grid/SFCXVariable.h>
#include <Uintah/Grid/SFCYVariable.h>
#include <Uintah/Grid/SFCZVariable.h>
#include <SCICore/Util/NotFinished.h>
#include <Uintah/Components/Arches/Arches.h>

using namespace Uintah::ArchesSpace;
using namespace std;

//****************************************************************************
// Default constructor for PressureSolver
//****************************************************************************
PressureSolver::PressureSolver(const ArchesLabel* label,
			       TurbulenceModel* turb_model,
			       BoundaryCondition* bndry_cond,
			       PhysicalConstants* physConst): 
                                      d_lab(label),
                                     d_turbModel(turb_model), 
                                     d_boundaryCondition(bndry_cond),
				     d_physicalConsts(physConst)
{
  d_pressureVars = scinew ArchesVariables();

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
  //create a new data warehouse to store matrix coeff
  // and source terms. It gets reinitialized after every 
  // pressure solve.
  DataWarehouseP matrix_dw = sched->createDataWarehouse(new_dw);
  

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
  sched_buildLinearMatrix(level, sched, new_dw, matrix_dw, delta_t);

  //residual at the start of linear solve
  // this can be part of linear solver
#if 0
  calculateResidual(level, sched, new_dw, matrix_dw);
  calculateOrderMagnitude(level, sched, new_dw, matrix_dw);
#endif

  // Schedule the pressure solve
  // require : pressureIN, presCoefPBLM, presNonLinSrcPBLM
  // compute : presResidualPS, presCoefPS, presNonLinSrcPS, pressurePS
  //d_linearSolver->sched_pressureSolve(level, sched, new_dw, matrix_dw);
  sched_pressureLinearSolve(level, sched, new_dw, matrix_dw);

  // Schedule Calculation of pressure norm
  // require :
  // compute :
  //sched_normPressure(level, sched, new_dw, matrix_dw);
  sched_normPressure(level, sched, old_dw, new_dw);
  
}

//****************************************************************************
// Schedule build of linear matrix
//****************************************************************************
void 
PressureSolver::sched_buildLinearMatrix(const LevelP& level,
					SchedulerP& sched,
					DataWarehouseP& new_dw,
					DataWarehouseP& matrix_dw,
					double delta_t)
{
  for(Level::const_patchIterator iter=level->patchesBegin();
      iter != level->patchesEnd(); iter++){
    const Patch* patch=*iter;
    {
      Task* tsk = scinew Task("PressureSolver::BuildCoeff",
			   patch, new_dw, matrix_dw, this,
			   &PressureSolver::buildLinearMatrix, delta_t);

      int numGhostCells = 0;
      int matlIndex = 0;
      int nofStencils = 7;
      DataWarehouseP d_old_dw = new_dw->getTop();
      // Requires
      // from old_dw for time integration
      // get old_dw from getTop function
      tsk->requires(d_old_dw, d_lab->d_cellTypeLabel, matlIndex, patch, 
		    Ghost::None, numGhostCells);
      tsk->requires(d_old_dw, d_lab->d_densityCPLabel, matlIndex, patch, 
		    Ghost::None, numGhostCells);
      tsk->requires(d_old_dw, d_lab->d_uVelocitySPBCLabel, matlIndex, patch, 
		    Ghost::None, numGhostCells);
      tsk->requires(d_old_dw, d_lab->d_vVelocitySPBCLabel, matlIndex, patch, 
		    Ghost::None, numGhostCells);
      tsk->requires(d_old_dw, d_lab->d_wVelocitySPBCLabel, matlIndex, patch, 
		    Ghost::None, numGhostCells);
      // from new_dw
      tsk->requires(new_dw, d_lab->d_pressureINLabel, matlIndex, patch, 
		    Ghost::None, numGhostCells);
      tsk->requires(new_dw, d_lab->d_densityINLabel, matlIndex, patch, 
		    Ghost::None, numGhostCells);
      tsk->requires(new_dw, d_lab->d_viscosityINLabel, matlIndex, patch, 
		    Ghost::None, numGhostCells);
      tsk->requires(new_dw, d_lab->d_uVelocitySIVBCLabel, matlIndex, patch, 
		    Ghost::None, numGhostCells);
      tsk->requires(new_dw, d_lab->d_vVelocitySIVBCLabel, matlIndex, patch, 
		    Ghost::None, numGhostCells);
      tsk->requires(new_dw, d_lab->d_wVelocitySIVBCLabel, matlIndex, patch, 
		    Ghost::None, numGhostCells);

      /// requires convection coeff because of the nodal
      // differencing
      // computes all the components of velocity
      for (int ii = 0; ii < nofStencils; ii++) {
	tsk->computes(matrix_dw, d_lab->d_uVelConvCoefPBLMLabel, ii, patch);
	tsk->computes(matrix_dw, d_lab->d_vVelConvCoefPBLMLabel, ii, patch);
	tsk->computes(matrix_dw, d_lab->d_wVelConvCoefPBLMLabel, ii, patch);
	tsk->computes(matrix_dw, d_lab->d_uVelCoefPBLMLabel, ii, patch);
	tsk->computes(matrix_dw, d_lab->d_vVelCoefPBLMLabel, ii, patch);
	tsk->computes(matrix_dw, d_lab->d_wVelCoefPBLMLabel, ii, patch);
	tsk->computes(matrix_dw, d_lab->d_presCoefPBLMLabel, ii, patch);
      }

      tsk->computes(matrix_dw, d_lab->d_uVelLinSrcPBLMLabel, matlIndex, patch);
      tsk->computes(matrix_dw, d_lab->d_vVelLinSrcPBLMLabel, matlIndex, patch);
      tsk->computes(matrix_dw, d_lab->d_wVelLinSrcPBLMLabel, matlIndex, patch);
      tsk->computes(matrix_dw, d_lab->d_uVelNonLinSrcPBLMLabel, matlIndex, patch);
      tsk->computes(matrix_dw, d_lab->d_vVelNonLinSrcPBLMLabel, matlIndex, patch);
      tsk->computes(matrix_dw, d_lab->d_wVelNonLinSrcPBLMLabel, matlIndex, patch);
      tsk->computes(matrix_dw, d_lab->d_presNonLinSrcPBLMLabel, matlIndex, patch);
     
      sched->addTask(tsk);
    }

  }
}


//****************************************************************************
// Schedule solver for linear matrix
//****************************************************************************
void 
PressureSolver::sched_pressureLinearSolve(const LevelP& level,
					  SchedulerP& sched,
					  DataWarehouseP& new_dw,
					  DataWarehouseP& matrix_dw)
{
  for(Level::const_patchIterator iter=level->patchesBegin();
      iter != level->patchesEnd(); iter++){
    const Patch* patch=*iter;
    {
      Task* tsk = scinew Task("PressureSolver::PressLinearSolve",
			   patch, new_dw, matrix_dw, this,
			   &PressureSolver::pressureLinearSolve);

      int numGhostCells = 0;
      int matlIndex = 0;
      int nofStencils = 7;
      // Requires
      // coefficient for the variable for which solve is invoked
      tsk->requires(new_dw, d_lab->d_pressureINLabel, matlIndex, patch, 
		    Ghost::None, numGhostCells);
      for (int ii = 0; ii < nofStencils; ii++)
	tsk->requires(matrix_dw, d_lab->d_presCoefPBLMLabel, ii, patch, 
		      Ghost::None, numGhostCells);
      tsk->requires(matrix_dw, d_lab->d_presNonLinSrcPBLMLabel, matlIndex, patch, 
		    Ghost::None, numGhostCells);
      // computes global residual
      //      tsk->computes(new_dw, d_lab->d_presResidPSLabel, matlIndex, patch);
      //      tsk->computes(new_dw, d_lab->d_presTruncPSLabel, matlIndex, patch);
      tsk->computes(new_dw, d_lab->d_pressurePSLabel, matlIndex, patch);

      sched->addTask(tsk);
    }
  }
}


//****************************************************************************
// Actually build of linear matrix
//****************************************************************************
void 
PressureSolver::buildLinearMatrix(const ProcessorGroup* pc,
				  const Patch* patch,
				  DataWarehouseP& new_dw,
				  DataWarehouseP& matrix_dw,
				  double delta_t)
{
  // compute all three componenets of velocity stencil coefficients
  int matlIndex = 0;
  int numGhostCells = 0;
  int nofStencils = 7;
  DataWarehouseP d_old_dw = new_dw->getTop();
  // Get the required data
  new_dw->get(d_pressureVars->density, d_lab->d_densityINLabel, 
	      matlIndex, patch, Ghost::None, numGhostCells);
  new_dw->get(d_pressureVars->viscosity, d_lab->d_viscosityINLabel, 
	      matlIndex, patch, Ghost::None, numGhostCells);
  new_dw->get(d_pressureVars->pressure, d_lab->d_pressureINLabel, 
	      matlIndex, patch, Ghost::None, numGhostCells);
  // Get the PerPatch CellInformation data
  PerPatch<CellInformation*> cellInfoP;
  d_old_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
  //  old_dw->get(cellInfoP, d_cellInfoLabel, matlIndex, patch);
  //  if (old_dw->exists(d_cellInfoLabel, patch)) 
  //  old_dw->get(cellInfoP, d_cellInfoLabel, matlIndex, patch);
  //else {
  //  cellInfoP.setData(scinew CellInformation(patch));
  //  old_dw->put(cellInfoP, d_cellInfoLabel, matlIndex, patch);
  //}
  CellInformation* cellinfo = cellInfoP;
  new_dw->get(d_pressureVars->uVelocity, d_lab->d_uVelocitySIVBCLabel, 
	      matlIndex, patch, Ghost::None, numGhostCells);
  new_dw->get(d_pressureVars->vVelocity, d_lab->d_vVelocitySIVBCLabel, 
	      matlIndex, patch, Ghost::None, numGhostCells);
  new_dw->get(d_pressureVars->wVelocity, d_lab->d_wVelocitySIVBCLabel, 
	      matlIndex, patch, Ghost::None, numGhostCells);
  d_old_dw->get(d_pressureVars->old_uVelocity, d_lab->d_uVelocitySPBCLabel, 
		matlIndex, patch, Ghost::None, numGhostCells);
  d_old_dw->get(d_pressureVars->old_vVelocity, d_lab->d_vVelocitySPBCLabel, 
		matlIndex, patch, Ghost::None, numGhostCells);
  d_old_dw->get(d_pressureVars->old_wVelocity, d_lab->d_wVelocitySPBCLabel, 
		matlIndex, patch, Ghost::None, numGhostCells);
  d_old_dw->get(d_pressureVars->old_density, d_lab->d_densityCPLabel, 
		matlIndex, patch, Ghost::None, numGhostCells);
  d_old_dw->get(d_pressureVars->cellType, d_lab->d_cellTypeLabel, 
		matlIndex, patch, Ghost::None, numGhostCells);
  
  for(int index = 1; index <= Arches::NDIM; ++index) {
    switch (index) {
    case Arches::XDIR:
      matrix_dw->allocate(d_pressureVars->variableCalledDU, d_lab->d_DUPBLMLabel,
			  matlIndex, patch);
      break;
    case Arches::YDIR:
      matrix_dw->allocate(d_pressureVars->variableCalledDV, d_lab->d_DUPBLMLabel,
			  matlIndex, patch);
      break;
    case Arches::ZDIR:
      matrix_dw->allocate(d_pressureVars->variableCalledDW, d_lab->d_DWPBLMLabel,
			  matlIndex, patch);
      break;
    default:
      throw InvalidValue("invalid index for velocity in PressureSolver"); 
    }

    for (int ii = 0; ii < nofStencils; ii++) {
      switch(index) {
      case Arches::XDIR:
	matrix_dw->allocate(d_pressureVars->uVelocityCoeff[ii], 
			    d_lab->d_uVelCoefPBLMLabel, ii, patch);
	matrix_dw->allocate(d_pressureVars->uVelocityConvectCoeff[ii], 
			    d_lab->d_uVelConvCoefPBLMLabel, ii, patch);
	break;
      case Arches::YDIR:
	matrix_dw->allocate(d_pressureVars->vVelocityCoeff[ii], 
			    d_lab->d_vVelCoefPBLMLabel, ii, patch);
	matrix_dw->allocate(d_pressureVars->vVelocityConvectCoeff[ii],
			    d_lab->d_vVelConvCoefPBLMLabel, ii, patch);
	break;
      case Arches::ZDIR:
	matrix_dw->allocate(d_pressureVars->wVelocityCoeff[ii], 
			    d_lab->d_wVelCoefPBLMLabel, ii, patch);
	matrix_dw->allocate(d_pressureVars->wVelocityConvectCoeff[ii], 
			    d_lab->d_wVelConvCoefPBLMLabel, ii, patch);
	break;
      default:
	throw InvalidValue("invalid index for velocity in PressureSolver");
      }
    }
    // Calculate Velocity Coeffs :
    //  inputs : [u,v,w]VelocitySIVBC, densityIN, viscosityIN
    //  outputs: [u,v,w]VelCoefPBLM, [u,v,w]VelConvCoefPBLM 
    d_discretize->calculateVelocityCoeff(pc, patch, new_dw, matrix_dw, 
					 delta_t, index, 
					 Arches::PRESSURE, 
					 cellinfo, d_pressureVars);
    // Calculate Velocity source
    //  inputs : [u,v,w]VelocitySIVBC, densityIN, viscosityIN
    //  outputs: [u,v,w]VelLinSrcPBLM, [u,v,w]VelNonLinSrcPBLM
    // get data
    // allocate
    switch(index) {
    case Arches::XDIR:
      matrix_dw->allocate(d_pressureVars->uVelLinearSrc, 
			  d_lab->d_uVelLinSrcPBLMLabel, matlIndex, patch);
      matrix_dw->allocate(d_pressureVars->uVelNonlinearSrc, 
			  d_lab->d_uVelNonLinSrcPBLMLabel,
			  matlIndex, patch);
      break;
    case Arches::YDIR:
      matrix_dw->allocate(d_pressureVars->vVelLinearSrc, 
			  d_lab->d_vVelLinSrcPBLMLabel, matlIndex, patch);
      matrix_dw->allocate(d_pressureVars->vVelNonlinearSrc, 
			  d_lab->d_vVelNonLinSrcPBLMLabel, matlIndex, patch);
      break;
    case Arches::ZDIR:
      matrix_dw->allocate(d_pressureVars->wVelLinearSrc, 
			  d_lab->d_wVelLinSrcPBLMLabel, matlIndex, patch);
      matrix_dw->allocate(d_pressureVars->wVelNonlinearSrc,
			  d_lab->d_wVelNonLinSrcPBLMLabel, matlIndex, patch);
      break;
    default:
      throw InvalidValue("Invalid index in PressureSolver for calcVelSrc");
    }
    d_source->calculateVelocitySource(pc, patch, new_dw, matrix_dw, 
				      delta_t, index,
				      Arches::PRESSURE,
				      cellinfo, d_pressureVars);
    // Calculate the Velocity BCS
    //  inputs : densityCP, [u,v,w]VelocitySIVBC, [u,v,w]VelCoefPBLM
    //           [u,v,w]VelLinSrcPBLM, [u,v,w]VelNonLinSrcPBLM
    //  outputs: [u,v,w]VelCoefPBLM, [u,v,w]VelLinSrcPBLM, 
    //           [u,v,w]VelNonLinSrcPBLM
    
    d_boundaryCondition->velocityBC(pc, patch, new_dw, matrix_dw, 
				    index,
				    Arches::PRESSURE, cellinfo, d_pressureVars);

    // Modify Velocity Mass Source
    //  inputs : [u,v,w]VelocitySIVBC, [u,v,w]VelCoefPBLM, 
    //           [u,v,w]VelConvCoefPBLM, [u,v,w]VelLinSrcPBLM, 
    //           [u,v,w]VelNonLinSrcPBLM
    //  outputs: [u,v,w]VelLinSrcPBLM, [u,v,w]VelNonLinSrcPBLM
    d_source->modifyVelMassSource(pc, patch, new_dw, matrix_dw, delta_t, index,
				  Arches::PRESSURE, d_pressureVars);

    // Calculate Velocity diagonal
    //  inputs : [u,v,w]VelCoefPBLM, [u,v,w]VelLinSrcPBLM
    //  outputs: [u,v,w]VelCoefPBLM
    d_discretize->calculateVelDiagonal(pc, patch, new_dw, matrix_dw, 
				       index,
				       Arches::PRESSURE, d_pressureVars);
    std::cerr << "Done building matrix for press coeff" << endl;
  }

  // Calculate Pressure Coeffs
  //  inputs : densityIN, pressureIN, [u,v,w]VelCoefPBLM[Arches::AP]
  //  outputs: presCoefPBLM[Arches::AE..AB] 
  for (int ii = 0; ii < nofStencils; ii++) 
    matrix_dw->allocate(d_pressureVars->pressCoeff[ii], 
			d_lab->d_presCoefPBLMLabel, ii, patch);
  
  d_discretize->calculatePressureCoeff(pc, patch, new_dw, matrix_dw, 
				       delta_t, cellinfo, d_pressureVars);

  // Calculate Pressure Source
  //  inputs : pressureSPBC, [u,v,w]VelocitySIVBC, densityCP,
  //           [u,v,w]VelCoefPBLM, [u,v,w]VelNonLinSrcPBLM
  //  outputs: presLinSrcPBLM, presNonLinSrcPBLM
  // Allocate space
  matrix_dw->allocate(d_pressureVars->pressLinearSrc, 
		      d_lab->d_presLinSrcPBLMLabel, matlIndex, patch);
  matrix_dw->allocate(d_pressureVars->pressNonlinearSrc, 
		      d_lab->d_presNonLinSrcPBLMLabel, matlIndex, patch);

  d_source->calculatePressureSource(pc, patch, new_dw, matrix_dw, delta_t,
				    cellinfo, d_pressureVars);

  // Calculate Pressure BC
  //  inputs : pressureIN, presCoefPBLM
  //  outputs: presCoefPBLM
  d_boundaryCondition->pressureBC(pc, patch, new_dw, matrix_dw, 
				  cellinfo, d_pressureVars);

  // Calculate Pressure Diagonal
  //  inputs : presCoefPBLM, presLinSrcPBLM
  //  outputs: presCoefPBLM 
  d_discretize->calculatePressDiagonal(pc, patch, new_dw, matrix_dw, 
				       d_pressureVars);
  
  // put required vars
  for (int ii = 0; ii < nofStencils; ii++) {
    matrix_dw->put(d_pressureVars->pressCoeff[ii], d_lab->d_presCoefPBLMLabel, 
		   ii, patch);
  }
  matrix_dw->put(d_pressureVars->pressNonlinearSrc, 
		 d_lab->d_presNonLinSrcPBLMLabel, matlIndex, patch);
  std::cerr << "Done building matrix for press coeff" << endl;

}


// Actual linear solve
void 
PressureSolver::pressureLinearSolve (const ProcessorGroup* pc,
				     const Patch* patch,
				     DataWarehouseP& new_dw,
				     DataWarehouseP& matrix_dw)
{
  int matlIndex = 0;
  int numGhostCells = 0;
  int nofStencils = 7;
  // Get the required data
  new_dw->get(d_pressureVars->pressure, d_lab->d_pressureINLabel, 
	      matlIndex, patch, Ghost::None, numGhostCells);
  for (int ii = 0; ii < nofStencils; ii++) 
    matrix_dw->get(d_pressureVars->pressCoeff[ii], d_lab->d_presCoefPBLMLabel, 
		   ii, patch, Ghost::None, numGhostCells);
  matrix_dw->get(d_pressureVars->pressNonlinearSrc, 
		 d_lab->d_presNonLinSrcPBLMLabel, 
		 matlIndex, patch, Ghost::None, numGhostCells);
  // compute eqn residual, L1 norm
  matrix_dw->allocate(d_pressureVars->residualPressure, d_lab->d_pressureRes,
			  matlIndex, patch);
  d_linearSolver->computePressResidual(pc, patch, new_dw, matrix_dw, 
				       d_pressureVars);
  new_dw->put(sum_vartype(d_pressureVars->residPress), d_lab->d_presResidPSLabel);
  new_dw->put(sum_vartype(d_pressureVars->truncPress), d_lab->d_presTruncPSLabel);
  // apply underelaxation to eqn
  d_linearSolver->computePressUnderrelax(pc, patch, new_dw, matrix_dw,
					 d_pressureVars);
  // put back computes matrix coeffs and nonlinear source terms 
  // modified as a result of underrelaxation 
  // into the matrix datawarehouse
#if 0
  for (int ii = 0; ii < nofStencils; ii++) {
    matrix_dw->put(d_pressureVars->pressCoeff[ii], d_lab->d_presCoefPSLabel, ii, patch);
  }
  matrix_dw->put(d_pressureVars->pressNonLinSrc, d_lab->d_presNonLinSrcPSLabel, 
		 matlIndex, patch);
#endif
  // for parallel code lisolve will become a recursive task and 
  // will make the following subroutine separate
  d_linearSolver->pressLisolve(pc, patch, new_dw, matrix_dw, d_pressureVars);
  // put back the results
  new_dw->put(d_pressureVars->pressure, d_lab->d_pressurePSLabel, 
	      matlIndex, patch);
}
  
  

//****************************************************************************
// Schedule the creation of the .. more documentation here
//****************************************************************************
void 
PressureSolver::sched_normPressure(const LevelP& ,
		                   SchedulerP& ,
		                   DataWarehouseP& ,
		                   DataWarehouseP& )
{
}  

//****************************************************************************
// Actually do normPressure
//****************************************************************************
void 
PressureSolver::normPressure(const Patch* ,
	                     SchedulerP& ,
	                     const DataWarehouseP& ,
	                     DataWarehouseP& )
{
}

//
// $Log$
// Revision 1.42  2000/08/01 23:28:43  skumar
// Added residual calculation procedure and modified templates in linear
// solver.  Added template for order-of-magnitude term calculation.
//
// Revision 1.41  2000/08/01 06:18:38  bbanerje
// Made ScalarSolver similar to PressureSolver and MomentumSolver.
//
// Revision 1.40  2000/07/28 02:31:00  rawat
// moved all the labels in ArchesLabel. fixed some bugs and added matrix_dw to store matrix
// coeffecients
//
// Revision 1.37  2000/07/17 22:06:58  rawat
// modified momentum source
//
// Revision 1.36  2000/07/14 03:45:46  rawat
// completed velocity bc and fixed some bugs
//
// Revision 1.35  2000/07/13 06:32:10  bbanerje
// Labels are once more consistent for one iteration.
//
// Revision 1.34  2000/07/12 23:59:21  rawat
// added wall bc for u-velocity
//
// Revision 1.33  2000/07/12 22:15:02  bbanerje
// Added pressure Coef .. will do until Kumar's code is up and running
//
// Revision 1.32  2000/07/12 07:35:46  bbanerje
// Added stuff for mascal : Rawat: Labels and dataWarehouse in velsrc need to be corrected.
//
// Revision 1.31  2000/07/11 15:46:28  rawat
// added setInitialGuess in PicardNonlinearSolver and also added uVelSrc
//
// Revision 1.30  2000/07/08 23:42:55  bbanerje
// Moved all enums to Arches.h and made corresponding changes.
//
// Revision 1.29  2000/07/08 08:03:34  bbanerje
// Readjusted the labels upto uvelcoef, removed bugs in CellInformation,
// made needed changes to uvelcoef.  Changed from StencilMatrix::AE etc
// to Arches::AE .. doesn't like enums in templates apparently.
//
// Revision 1.28  2000/07/07 23:07:45  rawat
// added inlet bc's
//
// Revision 1.27  2000/07/03 05:30:15  bbanerje
// Minor changes for inlbcs dummy code to compile and work. densitySIVBC is no more.
//
// Revision 1.26  2000/06/29 22:56:43  bbanerje
// Changed FCVars to SFC[X,Y,Z]Vars, and added the neceesary getIndex calls.
//
// Revision 1.25  2000/06/22 23:06:35  bbanerje
// Changed velocity related variables to FCVariable type.
// ** NOTE ** We may need 3 types of FCVariables (one for each direction)
//
// Revision 1.24  2000/06/21 07:51:00  bbanerje
// Corrected new_dw, old_dw problems, commented out intermediate dw (for now)
// and made the stuff go through schedule_time_advance.
//
// Revision 1.23  2000/06/21 06:49:21  bbanerje
// Straightened out some of the problems in data location .. still lots to go.
//
// Revision 1.22  2000/06/21 06:12:12  bbanerje
// Added missing VarLabel* mallocs .
//
// Revision 1.21  2000/06/18 01:20:16  bbanerje
// Changed names of varlabels in source to reflect the sequence of tasks.
// Result : Seg Violation in addTask in MomentumSolver
//
// Revision 1.20  2000/06/17 07:06:25  sparker
// Changed ProcessorContext to ProcessorGroup
//
// Revision 1.19  2000/06/16 04:25:40  bbanerje
// Uncommented BoundaryCondition related stuff.
//
// Revision 1.18  2000/06/14 20:40:49  rawat
// modified boundarycondition for physical boundaries and
// added CellInformation class
//
// Revision 1.17  2000/06/07 06:13:55  bbanerje
// Changed CCVariable<Vector> to CCVariable<double> for most cases.
// Some of these variables may not be 3D Vectors .. they may be Stencils
// or more than 3D arrays. Need help here.
//
// Revision 1.15  2000/06/04 23:57:46  bbanerje
// Updated Arches to do ScheduleTimeAdvance.
//
// Revision 1.14  2000/06/04 22:40:14  bbanerje
// Added Cocoon stuff, changed task, require, compute, get, put arguments
// to reflect new declarations. Changed sub.mk to include all the new files.
//
//
