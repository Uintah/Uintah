//----- MomentumSolver.cc ----------------------------------------------

/* REFERENCED */
static char *id="@(#) $Id$";

#include <Uintah/Components/Arches/MomentumSolver.h>
#include <Uintah/Components/Arches/RBGSSolver.h>
#include <Uintah/Components/Arches/Discretization.h>
#include <Uintah/Components/Arches/Source.h>
#include <Uintah/Components/Arches/BoundaryCondition.h>
#include <Uintah/Components/Arches/TurbulenceModel.h>
#include <Uintah/Components/Arches/PhysicalConstants.h>
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
// Default constructor for MomentumSolver
//****************************************************************************
MomentumSolver::
MomentumSolver(const ArchesLabel* label, TurbulenceModel* turb_model,
	       BoundaryCondition* bndry_cond,
	       PhysicalConstants* physConst) : 
                                   d_lab(label),
                                   d_turbModel(turb_model), 
                                   d_boundaryCondition(bndry_cond),
				   d_physicalConsts(physConst)
{
}

//****************************************************************************
// Destructor
//****************************************************************************
MomentumSolver::~MomentumSolver()
{
}

//****************************************************************************
// Problem Setup 
//****************************************************************************
void 
MomentumSolver::problemSetup(const ProblemSpecP& params)
{
  ProblemSpecP db = params->findBlock("MomentumSolver");
  string finite_diff;
  db->require("finite_difference", finite_diff);
  if (finite_diff == "second") 
    d_discretize = scinew Discretization();
  else {
    throw InvalidValue("Finite Differencing scheme "
		       "not supported: " + finite_diff);
  }
  // make source and boundary_condition objects
  d_source = scinew Source(d_turbModel, d_physicalConsts);
  string linear_sol;
  db->require("linear_solver", linear_sol);
  if (linear_sol == "linegs")
    d_linearSolver = scinew RBGSSolver();
  else {
    throw InvalidValue("linear solver option"
		       " not supported" + linear_sol);
  }
  d_linearSolver->problemSetup(db);
}

//****************************************************************************
// Schedule linear momentum solve
//****************************************************************************
void 
MomentumSolver::solve(const LevelP& level,
		      SchedulerP& sched,
		      DataWarehouseP& old_dw,
		      DataWarehouseP& new_dw,
		      double /*time*/, double delta_t, int index)
{
  //create a new data warehouse to store matrix coeff
  // and source terms. It gets reinitialized after every 
  // velocity solve.
  //  DataWarehouseP matrix_dw = sched->createDataWarehouse(new_dw);

  //computes stencil coefficients and source terms
  // require : pressureCPBC, [u,v,w]VelocityCPBC, densityIN, viscosityIN (new_dw)
  //           [u,v,w]SPBC, densityCP (old_dw)
  // compute : [u,v,w]VelCoefPBLM, [u,v,w]VelConvCoefPBLM
  //           [u,v,w]VelLinSrcPBLM, [u,v,w]VelNonLinSrcPBLM
  sched_buildLinearMatrix(level, sched, old_dw, new_dw, delta_t, index);
    
  // Schedules linear velocity solve
  // require : [u,v,w]VelocityCPBC, [u,v,w]VelCoefPBLM,
  //           [u,v,w]VelLinSrcPBLM, [u,v,w]VelNonLinSrcPBLM
  // compute : [u,v,w]VelResidualMS, [u,v,w]VelCoefMS, 
  //           [u,v,w]VelNonLinSrcMS, [u,v,w]VelLinSrcMS,
  //           [u,v,w]VelocityMS
  //  d_linearSolver->sched_velSolve(level, sched, new_dw, matrix_dw, index);
  sched_velocityLinearSolve(level, sched, old_dw, new_dw, delta_t, index);
    
}

//****************************************************************************
// Schedule the build of the linear matrix
//****************************************************************************
void 
MomentumSolver::sched_buildLinearMatrix(const LevelP& level,
					SchedulerP& sched,
					DataWarehouseP& old_dw,
					DataWarehouseP& new_dw,
					double delta_t, int index)
{
  for(Level::const_patchIterator iter=level->patchesBegin();
      iter != level->patchesEnd(); iter++){
    const Patch* patch=*iter;
    {
      // steve: requires two arguments
      // Task* tsk = scinew Task("MomentumSolver::BuildCoeff",
	// 		      patch, old_dw, new_dw, this,
	// 		      Discretization::buildLinearMatrix,
	// 		      delta_t, index);
      Task* tsk = scinew Task("MomentumSolver::BuildCoeff",
			      patch, old_dw, new_dw, this,
			      &MomentumSolver::buildLinearMatrix,
			      delta_t, index);

      int numGhostCells = 1;
      int zeroGhostCells = 0;
      int matlIndex = 0;
      int nofStencils = 7;
      // get old_dw from sched
      // from old_dw for time integration
      //      DataWarehouseP old_dw = new_dw->getTop();
      tsk->requires(old_dw, d_lab->d_densityCPLabel, matlIndex, patch, 
		    Ghost::None, zeroGhostCells);
      tsk->requires(old_dw, d_lab->d_uVelocitySPBCLabel, matlIndex, patch, 
		    Ghost::None, zeroGhostCells);
      tsk->requires(old_dw, d_lab->d_vVelocitySPBCLabel, matlIndex, patch, 
		    Ghost::None, zeroGhostCells);
      tsk->requires(old_dw, d_lab->d_wVelocitySPBCLabel, matlIndex, patch, 
		    Ghost::None, zeroGhostCells);
      tsk->requires(old_dw, d_lab->d_cellTypeLabel, matlIndex, patch, 
		    Ghost::AroundCells, numGhostCells);

      // from new_dw
      tsk->requires(new_dw, d_lab->d_densityINLabel, matlIndex, patch, 
		    Ghost::AroundCells, numGhostCells+1);
      tsk->requires(new_dw, d_lab->d_viscosityINLabel, matlIndex, patch, 
		    Ghost::AroundCells, numGhostCells);
      tsk->requires(new_dw, d_lab->d_pressureSPBCLabel, matlIndex, patch, 
		    Ghost::AroundCells, numGhostCells);
      switch (index) {
      case Arches::XDIR:
	tsk->requires(new_dw, d_lab->d_uVelocityCPBCLabel, matlIndex, patch, 
		      Ghost::AroundCells, numGhostCells);
	tsk->requires(new_dw, d_lab->d_vVelocityCPBCLabel, matlIndex, patch, 
		      Ghost::AroundCells, numGhostCells);
	tsk->requires(new_dw, d_lab->d_wVelocityCPBCLabel, matlIndex, patch, 
		      Ghost::AroundCells, numGhostCells);
	break;
      case Arches::YDIR:
	// use new uvelocity for v coef calculation
	tsk->requires(new_dw, d_lab->d_uVelocityCPBCLabel, matlIndex, patch, 
		      Ghost::AroundCells, numGhostCells);
	tsk->requires(new_dw, d_lab->d_vVelocityCPBCLabel, matlIndex, patch, 
		      Ghost::AroundCells, numGhostCells);
	tsk->requires(new_dw, d_lab->d_wVelocityCPBCLabel, matlIndex, patch, 
		      Ghost::AroundCells, numGhostCells);
	break;
      case Arches::ZDIR:
	// use new uvelocity for v coef calculation
	tsk->requires(new_dw, d_lab->d_uVelocityCPBCLabel, matlIndex, patch, 
		      Ghost::AroundCells, numGhostCells);
	tsk->requires(new_dw, d_lab->d_vVelocityCPBCLabel, matlIndex, patch, 
		      Ghost::AroundCells, numGhostCells);
	tsk->requires(new_dw, d_lab->d_wVelocityCPBCLabel, matlIndex, patch, 
		      Ghost::AroundCells, numGhostCells);
	break;
      default:
	throw InvalidValue("Invalid index in MomentumSolver");
      }
	

      /// requires convection coeff because of the nodal
      // differencing
      // computes index components of velocity
      switch (index) {
      case Arches::XDIR:
	for (int ii = 0; ii < nofStencils; ii++) {
	  tsk->computes(new_dw, d_lab->d_uVelCoefMBLMLabel, ii, patch);
	}
	tsk->computes(new_dw, d_lab->d_uVelLinSrcMBLMLabel, 
		      matlIndex, patch);
	tsk->computes(new_dw, d_lab->d_uVelNonLinSrcMBLMLabel, 
		      matlIndex, patch);
	break;
      case Arches::YDIR:
	for (int ii = 0; ii < nofStencils; ii++) {
	  tsk->computes(new_dw, d_lab->d_vVelCoefMBLMLabel, ii, patch);
	}
	tsk->computes(new_dw, d_lab->d_vVelLinSrcMBLMLabel, 
		      matlIndex, patch);
	tsk->computes(new_dw, d_lab->d_vVelNonLinSrcMBLMLabel, 
		      matlIndex, patch);
	break;
      case Arches::ZDIR:
	for (int ii = 0; ii < nofStencils; ii++) {
	  tsk->computes(new_dw, d_lab->d_wVelCoefMBLMLabel, ii, patch);
	}
	tsk->computes(new_dw, d_lab->d_wVelLinSrcMBLMLabel, 
		      matlIndex, patch);
	tsk->computes(new_dw, d_lab->d_wVelNonLinSrcMBLMLabel, 
		      matlIndex, patch);
	break;
      default:
	throw InvalidValue("Invalid index in MomentumSolver");
      }

      sched->addTask(tsk);
    }
  }
}

void
MomentumSolver::sched_velocityLinearSolve(const LevelP& level,
					  SchedulerP& sched,
					  DataWarehouseP& old_dw,
					  DataWarehouseP& new_dw,
					  double delta_t, int index)
{
  for(Level::const_patchIterator iter=level->patchesBegin();
      iter != level->patchesEnd(); iter++){
    const Patch* patch=*iter;
    {
      Task* tsk = scinew Task("MomentumSolver::VelLinearSolve",
			   patch, old_dw, new_dw, this,
			   &MomentumSolver::velocityLinearSolve, delta_t, index);

      int numGhostCells = 1;
      int zeroGhostCells = 0;
      int matlIndex = 0;
      int nofStencils = 7;
      //      DataWarehouseP old_dw = new_dw->getTop();
      tsk->requires(old_dw, d_lab->d_densityCPLabel, matlIndex, patch, 
		    Ghost::None, zeroGhostCells);
      switch(index) {
      case Arches::XDIR:
	tsk->requires(old_dw, d_lab->d_uVelocitySPBCLabel, matlIndex, patch, 
		      Ghost::None, zeroGhostCells);
	// coefficient for the variable for which solve is invoked
	tsk->requires(new_dw, d_lab->d_uVelocityCPBCLabel, matlIndex, patch, 
		      Ghost::AroundCells, numGhostCells);
	for (int ii = 0; ii < nofStencils; ii++) 
	  tsk->requires(new_dw, d_lab->d_uVelCoefMBLMLabel, ii, patch, 
			Ghost::None, zeroGhostCells);
	tsk->requires(new_dw, d_lab->d_uVelNonLinSrcMBLMLabel, 
		      matlIndex, patch, Ghost::None, zeroGhostCells);
	tsk->computes(new_dw, d_lab->d_uVelocitySPBCLabel, matlIndex, patch);
	
	break;
      case Arches::YDIR:
	tsk->requires(old_dw, d_lab->d_vVelocitySPBCLabel, matlIndex, patch, 
		      Ghost::None, zeroGhostCells);
	// coefficient for the variable for which solve is invoked
	tsk->requires(new_dw, d_lab->d_vVelocityCPBCLabel, matlIndex, patch, 
		      Ghost::AroundCells, numGhostCells);
	for (int ii = 0; ii < nofStencils; ii++) 
	  tsk->requires(new_dw, d_lab->d_vVelCoefMBLMLabel, ii, patch, 
			Ghost::None, zeroGhostCells);
	tsk->requires(new_dw, d_lab->d_vVelNonLinSrcMBLMLabel, 
		      matlIndex, patch, Ghost::None, zeroGhostCells);
	tsk->computes(new_dw, d_lab->d_vVelocitySPBCLabel, matlIndex, patch);
	break;
      case Arches::ZDIR:
	tsk->requires(old_dw, d_lab->d_wVelocitySPBCLabel, matlIndex, patch, 
		      Ghost::None, zeroGhostCells);
	// coefficient for the variable for which solve is invoked
	tsk->requires(new_dw, d_lab->d_wVelocityCPBCLabel, matlIndex, patch, 
		      Ghost::AroundCells, numGhostCells);
	for (int ii = 0; ii < nofStencils; ii++) 
	  tsk->requires(new_dw, d_lab->d_wVelCoefMBLMLabel, ii, patch, 
			Ghost::None, zeroGhostCells);
	tsk->requires(new_dw, d_lab->d_wVelNonLinSrcMBLMLabel, 
		      matlIndex, patch, Ghost::None, zeroGhostCells);
	tsk->computes(new_dw, d_lab->d_wVelocitySPBCLabel, matlIndex, patch);
	break;
      default:
	throw InvalidValue("INValid velocity index for linearSolver");
      }
      sched->addTask(tsk);
    }
  }
}
      
//****************************************************************************
// Actual build of the linear matrix
//****************************************************************************
void 
MomentumSolver::buildLinearMatrix(const ProcessorGroup* pc,
				  const Patch* patch,
				  DataWarehouseP& old_dw,
				  DataWarehouseP& new_dw,
				  double delta_t, int index)
{
  ArchesVariables velocityVars;
  int matlIndex = 0;
  int numGhostCells = 1;
  int zeroGhostCells = 0;
  int nofStencils = 7;
  //  DataWarehouseP old_dw = new_dw->getTop();
    // Get the required data
  new_dw->get(velocityVars.pressure, d_lab->d_pressureSPBCLabel, 
	      matlIndex, patch, Ghost::AroundCells, numGhostCells);
  new_dw->get(velocityVars.density, d_lab->d_densityINLabel, 
	      matlIndex, patch, Ghost::AroundCells, numGhostCells+1);
  new_dw->get(velocityVars.viscosity, d_lab->d_viscosityINLabel, 
	      matlIndex, patch, Ghost::AroundCells, numGhostCells);
  // Get the PerPatch CellInformation data
  PerPatch<CellInformation*> cellInfoP;
  // get old_dw from getTop function
  old_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
  //  old_dw->get(cellInfoP, d_cellInfoLabel, matlIndex, patch);
  //  if (old_dw->exists(d_cellInfoLabel, patch)) 
  //  old_dw->get(cellInfoP, d_cellInfoLabel, matlIndex, patch);
  //else {
  //  cellInfoP.setData(scinew CellInformation(patch));
  //  old_dw->put(cellInfoP, d_cellInfoLabel, matlIndex, patch);
  //}
  CellInformation* cellinfo = cellInfoP;
  old_dw->get(velocityVars.old_density, d_lab->d_densityCPLabel, 
	      matlIndex, patch, Ghost::None, zeroGhostCells);
  old_dw->get(velocityVars.cellType, d_lab->d_cellTypeLabel, 
	      matlIndex, patch, Ghost::AroundCells, numGhostCells);
  // for explicit coeffs will be computed using the old u, v, and w
  // change it for implicit solve
  switch (index) {
  case Arches::XDIR:
    new_dw->get(velocityVars.uVelocity, d_lab->d_uVelocityCPBCLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells+1);
    new_dw->get(velocityVars.vVelocity, d_lab->d_vVelocityCPBCLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells+1);
    new_dw->get(velocityVars.wVelocity, d_lab->d_wVelocityCPBCLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells+1);
    old_dw->get(velocityVars.old_uVelocity, d_lab->d_uVelocitySPBCLabel, 
		matlIndex, patch, Ghost::None, zeroGhostCells);
    cerr << "in moment solve just before allocate" << index << endl;
    //    new_dw->allocate(velocityVars.variableCalledDU, d_lab->d_DUMBLMLabel,
    //			matlIndex, patch);

    for (int ii = 0; ii < nofStencils; ii++) {
      new_dw->allocate(velocityVars.uVelocityCoeff[ii], 
			  d_lab->d_uVelCoefMBLMLabel, ii, patch);
      new_dw->allocate(velocityVars.uVelocityConvectCoeff[ii], 
			  d_lab->d_uVelConvCoefMBLMLabel, ii, patch);
    }
    new_dw->allocate(velocityVars.uVelLinearSrc, 
			d_lab->d_uVelLinSrcMBLMLabel, matlIndex, patch);
    new_dw->allocate(velocityVars.uVelNonlinearSrc, 
			d_lab->d_uVelNonLinSrcMBLMLabel, matlIndex, patch);
    cerr << "in moment solve just after allocate" << index << endl;
    break;
  case Arches::YDIR:
    // getting new value of u velocity
    new_dw->get(velocityVars.uVelocity, d_lab->d_uVelocityCPBCLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells+1);
    new_dw->get(velocityVars.vVelocity, d_lab->d_vVelocityCPBCLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells+1);
    new_dw->get(velocityVars.wVelocity, d_lab->d_wVelocityCPBCLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells+1);
    old_dw->get(velocityVars.old_vVelocity, d_lab->d_vVelocitySPBCLabel, 
		matlIndex, patch, Ghost::None, zeroGhostCells);
    cerr << "in moment solve just before allocate" << index << endl;
    //    new_dw->allocate(velocityVars.variableCalledDV, d_lab->d_DVMBLMLabel,
    //			matlIndex, patch);
    for (int ii = 0; ii < nofStencils; ii++) {
      new_dw->allocate(velocityVars.vVelocityCoeff[ii], 
			  d_lab->d_vVelCoefMBLMLabel, ii, patch);
      new_dw->allocate(velocityVars.vVelocityConvectCoeff[ii], 
			  d_lab->d_vVelConvCoefMBLMLabel, ii, patch);
    }
    new_dw->allocate(velocityVars.vVelLinearSrc, 
			d_lab->d_vVelLinSrcMBLMLabel, matlIndex, patch);
    new_dw->allocate(velocityVars.vVelNonlinearSrc, 
			d_lab->d_vVelNonLinSrcMBLMLabel, matlIndex, patch);

    break;
  case Arches::ZDIR:
    // getting new value of u velocity
    new_dw->get(velocityVars.uVelocity, d_lab->d_uVelocityCPBCLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells+1);
    new_dw->get(velocityVars.vVelocity, d_lab->d_vVelocityCPBCLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells+1);
    new_dw->get(velocityVars.wVelocity, d_lab->d_wVelocityCPBCLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells+1);
    old_dw->get(velocityVars.old_wVelocity, d_lab->d_wVelocitySPBCLabel, 
		matlIndex, patch, Ghost::None, zeroGhostCells);
    //    new_dw->allocate(velocityVars.variableCalledDW, d_lab->d_DWMBLMLabel,
    //			matlIndex, patch);
    for (int ii = 0; ii < nofStencils; ii++) {
      new_dw->allocate(velocityVars.wVelocityCoeff[ii], 
			  d_lab->d_wVelCoefMBLMLabel, ii, patch);
      new_dw->allocate(velocityVars.wVelocityConvectCoeff[ii], 
			  d_lab->d_wVelConvCoefMBLMLabel, ii, patch);
    }
    new_dw->allocate(velocityVars.wVelLinearSrc, 
			d_lab->d_wVelLinSrcMBLMLabel, matlIndex, patch);
    new_dw->allocate(velocityVars.wVelNonlinearSrc, 
			d_lab->d_wVelNonLinSrcMBLMLabel, matlIndex, patch);

    break;
  default:
    throw InvalidValue("Invalid index in MomentumSolver");
  }
  // compute ith componenet of velocity stencil coefficients
  // inputs : [u,v,w]VelocityCPBC, densityIN, viscosityIN
  // outputs: [u,v,w]VelConvCoefPBLM, [u,v,w]VelCoefPBLM
  d_discretize->calculateVelocityCoeff(pc, patch, old_dw, new_dw, 
				       delta_t, index,
				       Arches::MOMENTUM,
				       cellinfo, &velocityVars);

  // Calculate velocity source
  // inputs : [u,v,w]VelocityCPBC, densityIN, viscosityIN ( new_dw), 
  //          [u,v,w]VelocitySPBC, densityCP( old_dw), 
  // outputs: [u,v,w]VelLinSrcPBLM, [u,v,w]VelNonLinSrcPBLM
  d_source->calculateVelocitySource(pc, patch, old_dw, new_dw, 
				    delta_t, index,
				    Arches::MOMENTUM,
				    cellinfo, &velocityVars);

  // Velocity Boundary conditions
  //  inputs : densityIN, [u,v,w]VelocityCPBC, [u,v,w]VelCoefPBLM
  //           [u,v,w]VelLinSrcPBLM, [u,v,w]VelNonLinSrcPBLM
  //  outputs: [u,v,w]VelCoefPBLM, [u,v,w]VelLinSrcPBLM, 
  //           [u,v,w]VelNonLinSrcPBLM
  d_boundaryCondition->velocityBC(pc, patch, old_dw, new_dw, 
				  index,
				  Arches::MOMENTUM,
				  cellinfo, &velocityVars);

  // Modify Velocity Mass Source
  //  inputs : [u,v,w]VelocityCPBC, [u,v,w]VelCoefPBLM, 
  //           [u,v,w]VelConvCoefPBLM, [u,v,w]VelLinSrcPBLM, 
  //           [u,v,w]VelNonLinSrcPBLM
  //  outputs: [u,v,w]VelLinSrcPBLM, [u,v,w]VelNonLinSrcPBLM
  d_source->modifyVelMassSource(pc, patch, old_dw,
				new_dw, delta_t, index,
				Arches::MOMENTUM, &velocityVars);

  // Calculate Velocity Diagonal
  //  inputs : [u,v,w]VelCoefPBLM, [u,v,w]VelLinSrcPBLM
  //  outputs: [u,v,w]VelCoefPBLM
  d_discretize->calculateVelDiagonal(pc, patch, old_dw, new_dw, 
				     index,
				     Arches::MOMENTUM, &velocityVars);

  // Add the pressure source terms
  // inputs :[u,v,w]VelNonlinSrcMBLM, [u,v,w]VelCoefMBLM, pressureCPBC, 
  //          densityCP(old_dw), 
  // [u,v,w]VelocityCPBC
  // outputs:[u,v,w]VelNonlinSrcMBLM

  d_source->addPressureSource(pc, patch, old_dw, new_dw, delta_t, index,
			      cellinfo, &velocityVars);
  cerr << "in moment solve just before build matrix" << index << endl;
    // put required vars
  switch (index) {
  case Arches::XDIR:
    for (int ii = 0; ii < nofStencils; ii++) {
      new_dw->put(velocityVars.uVelocityCoeff[ii], 
		     d_lab->d_uVelCoefMBLMLabel, ii, patch);

    }
    new_dw->put(velocityVars.uVelNonlinearSrc, 
		   d_lab->d_uVelNonLinSrcMBLMLabel, matlIndex, patch);
  break;
  case Arches::YDIR:
    for (int ii = 0; ii < nofStencils; ii++) {
      new_dw->put(velocityVars.vVelocityCoeff[ii], 
		     d_lab->d_vVelCoefMBLMLabel, ii, patch);
    }
    new_dw->put(velocityVars.vVelNonlinearSrc, 
		   d_lab->d_vVelNonLinSrcMBLMLabel, matlIndex, patch);
  break;
  
  case Arches::ZDIR:
    for (int ii = 0; ii < nofStencils; ii++) {
      new_dw->put(velocityVars.wVelocityCoeff[ii], 
		     d_lab->d_wVelCoefMBLMLabel, ii, patch);
    }
    new_dw->put(velocityVars.wVelNonlinearSrc, 
		   d_lab->d_wVelNonLinSrcMBLMLabel, matlIndex, patch);
  break;
  default:
    throw InvalidValue("Invalid index in MomentumSolver");
  }
}

void 
MomentumSolver::velocityLinearSolve(const ProcessorGroup* pc,
				    const Patch* patch,
				    DataWarehouseP& old_dw,
				    DataWarehouseP& new_dw,
				    double delta_t, int index)
{
  ArchesVariables velocityVars;
  int matlIndex = 0;
  int numGhostCells = 1;
  int zeroGhostCells = 0;
  int nofStencils = 7;
  //  DataWarehouseP old_dw = new_dw->getTop();
  // Get the PerPatch CellInformation data
  PerPatch<CellInformation*> cellInfoP;
  // get old_dw from getTop function
  old_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
  //  old_dw->get(cellInfoP, d_cellInfoLabel, matlIndex, patch);
  //  if (old_dw->exists(d_cellInfoLabel, patch)) 
  //  old_dw->get(cellInfoP, d_cellInfoLabel, matlIndex, patch);
  //else {
  //  cellInfoP.setData(scinew CellInformation(patch));
  //  old_dw->put(cellInfoP, d_cellInfoLabel, matlIndex, patch);
  //}
  CellInformation* cellinfo = cellInfoP;
  old_dw->get(velocityVars.old_density, d_lab->d_densityCPLabel, 
	      matlIndex, patch, Ghost::None, zeroGhostCells);
  switch (index) {
  case Arches::XDIR:
    new_dw->get(velocityVars.uVelocity, d_lab->d_uVelocityCPBCLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
    for (int ii = 0; ii < nofStencils; ii++)
      new_dw->get(velocityVars.uVelocityCoeff[ii], 
		     d_lab->d_uVelCoefMBLMLabel, 
		     ii, patch, Ghost::None, zeroGhostCells);
    new_dw->get(velocityVars.uVelNonlinearSrc, 
		   d_lab->d_uVelNonLinSrcMBLMLabel,
		   matlIndex, patch, Ghost::None, zeroGhostCells);
    new_dw->allocate(velocityVars.residualUVelocity, d_lab->d_uVelocityRes,
			  matlIndex, patch);

    break;
  case Arches::YDIR:
    new_dw->get(velocityVars.vVelocity, d_lab->d_vVelocityCPBCLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
    // initial guess for explicit calculations
    for (int ii = 0; ii < nofStencils; ii++)
      new_dw->get(velocityVars.vVelocityCoeff[ii], 
		     d_lab->d_vVelCoefMBLMLabel, 
		     ii, patch, Ghost::None, zeroGhostCells);
    new_dw->get(velocityVars.vVelNonlinearSrc, 
		   d_lab->d_vVelNonLinSrcMBLMLabel,
		   matlIndex, patch, Ghost::None, zeroGhostCells);
    new_dw->allocate(velocityVars.residualVVelocity, d_lab->d_vVelocityRes,
			  matlIndex, patch);
    break; 
  case Arches::ZDIR:
    new_dw->get(velocityVars.wVelocity, d_lab->d_wVelocityCPBCLabel, 
		matlIndex, patch, Ghost::AroundCells, numGhostCells);

    for (int ii = 0; ii < nofStencils; ii++)
      new_dw->get(velocityVars.wVelocityCoeff[ii], 
		     d_lab->d_wVelCoefMBLMLabel, 
		     ii, patch, Ghost::None, zeroGhostCells);
    new_dw->get(velocityVars.wVelNonlinearSrc, 
		   d_lab->d_wVelNonLinSrcMBLMLabel,
		   matlIndex, patch, Ghost::None, zeroGhostCells);
    new_dw->allocate(velocityVars.residualWVelocity, d_lab->d_wVelocityRes,
			  matlIndex, patch);
    break;  
  default:
    throw InvalidValue("Invalid index in MomentumSolver");
  }
  
  // compute eqn residual
#if 0
  d_linearSolver->computeVelResidual(pc, patch, new_dw, new_dw, index, 
				     &velocityVars);
  // put the summed residuals into new_dw
  switch (index) {
  case Arches::XDIR:
    new_dw->put(sum_vartype(velocityVars.residUVel), d_lab->d_uVelResidPSLabel);
    new_dw->put(sum_vartype(velocityVars.truncUVel), d_lab->d_uVelTruncPSLabel);
    break;
  case Arches::YDIR:
    new_dw->put(sum_vartype(velocityVars.residVVel), d_lab->d_vVelResidPSLabel);
    new_dw->put(sum_vartype(velocityVars.truncVVel), d_lab->d_vVelTruncPSLabel);
    break;
  case Arches::ZDIR:
    new_dw->put(sum_vartype(velocityVars.residWVel), d_lab->d_wVelResidPSLabel);
    new_dw->put(sum_vartype(velocityVars.truncWVel), d_lab->d_wVelTruncPSLabel);
    break;
  default:
    throw InvalidValue("Invalid index in MomentumSolver");  
  }
#endif

  // apply underelax to eqn
  d_linearSolver->computeVelUnderrelax(pc, patch, old_dw, new_dw, index, 
				     &velocityVars);
  // initial guess for explicit calculation
#if 0
  new_dw->get(velocityVars.old_uVelocity, d_lab->d_uVelocityCPBCLabel, 
	      matlIndex, patch, Ghost::AroundCells, numGhostCells);
  new_dw->get(velocityVars.old_vVelocity, d_lab->d_vVelocityCPBCLabel, 
	      matlIndex, patch, Ghost::AroundCells, numGhostCells);
  new_dw->get(velocityVars.old_wVelocity, d_lab->d_wVelocityCPBCLabel, 
	      matlIndex, patch, Ghost::AroundCells, numGhostCells);
#endif

  new_dw->allocate(velocityVars.old_uVelocity, d_lab->d_old_uVelocityGuess,
			  matlIndex, patch);
  new_dw->allocate(velocityVars.old_vVelocity, d_lab->d_old_vVelocityGuess,
			  matlIndex, patch);
  new_dw->allocate(velocityVars.old_wVelocity, d_lab->d_old_wVelocityGuess,
			  matlIndex, patch);

  velocityVars.old_uVelocity.copy(velocityVars.uVelocity);
  velocityVars.old_vVelocity.copy(velocityVars.vVelocity);
  velocityVars.old_wVelocity.copy(velocityVars.wVelocity);

  // make it a separate task later
  d_linearSolver->velocityLisolve(pc, patch, old_dw, new_dw, index, delta_t, 
				  &velocityVars, cellinfo, d_lab);
  // put back the results
  switch (index) {
  case Arches::XDIR:
    new_dw->put(velocityVars.uVelocity, d_lab->d_uVelocitySPBCLabel, 
				     matlIndex, patch);
    break;
  case Arches::YDIR:
    new_dw->put(velocityVars.vVelocity, d_lab->d_vVelocitySPBCLabel,
				      matlIndex, patch);
    break;
  case Arches::ZDIR:
    new_dw->put(velocityVars.wVelocity, d_lab->d_wVelocitySPBCLabel, 
				     matlIndex, patch);
    break;
  default:
    throw InvalidValue("Invalid index in MomentumSolver");  
  }
}
    
  

  
//
// $Log$
// Revision 1.31  2000/10/09 17:06:24  rawat
// modified momentum solver for multi-patch
//
// Revision 1.30  2000/09/20 18:05:33  sparker
// Adding support for Petsc and per-processor tasks
//
// Revision 1.29  2000/09/20 16:56:16  rawat
// added some petsc parallel stuff and fixed some bugs
//
// Revision 1.28  2000/09/14 17:04:54  rawat
// converting arches to multipatch
//
// Revision 1.27  2000/09/07 23:07:17  rawat
// fixed some bugs in bc and added pressure solver using petsc
//
// Revision 1.26  2000/08/15 00:23:32  rawat
// added explicit solve for momentum and scalar eqns
//
// Revision 1.25  2000/08/14 02:34:57  bbanerje
// Removed a small buf in sum_vars for residual in MomentumSolver and ScalarSolver
//
// Revision 1.24  2000/08/12 23:53:19  bbanerje
// Added Linegs part to the solver.
//
// Revision 1.23  2000/08/10 21:29:09  rawat
// fixed a bug in cellinformation
//
// Revision 1.22  2000/08/01 23:28:43  skumar
// Added residual calculation procedure and modified templates in linear
// solver.  Added template for order-of-magnitude term calculation.
//
// Revision 1.21  2000/08/01 06:18:37  bbanerje
// Made ScalarSolver similar to PressureSolver and MomentumSolver.
//
// Revision 1.20  2000/07/28 02:31:00  rawat
// moved all the labels in ArchesLabel. fixed some bugs and added matrix_dw to store matrix
// coeffecients
//
// Revision 1.17  2000/07/17 22:06:58  rawat
// modified momentum source
//
// Revision 1.16  2000/07/12 07:35:46  bbanerje
// Added stuff for mascal : Rawat: Labels and dataWarehouse in velsrc need to be corrected.
//
// Revision 1.15  2000/07/08 23:42:54  bbanerje
// Moved all enums to Arches.h and made corresponding changes.
//
// Revision 1.14  2000/07/03 05:30:14  bbanerje
// Minor changes for inlbcs dummy code to compile and work. densitySIVBC is no more.
//
// Revision 1.13  2000/06/29 22:56:42  bbanerje
// Changed FCVars to SFC[X,Y,Z]Vars, and added the neceesary getIndex calls.
//
// Revision 1.12  2000/06/22 23:06:34  bbanerje
// Changed velocity related variables to FCVariable type.
// ** NOTE ** We may need 3 types of FCVariables (one for each direction)
//
// Revision 1.11  2000/06/21 07:51:00  bbanerje
// Corrected new_dw, old_dw problems, commented out intermediate dw (for now)
// and made the stuff go through schedule_time_advance.
//
// Revision 1.10  2000/06/21 06:00:12  bbanerje
// Added two labels that were causing seg violation.
//
// Revision 1.9  2000/06/18 01:20:15  bbanerje
// Changed names of varlabels in source to reflect the sequence of tasks.
// Result : Seg Violation in addTask in MomentumSolver
//
// Revision 1.8  2000/06/17 07:06:24  sparker
// Changed ProcessorContext to ProcessorGroup
//
// Revision 1.7  2000/06/16 04:25:40  bbanerje
// Uncommented BoundaryCondition related stuff.
//
// Revision 1.6  2000/06/14 20:40:49  rawat
// modified boundarycondition for physical boundaries and
// added CellInformation class
//
// Revision 1.5  2000/06/07 06:13:54  bbanerje
// Changed CCVariable<Vector> to CCVariable<double> for most cases.
// Some of these variables may not be 3D Vectors .. they may be Stencils
// or more than 3D arrays. Need help here.
//
// Revision 1.4  2000/06/04 22:40:13  bbanerje
// Added Cocoon stuff, changed task, require, compute, get, put arguments
// to reflect new declarations. Changed sub.mk to include all the new files.
//
//

