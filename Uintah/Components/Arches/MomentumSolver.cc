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
#include <Uintah/Interface/ProblemSpec.h>
#include <Uintah/Grid/Level.h>
#include <Uintah/Grid/Task.h>
#include <SCICore/Util/NotFinished.h>
#include <Uintah/Components/Arches/Arches.h>

using namespace Uintah::ArchesSpace;
using namespace std;

//****************************************************************************
// Private constructor for MomentumSolver
//****************************************************************************
MomentumSolver::MomentumSolver()
{
}

//****************************************************************************
// Default constructor for MomentumSolver
//****************************************************************************
MomentumSolver::MomentumSolver(TurbulenceModel* turb_model,
			       BoundaryCondition* bndry_cond,
			       PhysicalConstants* physConst) : 
                                   d_turbModel(turb_model), 
                                   d_boundaryCondition(bndry_cond),
				   d_physicalConsts(physConst),
				   d_generation(0)
{
  d_pressureLabel = scinew VarLabel("pressure",
				    CCVariable<double>::getTypeDescription() );
  // BB : (tmp) velocity is set as CCVariable (should be FCVariable)
  d_uVelocityLabel = scinew VarLabel("uVelocity",
				     CCVariable<double>::getTypeDescription() );
  d_vVelocityLabel = scinew VarLabel("vVelocity",
				     CCVariable<double>::getTypeDescription() );
  d_wVelocityLabel = scinew VarLabel("wVelocity",
				     CCVariable<double>::getTypeDescription() );
  d_densityLabel = scinew VarLabel("density",
				   CCVariable<double>::getTypeDescription() );
  d_viscosityLabel = scinew VarLabel("viscosity",
				     CCVariable<double>::getTypeDescription() );
  // BB : (tmp) velocity is set as CCVariable (should be FCVariable)
  d_uVelConvCoefLabel = scinew VarLabel("uVelocityConvectCoeff",
					CCVariable<double>::getTypeDescription() );
  d_vVelConvCoefLabel = scinew VarLabel("vVelocityConvectCoeff",
					CCVariable<double>::getTypeDescription() );
  d_wVelConvCoefLabel = scinew VarLabel("wVelocityConvectCoeff",
					CCVariable<double>::getTypeDescription() );
  // BB : (tmp) velocity is set as CCVariable (should be FCVariable)
  d_uVelCoefLabel = scinew VarLabel("uVelocityCoeff",
				    CCVariable<double>::getTypeDescription() );
  d_vVelCoefLabel = scinew VarLabel("vVelocityCoeff",
				    CCVariable<double>::getTypeDescription() );
  d_wVelCoefLabel = scinew VarLabel("wVelocityCoeff",
				    CCVariable<double>::getTypeDescription() );
  // BB : (tmp) velocity is set as CCVariable (should be FCVariable)
  d_uVelLinSrcLabel = scinew VarLabel("uVelLinearSource",
				      CCVariable<double>::getTypeDescription() );
  d_vVelLinSrcLabel = scinew VarLabel("vVelLinearSource",
				      CCVariable<double>::getTypeDescription() );
  d_wVelLinSrcLabel = scinew VarLabel("wVelLinearSource",
				      CCVariable<double>::getTypeDescription() );
  // BB : (tmp) velocity is set as CCVariable (should be FCVariable)
  d_uVelNonLinSrcLabel = scinew VarLabel("uVelNonlinearSource",
					 CCVariable<double>::getTypeDescription() );
  d_vVelNonLinSrcLabel = scinew VarLabel("vVelNonlinearSource",
					 CCVariable<double>::getTypeDescription() );
  d_wVelNonLinSrcLabel = scinew VarLabel("wVelNonlinearSource",
					 CCVariable<double>::getTypeDescription() );
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
    //throw InvalidValue("Finite Differencing scheme "
	//	       "not supported: " + finite_diff, db);
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
    //throw InvalidValue("linear solver option"
	//	       " not supported" + linear_sol, db);
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
		      double time, double delta_t, int index)
{
  //create a new data warehouse to store matrix coeff
  // and source terms. It gets reinitialized after every 
  // pressure solve.
  DataWarehouseP matrix_dw = sched->createDataWarehouse(d_generation);

  //computes stencil coefficients and source terms
  sched_buildLinearMatrix(level, sched, new_dw, matrix_dw, delta_t, index);
    
  d_linearSolver->sched_velSolve(level, sched, new_dw, matrix_dw, index);
    
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

      int numGhostCells = 0;
      int matlIndex = 0;
      tsk->requires(old_dw, d_pressureLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(old_dw, d_uVelocityLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(old_dw, d_vVelocityLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(old_dw, d_wVelocityLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(old_dw, d_densityLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(old_dw, d_viscosityLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);

      /// requires convection coeff because of the nodal
      // differencing
      // computes index components of velocity
      tsk->computes(new_dw, d_uVelConvCoefLabel, matlIndex, patch);
      tsk->computes(new_dw, d_vVelConvCoefLabel, matlIndex, patch);
      tsk->computes(new_dw, d_wVelConvCoefLabel, matlIndex, patch);
      tsk->computes(new_dw, d_uVelCoefLabel, matlIndex, patch);
      tsk->computes(new_dw, d_vVelCoefLabel, matlIndex, patch);
      tsk->computes(new_dw, d_wVelCoefLabel, matlIndex, patch);
      tsk->computes(new_dw, d_uVelLinSrcLabel, matlIndex, patch);
      tsk->computes(new_dw, d_vVelLinSrcLabel, matlIndex, patch);
      tsk->computes(new_dw, d_wVelLinSrcLabel, matlIndex, patch);
      tsk->computes(new_dw, d_uVelNonLinSrcLabel, matlIndex, patch);
      tsk->computes(new_dw, d_vVelNonLinSrcLabel, matlIndex, patch);
      tsk->computes(new_dw, d_wVelNonLinSrcLabel, matlIndex, patch);

      sched->addTask(tsk);
    }
  }
}

//****************************************************************************
// Actual build of the linear matrix
//****************************************************************************
void MomentumSolver::buildLinearMatrix(const ProcessorContext* pc,
				       const Patch* patch,
				       DataWarehouseP& old_dw,
				       DataWarehouseP& new_dw,
				       double delta_t, int index)
{
  // compute ith componenet of velocity stencil coefficients
  d_discretize->calculateVelocityCoeff(pc, patch, old_dw,
				       new_dw, delta_t, index);
  d_source->calculateVelocitySource(pc, patch, old_dw,
				    new_dw, delta_t, index);
#ifdef WONT_COMPILE_YET
  d_boundaryCondition->velocityBC(pc, patch, old_dw,
				  new_dw, delta_t, index);
#endif
  // similar to mascal
  d_source->modifyVelMassSource(pc, patch, old_dw,
			     new_dw, delta_t, index);
  d_discretize->calculateVelDiagonal(pc, patch, old_dw,
				     new_dw, index);
  d_source->addPressureSource(pc, patch, old_dw,
			     new_dw, index);

}

//
// $Log$
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

