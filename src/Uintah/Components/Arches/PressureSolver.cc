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
// Private constructor for PressureSolver
//****************************************************************************
PressureSolver::PressureSolver()
{
}

//****************************************************************************
// Default constructor for PressureSolver
//****************************************************************************
PressureSolver::PressureSolver(int nDim,
			       TurbulenceModel* turb_model,
			       BoundaryCondition* bndry_cond,
			       PhysicalConstants* physConst): 
                                     d_NDIM(nDim),
                                     d_turbModel(turb_model), 
                                     d_boundaryCondition(bndry_cond),
				     d_physicalConsts(physConst),
				     d_generation(0)
{
  d_pressureLabel = scinew VarLabel("pressure",
				    CCVariable<double>::getTypeDescription() );
  // BB : (tmp) velocity is set as CCVariable (should be FCVariable)
  d_velocityLabel = scinew VarLabel("velocity",
				    CCVariable<Vector>::getTypeDescription() );
  d_densityLabel = scinew VarLabel("density",
				   CCVariable<double>::getTypeDescription() );
  d_viscosityLabel = scinew VarLabel("viscosity",
				     CCVariable<double>::getTypeDescription() );
  // BB : (tmp) velocity is set as CCVariable (should be FCVariable)
  d_velConvCoefLabel = scinew VarLabel("velocityConvectCoeff",
				       CCVariable<Vector>::getTypeDescription() );
  // BB : (tmp) velocity is set as CCVariable (should be FCVariable)
  d_velCoefLabel = scinew VarLabel("velocityCoeff",
				   CCVariable<Vector>::getTypeDescription() );
  // BB : (tmp) velocity is set as CCVariable (should be FCVariable)
  d_velLinSrcLabel = scinew VarLabel("velLinearSource",
				     CCVariable<Vector>::getTypeDescription() );
  // BB : (tmp) velocity is set as CCVariable (should be FCVariable)
  d_velNonLinSrcLabel = scinew VarLabel("velNonlinearSource",
					CCVariable<Vector>::getTypeDescription() );
  d_presCoefLabel = scinew VarLabel("pressureCoeff",
				       CCVariable<double>::getTypeDescription() );
  d_presLinSrcLabel = scinew VarLabel("pressureLinearSource",
				       CCVariable<double>::getTypeDescription() );
  d_presNonLinSrcLabel = scinew VarLabel("pressureNonlinearSource",
				       CCVariable<double>::getTypeDescription() );
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
    throw InvalidValue("Linear solver option"
		       " not supported" + linear_sol);
    //throw InvalidValue("linear solver option"
	//	       " not supported" + linear_sol, db);
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
  DataWarehouseP matrix_dw = sched->createDataWarehouse(d_generation);
  ++d_generation;

  //computes stencil coefficients and source terms
  sched_buildLinearMatrix(level, sched, new_dw, matrix_dw, delta_t);

  //residual at the start of linear solve
  // this can be part of linear solver
#if 0
  calculateResidual(level, sched, new_dw, matrix_dw);
  calculateOrderMagnitude(level, sched, new_dw, matrix_dw);
#endif

  d_linearSolver->sched_pressureSolve(level, sched, new_dw, matrix_dw);
  sched_normPressure(level, sched, new_dw, new_dw);
  
}

//****************************************************************************
// Schedule build of linear matrix
//****************************************************************************
void 
PressureSolver::sched_buildLinearMatrix(const LevelP& level,
					SchedulerP& sched,
					DataWarehouseP& old_dw,
					DataWarehouseP& new_dw,
					double delta_t)
{
  for(Level::const_patchIterator iter=level->patchesBegin();
      iter != level->patchesEnd(); iter++){
    const Patch* patch=*iter;
    {
      Task* tsk = scinew Task("PressureSolver::BuildCoeff",
			   patch, old_dw, new_dw, this,
			   &PressureSolver::buildLinearMatrix, delta_t);

      int numGhostCells = 0;
      int matlIndex = 0;
      tsk->requires(old_dw, d_pressureLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(old_dw, d_velocityLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(old_dw, d_densityLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(old_dw, d_viscosityLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);

      /// requires convection coeff because of the nodal
      // differencing
      // computes all the components of velocity
      tsk->computes(new_dw, d_velConvCoefLabel, matlIndex, patch);
      tsk->computes(new_dw, d_velCoefLabel, matlIndex, patch);
      tsk->computes(new_dw, d_velLinSrcLabel, matlIndex, patch);
      tsk->computes(new_dw, d_velNonLinSrcLabel, matlIndex, patch);
      tsk->computes(new_dw, d_presCoefLabel, matlIndex, patch);
      tsk->computes(new_dw, d_presLinSrcLabel, matlIndex, patch);
      tsk->computes(new_dw, d_presNonLinSrcLabel, matlIndex, patch);
     
      sched->addTask(tsk);
    }

  }
}

//****************************************************************************
// Actually build of linear matrix
//****************************************************************************
void 
PressureSolver::buildLinearMatrix(const ProcessorContext* pc,
				  const Patch* patch,
				  DataWarehouseP& old_dw,
				  DataWarehouseP& new_dw,
				  double delta_t)
{
  // compute all three componenets of velocity stencil coefficients
  for(int index = 1; index <= d_NDIM; ++index) {
    d_discretize->calculateVelocityCoeff(pc, patch, old_dw,
					 new_dw,delta_t, index);
    d_source->calculateVelocitySource(pc, patch, old_dw,
				      new_dw,delta_t, index);
#ifdef WONT_COMPILE_YET
    d_boundaryCondition->velocityBC(pc, patch, old_dw,
				    new_dw,delta_t, index);
#endif
    // similar to mascal
    d_source->modifyVelMassSource(pc, patch, old_dw, new_dw, delta_t, index);

    d_discretize->calculateVelDiagonal(pc, patch, old_dw,
				       new_dw, index);
  }
  d_discretize->calculatePressureCoeff(pc, patch, old_dw, new_dw, delta_t);
  d_source->calculatePressureSource(pc, patch, old_dw, new_dw, delta_t);
#ifdef WONT_COMPILE_YET
  d_boundaryCondition->pressureBC(pc, patch, old_dw,
				  new_dw,delta_t);
#endif
  d_discretize->calculatePressDiagonal(pc, patch, old_dw, new_dw);

}

//****************************************************************************
// Schedule the creation of the .. more documentation here
//****************************************************************************
void 
PressureSolver::
sched_normPressure(const LevelP& ,
		   SchedulerP& ,
		   DataWarehouseP& ,
		   DataWarehouseP& )
{
}  

//****************************************************************************
// Actually do normPressure
//****************************************************************************
void 
PressureSolver::
normPressure(const Patch* ,
	     SchedulerP& ,
	     const DataWarehouseP& ,
	     DataWarehouseP& )
{
}

//
// $Log$
// Revision 1.16  2000/06/05 01:01:57  tan
// Made PressureSolver::sched_normPressure and
// PressureSolver::normPressure compilable.
//
// Revision 1.15  2000/06/04 23:57:46  bbanerje
// Updated Arches to do ScheduleTimeAdvance.
//
// Revision 1.14  2000/06/04 22:40:14  bbanerje
// Added Cocoon stuff, changed task, require, compute, get, put arguments
// to reflect new declarations. Changed sub.mk to include all the new files.
//
//
