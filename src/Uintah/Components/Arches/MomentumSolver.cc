#include <Uintah/Components/Arches/MomentumSolver.h>
#include <Uintah/Components/Arches/Discretization.h>
#include <Uintah/Components/Arches/Source.h>
#include <Uintah/Components/Arches/BoundaryCondition.h>
#include <Uintah/Components/Arches/TurbulenceModel.h>
#include <Uintah/Components/Arches/PhysicalConstants.h>
#include <Uintah/Exceptions/InvalidValue.h>
#include <Uintah/Interface/Scheduler.h>
#include <Uintah/Interface/ProblemSpec.h>
#include <SCICore/Util/NotFinished.h>
#include <Uintah/Components/Arches/Arches.h>

using namespace Uintah::Arches;
using namespace std;

MomentumSolver::MomentumSolver()
{
}

MomentumSolver::MomentumSolver(TurbulenceModel* turb_model,
			       BoundaryCondition* bndry_cond,
			       PhysicalConstants* physConst)
: d_turbModel(turb_model), d_boundaryCondition(bndry_cond),
  d_physicalConsts(physConst)
{
}

MomentumSolver::~MomentumSolver()
{
}

void MomentumSolver::problemSetup(const ProblemSpecP& params)
{
  ProblemSpecP db = params->findBlock("MomentumSolver");
  string finite_diff;
  db->require("finite_difference", finite_diff);
  if (finite_diff == "Secondorder") 
    d_discretize = scinew Discretization();
  else 
    throw InvalidValue("Finite Differencing scheme "
		       "not supported: " + finite_diff, db);
  // make source and boundary_condition objects
  d_source = Source(d_turbModel, d_physicalConsts);
  string linear_sol;
  db->require("linear_solver", linear_sol);
  if (linear_sol == "RBGaussSeidel")
    d_linearSolver = scinew RBGSSolver();
  else 
    throw InvalidValue("linear solver option"
		       " not supported" + linear_sol, db);
  d_linearSolver->problemSetup(db);
}

void MomentumSolver::solve(double time, double delta_t, int index,
			   const LevelP& level,
			   SchedulerP& sched,
			   const DataWarehouseP& old_dw,
			   DataWarehouseP& new_dw)
{
  //create a new data warehouse to store matrix coeff
  // and source terms. It gets reinitialized after every 
  // pressure solve.
  DataWarehouseP matrix_dw = sched->createDataWarehouse();
  //computes stencil coefficients and source terms
  sched_buildLinearMatrix(delta_t, index, level, sched, new_dw, matrix_dw);
    
  d_linearSolver->sched_velSolve(level, sched, new_dw, matrix_dw);
    
}

void MomentumSolver::sched_buildLinearMatrix(double delta_t, int index,
					     const LevelP& level,
					     SchedulerP& sched,
					     const DataWarehouseP& old_dw,
					     DataWarehouseP& new_dw)
{
  for(Level::const_patchIterator iter=level->patchesBegin();
      iter != level->patchesEnd(); iter++){
    const Patch* patch=*iter;
    {
      // steve: requires two arguments
      Task* tsk = scinew Task("MomentumSolver::BuildCoeff",
			      patch, old_dw, new_dw, this,
			      Discretization::buildLinearMatrix,
			      delta_t, index);
      tsk->requires(old_dw, "velocity", patch, 1,
		    FCVariable<Vector>::getTypeDescription());
      tsk->requires(old_dw, "density", patch, 1,
		    CCVariable<double>::getTypeDescription());
      tsk->requires(old_dw, "viscosity", patch, 1,
		    CCVariable<double>::getTypeDescription());
      tsk->requires(old_dw, "pressure", patch, 1,
		    CCVariable<double>::getTypeDescription());
      /// requires convection coeff because of the nodal
      // differencing
      // computes index components of velocity
      tsk->computes(new_dw, "VelocityConvectCoeff", patch, 0,index,
		    FCVariable<Vector>::getTypeDescription());
      tsk->computes(new_dw, "VelocityCoeff", patch, 0, index,
		    FCVariable<Vector>::getTypeDescription());
      tsk->computes(new_dw, "VelLinearSource", patch, 0, index,
		    FCVariable<Vector>::getTypeDescription());
      tsk->computes(new_dw, "VelNonlinearSource", patch, 0, index,
		    FCVariable<Vector>::getTypeDescription());
      sched->addTask(tsk);
    }

  }
}


void MomentumSolver::buildLinearMatrix(const ProcessorContext* pc,
				       const Patch* patch,
				       const DataWarehouseP& old_dw,
				       DataWarehouseP& new_dw,
				       double delta_t, const int index)
{
  // compute ith componenet of velocity stencil coefficients
  d_discretize->calculateVelocityCoeff(pc, patch, old_dw,
				       new_dw, delta_t, index);
  d_source->calculateVelocitySource(pc, patch, old_dw,
				    new_dw, delta_t, index);
  d_boundaryCondition->velocityBC(pc, patch, old_dw,
				  new_dw, delta_t, index);
  // similar to mascal
  d_source->modifyVelMassSource(pc, patch, old_dw,
			     new_dw, delta_t, index);
  d_discretize->calculateVelDiagonal(pc, patch, old_dw,
				     new_dw, index);
  d_source->addPressureSource(pc, patch, old_dw,
			     new_dw, index);

}



