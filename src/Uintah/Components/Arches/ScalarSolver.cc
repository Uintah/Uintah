#include <Uintah/Components/Arches/ScalarSolver.h>
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

using Uintah::Components::ScalarSolver;
using namespace std;

ScalarSolver::ScalarSolver()
{
}

ScalarSolver::ScalarSolver(TurbulenceModel* turb_model,
			   BoundaryCondition* bndry_cond,
			   PhysicalConstants* physConst)
: d_turbModel(turb_model), d_boundaryCondition(bndry_cond),
  d_physicalConsts(physConst)
{
}

ScalarSolver::~ScalarSolver()
{
}

void ScalarSolver::problemSetup(const ProblemSpecP& params)
{
  ProblemSpecP db = params->findBlock("ScalarSolver");
  string finite_diff;
  db->require("finite_difference", finite_diff);
  if (finite_diff == "Secondorder") 
    d_discretize = new Discretization();
  else 
    throw InvalidValue("Finite Differencing scheme "
		       "not supported: " + finite_diff, db);
  // make source and boundary_condition objects
  d_source = Source(d_turbModel, d_physicalConsts);
  string linear_sol;
  db->require("linear_solver", linear_sol);
  if (linear_sol == "RBGaussSeidel")
    d_linearSolver = new RBGSSolver();
  else 
    throw InvalidValue("linear solver option"
		       " not supported" + linear_sol, db);
  d_linearSolver->problemSetup(db);
}

void ScalarSolver::solve(double time, double delta_t, int index,
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
    
  d_linearSolver->sched_scalarSolve(index, level, sched, new_dw, matrix_dw);
    
}

void ScalarSolver::sched_buildLinearMatrix(double delta_t, int index,
					   const LevelP& level,
					   SchedulerP& sched,
					   const DataWarehouseP& old_dw,
					   DataWarehouseP& new_dw)
{
  for(Level::const_regionIterator iter=level->regionsBegin();
      iter != level->regionsEnd(); iter++){
    const Region* region=*iter;
    {
      // steve: requires two arguments
      Task* tsk = new Task("ScalarSolver::BuildCoeff",
			   region, old_dw, new_dw, this,
			   Discretization::buildLinearMatrix,
			   delta_t, index);
      tsk->requires(old_dw, "velocity", region, 1,
		    FCVariable<Vector>::getTypeDescription());
      tsk->requires(old_dw, "density", region, 1,
		    CCVariable<double>::getTypeDescription());
      tsk->requires(old_dw, "viscosity", region, 1,
		    CCVariable<double>::getTypeDescription());
      tsk->requires(old_dw, "scalar", region, 1,
		    CCVariable<double>::getTypeDescription());
      /// requires convection coeff because of the nodal
      // differencing
      // computes all the components of velocity
      // added one more argument of index to specify scalar component
      tsk->computes(new_dw, "ScalarCoeff", region, 0, index,
		    FCVariable<Vector>::getTypeDescription());
      tsk->computes(new_dw, "ScalarLinearSource", region, 0,index,
		    FCVariable<Vector>::getTypeDescription());
      tsk->computes(new_dw, "ScalarNonlinearSource", region, 0,index,
		    FCVariable<Vector>::getTypeDescription());
      sched->addTask(tsk);
    }

  }
}


void ScalarSolver::buildLinearMatrix(const ProcessorContext* pc,
				     const Region* region,
				     const DataWarehouseP& old_dw,
				     DataWarehouseP& new_dw,
				     double delta_t, const int index)
{
  // compute ith componenet of velocity stencil coefficients
  d_discretize->calculateScalarCoeff(pc, region, old_dw,
				       new_dw, delta_t, index);
  d_source->calculateScalarSource(pc, region, old_dw,
				    new_dw, delta_t, index);
  d_boundaryCondition->scalarBC(pc, region, old_dw,
				  new_dw, delta_t, index);
  // similar to mascal
  d_source->modifyScalarMassSource(pc, region, old_dw,
				   new_dw, delta_t, index);
  d_discretize->calculateScalarDiagonal(pc, region, old_dw,
				     new_dw, index);
}



