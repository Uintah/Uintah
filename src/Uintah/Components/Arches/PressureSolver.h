/**************************************
CLASS
   PressureSolver
   
   Class PressureSolver linearizes and solves pressure
   equation on a grid hierarchy


GENERAL INFORMATION
   PressureSolver.h - declaration of the class
   
   Author: Rajesh Rawat (rawat@crsim.utah.edu)
   
   Creation Date:   Mar 1, 2000
   
   C-SAFE 
   
   Copyright U of U 2000

KEYWORDS


DESCRIPTION
   Class PressureSolver linearizes and solves pressure
   equation on a grid hierarchy


WARNING
   none

************************************************************************/
#ifndef included_PressureSolver
#define included_PressureSolver

#include <Uintah/Grid/LevelP.h>
#include <Uintah/Interface/SchedulerP.h>
#include <Uintah/Grid/CCVariable.h>
#include <Uintah/Interface/ProblemSpecP.h>
#include <Uintah/Interface/DataWarehouseP.h>

namespace Uintah {
    namespace Components {


#ifndef LACKS_NAMESPACE
using namespace Uintah::Grid;
using namespace Uintah::Components;
using namespace Uintah::Interface;
#endif
class TurbulenceModel;
class PhysicalConstants;
class Discretization;
class Source;
class BoundaryCondition;
class LinearSolver;

class PressureSolver
{

 public:

  // GROUP: Constructors:
  ////////////////////////////////////////////////////////////////////////
  //
  // Construct an instance of the Pressure solver.
  //
  // PRECONDITIONS
  //
  // POSTCONDITIONS
  //   A linear level solver is partially constructed.  
  //
  // Default constructor.
   PressureSolver();
   PressureSolver(TurbulenceModel* turb_model, BoundaryCondition* bndry_cond,
		  PhysicalConstants* physConst);

  // GROUP: Destructors:
  ////////////////////////////////////////////////////////////////////////
  // Destructor
   ~PressureSolver();

   // sets parameters at start time
   void problemSetup(const ProblemSpecP& params);

   // linearize eqn
   void sched_buildLinearMatrix(double delta_t,
				const LevelP& level,
				SchedulerP& sched,
				const DataWarehouseP& old_dw,
				DataWarehouseP& new_dw);
 
   ////////////////////////////////////////////////////////////////////////
   // solve linearized pressure equation
   void solve(double time, double delta_t, 
	      const LevelP& level,
	      SchedulerP& sched,
	      const DataWarehouseP& old_dw,
	      DataWarehouseP& new_dw);
   
   void sched_normPressure(const LevelP& level,
			   SchedulerP& sched,
			   const DataWarehouseP& old_dw,
			   DataWarehouseP& new_dw);  
 private:

   void buildLinearMatrix(const Region* region,
			  SchedulerP& sched,
			  const DataWarehouseP& old_dw,
			  DataWarehouseP& new_dw,
			  double delta_t);
   void normPressure(const Region* region,
		     SchedulerP& sched,
		     const DataWarehouseP& old_dw,
		     DataWarehouseP& new_dw);
   // computes coefficients
   Discretization* d_discretize;

   // computes sources
   Source* d_source;

   // linear solver
   LinearSolver* d_linearSolver;

   // turbulence model
   TurbulenceModel* d_turbModel;
   // boundary condition
   BoundaryCondition* d_boundaryCondition;
   // physical constants
   PhysicalConstants* d_physicalConsts;
// GROUP: Data members.

   ////////////////////////////////////////////////////////////////////////
   // Maximum number of iterations to take before stopping/giving up.
   int d_maxIterations;

   // underrealaxation parameter, read from an input database
   double d_underrelax;
   //reference points for the solvers
   Array1 d_pressRef;

};

    }
}

#endif
