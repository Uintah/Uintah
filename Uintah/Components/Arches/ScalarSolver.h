/**************************************
CLASS
   ScalarSolver
   
   Class ScalarSolver linearizes and solves momentum
   equation on a grid hierarchy


GENERAL INFORMATION
   ScalarSolver.h - declaration of the class
   
   Author: Rajesh Rawat (rawat@crsim.utah.edu)
   
   Creation Date:   Mar 1, 2000
   
   C-SAFE 
   
   Copyright U of U 2000

KEYWORDS


DESCRIPTION
   Class ScalarSolver linearizes and solves scalar
   equation on a grid hierarchy


WARNING
   none

************************************************************************/
#ifndef included_ScalarSolver
#define included_ScalarSolver

#include <Uintah/Grid/LevelP.h>
#include <Uintah/Interface/SchedulerP.h>
#include <Uintah/Grid/CCVariable.h>
#include <Uintah/Interface/ProblemSpecP.h>
#include <Uintah/Interface/DataWarehouseP.h>

namespace Uintah {
    namespace ArchesSpace {
class TurbulenceModel;
class PhysicalConstants;
class Discretization;
class Source;
class BoundaryCondition;
class LinearSolver;

class ScalarSolver
{

 public:

  // GROUP: Constructors:
  ////////////////////////////////////////////////////////////////////////
  //
  // Construct an instance of the Scalar solver.
  //
  // PRECONDITIONS
  //
  // POSTCONDITIONS
  //   A linear level solver is partially constructed.  
  //
  // Default constructor.
   ScalarSolver();
   ScalarSolver(TurbulenceModel* turb_model, BoundaryCondition* bndry_cond,
		PhysicalConstants* physConst);

  // GROUP: Destructors:
  ////////////////////////////////////////////////////////////////////////
  // Destructor
   ~ScalarSolver();

   // sets parameters at start time
   void problemSetup(const ProblemSpecP& params);

   // linearize eqn
   void sched_buildLinearMatrix(double delta_t, int index,
				const LevelP& level,
				SchedulerP& sched,
				const DataWarehouseP& old_dw,
				DataWarehouseP& new_dw);
 
   ////////////////////////////////////////////////////////////////////////
   // solve linearized momentum equation
   void solve(double time, double delta_t, int index, 
	      const LevelP& level,
	      SchedulerP& sched,
	      const DataWarehouseP& old_dw,
	      DataWarehouseP& new_dw);
   
 private:

   void buildLinearMatrix(const ProcessorContext* pc,
			  const Region* region,
			  const DataWarehouseP& old_dw,
			  DataWarehouseP& new_dw,
			  double delta_t, const int index);
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


};

    }
}

#endif
