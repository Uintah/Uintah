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
   PressureSolver(TurbulenceModel* turb_model, BoundaryCondition* bndry_cond);

  // GROUP: Destructors:
  ////////////////////////////////////////////////////////////////////////
  // Destructor
   ~PressureSolver();

   // access functions
   CCVariable<double> getPressure() const;
   double getResidual() const;
   double getOrderMagnitude() const;
   // sets parameters at start time
   void problemSetup(const ProblemSpecP& params);

   // linearize eqn
   void buildLinearMatrix(const LevelP& level,
			  SchedulerP& sched,
			  const DataWarehouseP& old_dw,
			  DataWarehouseP& new_dw);
 
   ////////////////////////////////////////////////////////////////////////
   // solve linearized pressure equation
   void solve(const LevelP& level,
	      SchedulerP& sched,
	      const DataWarehouseP& old_dw,
	      DataWarehouseP& new_dw);
   
    void sched_modifyCoeff(const LevelP& level,
			   SchedulerP& sched,
			   const DataWarehouseP& old_dw,
			   DataWarehouseP& new_dw);
  
 private:

   // functions
   void calculateResidual(const LevelP& level,
			  SchedulerP& sched,
			  const DataWarehouseP& old_dw,
			  DataWarehouseP& new_dw);

    void calculateOrderMagnitude(const LevelP& level,
				 SchedulerP& sched,
				 const DataWarehouseP& old_dw,
				 DataWarehouseP& new_dw);
      // Modify coefficients
   void modifyCoeff(const Region* region,
		    SchedulerP& sched,
		    const DataWarehouseP& old_dw,
		    DataWarehouseP& new_dw);

   void underrelaxEqn(const LevelP& level,
		      SchedulerP& sched,
		      const DataWarehouseP& old_dw,
		      DataWarehouseP& new_dw);

   // computes coefficients
   Discretization* d_discretize;

   // computes sources
   Source* d_source;

   // computes boundary conditions
   BoundaryCondition* d_boundaryCondition;

   // linear solver
   LinearSolver* d_linearSolver;

   // turbulence model
   TurbulenceModel* d_turbModel;
   // boundary condition
   BoundaryCondition* d_boundaryCondition;

// GROUP: Data members.

   ////////////////////////////////////////////////////////////////////////
   // Maximum number of iterations to take before stopping/giving up.
   int d_maxIterations;

   // underrealaxation parameter, read from an input database
   double d_underrelax;
   //reference points for the solvers
   int d_ipref, d_jpref, d_kpref;

};

    }
}

#endif
