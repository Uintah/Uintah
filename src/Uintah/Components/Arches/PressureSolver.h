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




#ifndef LACKS_NAMESPACE
using namespace UINTAH;
#endif

class Discretization;
class Source;
class BoundaryCondition;
class LinearSolver;

class PressureSolver :
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

  // GROUP: Destructors:
  ////////////////////////////////////////////////////////////////////////
  // Destructor
   ~PressureSolver();

   // access functions
   const CCVariable <vector> getPressure() const;
   const double getResidual() const;
   const double getOrderMagnitude() const;
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

   calculateOrderMagnitude(const LevelP& level,
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

   ////////////////////////////////////////////////////////////////////////
   // Read input data from specified database and initializes solver.
   void getFromInput(Pointer<Database> input_db,
			    bool is_from_restart);

   // computes coefficients
   Discretization* d_discretize;

   // computes sources
   Source* d_source;

   // computes boundary conditions
   BoundaryCondition* d_bndryCond;

   // linear solver
   LinearSolver* d_linearSolver;

// GROUP: Data members.

   ////////////////////////////////////////////////////////////////////////
   // Maximum number of iterations to take before stopping/giving up.
   int d_maxIterations;
   // non-linear L-1 norm residual, computed before linear solve
   SoleVariable<double> d_residual;
   // order of magnitude term, required to normalize residual
   SoleVariable<double>  d_orderMagnitude;
   // underrealaxation parameter, read from an input database
   double d_underrelax;
   //reference points for the solvers
   int d_ipref, d_jpref, d_kpref;

};
#endif
