/**************************************
CLASS
   ImplicitTimeIntegrator
   
   Class ImplicitIntegrator controls the initialization 
   and time integration over the entire mesh hierarchy for a 
   time dependent problem.  

GENERAL INFORMATION
   ImplicitTimeIntegrator.h - declaration of the class
   
   Author: Rajesh Rawat (rawat@crsim.utah.edu)
   
   Creation Date:   Mar 1, 2000
   
   C-SAFE 
   
   Copyright U of U 2000

KEYWORDS


DESCRIPTION
   This class provides routines needed to integrate a system 
   of N-S equations on a mesh hierarchy using
   a simple implicit time stepping scheme  
   with a global time step (no local time stepping). 
   Methods in the class initialize the hierarhcy, determine advance
   timestep, advance the hierarchy data through one timestep.


WARNING
none
****************************************/

#ifndef included_ImplicitTimeIntegrator
#define included_ImplicitTimeIntegrator

#incldue <sgi_stl_warnings_off.h>
#include <iosfwd>
#incldue <sgi_stl_warnings_on.h>
#include "NonlinearSolver.h"


#ifndef NULL
#define NULL (0)
#endif
#ifndef UNDEFINED
#define UNDEFINED -1
#endif

class ImplicitTimeIntegrator:
{
public:
  //////////////////////////////////////////////////////////////////////// 
  // Constructor for ImplicitTimeIntegrator initializes
  // integration parameters to default values.  Other data
  //  members are read in from
  // the specified input or restart databases.
  // Do we've support for smart pointers in ?
  // hierarchy: pointer to a mesh in this case it would be single level
  ImplicitTimeIntegrator(
      Pointer<Database> input_db,
      Pointer<Database> restart_db,
      NonlinearSolver* solver,
      Pointer<PatchHierarchy> hierarchy);

  // virtual destructor
  virtual ~ImplicitTimeIntegrator();

  /**
   * Return a suitable time increment over which to integrate the
   * patches in the hierarchy.  A minimum is taken over the increment 
   * computed on each patch in the hierarchy.
   */
  double getTimestep(
     const Pointer<PatchHierarchy> hierarchy,
     const double time ) const;

  // GROUP: Access Functions:
  ////////////////////////////////////////////////////////////////////////
  // Read input data from specified database and initialize integrator.
  virtual void getFromInput(Pointer<Database> input_db,
			    bool is_from_restart);

  /**
   * Read in values from restart database.
   *
   * When assertion checking is active, the database pointer must be non-null.
   */
  virtual void getFromRestart(tbox_Pointer<tbox_Database> db);
 
  //////////////////////////////////////////////////////////////////////// 
  // Advance the hierarchy through to specified time and 
  // return subsequent time increment.  
  double advanceHierarchy(const Pointer<PatchHierarchy> hierarchy,
			  const double time, const double dt,
			  int &errfail);
 
  ////////////////////////////////////////////////////////////////////////  
  // Advance the hierarchy to the final time by continually calling
  // advanceHierarchy.
  int solveHierarchy(double end_time);

  ////////////////////////////////////////////////////////////////////////  
  // Initialize the data on a level at simulation start time.
  virtual void initializeLevelData(const Pointer<PatchHierarchy> 
				   hierarchy, const double startTime);

  ////////////////////////////////////////////////////////////////////////  
  // Return const reference to the integration time for the hierarchy.
  // Note that this is the time at the time step being solved for.
  const double& getIntegratorTime() const;

  ////////////////////////////////////////////////////////////////////////  
  // Return const reference to current time for the hierarchy.
  // Note that this is the time of the last succesfull step, use 
  // getIntegratorTime to get the current time being solved for. 
  const double& getCurrentTime() const;

  ////////////////////////////////////////////////////////////////////////  
  // Return const reference to initial integration time.
  const double& getStartTime() const;
 
  //////////////////////////////////////////////////////////////////////// 
  // Return const reference to final integration time.
  const double& getEndTime() const;

  //////////////////////////////////////////////////////////////////////// 
  // Return const reference to current timestep being used for time 
  // integration on the hierarchy.
  const double& getDtActual() const; 

  ////////////////////////////////////////////////////////////////////////  
  // Return const reference to patch hierarchy pointer.
  const Pointer<PatchHierarchy>& getPatchHierarchy() const;

  //////////////////////////////////////////////////////////////////////// 
  // Shut down the ImplicitIntegrator 
  void finalize();

  //////////////////////////////////////////////////////////////////////// 
  // Print out data representation of the class when an unrecoverable 
  // run-time exception is thrown.
  void printClassData(ostream& os) const; 
  
private:

  /*
   * Patch hierarchy over which integrator performs integration.
   */
  Pointer<PatchHierarchy> d_patch_hierarchy; 
  
  /*
   * The nonlinear solver.
   */
  NonlinearSolver* d_solver;

  /*
   * Integrator data read from input or set at initialization.
   */
  double d_start_time;
  double d_end_time;
  double d_delta_dt;
  double d_max_dt;
  double d_current_time;
};
#endif  
  
