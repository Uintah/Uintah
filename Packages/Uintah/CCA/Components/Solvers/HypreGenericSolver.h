#ifndef Packages_Uintah_CCA_Components_Solvers_HypreGenericSolver_h
#define Packages_Uintah_CCA_Components_Solvers_HypreGenericSolver_h

/*--------------------------------------------------------------------------
CLASS
   HypreGenericSolver
   
   A generic Hypre solver driver.

GENERAL INFORMATION

   File: HypreGenericSolver.h

   Oren E. Livne
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2005 SCI Group

KEYWORDS
   HypreDriver, HypreSolverParams.

DESCRIPTION
   Class HypreGenericSolver is a base class for Hypre solvers. It uses the
   generic HypreDriver and fetches only the data it can work with (either
   Struct, SStruct, or 

   preconditioners. It does not know about the internal Hypre interfaces
   like Struct and SStruct. Instead, it uses the generic HypreDriver
   and newSolver to determine the specific Hypre
   interface and solver, based on the parameters in HypreSolverParams.
   The solver is called through the solve() function. This is also the
   task-scheduled function in AMRSolver::scheduleSolve() that is
   activated by Components/ICE/impICE.cc.
  
WARNING
   solve() is a generic function for all Types, but this may need to change
   in the future. Currently only CC is implemented for the pressure solver
   in implicit [AMR] ICE.
   --------------------------------------------------------------------------*/

#include <Packages/Uintah/CCA/Ports/SolverInterface.h>
#include <Packages/Uintah/CCA/Components/Solvers/HypreSolverParams.h>
#include <Packages/Uintah/CCA/Components/Solvers/HypreTypes.h>

namespace Uintah {
  
  // Forward declarations
  class HypreGenericPrecond;
  class HypreDriver;

  //---------- Types ----------
  
  class HypreGenericSolver {

    //========================== PUBLIC SECTION ==========================
  public:
  
    // Solver results are output to this struct 
    struct Results {
      int        numIterations;   // Number of solver iterations performed
      double     finalResNorm;    // Final residual norm ||A*x-b||_2
    };
    
    HypreGenericSolver(HypreDriver* driver,
                       HypreGenericPrecond* precond,
                       const Priorities& priority);
    virtual ~HypreGenericSolver(void);

    // Data member unmodifyable access
    HypreDriver*       getDriver(void) { return _driver; }

    // Data member unmodifyable access
    const HypreDriver* getDriver(void) const { return _driver; }
    const Results&     getResults(void) const { return _results; }
    const bool&        requiresPar(void) const { return _requiresPar; }

    void         assertInterface(void);
    virtual void solve(void) = 0;

    //========================== PROTECTED SECTION ==========================
  protected:

    //---------- Data members ----------
    HypreDriver*             _driver;        // Hypre data containers
    HypreGenericPrecond*     _precond;       // Preconditioner (optional)
    Priorities               _priority;      // Prioritized acceptable drivers
    bool                     _requiresPar;   // Do we need PAR or not?
    Results                  _results;       // Solver results stored here
    
 }; // end class HypreGenericSolver

  // Utilities
  HypreGenericSolver* newHypreSolver(const SolverType& solverType,
                                     HypreDriver* driver,
                                     HypreGenericPrecond* precond);
  SolverType          getSolverType(const std::string& solverTitle);

} // end namespace Uintah

#endif // Packages_Uintah_CCA_Components_Solvers_HypreGenericSolver_h
