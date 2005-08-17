/*--------------------------------------------------------------------------
CLASS
   HypreGenericPrecond
   
   A generic Hypre preconditioner driver.

GENERAL INFORMATION

   File: HypreGenericPrecond.h

   Oren E. Livne
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2005 SCI Group

KEYWORDS
   HypreDriver, HypreSolverParams.

DESCRIPTION
   Class HypreGenericPrecond is a base class for Hypre solvers. It uses the
   generic HypreDriver and fetches only the data it can work with (either
   Struct, SStruct, or 

   preconditioners. It does not know about the internal Hypre interfaces
   like Struct and SStruct. Instead, it uses the generic HypreDriver
   and newSolver to determine the specific Hypre
   interface and solver, based on the parameters in HypreSolverParams.
   The solver is called through the solve() function. This is also the
   task-scheduled function in HypreSolverAMR::scheduleSolve() that is
   activated by Components/ICE/impICE.cc.
  
WARNING
   solve() is a generic function for all Types, but this may need to change
   in the future. Currently only CC is implemented for the pressure solver
   in implicit [AMR] ICE.
   --------------------------------------------------------------------------*/
#ifndef Packages_Uintah_CCA_Components_Solvers_HypreGenericPrecond_h
#define Packages_Uintah_CCA_Components_Solvers_HypreGenericPrecond_h

#include <Packages/Uintah/CCA/Ports/SolverInterface.h>
#include <Packages/Uintah/CCA/Components/Solvers/HypreTypes.h>

namespace Uintah {
  
  // Forward declarations
  class HypreSolverParams;
  class HypreGenericSolver;

  //---------- Types ----------
  
  class HypreGenericPrecond {

    //========================== PUBLIC SECTION ==========================
  public:
  
    HypreGenericPrecond(const Priorities& priority);
    virtual ~HypreGenericPrecond(void) {}

    // Data member modifyable access
    HypreGenericSolver*   getSolver(void) { return _solver; }
    HYPRE_PtrToSolverFcn& getPrecond(void) { return _precond; }
    HYPRE_PtrToSolverFcn& getPCSetup(void) { return _pcsetup; }
    HYPRE_Solver&         getPrecondSolver(void) { return _precond_solver; }
    Priorities&           getPriority(void) { return _priority; }
    
    // Data member unmodifyable access
    const HypreGenericSolver*   getSolver(void) const { return _solver; }
    const HYPRE_PtrToSolverFcn& getPrecond(void) const { return _precond; }
    const HYPRE_PtrToSolverFcn& getPCSetup(void) const { return _pcsetup; }
    const HYPRE_Solver&         getPrecondSolver(void) const { return _precond_solver; }
    const Priorities&           getPriority(void) const { return _priority; }

    virtual void setup(HypreGenericSolver* solver) = 0;

    //========================== PROTECTED SECTION ==========================
  protected:

    //---------- Data members ----------
    HypreGenericSolver*      _solver;         // The calling solver
    HYPRE_PtrToSolverFcn     _precond;        // Hypre ptr-to-function
    HYPRE_PtrToSolverFcn     _pcsetup;        // Hypre ptr-to-function
    HYPRE_Solver             _precond_solver; // Hypre precond object   
    Priorities               _priority;       // Prioritized acceptable drivers

  }; // end class HypreGenericPrecond

  // Utilities
  HypreGenericPrecond* newHyprePrecond(const PrecondType& precondType);
  PrecondType          getPrecondType(const std::string& precondTitle);

} // end namespace Uintah

#endif // Packages_Uintah_CCA_Components_Solvers_HypreGenericPrecond_h
