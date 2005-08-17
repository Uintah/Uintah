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

  //---------- Types ----------
  
  class HypreGenericPrecond {

    //========================== PUBLIC SECTION ==========================
  public:
  
    virtual ~HypreGenericPrecond(void) {}

    void assertInterface(const int acceptableInterface);

    //========================== PROTECTED SECTION ==========================
  protected:

    //---------- Data members ----------
    HypreGenericSolver*      _solver;       // Hypre system interface
    const ProcessorGroup*    _pg;
    const HypreSolverParams* _params;
    HYPRE_PtrToSolverFcn     _precond;
    HYPRE_PtrToSolverFcn     _pcsetup;
    HYPRE_Solver             _precond_solver;

    //========================== PRIVATE SECTION ==========================
  private:
    // Ensure that this class cannot be instantiated
    HypreGenericPrecond(const HypreInterface& interface,
                 const ProcessorGroup* pg,
                 const HypreSolverParams* params,
                 const int acceptableInterface)
      {
        assertInterface(acceptableInterface);
      }

  }; // end class HypreGenericPrecond

  // Utilities
  HypreGenericPrecond* newHypreGenericPrecond(const PrecondType& precondType,
                                const HypreInterface& interface,
                                const ProcessorGroup* pg,
                                const HypreSolverParams* params);
  PrecondType   getPrecondType(const std::string& precondTitle);

} // end namespace Uintah

#endif // Packages_Uintah_CCA_Components_Solvers_HypreGenericPrecond_h
