/*--------------------------------------------------------------------------
CLASS
   HyprePrecond
   
   A generic Hypre preconditioner driver.

GENERAL INFORMATION

   File: HyprePrecond.h

   Oren E. Livne
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2005 SCI Group

KEYWORDS
   HypreDriver, HypreSolverParams.

DESCRIPTION
   Class HyprePrecond is a base class for Hypre solvers. It uses the
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
#ifndef Packages_Uintah_CCA_Components_Solvers_HyprePrecond_h
#define Packages_Uintah_CCA_Components_Solvers_HyprePrecond_h

#include <Packages/Uintah/CCA/Components/Solvers/HypreDriver.h>

namespace Uintah {
  
  // Forward declarations
  template <class Types> class HypreDriver;

  //---------- Types ----------
  
  enum PrecondType {
    PrecondNA, // No preconditioner, use solver directly
    PrecondSMG, PrecondPFMG, PrecondSparseMSG, PrecondJacobi,
    PrecondDiagonal, PrecondAMG, PrecondFAC
  };
  
  class HyprePrecond {

    //========================== PUBLIC SECTION ==========================
  public:
  
    HyprePrecond(const HypreInterface& interface,
                 const ProcessorGroup* pg,
                 const HypreSolverParams* params,
                 const int acceptableInterface);
    virtual ~HyprePrecond(void);

    void         assertInterface(const int acceptableInterface);
    virtual void setup(void) = 0;
    virtual void destroy(void) = 0;

    //========================== PROTECTED SECTION ==========================
  protected:

    //---------- Data members ----------
    HypreInterface           _interface;       // Hypre system interface
    const ProcessorGroup*    _pg;
    const HypreSolverParams* _params;
    //    PrecondType              _precondType;     // Hypre preconditioner type
    HYPRE_PtrToSolverFcn     _precond;
    HYPRE_PtrToSolverFcn     _pcsetup;
    HYPRE_Solver             _precond_solver;

  }; // end class HyprePrecond

  // Utilities
  PrecondType   precondFromTitle(const std::string& precondTitle);
  HyprePrecond* newHyprePrecond(const PrecondType& precondType,
                                const HypreInterface& interface,
                                const ProcessorGroup* pg,
                                const HypreSolverParams* params);
} // end namespace Uintah

#endif // Packages_Uintah_CCA_Components_Solvers_HyprePrecond_h
