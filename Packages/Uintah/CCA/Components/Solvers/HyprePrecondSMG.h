/*--------------------------------------------------------------------------
CLASS
   HyprePrecondSMG
   
   A generic Hypre preconditioner driver.

GENERAL INFORMATION

   File: HyprePrecondSMG.h

   Oren E. Livne
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2005 SCI Group

KEYWORDS
   HypreDriver, HypreSolverParams.

DESCRIPTION
   Class HyprePrecondSMG is a base class for Hypre solvers. It uses the
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
#ifndef Packages_Uintah_CCA_Components_Solvers_HyprePrecondSMG_h
#define Packages_Uintah_CCA_Components_Solvers_HyprePrecondSMG_h

#include <Packages/Uintah/CCA/Components/Solvers/HypreGenericPrecond.h>

namespace Uintah {
  
  //---------- Types ----------
  
  class HyprePrecondSMG : public HypreGenericPrecond {

    //========================== PUBLIC SECTION ==========================
  public:
  
    HyprePrecondSMG(void) : HypreGenericPrecond(initPriority()) {}
    virtual ~HyprePrecondSMG(void);

    virtual void setup(void);
    
    //========================== PROTECTED SECTION ==========================
  protected:
    static Priorities initPriority(void);

  }; // end class HyprePrecondSMG

} // end namespace Uintah

#endif // Packages_Uintah_CCA_Components_Solvers_HyprePrecondSMG_h
