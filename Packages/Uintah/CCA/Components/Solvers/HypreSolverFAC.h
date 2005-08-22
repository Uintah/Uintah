#ifndef Packages_Uintah_CCA_Components_Solvers_HypreSolverFAC_h
#define Packages_Uintah_CCA_Components_Solvers_HypreSolverFAC_h

/*--------------------------------------------------------------------------
CLASS
   HypreSolverFAC
   
   A Hypre CG (conjugate gradient) solver.

GENERAL INFORMATION

   File: HypreSolverFAC.h

   Oren E. Livne
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2005 SCI Group

KEYWORDS
   HypreDriver, HypreGenericSolver, HypreSolverParams.

DESCRIPTION
   Class HyprePrecondCG sets up and destroys the Hypre conjugate gradient
   solver. It can optionally employ a preconditioner.

WARNING
   Works with Hypre Struct interface only.
   --------------------------------------------------------------------------*/

#include <Packages/Uintah/CCA/Components/Solvers/HypreGenericSolver.h>

namespace Uintah {
  
  class HypreDriver;

  //---------- Types ----------
  
  class HypreSolverFAC : public HypreGenericSolver {

    //========================== PUBLIC SECTION ==========================
  public:
  
    HypreSolverFAC(HypreDriver* driver,
                    HypreGenericPrecond* precond) :
      HypreGenericSolver(driver,precond,initPriority()) {}
    virtual ~HypreSolverFAC(void) {}

    virtual void solve(void);

    //========================== PRIVATE SECTION ==========================
  private:
    static Priorities initPriority(void);

  }; // end class HypreSolverFAC

} // end namespace Uintah

#endif // Packages_Uintah_CCA_Components_Solvers_HypreSolverFAC_h
