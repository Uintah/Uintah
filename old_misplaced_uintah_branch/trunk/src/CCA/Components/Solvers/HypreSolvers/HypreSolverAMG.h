#ifndef Packages_Uintah_CCA_Components_Solvers_HypreSolverAMG_h
#define Packages_Uintah_CCA_Components_Solvers_HypreSolverAMG_h

/*--------------------------------------------------------------------------
CLASS
   HypreSolverAMG
   
   A Hypre CG (conjugate gradient) solver.

GENERAL INFORMATION

   File: HypreSolverAMG.h

   Oren E. Livne
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2005 SCI Group

KEYWORDS
   HypreDriver, HypreSolverBase, HypreSolverParams.

DESCRIPTION
   Class HypreSolverCG sets up and destroys the Hypre conjugate gradient
   solver. It can optionally employ a preconditioner.

WARNING
   Works with Hypre Struct interface only.
   --------------------------------------------------------------------------*/

#include <CCA/Components/Solvers/HypreSolvers/HypreSolverBase.h>

namespace Uintah {
  
  class HypreDriver;

  //---------- Types ----------
  
  class HypreSolverAMG : public HypreSolverBase {

    //========================== PUBLIC SECTION ==========================
  public:
  
    HypreSolverAMG(HypreDriver* driver,
                    HyprePrecondBase* precond) :
      HypreSolverBase(driver,precond,initPriority()) {}
    virtual ~HypreSolverAMG(void) {}

    virtual void solve(void);

    //========================== PRIVATE SECTION ==========================
  private:
    static Priorities initPriority(void);

  }; // end class HypreSolverAMG

} // end namespace Uintah

#endif // Packages_Uintah_CCA_Components_Solvers_HypreSolverAMG_h
