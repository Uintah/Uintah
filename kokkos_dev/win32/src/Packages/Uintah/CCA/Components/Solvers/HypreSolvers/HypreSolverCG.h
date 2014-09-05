#ifndef Packages_Uintah_CCA_Components_Solvers_HypreSolverCG_h
#define Packages_Uintah_CCA_Components_Solvers_HypreSolverCG_h

/*--------------------------------------------------------------------------
CLASS
   HypreSolverCG
   
   A Hypre CG (conjugate gradient) solver.

GENERAL INFORMATION

   File: HypreSolverCG.h

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

#include <Packages/Uintah/CCA/Components/Solvers/HypreSolvers/HypreSolverBase.h>

namespace Uintah {
  
  class HypreDriver;

  //---------- Types ----------
  
  class HypreSolverCG : public HypreSolverBase {

    //========================== PUBLIC SECTION ==========================
  public:
  
    HypreSolverCG(HypreDriver* driver,
                  HyprePrecondBase* precond) :
      HypreSolverBase(driver,precond,initPriority()) {}
    virtual ~HypreSolverCG(void) {}

    virtual void solve(void);

    //========================== PRIVATE SECTION ==========================
  private:
    static Priorities initPriority(void);

  }; // end class HypreSolverCG

} // end namespace Uintah

#endif // Packages_Uintah_CCA_Components_Solvers_HypreSolverCG_h
