#ifndef Packages_Uintah_CCA_Components_Solvers_HypreSolverHybrid_h
#define Packages_Uintah_CCA_Components_Solvers_HypreSolverHybrid_h

/*--------------------------------------------------------------------------
CLASS
   HypreSolverHybrid
   
   A Hypre Hybrid (some hybrid conjugate gradient) solver.

GENERAL INFORMATION

   File: HypreSolverHybrid.h

   Oren E. Livne
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2005 SCI Group

KEYWORDS
   HypreDriver, HypreGenericSolver, HypreSolverParams.

DESCRIPTION
   Class HyprePrecondHybrid sets up and destroys the Hypre
   hybrid conjugate gradient solver. It can optionally employ a
   preconditioner.

WARNING
   Works with Hypre Struct interface only.
   --------------------------------------------------------------------------*/

#include <Packages/Uintah/CCA/Components/Solvers/HypreGenericSolver.h>

namespace Uintah {
  
  class HypreDriver;

  //---------- Types ----------
  
  class HypreSolverHybrid : public HypreGenericSolver {

    //========================== PUBLIC SECTION ==========================
  public:
  
    HypreSolverHybrid(HypreDriver* driver,
                    HypreGenericPrecond* precond) :
      HypreGenericSolver(driver,precond,initPriority()) {}
    virtual ~HypreSolverHybrid(void) {}

    virtual void solve(void);

    //========================== PRIVATE SECTION ==========================
  private:
    static Priorities initPriority(void);

  }; // end class HypreSolverHybrid

} // end namespace Uintah

#endif // Packages_Uintah_CCA_Components_Solvers_HypreSolverHybrid_h
