#ifndef Packages_Uintah_CCA_Components_Solvers_HypreSolverSMG_h
#define Packages_Uintah_CCA_Components_Solvers_HypreSolverSMG_h

/*--------------------------------------------------------------------------
CLASS
   HypreSolverSMG
   
   A Hypre SMG (geometric multigrid #1) solver.

GENERAL INFORMATION

   File: HypreSolverSMG.h

   Oren E. Livne
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2005 SCI Group

KEYWORDS
   HypreDriver, HypreSolverBase, HypreSolverParams.

DESCRIPTION
   Class HyprePrecondSMG sets up and destroys the Hypre SMG
   solver to be used with Hypre solvers. SMG is a geometric multigrid
   solver well suited for Poisson or a diffusion operator with a
   smooth diffusion coefficient.  SMG is used as to solve or as a
   preconditioner is often used with CG or GMRES.

WARNING
      Works with Hypre Struct interface only.
   --------------------------------------------------------------------------*/

#include <Packages/Uintah/CCA/Components/Solvers/HypreSolvers/HypreSolverBase.h>

namespace Uintah {
  
  class HypreDriver;

  //---------- Types ----------
  
  class HypreSolverSMG : public HypreSolverBase {

    //========================== PUBLIC SECTION ==========================
  public:
  
    HypreSolverSMG(HypreDriver* driver,
                    HyprePrecondBase* precond) :
      HypreSolverBase(driver,precond,initPriority()) {}
    virtual ~HypreSolverSMG(void) {}

    virtual void solve(void);

    //========================== PRIVATE SECTION ==========================
  private:
    static Priorities initPriority(void);

  }; // end class HypreSolverSMG

} // end namespace Uintah

#endif // Packages_Uintah_CCA_Components_Solvers_HypreSolverSMG_h
