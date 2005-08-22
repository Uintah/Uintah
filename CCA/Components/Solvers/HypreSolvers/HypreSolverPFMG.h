#ifndef Packages_Uintah_CCA_Components_Solvers_HypreSolverPFMG_h
#define Packages_Uintah_CCA_Components_Solvers_HypreSolverPFMG_h

/*--------------------------------------------------------------------------
CLASS
   HypreSolverPFMG
   
   A Hypre PFMG (geometric multigrid #2) solver.

GENERAL INFORMATION

   File: HypreSolverPFMG.h

   Oren E. Livne
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2005 SCI Group

KEYWORDS
   HypreDriver, HypreSolverBase, HypreSolverParams.

DESCRIPTION
   Class HyprePrecondPFMG sets up and destroys the Hypre PFMG
   solver to be used with Hypre solvers. PFMG is the Parallel
   Full Multigrid solver that uses geometric interpolation and
   restriction transfers and well suited for Poisson or a diffusion
   operator with a smooth diffusion coefficient.
   PFMG is used as to solve or as a
   preconditioner is often used with CG or GMRES.

WARNING
      Works with Hypre Struct interface only.
   --------------------------------------------------------------------------*/

#include <Packages/Uintah/CCA/Components/Solvers/HypreSolverBase.h>

namespace Uintah {
  
  class HypreDriver;

  //---------- Types ----------
  
  class HypreSolverPFMG : public HypreSolverBase {

    //========================== PUBLIC SECTION ==========================
  public:
  
    HypreSolverPFMG(HypreDriver* driver,
                    HyprePrecondBase* precond) :
      HypreSolverBase(driver,precond,initPriority()) {}
    virtual ~HypreSolverPFMG(void) {}

    virtual void solve(void);

    //========================== PRIVATE SECTION ==========================
  private:
    static Priorities initPriority(void);

  }; // end class HypreSolverPFMG

} // end namespace Uintah

#endif // Packages_Uintah_CCA_Components_Solvers_HypreSolverPFMG_h
