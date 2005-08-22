#ifndef Packages_Uintah_CCA_Components_Solvers_HypreSolverGMRES_h
#define Packages_Uintah_CCA_Components_Solvers_HypreSolverGMRES_h

/*--------------------------------------------------------------------------
CLASS
   HypreSolverGMRES
   
   A Hypre GMRES (generalized minimum residual) solver.

GENERAL INFORMATION

   File: HypreSolverGMRES.h

   Oren E. Livne
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2005 SCI Group

KEYWORDS
   HypreDriver, HypreSolverBase, HypreSolverParams.

DESCRIPTION
   Class HyprePrecondGMRES sets up and destroys the Hypre GMRES solver.
   It can optionally employ a preconditioner. Unlike CG and Hybrid, it can
   work with non-symmetric matrices.

WARNING
   Works with Hypre Struct interface only.
   --------------------------------------------------------------------------*/

#include <Packages/Uintah/CCA/Components/Solvers/HypreSolverBase.h>

namespace Uintah {
  
  class HypreDriver;

  //---------- Types ----------
  
  class HypreSolverGMRES : public HypreSolverBase {

    //========================== PUBLIC SECTION ==========================
  public:
  
    HypreSolverGMRES(HypreDriver* driver,
                    HyprePrecondBase* precond) :
      HypreSolverBase(driver,precond,initPriority()) {}
    virtual ~HypreSolverGMRES(void) {}

    virtual void solve(void);

    //========================== PRIVATE SECTION ==========================
  private:
    static Priorities initPriority(void);

  }; // end class HypreSolverGMRES

} // end namespace Uintah

#endif // Packages_Uintah_CCA_Components_Solvers_HypreSolverGMRES_h
