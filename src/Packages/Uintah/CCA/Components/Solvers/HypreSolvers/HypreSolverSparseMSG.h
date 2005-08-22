#ifndef Packages_Uintah_CCA_Components_Solvers_HypreSolverSparseMSG_h
#define Packages_Uintah_CCA_Components_Solvers_HypreSolverSparseMSG_h

/*--------------------------------------------------------------------------
CLASS
   HypreSolverSparseMSG
   
   A Hypre SparseMSG (distributed sparse linear solver?) solver.

GENERAL INFORMATION

   File: HypreSolverSparseMSG.h

   Oren E. Livne
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2005 SCI Group

KEYWORDS
   HypreDriver, HypreSolverBase, HypreSolverParams.

DESCRIPTION
   Class HyprePrecondSparseMSG sets up and destroys the Hypre SparseMSG
   solver to be used with Hypre solvers.

WARNING
      Works with Hypre Struct interface only.
   --------------------------------------------------------------------------*/

#include <Packages/Uintah/CCA/Components/Solvers/HypreSolverBase.h>

namespace Uintah {
  
  class HypreDriver;

  //---------- Types ----------
  
  class HypreSolverSparseMSG : public HypreSolverBase {

    //========================== PUBLIC SECTION ==========================
  public:
  
    HypreSolverSparseMSG(HypreDriver* driver,
                    HyprePrecondBase* precond) :
      HypreSolverBase(driver,precond,initPriority()) {}
    virtual ~HypreSolverSparseMSG(void) {}

    virtual void solve(void);

    //========================== PRIVATE SECTION ==========================
  private:
    static Priorities initPriority(void);

  }; // end class HypreSolverSparseMSG

} // end namespace Uintah

#endif // Packages_Uintah_CCA_Components_Solvers_HypreSolverSparseMSG_h
