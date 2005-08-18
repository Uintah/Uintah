#ifndef Packages_Uintah_CCA_Components_Solvers_HyprePrecondSMG_h
#define Packages_Uintah_CCA_Components_Solvers_HyprePrecondSMG_h

/*--------------------------------------------------------------------------
CLASS
   HyprePrecondSMG
   
   A Hypre SMG (geometric multigrid #1) preconditioner.

GENERAL INFORMATION

   File: HyprePrecondSMG.h

   Oren E. Livne
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2005 SCI Group

KEYWORDS
   HypreGenericSolver, Precond, HypreGenericSolver, HypreSolverParams.

DESCRIPTION
   Class HyprePrecondSMG sets up and destroys the Hypre SMG preconditioner
   to be used with Hypre solvers. SMG preconditioner is often used with CG
   or GMRES.
  
WARNING
   Works with Hypre Struct interface only.
   --------------------------------------------------------------------------*/

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
