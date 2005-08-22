#ifndef Packages_Uintah_CCA_Components_Solvers_HyprePrecondSparseMSG_h
#define Packages_Uintah_CCA_Components_Solvers_HyprePrecondSparseMSG_h

/*--------------------------------------------------------------------------
CLASS
   HyprePrecondSparseMSG
   
   A Hypre SparseMSG (distributed sparse linear solver?) preconditioner.

GENERAL INFORMATION

   File: HyprePrecondSparseMSG.h

   Oren E. Livne
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2005 SCI Group

KEYWORDS
   HypreSolverBase, Precond, HypreSolverBase, HypreSolverParams.

DESCRIPTION 
   Class HyprePrecondSparseMSG sets up and destroys the Hypre
   SparseMSG preconditioner to be used with Hypre solvers. SparseMSG
   is the distributed sparse linear solver. SparseMSG preconditioner
   is often used with CG or GMRES.
  
WARNING
   Works with Hypre Struct interface only.
   --------------------------------------------------------------------------*/

#include <Packages/Uintah/CCA/Components/Solvers/HyprePrecondBase.h>

namespace Uintah {
  
  //---------- Types ----------
  
  class HyprePrecondSparseMSG : public HyprePrecondBase {

    //========================== PUBLIC SECTION ==========================
  public:
  
    HyprePrecondSparseMSG(void) : HyprePrecondBase(initPriority()) {}
    virtual ~HyprePrecondSparseMSG(void);

    virtual void setup(void);
    
    //========================== PROTECTED SECTION ==========================
  protected:
    static Priorities initPriority(void);

  }; // end class HyprePrecondSparseMSG

} // end namespace Uintah

#endif // Packages_Uintah_CCA_Components_Solvers_HyprePrecondSparseMSG_h
