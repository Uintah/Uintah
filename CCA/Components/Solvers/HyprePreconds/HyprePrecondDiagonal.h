#ifndef Packages_Uintah_CCA_Components_Solvers_HyprePrecondDiagonal_h
#define Packages_Uintah_CCA_Components_Solvers_HyprePrecondDiagonal_h

/*--------------------------------------------------------------------------
CLASS
   HyprePrecondDiagonal
   
   A Hypre Diagonal (diagonal scaling) preconditioner.

GENERAL INFORMATION

   File: HyprePrecondDiagonal.h

   Oren E. Livne
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2005 SCI Group

KEYWORDS
   HypreSolverBase, Precond, HypreSolverBase, HypreSolverParams.

DESCRIPTION 
   Class HyprePrecondDiagonal sets up and destroys the Hypre
   Diagonal preconditioner to be used with Hypre solvers. Diagonal runs a
   diagonal scaling so that diag(A) = I after scaling. It is suited for
   symmetric and non-symmetric matrices.

WARNING
   Works with Hypre Struct interface only.
   --------------------------------------------------------------------------*/

#include <Packages/Uintah/CCA/Components/Solvers/HyprePrecondBase.h>

namespace Uintah {
  
  //---------- Types ----------
  
  class HyprePrecondDiagonal : public HyprePrecondBase {

    //========================== PUBLIC SECTION ==========================
  public:
  
    HyprePrecondDiagonal(void) : HyprePrecondBase(initPriority()) {}
    virtual ~HyprePrecondDiagonal(void);

    virtual void setup(void);
    
    //========================== PROTECTED SECTION ==========================
  protected:
    static Priorities initPriority(void);

  }; // end class HyprePrecondDiagonal

} // end namespace Uintah

#endif // Packages_Uintah_CCA_Components_Solvers_HyprePrecondDiagonal_h
