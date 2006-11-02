#ifndef Packages_Uintah_CCA_Components_Solvers_HyprePrecondJacobi_h
#define Packages_Uintah_CCA_Components_Solvers_HyprePrecondJacobi_h

/*--------------------------------------------------------------------------
CLASS
   HyprePrecondJacobi
   
   A Hypre Jacobi (Jacobi relaxation) preconditioner.

GENERAL INFORMATION

   File: HyprePrecondJacobi.h

   Oren E. Livne
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2005 SCI Group

KEYWORDS
   HypreSolverBase, Precond, HypreSolverBase, HypreSolverParams.

DESCRIPTION 
   Class HyprePrecondJacobi sets up and destroys the Hypre
   Jacobi preconditioner to be used with Hypre solvers. Jacobi runs a few
   Jacobi relaxation sweeps (I - D^{-1} A for a matrix A with diagonal D).
   is the distributed sparse linear solver. Jacobi preconditioner
   is often used with CG (I think it works only for symmetric matrices).
  
WARNING
   Works with Hypre Struct interface only and probably good only for symmetric
   matrices.
   --------------------------------------------------------------------------*/

#include <Packages/Uintah/CCA/Components/Solvers/HyprePreconds/HyprePrecondBase.h>

namespace Uintah {
  
  //---------- Types ----------
  
  class HyprePrecondJacobi : public HyprePrecondBase {

    //========================== PUBLIC SECTION ==========================
  public:
  
    HyprePrecondJacobi(void) : HyprePrecondBase(initPriority()) {}
    virtual ~HyprePrecondJacobi(void);

    virtual void setup(void);
    
    //========================== PROTECTED SECTION ==========================
  protected:
    static Priorities initPriority(void);

  }; // end class HyprePrecondJacobi

} // end namespace Uintah

#endif // Packages_Uintah_CCA_Components_Solvers_HyprePrecondJacobi_h
