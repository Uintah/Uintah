#ifndef Packages_Uintah_CCA_Components_Solvers_HyprePrecondPFMG_h
#define Packages_Uintah_CCA_Components_Solvers_HyprePrecondPFMG_h

/*--------------------------------------------------------------------------
CLASS
   HyprePrecondPFMG
   
   A Hypre PFMG (geometric multigrid #2) preconditioner.

GENERAL INFORMATION

   File: HyprePrecondPFMG.h

   Oren E. Livne
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2005 SCI Group

KEYWORDS
   HypreSolverBase, Precond, HypreSolverBase, HypreSolverParams.

DESCRIPTION
   Class HyprePrecondPFMG sets up and destroys the Hypre PFMG
   preconditioner to be used with Hypre solvers. PFMG is the Parallel
   Full Multigrid solver that uses geometric interpolation and
   restriction transfers and well suited for Poisson or a diffusion
   operator with a smooth diffusion coefficient.
   PFMG preconditioner is often used with CG or GMRES.
  
WARNING
   Works with Hypre Struct interface only.
   --------------------------------------------------------------------------*/

#include <Packages/Uintah/CCA/Components/Solvers/HyprePreconds/HyprePrecondBase.h>

namespace Uintah {
  
  //---------- Types ----------
  
  class HyprePrecondPFMG : public HyprePrecondBase {

    //========================== PUBLIC SECTION ==========================
  public:
  
    HyprePrecondPFMG(void) : HyprePrecondBase(initPriority()) {}
    virtual ~HyprePrecondPFMG(void);

    virtual void setup(void);
    
    //========================== PROTECTED SECTION ==========================
  protected:
    static Priorities initPriority(void);

  }; // end class HyprePrecondPFMG

} // end namespace Uintah

#endif // Packages_Uintah_CCA_Components_Solvers_HyprePrecondPFMG_h
