#ifndef Packages_Uintah_CCA_Components_Solvers_HypreTypes_h
#define Packages_Uintah_CCA_Components_Solvers_HypreTypes_h

/*--------------------------------------------------------------------------
CLASS
   HypreTypes
   
   Wrapper of a Hypre solver for a particular variable type.

GENERAL INFORMATION

   File: HypreTypes.h

   Oren E. Livne
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2005 SCI Group

KEYWORDS

DESCRIPTION 

WARNING

--------------------------------------------------------------------------*/

namespace Uintah {

  //---------- Types ----------
  
  enum HypreInterface {       // Hypre system interface for the solver
    HypreStruct      = 0x1,
    HypreSStruct     = 0x2,
    HypreParCSR      = 0x4,
    HypreInterfaceNA = 0x8
  };
  
  enum BoxSide {  // Left/right boundary in each dim
    LeftSide  = -1,
    RightSide = 1,
    BoxSideNA = 3
  };
  
  enum SolverType {
    SMG, PFMG, SparseMSG, CG, Hybrid, GMRES, AMG, FAC
  };
  
  enum PrecondType {
    PrecondNA, // No preconditioner, use solver directly
    PrecondSMG, PrecondPFMG, PrecondSparseMSG, PrecondJacobi,
    PrecondDiagonal, PrecondAMG, PrecondFAC
  };
  
} // end namespace Uintah

#endif // Packages_Uintah_CCA_Components_Solvers_HypreTypes_h
