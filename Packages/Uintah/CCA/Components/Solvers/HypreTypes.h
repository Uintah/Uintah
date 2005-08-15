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
   HYPRE_Struct, HYPRE_SStruct, HYPRE_ParCSR,
   HypreGenericSolver, HypreSolverParams, RefCounted, solve, HypreSolverAMR.

DESCRIPTION 
   Class HypreTypes is a wrapper for calling Hypre solvers
   and preconditioners. It allows access to multiple Hypre interfaces:
   Struct (structured grid), SStruct (composite AMR grid), ParCSR
   (parallel compressed sparse row representation of the matrix).
   Only one of them is activated per solver, by running
   makeLinearSystem(), thereby creating the objects (usually A,b,x) of
   the specific interface.  The solver can then be constructed from
   the interface data. If required by the solver, HypreTypes converts
   the data from one Hypre interface type to another.  HypreTypes is
   also responsible for deleting all Hypre objects.
   HypreGenericSolver::newSolver determines the specific Hypre
   interface and solver, based on the parameters in HypreSolverParams.
   HypreTypes::solve() is the task-scheduled function in
   HypreSolverAMR::scheduleSolve() that is activated by
   Components/ICE/impICE.cc.

WARNING
   * solve() is a generic function for all Types, but this *might* need
   to change in the future. Currently only CC is implemented for the
   pressure solver in implicit [AMR] ICE.
   * If we intend to use other Hypre system interfaces (e.g., IJ interface),
   their data types (Matrix, Vector, etc.) should be added to the data
   members of this class. Specific solvers deriving from HypreSolverGeneric
   should have a specific Hypre interface they work with that exists in
   HypreTypes.
   * Each new interface requires its own makeLinearSystem() -- construction
   of the linear system objects, and getSolution() -- getting the solution
   vector back to Uintah.
   * This interface is written for Hypre 1.9.0b (released 2005). However,
   it may still work with the Struct solvers in earlier Hypre versions (e.g., 
   1.7.7).
   --------------------------------------------------------------------------*/
#ifndef Packages_Uintah_CCA_Components_Solvers_HypreTypes_h
#define Packages_Uintah_CCA_Components_Solvers_HypreTypes_h

#include <Core/Thread/Time.h>
#include <Core/Util/DebugStream.h>
#include <Packages/Uintah/CCA/Ports/SolverInterface.h>
#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>
#include <Packages/Uintah/CCA/Components/Solvers/HypreGenericSolver.h>

namespace Uintah {

  // Forward declarations
  class HypreSolverParams;
  
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
