/*--------------------------------------------------------------------------
CLASS
   HypreSolverAMR
   
   A Hypre solver component for AMR grids.

GENERAL INFORMATION

   File: HypreSolverAMR.h

   Oren E. Livne
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2005 SCI Group

KEYWORDS
  HypreSolverAMR, HypreDriver, HypreSolverParams, HypreGenericSolver.

DESCRIPTION 
   Class HypreSolverAMR is the main solver component that
   interfaces to Hypre's structured and semi-structured system
   interfaces. It requires Uintah data from the ICE component
   (currently implemented for the elliptic pressure equation in
   implicit ICE AMR mode. HypreSolverAMR schedules a task in sus
   called scheduleSolve() that carries out the solve operation. It is
   based on the one-level solver class HypreSolver2.
  
WARNING
   * This interface is written for Hypre 1.9.0b (released 2005). However,
   it may still work with the Struct solvers in earlier Hypre versions (e.g., 
   1.7.7).
   --------------------------------------------------------------------------*/
#ifndef Packages_Uintah_CCA_Components_Solvers_HypreSolverAMR_h
#define Packages_Uintah_CCA_Components_Solvers_HypreSolverAMR_h

#include <Packages/Uintah/CCA/Ports/SolverInterface.h>
#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>

namespace Uintah {
  class HypreSolverAMR :
    public SolverInterface, public UintahParallelComponent { 

    //========================== PUBLIC SECTION ==========================
  public:

    HypreSolverAMR(const ProcessorGroup* myworld);
    virtual ~HypreSolverAMR();

    // Load solver parameters from input struct
    virtual SolverParameters* readParameters(ProblemSpecP& params,
                                             const std::string& name);

    // Main task that solves the pressure equation and returns
    // cell-centered pressure. In the future we can also implement
    // here solutions of other variable types, like node-centered.
    virtual void scheduleSolve(const LevelP& level, SchedulerP& sched,
                               const MaterialSet* matls,
                               const VarLabel* A,    
                               Task::WhichDW which_A_dw,  
                               const VarLabel* x,
                               bool modifies_x,
                               const VarLabel* b,    
                               Task::WhichDW which_b_dw,  
                               const VarLabel* guess,
                               Task::WhichDW guess_dw,
                               const SolverParameters* params);
    
    //========================== PRIVATE SECTION ==========================
    private:
  };
}

#endif // Packages_Uintah_CCA_Components_Solvers_HypreSolverAMR_h
