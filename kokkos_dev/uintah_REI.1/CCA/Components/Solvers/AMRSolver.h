#ifndef Packages_Uintah_CCA_Components_Solvers_AMRSolver_h
#define Packages_Uintah_CCA_Components_Solvers_AMRSolver_h

/*--------------------------------------------------------------------------
CLASS
   AMRSolver
   
   A Hypre solver component for AMR grids.

GENERAL INFORMATION

   File: AMRSolver.h

   Oren E. Livne
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2005 SCI Group

KEYWORDS
  AMRSolver, HypreDriver, HypreSolverParams, HypreSolverBase.

DESCRIPTION 
   Class AMRSolver is the main solver component that
   interfaces to Hypre's structured and semi-structured system
   interfaces.
  
WARNING
   * This interface is written for Hypre 1.9.0b (released 2005).
   --------------------------------------------------------------------------*/

#include <Packages/Uintah/CCA/Ports/SolverInterface.h>
#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>

namespace Uintah {
  class AMRSolver :
    public SolverInterface, public UintahParallelComponent { 

   
  public:

    AMRSolver(const ProcessorGroup* myworld);
    virtual ~AMRSolver();

    virtual SolverParameters* readParameters(ProblemSpecP& params,
                                             const std::string& name);

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
                               
    virtual string getName();
    
    private:
  };
}

#endif 
