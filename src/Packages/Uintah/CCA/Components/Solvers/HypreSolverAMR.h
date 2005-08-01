/*--------------------------------------------------------------------------
 * File: HypreSolverAMR.h
 *
 * Header file for the interface to Hypre's semi-structured matrix
 * interface and corresponding solvers.  class receives Uintah data
 * for the elliptic pressure equation in implicit ICE AMR mode, calls
 * a Hypre solver using HypreSolverWrap, and returns the pressure
 * results into Uintah. HypreSolverAMR schedules a task in sus called
 * scheduleSolve() that carries out these operations. It is based on
 * the one-level solver class HypreSolver2.
 *
 * Note: This interface is written for Hypre version 1.9.0b (released 2005)
 * or later.
 *--------------------------------------------------------------------------*/
#ifndef Packages_Uintah_CCA_Components_Solvers_HypreSolverAMR_h
#define Packages_Uintah_CCA_Components_Solvers_HypreSolverAMR_h

#include <Packages/Uintah/CCA/Ports/SolverInterface.h>
#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>

namespace Uintah {
  class HypreSolverAMR :
    public SolverInterface, public UintahParallelComponent { 
  public:
    HypreSolverAMR(const ProcessorGroup* myworld);
    virtual ~HypreSolverAMR();

    /* Load solver parameters from input struct */
    virtual SolverParameters* readParameters(ProblemSpecP& params,
                                             const std::string& name);

    /* Main task that solves the pressure equation and returns
       cell-centered pressure. In the future we can also implement here
       solutions of other variable types, like node-centered. */
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
  private:
  };
}

#endif // Packages_Uintah_CCA_Components_Solvers_HypreSolverAMR_h
