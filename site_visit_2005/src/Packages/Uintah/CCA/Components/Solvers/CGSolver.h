
#ifndef Packages_Uintah_CCA_Components_Solvers_CGSolver_h
#define Packages_Uintah_CCA_Components_Solvers_CGSolver_h

#include <Packages/Uintah/CCA/Ports/SolverInterface.h>
#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>

namespace Uintah {
  class CGSolver : public SolverInterface, public UintahParallelComponent { 
  public:
    CGSolver(const ProcessorGroup* myworld);
    virtual ~CGSolver();

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
                               Task::WhichDW which_guess_dw,
                               const SolverParameters* params);
  private:
  };
}

#endif

