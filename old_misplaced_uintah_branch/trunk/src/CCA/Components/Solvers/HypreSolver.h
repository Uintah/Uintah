#ifndef Packages_Uintah_CCA_Components_Solvers_HypreSolver_h
#define Packages_Uintah_CCA_Components_Solvers_HypreSolver_h

#include <CCA/Ports/SolverInterface.h>
#include <Core/Parallel/UintahParallelComponent.h>

namespace Uintah {
  class HypreSolver2 : public SolverInterface, public UintahParallelComponent {
  public:
    HypreSolver2(const ProcessorGroup* myworld);
    virtual ~HypreSolver2();

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

#endif // Packages_Uintah_CCA_Components_Solvers_HypreSolver_h
