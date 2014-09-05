
#ifndef Packages_Uintah_CCA_Components_Solvers_DirectSolve_h
#define Packages_Uintah_CCA_Components_Solvers_DirectSolve_h

#include <Packages/Uintah/CCA/Ports/SolverInterface.h>
#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>

namespace Uintah {
  class DirectSolve : public SolverInterface, public UintahParallelComponent { 
  public:
    DirectSolve(const ProcessorGroup* myworld);
    virtual ~DirectSolve();

    virtual SolverParameters* readParameters(ProblemSpecP& params,
					     const std::string& name);

    virtual void scheduleSolve(const LevelP& level, SchedulerP& sched,
			       const MaterialSet* matls,
			       const VarLabel* A, const VarLabel* x,
			       bool modifies_x,
			       const VarLabel* b, const VarLabel* guess,
			       Task::WhichDW guess_dw,
			       const SolverParameters* params);
  private:
  };
}

#endif

