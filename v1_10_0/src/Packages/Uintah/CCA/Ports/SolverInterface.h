
#ifndef Packages_Uintah_CCA_Ports_SolverInterace_h
#define Packages_Uintah_CCA_Ports_SolverInterace_h

#include <Packages/Uintah/Core/Parallel/UintahParallelPort.h>
#include <Packages/Uintah/Core/Grid/LevelP.h>
#include <Packages/Uintah/CCA/Ports/SchedulerP.h>
#include <Packages/Uintah/Core/Grid/ComputeSet.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <string>

namespace Uintah {
  class VarLabel;
  class SolverParameters {
  public:
    virtual ~SolverParameters();
  };
  class SolverInterface : public UintahParallelPort {
  public:
    SolverInterface();
    virtual ~SolverInterface();

    virtual SolverParameters* readParameters(const ProblemSpecP& params,
					     const std::string& name) = 0;

    virtual void scheduleSolve(const LevelP& level, SchedulerP& sched,
			       const MaterialSet* matls,
			       const VarLabel* A, const VarLabel* x,
			       bool modifies_x,
			       const VarLabel* b, const VarLabel* guess,
			       Task::WhichDW guess_dw,
			       const SolverParameters* params) = 0;
  private:
    SolverInterface(const SolverInterface&);
    SolverInterface& operator=(const SolverInterface&);
  };
}

#endif
