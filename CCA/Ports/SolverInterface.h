
#ifndef Packages_Uintah_CCA_Ports_SolverInterace_h
#define Packages_Uintah_CCA_Ports_SolverInterace_h

#include <Packages/Uintah/Core/Parallel/UintahParallelPort.h>
#include <Packages/Uintah/Core/Grid/LevelP.h>
#include <Packages/Uintah/CCA/Ports/SchedulerP.h>
#include <Packages/Uintah/Core/Grid/ComputeSet.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

namespace Uintah {
  class VarLabel;
  class SolverParameters {
  public:
    SolverParameters() : solveOnExtraCells(false) {}
    void setSolveOnExtraCells(bool s) {
      solveOnExtraCells = s;
    }
    bool getSolveOnExtraCells() const {
      return solveOnExtraCells;
    }
    virtual ~SolverParameters();
  private:
    bool solveOnExtraCells;
  };
  class SolverInterface : public UintahParallelPort {
  public:
    SolverInterface();
    virtual ~SolverInterface();

    virtual SolverParameters* readParameters(ProblemSpecP& params,
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
