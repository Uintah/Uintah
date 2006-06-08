
#ifndef Packages_Uintah_CCA_Ports_SolverInterace_h
#define Packages_Uintah_CCA_Ports_SolverInterace_h

#include <Packages/Uintah/Core/Parallel/UintahParallelPort.h>
#include <Packages/Uintah/Core/Grid/LevelP.h>
#include <Packages/Uintah/CCA/Ports/SchedulerP.h>
#include <Packages/Uintah/Core/Grid/Variables/ComputeSet.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

#include <Packages/Uintah/CCA/Ports/share.h>

namespace Uintah {
  class VarLabel;
  class SCISHARE SolverParameters {
  public:
    SolverParameters() : solveOnExtraCells(false), residualNormalization(1) {}
    void setSolveOnExtraCells(bool s) {
      solveOnExtraCells = s;
    }
    bool getSolveOnExtraCells() const {
      return solveOnExtraCells;
    }
    void setResidualNormalizationFactor(double s) {
      residualNormalization = s;
    }
    double getResidualNormalizationFactor() const {
      return residualNormalization;
    }
    virtual ~SolverParameters();
  private:
    bool solveOnExtraCells;
    double residualNormalization;
  };
  
  class SCISHARE SolverInterface : public UintahParallelPort {
  public:
    SolverInterface();
    virtual ~SolverInterface();

    virtual SolverParameters* readParameters(ProblemSpecP& params,
					     const std::string& name) = 0;
                            
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
			          const SolverParameters* params) = 0;
                               
  virtual string getName()=0;
  
  private: 
    SolverInterface(const SolverInterface&);
    SolverInterface& operator=(const SolverInterface&);
  };
}

#endif
