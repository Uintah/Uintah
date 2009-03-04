/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/



#ifndef Uintah_CCA_Ports_SolverInterace_h
#define Uintah_CCA_Ports_SolverInterace_h

#include <Uintah/Core/Parallel/UintahParallelPort.h>
#include <Uintah/Core/Grid/LevelP.h>
#include <Uintah/CCA/Ports/SchedulerP.h>
#include <Uintah/Core/Grid/Variables/ComputeSet.h>
#include <Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Uintah/Core/Grid/Task.h>
#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

#include <Uintah/CCA/Ports/uintahshare.h>

namespace Uintah {
  class VarLabel;
  class UINTAHSHARE SolverParameters {
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
  
  class UINTAHSHARE SolverInterface : public UintahParallelPort {
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
