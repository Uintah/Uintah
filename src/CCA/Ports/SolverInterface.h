/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */


#ifndef Packages_Uintah_CCA_Ports_SolverInterace_h
#define Packages_Uintah_CCA_Ports_SolverInterace_h

#include <Core/Parallel/UintahParallelPort.h>
#include <Core/Grid/LevelP.h>
#include <CCA/Ports/SchedulerP.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/SimulationState.h>
#include <string>


namespace Uintah {
  class VarLabel;
  class SolverParameters {
  public:
    SolverParameters() : useStencil4(false), solveOnExtraCells(false), residualNormalizationFactor(1), 
                        restartableTimestep(false), outputFileName("NULL") {}
    
    void setSolveOnExtraCells(bool s) {
      solveOnExtraCells = s;
    }
    
    bool getSolveOnExtraCells() const {
      return solveOnExtraCells;
    }
    
    void setUseStencil4(bool s) {
      useStencil4 = s;
    }
    
    bool getUseStencil4() const {
      return useStencil4;
    }
    
    void setResidualNormalizationFactor(double s) {
      residualNormalizationFactor = s;
    }
    
    double getResidualNormalizationFactor() const {
      return residualNormalizationFactor;
    }
    
    //If convergence fails call for the timestep to be restarted.
    void setRestartTimestepOnFailure(bool s){
      restartableTimestep=s;
    }
    
    bool getRestartTimestepOnFailure() const {
      return restartableTimestep;
    }
    
    // Used for outputting A, X & B to files
    void setOutputFileName(std::string s){
      outputFileName=s;
    }
    
    void getOutputFileName(vector<string>& fname) const {
      fname.push_back( "A" + outputFileName );
      fname.push_back( "b" + outputFileName );
      fname.push_back( "x" + outputFileName );
    }
    
    
    virtual ~SolverParameters();
  private:
    bool   useStencil4;
    bool   solveOnExtraCells;
    double residualNormalizationFactor;
    bool   restartableTimestep;
    std::string outputFileName;
  };
  
  class SolverInterface : public UintahParallelPort {
  public:
    SolverInterface();
    virtual ~SolverInterface();

    virtual SolverParameters* readParameters(ProblemSpecP& params,
					     const std::string& name,
                                             SimulationStateP& state) = 0;

    virtual void scheduleInitialize(const LevelP& level,SchedulerP& sched,
                                    const MaterialSet* matls);
                            
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
                               const SolverParameters* params,
                               bool modifies_hypre = false) = 0;
#if 0
    virtual void scheduleTransferData(const LevelP& level, SchedulerP& sched,
                                      const MaterialSet* matls);
#endif

    virtual string getName()=0;
  
  private: 
    SolverInterface(const SolverInterface&);
    SolverInterface& operator=(const SolverInterface&);
  };
}

#endif
