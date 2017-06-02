/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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
  

KEYWORDS
  AMRSolver, HypreDriver, HypreSolverParams, HypreSolverBase.

DESCRIPTION 
   Class AMRSolver is the main solver component that
   interfaces to Hypre's structured and semi-structured system
   interfaces.
  
WARNING
   * This interface is written for Hypre 1.9.0b (released 2005).
   --------------------------------------------------------------------------*/

#include <CCA/Ports/SolverInterface.h>
#include <Core/Parallel/UintahParallelComponent.h>

namespace Uintah {

  class AMRSolver :
    public SolverInterface, public UintahParallelComponent { 

   
  public:

    AMRSolver(const ProcessorGroup* myworld);
    virtual ~AMRSolver();

    virtual SolverParameters* readParameters(ProblemSpecP& params,
                                             const std::string& name,
                                             SimulationStateP& state);

    virtual SolverParameters* readParameters(ProblemSpecP& params,
                                             const std::string& name);

    virtual void scheduleSolve( const LevelP           & level,
                                      SchedulerP       & sched,
                                const MaterialSet      * matls,
                                const VarLabel         * A,    
                                      Task::WhichDW      which_A_dw,  
                                const VarLabel         * x,
                                      bool               modifies_x,
                                const VarLabel         * b,    
                                      Task::WhichDW      which_b_dw,  
                                const VarLabel         * guess,
                                      Task::WhichDW      which_guess_dw,
                                const SolverParameters * params,
                                      bool               modifies_hypre = false );
                               
    virtual std::string getName();
    
    // AMRSolver does not require initialization... but we need an empty
   // routine to satisfy inheritance.
    virtual void scheduleInitialize( const LevelP      & level,
                                           SchedulerP  & sched,
                                     const MaterialSet * matls ) {}

  private:

  };
}

#endif 
