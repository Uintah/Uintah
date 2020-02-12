/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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

#ifndef Packages_Uintah_CCA_Components_Solvers_CGSolver_h
#define Packages_Uintah_CCA_Components_Solvers_CGSolver_h

#include <CCA/Components/Solvers/SolverCommon.h>

namespace Uintah {


  //______________________________________________________________________
  //
  class CGSolverParams : public SolverParameters {
  public:
    double tolerance;
    double initial_tolerance;
    int     maxiterations;

    enum Norm {
      L1, L2, LInfinity
    };
    
    Norm norm;
    
    enum Criteria {
      Absolute, Relative
    };
    
    Criteria criteria;
    
    CGSolverParams()
      : tolerance(1.e-8)
      , initial_tolerance(1.e-15)
      , norm(L2)
      , criteria(Relative)
    {}
    
    ~CGSolverParams() {}
  };


  //______________________________________________________________________
  //
  class CGSolver : public SolverCommon { 

  public:

    CGSolver( const ProcessorGroup * myworld );
    virtual ~CGSolver();

    virtual void readParameters(       ProblemSpecP     & params,
                                 const std::string      & name );

    virtual SolverParameters * getParameters(){ return m_params;}

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
                                      bool               isFirstSolve = true );

    virtual std::string getName();

    // CGSolver does not require initialization... but we need an empty
    // routine to satisfy inheritance.
    virtual void scheduleInitialize( const LevelP      & level,
                                           SchedulerP  & sched,
                                     const MaterialSet * matls ) {}
                                     
    virtual void scheduleRestartInitialize( const LevelP      & level,
                                                  SchedulerP  & sched,
                                            const MaterialSet * matls) {}
                                            
  private:
    CGSolverParams* m_params = nullptr;
  };

} // end namespace Uintah

#endif // Packages_Uintah_CCA_Components_Solvers_CGSolver_h
