/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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

#ifndef Packages_Uintah_CCA_Components_Solvers_SolverCommon_h
#define Packages_Uintah_CCA_Components_Solvers_SolverCommon_h

#include <CCA/Ports/SolverInterface.h>
#include <Core/Parallel/UintahParallelComponent.h>

namespace Uintah {

  class ApplicationInterface;

  //______________________________________________________________________
  //
  class SolverCommon : public UintahParallelComponent, public SolverInterface { 

  public:

    SolverCommon(const ProcessorGroup* myworld);
    virtual ~SolverCommon();
    
    // Methods for managing the components attached via the ports.
    virtual void setComponents( UintahParallelComponent *comp ) {};
    virtual void getComponents();
    virtual void releaseComponents();
    
    virtual void readParameters( ProblemSpecP & params,
                                 const std::string  & name ) = 0;

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
                                      bool               isFirstSolve = true ) = 0;
                               
    virtual std::string getName() = 0;

    // SolverCommon does not require initialization... but we need an empty
    // routine to satisfy inheritance. 
    virtual void scheduleInitialize( const LevelP      & level,
                                               SchedulerP  & sched,
                                         const MaterialSet * matls ) = 0;

    virtual void scheduleRestartInitialize( const LevelP      & level,
                                                  SchedulerP  & sched,
                                            const MaterialSet * matls) = 0;

  protected:
    ApplicationInterface * m_application  {nullptr};
    
    const ProcessorGroup * m_myworld;

  };
  
} // end namespace Uintah

#endif // Packages_Uintah_CCA_Components_Solvers_SolverCommon_h
