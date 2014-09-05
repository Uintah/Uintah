/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
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


#ifndef Packages_Uintah_CCA_Components_Solvers_HypreSolver_h
#define Packages_Uintah_CCA_Components_Solvers_HypreSolver_h

#include <CCA/Ports/SolverInterface.h>
#include <Core/Parallel/UintahParallelComponent.h>
/**
 *  @class  HypreSolver2
 *  @author Steve Parker
 *  @author Todd Haraman
 *  @author John Schmidt
 *  @author James Sutherland
 *  @author Oren Livne
 *  @brief  Uintah hypre solver interface.
 *  Allows the solution of a linear system of the form \[ \mathbf{A} \mathbf{x} = \mathbf{b}\] where \[\mathbf{A}\] is
 *  stencil7 matrix.
 *
 */

namespace Uintah {
  class HypreSolver2 : public SolverInterface, public UintahParallelComponent {
  public:
    HypreSolver2(const ProcessorGroup* myworld);
    virtual ~HypreSolver2();

    virtual SolverParameters* readParameters(ProblemSpecP& params,
                                             const std::string& name);
    
    /**
     *  @brief Schedules the solution of the linear system \[ \mathbf{A} \mathbf{x} = \mathbf{b}\].
     *
     *  @param level A reference to the level on which the system is solved.
     *  @param sched A reference to the Uintah scheduler.
     *  @param matls A pointer to the MaterialSet.
     *  @param A Varlabel of the coefficient matrix \[\mathbf{A}\]
     *  @param which_A_dw The datawarehouse in which the coefficient matrix lives.
     *  @param x The varlabel of the solution vector.
     *  @param modifies_x A boolean that specifies the behaviour of the task 
                          associated with the ScheduleSolve. If set to true,
                          then the task will only modify x. Otherwise, it will
                          compute x. This is a key option when you are computing
                          x in another place.
     * @param b The VarLabel of the right hand side vector.
     * @param which_b_dw Specifies the datawarehouse in which b lives.
     * @param guess VarLabel of the initial guess.
     * @param guess_dw Specifies the datawarehouse of the initial guess.
     * @param params Specifies the solver parameters usually parsed from the input file.
     *
     */    
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
