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

#ifndef Packages_Uintah_CCA_Components_Solvers_HypreSolver_h
#define Packages_Uintah_CCA_Components_Solvers_HypreSolver_h

#define HYPRE_TIMING
#undef HYPRE_TIMING

#include <CCA/Ports/SolverInterface.h>
#include <Core/Parallel/UintahParallelComponent.h>
#include <Core/Util/RefCounted.h>
#include <Core/Util/Handle.h>
#include <HYPRE_struct_ls.h>
#include <HYPRE_krylov.h>
#include <iostream>
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

  

  enum SolverType {
    smg,
    pfmg,
    sparsemsg,
    pcg,
    hybrid,
    gmres,
    jacobi,
    diagonal
    };


  struct hypre_solver_struct : public RefCounted {
    bool created_solver;
    bool created_precond_solver;
    SolverType solver_type;
    SolverType precond_solver_type;
    HYPRE_StructSolver* solver;
    HYPRE_StructSolver* precond_solver;
    HYPRE_StructMatrix* HA;
    HYPRE_StructVector* HB;
    HYPRE_StructVector* HX;
    
    hypre_solver_struct() {
      created_solver=false;
      created_precond_solver=false;
      solver_type=smg;
      precond_solver_type=diagonal;
      solver=0;
      precond_solver=0;
      HA=0;
      HB=0;
      HX=0;
    };
    virtual ~hypre_solver_struct() {
      if (created_solver) {
        HYPRE_StructMatrixDestroy(*HA);
        HYPRE_StructVectorDestroy(*HB);
        HYPRE_StructVectorDestroy(*HX);
      }
      if (created_solver)
        switch (solver_type) {
        case smg:
          HYPRE_StructSMGDestroy(*solver);
          break;
        case pfmg:
          HYPRE_StructPFMGDestroy(*solver);
          break;
        case sparsemsg:
          HYPRE_StructSparseMSGDestroy(*solver);
          break;
        case pcg:
          HYPRE_StructPCGDestroy(*solver);
          break;
        case gmres:
          HYPRE_StructGMRESDestroy(*solver);
          break;
        case jacobi:
          HYPRE_StructJacobiDestroy(*solver);
          break;
        default:
          throw InternalError("HypreSolver given a bad solver type!", 
                              __FILE__, __LINE__);
        }

      if (created_precond_solver)
        switch (precond_solver_type) {
        case smg:
          HYPRE_StructSMGDestroy(*precond_solver);
          break;
        case pfmg:
          HYPRE_StructPFMGDestroy(*precond_solver);
          break;
        case sparsemsg:
          HYPRE_StructSparseMSGDestroy(*precond_solver);
          break;
        case pcg:
          HYPRE_StructPCGDestroy(*precond_solver);
          break;
        case gmres:
          HYPRE_StructGMRESDestroy(*precond_solver);
          break;
        case jacobi:
          HYPRE_StructJacobiDestroy(*precond_solver);
          break;
        default:
          throw InternalError("HypreSolver given a bad solver type!", 
                              __FILE__, __LINE__);
      }

      if (HA) {
        delete HA;  
        HA = 0;
      }
      if (HB){
        delete HB;  
        HB = 0;
      }
      if (HX) {
        delete HX;  
        HX = 0;
      }
      if (solver) {
      delete solver;
      solver = 0;
      }
      if (precond_solver) {
      delete precond_solver;
      precond_solver = 0;
      }
    };
  };

  typedef Handle<hypre_solver_struct> hypre_solver_structP;

  class HypreSolver2 : public SolverInterface, public UintahParallelComponent {
  public:
    HypreSolver2(const ProcessorGroup* myworld);
    virtual ~HypreSolver2();

    virtual SolverParameters* readParameters(ProblemSpecP& params,
                                             const std::string& name,
                                             SimulationStateP& state);

    /**
     *  @brief Schedules the solution of the linear system \[ \mathbf{A} \mathbf{x} = \mathbf{b}\].
     *
     *  @param level A reference to the level on which the system is solved.
     *  @param sched A reference to the Uintah scheduler.
     *  @param matls A pointer to the MaterialSet.
     *  @param A Varlabel of the coefficient matrix \[\mathbf{A}\]
     *  @param which_A_dw The datawarehouse in which the coefficient matrix lives.
     *  @param x The varlabel of the solutio1n vector.
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
                               const SolverParameters* params,
                               bool modifies_hypre = false);

    virtual void scheduleInitialize(const LevelP& level, SchedulerP& sched,
                                    const MaterialSet* matls);

    virtual string getName();

    void allocateHypreMatrices(DataWarehouse* new_dw);

  private:
    void initialize(const ProcessorGroup*,
                    const PatchSubset* patches, const MaterialSubset* matls,
                    DataWarehouse* old_dw, DataWarehouse* new_dw);

    const VarLabel* hypre_solver_label;
  };
}

#endif // Packages_Uintah_CCA_Components_Solvers_HypreSolver_h
