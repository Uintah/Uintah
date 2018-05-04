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

#ifndef Packages_Uintah_CCA_Components_Solvers_HypreSolver_h
#define Packages_Uintah_CCA_Components_Solvers_HypreSolver_h

//#define HYPRE_TIMING

#include <CCA/Components/Solvers/SolverCommon.h>

#include <Core/Grid/SimulationState.h>
#include <Core/Util/Handle.h>
#include <Core/Util/RefCounted.h>

#include <HYPRE_struct_ls.h>
#include <HYPRE_krylov.h>

#include <iostream>

/**
 *  @class  HypreSolver2
 *  @author Steve Parker
 *  @author Todd Haraman
 *  @author John Schmidt
 *  @author Tony Saad
 *  @author James Sutherland
 *  @author Oren Livne
 *  @brief  Uintah hypre solver interface.
 *  Allows the solution of a linear system of the form \[ \mathbf{A} \mathbf{x} = \mathbf{b}\] where \[\mathbf{A}\] is
 *  stencil7 matrix.
 *
 */

namespace Uintah {

  //______________________________________________________________________
  //
  class HypreParams : public SolverParameters {
  public:
    HypreParams(){}
    
    ~HypreParams() {}
    
    // Parameters common for all Hypre Solvers
    std::string solvertype;         // String corresponding to solver type
    std::string precondtype;        // String corresponding to preconditioner type
    double      tolerance;          // Residual tolerance for solver
    double      precond_tolerance;  // Tolerance for preconditioner
    int         maxiterations;      // Maximum # iterations allowed
    int         precond_maxiters;   // Preconditioner max iterations
    int         logging;            // Log Hypre solver (using Hypre options)
    int         solveFrequency;     // Frequency for solving the linear system. timestep % solveFrequency
    int         relax_type;         // relaxation type
    
    // SMG parameters
    int    npre;               // # pre relaxations for Hypre SMG solver
    int    npost;              // # post relaxations for Hypre SMG solver
    
    // PFMG parameters
    int    skip;               // Hypre PFMG parameter
    
    // SparseMSG parameters
    int    jump;               // Hypre Sparse MSG parameter
  };

  //______________________________________________________________________
  //
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

  //______________________________________________________________________
  //
  struct hypre_solver_struct : public RefCounted {
    bool                 created_solver;
    bool                 created_precond_solver;
    SolverType           solver_type;
    SolverType           precond_solver_type;
    
    //  *_p = pointer
    HYPRE_StructSolver * solver_p;
    HYPRE_StructSolver * precond_solver_p;
    HYPRE_StructMatrix * HA_p;
    HYPRE_StructVector * HB_p;
    HYPRE_StructVector * HX_p;

    //__________________________________
    //
    hypre_solver_struct() {
      created_solver         = false;
      created_precond_solver = false;
      solver_type            = smg;
      precond_solver_type    = diagonal;
      solver_p              = 0;
      precond_solver_p      = 0;
      HA_p = 0;
      HB_p = 0;
      HX_p = 0;
    };
    //__________________________________
    //
    void print()
    {
      std::cout << "  Solver:  created: " << created_solver         << " type: " << solver_type         
                << " solver: " << &solver_p <<  " " << *solver_p << "\n";
                
      std::cout << "  Precond: created: " << created_precond_solver << " type: " << precond_solver_type 
                << " solver: " << &precond_solver_p << " " << *solver_p << "\n";
    };
    
    //__________________________________
    //
    virtual ~hypre_solver_struct() {
      if (created_solver) {
        HYPRE_StructMatrixDestroy( *HA_p );
        HYPRE_StructVectorDestroy( *HB_p );
        HYPRE_StructVectorDestroy( *HX_p );
      }
      if (created_solver)
        switch (solver_type) {
        case smg:
          HYPRE_StructSMGDestroy(*solver_p);
          break;
        case pfmg:
          HYPRE_StructPFMGDestroy(*solver_p);
          break;
        case sparsemsg:
          HYPRE_StructSparseMSGDestroy(*solver_p);
          break;
        case pcg:
          HYPRE_StructPCGDestroy(*solver_p);
          break;
        case gmres:
          HYPRE_StructGMRESDestroy(*solver_p);
          break;
        case jacobi:
          HYPRE_StructJacobiDestroy(*solver_p);
          break;
        default:
          throw InternalError( "HypreSolver given a bad solver type!",
                               __FILE__, __LINE__ );
        }

      if (created_precond_solver)
        switch (precond_solver_type) {
        case smg:
          HYPRE_StructSMGDestroy(*precond_solver_p);
          break;
        case pfmg:
          HYPRE_StructPFMGDestroy(*precond_solver_p);
          break;
        case sparsemsg:
          HYPRE_StructSparseMSGDestroy(*precond_solver_p);
          break;
        case pcg:
          HYPRE_StructPCGDestroy(*precond_solver_p);
          break;
        case gmres:
          HYPRE_StructGMRESDestroy(*precond_solver_p);
          break;
        case jacobi:
          HYPRE_StructJacobiDestroy(*precond_solver_p);
          break;
        default:
          throw InternalError("HypreSolver given a bad solver type!",
                              __FILE__, __LINE__);
      }

      if (HA_p) {
        delete HA_p;  
        HA_p = 0;
      }
      if (HB_p){
        delete HB_p;  
        HB_p = 0;
      }
      if (HX_p) {
        delete HX_p;  
        HX_p = 0;
      }
      if (solver_p) {
        delete solver_p;
        solver_p = 0;
      }
      if (precond_solver_p) {
        delete precond_solver_p;
        precond_solver_p = 0;
      }
    };
  };

  typedef Handle<hypre_solver_struct> hypre_solver_structP;

  void swapbytes( Uintah::hypre_solver_structP& );
  
  // Note the general template for SoleVariable::readNormal will not
  // recognize the swapbytes correctly. So specialize it here.
  // Somewhat moot because the swapbytes for hypre_solver_structP is
  // not implemented.
  template<>
  inline void SoleVariable<hypre_solver_structP>::readNormal(std::istream& in, bool swapBytes)
  {
    ssize_t linesize = (ssize_t)(sizeof(hypre_solver_structP));
    
    hypre_solver_structP val;
    
    in.read((char*) &val, linesize);
    
    if (swapBytes)
      Uintah::swapbytes(val);
    
    value = val;
  }
  
  //______________________________________________________________________
  //
  class HypreSolver2 : public SolverCommon {
  public:
    HypreSolver2(const ProcessorGroup* myworld);
    virtual ~HypreSolver2();

    virtual void readParameters(       ProblemSpecP & params,
                                 const std::string  & name  );
                                 
    virtual SolverParameters * getParameters(){ return m_params; }

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
    virtual void scheduleSolve( const LevelP           & level_in,
                                      SchedulerP       & sched_in,
                                const MaterialSet      * matls_in,
                                const VarLabel         * A_in,
                                      Task::WhichDW      which_A_dw_in,
                                const VarLabel         * x_in,
                                      bool               modifies_x_in,
                                const VarLabel         * b_in,
                                      Task::WhichDW      which_b_dw_in,
                                const VarLabel         * guess_in,
                                      Task::WhichDW      which_guess_dw_in,
                                      bool               isFirstSolve_in = true );

    virtual void scheduleInitialize( const LevelP      & level,
                                           SchedulerP  & sched,
                                     const MaterialSet * matls );
                                     
    virtual void scheduleRestartInitialize( const LevelP      & level,
                                                  SchedulerP  & sched,
                                            const MaterialSet * matls);

    virtual std::string getName();

    void allocateHypreMatrices( DataWarehouse * new_dw );

  private:
    void initialize( const ProcessorGroup *,
                     const PatchSubset    * patches,
                     const MaterialSubset * matls,
                           DataWarehouse  * old_dw,
                           DataWarehouse  * new_dw );
                           
    

    const VarLabel               * m_timeStepLabel;
    std::vector<const VarLabel*>   hypre_solver_label;
    int                            m_num_hypre_threads{1};
    
    
    HypreParams * m_params = nullptr;
    
  };
}

#endif // Packages_Uintah_CCA_Components_Solvers_HypreSolver_h
