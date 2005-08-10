/*--------------------------------------------------------------------------
 * File: HypreDriver.cc
 *
 * Implementation of a wrapper of a Hypre solver for a particular variable
 * type. 
 *--------------------------------------------------------------------------*/
// TODO: (taken from HypreSolver.cc)
// Matrix file - why are ghosts there?
// Read hypre options from input file
// 3D performance
// Logging?
// Report mflops
// Use a symmetric matrix whenever possible
// More efficient set?
// Reuse some data between solves?
// Where is the initial guess taken from and where to read & print it here?
//   (right now in initialize() and solve()).

#include <Packages/Uintah/CCA/Components/Solvers/HypreDriver.h>
#include <Packages/Uintah/CCA/Components/Solvers/HypreDriverStruct.h>
#include <Packages/Uintah/CCA/Components/Solvers/HypreDriverSStruct.h>
#include <Packages/Uintah/CCA/Components/Solvers/HypreGenericSolver.h>
#include <Packages/Uintah/CCA/Components/Solvers/MatrixUtil.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/NCVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCZVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/Core/Grid/Variables/Stencil7.h>
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/Core/Exceptions/ConvergenceFailure.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/CCA/Ports/LoadBalancer.h>
#include <Core/Containers/Array1.h>
#include <Core/Containers/Array2.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Math/MiscMath.h>
#include <Core/Math/MinMax.h>
#include <Core/Thread/Time.h>
#include <Core/Util/DebugStream.h>
#include <iomanip>

// hypre includes
//#define HYPRE_TIMING
#ifndef HYPRE_TIMING
#ifndef hypre_ClearTiming
// This isn't in utilities.h for some reason...
#define hypre_ClearTiming()
#endif
#endif

using namespace Uintah;
//__________________________________
//  To turn on normal output
//  setenv SCI_DEBUG "HYPRE_DOING_COUT:+"

static DebugStream cout_doing("HYPRE_DOING_COUT", false);

namespace Uintah {

  /*_____________________________________________________________________
    class HypreDriver implementation common to all variable types
    and all Hypre interfaces
    _____________________________________________________________________*/

  template<class Types>
  HypreDriver<Types>::~HypreDriver<Types>(void)
  {
    cerr << "Destroying Solver object" << "\n";
    /* Destroy graph objects */
    /* Destroy matrix, RHS, solution objects */
    
    if (_activeInterface & Struct) {
      cerr << "Destroying Struct matrix, RHS, solution objects" << "\n";
      HYPRE_StructMatrixDestroy(_A_Struct);
      HYPRE_StructVectorDestroy(_b_Struct);
      HYPRE_StructVectorDestroy(_x_Struct);
    }
    if (_activeInterface & SStruct) {
      cerr << "Destroying SStruct matrix, RHS, solution objects" << "\n";
      HYPRE_SStructMatrixDestroy(_A_SStruct);
      HYPRE_SStructVectorDestroy(_b_SStruct);
      HYPRE_SStructVectorDestroy(_x_SStruct);
      cerr << "Destroying graph objects" << "\n";
      HYPRE_SStructGraphDestroy(_graph_SStruct);
    }
    if (_activeInterface & ParCSR) {
      cerr << "Destroying ParCSR matrix, RHS, solution objects" << "\n";
      HYPRE_ParCSRMatrixDestroy(_A_Par);
      HYPRE_ParVectorDestroy(_b_Par);
      HYPRE_ParVectorDestroy(_x_Par);
    }
  }

  template<class Types>
  void
  HypreDriver<Types>::solve(const ProcessorGroup* pg,
                            const PatchSubset* patches,
                            const MaterialSubset* matls,
                            DataWarehouse* old_dw,
                            DataWarehouse* new_dw,
                            Handle<HypreDriver<Types> >)
    /*_____________________________________________________________________
      Function HypreDriver::solve~
      Main solve function.
      _____________________________________________________________________*/
  {
    typedef typename Types::sol_type sol_type;
    cout_doing << "HypreSolverAMR<Types>::solve()" << endl;

    DataWarehouse* A_dw = new_dw->getOtherDataWarehouse(which_A_dw);
    DataWarehouse* b_dw = new_dw->getOtherDataWarehouse(which_b_dw);
    DataWarehouse* guess_dw = new_dw->getOtherDataWarehouse(which_guess_dw);
    
    // Check parameter correctness
    cerr << "Checking arguments and parameters ... ";
    HypreGenericSolver::SolverType solverType =
      getSolverType(p->solverTitle);
    const int numLevels = new_dw->getGrid()->numLevels();
    if ((solverType == HypreGenericSolver::FAC) && (numLevels < 2)) {
      cerr << "\n\nFAC solver needs a 3D problem and at least 2 levels."
           << "\n";
      clean();
      exit(1);
    }

    for(int m = 0;m<matls->size();m++){
      int matl = matls->get(m);

      /* Construct Hypre linear system for the specific variable type
         and Hypre interface */
      makeLinearSystemStruct();
    
      /* Construct Hypre solver object that uses the hypreInterface we
         chose. Specific solver object is arbitrated in HypreGenericSolver
         according to param->solverType. */
      HypreGenericSolver::SolverType solverType =
        HypreGenericSolver::solverFromTitle(params->solverTitle);
      HypreGenericSolver* _hypreSolver =
        HypreGenericSolver::newSolver(solverType,_hypreInterface);

      /* Solve the linear system */
      double solve_start = Time::currentSeconds();
      _hypresolver->setup();  // Depends only on A
      _hypresolver->solve();  // Depends on A and b
      double solve_dt = Time::currentSeconds()-solve_start;

      /* Check if converged, print solve statistics */
      const HypreGenericSolver::Results& results = _hypreSolver->getResults();
      const double& finalResNorm = results->finalResNorm;
      if ((finalResNorm > params->tolerance) ||
          (finite(finalResNorm) == 0)) {
        if (params->restart){
          if(pg->myrank() == 0)
            cerr << "HypreSolver not converged in " << results.numIterations 
                 << "iterations, final residual= " << finalResNorm
                 << ", requesting smaller timestep\n";
          //new_dw->abortTimestep();
          //new_dw->restartTimestep();
        } else {
          throw ConvergenceFailure("HypreSolver variable: "
                                   +X_label->getName()+
                                   ",solver: "+params->solverTitle+
                                   ", preconditioner: "+params->precondTitle,
                                   num_iterations, final_res_norm,
                                   params->tolerance,__FILE__,__LINE__);
        }
      } // if (finalResNorm is ok)

      /* Get the solution x values back into Uintah */
      getSolutionStruct(matl);

      /*-----------------------------------------------------------
       * Print the solution and other info
       *-----------------------------------------------------------*/
      linePrint("-",50);
      dbg0 << "Print the solution vector" << "\n";
      linePrint("-",50);
      solver->printSolution("output_x1");
      dbg0 << "Iterations = " << solver->_results.numIterations << "\n";
      dbg0 << "Final Relative Residual Norm = "
           << solver->_results.finalResNorm << "\n";
      dbg0 << "" << "\n";
      
      delete _hypreSolver;
      clear(); // Destroy Hypre objects

      double dt=Time::currentSeconds()-tstart;
      if(pg->myrank() == 0){
        cerr << "Solve of " << X_label->getName() 
             << " on level " << level->getIndex()
             << " completed in " << dt 
             << " seconds (solve only: " << solve_dt 
             << " seconds, " << num_iterations 
             << " iterations, residual=" << final_res_norm << ")\n";
      }
      tstart = Time::currentSeconds();
    } // for m (matls loop)
  } // end solve() for

#if 0
  template<class Types>
  void HypreDriver<Types>::setupPrecond(const ProcessorGroup* pg,
                                        HYPRE_PtrToSolverFcn& precond,
                                        HYPRE_PtrToSolverFcn& pcsetup,
                                        HYPRE_StructSolver& precond_solver)
    /*_____________________________________________________________________
      Function HypreDriver::setupPrecond
      Set up and initialize the Hypre preconditioner, if we use one.
      _____________________________________________________________________*/
  {
    switch (params->precondType) {
    case HypreSolverParams::PrecondNA:
      {
        /* No preconditioner, do nothing */
        break;
      } // case HypreSolverParams::PrecondNA

    case HypreSolverParams::PrecondSMG:
      /* use symmetric SMG as preconditioner */
      {
        HYPRE_StructSMGCreate(pg->getComm(), &precond_solver);
        HYPRE_StructSMGSetMemoryUse(precond_solver, 0);
        HYPRE_StructSMGSetMaxIter(precond_solver, 1);
        HYPRE_StructSMGSetTol(precond_solver, 0.0);
        HYPRE_StructSMGSetZeroGuess(precond_solver);
        HYPRE_StructSMGSetNumPreRelax(precond_solver, params->nPre);
        HYPRE_StructSMGSetNumPostRelax(precond_solver, params->nPost);
        HYPRE_StructSMGSetLogging(precond_solver, 0);
        precond = (HYPRE_PtrToSolverFcn)HYPRE_StructSMGSolve;
        pcsetup = (HYPRE_PtrToSolverFcn)HYPRE_StructSMGSetup;
        break;
      } // case HypreSolverParams::PrecondSMG

    case HypreSolverParams::PrecondPFMG:
      /* use symmetric PFMG as preconditioner */
      {
        HYPRE_StructPFMGCreate(pg->getComm(), &precond_solver);
        HYPRE_StructPFMGSetMaxIter(precond_solver, 1);
        HYPRE_StructPFMGSetTol(precond_solver, 0.0);
        HYPRE_StructPFMGSetZeroGuess(precond_solver);
        /* weighted Jacobi = 1; red-black GS = 2 */
        HYPRE_StructPFMGSetRelaxType(precond_solver, 1);
        HYPRE_StructPFMGSetNumPreRelax(precond_solver, params->nPre);
        HYPRE_StructPFMGSetNumPostRelax(precond_solver, params->nPost);
        HYPRE_StructPFMGSetSkipRelax(precond_solver, params->skip);
        HYPRE_StructPFMGSetLogging(precond_solver, 0);
        precond = (HYPRE_PtrToSolverFcn)HYPRE_StructPFMGSolve;
        pcsetup = (HYPRE_PtrToSolverFcn)HYPRE_StructPFMGSetup;
        break;
      } // case HypreSolverParams::PrecondPFMG

    case HypreSolverParams::PrecondSparseMSG:
      /* use symmetric SparseMSG as preconditioner */
      {
        HYPRE_StructSparseMSGCreate(pg->getComm(), &precond_solver);
        HYPRE_StructSparseMSGSetMaxIter(precond_solver, 1);
        HYPRE_StructSparseMSGSetJump(precond_solver, params->jump);
        HYPRE_StructSparseMSGSetTol(precond_solver, 0.0);
        HYPRE_StructSparseMSGSetZeroGuess(precond_solver);
        /* weighted Jacobi = 1; red-black GS = 2 */
        HYPRE_StructSparseMSGSetRelaxType(precond_solver, 1);
        HYPRE_StructSparseMSGSetNumPreRelax(precond_solver, params->nPre);
        HYPRE_StructSparseMSGSetNumPostRelax(precond_solver, params->nPost);
        HYPRE_StructSparseMSGSetLogging(precond_solver, 0);
        precond = (HYPRE_PtrToSolverFcn)HYPRE_StructSparseMSGSolve;
        pcsetup = (HYPRE_PtrToSolverFcn)HYPRE_StructSparseMSGSetup;
        break;
      } // case HypreSolverParams::PrecondSparseMSG

    case HypreSolverParams::PrecondJacobi:
      /* use two-step Jacobi as preconditioner */
      {
        HYPRE_StructJacobiCreate(pg->getComm(), &precond_solver);
        HYPRE_StructJacobiSetMaxIter(precond_solver, 2);
        HYPRE_StructJacobiSetTol(precond_solver, 0.0);
        HYPRE_StructJacobiSetZeroGuess(precond_solver);
        precond = (HYPRE_PtrToSolverFcn)HYPRE_StructJacobiSolve;
        pcsetup = (HYPRE_PtrToSolverFcn)HYPRE_StructJacobiSetup;
        break;
      } // case HypreSolverParams::PrecondJacobi

    case HypreSolverParams::PrecondDiagonal:
      /* use diagonal scaling as preconditioner */
      {
#ifdef HYPRE_USE_PTHREADS
        for (i = 0; i < hypre_NumThreads; i++)
          precond[i] = NULL;
#else
        precond = NULL;
#endif
        precond = (HYPRE_PtrToSolverFcn)HYPRE_StructDiagScale;
        pcsetup = (HYPRE_PtrToSolverFcn)HYPRE_StructDiagScaleSetup;
        break;
      } // case HypreSolverParams::PrecondDiagonal

    default:
      // This should have been caught in readParameters...
      throw InternalError("Unknown preconditionertype: "
                          +params->precondTitle,
                          __FILE__, __LINE__);
    } // end switch (param->precondType)
  } // end setupPrecond()

  template<class Types>
  void HypreDriver<Types>::destroyPrecond
  (HYPRE_StructSolver& precond_solver)
    /*_____________________________________________________________________
      Function HypreDriver::destroyPrecond
      Destroy (+free) the Hypre preconditioner.
      _____________________________________________________________________*/
  {
    switch (params->precondType) {
    case HypreSolverParams::PrecondNA:
      {
        /* No preconditioner, do nothing */
        break;
      } // case HypreSolverParams::PrecondNA
    case HypreSolverParams::PrecondSMG:
      {
        HYPRE_StructSMGDestroy(precond_solver);
        break;
      } // case HypreSolverParams::PrecondSMG

    case HypreSolverParams::PrecondPFMG:
      {
        HYPRE_StructPFMGDestroy(precond_solver);
        break;
      } // case HypreSolverParams::PrecondPFMG

    case HypreSolverParams::PrecondSparseMSG:
      {
        HYPRE_StructSparseMSGDestroy(precond_solver);
        break;
      } // case HypreSolverParams::PrecondSparseMSG
      
    case HypreSolverParams::PrecondJacobi:
      {
        HYPRE_StructJacobiDestroy(precond_solver);
        break;
      } // case HypreSolverParams::PrecondJacobi

    case HypreSolverParams::PrecondDiagonal:
      /* Nothing to destroy for diagonal preconditioner */
      {
        break;
      } // case HypreSolverParams::PrecondDiagonal

    default:
      // This should have been caught in readParameters...
      throw InternalError("Unknown preconditionertype in destroyPrecond: "
                          +params->precondType, __FILE__, __LINE__);
    } // end switch (param->precondType)
  } // end destroyPrecond()

#endif

  std::ostream&
  operator << (std::ostream& os, const CoarseFineViewpoint& v)
    // Write side s to the stream os.
  {
    if      (v == DoingCoarseToFine) os << "CoarseToFine";
    else if (v == DoingFineToCoarse) os << "FineToCoarse";
    else os << "N/A";
    return os;
  }


  std::ostream&
  operator << (std::ostream& os, const ConstructionStatus& s)
  {
    if      (s == DoingGraph ) os << "Graph ";
    else if (s == DoingMatrix) os << "Matrix";
    else os << "ST WRONG!!!";
    return os;
  }

  BoxSide& operator++(BoxSide &s)
  {
    return s = BoxSide(s+2);
  }
  
  std::ostream&
  operator << (std::ostream& os, const BoxSide& s)
    // Write side s to the stream os.
  {
    if      (s == LeftSide ) os << "Left ";
    else if (s == RightSide) os << "Right";
    else os << "N/A";
    return os;
  }

  template<class Types>
  HypreDriver<Types>*
  newHypreDriver(const HypreInterface& interface,
                 const Level* level,
                 const MaterialSet* matlset,
                 const VarLabel* A, Task::WhichDW which_A_dw,
                 const VarLabel* x, bool modifies_x,
                 const VarLabel* b, Task::WhichDW which_b_dw,
                 const VarLabel* guess,
                 Task::WhichDW which_guess_dw,
                 const HypreSolverParams* params)
  {
    switch (interface) {
    case HypreInterface::Struct: 
      {
        return new HypreDriverStruct<Types>
          (level, matls, A, which_A_dw,
           x, modifies_x, b, which_b_dw, guess, 
           which_guess_dw, dparams);
      }
    case HypreInterface::SStruct:
      {
        return new HypreDriverSStruct<Types>
          (level, matls, A, which_A_dw,
           x, modifies_x, b, which_b_dw, guess, 
           which_guess_dw, dparams);
      }
    default:
      throw InternalError("Unsupported Hypre Interface: "+interface,
                          __FILE__, __LINE__);
    } // end switch (interface)
  }

  double harmonicAvg(const Point& x,
                     const Point& y,
                     const Point& z,
                     const double& Ax,
                     const double& Ay)
    /*_____________________________________________________________________
      Function harmonicAvg~: 
      Harmonic average of the diffusion coefficient.
      A = harmonicAvg(X,Y,Z) returns the harmonic average of the
      diffusion coefficient a(T) (T in R^D) along the line connecting
      the points X,Y in R^D. That is, A = 1/(integral_0^1
      1/a(t1(s),...,tD(s)) ds), where td(s) = x{d} + s*(y{d} -
      x{d})/norm(y-x) is the arclength parameterization of the
      d-coordinate of the line x-y, d = 1...D.  We assume that A is
      piecewise constant with jump at Z (X,Y are normally cell centers
      and Z at the cell face). X,Y,Z are Dx1 location arrays.  In
      general, A can be analytically computed for the specific cases we
      consider; in general, use some simple quadrature formula for A
      from discrete a-values. This can be implemented by the derived
      test cases from Param.

      ### NOTE: ### If we use a different
      refinement ratio in different dimensions, near the interface we
      may need to compute A along lines X-Y that cross more than one
      cell boundary. This is currently ignored and we assume all lines
      cut one cell interface only.
      _____________________________________________________________________*/

  {
    const int numDims = 3;
    /* Compute distances x-y and x-z */
    double dxy = 0.0, dxz = 0.0;
    for (int d = 0; d < numDims; d++) {
      dxy += pow(fabs(y(d) - x(d)),2.0);
      dxz += pow(fabs(z(d) - x(d)),2.0);
    }
    double K = sqrt(dxz/dxy);
    return (Ax*Ay)/((1-K)*Ax + K*Ay);
  }

} // end namespace Uintah
