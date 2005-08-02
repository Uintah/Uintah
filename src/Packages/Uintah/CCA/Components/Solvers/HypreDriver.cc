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

#include <Packages/Uintah/CCA/Components/Solvers/HypreDriver.h>
#include <Packages/Uintah/CCA/Components/Solvers/HypreSolverParams.h>
#include <Packages/Uintah/CCA/Components/Solvers/HypreInterface.h>
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
    class HypreDriver implementation for CC variables
    _____________________________________________________________________*/

  template<class Types>
  void HypreDriver<Types>::solve(const ProcessorGroup* pg,
                                 const PatchSubset* patches,
                                 const MaterialSubset* matls,
                                 DataWarehouse* old_dw,
                                 DataWarehouse* new_dw,
                                 Handle<HypreDriver<Types> >)
    /*_____________________________________________________________________
      Function HypreDriver::solve
      Main solve function.
      _____________________________________________________________________*/
  {
    typedef typename Types::sol_type sol_type;
    cout_doing << "HypreSolverAMR<CCTypes>::solve" << endl;

    DataWarehouse* A_dw = new_dw->getOtherDataWarehouse(which_A_dw);
    DataWarehouse* b_dw = new_dw->getOtherDataWarehouse(which_b_dw);
    DataWarehouse* guess_dw = new_dw->getOtherDataWarehouse(which_guess_dw);
    
    for(int m = 0;m<matls->size();m++){
      int matl = matls->get(m);

      /* Decide which Hypre interface to use */
      const int numLevels = new_dw->getGrid()->numLevels();
      HypreInterface::InterfaceType interfaceType;
      if (numLevels == 1) {
        /* A uniform grid */
        interfaceType = HypreInterface::Struct;
      } else {
        /* Composite grid of uniform patches */
        interfaceType = HypreInterface::SStruct;
      }

      /* Construct Hypre linear system for the specific variable type
         and Hypre interface */
      HypreInterface hypreInterface(params);
      hypreInterface.makeLinearSystemStruct<Types>(interfaceType);
    
      /* Construct Hypre solver object that uses the hypreInterface we
         chose. Specific solver object is arbitrated in HypreGenericSolver
         according to param->solverType. */
      HypreGenericSolver::SolveType solverType =
        HypreGenericSolver::solverFromTitle(params->solverTitle);
      HypreGenericSolver* _hypreSolver =
        HypreGenericSolver::newSolver(solverType,_hypreInterface);

      /* Solve the linear system */
      double solve_start = Time::currentSeconds();
      _hypreSolver->solve();
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
      switch (_hypreInterface) {
      case HypreSolverParams::Struct:
        {
          getSolutionStruct(matl);
          break;
        }
      case HypreSolverParams::SStruct:
        {
          getSolutionSStruct(matl);
          break;
        }
      default:
        {
          throw InternalError("Unknown Hypre interface for getSolution: "
                              +hypreInterface,__FILE__, __LINE__);
        }
      } // end switch (_hypreInterface)

      HYPRE_StructMatrixDestroy(HA);
      HYPRE_StructVectorDestroy(HB);
      HYPRE_StructVectorDestroy(HX);
      HYPRE_StructStencilDestroy(stencil);
      HYPRE_StructGridDestroy(grid);

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


  // Depends on Types
  void HypreDriver::makeLinearSystemStruct(void)
  {
    ASSERTEQ(sizeof(Stencil7), 7*sizeof(double));
    double tstart = Time::currentSeconds();

    // Setup matrix
    HYPRE_StructGrid grid;
    HYPRE_StructGridCreate(pg->getComm(), 3, &grid);

    for(int p=0;p<patches->size();p++){
      const Patch* patch = patches->get(p);
      Patch::VariableBasis basis =
        Patch::translateTypeToBasis(sol_type::getTypeDescription()
                                    ->getType(), true);
      IntVector ec = params->getSolveOnExtraCells() ?
        IntVector(0,0,0) : -level->getExtraCells();
      IntVector l = patch->getLowIndex(basis, ec);
      IntVector h1 = patch->getHighIndex(basis, ec)-IntVector(1,1,1);

      HYPRE_StructGridSetExtents(grid, l.get_pointer(), h1.get_pointer());
    }
    HYPRE_StructGridAssemble(grid);

    // Create the stencil
    HYPRE_StructStencil stencil;
    if(params->symmetric){
      HYPRE_StructStencilCreate(3, 4, &stencil);
      int offsets[4][3] = {{0,0,0},
                           {-1,0,0},
                           {0,-1,0},
                           {0,0,-1}};
      for(int i=0;i<4;i++)
        HYPRE_StructStencilSetElement(stencil, i, offsets[i]);
    } else {
      HYPRE_StructStencilCreate(3, 7, &stencil);
      int offsets[7][3] = {{0,0,0},
                           {1,0,0}, {-1,0,0},
                           {0,1,0}, {0,-1,0},
                           {0,0,1}, {0,0,-1}};
      for(int i=0;i<7;i++)
        HYPRE_StructStencilSetElement(stencil, i, offsets[i]);
    }

    // Create the matrix
    HYPRE_StructMatrix HA;
    HYPRE_StructMatrixCreate(pg->getComm(), grid, stencil, &HA);
    HYPRE_StructMatrixSetSymmetric(HA, params->symmetric);
    int ghost[] = {1,1,1,1,1,1};
    HYPRE_StructMatrixSetNumGhost(HA, ghost);
    HYPRE_StructMatrixInitialize(HA);

    for(int p=0;p<patches->size();p++){
      const Patch* patch = patches->get(p);

      // Get the data
      CCTypes::matrix_type A;
      A_dw->get(A, A_label, matl, patch, Ghost::None, 0);

      Patch::VariableBasis basis =
        Patch::translateTypeToBasis(sol_type::getTypeDescription()
                                    ->getType(), true);
      IntVector ec = params->getSolveOnExtraCells() ?
        IntVector(0,0,0) : -level->getExtraCells();
      IntVector l = patch->getLowIndex(basis, ec);
      IntVector h = patch->getHighIndex(basis, ec);

      // Feed it to Hypre
      if(params->symmetric){
        double* values = new double[(h.x()-l.x())*4];	
        int stencil_indices[] = {0,1,2,3};
        for(int z=l.z();z<h.z();z++){
          for(int y=l.y();y<h.y();y++){
            const Stencil7* AA = &A[IntVector(l.x(), y, z)];
            double* p = values;
            for(int x=l.x();x<h.x();x++){
              *p++ = AA->p;
              *p++ = AA->w;
              *p++ = AA->s;
              *p++ = AA->b;
              AA++;
            }
            IntVector ll(l.x(), y, z);
            IntVector hh(h.x()-1, y, z);
            HYPRE_StructMatrixSetBoxValues(HA,
                                           ll.get_pointer(),
                                           hh.get_pointer(),
                                           4, stencil_indices, values);

          }
        }
        delete[] values;
      } else {
        int stencil_indices[] = {0,1,2,3,4,5,6};
        for(int z=l.z();z<h.z();z++){
          for(int y=l.y();y<h.y();y++){
            const double* values = &A[IntVector(l.x(), y, z)].p;
            IntVector ll(l.x(), y, z);
            IntVector hh(h.x()-1, y, z);
            HYPRE_StructMatrixSetBoxValues(HA,
                                           ll.get_pointer(),
                                           hh.get_pointer(),
                                           7, stencil_indices,
                                           const_cast<double*>(values));
          }
        }
      }
    }
    HYPRE_StructMatrixAssemble(HA);

    // Create the rhs
    HYPRE_StructVector HB;
    HYPRE_StructVectorCreate(pg->getComm(), grid, &HB);
    HYPRE_StructVectorInitialize(HB);

    for(int p=0;p<patches->size();p++){
      const Patch* patch = patches->get(p);

      // Get the data
      CCTypes::const_type B;
      b_dw->get(B, B_label, matl, patch, Ghost::None, 0);

      Patch::VariableBasis basis =
        Patch::translateTypeToBasis(sol_type::getTypeDescription()
                                    ->getType(), true);
      IntVector ec = params->getSolveOnExtraCells() ?
        IntVector(0,0,0) : -level->getExtraCells();
      IntVector l = patch->getLowIndex(basis, ec);
      IntVector h = patch->getHighIndex(basis, ec);

      // Feed it to Hypre
      for(int z=l.z();z<h.z();z++){
        for(int y=l.y();y<h.y();y++){
          const double* values = &B[IntVector(l.x(), y, z)];
          IntVector ll(l.x(), y, z);
          IntVector hh(h.x()-1, y, z);
          HYPRE_StructVectorSetBoxValues(HB,
                                         ll.get_pointer(),
                                         hh.get_pointer(),
                                         const_cast<double*>(values));
        }
      }
    }
    HYPRE_StructVectorAssemble(HB);

    // Create the solution vector
    HYPRE_StructVector HX;
    HYPRE_StructVectorCreate(pg->getComm(), grid, &HX);
    HYPRE_StructVectorInitialize(HX);

    for(int p=0;p<patches->size();p++){
      const Patch* patch = patches->get(p);

      if(guess_label){
        CCTypes::const_type X;
        guess_dw->get(X, guess_label, matl, patch, Ghost::None, 0);

        // Get the initial guess
        Patch::VariableBasis basis =
          Patch::translateTypeToBasis(sol_type::getTypeDescription()
                                      ->getType(), true);
        IntVector ec = params->getSolveOnExtraCells() ?
          IntVector(0,0,0) : -level->getExtraCells();
        IntVector l = patch->getLowIndex(basis, ec);
        IntVector h = patch->getHighIndex(basis, ec);

        // Feed it to Hypre
        for(int z=l.z();z<h.z();z++){
          for(int y=l.y();y<h.y();y++){
            const double* values = &X[IntVector(l.x(), y, z)];
            IntVector ll(l.x(), y, z);
            IntVector hh(h.x()-1, y, z);
            HYPRE_StructVectorSetBoxValues(HX,
                                           ll.get_pointer(),
                                           hh.get_pointer(),
                                           const_cast<double*>(values));
          }
        }
      }  // initialGuess
    } // patch loop
    HYPRE_StructVectorAssemble(HX);
  } // end HypreDriver::makeLinearSystemStruct()


  void HypreDriver::getSolutionStruct(void)
  {
    for(int p=0;p<patches->size();p++){
      const Patch* patch = patches->get(p);

      Patch::VariableBasis basis =
        Patch::translateTypeToBasis(sol_type::getTypeDescription()
                                    ->getType(), true);
      IntVector ec = params->getSolveOnExtraCells() ?
        IntVector(0,0,0) : -level->getExtraCells();
      IntVector l = patch->getLowIndex(basis, ec);
      IntVector h = patch->getHighIndex(basis, ec);
      CellIterator iter(l, h);

      sol_type Xnew;
      if(modifies_x)
        new_dw->getModifiable(Xnew, X_label, matl, patch);
      else
        new_dw->allocateAndPut(Xnew, X_label, matl, patch);
	
      // Get the solution back from hypre
      for(int z=l.z();z<h.z();z++){
        for(int y=l.y();y<h.y();y++){
          const double* values = &Xnew[IntVector(l.x(), y, z)];
          IntVector ll(l.x(), y, z);
          IntVector hh(h.x()-1, y, z);
          HYPRE_StructVectorGetBoxValues(HX,
                                         ll.get_pointer(),
                                         hh.get_pointer(),
                                         const_cast<double*>(values));
        }
      }
    }
  } // end HypreDriver::getSolutionStruct()


} // end namespace Uintah
