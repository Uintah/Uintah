//--------------------------------------------------------------------------
// File: HypreDriverStruct.cc
//
// Implementation of the interface from Uintah to Hypre's Struct system
// interface, for cell-centered variables (e.g., pressure in implicit ICE).
//--------------------------------------------------------------------------
#include <Packages/Uintah/CCA/Components/Solvers/HypreSolverParams.h>
#include <Packages/Uintah/CCA/Components/Solvers/HypreDriverStruct.h>
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

  //#####################################################################
  // class HypreDriver implementation common to all variable types
  //#####################################################################
  
  HypreDriverStruct::HypreDriverStruct
  (const Level* level,
   const MaterialSet* matlset,
   const VarLabel* A, Task::WhichDW which_A_dw,
   const VarLabel* x, bool modifies_x,
   const VarLabel* b, Task::WhichDW which_b_dw,
   const VarLabel* guess,
   Task::WhichDW which_guess_dw,
   const HypreSolverParams* params) :
    //___________________________________________________________________
    // HypreDriverStruct constructor
    //___________________________________________________________________
    HypreDriver(level,matlset,A,which_A_dw,x,modifies_x,
                b,which_b_dw,guess,which_guess_dw,params)
  {
  }

  HypreDriverStruct::~HypreDriverStruct(void)
    //___________________________________________________________________
    // HypreDriverStruct destructor
    //___________________________________________________________________
  {
    HYPRE_StructMatrixDestroy(_HA);
    HYPRE_StructVectorDestroy(_HB);
    HYPRE_StructVectorDestroy(_HX);
    HYPRE_StructStencilDestroy(stencil);
    HYPRE_StructGridDestroy(grid);
  }

  void HypreDriverStruct::setupPrecond(void)
    /*_____________________________________________________________________
      Function HypreDriverStruct::setupPrecond
      Set up and initialize the Hypre preconditioner to be used by
      an SStruct solver. Preconditioner type is determined by params.
      _____________________________________________________________________*/
  {
    switch (params->precondType) {
    case HypreGenericSolver::PrecondNA:
      {
        /* No preconditioner, do nothing */
        break;
      } // case HypreGenericSolver::PrecondNA

    case HypreGenericSolver::PrecondSMG:
      /* use symmetric SMG as preconditioner */
      {
        HYPRE_StructSMGCreate(_pg->getComm(), precond_solver);
        HYPRE_StructSMGSetMemoryUse(*precond_solver, 0);
        HYPRE_StructSMGSetMaxIter(precond_solver, 1);
        HYPRE_StructSMGSetTol(precond_solver, 0.0);
        HYPRE_StructSMGSetZeroGuess(precond_solver);
        HYPRE_StructSMGSetNumPreRelax(precond_solver, params->nPre);
        HYPRE_StructSMGSetNumPostRelax(precond_solver, params->nPost);
        HYPRE_StructSMGSetLogging(precond_solver, 0);
        precond = (HYPRE_PtrToSolverFcn)HYPRE_StructSMGSolve;
        pcsetup = (HYPRE_PtrToSolverFcn)HYPRE_StructSMGSetup;
        break;
      } // case HypreGenericSolver::PrecondSMG

    case HypreGenericSolver::PrecondPFMG:
      /* use symmetric PFMG as preconditioner */
      {
        HYPRE_StructPFMGCreate(_pg->getComm(), &precond_solver);
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
      } // case HypreGenericSolver::PrecondPFMG

    case HypreGenericSolver::PrecondSparseMSG:
      /* use symmetric SparseMSG as preconditioner */
      {
        HYPRE_StructSparseMSGCreate(_pg->getComm(), &precond_solver);
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
      } // case HypreGenericSolver::PrecondSparseMSG

    case HypreGenericSolver::PrecondJacobi:
      /* use two-step Jacobi as preconditioner */
      {
        HYPRE_StructJacobiCreate(_pg->getComm(), &precond_solver);
        HYPRE_StructJacobiSetMaxIter(precond_solver, 2);
        HYPRE_StructJacobiSetTol(precond_solver, 0.0);
        HYPRE_StructJacobiSetZeroGuess(precond_solver);
        precond = (HYPRE_PtrToSolverFcn)HYPRE_StructJacobiSolve;
        pcsetup = (HYPRE_PtrToSolverFcn)HYPRE_StructJacobiSetup;
        break;
      } // case HypreGenericSolver::PrecondJacobi

    case HypreGenericSolver::PrecondDiagonal:
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
      } // case HypreGenericSolver::PrecondDiagonal

    default:
      // This should have been caught in readParameters...
      throw InternalError("Unknown preconditionertype: "
                          +params->precondTitle,
                          __FILE__, __LINE__);
    } // end switch (param->precondType)
  } // end setupPrecond()

  void HypreDriverStruct::destroyPrecond(void)
    /*_____________________________________________________________________
      Function HypreDriverStruct::destroyPrecond
      Destroy the Hypre preconditioner object used by an SStruct solver.
      _____________________________________________________________________*/
  {
    switch (params->precondType) {
    case HypreGenericSolver::PrecondNA:
      {
        /* No preconditioner, do nothing */
        break;
      } // case HypreGenericSolver::PrecondNA
    case HypreGenericSolver::PrecondSMG:
      {
        HYPRE_StructSMGDestroy(precond_solver);
        break;
      } // case HypreGenericSolver::PrecondSMG

    case HypreGenericSolver::PrecondPFMG:
      {
        HYPRE_StructPFMGDestroy(precond_solver);
        break;
      } // case HypreGenericSolver::PrecondPFMG

    case HypreGenericSolver::PrecondSparseMSG:
      {
        HYPRE_StructSparseMSGDestroy(precond_solver);
        break;
      } // case HypreGenericSolver::PrecondSparseMSG
      
    case HypreGenericSolver::PrecondJacobi:
      {
        HYPRE_StructJacobiDestroy(precond_solver);
        break;
      } // case HypreGenericSolver::PrecondJacobi

    case HypreGenericSolver::PrecondDiagonal:
      /* Nothing to destroy for diagonal preconditioner */
      {
        break;
      } // case HypreGenericSolver::PrecondDiagonal

    default:
      // This should have been caught in readParameters...
      throw InternalError("Unknown preconditionertype in destroyPrecond: "
                          +params->precondType, __FILE__, __LINE__);
    } // end switch (param->precondType)
  } // end destroyPrecond()

  //#####################################################################
  // class HypreDriver implementation for CC variable type
  //#####################################################################

  void
  HypreDriverStruct::makeLinearSystem_CC(const int matl)
    //___________________________________________________________________
    // Function HypreDriverStruct::makeLinearSystem_CC~
    // Construct the linear system for CC variables (e.g. pressure),
    // for the Hypre Struct interface (1-level problem / uniform grid).
    // We set up the matrix at all patches of the "level" data member.
    // matl is a fake material index. We always have one material here,
    // matl=0 (pressure).
    //_____________________________________________________________________
  {
    typedef CCTypes::sol_type sol_type;
    ASSERTEQ(sizeof(Stencil7), 7*sizeof(double));

    //==================================================================
    // Set up the grid
    //==================================================================
    HYPRE_StructGridCreate(_pg->getComm(), 3, &_grid);

    for(int p=0;p<_patches->size();p++){
      const Patch* patch = _patches->get(p);
      Patch::VariableBasis basis =
        Patch::translateTypeToBasis(sol_type::getTypeDescription()
                                    ->getType(), true);
      IntVector ec = params->getSolveOnExtraCells() ?
        IntVector(0,0,0) : -level->getExtraCells();
      IntVector l = patch->getLowIndex(basis, ec);
      IntVector h1 = patch->getHighIndex(basis, ec)-IntVector(1,1,1);

      HYPRE_StructGridSetExtents(_grid, l.get_pointer(), h1.get_pointer());
    }
    HYPRE_StructGridAssemble(_grid);

    //==================================================================
    // Set up the stencil
    //==================================================================
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

    //==================================================================
    // Set up the Struct left-hand-side matrix _HA
    //==================================================================
    HYPRE_StructMatrixCreate(_pg->getComm(), _grid, stencil, &_HA);
    HYPRE_StructMatrixSetSymmetric(_HA, params->symmetric);
    int ghost[] = {1,1,1,1,1,1};
    HYPRE_StructMatrixSetNumGhost(_HA, ghost);
    HYPRE_StructMatrixInitialize(_HA);

    for(int p=0;p<_patches->size();p++){
      const Patch* patch = _patches->get(p);

      // Get the data from Uintah
      CCTypes::matrix_type A;
      _A_dw->get(A, A_label, matl, patch, Ghost::None, 0);

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
            HYPRE_StructMatrixSetBoxValues(_HA,
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
            HYPRE_StructMatrixSetBoxValues(_HA,
                                           ll.get_pointer(),
                                           hh.get_pointer(),
                                           7, stencil_indices,
                                           const_cast<double*>(values));
          }
        }
      }
    }
    HYPRE_StructMatrixAssemble(_HA);

    //==================================================================
    // Set up the Struct right-hand-side vector _HB
    //==================================================================
    HYPRE_StructVectorCreate(_pg->getComm(), _grid, &_HB);
    HYPRE_StructVectorInitialize(_HB);

    for(int p=0;p<_patches->size();p++){
      const Patch* patch = _patches->get(p);

      // Get the data from Uintah
      CCTypes::const_type B;
      _b_dw->get(B, B_label, matl, patch, Ghost::None, 0);

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
          HYPRE_StructVectorSetBoxValues(_HB,
                                         ll.get_pointer(),
                                         hh.get_pointer(),
                                         const_cast<double*>(values));
        }
      }
    }
    HYPRE_StructVectorAssemble(_HB);

    //==================================================================
    // Set up the Struct solution vector _HX
    //==================================================================
    HYPRE_StructVectorCreate(_pg->getComm(), _grid, &_HX);
    HYPRE_StructVectorInitialize(_HX);

    for(int p=0;p<_patches->size();p++){
      const Patch* patch = _patches->get(p);

      if(guess_label){
        CCTypes::const_type X;
        _guess_dw->get(X, guess_label, matl, patch, Ghost::None, 0);

        // Get the initial guess from Uintah
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
            HYPRE_StructVectorSetBoxValues(_HX,
                                           ll.get_pointer(),
                                           hh.get_pointer(),
                                           const_cast<double*>(values));
          }
        }
      }  // initialGuess
    } // patch loop
    HYPRE_StructVectorAssemble(_HX);
  } // end HypreDriverStruct::makeLinearSystem()


  void
  HypreDriverStruct::getSolution_CC(const int matl)
    //_____________________________________________________________________
    // Function HypreDriverStruct::getSolution~
    // Get the solution vector for a 1-level, CC variable problem from
    // the Hypre Struct interface.
    //_____________________________________________________________________*/
  {
    //    typedef CCTypes::sol_type sol_type;
    for(int p=0;p<_patches->size();p++){
      const Patch* patch = _patches->get(p);

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
        _new_dw->getModifiable(Xnew, X_label, matl, patch);
      else
        _new_dw->allocateAndPut(Xnew, X_label, matl, patch);
	
      // Get the solution back from hypre
      for(int z=l.z();z<h.z();z++){
        for(int y=l.y();y<h.y();y++){
          const double* values = &Xnew[IntVector(l.x(), y, z)];
          IntVector ll(l.x(), y, z);
          IntVector hh(h.x()-1, y, z);
          HYPRE_StructVectorGetBoxValues(_HX,
                                         ll.get_pointer(),
                                         hh.get_pointer(),
                                         const_cast<double*>(values));
        }
      }
    }
  } // end HypreDriverStruct::getSolution()

  void
  HypreDriverStruct::printMatrix(const string& fileName /* =  "output" */)
  {
    cout_doing << "HypreDriverStruct::printMatrix() begin" << "\n";
    if (!_param->printSystem) return;
    HYPRE_SStructMatrixPrint((fileName + ".sstruct").c_str(), _A, 0);
    if (_requiresPar) {
      HYPRE_ParCSRMatrixPrint(_parA, (fileName + ".par").c_str());
      /* Print CSR matrix in IJ format, base 1 for rows and cols */
      HYPRE_ParCSRMatrixPrintIJ(_parA, 1, 1, (fileName + ".ij").c_str());
    }
    cout_doing << "HypreDriverStruct::printMatrix() end" << "\n";
  }

  void
  HypreDriverStruct::printRHS(const string& fileName /* =  "output_b" */)
  {
    if (!_param->printSystem) return;
    HYPRE_SStructVectorPrint(fileName.c_str(), _b, 0);
    if (_requiresPar) {
      HYPRE_ParVectorPrint(_parB, (fileName + ".par").c_str());
    }
  }

  void
  HypreDriverStruct::printSolution(const string& fileName /* =  "output_x" */)
  {
    if (!_param->printSystem) return;
    HYPRE_SStructVectorPrint(fileName.c_str(), _x, 0);
    if (_requiresPar) {
      HYPRE_ParVectorPrint(_parX, (fileName + ".par").c_str());
    }
  }

} // end namespace Uintah
