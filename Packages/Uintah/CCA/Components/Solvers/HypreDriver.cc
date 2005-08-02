/*--------------------------------------------------------------------------
 * File: HypreSolverWrap.cc
 *
 * Generic interface to Hypre's structured- and semi-structured matrix
 * interface and corresponding solvers. This is a template-"switch" on
 * different variable types (CC, NC, etc.).
 *--------------------------------------------------------------------------*/
// TODO (taken from HypreSolver.cc):
// Matrix file - why are ghosts there?
// Read hypre options from input file
// 3D performance
// Logging?
// Report mflops
// Use a symmetric matrix
// More efficient set?
// Reuse some data between solves?

#include <Packages/Uintah/CCA/Components/Solvers/HypreSolverWrap.h>
#include <Packages/Uintah/CCA/Components/Solvers/HypreSolverParams.h>
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
    class HypreSolverWrap implementation for CC variables
    _____________________________________________________________________*/

  template<>
  void HypreSolverWrap<CCTypes>::solve(const ProcessorGroup* pg,
                                       const PatchSubset* patches,
                                       const MaterialSubset* matls,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw,
                                       Handle<HypreSolverWrap<CCTypes> >)
    /*_____________________________________________________________________
      Function HypreSolverWrap::solve
      Main solve function.
      _____________________________________________________________________*/
  {
    typedef CCTypes::sol_type sol_type;
    cout_doing << "HypreSolverAMR<CCTypes>::solve" << endl;

    DataWarehouse* A_dw = new_dw->getOtherDataWarehouse(which_A_dw);
    DataWarehouse* b_dw = new_dw->getOtherDataWarehouse(which_b_dw);
    DataWarehouse* guess_dw = new_dw->getOtherDataWarehouse(which_guess_dw);
    
    /* Decide which Hypre data type to use */
    const int numLevels = new_dw->getGrid()->numLevels();
    if (numLevels == 1) {
      /* A uniform grid */
      _hypreInterface = HypreSolverParams::Struct;
    } else {
      /* Composite grid of uniform patches */
      _hypreInterface = HypreSolverParams::SStruct;
    }
    
    /* Construct linear system in the format chosen */
    _hypreData = new HypreData(_hypreInterface);

    /* Construct empty Hypre solver object */
    _hypreSolver = new Solver(_hypreData);
    
#if 0
    /* Construct Hypre linear system */
    switch (_hypreInterface) {
    case HypreSolverParams::Struct:
      {
        makeLinearSystemStruct();
        break;
      }
    case HypreSolverParams::SStruct:
      {
        makeLinearSystemSStruct();
        break;
      }
    default:
      {
        throw InternalError("Unknown Hypre interface type: "
                            +hypreInterface,__FILE__, __LINE__);
      }
    }
#endif

    // Solve the system
    double solve_start = Time::currentSeconds();
    int    num_iterations;
    double final_res_norm;

    switch (params->solverType) {
    case HypreSolverParams::SMG: {
      int time_index = hypre_InitializeTiming("SMG Setup");
      hypre_BeginTiming(time_index);

      HYPRE_StructSolver  solver;
      HYPRE_StructSMGCreate(MPI_COMM_WORLD, &solver);
      HYPRE_StructSMGSetMemoryUse(solver, 0);
      HYPRE_StructSMGSetMaxIter(solver, params->maxIterations);
      HYPRE_StructSMGSetTol(solver, params->tolerance);
      HYPRE_StructSMGSetRelChange(solver, 0);
      HYPRE_StructSMGSetNumPreRelax(solver, params->nPre);
      HYPRE_StructSMGSetNumPostRelax(solver, params->nPost);
      HYPRE_StructSMGSetLogging(solver, params->logging);
      HYPRE_StructSMGSetup(solver, HA, HB, HX);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", pg->getComm());
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("SMG Solve");
      hypre_BeginTiming(time_index);

      HYPRE_StructSMGSolve(solver, HA, HB, HX);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", pg->getComm());
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
   
      HYPRE_StructSMGGetNumIterations(solver, &num_iterations);
      HYPRE_StructSMGGetFinalRelativeResidualNorm(solver, &final_res_norm);
      HYPRE_StructSMGDestroy(solver);

      break;
    } // case HypreSolverParams::SMG

    case HypreSolverParams::PFMG: {
      int time_index = hypre_InitializeTiming("PFMG Setup");
      hypre_BeginTiming(time_index);

      HYPRE_StructSolver  solver;
      HYPRE_StructPFMGCreate(pg->getComm(), &solver);
      HYPRE_StructPFMGSetMaxIter(solver, params->maxIterations);
      HYPRE_StructPFMGSetTol(solver, params->tolerance);
      HYPRE_StructPFMGSetRelChange(solver, 0);
      /* weighted Jacobi = 1; red-black GS = 2 */
      HYPRE_StructPFMGSetRelaxType(solver, 1);
      HYPRE_StructPFMGSetNumPreRelax(solver, params->nPre);
      HYPRE_StructPFMGSetNumPostRelax(solver, params->nPost);
      HYPRE_StructPFMGSetSkipRelax(solver, params->skip);
      HYPRE_StructPFMGSetLogging(solver, params->logging);
      HYPRE_StructPFMGSetup(solver, HA, HB, HX);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", pg->getComm());
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("PFMG Solve");
      hypre_BeginTiming(time_index);
        
      HYPRE_StructPFMGSolve(solver, HA, HB, HX);
	
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", pg->getComm());
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
      
      HYPRE_StructPFMGGetNumIterations(solver, &num_iterations);
      HYPRE_StructPFMGGetFinalRelativeResidualNorm(solver, &final_res_norm);
      HYPRE_StructPFMGDestroy(solver);

      break;
    }  // case HypreSolverParams::PFMG

    case HypreSolverParams::SparseMSG: {
      int time_index = hypre_InitializeTiming("SparseMSG Setup");
      hypre_BeginTiming(time_index);

      HYPRE_StructSolver  solver;
      HYPRE_StructSparseMSGCreate(pg->getComm(), &solver);
      HYPRE_StructSparseMSGSetMaxIter(solver, params->maxIterations);
      HYPRE_StructSparseMSGSetJump(solver, params->jump);
      HYPRE_StructSparseMSGSetTol(solver, params->tolerance);
      HYPRE_StructSparseMSGSetRelChange(solver, 0);
      /* weighted Jacobi = 1; red-black GS = 2 */
      HYPRE_StructSparseMSGSetRelaxType(solver, 1);
      HYPRE_StructSparseMSGSetNumPreRelax(solver, params->nPre);
      HYPRE_StructSparseMSGSetNumPostRelax(solver, params->nPost);
      HYPRE_StructSparseMSGSetLogging(solver, params->logging);
      HYPRE_StructSparseMSGSetup(solver, HA, HB, HX);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", pg->getComm());
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("SparseMSG Solve");
      hypre_BeginTiming(time_index);

      HYPRE_StructSparseMSGSolve(solver, HA, HB, HX);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", pg->getComm());
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
   
      HYPRE_StructSparseMSGGetNumIterations(solver, &num_iterations);
      HYPRE_StructSparseMSGGetFinalRelativeResidualNorm(solver,
                                                        &final_res_norm);
      HYPRE_StructSparseMSGDestroy(solver);

      break;
    } // case HypreSolverParams::SparseMSG

    case HypreSolverParams::CG: {
      int time_index = hypre_InitializeTiming("PCG Setup");
      hypre_BeginTiming(time_index);

      HYPRE_StructSolver solver;
      HYPRE_StructPCGCreate(pg->getComm(), &solver);
      HYPRE_PCGSetMaxIter( (HYPRE_Solver)solver, params->maxIterations);
      HYPRE_PCGSetTol( (HYPRE_Solver)solver, params->tolerance);
      HYPRE_PCGSetTwoNorm( (HYPRE_Solver)solver, 1 );
      HYPRE_PCGSetRelChange( (HYPRE_Solver)solver, 0 );
      HYPRE_PCGSetLogging( (HYPRE_Solver)solver, params->logging);

      HYPRE_PtrToSolverFcn precond;
      HYPRE_PtrToSolverFcn precond_setup;
      HYPRE_StructSolver precond_solver;
      setupPrecond(pg, precond, precond_setup, precond_solver);
      HYPRE_PCGSetPrecond((HYPRE_Solver)solver, precond,
                          precond_setup, (HYPRE_Solver)precond_solver);
      HYPRE_PCGSetup( (HYPRE_Solver)solver, (HYPRE_Matrix)HA,
                      (HYPRE_Vector)HB, (HYPRE_Vector)HX);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", pg->getComm());
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
   
      time_index = hypre_InitializeTiming("PCG Solve");
      hypre_BeginTiming(time_index);

      HYPRE_PCGSolve( (HYPRE_Solver)solver, (HYPRE_Matrix)HA, (HYPRE_Vector)HB, (HYPRE_Vector)HX);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", pg->getComm());
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_PCGGetNumIterations( (HYPRE_Solver)solver, &num_iterations );
      HYPRE_PCGGetFinalRelativeResidualNorm( (HYPRE_Solver)solver, &final_res_norm );
      HYPRE_StructPCGDestroy(solver);

      destroyPrecond(precond_solver);

      break;
    } // case HypreSolverParams::CG

    case HypreSolverParams::Hybrid: {
      /*-----------------------------------------------------------
       * Solve the system using Hybrid
       *-----------------------------------------------------------*/
      int time_index = hypre_InitializeTiming("Hybrid Setup");
      hypre_BeginTiming(time_index);

      HYPRE_StructSolver  solver;
      HYPRE_StructHybridCreate(pg->getComm(), &solver);
      HYPRE_StructHybridSetDSCGMaxIter(solver, 100);
      HYPRE_StructHybridSetPCGMaxIter(solver, params->maxIterations);
      HYPRE_StructHybridSetTol(solver, params->tolerance);
      HYPRE_StructHybridSetConvergenceTol(solver, 0.90);
      HYPRE_StructHybridSetTwoNorm(solver, 1);
      HYPRE_StructHybridSetRelChange(solver, 0);
      HYPRE_StructHybridSetLogging(solver, params->logging);

      HYPRE_PtrToSolverFcn precond;
      HYPRE_PtrToSolverFcn precond_setup;
      HYPRE_StructSolver precond_solver;
      setupPrecond(pg, precond, precond_setup, precond_solver);
      HYPRE_StructHybridSetPrecond(solver,
                                   (HYPRE_PtrToStructSolverFcn)precond,
                                   (HYPRE_PtrToStructSolverFcn)precond_setup,
                                   (HYPRE_StructSolver)precond_solver);

      HYPRE_StructHybridSetup(solver, HA, HB, HX);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", pg->getComm());
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
   
      time_index = hypre_InitializeTiming("Hybrid Solve");
      hypre_BeginTiming(time_index);

      HYPRE_StructHybridSolve(solver, HA, HB, HX);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", pg->getComm());
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_StructHybridGetNumIterations(solver, &num_iterations);
      HYPRE_StructHybridGetFinalRelativeResidualNorm(solver, &final_res_norm);
      HYPRE_StructHybridDestroy(solver);

      destroyPrecond(precond_solver);

      break;
    } // case HypreSolverParams::Hybrid

    case HypreSolverParams::GMRES: {
      int time_index = hypre_InitializeTiming("GMRES Setup");
      hypre_BeginTiming(time_index);

      HYPRE_StructSolver solver;
      HYPRE_StructGMRESCreate(pg->getComm(), &solver);
      HYPRE_GMRESSetMaxIter( (HYPRE_Solver)solver, params->maxIterations);
      HYPRE_GMRESSetTol( (HYPRE_Solver)solver, params->tolerance );
      HYPRE_GMRESSetRelChange( (HYPRE_Solver)solver, 0 );
      HYPRE_GMRESSetLogging( (HYPRE_Solver)solver, params->logging);
      HYPRE_PtrToSolverFcn precond;
      HYPRE_PtrToSolverFcn precond_setup;
      HYPRE_StructSolver precond_solver;
      setupPrecond(pg, precond, precond_setup, precond_solver);
      HYPRE_GMRESSetPrecond((HYPRE_Solver)solver, precond, precond_setup,
                            (HYPRE_Solver)precond_solver);
      HYPRE_GMRESSetup( (HYPRE_Solver)solver, (HYPRE_Matrix)HA,
                        (HYPRE_Vector)HB, (HYPRE_Vector)HX);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", pg->getComm());
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
   
      time_index = hypre_InitializeTiming("GMRES Solve");
      hypre_BeginTiming(time_index);

      HYPRE_GMRESSolve( (HYPRE_Solver)solver, (HYPRE_Matrix)HA, (HYPRE_Vector)HB, (HYPRE_Vector)HX);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", pg->getComm());
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_GMRESGetNumIterations( (HYPRE_Solver)solver, &num_iterations);
      HYPRE_GMRESGetFinalRelativeResidualNorm( (HYPRE_Solver)solver, &final_res_norm);
      HYPRE_StructGMRESDestroy(solver);
	
      destroyPrecond(precond_solver);
      break;

    } // case HypreSolverParams::GMRES
          
    default:
      throw InternalError("Unknown solver type: "+params->solverType,
                          __FILE__, __LINE__);
    } // end switch (param->solverType)
        
    if(final_res_norm > params->tolerance || finite(final_res_norm) == 0){
      if(params->restart){
        if(pg->myrank() == 0)
          cerr << "HypreSolver not converged in " << num_iterations 
               << "iterations, final residual= " << final_res_norm 
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
    }
    double solve_dt = Time::currentSeconds()-solve_start;

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
} // end solve() for <CCTypes>

template<class Types>
void HypreSolverWrap<Types>::setupPrecond(const ProcessorGroup* pg,
                                          HYPRE_PtrToSolverFcn& precond,
                                          HYPRE_PtrToSolverFcn& pcsetup,
                                          HYPRE_StructSolver& precond_solver)
  /*_____________________________________________________________________
    Function HypreSolverWrap::setupPrecond
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
void HypreSolverWrap<Types>::destroyPrecond
(HYPRE_StructSolver& precond_solver)
  /*_____________________________________________________________________
    Function HypreSolverWrap::destroyPrecond
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

void HypreSolverWrap::makeLinearSystemStruct(void)
{
  ASSERTEQ(sizeof(Stencil7), 7*sizeof(double));
  double tstart = Time::currentSeconds();
  for(int m = 0;m<matls->size();m++){
    int matl = matls->get(m);

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
  } // end HypreSolverWrap::makeLinearSystemStruct()

} // end namespace Uintah
