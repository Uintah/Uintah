// TODO:
// Matrix file - why are ghosts there?
// Read hypre options from input file
// 3D performance
// Logging?
// Report mflops
// Use a symmetric matrix
// More efficient set?
// Reuse some data between solves?
#include <Packages/Uintah/CCA/Components/Solvers/HypreSolver.h>
#include <Packages/Uintah/CCA/Components/Solvers/MatrixUtil.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>
#include <Packages/Uintah/Core/Grid/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/SFCZVariable.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Packages/Uintah/Core/Grid/Stencil7.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/Core/Exceptions/ConvergenceFailure.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Core/Containers/Array1.h>
#include <Core/Containers/Array2.h>
#include <Core/Math/MiscMath.h>
#include <Core/Math/MinMax.h>
#include <Core/Thread/Time.h>
#include <Core/Util/DebugStream.h>
#include <iomanip>

// hypre includes
//#define HYPRE_TIMING
#include <utilities.h>
#include <HYPRE_struct_ls.h>
#include <krylov.h>
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

HypreSolver2::HypreSolver2(const ProcessorGroup* myworld)
  : UintahParallelComponent(myworld)
{
}

HypreSolver2::~HypreSolver2()
{
}

class HypreSolver2Params : public SolverParameters {
public:
  HypreSolver2Params()
  {
  }
  ~HypreSolver2Params() {}
  string solvertype;
  string precondtype;
  double tolerance;
  int maxiterations;
  int npre;
  int npost;
  int skip;
  int jump;
  int logging;
};

template<class Types>
class HypreStencil7 : public RefCounted {
public:
  HypreStencil7(const Level* level,
		const MaterialSet* matlset,
		const VarLabel* A,
		const VarLabel* x, bool modifies_x,
		const VarLabel* b,
		const VarLabel* guess, Task::WhichDW guess_dw,
		const HypreSolver2Params* params)
    : level(level), matlset(matlset),
      A_label(A), X_label(x), B_label(b),
      modifies_x(modifies_x),
      guess_label(guess), guess_dw(guess_dw), params(params)
  {
  }

  virtual ~HypreStencil7() {
  }

  //______________________________________________________________________
  void solve(const ProcessorGroup* pg, const PatchSubset* patches,
	     const MaterialSubset* matls,
	     DataWarehouse* old_dw, DataWarehouse* new_dw,
	     Handle<HypreStencil7<Types> >)
  {
    typedef typename Types::sol_type sol_type;
    cout_doing << "HypreSolver2::solve" << endl;

    ASSERTEQ(sizeof(Stencil7), 7*sizeof(double));
    double tstart = Time::currentSeconds();
    for(int m = 0;m<matls->size();m++){
      int matl = matls->get(m);

      // Setup matrix
      HYPRE_StructGrid grid;
      HYPRE_StructGridCreate(pg->getComm(), 3, &grid);

      for(int p=0;p<patches->size();p++){
	const Patch* patch = patches->get(p);
	Patch::VariableBasis basis = Patch::translateTypeToBasis(sol_type::getTypeDescription()->getType(), true);
	IntVector ec = params->getSolveOnExtraCells() ?
	  IntVector(0,0,0) : -level->getExtraCells();
	IntVector l = patch->getLowIndex(basis, ec);
	IntVector h1 = patch->getHighIndex(basis, ec)-IntVector(1,1,1);

	HYPRE_StructGridSetExtents(grid, l.get_pointer(), h1.get_pointer());
      }
      HYPRE_StructGridAssemble(grid);


      // Create the stencil
      HYPRE_StructStencil stencil;
      HYPRE_StructStencilCreate(3, 7, &stencil);
      int offsets[7][3] = {{0,0,0},
			   {1,0,0}, {-1,0,0},
			   {0,1,0}, {0,-1,0},
			   {0,0,1}, {0,0,-1}};
      for(int i=0;i<7;i++)
	HYPRE_StructStencilSetElement(stencil, i, offsets[i]);

      // Create the matrix
      HYPRE_StructMatrix HA;
      HYPRE_StructMatrixCreate(pg->getComm(), grid, stencil, &HA);
      HYPRE_StructMatrixSetSymmetric(HA, 0);
      int ghost[] = {1,1,1,1,1,1};
      HYPRE_StructMatrixSetNumGhost(HA, ghost);
      HYPRE_StructMatrixInitialize(HA);

      for(int p=0;p<patches->size();p++){
	const Patch* patch = patches->get(p);

	// Get the data
	typename Types::matrix_type A;
	new_dw->get(A, A_label, matl, patch, Ghost::None, 0);

	Patch::VariableBasis basis = Patch::translateTypeToBasis(sol_type::getTypeDescription()->getType(), true);
	IntVector ec = params->getSolveOnExtraCells() ?
	  IntVector(0,0,0) : -level->getExtraCells();
	IntVector l = patch->getLowIndex(basis, ec);
	IntVector h = patch->getHighIndex(basis, ec);

	// Feed it to Hypre
	for(int z=l.z();z<h.z();z++){
	  for(int y=l.y();y<h.y();y++){
	    int stencil_indices[] = {0,1,2,3,4,5,6};
	    const double* values = &A[IntVector(l.x(), y, z)].p;
	    IntVector ll(l.x(), y, z);
	    IntVector hh(h.x()-1, y, z);
	    HYPRE_StructMatrixSetBoxValues(HA,
					   ll.get_pointer(), hh.get_pointer(),
					   7, stencil_indices,
					   const_cast<double*>(values));
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
	typename Types::const_type B;
	new_dw->get(B, B_label, matl, patch, Ghost::None, 0);

	Patch::VariableBasis basis = Patch::translateTypeToBasis(sol_type::getTypeDescription()->getType(), true);
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
					   ll.get_pointer(), hh.get_pointer(),
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
	  typename Types::const_type X;
	  if(guess_dw == Task::OldDW)
	    old_dw->get(X, guess_label, matl, patch, Ghost::None, 0);
	  else
	    new_dw->get(X, guess_label, matl, patch, Ghost::None, 0);

	  // Get the initial guess
	  Patch::VariableBasis basis = Patch::translateTypeToBasis(sol_type::getTypeDescription()->getType(), true);
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
					     ll.get_pointer(), hh.get_pointer(),
					     const_cast<double*>(values));
	    }
	  }
	}
      }
      HYPRE_StructVectorAssemble(HX);

      double solve_start = Time::currentSeconds();
      int num_iterations;
      double final_res_norm;
      // Solve the system
      if (params->solvertype == "SMG" || params->solvertype == "smg"){
	int time_index = hypre_InitializeTiming("SMG Setup");
	hypre_BeginTiming(time_index);

	HYPRE_StructSolver  solver;
	HYPRE_StructSMGCreate(MPI_COMM_WORLD, &solver);
	HYPRE_StructSMGSetMemoryUse(solver, 0);
	HYPRE_StructSMGSetMaxIter(solver, params->maxiterations);
	HYPRE_StructSMGSetTol(solver, params->tolerance);
	HYPRE_StructSMGSetRelChange(solver, 0);
	HYPRE_StructSMGSetNumPreRelax(solver, params->npre);
	HYPRE_StructSMGSetNumPostRelax(solver, params->npost);
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
      } else if(params->solvertype == "PFMG" || params->solvertype == "pfmg"){
	int time_index = hypre_InitializeTiming("PFMG Setup");
	hypre_BeginTiming(time_index);

	HYPRE_StructSolver  solver;
	HYPRE_StructPFMGCreate(pg->getComm(), &solver);
	HYPRE_StructPFMGSetMaxIter(solver, params->maxiterations);
	HYPRE_StructPFMGSetTol(solver, params->tolerance);
	HYPRE_StructPFMGSetRelChange(solver, 0);
	/* weighted Jacobi = 1; red-black GS = 2 */
	HYPRE_StructPFMGSetRelaxType(solver, 1);
	HYPRE_StructPFMGSetNumPreRelax(solver, params->npre);
	HYPRE_StructPFMGSetNumPostRelax(solver, params->npost);
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
      } else if(params->solvertype == "SparseMSG" || params->solvertype == "sparsemsg"){
	int time_index = hypre_InitializeTiming("SparseMSG Setup");
	hypre_BeginTiming(time_index);

	HYPRE_StructSolver  solver;
	HYPRE_StructSparseMSGCreate(pg->getComm(), &solver);
	HYPRE_StructSparseMSGSetMaxIter(solver, params->maxiterations);
	HYPRE_StructSparseMSGSetJump(solver, params->jump);
	HYPRE_StructSparseMSGSetTol(solver, params->tolerance);
	HYPRE_StructSparseMSGSetRelChange(solver, 0);
	/* weighted Jacobi = 1; red-black GS = 2 */
	HYPRE_StructSparseMSGSetRelaxType(solver, 1);
	HYPRE_StructSparseMSGSetNumPreRelax(solver, params->npre);
	HYPRE_StructSparseMSGSetNumPostRelax(solver, params->npost);
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
      } else if(params->solvertype == "CG" || params->solvertype == "cg" || params->solvertype == "conjugategradient" || params->solvertype == "PCG" || params->solvertype == "cg"){
	int time_index = hypre_InitializeTiming("PCG Setup");
	hypre_BeginTiming(time_index);

	HYPRE_StructSolver solver;
	HYPRE_StructPCGCreate(pg->getComm(), &solver);
	HYPRE_PCGSetMaxIter( (HYPRE_Solver)solver, params->maxiterations);
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
      } else if(params->solvertype == "Hybrid" || params->solvertype == "hybrid"){
	/*-----------------------------------------------------------
	 * Solve the system using Hybrid
	 *-----------------------------------------------------------*/

	int time_index = hypre_InitializeTiming("Hybrid Setup");
	hypre_BeginTiming(time_index);

	HYPRE_StructSolver  solver;
	HYPRE_StructHybridCreate(pg->getComm(), &solver);
	HYPRE_StructHybridSetDSCGMaxIter(solver, 100);
	HYPRE_StructHybridSetPCGMaxIter(solver, params->maxiterations);
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
      } else if(params->solvertype == "GMRES" || params->solvertype == "gmres"){
	int time_index = hypre_InitializeTiming("GMRES Setup");
	hypre_BeginTiming(time_index);

	HYPRE_StructSolver solver;
	HYPRE_StructGMRESCreate(pg->getComm(), &solver);
	HYPRE_GMRESSetMaxIter( (HYPRE_Solver)solver, params->maxiterations);
	HYPRE_GMRESSetTol( (HYPRE_Solver)solver, params->tolerance );
	HYPRE_GMRESSetRelChange( (HYPRE_Solver)solver, 0 );
	HYPRE_GMRESSetLogging( (HYPRE_Solver)solver, params->logging);
	HYPRE_PtrToSolverFcn precond;
	HYPRE_PtrToSolverFcn precond_setup;
	HYPRE_StructSolver precond_solver;
	setupPrecond(pg, precond, precond_setup, precond_solver);
	HYPRE_GMRESSetPrecond((HYPRE_Solver)solver, precond, precond_setup,
			      (HYPRE_Solver)precond_solver);
	HYPRE_GMRESSetup( (HYPRE_Solver)solver, (HYPRE_Matrix)HA, (HYPRE_Vector)HB, (HYPRE_Vector)HX);

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
      } else {
	throw InternalError("Unknown solver type: "+params->solvertype);
      }
      if(final_res_norm > params->tolerance || finite(final_res_norm) == 0)
	throw ConvergenceFailure("HypreSolver variable: "+X_label->getName()+", solver: "+params->solvertype+", preconditioner: "+params->precondtype,
				 num_iterations, final_res_norm,
				 params->tolerance);
      double solve_dt = Time::currentSeconds()-solve_start;

      for(int p=0;p<patches->size();p++){
	const Patch* patch = patches->get(p);

	Patch::VariableBasis basis = Patch::translateTypeToBasis(sol_type::getTypeDescription()->getType(), true);
	IntVector ec = params->getSolveOnExtraCells() ?
	  IntVector(0,0,0) : -level->getExtraCells();
	IntVector l = patch->getLowIndex(basis, ec);
	IntVector h = patch->getHighIndex(basis, ec);
	CellIterator iter(l, h);

	typename Types::sol_type Xnew;
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
					   ll.get_pointer(), hh.get_pointer(),
					   const_cast<double*>(values));
	  }
	}
      }

#if 0
      {
	static int count=0;
	count++;
	ostringstream name;
	name << "A.dat." << new_dw->getID() << "." << count;
	HYPRE_StructMatrixPrint(name.str().c_str(), HA, 1);
      }
      {
	static int count=0;
	count++;
	ostringstream name;
	name << "B.dat." << new_dw->getID() << "." << count;
	HYPRE_StructVectorPrint(name.str().c_str(), HB, 1);
      }
      {
	static int count=0;
	count++;
	ostringstream name;
	name << "X.dat." << new_dw->getID() << "." << count;
	HYPRE_StructVectorPrint(name.str().c_str(), HX, 1);
      }
#endif

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
    }
  }

  void setupPrecond(const ProcessorGroup* pg,
		    HYPRE_PtrToSolverFcn& precond,
		    HYPRE_PtrToSolverFcn& pcsetup,
		    HYPRE_StructSolver& precond_solver){
    if(params->precondtype == "SMG" || params->precondtype == "smg"){
      /* use symmetric SMG as preconditioner */
      HYPRE_StructSMGCreate(pg->getComm(), &precond_solver);
      HYPRE_StructSMGSetMemoryUse(precond_solver, 0);
      HYPRE_StructSMGSetMaxIter(precond_solver, 1);
      HYPRE_StructSMGSetTol(precond_solver, 0.0);
      HYPRE_StructSMGSetZeroGuess(precond_solver);
      HYPRE_StructSMGSetNumPreRelax(precond_solver, params->npre);
      HYPRE_StructSMGSetNumPostRelax(precond_solver, params->npost);
      HYPRE_StructSMGSetLogging(precond_solver, 0);
      precond = (HYPRE_PtrToSolverFcn)HYPRE_StructSMGSolve;
      pcsetup = (HYPRE_PtrToSolverFcn)HYPRE_StructSMGSetup;
    } else if(params->precondtype == "PFMG" || params->precondtype == "pfmg"){
      /* use symmetric PFMG as preconditioner */
      HYPRE_StructPFMGCreate(pg->getComm(), &precond_solver);
      HYPRE_StructPFMGSetMaxIter(precond_solver, 1);
      HYPRE_StructPFMGSetTol(precond_solver, 0.0);
      HYPRE_StructPFMGSetZeroGuess(precond_solver);
      /* weighted Jacobi = 1; red-black GS = 2 */
      HYPRE_StructPFMGSetRelaxType(precond_solver, 1);
      HYPRE_StructPFMGSetNumPreRelax(precond_solver, params->npre);
      HYPRE_StructPFMGSetNumPostRelax(precond_solver, params->npost);
      HYPRE_StructPFMGSetSkipRelax(precond_solver, params->skip);
      HYPRE_StructPFMGSetLogging(precond_solver, 0);
      precond = (HYPRE_PtrToSolverFcn)HYPRE_StructPFMGSolve;
      pcsetup = (HYPRE_PtrToSolverFcn)HYPRE_StructPFMGSetup;
    } else if(params->precondtype == "SparseMSG" || params->precondtype == "sparsemsg"){
      /* use symmetric SparseMSG as preconditioner */
      HYPRE_StructSparseMSGCreate(pg->getComm(), &precond_solver);
      HYPRE_StructSparseMSGSetMaxIter(precond_solver, 1);
      HYPRE_StructSparseMSGSetJump(precond_solver, params->jump);
      HYPRE_StructSparseMSGSetTol(precond_solver, 0.0);
      HYPRE_StructSparseMSGSetZeroGuess(precond_solver);
      /* weighted Jacobi = 1; red-black GS = 2 */
      HYPRE_StructSparseMSGSetRelaxType(precond_solver, 1);
      HYPRE_StructSparseMSGSetNumPreRelax(precond_solver, params->npre);
      HYPRE_StructSparseMSGSetNumPostRelax(precond_solver, params->npost);
      HYPRE_StructSparseMSGSetLogging(precond_solver, 0);
      precond = (HYPRE_PtrToSolverFcn)HYPRE_StructSparseMSGSolve;
      pcsetup = (HYPRE_PtrToSolverFcn)HYPRE_StructSparseMSGSetup;
    } else if(params->precondtype == "Jacobi" || params->precondtype == "jacobi"){
      /* use two-step Jacobi as preconditioner */
      HYPRE_StructJacobiCreate(pg->getComm(), &precond_solver);
      HYPRE_StructJacobiSetMaxIter(precond_solver, 2);
      HYPRE_StructJacobiSetTol(precond_solver, 0.0);
      HYPRE_StructJacobiSetZeroGuess(precond_solver);
      precond = (HYPRE_PtrToSolverFcn)HYPRE_StructJacobiSolve;
      pcsetup = (HYPRE_PtrToSolverFcn)HYPRE_StructJacobiSetup;
    } else if(params->precondtype == "Diagonal" || params->precondtype == "diagonal"){
      /* use diagonal scaling as preconditioner */
#ifdef HYPRE_USE_PTHREADS
      for (i = 0; i < hypre_NumThreads; i++)
	precond[i] = NULL;
#else
      precond = NULL;
#endif
      precond = (HYPRE_PtrToSolverFcn)HYPRE_StructDiagScale;
      pcsetup = (HYPRE_PtrToSolverFcn)HYPRE_StructDiagScaleSetup;
    } else {
      // This should have been caught in readParameters...
      throw InternalError("Unknown preconditionertype: "+params->precondtype);
    }
  }
  void destroyPrecond(HYPRE_StructSolver precond_solver){
    if(params->precondtype == "SMG" || params->precondtype == "smg"){
      HYPRE_StructSMGDestroy(precond_solver);
    } else if(params->precondtype == "PFMG" || params->precondtype == "pfmg"){
      HYPRE_StructPFMGDestroy(precond_solver);
    } else if(params->precondtype == "SparseMSG" || params->precondtype == "sparsemsg"){
      HYPRE_StructSparseMSGDestroy(precond_solver);
    } else if(params->precondtype == "Jacobi" || params->precondtype == "jacobi"){
      HYPRE_StructJacobiDestroy(precond_solver);
    } else if(params->precondtype == "Diagonal" || params->precondtype == "diagonal"){
    } else {
      // This should have been caught in readParameters...
      throw InternalError("Unknown preconditionertype in destroyPrecond: "+params->precondtype);
    }
  }

private:
  const Level* level;
  const MaterialSet* matlset;
  const VarLabel* A_label;
  const VarLabel* X_label;
  const VarLabel* B_label;
  bool modifies_x;
  const VarLabel* guess_label;
  Task::WhichDW guess_dw;
  const HypreSolver2Params* params;
};

SolverParameters* HypreSolver2::readParameters(ProblemSpecP& params, const string& varname)
{
  HypreSolver2Params* p = new HypreSolver2Params();
  bool found=false;
  if(params){
    for(ProblemSpecP param = params->findBlock("Parameters"); param != 0;
	param = param->findNextBlock("Parameters")) {
      string variable;
      if(param->getAttribute("variable", variable) && variable != varname)
	continue;
      param->getWithDefault("solver", p->solvertype, "smg");
      param->getWithDefault("preconditioner", p->precondtype, "diagonal");
      param->getWithDefault("tolerance", p->tolerance, 1.e-10);
      param->getWithDefault("maxiterations", p->maxiterations, 75);
      param->getWithDefault("npre", p->npre, 1);
      param->getWithDefault("npost", p->npost, 1);
      param->getWithDefault("skip", p->skip, 0);
      param->getWithDefault("jump", p->jump, 0);
      param->getWithDefault("logging", p->logging, 0);
      found=true;
    }
  }
  if(!found){
    p->solvertype = "smg";
    p->precondtype = "diagonal";
    p->tolerance = 1.e-10;
    p->maxiterations = 75;
    p->npre = 1;
    p->npost = 1;
    p->skip = 0;
    p->jump = 0;
    p->logging = 0;
  }
  return p;
}

void HypreSolver2::scheduleSolve(const LevelP& level, SchedulerP& sched,
			     const MaterialSet* matls,
			     const VarLabel* A, const VarLabel* x,
			     bool modifies_x,
                             const VarLabel* b, const VarLabel* guess,
			     Task::WhichDW guess_dw,
			     const SolverParameters* params)
{
  Task* task;
  // The extra handle arg ensures that the stencil7 object will get freed
  // when the task gets freed.  The downside is that the refcount gets
  // tweaked everytime solve is called.

  TypeDescription::Type domtype = A->typeDescription()->getType();
  ASSERTEQ(domtype, x->typeDescription()->getType());
  ASSERTEQ(domtype, b->typeDescription()->getType());
  const HypreSolver2Params* dparams = dynamic_cast<const HypreSolver2Params*>(params);
  if(!dparams)
    throw InternalError("Wrong type of params passed to cg solver!");

  switch(domtype){
  case TypeDescription::SFCXVariable:
    {
      HypreStencil7<SFCXTypes>* that = new HypreStencil7<SFCXTypes>(level.get_rep(), matls, A, x, modifies_x, b, guess, guess_dw, dparams);
      Handle<HypreStencil7<SFCXTypes> > handle = that;
      task = scinew Task("Matrix solve", that, &HypreStencil7<SFCXTypes>::solve, handle);
    }
    break;
  case TypeDescription::SFCYVariable:
    {
      HypreStencil7<SFCYTypes>* that = new HypreStencil7<SFCYTypes>(level.get_rep(), matls, A, x, modifies_x, b, guess, guess_dw, dparams);
      Handle<HypreStencil7<SFCYTypes> > handle = that;
      task = scinew Task("Matrix solve", that, &HypreStencil7<SFCYTypes>::solve, handle);
    }
    break;
  case TypeDescription::SFCZVariable:
    {
      HypreStencil7<SFCZTypes>* that = new HypreStencil7<SFCZTypes>(level.get_rep(), matls, A, x, modifies_x, b, guess, guess_dw, dparams);
      Handle<HypreStencil7<SFCZTypes> > handle = that;
      task = scinew Task("Matrix solve", that, &HypreStencil7<SFCZTypes>::solve, handle);
    }
    break;
  case TypeDescription::CCVariable:
    {
      HypreStencil7<CCTypes>* that = new HypreStencil7<CCTypes>(level.get_rep(), matls, A, x, modifies_x, b, guess, guess_dw, dparams);
      Handle<HypreStencil7<CCTypes> > handle = that;
      task = scinew Task("Matrix solve", that, &HypreStencil7<CCTypes>::solve, handle);
    }
    break;
  case TypeDescription::NCVariable:
    {
      HypreStencil7<NCTypes>* that = new HypreStencil7<NCTypes>(level.get_rep(), matls, A, x, modifies_x, b, guess, guess_dw, dparams);
      Handle<HypreStencil7<NCTypes> > handle = that;
      task = scinew Task("Matrix solve", that, &HypreStencil7<NCTypes>::solve, handle);
    }
    break;
  default:
    throw InternalError("Unknown variable type in scheduleSolve");
  }

  task->requires(Task::NewDW, A, Ghost::None, 0);
  if(modifies_x)
    task->modifies(x);
  else
    task->computes(x);

  task->requires(Task::NewDW, b, Ghost::None, 0);
  sched->addTask(task, level->eachPatch(), matls);
}

} // end namespace Uintah
