// TODO:
// Logging?
// Destroy stuff
// Use a symmetric matrix
// Read hypre options from input file
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

#include <HYPRE_struct_ls.h>

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
	} else {
	  // Zero initial guess
	  Patch::VariableBasis basis = Patch::translateTypeToBasis(sol_type::getTypeDescription()->getType(), true);
	  IntVector ec = params->getSolveOnExtraCells() ?
	    IntVector(0,0,0) : -level->getExtraCells();
	  IntVector l = patch->getLowIndex(basis, ec);
	  IntVector h = patch->getHighIndex(basis, ec);

	  // Feed it to Hypre
	  int size = h.x()-l.x();
	  double* values = new double[size];
	  for(int i=0;i<size;i++)
	    values[i]=0;
	  for(int z=l.z();z<h.z();z++){
	    for(int y=l.y();y<h.y();y++){
	      IntVector ll(l.x(), y, z);
	      IntVector hh(h.x()-1, y, z);
	      HYPRE_StructVectorSetBoxValues(HX,
					     ll.get_pointer(), hh.get_pointer(),
					     values);
	    }
	  }
	}
      }
      HYPRE_StructVectorAssemble(HX);

      double solve_start = Time::currentSeconds();
      HYPRE_StructSolver solver;
      HYPRE_StructSMGCreate(pg->getComm(), &solver);
      HYPRE_StructSMGSetMemoryUse(solver, 0);
      HYPRE_StructSMGSetMaxIter(solver, 75);
      HYPRE_StructSMGSetTol(solver, 1.e-14);
      HYPRE_StructSMGSetRelChange(solver, 0);
      HYPRE_StructSMGSetNumPreRelax(solver, 1);
      HYPRE_StructSMGSetNumPostRelax(solver, 1);
      HYPRE_StructSMGSetLogging(solver, 1);
      HYPRE_StructSMGSetup(solver, HA, HB, HX);
      HYPRE_StructSMGSolve(solver, HA, HB, HX);
      double solve_dt = Time::currentSeconds()-solve_start;

      int num_iterations;
      HYPRE_StructSMGGetNumIterations(solver, &num_iterations);
      double final_res_norm;
      HYPRE_StructSMGGetFinalRelativeResidualNorm(solver, &final_res_norm);
      HYPRE_StructSMGDestroy(solver);

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

SolverParameters* HypreSolver2::readParameters(const ProblemSpecP& params, const string& varname)
{
  HypreSolver2Params* p = new HypreSolver2Params();
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
