// TODO:
// This could be made somewhat faster by using a banded matrix
#include <Packages/Uintah/CCA/Components/Solvers/DirectSolve.h>
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

using namespace Uintah;
//__________________________________
//  To turn on normal output
//  setenv SCI_DEBUG "CGSOLVER_DOING_COUT:+"

static DebugStream cout_doing("CGSOLVER_DOING_COUT", false);

namespace Uintah {

DirectSolve::DirectSolve(const ProcessorGroup* myworld)
  : UintahParallelComponent(myworld)
{
}

DirectSolve::~DirectSolve()
{
}

class DirectSolveParams : public SolverParameters {
public:
  DirectSolveParams()
  {
  }
  ~DirectSolveParams() {}
};

template<class Types>
class DirectStencil7 : public RefCounted {
public:
  DirectStencil7(const Level* level,
		 const MaterialSet* matlset,
		 const VarLabel* A,
		 const VarLabel* x, bool modifies_x,
		 const VarLabel* b,
		 const DirectSolveParams* params)
    : level(level), matlset(matlset),
      A_label(A), X_label(x), B_label(b),
      modifies_x(modifies_x), params(params)
  {
  }

  virtual ~DirectStencil7() {
  }

  //______________________________________________________________________
  void solve(const ProcessorGroup* pg, const PatchSubset* patches,
	     const MaterialSubset* matls,
	     DataWarehouse* old_dw, DataWarehouse* new_dw,
	     Handle<DirectStencil7<Types> >)
  {
    cout_doing << "DirectSolve::solve" << endl;
    double tstart = Time::currentSeconds();
    long64 flops = 0, memrefs = 0;
    for(int m = 0;m<matls->size();m++){
      int matl = matls->get(m);
      typedef typename Types::sol_type sol_type;
      cout_doing << "DirectSolve on matl " << matl << endl;
      ASSERTEQ(patches->size(), 1);
      const Patch* patch = patches->get(0);
      typename Types::const_type B;
      new_dw->get(B, B_label, matl, patch, Ghost::None, 0);
      typename Types::matrix_type A;
      new_dw->get(A, A_label, matl, patch, Ghost::None, 0);

      typename Types::sol_type X;
      if(modifies_x)
	new_dw->getModifiable(X, X_label, matl, patch);
      else
	new_dw->allocateAndPut(X, X_label, matl, patch);
      
      Patch::VariableBasis basis = Patch::translateTypeToBasis(sol_type::getTypeDescription()->getType(), true);
      IntVector ec = params->getSolveOnExtraCells() ?
	IntVector(0,0,0) : -level->getExtraCells();
      IntVector l = patch->getLowIndex(basis, ec);
      IntVector h = patch->getHighIndex(basis, ec);
      CellIterator iter(l, h);

      IntVector size = h-l;
      int totsize = size.x()*size.y()*size.z();

      Array2<double> a(totsize, totsize);
      a.initialize(0);
      Array1<double> x(totsize);
      Array1<double> b(totsize);

      for(CellIterator i(iter); !i.done(); i++){
	IntVector idx = *i;
	IntVector idx2 = idx-l;
	int row = idx2.x()+idx2.y()*size.x()+idx2.z()*size.x()*size.y();
	b[row] = B[idx];
	const Stencil7& S = A[idx];
	a(row, row) = S.p;
	if(idx.x() > l.x()){
	  a(row, row-1) = S.w;
	} else {
	  ASSERTEQ(S.w, 0);
	}
	if(idx.x() < h.x()-1){
	  a(row, row+1) = S.e;
	} else {
	  ASSERTEQ(S.e, 0);
	}
	if(idx.y() > l.y()){
	  a(row, row-size.x()) = S.s;
	} else {
	  ASSERTEQ(S.s, 0);
	}
	if(idx.y() < h.y()-1){
	  a(row, row+size.x()) = S.n;
	} else {
	  ASSERTEQ(S.n, 0);
	}
	if(idx.z() > l.z()){
	  a(row, row-size.x()*size.y()) = S.b;
	} else {
	  ASSERTEQ(S.b, 0);
	}
	if(idx.z() < h.z()-1){
	  a(row, row+size.x()*size.y()) = S.t;
	} else {
	  ASSERTEQ(S.t, 0);
	}
      }

      // Check for symmetry
      for(int i=0;i<totsize;i++){
	for(int j=i+1;j<totsize;j++){
	  ASSERTEQ(a(i, j), a(j, i));
	}
      }

      int rows = totsize;
      for(int i=0;i<rows;i++)
	x[i]=b[i];

      // Gauss-Jordan with no pivoting
      for(int i=0;i<rows;i++){
	ASSERT(a(i, i) != 0);
	double denom=1./a(i, i);
	x[i]*=denom;
	for(int j=0;j<rows;j++){
	  a(i, j)*=denom;
	}
	flops += rows+1;
	memrefs += (rows+1)*sizeof(double);
	for(int j=i+1;j<rows;j++){
	  double factor=a(j, i);
	  x[j]-=factor*x[i];
	  flops += 2*rows+2;
	  memrefs += (2*rows+3)*sizeof(double);
	  for(int k=0;k<rows;k++){
	    a(j, k)-=factor*a(i, k);
	  }
	}
      }

      // Back-substitution
      for(int i=1;i<rows;i++){
	for(int j=0;j<i;j++){
	  double factor=a(j, i);
	  x[j]-=factor*x[i];
	  memrefs += (2*rows+2)*sizeof(double);
	  flops += (2*rows+2)*sizeof(double);
	  for(int k=0;k<rows;k++){
	    a(j, k)-=factor*a(i, k);
	  }
	}
      }

      for(CellIterator i(iter); !i.done(); i++){
	IntVector idx = *i;
	IntVector idx2 = idx-l;
	int row = idx2.x()+idx2.y()*size.x()+idx2.z()*size.x()*size.y();
	X[idx] = x[row];
      }
    }
    double dt=Time::currentSeconds()-tstart;
    double mflops = (double(flops)*1.e-6)/dt;
    double memrate = (double(memrefs)*1.e-9)/dt;
    if(pg->myrank() == 0){
      cerr << "Solve of " << X_label->getName() 
	   << " on level " << level->getIndex()
           << " completed in " << dt << " seconds (" 
	   << mflops << " MFLOPS, " << memrate << " GB/sec)\n";
    }
  }
    
private:
  const Level* level;
  const MaterialSet* matlset;
  const VarLabel* A_label;
  const VarLabel* X_label;
  const VarLabel* B_label;
  bool modifies_x;
  const DirectSolveParams* params;
};

SolverParameters* DirectSolve::readParameters(ProblemSpecP& params, const string& varname)
{
  DirectSolveParams* p = new DirectSolveParams();
  return p;
}

void DirectSolve::scheduleSolve(const LevelP& level, SchedulerP& sched,
			     const MaterialSet* matls,
			     const VarLabel* A, const VarLabel* x,
			     bool modifies_x,
                             const VarLabel* b, const VarLabel* guess,
			     Task::WhichDW guess_dw,
			     const SolverParameters* params)
{
  if(level->numPatches() != 1)
    throw InternalError("DirectSolve only works with 1 patch");

  Task* task;
  // The extra handle arg ensures that the stencil7 object will get freed
  // when the task gets freed.  The downside is that the refcount gets
  // tweaked everytime solve is called.

  TypeDescription::Type domtype = A->typeDescription()->getType();
  ASSERTEQ(domtype, x->typeDescription()->getType());
  ASSERTEQ(domtype, b->typeDescription()->getType());
  const DirectSolveParams* dparams = dynamic_cast<const DirectSolveParams*>(params);
  if(!dparams)
    throw InternalError("Wrong type of params passed to cg solver!");

  switch(domtype){
  case TypeDescription::SFCXVariable:
    {
      DirectStencil7<SFCXTypes>* that = new DirectStencil7<SFCXTypes>(level.get_rep(), matls, A, x, modifies_x, b, dparams);
      Handle<DirectStencil7<SFCXTypes> > handle = that;
      task = scinew Task("Matrix solve", that, &DirectStencil7<SFCXTypes>::solve, handle);
    }
    break;
  case TypeDescription::SFCYVariable:
    {
      DirectStencil7<SFCYTypes>* that = new DirectStencil7<SFCYTypes>(level.get_rep(), matls, A, x, modifies_x, b, dparams);
      Handle<DirectStencil7<SFCYTypes> > handle = that;
      task = scinew Task("Matrix solve", that, &DirectStencil7<SFCYTypes>::solve, handle);
    }
    break;
  case TypeDescription::SFCZVariable:
    {
      DirectStencil7<SFCZTypes>* that = new DirectStencil7<SFCZTypes>(level.get_rep(), matls, A, x, modifies_x, b, dparams);
      Handle<DirectStencil7<SFCZTypes> > handle = that;
      task = scinew Task("Matrix solve", that, &DirectStencil7<SFCZTypes>::solve, handle);
    }
    break;
  case TypeDescription::CCVariable:
    {
      DirectStencil7<CCTypes>* that = new DirectStencil7<CCTypes>(level.get_rep(), matls, A, x, modifies_x, b, dparams);
      Handle<DirectStencil7<CCTypes> > handle = that;
      task = scinew Task("Matrix solve", that, &DirectStencil7<CCTypes>::solve, handle);
    }
    break;
  case TypeDescription::NCVariable:
    {
      DirectStencil7<NCTypes>* that = new DirectStencil7<NCTypes>(level.get_rep(), matls, A, x, modifies_x, b, dparams);
      Handle<DirectStencil7<NCTypes> > handle = that;
      task = scinew Task("Matrix solve", that, &DirectStencil7<NCTypes>::solve, handle);
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
