
#include <Packages/Uintah/CCA/Components/Solvers/CGSolver.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>
#include <Packages/Uintah/Core/Grid/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/SFCZVariable.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Packages/Uintah/Core/Grid/Stencil7.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/CCA/Ports/LoadBalancer.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Core/Math/MiscMath.h>
#include <Core/Math/MinMax.h>
#include <Core/Thread/Time.h>
#include <iomanip>

using namespace Uintah;

static void Mult(Array3<double>& B, const Array3<Stencil7>& A,
		 const Array3<double>& X, CellIterator iter, long64& flops)
{
  IntVector l(X.getLowIndex());
  IntVector h1(X.getHighIndex()-IntVector(1,1,1));
  // Center
#if 1
  ASSERT(B.getWindow()->getOffset() == IntVector(0,0,0));
  ASSERT(A.getWindow()->getOffset() == IntVector(0,0,0));
  ASSERT(X.getWindow()->getOffset() == IntVector(0,0,0));
  IntVector  ll(iter.begin());
  IntVector hh(iter.end());
  double*** cb = B.get3DPointer();
  Stencil7*** ca = A.get3DPointer();
  double*** cx = X.get3DPointer();
  for(int z=ll.z();z<hh.z();z++){
    for(int y=ll.y();y<hh.y();y++){
      for(int x=ll.x();x<hh.x();x++){
	Stencil7* AA = &ca[z][y][x];
	double result = AA->p*cx[z][y][x];
	if(x > l.x())
	  result += AA->w*cx[z][y][x-1];
	if(x < h1.x())
	  result += AA->e*cx[z][y][x+1];
	if(y > l.y())
	  result += AA->s*cx[z][y-1][x];
	if(y < h1.y())
	  result += AA->n*cx[z][y+1][x];
	if(z > l.z())
	  result += AA->b*cx[z-1][y][x];
	if(z < h1.z())
	  result += AA->t*cx[z+1][y][x];
	cb[z][y][x] = result;
      }
    }
  }
  //mult(cb, ca, cx, &cll, &chh, &cl, &ch1);
#else
  for(; !iter.done(); ++iter){
    IntVector idx = *iter;
    const Stencil7& AA = A[idx];
    double result = AA.p*X[idx];
    if(idx.x() > l.x())
      result += AA.w*X[idx+IntVector(-1,0,0)];
    if(idx.x() < h1.x())
      result += AA.e*X[idx+IntVector(1,0,0)];
    if(idx.y() > l.y())
      result += AA.s*X[idx+IntVector(0,-1,0)];
    if(idx.y() < h1.y())
      result += AA.n*X[idx+IntVector(0,1,0)];
    if(idx.z() > l.z())
      result += AA.b*X[idx+IntVector(0,0,-1)];
    if(idx.z() < h1.z())
      result += AA.t*X[idx+IntVector(0,0,1)];
    B[idx] = result;
  }
#endif
  IntVector diff = iter.end()-iter.begin();
  flops += 13*diff.x()*diff.y()*diff.z();
}

static void Sub(Array3<double>& r, const Array3<double>& a,
		const Array3<double>& b,
		CellIterator iter, long64& flops)
{
  for(; !iter.done(); ++iter)
    r[*iter] = a[*iter]-b[*iter];
  IntVector diff = iter.end()-iter.begin();
  flops += diff.x()*diff.y()*diff.z();
}

static void DivDiagonal(Array3<double>& r, const Array3<double>& a,
			const Array3<Stencil7>& A,
			CellIterator iter, long64& flops)
{
  for(; !iter.done(); ++iter)
    r[*iter] = a[*iter]/A[*iter].p;
  IntVector diff = iter.end()-iter.begin();
  flops += diff.x()*diff.y()*diff.z();
}

static double L1(const Array3<double>& a, CellIterator iter, long64& flops)
{
  double sum=0;
  for(; !iter.done(); ++iter)
    sum += Abs(a[*iter]);
  IntVector diff = iter.end()-iter.begin();
  flops += 2*diff.x()*diff.y()*diff.z();
  return sum;
}

static double LInf(const Array3<double>& a, CellIterator iter, long64& flops)
{
  double max=0;
  for(; !iter.done(); ++iter)
    max = Max(max, Abs(a[*iter]));
  IntVector diff = iter.end()-iter.begin();
  flops += 2*diff.x()*diff.y()*diff.z();
  return max;
}

static double Dot(const Array3<double>& a, const Array3<double>& b,
		  CellIterator iter, long64& flops)
{
  double sum=0;
  for(; !iter.done(); ++iter)
    sum += a[*iter]*b[*iter];
  IntVector diff = iter.end()-iter.begin();
  flops += 2*diff.x()*diff.y()*diff.z();
  return sum;
}

static void ScMult_Add(Array3<double>& r, double s,
		       const Array3<double>& a, const Array3<double>& b,
		       CellIterator iter, long64& flops)
{
  for(; !iter.done(); ++iter)
    r[*iter] = s*a[*iter]+b[*iter];
  IntVector diff = iter.end()-iter.begin();
  flops += 2*diff.x()*diff.y()*diff.z();
}

namespace Uintah {

CGSolver::CGSolver(const ProcessorGroup* myworld)
  : UintahParallelComponent(myworld)
{
}

CGSolver::~CGSolver()
{
}

double*** get3DPointer(const Array3<double>& X)
{
  return X.get3DPointer();
}

Stencil7*** get3DPointer(const Array3<Stencil7>& X)
{
  return X.get3DPointer();
}

class SFCXTypes {
public:
  typedef constSFCXVariable<Stencil7> matrix_type;
  typedef constSFCXVariable<double> const_type;
  typedef SFCXVariable<double> sol_type;
};

class SFCYTypes {
public:
  typedef constSFCYVariable<Stencil7> matrix_type;
  typedef constSFCYVariable<double> const_type;
  typedef SFCYVariable<double> sol_type;
};

class SFCZTypes {
public:
  typedef constSFCZVariable<Stencil7> matrix_type;
  typedef constSFCZVariable<double> const_type;
  typedef SFCZVariable<double> sol_type;
};

class CCTypes {
public:
  typedef constCCVariable<Stencil7> matrix_type;
  typedef constCCVariable<double> const_type;
  typedef CCVariable<double> sol_type;
};

class NCTypes {
public:
  typedef constNCVariable<Stencil7> matrix_type;
  typedef constNCVariable<double> const_type;
  typedef NCVariable<double> sol_type;
};

class CGSolverParams : public SolverParameters {
public:
  double tolerance;
  double initial_tolerance;
  enum Norm {
    L1, L2, LInfinity
  };
  Norm norm;
  enum Criteria {
    Absolute, Relative
  };
  Criteria criteria;
  CGSolverParams()
    : tolerance(1.e-8), initial_tolerance(1.e-15), norm(L2), criteria(Relative)
  {
  }
  ~CGSolverParams() {}
};

template<class Types>
class CGStencil7 : public RefCounted {
public:
  CGStencil7(Scheduler* sched, const ProcessorGroup* world, const Level* level,
	     const MaterialSet* matlset, Ghost::GhostType Around,
	     const VarLabel* A, const VarLabel* x, bool modifies_x,
	     const VarLabel* b,
	     const VarLabel* guess, Task::WhichDW guess_dw,
	     const CGSolverParams* params)
    : sched(sched), world(world), level(level), matlset(matlset),
      Around(Around), A_label(A), X_label(x), B_label(b),
      guess_label(guess), guess_dw(guess_dw), params(params),
      modifies_x(modifies_x)
  {
    typedef typename Types::sol_type sol_type;
    R_label = VarLabel::create(A->getName()+" R", sol_type::getTypeDescription());
    D_label = VarLabel::create(A->getName()+" D", sol_type::getTypeDescription());
    Q_label = VarLabel::create(A->getName()+" Q", sol_type::getTypeDescription());
    d_label = VarLabel::create(A->getName()+" d", sum_vartype::getTypeDescription());
    aden_label = VarLabel::create(A->getName()+" aden", sum_vartype::getTypeDescription());
    VarLabel* tmp_flop_label = VarLabel::create(A->getName()+" flops", sumlong_vartype::getTypeDescription());
    tmp_flop_label->allowMultipleComputes();
    flop_label = tmp_flop_label;
    switch(params->norm){
    case CGSolverParams::L1:
      err_label = VarLabel::create(A->getName()+" err", sum_vartype::getTypeDescription());
      break;
    case CGSolverParams::L2:
      err_label = d_label;  // Uses d
      break;
    case CGSolverParams::LInfinity:
      err_label = VarLabel::create(A->getName()+" err", max_vartype::getTypeDescription());
      break;
    }
  }

  virtual ~CGStencil7() {
    VarLabel::destroy(R_label);
    VarLabel::destroy(D_label);
    VarLabel::destroy(Q_label);
    VarLabel::destroy(d_label);
    VarLabel::destroy(flop_label);
    if(err_label != d_label)
      VarLabel::destroy(err_label);
    VarLabel::destroy(aden_label);
  }

  void step1(const ProcessorGroup*, const PatchSubset* patches,
	     const MaterialSubset* matls,
	     DataWarehouse* old_dw, DataWarehouse* new_dw)
  {
    DataWarehouse* parent_new_dw = new_dw->getOtherDataWarehouse(Task::ParentNewDW);
    // Step 1 - requires A(parent), D(old, 1 ghost) computes aden(new)
    for(int p=0;p<patches->size();p++){
      const Patch* patch = patches->get(p);
      for(int m = 0;m<matls->size();m++){
	int matl = matls->get(m);

	typename Types::sol_type Q;
	new_dw->allocateAndPut(Q, Q_label, matl, patch);

	typename Types::matrix_type A;
	parent_new_dw->get(A, A_label, matl, patch, Ghost::None, 0);

	typename Types::const_type D;
	old_dw->get(D, D_label, matl, patch, Around, 1);

	typedef typename Types::sol_type sol_type;
	Patch::VariableBasis basis = Patch::translateTypeToBasis(sol_type::getTypeDescription()->getType(), true);
	IntVector l = patch->getLowIndex(basis);
	IntVector h = patch->getHighIndex(basis);
	CellIterator iter(l, h);

	// Q = A*D
	long64 flops = 0;
	// Must be qualified with :: for the IBM xlC compiler.
	::Mult(Q, A, D, iter, flops);
	double aden = ::Dot(D, Q, iter, flops);
	new_dw->put(sum_vartype(aden), aden_label);

	new_dw->put(sumlong_vartype(flops), flop_label);
      }
    }
  }

  void step2(const ProcessorGroup*, const PatchSubset* patches,
	     const MaterialSubset* matls,
	     DataWarehouse* old_dw, DataWarehouse* new_dw)
  {
    DataWarehouse* parent_new_dw = new_dw->getOtherDataWarehouse(Task::ParentNewDW);
    for(int p=0;p<patches->size();p++){
      const Patch* patch = patches->get(p);
      for(int m = 0;m<matls->size();m++){
	int matl = matls->get(m);
	typedef typename Types::sol_type sol_type;
	Patch::VariableBasis basis = Patch::translateTypeToBasis(sol_type::getTypeDescription()->getType(), true);
	IntVector l = patch->getLowIndex(basis);
	IntVector h = patch->getHighIndex(basis);
	CellIterator iter(l, h);

	// Step 2 - requires d(old), aden(new) D(old), X(old) R(old)  computes X, R, Q, d
	typename Types::const_type D;
	old_dw->get(D, D_label, matl, patch, Ghost::None, 0);

	typename Types::const_type X, R;
	old_dw->get(X, X_label, matl, patch, Ghost::None, 0);
	old_dw->get(R, R_label, matl, patch, Ghost::None, 0);
	typename Types::sol_type Xnew, Rnew;
	new_dw->allocateAndPut(Xnew, X_label, matl, patch, Ghost::None, 0);
	new_dw->allocateAndPut(Rnew, R_label, matl, patch, Ghost::None, 0);
	typename Types::sol_type Q;
	new_dw->getModifiable(Q, Q_label, matl, patch);

	sum_vartype aden;
	new_dw->get(aden, aden_label);

	sum_vartype d;
	old_dw->get(d, d_label);

	typename Types::matrix_type A;
	parent_new_dw->get(A, A_label, matl, patch, Ghost::None, 0);

	long64 flops = 0;
	double a=d/aden;

#if 0
	// X = a*D+X
	ScMult_Add(Xnew, a, D, X, iter, flops);
	// R = -a*Q+R
	ScMult_Add(Rnew, -a, Q, R, iter, flops);

	// Simple Preconditioning...
	DivDiagonal(Q, Rnew, A, iter, flops);

	// Calculate coefficient bk and direction vectors p and pp
	double dnew=Dot(Q, Rnew, iter, flops);

	// Calculate error term
	switch(params->norm){
	case CGSolverParams::L1:
	  {
	    double err = L1(Q, iter, flops);
	    new_dw->put(sum_vartype(err), err_label);
	  }
	  break;
	case CGSolverParams::L2:
	  // Nothing...
	  break;
	case CGSolverParams::LInfinity:
	  {
	    double err = LInf(Q, iter, flops);
	    new_dw->put(max_vartype(err), err_label);
	  }
	  break;
	}
#else
	double*** pXnew = Xnew.get3DPointer();
	double*** pD = get3DPointer(D);
	double*** pRnew = Rnew.get3DPointer();
	double*** pR = get3DPointer(R);
	double*** pQ = Q.get3DPointer();
	Stencil7*** pA = get3DPointer(A);
	double*** pX = get3DPointer(X);
	IntVector  ll(iter.begin());
	IntVector hh(iter.end());
	double dnew = 0;
	double err = 0;
	for(int z=ll.z();z<hh.z();z++){
	  for(int y=ll.y();y<hh.y();y++){
	    for(int x=ll.x();x<hh.x();x++){
	      // X = a*D+X
	      pXnew[z][y][x] = a*pD[z][y][x]+pX[z][y][x];
	      // R = -a*Q+R
	      double tmp1 = pRnew[z][y][x] = pR[z][y][x]-a*pQ[z][y][x];

	      // Simple Preconditioning...
	      double tmp2 = pQ[z][y][x] = tmp1/pA[z][y][x].p;

	      // Calculate coefficient bk and direction vectors p and pp
	      dnew += tmp1*tmp2;
	      double tmp3 = tmp2<0?-tmp2:tmp2;
	      err = tmp3 > err ? tmp3:err;
	    }
	  }
	}
	new_dw->put(max_vartype(err), err_label);
	IntVector diff = iter.end()-iter.begin();
	flops += 9*diff.x()*diff.y()*diff.z();
#endif
	new_dw->put(sum_vartype(dnew), d_label);
	new_dw->put(sumlong_vartype(flops), flop_label);
      }
    }
  }

  void step3(const ProcessorGroup*, const PatchSubset* patches,
	     const MaterialSubset* matls,
	     DataWarehouse* old_dw, DataWarehouse* new_dw)
  {
    for(int p=0;p<patches->size();p++){
      const Patch* patch = patches->get(p);
      for(int m = 0;m<matls->size();m++){
	int matl = matls->get(m);
	typedef typename Types::sol_type sol_type;
	Patch::VariableBasis basis = Patch::translateTypeToBasis(sol_type::getTypeDescription()->getType(), true);
	IntVector l = patch->getLowIndex(basis);
	IntVector h = patch->getHighIndex(basis);
	CellIterator iter(l, h);

	sum_vartype dnew, dold;
	old_dw->get(dold, d_label);
	new_dw->get(dnew, d_label);
	typename Types::const_type Q;
	new_dw->get(Q, Q_label, matl, patch, Ghost::None, 0);

	typename Types::const_type D;
	old_dw->get(D, D_label, matl, patch, Ghost::None, 0);

	// Step 3 - requires D(old), Q(new), d(new), d(old), computes D
	double b=dnew/dold;

	// D = b*D+Q
	typename Types::sol_type Dnew;
	new_dw->allocateAndPut(Dnew, D_label, matl, patch, Ghost::None, 0);
	long64 flops = 0;
	::ScMult_Add(Dnew, b, D, Q, iter, flops);
	new_dw->put(sumlong_vartype(flops), flop_label);
      }
    }
  }

  void setup(const ProcessorGroup*, const PatchSubset* patches,
	     const MaterialSubset* matls,
	     DataWarehouse*, DataWarehouse* new_dw)
  {
    DataWarehouse* parent_new_dw = new_dw->getOtherDataWarehouse(Task::ParentNewDW);
    DataWarehouse* parent_old_dw = new_dw->getOtherDataWarehouse(Task::ParentOldDW);
    for(int p=0;p<patches->size();p++){
      const Patch* patch = patches->get(p);
      for(int m = 0;m<matls->size();m++){
	int matl = matls->get(m);
	typedef typename Types::sol_type sol_type;
	Patch::VariableBasis basis = Patch::translateTypeToBasis(sol_type::getTypeDescription()->getType(), true);
	IntVector l = patch->getLowIndex(basis);
	IntVector h = patch->getHighIndex(basis);
	CellIterator iter(l, h);

	typename Types::sol_type R, Xnew;
	new_dw->allocateAndPut(R, R_label, matl, patch);
	new_dw->allocateAndPut(Xnew, X_label, matl, patch);
	typename Types::const_type B;
	parent_new_dw->get(B, B_label, matl, patch, Ghost::None, 0);

	typename Types::matrix_type A;
	parent_new_dw->get(A, A_label, matl, patch, Ghost::None, 0);

	long64 flops = 0;
	if(guess_label){
	  typename Types::const_type X;
	  if(guess_dw == Task::OldDW)
	    parent_old_dw->get(X, guess_label, matl, patch, Around, 1);
	  else
	    parent_new_dw->get(X, guess_label, matl, patch, Around, 1);

	  // R = A*X
	  ::Mult(R, A, X, iter, flops);

	  // R = B-R
	  ::Sub(R, B, R, iter, flops);
	  Xnew.copy(X, iter.begin(), iter.end());
	} else {
	  R.copy(B);
	  Xnew.initialize(0);
	}

	// D = R/Ap
	typename Types::sol_type D;
	new_dw->allocateAndPut(D, D_label, matl, patch);
	::DivDiagonal(D, R, A, iter, flops);

	double dnew = ::Dot(R, D, iter, flops);
	new_dw->put(sum_vartype(dnew), d_label);
	switch(params->norm){
	case CGSolverParams::L1:
	  {
	    double err = ::L1(R, iter, flops);
	    new_dw->put(sum_vartype(err), err_label);
	  }
	  break;
	case CGSolverParams::L2:
	  // Nothing...
	  break;
	case CGSolverParams::LInfinity:
	  {
	    double err = ::LInf(R, iter, flops);
	    new_dw->put(max_vartype(err), err_label);
	  }
	  break;
	}
	new_dw->put(sumlong_vartype(flops), flop_label);
      }
    }
  }

  void solve(const ProcessorGroup*, const PatchSubset* patches,
	     const MaterialSubset* matls,
	     DataWarehouse* old_dw, DataWarehouse* new_dw,
	     Handle<CGStencil7<Types> >)
  {
    double tstart = Time::currentSeconds();
    SchedulerP subsched = sched->createSubScheduler();
    DataWarehouse::ScrubMode old_dw_scrubmode = old_dw->setScrubbing(DataWarehouse::ScrubNone);
    DataWarehouse::ScrubMode new_dw_scrubmode = new_dw->setScrubbing(DataWarehouse::ScrubNone);
    subsched->initialize(3, 1, old_dw, new_dw);
    subsched->clearMappings();
    subsched->mapDataWarehouse(Task::ParentOldDW, 0);
    subsched->mapDataWarehouse(Task::ParentNewDW, 1);
    subsched->mapDataWarehouse(Task::OldDW, 2);
    subsched->mapDataWarehouse(Task::NewDW, 3);
  
    GridP grid = level->getGrid();
    IntVector l, h;
    level->findCellIndexRange(l, h);

    int niter=0;
    int toomany=0;
    IntVector diff(h-l);
    int size = diff.x()*diff.y()*diff.z();
    if(toomany == 0)
      toomany=2*size;

    subsched->advanceDataWarehouse(grid);

    // Schedule the setup
    Task* task = scinew Task("CGSolve setup", this, &CGStencil7<Types>::setup);
    task->requires(Task::ParentNewDW, B_label, Ghost::None, 0);
    task->requires(Task::ParentNewDW, A_label, Ghost::None, 0);
    if(guess_label){
      if(guess_dw == Task::OldDW)
	task->requires(Task::ParentOldDW, guess_label, Around, 1);
      else
	task->requires(Task::ParentNewDW, guess_label, Around, 1);
    }
    task->computes(R_label);
    task->computes(X_label);
    task->computes(D_label);
    task->computes(d_label);
    if(params->norm != CGSolverParams::L2)
      task->computes(err_label);
    task->computes(flop_label);
    subsched->addTask(task, level->eachPatch(), matlset);

    subsched->compile(world);
    subsched->get_dw(3)->setScrubbing(DataWarehouse::ScrubNone);
    subsched->execute(world);

    double e=0;
    switch(params->norm){
    case CGSolverParams::L1:
    case CGSolverParams::L2:
      {
	sum_vartype err;
	subsched->get_dw(3)->get(err, err_label);
	e=err;
      }
      break;
    case CGSolverParams::LInfinity:
      {
	max_vartype err;
	subsched->get_dw(3)->get(err, err_label);
	e=err;
      }
      break;
    }
    double err0=e;
    sumlong_vartype f;
    subsched->get_dw(3)->get(f, flop_label);
    long64 flops = f;
    if(!(e < params->initial_tolerance)) {
      subsched->initialize(3, 1, old_dw, new_dw);
      subsched->clearMappings();
      subsched->mapDataWarehouse(Task::ParentOldDW, 0);
      subsched->mapDataWarehouse(Task::ParentNewDW, 1);
      subsched->mapDataWarehouse(Task::OldDW, 2);
      subsched->mapDataWarehouse(Task::NewDW, 3);
  
      // Schedule the iteration
      // Step 1 - requires A(parent), D(old, 1 ghost) computes aden(new)
      task = scinew Task("CGSolve step1", this, &CGStencil7<Types>::step1);
      task->requires(Task::ParentNewDW, A_label, Ghost::None, 0);
      task->requires(Task::OldDW, D_label, Around, 1);
      task->computes(aden_label);
      task->computes(Q_label);
      task->computes(flop_label);
      subsched->addTask(task, level->eachPatch(), matlset);

      // Step 2 - requires d(old), aden(new) D(old), X(old) R(old)  computes X, R, Q, d
      task = scinew Task("CGSolve step2", this, &CGStencil7<Types>::step2);
      task->requires(Task::OldDW, d_label);
      task->requires(Task::NewDW, aden_label);
      task->requires(Task::OldDW, D_label, Ghost::None, 0);
      task->requires(Task::OldDW, X_label, Ghost::None, 0);
      task->requires(Task::OldDW, R_label, Ghost::None, 0);
      task->computes(X_label);
      task->computes(R_label);
      task->modifies(Q_label);
      task->computes(d_label);
      if(params->norm != CGSolverParams::L1)
	task->computes(err_label);
      task->computes(flop_label);
      subsched->addTask(task, level->eachPatch(), matlset);

      // Step 3 - requires D(old), Q(new), d(new), d(old), computes D
      task = scinew Task("CGSolve step3", this, &CGStencil7<Types>::step3);
      task->requires(Task::OldDW, D_label, Ghost::None, 0);
      task->requires(Task::NewDW, Q_label, Ghost::None, 0);
      task->requires(Task::NewDW, d_label);
      task->requires(Task::OldDW, d_label);
      task->computes(D_label);
      task->computes(flop_label);
      subsched->addTask(task, level->eachPatch(), matlset);
      subsched->compile(world);

      while(niter < toomany && !(e < params->tolerance)){
	niter++;
	subsched->advanceDataWarehouse(grid);
	subsched->get_dw(2)->setScrubbing(DataWarehouse::ScrubComplete);
	subsched->get_dw(3)->setScrubbing(DataWarehouse::ScrubNonPermanent);
	subsched->execute(world);
	switch(params->norm){
	case CGSolverParams::L1:
	case CGSolverParams::L2:
	  {
	    sum_vartype err;
	    subsched->get_dw(3)->get(err, err_label);
	    e=err;
	  }
	  break;
	case CGSolverParams::LInfinity:
	  {
	    max_vartype err;
	    subsched->get_dw(3)->get(err, err_label);
	    e=err;
	  }
	  break;
	}
	if(params->criteria == CGSolverParams::Relative)
	  e/=err0;
	sumlong_vartype f;
	subsched->get_dw(3)->get(f, flop_label);
	flops += f;
      }
    }

    // Pull the solution out of subsched new DW and put it in ours
    if(modifies_x){
      for(int p=0;p<patches->size();p++){
	const Patch* patch = patches->get(p);
	for(int m = 0;m<matls->size();m++){
	  int matl = matls->get(m);
	  typedef typename Types::sol_type sol_type;
	  Patch::VariableBasis basis = Patch::translateTypeToBasis(sol_type::getTypeDescription()->getType(), true);
	  IntVector l = patch->getLowIndex(basis);
	  IntVector h = patch->getHighIndex(basis);
	  CellIterator iter(l, h);

	  typename Types::sol_type Xnew;
	  new_dw->getModifiable(Xnew, X_label, matl, patch);
	  typename Types::const_type X;
	  subsched->get_dw(3)->get(X, X_label, matl, patch, Ghost::None, 0);
	  Xnew.copy(X);
	}
      }
    } else {
      new_dw->transferFrom(subsched->get_dw(3), X_label, patches, matls);
    }

    // Restore the scrubbing mode
    old_dw->setScrubbing(old_dw_scrubmode);
    new_dw->setScrubbing(new_dw_scrubmode);

    double dt=Time::currentSeconds()-tstart;
    double mflops = (double(flops)*1.e-6)/dt;
    cerr << "Solve of " << X_label->getName();
    if(niter < toomany)
      cerr << " completed in ";
    else
      cerr << " FAILED in ";
    cerr << niter << " iterations, " << dt << " seconds (" << mflops << " MFLOPS)\n";
  }
    
private:
  Scheduler* sched;
  const ProcessorGroup* world;
  const Level* level;
  const MaterialSet* matlset;
  Ghost::GhostType Around;
  const VarLabel* A_label;
  const VarLabel* X_label;
  const VarLabel* B_label;
  const VarLabel* R_label;
  const VarLabel* D_label;
  const VarLabel* Q_label;
  const VarLabel* d_label;
  const VarLabel* err_label;
  const VarLabel* aden_label;
  const VarLabel* guess_label;
  const VarLabel* flop_label;
  Task::WhichDW guess_dw;
  const CGSolverParams* params;
  bool modifies_x;
};

SolverParameters* CGSolver::readParameters(const ProblemSpecP& params, const string& varname)
{
  CGSolverParams* p = new CGSolverParams();
  if(params){
    for(ProblemSpecP param = params->findBlock("Parameters"); param != 0;
	param = param->findNextBlock("Parameters")) {
      string variable;
      if(param->getAttribute("variable", variable) && variable != varname)
	continue;
      param->get("initial_tolerance", p->initial_tolerance);
      param->get("tolerance", p->tolerance);
      string norm;
      if(param->get("norm", norm)){
	if(norm == "L1" || norm == "l1") {
	  p->norm = CGSolverParams::L1;
	} else if(norm == "L2" || norm == "l2") {
	  p->norm = CGSolverParams::L2;
	} else if(norm == "LInfinity" || norm == "linfinity") {
	  p->norm = CGSolverParams::LInfinity;
	} else {
	  throw ProblemSetupException("Unknown norm type: "+norm);
	}
      }
      string criteria;
      if(param->get("criteria", criteria)){
	if(criteria == "Absolute" || criteria == "absolute") {
	  p->criteria = CGSolverParams::Absolute;
	} else if(criteria == "Relative" || criteria == "relative") {
	  p->criteria = CGSolverParams::Relative;
	} else {
	  throw ProblemSetupException("Unknown criteria: "+criteria);
	}
      }
    }
  }
  if(p->norm == CGSolverParams::L2)
    p->tolerance *= p->tolerance;
  return p;
}

void CGSolver::scheduleSolve(const LevelP& level, SchedulerP& sched,
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
  const CGSolverParams* cgparams = dynamic_cast<const CGSolverParams*>(params);
  if(!cgparams)
    throw InternalError("Wrong type of params passed to cg solver!");

  Ghost::GhostType Around;

  switch(domtype){
  case TypeDescription::SFCXVariable:
    {
      Around = Ghost::AroundFaces;
      CGStencil7<SFCXTypes>* that = new CGStencil7<SFCXTypes>(sched.get_rep(), d_myworld, level.get_rep(), matls, Around, A, x, modifies_x, b, guess, guess_dw, cgparams);
      Handle<CGStencil7<SFCXTypes> > handle = that;
      task = scinew Task("Matrix solve", that, &CGStencil7<SFCXTypes>::solve, handle);
    }
    break;
  case TypeDescription::SFCYVariable:
    {
      Around = Ghost::AroundFaces;
      CGStencil7<SFCYTypes>* that = new CGStencil7<SFCYTypes>(sched.get_rep(), d_myworld, level.get_rep(), matls, Around, A, x, modifies_x, b, guess, guess_dw, cgparams);
      Handle<CGStencil7<SFCYTypes> > handle = that;
      task = scinew Task("Matrix solve", that, &CGStencil7<SFCYTypes>::solve, handle);
    }
    break;
  case TypeDescription::SFCZVariable:
    {
      Around = Ghost::AroundFaces;
      CGStencil7<SFCZTypes>* that = new CGStencil7<SFCZTypes>(sched.get_rep(), d_myworld, level.get_rep(), matls, Around, A, x, modifies_x, b, guess, guess_dw, cgparams);
      Handle<CGStencil7<SFCZTypes> > handle = that;
      task = scinew Task("Matrix solve", that, &CGStencil7<SFCZTypes>::solve, handle);
    }
    break;
  case TypeDescription::CCVariable:
    {
      Around = Ghost::AroundCells;
      CGStencil7<CCTypes>* that = new CGStencil7<CCTypes>(sched.get_rep(), d_myworld, level.get_rep(), matls, Around, A, x, modifies_x, b, guess, guess_dw, cgparams);
      Handle<CGStencil7<CCTypes> > handle = that;
      task = scinew Task("Matrix solve", that, &CGStencil7<CCTypes>::solve, handle);
    }
    break;
  case TypeDescription::NCVariable:
    {
      Around = Ghost::AroundNodes;
      CGStencil7<NCTypes>* that = new CGStencil7<NCTypes>(sched.get_rep(), d_myworld, level.get_rep(), matls, Around, A, x, modifies_x, b, guess, guess_dw, cgparams);
      Handle<CGStencil7<NCTypes> > handle = that;
      task = scinew Task("Matrix solve", that, &CGStencil7<NCTypes>::solve, handle);
    }
    break;
  default:
    throw InternalError("Unknown variable type in scheduleSolve");
  }

  task->requires(Task::NewDW, A, Ghost::None, 0);
  if(guess)
    task->requires(guess_dw, guess, Around, 1);
  if(modifies_x)
    task->modifies(guess);
  else
    task->computes(x);

  task->requires(Task::NewDW, b, Ghost::None, 0);
  task->hasSubScheduler();
  LoadBalancer* lb = sched->getLoadBalancer();
  const PatchSet* perproc_patches = lb->createPerProcessorPatchSet(level, 
								   d_myworld);
  sched->addTask(task, perproc_patches, matls);
}

} // end namespace Uintah
