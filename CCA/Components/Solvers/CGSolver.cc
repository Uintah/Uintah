
// TODO
//  - dynamic compilation tests
//  - Bench on cluster
//  - Can we do any better?  Tiling? ???

#include <Packages/Uintah/CCA/Components/Solvers/CGSolver.h>
#include <Packages/Uintah/CCA/Components/Solvers/MatrixUtil.h>
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
#include <Packages/Uintah/Core/Exceptions/ConvergenceFailure.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/CCA/Ports/LoadBalancer.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
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

#define OPT 1

void Mult(Array3<double>& B, const Array3<Stencil7>& A,
	  const Array3<double>& X, CellIterator iter,
	  const IntVector& l, const IntVector& h1, long64& flops, long64& memrefs)
{
  if(cout_doing.active())
    cout_doing << "CGSolver::Mult" << endl;
  // Center
#if OPT
  IntVector  ll(iter.begin());
  IntVector hh(iter.end());
  // Zlow
  int z=ll.z();
  if(z <= l.z()){
    for(int y=ll.y();y<hh.y();y++){
      const Stencil7* caa = &A[IntVector(ll.x(),y,z)]-ll.x();
      double* cbb = &B[IntVector(ll.x(),y,z)]-ll.x();
      const double* cx0 = &X[IntVector(ll.x(),y,z)]-ll.x();
      const double* cx1;
      if(y > l.y())
	cx1 = &X[IntVector(ll.x(),y-1,z)]-ll.x();
      else
	cx1 = 0;
      const double* cx2;
      if(y < h1.y())
	cx2 = &X[IntVector(ll.x(),y+1,z)]-ll.x();
      else
	cx2 = 0;
      const double* cx4 = &X[IntVector(ll.x(),y,z+1)]-ll.x();
      for(int x=ll.x();x<hh.x();x++){
        const Stencil7* AA = &caa[x];
        double result = AA->p*cx0[x];
        if(x > l.x())
          result += AA->w*cx0[x-1];
        if(x < h1.x())
          result += AA->e*cx0[x+1];
        if(y > l.y())
          result += AA->s*cx1[x];
        if(y < h1.y())
          result += AA->n*cx2[x];
	if(z < h1.z())
	  result += AA->t*cx4[x];
        cbb[x] = result;
      }
    }
    z++;
  }
  // Zmid
  for(;z<h1.z();z++){
    // Ylow
    int y=ll.y();
    if(y <= l.y()){
      const Stencil7* caa = &A[IntVector(ll.x(),y,z)]-ll.x();
      double* cbb = &B[IntVector(ll.x(),y,z)]-ll.x();
      const double* cx0 = &X[IntVector(ll.x(),y,z)]-ll.x();
      const double* cx2 = &X[IntVector(ll.x(),y+1,z)]-ll.x();
      const double* cx3 = &X[IntVector(ll.x(),y,z-1)]-ll.x();
      const double* cx4 = &X[IntVector(ll.x(),y,z+1)]-ll.x();
      for(int x=ll.x();x<hh.x();x++){
        const Stencil7* AA = &caa[x];
        double result = AA->p*cx0[x];
        if(x > l.x())
          result += AA->w*cx0[x-1];
        if(x < h1.x())
          result += AA->e*cx0[x+1];
	result += AA->n*cx2[x];
	result += AA->b*cx3[x];
	result += AA->t*cx4[x];
        cbb[x] = result;
      }
      y++;
    }
    // Ymid
    for(;y<h1.y();y++){
      const Stencil7* caa = &A[IntVector(ll.x(),y,z)]-ll.x();
      double* cbb = &B[IntVector(ll.x(),y,z)]-ll.x();
      const double* cx0 = &X[IntVector(ll.x(),y,z)]-ll.x();
      const double* cx1 = &X[IntVector(ll.x(),y-1,z)]-ll.x();
      const double* cx2 = &X[IntVector(ll.x(),y+1,z)]-ll.x();
      const double* cx3 = &X[IntVector(ll.x(),y,z-1)]-ll.x();
      const double* cx4 = &X[IntVector(ll.x(),y,z+1)]-ll.x();

      // Xlow
      int x=ll.x();
      if(x <= l.x()){
        const Stencil7* AA = &caa[x];
        double result = AA->p*cx0[x];
	result += AA->e*cx0[x+1];
	result += AA->s*cx1[x];
	result += AA->n*cx2[x];
	result += AA->b*cx3[x];
	result += AA->t*cx4[x];
        cbb[x] = result;
	x++;
      }
      // Xmid
      for(;x<h1.x();x++){
        const Stencil7* AA = &caa[x];
        double result = AA->p*cx0[x];
	result += AA->w*cx0[x-1];
	result += AA->e*cx0[x+1];
	result += AA->s*cx1[x];
	result += AA->n*cx2[x];
	result += AA->b*cx3[x];
	result += AA->t*cx4[x];
        cbb[x] = result;
      }
      // Xhigh
      if(x < hh.x()){
        const Stencil7* AA = &caa[x];
        double result = AA->p*cx0[x];
	result += AA->w*cx0[x-1];
	result += AA->s*cx1[x];
	result += AA->n*cx2[x];
	result += AA->b*cx3[x];
	result += AA->t*cx4[x];
        cbb[x] = result;
      }
    }
    // Yhigh
    if(y < hh.y()){
      const Stencil7* caa = &A[IntVector(ll.x(),y,z)]-ll.x();
      double* cbb = &B[IntVector(ll.x(),y,z)]-ll.x();
      const double* cx0 = &X[IntVector(ll.x(),y,z)]-ll.x();
      const double* cx1 = &X[IntVector(ll.x(),y-1,z)]-ll.x();
      const double* cx3 = &X[IntVector(ll.x(),y,z-1)]-ll.x();
      const double* cx4 = &X[IntVector(ll.x(),y,z+1)]-ll.x();
      for(int x=ll.x();x<hh.x();x++){
        const Stencil7* AA = &caa[x];
        double result = AA->p*cx0[x];
        if(x > l.x())
          result += AA->w*cx0[x-1];
        if(x < h1.x())
          result += AA->e*cx0[x+1];
	result += AA->s*cx1[x];
	result += AA->b*cx3[x];
	result += AA->t*cx4[x];
        cbb[x] = result;
      }
    }
  }
  // Zhigh
  if(z < hh.z()){
    for(int y=ll.y();y<hh.y();y++){
      const Stencil7* caa = &A[IntVector(ll.x(),y,z)]-ll.x();
      double* cbb = &B[IntVector(ll.x(),y,z)]-ll.x();
      const double* cx0 = &X[IntVector(ll.x(),y,z)]-ll.x();
      const double* cx1;
      if(y > l.y())
	cx1 = &X[IntVector(ll.x(),y-1,z)]-ll.x();
      else
	cx1 = 0;
      const double* cx2;
      if(y < h1.y())
	cx2 = &X[IntVector(ll.x(),y+1,z)]-ll.x();
      else
	cx2 = 0;
      const double* cx3 = &X[IntVector(ll.x(),y,z-1)]-ll.x();
      for(int x=ll.x();x<hh.x();x++){
        const Stencil7* AA = &caa[x];
        double result = AA->p*cx0[x];
        if(x > l.x())
          result += AA->w*cx0[x-1];
        if(x < h1.x())
          result += AA->e*cx0[x+1];
        if(y > l.y())
          result += AA->s*cx1[x];
        if(y < h1.y())
          result += AA->n*cx2[x];
	result += AA->b*cx3[x];
        cbb[x] = result;
      }
    }
  }
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
  memrefs += 15L*diff.x()*diff.y()*diff.z()*8L;
}

void Mult(Array3<double>& B, const Array3<Stencil7>& A,
	  const Array3<double>& X, CellIterator iter,
	  const IntVector& l, const IntVector& h1, long64& flops,
	  long64& memrefs, double& dotresult)
{
  if(cout_doing.active())
    cout_doing << "CGSolver::Mult" << endl;
  // Center
#if OPT
  double dot=0;
  IntVector  ll(iter.begin());
  IntVector hh(iter.end());
  // Zlow
  int z=ll.z();
  if(z <= l.z()){
    for(int y=ll.y();y<hh.y();y++){
      const Stencil7* caa = &A[IntVector(ll.x(),y,z)]-ll.x();
      double* cbb = &B[IntVector(ll.x(),y,z)]-ll.x();
      const double* cx0 = &X[IntVector(ll.x(),y,z)]-ll.x();
      const double* cx1;
      const double* cx2;
      if(y > l.y())
	cx1 = &X[IntVector(ll.x(),y-1,z)]-ll.x();
      else
	cx1 = 0;
      if(y < h1.y())
	cx2 = &X[IntVector(ll.x(),y+1,z)]-ll.x();
      else
	cx2 = 0;
      const double* cx4 = &X[IntVector(ll.x(),y,z+1)]-ll.x();
      for(int x=ll.x();x<hh.x();x++){
        const Stencil7* AA = &caa[x];
        double result = AA->p*cx0[x];
        if(x > l.x())
          result += AA->w*cx0[x-1];
        if(x < h1.x())
          result += AA->e*cx0[x+1];
        if(y > l.y())
          result += AA->s*cx1[x];
        if(y < h1.y())
          result += AA->n*cx2[x];
	if(z < h1.z())
	  result += AA->t*cx4[x];
        cbb[x] = result;
	dot += cx0[x]*result;
      }
    }
    z++;
  }
  // Zmid
  for(;z<h1.z();z++){
    // Ylow
    int y=ll.y();
    if(y <= l.y()){
      const Stencil7* caa = &A[IntVector(ll.x(),y,z)]-ll.x();
      double* cbb = &B[IntVector(ll.x(),y,z)]-ll.x();
      const double* cx0 = &X[IntVector(ll.x(),y,z)]-ll.x();
      const double* cx2 = &X[IntVector(ll.x(),y+1,z)]-ll.x();
      const double* cx3 = &X[IntVector(ll.x(),y,z-1)]-ll.x();
      const double* cx4 = &X[IntVector(ll.x(),y,z+1)]-ll.x();
      for(int x=ll.x();x<hh.x();x++){
        const Stencil7* AA = &caa[x];
        double result = AA->p*cx0[x];
        if(x > l.x())
          result += AA->w*cx0[x-1];
        if(x < h1.x())
          result += AA->e*cx0[x+1];
	result += AA->n*cx2[x];
	result += AA->b*cx3[x];
	result += AA->t*cx4[x];
        cbb[x] = result;
	dot += cx0[x]*result;
      }
      y++;
    }
    // Ymid
    for(;y<h1.y();y++){
      const Stencil7* caa = &A[IntVector(ll.x(),y,z)]-ll.x();
      double* cbb = &B[IntVector(ll.x(),y,z)]-ll.x();
      const double* cx0 = &X[IntVector(ll.x(),y,z)]-ll.x();
      const double* cx1 = &X[IntVector(ll.x(),y-1,z)]-ll.x();
      const double* cx2 = &X[IntVector(ll.x(),y+1,z)]-ll.x();
      const double* cx3 = &X[IntVector(ll.x(),y,z-1)]-ll.x();
      const double* cx4 = &X[IntVector(ll.x(),y,z+1)]-ll.x();

      // Xlow
      int x=ll.x();
      if(x <= l.x()){
        const Stencil7* AA = &caa[x];
        double result = AA->p*cx0[x];
	result += AA->e*cx0[x+1];
	result += AA->s*cx1[x];
	result += AA->n*cx2[x];
	result += AA->b*cx3[x];
	result += AA->t*cx4[x];
        cbb[x] = result;
	dot += cx0[x]*result;
	x++;
      }
      // Xmid
      for(;x<h1.x();x++){
        const Stencil7* AA = &caa[x];
        double result = AA->p*cx0[x];
	result += AA->w*cx0[x-1];
	result += AA->e*cx0[x+1];
	result += AA->s*cx1[x];
	result += AA->n*cx2[x];
	result += AA->b*cx3[x];
	result += AA->t*cx4[x];
        cbb[x] = result;
	dot += cx0[x]*result;
      }
      // Xhigh
      if(x < hh.x()){
        const Stencil7* AA = &caa[x];
        double result = AA->p*cx0[x];
	result += AA->w*cx0[x-1];
	result += AA->s*cx1[x];
	result += AA->n*cx2[x];
	result += AA->b*cx3[x];
	result += AA->t*cx4[x];
        cbb[x] = result;
	dot += cx0[x]*result;
      }
    }
    // Yhigh
    if(y < hh.y()){
      const Stencil7* caa = &A[IntVector(ll.x(),y,z)]-ll.x();
      double* cbb = &B[IntVector(ll.x(),y,z)]-ll.x();
      const double* cx0 = &X[IntVector(ll.x(),y,z)]-ll.x();
      const double* cx1 = &X[IntVector(ll.x(),y-1,z)]-ll.x();
      const double* cx3 = &X[IntVector(ll.x(),y,z-1)]-ll.x();
      const double* cx4 = &X[IntVector(ll.x(),y,z+1)]-ll.x();
      for(int x=ll.x();x<hh.x();x++){
        const Stencil7* AA = &caa[x];
        double result = AA->p*cx0[x];
        if(x > l.x())
          result += AA->w*cx0[x-1];
        if(x < h1.x())
          result += AA->e*cx0[x+1];
	result += AA->s*cx1[x];
	result += AA->b*cx3[x];
	result += AA->t*cx4[x];
        cbb[x] = result;
	dot += cx0[x]*result;
      }
    }
  }
  // Zhigh
  if(z < hh.z()){
    for(int y=ll.y();y<hh.y();y++){
      const Stencil7* caa = &A[IntVector(ll.x(),y,z)]-ll.x();
      double* cbb = &B[IntVector(ll.x(),y,z)]-ll.x();
      const double* cx0 = &X[IntVector(ll.x(),y,z)]-ll.x();
      const double* cx1;
      if(y > l.y())
	cx1 = &X[IntVector(ll.x(),y-1,z)]-ll.x();
      else
	cx1 = 0;
      const double* cx2;
      if(y < h1.y())
	cx2 = &X[IntVector(ll.x(),y+1,z)]-ll.x();
      else
	cx2 = 0;
      const double* cx3 = &X[IntVector(ll.x(),y,z-1)]-ll.x();
      for(int x=ll.x();x<hh.x();x++){
        const Stencil7* AA = &caa[x];
        double result = AA->p*cx0[x];
        if(x > l.x())
          result += AA->w*cx0[x-1];
        if(x < h1.x())
          result += AA->e*cx0[x+1];
        if(y > l.y())
          result += AA->s*cx1[x];
        if(y < h1.y())
          result += AA->n*cx2[x];
	result += AA->b*cx3[x];
        cbb[x] = result;
	dot += cx0[x]*result;
      }
    }
  }

#else
  double dot=0;
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
    dot+=result*X[idx];
  }
#endif
  dotresult=dot;
  IntVector diff = iter.end()-iter.begin();
  flops += 15*diff.x()*diff.y()*diff.z();
  memrefs += 16L*diff.x()*diff.y()*diff.z()*8L;
}

static void Sub(Array3<double>& r, const Array3<double>& a,
		const Array3<double>& b,
		CellIterator iter, long64& flops, long64& memrefs)
{
  if(cout_doing.active())
    cout_doing << "CGSolver::Sub" << endl;
  for(; !iter.done(); ++iter)
    r[*iter] = a[*iter]-b[*iter];
  IntVector diff = iter.end()-iter.begin();
  flops += diff.x()*diff.y()*diff.z();
  memrefs += diff.x()*diff.y()*diff.z()*3L*8L;
}

void Mult(Array3<double>& r, const Array3<double>& a,
	  const Array3<double>& b,
	  CellIterator iter, long64& flops, long64& memrefs)
{
  if(cout_doing.active())
    cout_doing << "CGSolver::Mult" << endl;
  for(; !iter.done(); ++iter)
    r[*iter] = a[*iter]*b[*iter];
  IntVector diff = iter.end()-iter.begin();
  flops += diff.x()*diff.y()*diff.z();
  memrefs += diff.x()*diff.y()*diff.z()*3L*8L;
}

#if 0
static void DivDiagonal(Array3<double>& r, const Array3<double>& a,
			const Array3<Stencil7>& A,
			CellIterator iter, long64& flops)
{
  for(; !iter.done(); ++iter)
    r[*iter] = a[*iter]/A[*iter].p;
  IntVector diff = iter.end()-iter.begin();
  flops += diff.x()*diff.y()*diff.z();
}
#endif

static void InverseDiagonal(Array3<double>& r, const Array3<Stencil7>& A,
			    CellIterator iter, long64& flops, long64& memrefs)
{
  if(cout_doing.active())
    cout_doing << "CGSolver::InverseDiagonal" << endl;
  for(; !iter.done(); ++iter)
    r[*iter] = 1./A[*iter].p;
  IntVector diff = iter.end()-iter.begin();
  flops += diff.x()*diff.y()*diff.z();
  memrefs += 2L*diff.x()*diff.y()*diff.z()*8L;
}

static double L1(const Array3<double>& a, CellIterator iter, long64& flops,
		 long64& memrefs)
{
  if(cout_doing.active())
    cout_doing << "CGSolver::L1" << endl;
  double sum=0;
  for(; !iter.done(); ++iter)
    sum += Abs(a[*iter]);
  IntVector diff = iter.end()-iter.begin();
  flops += 2*diff.x()*diff.y()*diff.z();
  memrefs += diff.x()*diff.y()*diff.z()*8L;
  return sum;
}

double LInf(const Array3<double>& a, CellIterator iter, long64& flops,
	    long64& memrefs)
{
  if(cout_doing.active())
    cout_doing << "CGSolver::Linf" << endl;
  double max=0;
  for(; !iter.done(); ++iter)
    max = Max(max, Abs(a[*iter]));
  IntVector diff = iter.end()-iter.begin();
  flops += 2*diff.x()*diff.y()*diff.z();
  memrefs += diff.x()*diff.y()*diff.z()*8L;
  return max;
}

double Dot(const Array3<double>& a, const Array3<double>& b,
	   CellIterator iter, long64& flops, long64& memrefs)
{
  if(cout_doing.active())
    cout_doing << "CGSolver::Dot" << endl;
  double sum=0;
  for(; !iter.done(); ++iter)
    sum += a[*iter]*b[*iter];
  IntVector diff = iter.end()-iter.begin();
  flops += 2*diff.x()*diff.y()*diff.z();
  memrefs += diff.x()*diff.y()*diff.z()*2L*8L;
  return sum;
}

void ScMult_Add(Array3<double>& r, double s,
		const Array3<double>& a, const Array3<double>& b,
		CellIterator iter, long64& flops, long64& memrefs)
{
  if(cout_doing.active())
    cout_doing << "CGSolver::ScMult_Add" << endl;
#if OPT
  IntVector ll(iter.begin());
  IntVector hh(iter.end());
  for(int z=ll.z();z<hh.z();z++){
    for(int y=ll.y();y<hh.y();y++){
      IntVector rowstart(ll.x(),y,z);
      double* ppr = &r[rowstart]-ll.x();
      const double* ppa = &a[rowstart]-ll.x();
      const double* ppb = &b[rowstart]-ll.x();
      for(int x=ll.x();x<hh.x();x++){
	ppr[x]=s*ppa[x]+ppb[x];
      }
    }
  }
#else
  for(; !iter.done(); ++iter)
    r[*iter] = s*a[*iter]+b[*iter];
#endif
  IntVector diff = iter.end()-iter.begin();
  flops += 2*diff.x()*diff.y()*diff.z();
  memrefs += diff.x()*diff.y()*diff.z()*3L*8L;
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
    diag_label = VarLabel::create(A->getName()+" inverse diagonal", sol_type::getTypeDescription());
    VarLabel* tmp_flop_label = VarLabel::create(A->getName()+" flops", sumlong_vartype::getTypeDescription());
    tmp_flop_label->allowMultipleComputes();
    flop_label = tmp_flop_label;
    VarLabel* tmp_memref_label = VarLabel::create(A->getName()+" memrefs", sumlong_vartype::getTypeDescription());
    tmp_memref_label->allowMultipleComputes();
    memref_label = tmp_memref_label;
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
    VarLabel::destroy(diag_label);
    VarLabel::destroy(flop_label);
    VarLabel::destroy(memref_label);
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
	IntVector ec = params->getSolveOnExtraCells() ?
	  IntVector(0,0,0) : -level->getExtraCells();
	IntVector l = patch->getLowIndex(basis, ec);
	IntVector h = patch->getHighIndex(basis, ec);
	CellIterator iter(l, h);

	IntVector ll(l);
	IntVector hh(h);
	ll -= IntVector(patch->getBCType(Patch::xminus) == Patch::Neighbor?1:0,
			patch->getBCType(Patch::yminus) == Patch::Neighbor?1:0,
			patch->getBCType(Patch::zminus) == Patch::Neighbor?1:0);

	hh += IntVector(patch->getBCType(Patch::xplus) == Patch::Neighbor?1:0,
			patch->getBCType(Patch::yplus) == Patch::Neighbor?1:0,
			patch->getBCType(Patch::zplus) == Patch::Neighbor?1:0);
	hh -= IntVector(1,1,1);

	// Q = A*D
	long64 flops = 0;
	long64 memrefs = 0;
	// Must be qualified with :: for the IBM xlC compiler.
	double aden;
	::Mult(Q, A, D, iter, ll, hh, flops, memrefs, aden);
	new_dw->put(sum_vartype(aden), aden_label);

	new_dw->put(sumlong_vartype(flops), flop_label);
	new_dw->put(sumlong_vartype(memrefs), memref_label);
      }
    }
  }

  void step2(const ProcessorGroup*, const PatchSubset* patches,
	     const MaterialSubset* matls,
	     DataWarehouse* old_dw, DataWarehouse* new_dw)
  {
    for(int p=0;p<patches->size();p++){
      const Patch* patch = patches->get(p);
      if(cout_doing.active())
	cout_doing << "CGSolver::step2 on patch" << patch->getID()<<endl;
      for(int m = 0;m<matls->size();m++){
	int matl = matls->get(m);
	typedef typename Types::sol_type sol_type;
	Patch::VariableBasis basis = Patch::translateTypeToBasis(sol_type::getTypeDescription()->getType(), true);
	IntVector ec = params->getSolveOnExtraCells() ?
	  IntVector(0,0,0) : -level->getExtraCells();
	IntVector l = patch->getLowIndex(basis, ec);
	IntVector h = patch->getHighIndex(basis, ec);
	CellIterator iter(l, h);

	// Step 2 - requires d(old), aden(new) D(old), X(old) R(old)  computes X, R, Q, d
	typename Types::const_type D;
	old_dw->get(D, D_label, matl, patch, Ghost::None, 0);

	typename Types::const_type X, R, diagonal;
	old_dw->get(X, X_label, matl, patch, Ghost::None, 0);
	old_dw->get(R, R_label, matl, patch, Ghost::None, 0);
	old_dw->get(diagonal, diag_label, matl, patch, Ghost::None, 0);
	typename Types::sol_type Xnew, Rnew;
	new_dw->allocateAndPut(Xnew, X_label, matl, patch, Ghost::None, 0);
	new_dw->allocateAndPut(Rnew, R_label, matl, patch, Ghost::None, 0);
	typename Types::sol_type Q;
	new_dw->getModifiable(Q, Q_label, matl, patch);

	sum_vartype aden;
	new_dw->get(aden, aden_label);

	sum_vartype d;
	old_dw->get(d, d_label);

	long64 flops = 0;
	long64 memrefs = 0;
	double a=d/aden;

#if OPT
	IntVector  ll(iter.begin());
	IntVector hh(iter.end());
	double dnew = 0;
	double err = 0;
	for(int z=ll.z();z<hh.z();z++){
	  for(int y=ll.y();y<hh.y();y++){
	    IntVector rowstart(ll.x(),y,z);
	    double* ppXnew = &Xnew[rowstart]-ll.x();
	    const double* ppD = &D[rowstart]-ll.x();
	    double* ppRnew = &Rnew[rowstart]-ll.x();
	    const double* ppR = &R[rowstart]-ll.x();
	    double* ppQ = &Q[rowstart]-ll.x();
	    const double* ppX = &X[rowstart]-ll.x();
	    const double* ppdiagonal = &diagonal[rowstart]-ll.x();
	    for(int x=ll.x();x<hh.x();x++){
	      // X = a*D+X
	      ppXnew[x] = a*ppD[x]+ppX[x];
	      // R = -a*Q+R
	      double tmp1 = ppRnew[x] = ppR[x]-a*ppQ[x];

	      // Simple Preconditioning...
	      double tmp2 = ppQ[x] = tmp1*ppdiagonal[x];

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
	memrefs += 7L*diff.x()*diff.y()*diff.z()*8L;
#else
	// X = a*D+X
	::ScMult_Add(Xnew, a, D, X, iter, flops, memrefs);
	// R = -a*Q+R
	::ScMult_Add(Rnew, -a, Q, R, iter, flops, memrefs);

	// Simple Preconditioning...
	::Mult(Q, Rnew, diagonal, iter, flops, memrefs);

	// Calculate coefficient bk and direction vectors p and pp
	double dnew = ::Dot(Q, Rnew, iter, flops, memrefs);

	// Calculate error term
	switch(params->norm){
	case CGSolverParams::L1:
	  {
	    double err = ::L1(Q, iter, flops, memrefs);
	    new_dw->put(sum_vartype(err), err_label);
	  }
	  break;
	case CGSolverParams::L2:
	  // Nothing...
	  break;
	case CGSolverParams::LInfinity:
	  {
	    double err = ::LInf(Q, iter, flops, memrefs);
	    new_dw->put(max_vartype(err), err_label);
	  }
	  break;
	}
#endif
	new_dw->put(sum_vartype(dnew), d_label);
	new_dw->put(sumlong_vartype(flops), flop_label);
	new_dw->put(sumlong_vartype(memrefs), memref_label);
      }
    }
    new_dw->transferFrom(old_dw, diag_label, patches, matls);
  }

  void step3(const ProcessorGroup*, const PatchSubset* patches,
	     const MaterialSubset* matls,
	     DataWarehouse* old_dw, DataWarehouse* new_dw)
  {
    for(int p=0;p<patches->size();p++){
      const Patch* patch = patches->get(p);
      if(cout_doing.active())
	cout_doing << "CGSolver::step3 on patch" << patch->getID()<<endl;
      for(int m = 0;m<matls->size();m++){
	int matl = matls->get(m);
	typedef typename Types::sol_type sol_type;
	Patch::VariableBasis basis = Patch::translateTypeToBasis(sol_type::getTypeDescription()->getType(), true);
	IntVector ec = params->getSolveOnExtraCells() ?
	  IntVector(0,0,0) : -level->getExtraCells();
	IntVector l = patch->getLowIndex(basis, ec);
	IntVector h = patch->getHighIndex(basis, ec);
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
	long64 memrefs = 0;
	::ScMult_Add(Dnew, b, D, Q, iter, flops, memrefs);
	new_dw->put(sumlong_vartype(flops), flop_label);
	new_dw->put(sumlong_vartype(memrefs), memref_label);
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
      if(cout_doing.active())
	cout_doing << "CGSolver::setup on patch " << patch->getID()<< endl;
      for(int m = 0;m<matls->size();m++){
	int matl = matls->get(m);
	typedef typename Types::sol_type sol_type;
	Patch::VariableBasis basis = Patch::translateTypeToBasis(sol_type::getTypeDescription()->getType(), true);
	IntVector ec = params->getSolveOnExtraCells() ?
	  IntVector(0,0,0) : -level->getExtraCells();
	IntVector l = patch->getLowIndex(basis, ec);
	IntVector h = patch->getHighIndex(basis, ec);
	CellIterator iter(l, h);

	typename Types::sol_type R, Xnew, diagonal;
	new_dw->allocateAndPut(R, R_label, matl, patch);
	new_dw->allocateAndPut(Xnew, X_label, matl, patch);
	new_dw->allocateAndPut(diagonal, diag_label, matl, patch);
	typename Types::const_type B;
       typename Types::matrix_type A;
	parent_new_dw->get(B, B_label, matl, patch, Ghost::None, 0);
       parent_new_dw->get(A, A_label, matl, patch, Ghost::None, 0);

	long64 flops = 0;
	long64 memrefs = 0;
	if(guess_label){
	  typename Types::const_type X;
	  if(guess_dw == Task::OldDW)
	    parent_old_dw->get(X, guess_label, matl, patch, Around, 1);
	  else
	    parent_new_dw->get(X, guess_label, matl, patch, Around, 1);

	  // R = A*X
	  IntVector ll(l);
	  IntVector hh(h);
	  ll -= IntVector(patch->getBCType(Patch::xminus) == Patch::Neighbor?1:0,
			  patch->getBCType(Patch::yminus) == Patch::Neighbor?1:0,
			  patch->getBCType(Patch::zminus) == Patch::Neighbor?1:0);

	  hh += IntVector(patch->getBCType(Patch::xplus) == Patch::Neighbor?1:0,
			  patch->getBCType(Patch::yplus) == Patch::Neighbor?1:0,
			  patch->getBCType(Patch::zplus) == Patch::Neighbor?1:0);
	  hh -= IntVector(1,1,1);

	  ::Mult(R, A, X, iter, ll, hh, flops, memrefs);

	  // R = B-R
	  ::Sub(R, B, R, iter, flops, memrefs);
	  Xnew.copy(X, iter.begin(), iter.end());
	} else {
	  R.copy(B);
	  Xnew.initialize(0);
	}

	// D = R/Ap
	typename Types::sol_type D;
	new_dw->allocateAndPut(D, D_label, matl, patch);
#if 0
	::DivDiagonal(D, R, A, iter, flops);
#else
	::InverseDiagonal(diagonal, A, iter, flops, memrefs);
	::Mult(D, R, diagonal, iter, flops, memrefs);
#endif

	double dnew = ::Dot(R, D, iter, flops, memrefs);
	new_dw->put(sum_vartype(dnew), d_label);
	switch(params->norm){
	case CGSolverParams::L1:
	  {
	    double err = ::L1(R, iter, flops, memrefs);
	    new_dw->put(sum_vartype(err), err_label);
	  }
	  break;
	case CGSolverParams::L2:
	  // Nothing...
	  break;
	case CGSolverParams::LInfinity:
	  {
	    double err = ::LInf(R, iter, flops, memrefs);
	    new_dw->put(max_vartype(err), err_label);
	  }
	  break;
	}
	new_dw->put(sumlong_vartype(flops), flop_label);
	new_dw->put(sumlong_vartype(memrefs), memref_label);
      }
    }
  }

  //______________________________________________________________________
  void solve(const ProcessorGroup* pg, const PatchSubset* patches,
	     const MaterialSubset* matls,
	     DataWarehouse* old_dw, DataWarehouse* new_dw,
	     Handle<CGStencil7<Types> >)
  {
    if(cout_doing.active())
      cout_doing << "CGSolver::solve" << endl;
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

    //__________________________________
    // Schedule the setup
    if(cout_doing.active())
      cout_doing << "CGSolver::schedule setup" << endl;
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
    task->computes(diag_label);
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
    subsched->get_dw(3)->get(f, memref_label);
    long64 memrefs = f;
    if(!(e < params->initial_tolerance)) {
      subsched->initialize(3, 1, old_dw, new_dw);
      subsched->clearMappings();
      subsched->mapDataWarehouse(Task::ParentOldDW, 0);
      subsched->mapDataWarehouse(Task::ParentNewDW, 1);
      subsched->mapDataWarehouse(Task::OldDW, 2);
      subsched->mapDataWarehouse(Task::NewDW, 3);
  
      //__________________________________
      // Step 1 - requires A(parent), D(old, 1 ghost) computes aden(new)
      if(cout_doing.active())
	cout_doing << "CGSolver::schedule Step 1" << endl;
      task = scinew Task("CGSolve step1", this, &CGStencil7<Types>::step1);
      task->requires(Task::ParentNewDW, A_label, Ghost::None, 0);
      task->requires(Task::OldDW,       D_label, Around, 1);
      task->computes(aden_label);
      task->computes(Q_label);
      task->computes(flop_label);
      subsched->addTask(task, level->eachPatch(), matlset);

      //__________________________________
      // schedule
      // Step 2 - requires d(old), aden(new) D(old), X(old) R(old)  computes X, R, Q, d
      if(cout_doing.active())
	cout_doing << "CGSolver::schedule Step 2" << endl;
      task = scinew Task("CGSolve step2", this, &CGStencil7<Types>::step2);
      task->requires(Task::OldDW, d_label);
      task->requires(Task::NewDW, aden_label);
      task->requires(Task::OldDW, D_label,    Ghost::None, 0);
      task->requires(Task::OldDW, X_label,    Ghost::None, 0);
      task->requires(Task::OldDW, R_label,    Ghost::None, 0);
      task->requires(Task::OldDW, diag_label, Ghost::None, 0);
      task->computes(X_label);
      task->computes(R_label);
      task->modifies(Q_label);
      task->computes(d_label);
      task->computes(diag_label);
      task->computes(flop_label);
      if(params->norm != CGSolverParams::L1) {
	task->computes(err_label);
      }
      subsched->addTask(task, level->eachPatch(), matlset);

      
      //__________________________________
      // schedule
      // Step 3 - requires D(old), Q(new), d(new), d(old), computes D
      if(cout_doing.active())
	cout_doing << "CGSolver::schedule Step 2" << endl;
      task = scinew Task("CGSolve step3", this, &CGStencil7<Types>::step3);
      task->requires(Task::OldDW, D_label, Ghost::None, 0);
      task->requires(Task::NewDW, Q_label, Ghost::None, 0);
      task->requires(Task::NewDW, d_label);
      task->requires(Task::OldDW, d_label);
      task->computes(D_label);
      task->computes(flop_label);
      subsched->addTask(task, level->eachPatch(), matlset);
      subsched->compile(world);
      
      //__________________________________
      //  Main iteration
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
	subsched->get_dw(3)->get(f, memref_label);
	memrefs += f;
      }
    }

    //__________________________________
    //  Pull the solution out of subsched new DW and put it in ours
    if(modifies_x){
      for(int p=0;p<patches->size();p++){
	const Patch* patch = patches->get(p);
	for(int m = 0;m<matls->size();m++){
	  int matl = matls->get(m);
	  typedef typename Types::sol_type sol_type;
	  Patch::VariableBasis basis = Patch::translateTypeToBasis(sol_type::getTypeDescription()->getType(), true);
	  IntVector ec = params->getSolveOnExtraCells() ?
	    IntVector(0,0,0) : -level->getExtraCells();
	  IntVector l = patch->getLowIndex(basis, ec);
	  IntVector h = patch->getHighIndex(basis, ec);
	  CellIterator iter(l, h);

	  typename Types::sol_type Xnew;
         typename Types::const_type X;
	  new_dw->getModifiable(Xnew, X_label, matl, patch);
	 
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
    double memrate = (double(memrefs)*1.e-9)/dt;
    if(pg->myrank() == 0){
      if(niter < toomany) {
        cerr << "Solve of " << X_label->getName() 
	      << " on level " << level->getIndex()
             << " completed in "
             << niter << " iterations, " << dt << " seconds (" 
	      << mflops<< " MFLOPS, " << memrate << " GB/sec)\n";
      }else{
        throw ConvergenceFailure("CGSolve variable: "+X_label->getName(), 
			  niter, e, params->tolerance);
      }
    }
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
  const VarLabel* diag_label;
  const VarLabel* Q_label;
  const VarLabel* d_label;
  const VarLabel* err_label;
  const VarLabel* aden_label;
  const VarLabel* guess_label;
  const VarLabel* flop_label;
  const VarLabel* memref_label;
  Task::WhichDW guess_dw;
  const CGSolverParams* params;
  bool modifies_x;
};

SolverParameters* CGSolver::readParameters(ProblemSpecP& params, const string& varname)
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
    task->modifies(x);
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
