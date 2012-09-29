/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <CCA/Components/Solvers/CGSolver.h>
#include <CCA/Components/Solvers/MatrixUtil.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/Stencil7.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Exceptions/ConvergenceFailure.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Ports/LoadBalancer.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Math/MiscMath.h>
#include <Core/Math/MinMax.h>
#include <Core/Thread/Time.h>
#include <Core/Util/DebugStream.h>
#include <iomanip>

using std::cout;
using namespace Uintah;
//__________________________________
//  To turn on normal output
//  setenv SCI_DEBUG "CGSOLVER_DOING_COUT:+"

static DebugStream cout_doing("CGSOLVER_DOING_COUT", false);

void Mult(Array3<double>& B, const Array3<Stencil7>& A,
          const Array3<double>& X, CellIterator iter,
          const IntVector& l, const IntVector& h1, long64& flops, long64& memrefs)
{
  if(cout_doing.active())
    cout_doing << "CGSolver::Mult" << endl;
  
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

  for(; !iter.done(); ++iter){
    r[*iter] = s*a[*iter]+b[*iter];
  }
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
  int maxiterations;
  
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
             const VarLabel* A, Task::WhichDW which_A_dw,
             const VarLabel* x, bool modifies_x,
             const VarLabel* b, Task::WhichDW which_b_dw,
             const VarLabel* guess, Task::WhichDW which_guess_dw,
             const CGSolverParams* params)
    : sched(sched), world(world), level(level), matlset(matlset),
      Around(Around), A_label(A), which_A_dw(which_A_dw), X_label(x),
      B_label(b), which_b_dw(which_b_dw),
      guess_label(guess), which_guess_dw(which_guess_dw), params(params),
      modifies_x(modifies_x)
  {
    switch(which_A_dw){
    case Task::OldDW:
      parent_which_A_dw = Task::ParentOldDW;
      break;
    case Task::NewDW:
      parent_which_A_dw = Task::ParentNewDW;
      break;
    default:
      throw ProblemSetupException("Unknown data warehouse for A matrix", __FILE__, __LINE__);
    }
    switch(which_b_dw){
    case Task::OldDW:
      parent_which_b_dw = Task::ParentOldDW;
      break;
    case Task::NewDW:
      parent_which_b_dw = Task::ParentNewDW;
      break;
    default:
      throw ProblemSetupException("Unknown data warehouse for b rhs", __FILE__, __LINE__);
    }
    switch(which_guess_dw){
    case Task::OldDW:
      parent_which_guess_dw = Task::ParentOldDW;
      break;
    case Task::NewDW:
      parent_which_guess_dw = Task::ParentNewDW;
      break;
    default:
      throw ProblemSetupException("Unknown data warehouse for initial guess", __FILE__, __LINE__);
    }
    typedef typename Types::sol_type sol_type;
    R_label     = VarLabel::create(A->getName()+" R", sol_type::getTypeDescription());
    D_label     = VarLabel::create(A->getName()+" D", sol_type::getTypeDescription());
    Q_label     = VarLabel::create(A->getName()+" Q", sol_type::getTypeDescription());
    d_label     = VarLabel::create(A->getName()+" d", sum_vartype::getTypeDescription());
    aden_label = VarLabel::create(A->getName()+" aden", sum_vartype::getTypeDescription());
    diag_label = VarLabel::create(A->getName()+" inverse diagonal", sol_type::getTypeDescription());
    
    tolerance_label = VarLabel::create("tolerance", sum_vartype::getTypeDescription());
    
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
    VarLabel::destroy(tolerance_label);
    
    if(err_label != d_label)
      VarLabel::destroy(err_label);
    VarLabel::destroy(aden_label);
  }
//______________________________________________________________________
//
  void step1(const ProcessorGroup*, const PatchSubset* patches,
             const MaterialSubset* matls,
             DataWarehouse* old_dw, DataWarehouse* new_dw)
  {
    DataWarehouse* A_dw = new_dw->getOtherDataWarehouse(parent_which_A_dw);
    // Step 1 - requires A(parent), D(old, 1 ghost) computes aden(new)
    for(int p=0;p<patches->size();p++){
      const Patch* patch = patches->get(p);
      for(int m = 0;m<matls->size();m++){
        int matl = matls->get(m);

        typename Types::sol_type Q;
        new_dw->allocateAndPut(Q, Q_label, matl, patch);

        typename Types::matrix_type A;
        A_dw->get(A, A_label, matl, patch, Ghost::None, 0);

        typename Types::const_type D;
        old_dw->get(D, D_label, matl, patch, Around, 1);

        typedef typename Types::sol_type sol_type;
        Patch::VariableBasis basis = Patch::translateTypeToBasis(sol_type::getTypeDescription()->getType(), true);

        IntVector l,h;
        if(params->getSolveOnExtraCells())
        {
          l = patch->getExtraLowIndex(basis, IntVector(0,0,0));
          h = patch->getExtraHighIndex(basis, IntVector(0,0,0));
        }
        else
        {
          l = patch->getLowIndex(basis);
          h = patch->getHighIndex(basis);
        }

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
//______________________________________________________________________
//
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

        IntVector l,h;
        if(params->getSolveOnExtraCells())
        {
          l = patch->getExtraLowIndex(basis, IntVector(0,0,0));
          h = patch->getExtraHighIndex(basis, IntVector(0,0,0));
        }
        else
        {
          l = patch->getLowIndex(basis);
          h = patch->getHighIndex(basis);
        }
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
        new_dw->put(sum_vartype(dnew), d_label);
        new_dw->put(sumlong_vartype(flops), flop_label);
        new_dw->put(sumlong_vartype(memrefs), memref_label);
      }
    }
    new_dw->transferFrom(old_dw, diag_label, patches, matls);
  }
//______________________________________________________________________
//
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

        IntVector l,h;
        if(params->getSolveOnExtraCells())
        {
          l = patch->getExtraLowIndex(basis, IntVector(0,0,0));
          h = patch->getExtraHighIndex(basis, IntVector(0,0,0));
        }
        else
        {
          l = patch->getLowIndex(basis);
          h = patch->getHighIndex(basis);
        }
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
//______________________________________________________________________
  void setup(const ProcessorGroup*, const PatchSubset* patches,
             const MaterialSubset* matls,
             DataWarehouse*, DataWarehouse* new_dw)
  {
    DataWarehouse* A_dw = new_dw->getOtherDataWarehouse(parent_which_A_dw);
    DataWarehouse* b_dw = new_dw->getOtherDataWarehouse(parent_which_b_dw);
    DataWarehouse* guess_dw = new_dw->getOtherDataWarehouse(parent_which_guess_dw);
    for(int p=0;p<patches->size();p++){
      const Patch* patch = patches->get(p);
      if(cout_doing.active())
        cout_doing << "CGSolver::setup on patch " << patch->getID()<< endl;
        
      for(int m = 0;m<matls->size();m++){
        int matl = matls->get(m);
        typedef typename Types::sol_type sol_type;
        Patch::VariableBasis basis = Patch::translateTypeToBasis(sol_type::getTypeDescription()->getType(), true);

        IntVector l,h;
        if(params->getSolveOnExtraCells())
        {
          l = patch->getExtraLowIndex(basis, IntVector(0,0,0));
          h = patch->getExtraHighIndex(basis, IntVector(0,0,0));
        }
        else
        {
          l = patch->getLowIndex(basis);
          h = patch->getHighIndex(basis);
        }
        CellIterator iter(l, h);

        typename Types::sol_type R, Xnew, diagonal;
        new_dw->allocateAndPut(R, R_label, matl, patch);
        new_dw->allocateAndPut(Xnew, X_label, matl, patch);
        new_dw->allocateAndPut(diagonal, diag_label, matl, patch);
        typename Types::const_type B;
        typename Types::matrix_type A;
        b_dw->get(B, B_label, matl, patch, Ghost::None, 0);
        A_dw->get(A, A_label, matl, patch, Ghost::None, 0);

        long64 flops = 0;
        long64 memrefs = 0;
        if(guess_label){
          typename Types::const_type X;
          guess_dw->get(X, guess_label, matl, patch, Around, 1);

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

        ::InverseDiagonal(diagonal, A, iter, flops, memrefs);
        ::Mult(D, R, diagonal, iter, flops, memrefs);

        double dnew = ::Dot(R, D, iter, flops, memrefs);
        new_dw->put(sum_vartype(dnew), d_label);
        new_dw->put( sum_vartype(params->tolerance), tolerance_label );
        
        
        // Calculate error term
        double residualNormalization = params->getResidualNormalizationFactor();       
        
        switch(params->norm){
        case CGSolverParams::L1:
          {
            double err = ::L1(R, iter, flops, memrefs);
            err /= residualNormalization;
            new_dw->put(sum_vartype(err), err_label);
          }
          break;
        case CGSolverParams::L2:
          // Nothing...
          break;
        case CGSolverParams::LInfinity:
          {
            double err = ::LInf(R, iter, flops, memrefs);
            err /= residualNormalization;
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
    subsched->initialize(3, 1);
    subsched->setParentDWs(old_dw, new_dw);
    subsched->clearMappings();
    subsched->mapDataWarehouse(Task::ParentOldDW, 0);
    subsched->mapDataWarehouse(Task::ParentNewDW, 1);
    subsched->mapDataWarehouse(Task::OldDW, 2);
    subsched->mapDataWarehouse(Task::NewDW, 3);
  
    GridP grid = level->getGrid();
    IntVector l, h;
    level->findCellIndexRange(l, h);

    int niter=0;

    subsched->advanceDataWarehouse(grid);

    //__________________________________
    // Schedule the setup
    if(cout_doing.active())
      cout_doing << "CGSolver::schedule setup" << endl;
    Task* task = scinew Task("CGSolver: schedule setup", this, &CGStencil7<Types>::setup);
    task->requires(parent_which_b_dw, B_label, Ghost::None, 0);
    task->requires(parent_which_A_dw, A_label, Ghost::None, 0);
    if(guess_label)
      task->requires(parent_which_guess_dw, guess_label, Around, 1);

    task->computes(memref_label);
    task->computes(R_label);
    task->computes(X_label);
    task->computes(D_label);
    task->computes(d_label);
    task->computes(tolerance_label);
    task->computes(diag_label);
    if(params->norm != CGSolverParams::L2){
      task->computes(err_label);
    }
    task->computes(flop_label);
    subsched->addTask(task, level->eachPatch(), matlset);

    subsched->compile();
    subsched->get_dw(3)->setScrubbing(DataWarehouse::ScrubNone);
    subsched->execute();    
    
    //__________________________________
    double tolerance=0;
    sum_vartype tol;
    subsched->get_dw(3)->get(tol, tolerance_label);
    tolerance=tol;
    
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

    //__________________________________
    if(!(e < params->initial_tolerance)) {
      subsched->initialize(3, 1);
      subsched->setParentDWs(old_dw, new_dw);
      subsched->clearMappings();
      subsched->mapDataWarehouse(Task::ParentOldDW, 0);
      subsched->mapDataWarehouse(Task::ParentNewDW, 1);
      subsched->mapDataWarehouse(Task::OldDW, 2);
      subsched->mapDataWarehouse(Task::NewDW, 3);
  
      //__________________________________
      // Step 1 - requires A(parent), D(old, 1 ghost) computes aden(new)
      if(cout_doing.active())
        cout_doing << "CGSolver::schedule Step 1" << endl;
      task = scinew Task("CGSolver: schedule step1", this, &CGStencil7<Types>::step1);
      task->requires(parent_which_A_dw, A_label, Ghost::None, 0);
      task->requires(Task::OldDW,       D_label, Around, 1);
      task->computes(aden_label);
      task->computes(Q_label);
      task->computes(flop_label);
      task->computes(memref_label);
      subsched->addTask(task, level->eachPatch(), matlset);

      //__________________________________
      // schedule
      // Step 2 - requires d(old), aden(new) D(old), X(old) R(old)  computes X, R, Q, d
      if(cout_doing.active())
        cout_doing << "CGSolver::schedule Step 2" << endl;
      task = scinew Task("CGSolver: schedule step2", this, &CGStencil7<Types>::step2);
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
      task->modifies(memref_label);
      if(params->norm != CGSolverParams::L1) {
        task->computes(err_label);
      }
      subsched->addTask(task, level->eachPatch(), matlset);

      
      //__________________________________
      // schedule
      // Step 3 - requires D(old), Q(new), d(new), d(old), computes D
      if(cout_doing.active())
        cout_doing << "CGSolver::schedule Step 3" << endl;
      task = scinew Task("CGSolver: schedule step3", this, &CGStencil7<Types>::step3);
      task->requires(Task::OldDW, D_label, Ghost::None, 0);
      task->requires(Task::NewDW, Q_label, Ghost::None, 0);
      task->requires(Task::NewDW, d_label);
      task->requires(Task::OldDW, d_label);
      task->computes(D_label);
      task->computes(flop_label);
      task->modifies(memref_label);
      subsched->addTask(task, level->eachPatch(), matlset);
      subsched->compile();
      
      //__________________________________
      //  Main iteration
      while(niter < params->maxiterations && !(e < tolerance)){
        niter++;
        subsched->advanceDataWarehouse(grid);
        subsched->get_dw(2)->setScrubbing(DataWarehouse::ScrubComplete);
        subsched->get_dw(3)->setScrubbing(DataWarehouse::ScrubNonPermanent);
        subsched->execute();
        
        //__________________________________
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
        if(params->criteria == CGSolverParams::Relative){
          e/=err0;
        }
        sumlong_vartype f;
        subsched->get_dw(3)->get(f, flop_label);
        flops += f;
        subsched->get_dw(3)->get(f, memref_label);
        memrefs += f;
      }
    }

    //__________________________________
    //  Pull the solution out of subsched new DW and put it into our X
    if(modifies_x){
      for(int p=0;p<patches->size();p++){
        const Patch* patch = patches->get(p);
        for(int m = 0;m<matls->size();m++){
          int matl = matls->get(m);
          typedef typename Types::sol_type sol_type;
          Patch::VariableBasis basis = Patch::translateTypeToBasis(sol_type::getTypeDescription()->getType(), true);

          IntVector l,h;
          if(params->getSolveOnExtraCells())
          {
            l = patch->getExtraLowIndex(basis, IntVector(0,0,0));
            h = patch->getExtraHighIndex(basis, IntVector(0,0,0));
          }
          else
          {
            l = patch->getLowIndex(basis);
            h = patch->getHighIndex(basis);
          }
          CellIterator iter(l, h);

          typename Types::sol_type Xnew;
          typename Types::const_type X;
          new_dw->getModifiable(Xnew, X_label, matl, patch);

          subsched->get_dw(3)->get(X, X_label, matl, patch, Ghost::None, 0);
          Xnew.copy(X, l, h);
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
      if(niter < params->maxiterations) {
        cout << "Solve of " << X_label->getName() 
              << " on level " << level->getIndex()
             << " completed in "
             << dt << " seconds ("
             << niter << " iterations, "
             << e << " residual, " 
              << mflops<< " MFLOPS, " << memrate << " GB/sec)\n";
      }else{
        if(params->getRestartTimestepOnFailure()){
           cout << "CGSolver not converging, requesting smaller timestep\n";
           cout << "    niters:   " << niter << "\n"
                << "    residual: " << e << endl;
          new_dw->abortTimestep();
          new_dw->restartTimestep();
        }else {
        throw ConvergenceFailure("CGSolve variable: "+X_label->getName(), 
                          niter, e, tolerance,__FILE__,__LINE__);
        }
      }
    }
  }
//______________________________________________________________________
//    
private:
  Scheduler* sched;
  const ProcessorGroup* world;
  const Level* level;
  const MaterialSet* matlset;
  Ghost::GhostType Around;
  const VarLabel* A_label;
  Task::WhichDW which_A_dw, parent_which_A_dw;
  const VarLabel* X_label;
  const VarLabel* B_label;
  Task::WhichDW which_b_dw, parent_which_b_dw;
  const VarLabel* guess_label;
  Task::WhichDW which_guess_dw, parent_which_guess_dw;

  const VarLabel* R_label;
  const VarLabel* D_label;
  const VarLabel* diag_label;
  const VarLabel* Q_label;
  const VarLabel* d_label;
  const VarLabel* err_label;
  const VarLabel* aden_label;
  const VarLabel* flop_label;
  const VarLabel* memref_label;
  const VarLabel* tolerance_label;

  const CGSolverParams* params;
  bool modifies_x;
};
//______________________________________________________________________
//
SolverParameters* CGSolver::readParameters(ProblemSpecP& params, 
                                           const string& varname,
                                           SimulationStateP& state)
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
      param->getWithDefault ("maxiterations",   p->maxiterations,  75);
      
      string norm;
      if(param->get("norm", norm)){
        if(norm == "L1" || norm == "l1") {
          p->norm = CGSolverParams::L1;
        } else if(norm == "L2" || norm == "l2") {
          p->norm = CGSolverParams::L2;
        } else if(norm == "LInfinity" || norm == "linfinity") {
          p->norm = CGSolverParams::LInfinity;
        } else {
          throw ProblemSetupException("Unknown norm type: "+norm, __FILE__, __LINE__);
        }
      }
      string criteria;
      if(param->get("criteria", criteria)){
        if(criteria == "Absolute" || criteria == "absolute") {
          p->criteria = CGSolverParams::Absolute;
        } else if(criteria == "Relative" || criteria == "relative") {
          p->criteria = CGSolverParams::Relative;
        } else {
          throw ProblemSetupException("Unknown criteria: "+criteria, __FILE__, __LINE__);
        }
      }
    }
  }
  
  if(p->norm == CGSolverParams::L2)
    p->tolerance *= p->tolerance;
  return p;
}

//______________________________________________________________________
//
void CGSolver::scheduleSolve(const LevelP& level, SchedulerP& sched,
                             const MaterialSet* matls,
                             const VarLabel* A,    Task::WhichDW which_A_dw,  
                             const VarLabel* x,
                             bool modifies_x,
                             const VarLabel* b,    Task::WhichDW which_b_dw,  
                             const VarLabel* guess,Task::WhichDW which_guess_dw,
                             const SolverParameters* params,
                             bool modifies_hypre)
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
    throw InternalError("Wrong type of params passed to cg solver!", __FILE__, __LINE__);

  Ghost::GhostType Around;

  switch(domtype){
  case TypeDescription::SFCXVariable:
    {
      Around = Ghost::AroundFaces;
      CGStencil7<SFCXTypes>* that = scinew CGStencil7<SFCXTypes>(sched.get_rep(), d_myworld, level.get_rep(), matls, Around, A, which_A_dw, x, modifies_x, b, which_b_dw, guess, which_guess_dw, cgparams);
      Handle<CGStencil7<SFCXTypes> > handle = that;
      task = scinew Task("CGSolver::Matrix solve(SFCX)", that, &CGStencil7<SFCXTypes>::solve, handle);
    }
    break;
  case TypeDescription::SFCYVariable:
    {
      Around = Ghost::AroundFaces;
      CGStencil7<SFCYTypes>* that = scinew CGStencil7<SFCYTypes>(sched.get_rep(), d_myworld, level.get_rep(), matls, Around, A, which_A_dw, x, modifies_x, b, which_b_dw, guess, which_guess_dw, cgparams);
      Handle<CGStencil7<SFCYTypes> > handle = that;
      task = scinew Task("CGSolver::Matrix solve(SFCY)", that, &CGStencil7<SFCYTypes>::solve, handle);
    }
    break;
  case TypeDescription::SFCZVariable:
    {
      Around = Ghost::AroundFaces;
      CGStencil7<SFCZTypes>* that = scinew CGStencil7<SFCZTypes>(sched.get_rep(), d_myworld, level.get_rep(), matls, Around, A, which_A_dw, x, modifies_x, b, which_b_dw, guess, which_guess_dw, cgparams);
      Handle<CGStencil7<SFCZTypes> > handle = that;
      task = scinew Task("CGSolver::Matrix solve(SFCZ)", that, &CGStencil7<SFCZTypes>::solve, handle);
    }
    break;
  case TypeDescription::CCVariable:
    {
      Around = Ghost::AroundCells;
      CGStencil7<CCTypes>* that = scinew CGStencil7<CCTypes>(sched.get_rep(), d_myworld, level.get_rep(), matls, Around, A, which_A_dw, x, modifies_x, b, which_b_dw, guess, which_guess_dw, cgparams);
      Handle<CGStencil7<CCTypes> > handle = that;
      task = scinew Task("CGSolver::Matrix solve(CC)", that, &CGStencil7<CCTypes>::solve, handle);
    }
    break;
  case TypeDescription::NCVariable:
    {
      Around = Ghost::AroundNodes;
      CGStencil7<NCTypes>* that = scinew CGStencil7<NCTypes>(sched.get_rep(), d_myworld, level.get_rep(), matls, Around, A, which_A_dw, x, modifies_x, b, which_b_dw, guess, which_guess_dw, cgparams);
      Handle<CGStencil7<NCTypes> > handle = that;
      task = scinew Task("CGSolver::Matrix solve(NC)", that, &CGStencil7<NCTypes>::solve, handle);
    }
    break;
  default:
    throw InternalError("Unknown variable type in scheduleSolve", __FILE__, __LINE__);
  }

  task->requires(which_A_dw, A, Ghost::None, 0);
  if(guess)
    task->requires(which_guess_dw, guess, Around, 1);
  if(modifies_x)
    task->modifies(x);
  else
    task->computes(x);

  task->requires(which_b_dw, b, Ghost::None, 0);
  task->hasSubScheduler();
  LoadBalancer* lb = sched->getLoadBalancer();
  const PatchSet* perproc_patches = lb->getPerProcessorPatchSet(level);
  sched->addTask(task, perproc_patches, matls);
}

string CGSolver::getName(){
  return "CGSolver";
}

} // end namespace Uintah
