/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <CCA/Components/Solvers/DirectSolve.h>
#include <Core/Grid/Level.h>
#include <CCA/Components/Solvers/MatrixUtil.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/Stencil7.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Containers/Array1.h>
#include <Core/Containers/Array2.h>
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
                 const VarLabel* A, Task::WhichDW which_A_dw,
                 const VarLabel* x, bool modifies_x,
                 const VarLabel* b, Task::WhichDW which_b_dw,
                 const DirectSolveParams* params)
    : level(level), matlset(matlset),
      A_label(A), which_A_dw(which_A_dw),
      X_label(x), 
      B_label(b), which_b_dw(which_b_dw),
      modifies_x(modifies_x),
      params(params)
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
    DataWarehouse* A_dw = new_dw->getOtherDataWarehouse(which_A_dw);
    DataWarehouse* b_dw = new_dw->getOtherDataWarehouse(which_b_dw);
    
    double tstart = Time::currentSeconds();
    long64 flops = 0, memrefs = 0;
    for(int m = 0;m<matls->size();m++){
      int matl = matls->get(m);
      typedef typename Types::sol_type sol_type;
      cout_doing << "DirectSolve on matl " << matl << endl;
      ASSERTEQ(patches->size(), 1);
      const Patch* patch = patches->get(0);
      typename Types::const_type B;
      b_dw->get(B, B_label, matl, patch, Ghost::None, 0);
      typename Types::matrix_type A;
      A_dw->get(A, A_label, matl, patch, Ghost::None, 0);

      typename Types::sol_type X;
      if(modifies_x)
        new_dw->getModifiable(X, X_label, matl, patch);
      else
        new_dw->allocateAndPut(X, X_label, matl, patch);

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
      cout << "Solve of " << X_label->getName() 
	   << " on level " << level->getIndex()
           << " completed in " << dt << " seconds (" 
	   << mflops << " MFLOPS, " << memrate << " GB/sec)\n";
    }
  }
    
private:
  const Level* level;
  const MaterialSet* matlset;
  const VarLabel* A_label;
  Task::WhichDW which_A_dw;
  const VarLabel* X_label;
  const VarLabel* B_label;
  Task::WhichDW which_b_dw;
  bool modifies_x;
  const DirectSolveParams* params;
};

SolverParameters* DirectSolve::readParameters(ProblemSpecP& params, 
                                              const string& varname,
                                              SimulationStateP& state)
{
  DirectSolveParams* p = scinew DirectSolveParams();
  return p;
}


SolverParameters* DirectSolve::readParameters(ProblemSpecP& params, 
                                              const string& varname)
{
  DirectSolveParams* p = scinew DirectSolveParams();
  return p;
}

void DirectSolve::scheduleSolve(const LevelP& level, SchedulerP& sched,
                                const MaterialSet* matls,
                                const VarLabel* A,    
                                Task::WhichDW which_A_dw,  
                                const VarLabel* x,
                                bool modifies_x,
                                const VarLabel* b,    
                                Task::WhichDW which_b_dw,  
                                const VarLabel* guess,Task::WhichDW guess_dw,
                                const SolverParameters* params,
                                bool modifies_hypre)
{
  if(level->numPatches() != 1)
    throw InternalError("DirectSolve only works with 1 patch", __FILE__, __LINE__);

  Task* task;
  // The extra handle arg ensures that the stencil7 object will get freed
  // when the task gets freed.  The downside is that the refcount gets
  // tweaked everytime solve is called.

  TypeDescription::Type domtype = A->typeDescription()->getType();
  ASSERTEQ(domtype, x->typeDescription()->getType());
  ASSERTEQ(domtype, b->typeDescription()->getType());
  const DirectSolveParams* dparams = dynamic_cast<const DirectSolveParams*>(params);
  if(!dparams)
    throw InternalError("Wrong type of params passed to Direct solver!", __FILE__, __LINE__);

  switch(domtype){
  case TypeDescription::SFCXVariable:
    {
      DirectStencil7<SFCXTypes>* that = scinew DirectStencil7<SFCXTypes>(level.get_rep(), matls, A, which_A_dw, x, modifies_x, b, which_b_dw, dparams);
      Handle<DirectStencil7<SFCXTypes> > handle = that;
      task = scinew Task("DirectSolve::Matrix solve (SFCX)", that, &DirectStencil7<SFCXTypes>::solve, handle);
    }
    break;
  case TypeDescription::SFCYVariable:
    {
      DirectStencil7<SFCYTypes>* that = scinew DirectStencil7<SFCYTypes>(level.get_rep(), matls, A, which_A_dw, x, modifies_x, b, which_b_dw, dparams);
      Handle<DirectStencil7<SFCYTypes> > handle = that;
      task = scinew Task("DirectSolve::Matrix solve (SFCY)", that, &DirectStencil7<SFCYTypes>::solve, handle);
    }
    break;
  case TypeDescription::SFCZVariable:
    {
      DirectStencil7<SFCZTypes>* that = scinew DirectStencil7<SFCZTypes>(level.get_rep(), matls, A, which_A_dw, x, modifies_x, b, which_b_dw, dparams);
      Handle<DirectStencil7<SFCZTypes> > handle = that;
      task = scinew Task("DirectSolve::Matrix solve (SFCZ)", that, &DirectStencil7<SFCZTypes>::solve, handle);
    }
    break;
  case TypeDescription::CCVariable:
    {
      DirectStencil7<CCTypes>* that = scinew DirectStencil7<CCTypes>(level.get_rep(), matls, A, which_A_dw, x, modifies_x, b, which_b_dw, dparams);
      Handle<DirectStencil7<CCTypes> > handle = that;
      task = scinew Task("DirectSolve::Matrix solve (CC)", that, &DirectStencil7<CCTypes>::solve, handle);
    }
    break;
  case TypeDescription::NCVariable:
    {
      DirectStencil7<NCTypes>* that = scinew DirectStencil7<NCTypes>(level.get_rep(), matls, A, which_A_dw, x, modifies_x, b, which_b_dw, dparams);
      Handle<DirectStencil7<NCTypes> > handle = that;
      task = scinew Task("DirectSolve::Matrix solve (NC)", that, &DirectStencil7<NCTypes>::solve, handle);
    }
    break;
  default:
    throw InternalError("Unknown variable type in scheduleSolve", __FILE__, __LINE__);
  }

  task->requires(which_A_dw, A, Ghost::None, 0);
  if(modifies_x)
    task->modifies(x);
  else
    task->computes(x);

  task->requires(which_b_dw, b, Ghost::None, 0);
  sched->addTask(task, level->eachPatch(), matls);
}

string DirectSolve::getName(){
  return "DirectSolve";
}

} // end namespace Uintah
