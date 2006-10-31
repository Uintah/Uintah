/*
  For more information, please see: http://software.sci.utah.edu

  The MIT License

  Copyright (c) 2004 Scientific Computing and Imaging Institute,
  University of Utah.

  
  Permission is hereby granted, free of charge, to any person obtaining a
  copy of this software and associated documentation files (the "Software"),
  to deal in the Software without restriction, including without limitation
  the rights to use, copy, modify, merge, publish, distribute, sublicense,
  and/or sell copies of the Software, and to permit persons to whom the
  Software is furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included
  in all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
  DEALINGS IN THE SOFTWARE.
*/


/*
 *  SolveLinearSystem.cc:  Unfinished modules
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <sci_defs/petsc_defs.h>

#include <Dataflow/Network/Module.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/SparseRowMatrix.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Dataflow/Network/NetworkEditor.h>
#include <Dataflow/Network/Ports/MatrixPort.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Geometry/Point.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MiscMath.h>
#include <Dataflow/GuiInterface/GuiVar.h>
#include <Core/Thread/Parallel.h>
#include <Core/Thread/SimpleReducer.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Mutex.h>

#include <stdio.h>
#include <math.h>

#include <iostream>
#include <sstream>

#ifdef PETSC_UNI

extern "C" {
#include <petsc.h>
#include <petscsles.h>
}

/*

This class (and the global instance below it) is used solely to
initialize PETSc at SCIRun startup, and finalize at shutdown.
DO NOT use it for anything else.

*/

class Initializer
{
public:

  Initializer() :
    petsc_is_initialized(false)
  {
    int argc = 0;
    char **argv = 0;
    petsc_is_initialized = true;
    PetscInitialize(&argc,&argv,0,0);
  }

  ~Initializer()
  {
    PetscFinalize();
  }

  bool is_initialized() { return petsc_is_initialized; }

private:

  bool petsc_is_initialized;

};

Initializer Petsc;

#endif

/* **************************************************************** */


namespace SCIRun {

Mutex PetscLock("SolveLinearSystem PETSc lock");

struct PStats {
  int flop;
  int memref;
  int gflop;
  int grefs;
  int pad[28];
};

class SolveLinearSystem;

struct CGData {
  SolveLinearSystem* module;
  WallClockTimer* timer;
  ColumnMatrix* rhs;
  ColumnMatrix* lhs;
  Matrix* mat;
  ColumnMatrix* diag;
  int niter;
  int toomany;
  ColumnMatrix* Z;
  ColumnMatrix* R;
  ColumnMatrix* P;
  // BiCG
  ColumnMatrix* Z1;
  ColumnMatrix* R1;
  ColumnMatrix* P1;
  Matrix *trans;
  double max_error;
  SimpleReducer reducer;
  int np;
  PStats* stats;
  double err;
  double bnorm;
  CGData();
  void clear();
};

class SolveLinearSystem : public Module {
  MatrixHandle solution;

#ifdef PETSC_UNI
  friend int PETSc_monitor(KSP,int n,PetscReal rnorm,void *);
  vector<double> PETSc_errlist;
  vector<double> PETSc_targetlist;
  vector<int> PETSc_targetidx;
  double PETSc_log_orig;
  double PETSc_log_targ;
  double PETSc_max_err;
  int PETSc_last_update;
  int PETSc_last_errupdate;

  void petsc_solve(const char*, const char*,Matrix* ,
		   ColumnMatrix *, ColumnMatrix *);
#endif

  void jacobi_sci(Matrix*,ColumnMatrix& , ColumnMatrix&);
  void conjugate_gradient_sci(Matrix*,ColumnMatrix&, ColumnMatrix&);
  void bi_conjugate_gradient_sci(Matrix*,ColumnMatrix&, ColumnMatrix&);

  void append_values(int niter, const vector<double>& errlist,
		     int& last_update, const vector<int>& targetidx,
		     const vector<double>& targetlist,
		     int& last_errupdate);

  void set_compute_time_stats(PStats *stats, double time, int nprocs);

public:
  bool init_parallel_conjugate_gradient();
  void parallel_conjugate_gradient(int proc);
  bool init_parallel_bi_conjugate_gradient();
  void parallel_bi_conjugate_gradient(int proc);
  SolveLinearSystem(GuiContext* ctx);
  virtual ~SolveLinearSystem();
  virtual void execute();
  void tcl_command( GuiArgs&, void * );

  GuiDouble target_error;
  GuiDouble flops;
  GuiDouble floprate;
  GuiDouble memrefs;
  GuiDouble memrate;
  GuiDouble orig_error;
  GuiString current_error;
  GuiString method;
  GuiString precond;
  GuiInt iteration;
  GuiInt maxiter;
  GuiInt use_previous_soln;
  GuiInt emit_partial;
  GuiInt emit_iter;
  bool ep;
  int epcount;
  GuiString status;
  GuiInt tcl_np;
  CGData data;
};


DECLARE_MAKER(SolveLinearSystem)

SolveLinearSystem::SolveLinearSystem(GuiContext* ctx)
  : Module("SolveLinearSystem", ctx, Filter, "Math", "SCIRun"),
    target_error(get_ctx()->subVar("target_error"), 0.001),
    flops(get_ctx()->subVar("flops"), 0.0),
    floprate(get_ctx()->subVar("floprate"), 0.0),
    memrefs(get_ctx()->subVar("memrefs"), 0.0),
    memrate(get_ctx()->subVar("memrate"), 0.0),
    orig_error(get_ctx()->subVar("orig_error"), 0.0),
    current_error(get_ctx()->subVar("current_error"), ""),
    method(get_ctx()->subVar("method"), "Conjugate Gradient & Precond. (SCI)"),
    precond(get_ctx()->subVar("precond"), "jacobi"),
    iteration(get_ctx()->subVar("iteration"), 0),
    maxiter(get_ctx()->subVar("maxiter"), 200),
    use_previous_soln(get_ctx()->subVar("use_previous_soln"), 1),
    emit_partial(get_ctx()->subVar("emit_partial"), 1),
    emit_iter(get_ctx()->subVar("emit_iter"), 50),
    status(get_ctx()->subVar("status"), ""),
    tcl_np(get_ctx()->subVar("np"), 4)
{
}


SolveLinearSystem::~SolveLinearSystem()
{
}


void
SolveLinearSystem::tcl_command(GuiArgs& args, void*userdata)
{
  if (args[1] == "petscenabled")
  {
#ifdef PETSC_UNI
    args.result("1");
#else
    args.result("0");
#endif
  }
  else
  {
    Module::tcl_command(args, userdata);
  }
}


void
SolveLinearSystem::set_compute_time_stats(PStats *stats, double time, int nprocs)
{
  time = Max(time, 0.000001);
  flops.set(nprocs * stats->gflop * 1.e9 + stats->flop);
  floprate.set(nprocs * (stats->gflop * 1.e3 + stats->flop * 1.e-6) / time);
  memrefs.set(nprocs * stats->grefs * 1.e9 + stats->memref);
  memrate.set(nprocs * (stats->grefs * 1.e3 + stats->memref * 1.e-6) / time);
}


void
SolveLinearSystem::execute()
{
  MatrixHandle matrix;
  if (!get_input_handle("Matrix", matrix)) return;

  MatrixHandle rhs;
  if (!get_input_handle("RHS", rhs)) return;

  if (use_previous_soln.get() && solution.get_rep() &&
     solution->nrows() == rhs->nrows())
  {
    solution.detach();
  }
  else
  {
    solution = scinew ColumnMatrix(rhs->nrows());
    solution->zero();
    string units;
    if (rhs->get_property("units", units))
      solution->set_property("units", units, false);
  }

  const int size = matrix->nrows();
  if (matrix->ncols() != size)
  {
    error("Matrix should be square, but is " +
	  to_string(size) + " x " + to_string(matrix->ncols()));
    return;
  }
  if (rhs->nrows() != size || rhs->ncols() != 1)
  {
    error("Matrix size mismatch.");
    return;
  }

  ColumnMatrix *rhsp = rhs->as_column();
  ColumnMatrix *solp = solution->as_column();
  Matrix* mat = matrix.get_rep();

  bool delete_rhsp = false;
  if (!rhsp) {
    rhsp = rhs->column();
    delete_rhsp = true;
  }

  ep = emit_partial.get();
  epcount = Max(1, emit_iter.get());
  const string meth = method.get();

  bool intermediate = false;

  if (meth == "Conjugate Gradient & Precond. (SCI)") {
    conjugate_gradient_sci(mat, *solp, *rhsp);
  } else if (meth == "BiConjugate Gradient & Precond. (SCI)") {
    bi_conjugate_gradient_sci(mat, *solp, *rhsp);
    intermediate = true;
  } else if (meth == "Jacobi & Precond. (SCI)") {
    jacobi_sci(mat, *solp, *rhsp);
#ifdef PETSC_UNI
  } else if (meth == "KSPRICHARDSON (PETSc)") {
    petsc_solve(precond.get().c_str(),(char*)KSPRICHARDSON,mat,rhsp,solp);
  } else if (meth == "KSPCHEBYCHEV (PETSc)") {
    petsc_solve(precond.get().c_str(),(char*)KSPCHEBYCHEV,mat,rhsp,solp);
  } else if (meth == "KSPCG (PETSc)") {
    petsc_solve(precond.get().c_str(),(char*)KSPCG,mat,rhsp,solp);
  } else if (meth == "KSPGMRES (PETSc)") {
    petsc_solve(precond.get().c_str(),(char*)KSPGMRES,mat,rhsp,solp);
  } else if (meth == "KSPTCQMR (PETSc)") {
    petsc_solve(precond.get().c_str(),(char*)KSPTCQMR,mat,rhsp,solp);
  } else if (meth == "KSPBCGS (PETSc)") {
    petsc_solve(precond.get().c_str(),(char*)KSPBCGS,mat,rhsp,solp);
  } else if (meth == "KSPCGS (PETSc)") {
    petsc_solve(precond.get().c_str(),(char*)KSPCGS,mat,rhsp,solp);
  } else if (meth == "KSPTFQMR (PETSc)") {
    petsc_solve(precond.get().c_str(),(char*)KSPTFQMR,mat,rhsp,solp);
  } else if (meth == "KSPCR (PETSc)") {
    petsc_solve(precond.get().c_str(),(char*)KSPCR,mat,rhsp,solp);
  } else if (meth == "KSPLSQR (PETSc)") {
    petsc_solve(precond.get().c_str(),(char*)KSPLSQR,mat,rhsp,solp);
  } else if (meth == "KSPBICG (PETSc)") {
    petsc_solve(precond.get().c_str(),(char*)KSPBICG,mat,rhsp,solp);
  } else if (meth == "KSPPREONLY (PETSc)") {
    petsc_solve(precond.get().c_str(),(char*)KSPPREONLY,mat,rhsp,solp);
#endif
  } else {
    error("Unknown method: " + meth);
    return;
  }

  send_output_handle("Solution", solution, false, intermediate);

  if (delete_rhsp) { delete rhsp; }
}


#ifdef PETSC_UNI

int
PETSc_monitor(KSP, int niter, PetscReal err, void *context)
{
  SolveLinearSystem *solver = (SolveLinearSystem *)context;

  if (niter == 1) solver->PETSc_log_orig=log(err);

  solver->PETSc_targetidx.add(niter);
  solver->PETSc_targetlist.add(solver->PETSc_max_err);
  solver->PETSc_errlist.add(err);
  if (niter == 1 || niter == 5 || niter%10 == 0)
  {
    solver->iteration.set(niter);
    solver->current_error.set(to_string(err));
    solver->append_values(niter,
			  solver->PETSc_errlist, solver->PETSc_last_update,
			  solver->PETSc_targetidx, solver->PETSc_targetlist,
			  solver->PETSc_last_errupdate);

    double progress = ((solver->PETSc_log_orig-log(err))/
                       (solver->PETSc_log_orig-log(solver->PETSc_max_err)));
    solver->update_progress(progress);
  }

  return 0;
}


void
SolveLinearSystem::petsc_solve(const char* prec, const char* meth,
                         Matrix* matrix, ColumnMatrix* rhs,
                         ColumnMatrix* sol)
{
  PetscLock.lock();

  if (!Petsc.is_initialized()) {
    error("FATAL: PETSc is uninitialized.  exiting.");
    return;
  }

  int i,j;
  int rows = matrix->nrows();
  int cols = matrix->ncols();

  remark(string("rows, cols: " + to_string(rows) + ", " + to_string(cols)));

  // Create PETSc vectors
  Vec x,b;
  VecCreate(PETSC_COMM_WORLD, &x);
  VecSetSizes(x,PETSC_DECIDE, rhs->nrows());
  //  VecSetType(x,VECSEQ);
  VecSetFromOptions(x);
  VecDuplicate(x,&b);

  // Copy SCIRun RHS Vector to PETSc
  int *ix = scinew int[rows];
  for (i = 0; i < rows; ++i) ix[i] = i;
  VecSetValues(b,rows,ix,rhs->get_data(),INSERT_VALUES);
  VecAssemblyBegin(b);
  VecAssemblyEnd(b);

  // Copy SCIRun Matrix to PETSc matrix
  Mat A;
  SparseRowMatrix *sparse = dynamic_cast<SparseRowMatrix *>(matrix);
  if (sparse)
  {
    remark("Using Sparse Matrix.");
    MatCreateSeqAIJWithArrays(PETSC_COMM_WORLD, rows, cols,
			      sparse->rows, sparse->columns, sparse->a, &A);
  }
  else
  {
    MatCreate(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE,rows, cols, &A);
    MatSetType(A,MATSEQDENSE);
    for (i=0;i<rows;++i) {
      for (j=0;j<cols;++j) {
	// MH - MatSetValue appears to be a macro which has badness in
	// it.
	//MatSetValue(A,i,j,(*matrix)[i][j],INSERT_VALUES);
	
	int _ierr,_row = i,_col = j;
	PetscScalar _va = (*matrix)[i][j];
	_ierr = MatSetValues(A,1,&_row,1,&_col,&_va,INSERT_VALUES);
	
	if (_ierr) {
	  //  return PetscError(438,"unknownfunction",
	  //                    "../src/Dataflow/Modules/Math/SolveLinearSystem.cc",
	  //                    "unknowndirectory/",_ierr,0," ");
	  cerr << "An error occured assigning the value to the PETSc matrix...\n";
	}
      }
    }
    MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);
  }

  // Create linear solver context
  SLES sles;
  SLESCreate(PETSC_COMM_WORLD,&sles);
  SLESSetOperators(sles,A,A,DIFFERENT_NONZERO_PATTERN);

  // Get handles to the method and preconditioner
  KSP  ksp;
  PC   pc;
  SLESGetKSP(sles,&ksp);
  SLESGetPC(sles,&pc);

  // Set the method and preconditioner
  PCSetType(pc,(char*)prec);
  KSPSetType(ksp,(char*)meth);

  PETSc_max_err = target_error.get();

  //Setup callback to chart graph
  get_gui()->execute(get_id()+" reset_graph");
  PETSc_errlist.remove_all();
  PETSc_last_update=1;
  PETSc_targetidx.remove_all();
  PETSc_targetlist.remove_all();
  KSPSetMonitor(ksp, PETSc_monitor, (void *)this, PETSC_NULL);

  // If user wants inital non-zero guess, copy previous solution to PETSc
  if (use_previous_soln.get())
  {
    VecSetValues(x,rows,ix,sol->get_data(),INSERT_VALUES);
    VecAssemblyBegin(x);
    VecAssemblyEnd(x);
    KSPSetInitialGuessNonzero(ksp,PETSC_TRUE);
  }

  // Set linear solver to stop at target error
  KSPSetTolerances(ksp,1.0e-100, PETSc_max_err,10.0,maxiter.get());

  // Solve the linear system
  int its = maxiter.get();
  SLESSolve(sles,b,x,&its);
  iteration.set(its);

  // Determine if solution converted or diverged
  KSPConvergedReason reason;
  KSPGetConvergedReason(ksp, &reason);
  if (reason < 0)
  {
    error("SolveLinearSystem(PETSc) diverged.");
  }
  else
  {
    get_gui()->execute(get_id()+" finish_graph");
    append_values(its, PETSc_errlist, PETSc_last_update,
		  PETSc_targetidx, PETSc_targetlist,
		  PETSc_last_errupdate);
  }

  // Copy the solution out of PETSc into SCIRun
  double *solution;
  VecGetArray(x,&solution);
  memcpy(sol->get_data(), solution, sizeof(double)*sol->nrows());
  VecRestoreArray(x,&solution);

  // Cleanup PETSc memory
  VecDestroy(b);
  VecDestroy(x);
  MatDestroy(A);
  SLESDestroy(sles);
  delete[] ix;

  PetscLock.unlock();

  get_gui()->execute("update idletasks");
}
#endif


void
SolveLinearSystem::append_values(int niter, const vector<double>& errlist,
                           int& last_update,
                           const vector<int>& targetidx,
                           const vector<double>& targetlist,
                           int& last_errupdate)
{
  std::ostringstream str;
  str << get_id() << " append_graph " << niter << " \"";
  unsigned int i;
  for (i = last_update; i < errlist.size(); i++)
  {
    const double err = MakeReal(errlist[i]);
    if (err < 1000000)
      str << i << " " << errlist[i] << " ";
    else
      str << i << " 1000000 ";
  }
  str << "\" \"";
  for (i = last_errupdate; i<targetidx.size(); i++)
  {
    str << targetidx[i] << " " << targetlist[i] << " ";
  }
  str << "\" ; update idletasks";
  get_gui()->execute(str.str().c_str());
  last_update = errlist.size();
  last_errupdate = targetidx.size();
}


void
SolveLinearSystem::jacobi_sci(Matrix* matrix, ColumnMatrix& lhs, ColumnMatrix& rhs)
{
  const int size = matrix->nrows();

  PStats stats_struct;
  PStats *stats = &stats_struct;

  stats->flop = 0;
  stats->memref = 0;
  stats->gflop = 0;
  stats->grefs = 0;
  set_compute_time_stats(stats, 0.0, 0);
  iteration.set(0);

  ColumnMatrix invdiag(size);

  // We should try to do a better job at preconditioning.
  for (int i=0; i<size; i++)
  {
    if (Abs(matrix->get(i,i) > 0.000001))
      invdiag[i] = 1.0 / matrix->get(i,i);
    else
      invdiag[i] = 1.0;
  }
  stats->flop += size;
  stats->memref = 2*size*(int)sizeof(double);

  ColumnMatrix Z(size);
  matrix->mult(lhs, Z, stats->flop, stats->memref);

  Sub(Z, Z, rhs, stats->flop, stats->memref);
  const double bnorm = rhs.vector_norm(stats->flop, stats->memref);
  double err = Z.vector_norm(stats->flop, stats->memref) / bnorm;
  if (err >= 10000000) err = 1000000;

  orig_error.set(err);
  current_error.set(to_string(err));

  int niter = 0;
  int toomany = maxiter.get();
  if (toomany == 0) { toomany = 2*size; }
  double max_error = target_error.get();

  stats->gflop += stats->flop / 1000000000;
  stats->flop = stats->flop % 1000000000;
  stats->grefs += stats->memref / 1000000000;
  stats->memref = stats->memref % 1000000000;
  set_compute_time_stats(stats, timer_.time(), 1);

  get_gui()->execute(get_id() + " reset_graph");
  vector<double> errlist;
  errlist.push_back(err);
  int last_update = 0;

  vector<int> targetidx;
  vector<double> targetlist;
  int last_errupdate = 0;
  targetidx.push_back(0);
  targetlist.push_back(max_error);

  append_values(1, errlist, last_update, targetidx,
                targetlist, last_errupdate);

  const double log_orig = log(err);
  const double log_targ = log(max_error);

  while (niter < toomany)
  {
    niter++;

    target_error.reset();
    const double new_error = target_error.get();
    if (new_error != max_error)
    {
      targetidx.push_back(niter);
      targetlist.push_back(max_error);
      max_error = new_error;
    }
    targetidx.push_back(niter);
    targetlist.push_back(max_error);
    if (err < max_error)
      break;
    if (err > 10)
    {
      error("Solution not converging!");
      break;
    }

    Mult(Z, invdiag, Z, stats->flop, stats->memref);
    ScMult_Add(lhs, 1, lhs, Z, stats->flop, stats->memref);

    matrix->mult(lhs, Z, stats->flop, stats->memref);
    Sub(Z, rhs, Z, stats->flop, stats->memref);
    err = Z.vector_norm(stats->flop, stats->memref) / bnorm;

    if (err >= 10000000) { err = 1000000; }

    errlist.push_back(err);

    stats->gflop += stats->flop / 1000000000;
    stats->flop = stats->flop % 1000000000;
    stats->grefs += stats->memref / 1000000000;
    stats->memref = stats->memref % 1000000000;

    if (niter == 1 || niter == 5 || niter%10 == 0)
    {
      iteration.set(niter);
      current_error.set(to_string(err));
      set_compute_time_stats(stats, timer_.time(), 1);

      append_values(niter, errlist, last_update, targetidx,
                    targetlist, last_errupdate);

      const double progress = (log_orig-log(err))/(log_orig-log_targ);
      update_progress(progress);

      if (ep && niter%epcount == 0)
      {
        MatrixHandle rhsH(rhs.clone());
        send_output_handle("Solution", rhsH, false, true);
      }
    }
  }
  iteration.set(niter);
  current_error.set(to_string(err));

  set_compute_time_stats(stats, timer_.time(), 1);

  get_gui()->execute(get_id()+" finish_graph");
  append_values(niter, errlist, last_update, targetidx,
                targetlist, last_errupdate);
}


CGData::CGData()
  : timer(0),
    diag(0),
    Z(0),
    R(0),
    P(0),
    Z1(0),
    R1(0),
    P1(0),
    trans(0),
    reducer("SolveLinearSystem reduction barrier"),
    stats(0)
{
}

void
CGData::clear()
{
  if (timer) { delete timer; timer = 0; }
  if (diag) { delete diag; diag = 0; }
  if (Z) { delete Z; Z = 0; }
  if (R) { delete R; R = 0; }
  if (P) { delete P; P = 0; }
  if (Z1) { delete Z1; Z1 = 0; }
  if (R1) { delete R1; R1 = 0; }
  if (P1) { delete P1; P1 = 0; }
  if (trans) { delete trans; trans = 0; }
  if (stats) { delete stats; stats = 0; }
}


void
SolveLinearSystem::conjugate_gradient_sci(Matrix* matrix,
                                    ColumnMatrix& lhs, ColumnMatrix& rhs)
{
  const int np = tcl_np.get();

  data.module = this;
  data.np = np;
  data.rhs = &rhs;
  data.lhs = &lhs;
  data.mat = matrix;
  data.timer = new WallClockTimer;
  data.stats = new PStats[data.np];

  if (init_parallel_conjugate_gradient())
  {
    Thread::parallel(this, &SolveLinearSystem::parallel_conjugate_gradient, data.np);
  }

  remark("Conjugate Gradient done in " +
         to_string(data.timer->time()) + " seconds.");

  data.clear();
}


bool
SolveLinearSystem::init_parallel_conjugate_gradient()
{
  Matrix* matrix = data.mat;
  PStats* stats = &data.stats[0];
  const int size = matrix->nrows();

  data.timer->clear();
  data.timer->start();
  set_compute_time_stats(stats, 0.0, 0);
  iteration.set(0);

  data.niter = 0;
  data.toomany = maxiter.get();
  if (data.toomany == 0) { data.toomany = 2*size; }

  if (data.rhs->vector_norm(stats->flop, stats->memref) < 0.0000001)
  {
    *data.lhs = *data.rhs;
    stats->memref += 2*size*sizeof(double);
    return false;
  }

  // We should try to do a better job at preconditioning.
  data.diag = new ColumnMatrix(size);
  ColumnMatrix& diag = *data.diag;
  for (int i=0; i<size; i++)
  {
    if (Abs(matrix->get(i,i) > 0.000001))
      diag[i] = 1.0 / matrix->get(i,i);
    else
      diag[i] = 1.0;
  }
  stats->flop += size;
  stats->memref += 2*size*sizeof(double);

  data.R = new ColumnMatrix(size);
  ColumnMatrix& R = *data.R;
  ColumnMatrix& lhs = *data.lhs;
  matrix->mult(lhs, R, stats->flop, stats->memref);

  ColumnMatrix& rhs = *data.rhs;
  Sub(R, rhs, R, stats->flop, stats->memref);
  data.bnorm = rhs.vector_norm(stats->flop, stats->memref);

  data.Z = new ColumnMatrix(size);
  ColumnMatrix& Z = *data.Z;
  matrix->mult(R, Z, stats->flop, stats->memref);

  data.P = new ColumnMatrix(size);

  data.err = R.vector_norm(stats->flop, stats->memref) / data.bnorm;
  if (data.err == 0)
  {
    lhs = rhs;
    stats->memref += 2*size*sizeof(double);
    return false;
  }
  else
  {
    if (data.err >= 1000000) { data.err = 1000000; }
  }

  data.max_error = target_error.get();

  stats->gflop += stats->flop / 1000000000;
  stats->flop = stats->flop % 1000000000;
  stats->grefs += stats->memref / 1000000000;
  stats->memref = stats->memref % 1000000000;
  orig_error.set(data.err);
  current_error.set(to_string(data.err));
  set_compute_time_stats(stats, data.timer->time(), 1);

  get_gui()->execute(get_id() + " reset_graph");
  vector<int> targetidx;
  vector<double> targetlist;
  vector<double> errlist;
  int last_update = 0;
  int last_errupdate = 0;
  errlist.push_back(data.err);
  targetidx.push_back(0);
  targetlist.push_back(data.max_error);
  append_values(1, errlist, last_update, targetidx,
                targetlist, last_errupdate);

  return true;
}


void
SolveLinearSystem::parallel_conjugate_gradient(int processor)
{
  Matrix* matrix = data.mat;
  PStats* stats = &data.stats[processor];
  const int size = matrix->nrows();

  const int beg = processor*size / data.np;
  const int end = (processor+1)*size / data.np;
  stats->flop = 0;
  stats->memref = 0;
  stats->gflop = 0;
  stats->grefs = 0;

  vector<int> targetidx;
  vector<double> targetlist;
  vector<double> errlist;
  int last_update = 0;
  int last_errupdate = 0;

  const double log_orig = log(data.err);
  const double log_targ = log(data.max_error);

  double err = data.err;
  double bkden = 0;
  while (data.niter < data.toomany)
  {
    ColumnMatrix& Z = *data.Z;
    ColumnMatrix& P = *data.P;
    if (processor == 0)
    {
      target_error.reset();
      const double new_error = target_error.get();
      if (new_error != data.max_error)
      {
	targetidx.push_back(data.niter+1);
	targetlist.push_back(data.max_error);
	data.max_error = new_error;
      }
      targetidx.push_back(data.niter);
      targetlist.push_back(data.max_error);
    }

    data.reducer.wait(data.np);

    if (err < 1.e-15 || err < data.max_error)
      break;

    if (processor == 0)
      data.niter++;

    // Simple Preconditioning.
    ColumnMatrix& diag = *data.diag;
    ColumnMatrix& R = *data.R;
    Mult(Z, R, diag, stats->flop, stats->memref, beg, end);

    // Calculate coefficient bk and direction vectors p and pp
    const double my_bknum = Dot(Z, R, stats->flop, stats->memref, beg, end);
    const double bknum = data.reducer.sum(processor, data.np, my_bknum);

    if (data.niter == 1)
    {
      Copy(P, Z, stats->flop, stats->memref, beg, end);
    }
    else
    {
      const double bk = bknum/bkden;
      ScMult_Add(P, bk, P, Z, stats->flop, stats->memref, beg, end);
    }
    data.reducer.wait(data.np);

    // Calculate coefficient ak, new iterate x and new residuals r and rr
    matrix->mult(P, Z, stats->flop, stats->memref, beg, end);
    bkden = bknum;

    const double my_akden = Dot(Z, P, stats->flop, stats->memref, beg, end);
    const double akden = data.reducer.sum(processor, data.np, my_akden);
    const double ak = bknum / akden;
    ColumnMatrix& lhs = *data.lhs;
    ScMult_Add(lhs, ak, P, lhs, stats->flop, stats->memref, beg, end);
    ScMult_Add(R, -ak, Z, R, stats->flop, stats->memref, beg, end);

    const double my_err = R.vector_norm(stats->flop, stats->memref,
                                        beg, end) / data.bnorm;
    err = data.reducer.sum(processor, data.np, my_err);
    if (err >= 1000000) { err = 1000000; }

    stats->gflop += stats->flop / 1000000000;
    stats->flop = stats->flop % 1000000000;
    stats->grefs += stats->memref / 1000000000;
    stats->memref = stats->memref % 1000000000;

    if (processor == 0)
    {
      errlist.push_back(err);

      stats->gflop += stats->flop / 1000000000;
      stats->flop = stats->flop % 1000000000;
      stats->grefs += stats->memref / 1000000000;
      stats->memref = stats->memref % 1000000000;

      if (data.niter == 1 || data.niter == 10 || data.niter%20 == 0)
      {
	if (data.niter <= 60 || data.niter%60 == 0)
        {
	  iteration.set(data.niter);
	  current_error.set(to_string(err));
          set_compute_time_stats(stats, data.timer->time(), data.np);
	  append_values(data.niter, errlist, last_update, targetidx,
			targetlist, last_errupdate);
	
	  if (err > 0)
          {
	    const double progress = (log_orig-log(err))/(log_orig-log_targ);
	    update_progress(progress);
	  }
	}

	if (ep && data.niter%epcount == 0)
        {
          MatrixHandle lhsH(lhs.clone());
          send_output_handle("Solution", lhsH, false, true);
        }
      }
    }
  }

  data.reducer.wait(data.np);

  if (processor == 0)
  {
    data.niter++;

    iteration.set(data.niter);
    current_error.set(to_string(err));
    data.timer->stop();
    const double time = data.timer->time();
    set_compute_time_stats(stats, time, data.np);
    remark("Done in " + to_string(time) + " seconds.");

    get_gui()->execute(get_id()+" finish_graph");
    append_values(data.niter, errlist, last_update, targetidx, targetlist,
		  last_errupdate);
  }
}


void
SolveLinearSystem::bi_conjugate_gradient_sci(Matrix* matrix,
				       ColumnMatrix& lhs, ColumnMatrix& rhs)
{
  const int np = tcl_np.get();

  data.module = this;
  data.np = np;
  data.rhs = &rhs;
  data.lhs = &lhs;
  data.mat = matrix;
  data.timer = new WallClockTimer;
  data.stats = new PStats[data.np];
  data.trans = matrix->transpose();

  if (init_parallel_bi_conjugate_gradient())
  {
    Thread::parallel(this, &SolveLinearSystem::parallel_bi_conjugate_gradient,
                     data.np);
  }

  remark("Bi Conjugate Gradient done in " +
         to_string(data.timer->time()) + " seconds.");

  data.clear();
}


bool
SolveLinearSystem::init_parallel_bi_conjugate_gradient()
{
  Matrix* matrix = data.mat;
  PStats* stats = &data.stats[0];
  const int size = matrix->nrows();

  data.timer->clear();
  data.timer->start();
  set_compute_time_stats(stats, 0.0, 0);
  iteration.set(0);

  // We should try to do a better job at preconditioning.
  data.diag = new ColumnMatrix(size);
  ColumnMatrix& diag = *data.diag;
  for (int i=0; i<size; i++)
  {
    if (Abs(matrix->get(i,i) > 0.000001))
      diag[i] = 1.0 / matrix->get(i,i);
    else
      diag[i] = 1.0;
  }
  stats->flop += size;
  stats->memref += 2*size*sizeof(double);

  data.R = new ColumnMatrix(size);
  ColumnMatrix& R = *data.R;
  ColumnMatrix& lhs = *data.lhs;
  matrix->mult(lhs, R, stats->flop, stats->memref);

  const ColumnMatrix& rhs = *data.rhs;
  Sub(R, rhs, R, stats->flop, stats->memref);
  data.bnorm = rhs.vector_norm(stats->flop, stats->memref);

  // BiCG
  data.R1 = new ColumnMatrix(size);
  ColumnMatrix& R1 = *data.R1;
  Copy(R1, R, stats->flop, stats->memref, 0, size);

  data.Z = new ColumnMatrix(size);

  // BiCG ??
  data.Z1 = new ColumnMatrix(size);

  data.P = new ColumnMatrix(size);

  // BiCG
  data.P1 = new ColumnMatrix(size);

  data.err = R.vector_norm(stats->flop, stats->memref) / data.bnorm;
  if (data.err == 0)
  {
    lhs = rhs;
    stats->memref += 2*size*sizeof(double);
    return false;
  }
  else
  {
    if (data.err >= 1000000) { data.err = 1000000; }
  }

  data.niter = 0;
  data.toomany = maxiter.get();
  if (data.toomany == 0)
    data.toomany = 2*size;
  data.max_error = target_error.get();

  stats->gflop += stats->flop / 1000000000;
  stats->flop = stats->flop % 1000000000;
  stats->grefs += stats->memref / 1000000000;
  stats->memref = stats->memref % 1000000000;
  orig_error.set(data.err);
  current_error.set(to_string(data.err));

  set_compute_time_stats(stats, data.timer->time(), 1);

  get_gui()->execute(get_id()+" reset_graph");
  vector<int> targetidx;
  vector<double> targetlist;
  vector<double> errlist;
  int last_update = 0;
  int last_errupdate = 0;
  errlist.push_back(data.err);
  targetidx.push_back(0);
  targetlist.push_back(data.max_error);

  append_values(1, errlist, last_update, targetidx,
                targetlist, last_errupdate);

  return true;
}


void
SolveLinearSystem::parallel_bi_conjugate_gradient(int processor)
{
  Matrix* matrix = data.mat;
  PStats* stats = &data.stats[processor];
  const int size = matrix->nrows();

  const int beg = processor*size / data.np;
  const int end = Min((processor+1)*size / data.np, size);
  stats->flop = 0;
  stats->memref = 0;
  stats->gflop = 0;
  stats->grefs = 0;

  vector<int> targetidx;
  vector<double> targetlist;
  vector<double> errlist;
  int last_update = 0;
  int last_errupdate = 0;

  const double log_orig = log(data.err);
  const double log_targ = log(data.max_error);

  double err = data.err;
  double bkden = 0;
  while (data.niter < data.toomany)
  {
    ColumnMatrix& Z = *data.Z;
    ColumnMatrix& P = *data.P;
    // BiCG
    ColumnMatrix& Z1 = *data.Z1;
    ColumnMatrix& P1 = *data.P1;

    if (processor == 0)
    {
      target_error.reset();
      double new_error = target_error.get();
      if (new_error != data.max_error)
      {
	targetidx.push_back(data.niter+1);
	targetlist.push_back(data.max_error);
	data.max_error = new_error;
      }
      targetidx.push_back(data.niter);
      targetlist.push_back(data.max_error);
    }

    data.reducer.wait(data.np);

    if (err < data.max_error)
      break;

    if ( processor == 0 )
      data.niter++;

    // Simple Preconditioning.
    ColumnMatrix& diag = *data.diag;
    ColumnMatrix& R = *data.R;
    Mult(Z, R, diag, stats->flop, stats->memref, beg, end);

    // BiCG
    ColumnMatrix& R1 = *data.R1;
    Mult(Z1, R1, diag, stats->flop, stats->memref, beg, end);

    // Calculate coefficient bk and direction vectors p and pp
    // BiCG - change R->R1
    const double my_bknum = Dot(Z, R1, stats->flop, stats->memref, beg, end);
    const double bknum = data.reducer.sum(processor, data.np, my_bknum);

    // BiCG
    if ( bknum == 0 ) {
      break;
    }

    if (data.niter == 1)
    {
      Copy(P, Z, stats->flop, stats->memref, beg, end);
      // BiCG
      Copy(P1, Z1, stats->flop, stats->memref, beg, end);
    }
    else
    {
      const double bk = bknum/bkden;
      ScMult_Add(P, bk, P, Z, stats->flop, stats->memref, beg, end);
      // BiCG
      ScMult_Add(P1, bk, P1, Z1, stats->flop, stats->memref, beg, end);
    }

    data.reducer.wait(data.np);

    // Calculate coefficient ak, new iterate x and new residuals r and rr
    matrix->mult(P, Z, stats->flop, stats->memref, beg, end);
    bkden = bknum;

    // BiCG
    data.trans->mult(P1, Z1, stats->flop, stats->memref, beg, end);

    // BiCG = change P -> P1
    const double my_akden = Dot(Z, P1, stats->flop, stats->memref, beg, end);
    const double akden = data.reducer.sum(processor, data.np, my_akden);
    const double ak = bknum / akden;
    ColumnMatrix& lhs = *data.lhs;
    ScMult_Add(lhs, ak, P, lhs, stats->flop, stats->memref, beg, end);
    ScMult_Add(R, -ak, Z, R, stats->flop, stats->memref, beg, end);
    // BiCG
    ScMult_Add(R1, -ak, Z1, R1, stats->flop, stats->memref, beg, end);

    const double my_err = R.vector_norm(stats->flop, stats->memref,
                                        beg, end) / data.bnorm;
    err = data.reducer.sum(processor, data.np, my_err);
    if (err >= 1000000) { err = 1000000; }

    stats->gflop += stats->flop/1000000000;
    stats->flop = stats->flop % 1000000000;
    stats->grefs += stats->memref / 1000000000;
    stats->memref = stats->memref % 1000000000;

    if (processor == 0)
    {
      errlist.push_back(err);

      stats->gflop += stats->flop / 1000000000;
      stats->flop = stats->flop % 1000000000;
      stats->grefs += stats->memref / 1000000000;
      stats->memref = stats->memref % 1000000000;

      if (data.niter == 1 || data.niter == 10 || data.niter%20 == 0)
      {
	if (data.niter <= 60 || data.niter%60 == 0)
        {
	  iteration.set(data.niter);
	  current_error.set(to_string(err));
          set_compute_time_stats(stats, data.timer->time(), data.np);
	  append_values(data.niter, errlist, last_update, targetidx,
			targetlist, last_errupdate);
	
	  if (err > 0)
          {
	    const double progress = (log_orig-log(err)) / (log_orig-log_targ);
	    update_progress(progress);
	  }
	}
      }
    }
  }

  data.reducer.wait(data.np);

  if (processor == 0)
  {
    data.niter++;

    iteration.set(data.niter);
    current_error.set(to_string(err));
    data.timer->stop();
    const double time = data.timer->time();
    set_compute_time_stats(stats, data.timer->time(), data.np);
    remark("Done in " + to_string(time) + " seconds.");

    get_gui()->execute(get_id()+" finish_graph");
    append_values(data.niter, errlist, last_update, targetidx, targetlist,
		  last_errupdate);
  }
}


} // End namespace SCIRun
