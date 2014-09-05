/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

/*
 *  SolveMatrix.cc:  Unfinished modules
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */


#include <sci_config.h>
#include <stdio.h>
#include <math.h>

#include <Dataflow/Network/Module.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/SparseRowMatrix.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Dataflow/Network/NetworkEditor.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Core/Geometry/Point.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MiscMath.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Thread/Parallel.h>
#include <Core/Thread/SimpleReducer.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Mutex.h>
#include <iostream>
#include <sstream>

#ifdef UNI_PETSC

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

Mutex PetscLock("SolveMatrix PETSc lock");

struct PStats {
  int flop;
  int memref;
  int gflop;
  int grefs;
  int pad[28];
};

class SolveMatrix;

struct CGData {
  SolveMatrix* module;
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
};

class SolveMatrix : public Module {
  MatrixIPort* matrixport;
  MatrixIPort* rhsport;
  MatrixOPort* solport;
  MatrixHandle solution;
  
#ifdef UNI_PETSC
  // the only contexts that this instance of SolveMatrix will use
  SLES sles;  
  KSP  ksp;
  PC   pc;   
  void petsc_solve(const char*, const char*,Matrix* ,
		   ColumnMatrix *, ColumnMatrix *);
#endif
  
  void jacobi_sci(Matrix*,ColumnMatrix& , ColumnMatrix&);
  void conjugate_gradient_sci(Matrix*,ColumnMatrix&, ColumnMatrix&);
  void bi_conjugate_gradient_sci(Matrix*,ColumnMatrix&, ColumnMatrix&);
  
  void append_values(int niter, const Array1<double>& errlist,
		     int& last_update, const Array1<int>& targetidx,
		     const Array1<double>& targetlist,
		     int& last_errupdate);
public:
  void parallel_conjugate_gradient(int proc);
  void parallel_bi_conjugate_gradient(int proc);
  SolveMatrix(const string& id);
  virtual ~SolveMatrix();
  virtual void execute();
  
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
  int ep;
  GuiString status;
  GuiInt tcl_np;
  CGData data;
};

extern "C" Module* make_SolveMatrix(const string& id) {
  return new SolveMatrix(id);
}

 
SolveMatrix::SolveMatrix(const string& id)
  : Module("SolveMatrix", id, Filter, "Math", "SCIRun"),
    target_error("target_error", id, this),
    flops("flops", id, this),
    floprate("floprate", id, this),
    memrefs("memrefs", id, this),
    memrate("memrate", id, this),
    orig_error("orig_error", id, this),
    current_error("current_error", id, this),
    method("method", id, this),
    precond("precond",id,this),
    iteration("iteration", id, this),
    maxiter("maxiter", id, this),
    use_previous_soln("use_previous_soln", id, this),
    emit_partial("emit_partial", id, this),
    status("status",id,this),
    tcl_np("np", id, this)
{
#ifdef UNI_PETSC
  SLESCreate(PETSC_COMM_WORLD,&sles);
#endif
}

SolveMatrix::~SolveMatrix()
{
#ifdef UNI_PETSC
  SLESDestroy(sles);
#endif
}

void SolveMatrix::execute()
{
  matrixport = (MatrixIPort *)get_iport("Matrix");
  rhsport = (MatrixIPort *)get_iport("RHS");
  solport = (MatrixOPort *)get_oport("Solution");
  
  if (!matrixport) {
    postMessage("Unable to initialize "+name+"'s iport\n");
    return;
  }
  if (!rhsport) {
    postMessage("Unable to initialize "+name+"'s iport\n");
    return;
  }
  if (!solport) {
    postMessage("Unable to initialize "+name+"'s oport\n");
    return;
  }
  
  MatrixHandle matrix;
  MatrixHandle rhs;
  
  int m = matrixport->get(matrix);
  int r = rhsport->get(rhs);
  
  if ( !r || !m ) {
    return;
  }
  
  if ( !matrix.get_rep() || !rhs.get_rep() ) {
    warning("No input.");
    solport->send(MatrixHandle(0));
    return;
  }
  
  if(use_previous_soln.get() && solution.get_rep() && 
     solution->nrows() == rhs->nrows()){
    solution.detach();
  } else {
    solution=scinew ColumnMatrix(rhs->nrows());
    solution->zero();
  }
  
  int size=matrix->nrows();
  if(matrix->ncols() != size){
    error("Matrix should be square, but is " +
	  to_string(size) + " x " + to_string(matrix->ncols()));
    return;
  }
  if(rhs->nrows() != size){
    error("Matrix size mismatch");
    return;
  }
  
  ColumnMatrix *rhsp = dynamic_cast<ColumnMatrix*>(rhs.get_rep());
  ColumnMatrix *solp = dynamic_cast<ColumnMatrix*>(solution.get_rep());
  Matrix* mat = matrix.get_rep();
  
  if (!rhsp) {
    error("rhs isn't a column!");
    return;
  }
  
  ep=emit_partial.get();
  string meth=method.get();

  if(meth == "Conjugate Gradient & Precond. (SCI)"){
    conjugate_gradient_sci(mat, *solp, *rhsp);
    solport->send(solution);
  } else if(meth == "BiConjugate Gradient & Precond. (SCI)"){
    bi_conjugate_gradient_sci(mat, *solp, *rhsp);
    solport->send_intermediate(solution);
  } else if(meth == "Jacoby & Precond. (SCI)"){
    jacobi_sci(mat, *solp, *rhsp);
    solport->send(solution);
#ifdef UNI_PETSC
  } else if(meth == "KSPRICHARDSON (PETSc)") {
    petsc_solve(precond.get().c_str(),(char*)KSPRICHARDSON,mat,rhsp,solp); 
    solport->send(solution);
  } else if(meth == "KSPCHEBYCHEV (PETSc)") {
    petsc_solve(precond.get().c_str(),(char*)KSPCHEBYCHEV,mat,rhsp,solp);
    solport->send(solution);
  } else if(meth == "KSPCG (PETSc)") {
    petsc_solve(precond.get().c_str(),(char*)KSPCG,mat,rhsp,solp);
    solport->send(solution);
  } else if(meth == "KSPGMRES (PETSc)") {
    petsc_solve(precond.get().c_str(),(char*)KSPGMRES,mat,rhsp,solp);
    solport->send(solution);
  } else if(meth == "KSPTCQMR (PETSc)") {
    petsc_solve(precond.get().c_str(),(char*)KSPTCQMR,mat,rhsp,solp);
    solport->send(solution);
  } else if(meth == "KSPBCGS (PETSc)") {
    petsc_solve(precond.get().c_str(),(char*)KSPBCGS,mat,rhsp,solp);
    solport->send(solution);
  } else if(meth == "KSPCGS (PETSc)") {
    petsc_solve(precond.get().c_str(),(char*)KSPCGS,mat,rhsp,solp);
    solport->send(solution);
  } else if(meth == "KSPTFQMR (PETSc)") {
    petsc_solve(precond.get().c_str(),(char*)KSPTFQMR,mat,rhsp,solp);
    solport->send(solution);
  } else if(meth == "KSPCR (PETSc)") {
    petsc_solve(precond.get().c_str(),(char*)KSPCR,mat,rhsp,solp);
    solport->send(solution);
  } else if(meth == "KSPLSQR (PETSc)") {
    petsc_solve(precond.get().c_str(),(char*)KSPLSQR,mat,rhsp,solp);
    solport->send(solution);
  } else if(meth == "KSPBICG (PETSc)") {
    petsc_solve(precond.get().c_str(),(char*)KSPBICG,mat,rhsp,solp);
    solport->send(solution);
  } else if(meth == "KSPPREONLY (PETSc)") {
    petsc_solve(precond.get().c_str(),(char*)KSPPREONLY,mat,rhsp,solp);
    solport->send(solution);
#endif
  } else {
    error("Unknown method: " + meth);
  }
}

#ifdef UNI_PETSC
void SolveMatrix::petsc_solve(const char* prec, const char* meth, 
			      Matrix* matrix, ColumnMatrix* rhs, 
			      ColumnMatrix* sol)
{
  Mat A;
  Vec x,b;

  PetscLock.lock();

  if (!Petsc.is_initialized()) {
    error("FATAL: PETSc is uninitialized.  exiting.");
    return;
  }

  int rows = matrix->nrows();
  int cols = matrix->ncols();

  remark(string("rows, cols: " + to_string(rows) + ", " + to_string(cols)));

  // create storage for the PETSc objects
  MatCreate(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE,
	    rows, cols, &A);
  MatSetType(A,MATSEQDENSE);
  VecCreate(PETSC_COMM_WORLD, PETSC_DECIDE, rows,&x);
  VecSetType(x,VEC_SEQ);
  VecDuplicate(x,&b);

  // copy the matrix and rhs vector to petsc
  int i,j;
  for (i=0;i<rows;++i) {
    for (j=0;j<cols;++j) { 
      MatSetValue(A,i,j,(*matrix)[i][j],INSERT_VALUES);
    }
  }
  VecSet(rhs->get_data(),b);
  MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);

  // set up the equation
  SLESSetOperators(sles,A,A,DIFFERENT_NONZERO_PATTERN);

  // get handles to the method and preconditioner
  SLESGetKSP(sles,&ksp);
  SLESGetPC(sles,&pc);

  // set the method and preconditioner
  KSPSetType(ksp,(char*)meth);
  PCSetType(pc,(char*)prec);

  KSPSetTolerances(ksp,target_error.get(),
		   PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);

  int its;
  SLESSolve(sles,b,x,&its);

  // copy the solution out of petsc
  double *s;
  VecGetArray(x,&s);
  for (i=0;i<rows;++i) {
    sol->put(i,s[i]);
  }
  VecRestoreArray(x,&s);

  // clean up
  VecDestroy(b);
  VecDestroy(x);
  MatDestroy(A);
  
  PetscLock.unlock();

  iteration.set(its);
  TCL::execute("update idletasks");
}
#endif

void SolveMatrix::append_values(int niter, const Array1<double>& errlist,
				int& last_update,
				const Array1<int>& targetidx,
				const Array1<double>& targetlist,
				int& last_errupdate)
{
    std::ostringstream str;
    str << id << " append_graph " << niter << " \"";
    int i;
    for(i=last_update;i<errlist.size();i++){
	if (errlist[i]<1000000) 
	    str << i << " " << errlist[i] << " ";
	else 
	    str << i << " 1000000 ";
    }
    str << "\" \"";
    for(i=last_errupdate;i<targetidx.size();i++){
	str << targetidx[i] << " " << targetlist[i] << " ";
    }
    str << "\" ; update idletasks";
    TCL::execute(str.str().c_str());
    last_update=errlist.size();
    last_errupdate=targetidx.size();
}

void SolveMatrix::jacobi_sci(Matrix* matrix,
			 ColumnMatrix& lhs, ColumnMatrix& rhs)
{
    int size=matrix->nrows();

    int flop=0;
    int memref=0;
    int gflop=0;
    int grefs=0;
    flops.set(flop);
    floprate.set(0);
    memrefs.set(memref);
    memrate.set(0);

    iteration.set(0);
    
    ColumnMatrix invdiag(size);
    // We should try to do a better job at preconditioning...
    int i;

    for(i=0;i<size;i++){
      if (Abs(matrix->get(i,i)>0.000001))
	invdiag[i]=1./matrix->get(i,i);
      else
	invdiag[i]=1;
    }
    flop+=size;
    memref=2*size*(int)sizeof(double);

    ColumnMatrix Z(size);
    matrix->mult(lhs, Z, flop, memref);

    Sub(Z, Z, rhs, flop, memref);
    double bnorm=rhs.vector_norm(flop, memref);
    double err=Z.vector_norm(flop, memref)/bnorm;

    if (!(err < 10000000)) err=1000000;

    orig_error.set(err);
    current_error.set(to_string(err));

    int niter=0;
    int toomany=maxiter.get();
    if(toomany == 0)
	toomany=2*size;
    double max_error=target_error.get();

    double time=timer.time();
    flops.set(gflop*1.e9+flop);
    floprate.set((gflop*1.e3+flop*1.e-6)/time);
    memrefs.set(grefs*1.e9+memref);
    memrate.set((grefs*1.e3+memref*1.e-6)/time);
    gflop+=flop/1000000000;
    flop=flop%1000000000;
    grefs+=memref/1000000000;
    memref=memref%1000000000;
    
    TCL::execute(id+" reset_graph");
    Array1<double> errlist;
    errlist.add(err);
    int last_update=0;

    Array1<int> targetidx;
    Array1<double> targetlist;
    int last_errupdate=0;
    targetidx.add(0);
    targetlist.add(max_error);

    append_values(1, errlist, last_update, targetidx, targetlist, last_errupdate);

    double log_orig=log(err);
    double log_targ=log(max_error);
    while(niter < toomany){
	niter++;

	double new_error;
	if(get_gui_doublevar(id, "target_error", new_error)
	   && new_error != max_error){
	    targetidx.add(niter);
	    targetlist.add(max_error);
	    max_error=new_error;
	}
	targetidx.add(niter);
	targetlist.add(max_error);
	if(err < max_error)
	    break;
	if(err > 10){
	    error("Solution not converging!");
	    break;
	}

	Mult(Z, invdiag, Z, flop, memref);
	ScMult_Add(lhs, 1, lhs, Z, flop, memref);
//	Sub(lhs, lhs, Z, flop, memref);

	matrix->mult(lhs, Z, flop, memref);
	Sub(Z, rhs, Z, flop, memref);
	err=Z.vector_norm(flop, memref)/bnorm;

	if (!(err < 10000000)) err=1000000;

	errlist.add(err);

	gflop+=flop/1000000000;
	flop=flop%1000000000;
	grefs+=memref/1000000000;
	memref=memref%1000000000;

	if(niter == 1 || niter == 5 || niter%10 == 0){
	    iteration.set(niter);
	    current_error.set(to_string(err));
	    double time=timer.time();
	    flops.set(gflop*1.e9+flop);
	    floprate.set((gflop*1.e3+flop*1.e-6)/time);
	    memrefs.set(grefs*1.e9+memref);
	    memrate.set((grefs*1.e3+memref*1.e-6)/time);

	    append_values(niter, errlist, last_update, targetidx, targetlist, last_errupdate);

	    double progress=(log_orig-log(err))/(log_orig-log_targ);
	    update_progress(progress);
	    if(ep && niter%50 == 0)
		solport->send_intermediate(rhs.clone());
	}
    }
    iteration.set(niter);
    current_error.set(to_string(err));

    time=timer.time();
    flops.set(gflop*1.e9+flop);
    floprate.set((gflop*1.e3+flop*1.e-6)/time);
    memrefs.set(grefs*1.e9+memref);
    memrate.set((grefs*1.e3+memref*1.e-6)/time);

    TCL::execute(id+" finish_graph");
    append_values(niter, errlist, last_update, targetidx, targetlist, last_errupdate);
}

CGData::CGData()
    : reducer("SolveMatrix reduction barrier")
{
}

void SolveMatrix::conjugate_gradient_sci(Matrix* matrix,
					 ColumnMatrix& lhs, ColumnMatrix& rhs)
{
  CPUTimer timer;
  timer.start();
  int np = tcl_np.get();

  data.module=this;
  data.np=np;
  data.rhs=&rhs;
  data.lhs=&lhs;
  data.mat=matrix;
  data.timer=new WallClockTimer;
  data.stats=new PStats[data.np];
  Thread::parallel(Parallel<SolveMatrix>(this, &SolveMatrix::parallel_conjugate_gradient),
		   data.np, true);
  delete data.timer;
  delete data.stats;
//  delete data;
  timer.stop();
}

void SolveMatrix::parallel_conjugate_gradient(int processor)
{
  Matrix* matrix=data.mat;
  PStats* stats=&data.stats[processor];
  int size=matrix->nrows();
  
  int beg=processor*size/data.np;
  int end=(processor+1)*size/data.np;
  stats->flop=0;
  stats->memref=0;
  stats->gflop=0;
  stats->grefs=0;
  Array1<int> targetidx;
  Array1<double> targetlist;
  Array1<double> errlist;
  
  int last_update=0;
  
  int last_errupdate=0;
  
  if(processor == 0){
    data.timer->clear();
    data.timer->start();
    flops.set(0);
    floprate.set(0);
    memrefs.set(0);
    memrate.set(0);
    iteration.set(0);
    data.niter=0;
    data.toomany=maxiter.get();
    if(data.toomany == 0)
      data.toomany=2*size;

    if (data.rhs->vector_norm(stats->flop, stats->memref) < 0.0000001) {
	*data.lhs=*data.rhs;
	data.niter=data.toomany+1;
	data.reducer.wait(data.np);
	return;
    }
        
    data.diag=new ColumnMatrix(size);
    // We should try to do a better job at preconditioning...
    int i;
    
    for(i=0;i<size;i++){
      ColumnMatrix& diag=*data.diag;
      if (Abs(matrix->get(i,i)>0.000001))
	diag[i]=1./matrix->get(i,i);
      else
	diag[i]=1;
    }
    stats->flop+=size;
    stats->memref+=2*size*sizeof(double);
    data.R=new ColumnMatrix(size);
    ColumnMatrix& R=*data.R;
    ColumnMatrix& lhs=*data.lhs;

    matrix->mult(lhs, R, stats->flop, stats->memref);    
    
    ColumnMatrix& rhs=*data.rhs;
    Sub(R, rhs, R, stats->flop, stats->memref);
    data.bnorm=rhs.vector_norm(stats->flop, stats->memref);
    
    data.Z=new ColumnMatrix(size);
    ColumnMatrix& Z=*data.Z;
    matrix->mult(R, Z, stats->flop, stats->memref);
    
    data.P=new ColumnMatrix(size);
//     ColumnMatrix& P=*data.P;
    data.err=R.vector_norm(stats->flop, stats->memref)/data.bnorm;

    if(data.err == 0){
      lhs=rhs;
      stats->memref+=2*size*sizeof(double);
      data.niter=data.toomany+1;
      data.reducer.wait(data.np);
      return;
    } else {
	int ev=(data.err<1000000);
	if (!ev) data.err=1000000;
    }

    data.max_error=target_error.get();
    
    stats->gflop+=stats->flop/1000000000;
    stats->flop=stats->flop%1000000000;
    stats->grefs+=stats->memref/1000000000;
    stats->memref=stats->memref%1000000000;
    orig_error.set(data.err);
    current_error.set(to_string(data.err));
    
    double time=data.timer->time();
    flops.set(stats->gflop*1.e9+stats->flop);
    floprate.set((stats->gflop*1.e3+stats->flop*1.e-6)/time);
    memrefs.set(stats->grefs*1.e9+stats->memref);
    memrate.set((stats->grefs*1.e3+stats->memref*1.e-6)/time);
    
    TCL::execute(id+" reset_graph");
    errlist.add(data.err);
    targetidx.add(0);
    targetlist.add(data.max_error);
    
    append_values(1, errlist, last_update, targetidx, targetlist, last_errupdate);
  }
  double log_orig=log(data.err);
  double log_targ=log(data.max_error);
  data.reducer.wait(data.np);
  double err=data.err;
  double bkden=0;
  while(data.niter < data.toomany){
//     if(err < data.max_error)
//       break;
    
    ColumnMatrix& Z=*data.Z;
    ColumnMatrix& P=*data.P;
    if(processor==0){
//       data.niter++;
      double new_error;
     if(get_gui_doublevar(id, "target_error", new_error)
	 && new_error != data.max_error){
	targetidx.add(data.niter+1);
	targetlist.add(data.max_error);
	data.max_error=new_error;
      }
      targetidx.add(data.niter);
      targetlist.add(data.max_error);
    }
    data.reducer.wait(data.np);
    if(err < data.max_error)
      break;

    if (processor == 0 )
      data.niter++;
    
    // Simple Preconditioning...
    ColumnMatrix& diag=*data.diag;
    ColumnMatrix& R=*data.R;
    Mult(Z, R, diag, stats->flop, stats->memref, beg, end);
    
    // Calculate coefficient bk and direction vectors p and pp
    double my_bknum=Dot(Z, R, stats->flop, stats->memref, beg, end);
    double bknum=data.reducer.sum(processor, data.np, my_bknum);

    if(data.niter==1){
      Copy(P, Z, stats->flop, stats->memref, beg, end);
    } else {
      double bk=bknum/bkden;
      ScMult_Add(P, bk, P, Z, stats->flop, stats->memref, beg, end);
    }
    data.reducer.wait(data.np);
    // Calculate coefficient ak, new iterate x and new residuals r and rr


    matrix->mult(P, Z, stats->flop, stats->memref, beg, end);
    bkden=bknum;
    double my_akden=Dot(Z, P, stats->flop, stats->memref, beg, end);

    double akden=data.reducer.sum(processor, data.np, my_akden);
    
    double ak=bknum/akden;
    ColumnMatrix& lhs=*data.lhs;
    ScMult_Add(lhs, ak, P, lhs, stats->flop, stats->memref, beg, end);
    ScMult_Add(R, -ak, Z, R, stats->flop, stats->memref, beg, end);
    
    double my_err=R.vector_norm(stats->flop, stats->memref, beg, end)/data.bnorm;
    err=data.reducer.sum(processor, data.np, my_err);
    int ev=(err<1000000);
    if (!ev) err=1000000;


    stats->gflop+=stats->flop/1000000000;
    stats->flop=stats->flop%1000000000;
    stats->grefs+=stats->memref/1000000000;
    stats->memref=stats->memref%1000000000;
    
    if(processor == 0){
      errlist.add(err);
      
      stats->gflop+=stats->flop/1000000000;
      stats->flop=stats->flop%1000000000;
      stats->grefs+=stats->memref/1000000000;
      stats->memref=stats->memref%1000000000;
      
      if(data.niter == 1 || data.niter == 10 || data.niter%20 == 0){
	if(data.niter <= 60 || data.niter%60 == 0){
	  iteration.set(data.niter);
	  current_error.set(to_string(err));
	  double time=timer.time();
	  flops.set(14*stats->gflop*1.e9+stats->flop);
	  floprate.set(14*(stats->gflop*1.e3+stats->flop*1.e-6)/time);                    memrefs.set(14*stats->grefs*1.e9+stats->memref);
	  memrate.set(14*(stats->grefs*1.e3+stats->memref*1.e-6)/time);
	  append_values(data.niter, errlist, last_update, targetidx,
			targetlist, last_errupdate);
	  
	  if(err > 0){
	    double progress = (log_orig-log(err))/(log_orig-log_targ);
	    update_progress(progress);
	  }
	}

	if(ep && data.niter%60 == 0)
	  solport->send_intermediate(lhs.clone());

      }
    }
  }
  if(processor == 0){
    data.niter++;
    
    iteration.set(data.niter);
    current_error.set(to_string(err));
    data.timer->stop();
    double time=data.timer->time();
    flops.set(14*stats->gflop*1.e9+stats->flop);
    floprate.set(14*(stats->gflop*1.e3+stats->flop*1.e-6)/time);
    memrefs.set(14*stats->grefs*1.e9+stats->memref);
    memrate.set(14*(stats->grefs*1.e3+stats->memref*1.e-6)/time);
    
    TCL::execute(id+" finish_graph");
    append_values(data.niter, errlist, last_update, targetidx, targetlist,
		  last_errupdate);
    
  }
//  data.reducer.wait(data.np);
}

void 
SolveMatrix::bi_conjugate_gradient_sci(Matrix* matrix,
				       ColumnMatrix& lhs, ColumnMatrix& rhs)
{
  CPUTimer timer;
  timer.start();
  int np = tcl_np.get();
  Matrix *trans = matrix->transpose();

//  data=new CGData;
  data.module=this;
  data.np=np;
  data.rhs=&rhs;
  data.lhs=&lhs;
  data.mat=matrix;
  data.timer=new WallClockTimer;
  data.stats=new PStats[data.np];
  data.trans = trans;
  
//   int i,p;
  Thread::parallel(Parallel<SolveMatrix>(this, &SolveMatrix::parallel_bi_conjugate_gradient),
		   data.np, true);
  delete data.timer;
  delete data.stats;
//  delete data;
  timer.stop();
  remark("bi_cg done in " + to_string(timer.time()) + " seconds");
}


void SolveMatrix::parallel_bi_conjugate_gradient(int processor)
{
  Matrix* matrix=data.mat;
  PStats* stats=&data.stats[processor];
  int size=matrix->nrows();
  
  int beg=processor*size/data.np;
  int end=(processor+1)*size/data.np;
  if ( end > size ) end = size;
  stats->flop=0;
  stats->memref=0;
  stats->gflop=0;
  stats->grefs=0;
  Array1<int> targetidx;
  Array1<double> targetlist;
  Array1<double> errlist;
  
  int last_update=0;
  
  int last_errupdate=0;

  if(processor == 0){
    data.timer->clear();
    data.timer->start();
    flops.set(0);
    floprate.set(0);
    memrefs.set(0);
    memrate.set(0);
    iteration.set(0);
    
    data.diag=new ColumnMatrix(size);
    // We should try to do a better job at preconditioning...
    int i;
    
    ColumnMatrix& diag=*data.diag;
    for(i=0;i<size;i++){
      if (Abs(matrix->get(i,i)>0.000001))
	diag[i]=1./matrix->get(i,i);
      else
	diag[i]=1;
    }
    stats->flop+=size;
    stats->memref+=2*size*sizeof(double);
    data.R=new ColumnMatrix(size);
    ColumnMatrix& R=*data.R;
    ColumnMatrix& lhs=*data.lhs;
    matrix->mult(lhs, R, stats->flop, stats->memref);
    
    ColumnMatrix& rhs=*data.rhs;
    Sub(R, rhs, R, stats->flop, stats->memref);
    data.bnorm=rhs.vector_norm(stats->flop, stats->memref);
    
    // BiCG
    data.R1=new ColumnMatrix(size);
    ColumnMatrix& R1=*data.R1;
    Copy(R1, R, stats->flop, stats->memref, 0, size);
    
    data.Z=new ColumnMatrix(size);
    //         ColumnMatrix& Z=*data.Z;
    //         matrix->mult(R, Z, stats->flop, stats->memref);
    
    // BiCG ??
    data.Z1=new ColumnMatrix(size);
    //         ColumnMatrix& Z1=*data.Z1;
    //         matrix->mult(R, Z, stats->flop, stats->memref);

    data.P=new ColumnMatrix(size);
    //         ColumnMatrix& P=*data.P;
    
    // BiCG
    data.P1=new ColumnMatrix(size);
    //         ColumnMatrix& P1=*data.P1;
    
    data.err=R.vector_norm(stats->flop, stats->memref)/data.bnorm;
    if(data.err == 0){
      lhs=rhs;
      stats->memref+=2*size*sizeof(double);
      data.niter=data.toomany+1;
      data.reducer.wait(data.np);
      return;
    } else {
	int ev=(data.err<1000000);
	if (!ev) data.err=1000000;
    }
    
    data.niter=0;
    data.toomany=maxiter.get();
    if(data.toomany == 0)
      data.toomany=2*size;
    data.max_error=target_error.get();
    
    stats->gflop+=stats->flop/1000000000;
    stats->flop=stats->flop%1000000000;
    stats->grefs+=stats->memref/1000000000;
    stats->memref=stats->memref%1000000000;
    orig_error.set(data.err);
    current_error.set(to_string(data.err));
    
    double time=data.timer->time();
    flops.set(stats->gflop*1.e9+stats->flop);
    floprate.set((stats->gflop*1.e3+stats->flop*1.e-6)/time);
    memrefs.set(stats->grefs*1.e9+stats->memref);
    memrate.set((stats->grefs*1.e3+stats->memref*1.e-6)/time);
    
    TCL::execute(id+" reset_graph");
    errlist.add(data.err);
    targetidx.add(0);
    targetlist.add(data.max_error);
    
    append_values(1, errlist, last_update, targetidx, targetlist, last_errupdate);
  }
  double log_orig=log(data.err);
  double log_targ=log(data.max_error);
  data.reducer.wait(data.np);
  double err=data.err;
  double bkden=0;


  while(data.niter < data.toomany){
    ColumnMatrix& Z=*data.Z;
    ColumnMatrix& P=*data.P;
    // BiCG
    ColumnMatrix& Z1=*data.Z1;
    ColumnMatrix& P1=*data.P1;
    
    if(processor==0){
      double new_error;
      if(get_gui_doublevar(id, "target_error", new_error)
	 && new_error != data.max_error){
	targetidx.add(data.niter+1);
	targetlist.add(data.max_error);
	data.max_error=new_error;
      }
      targetidx.add(data.niter);
      targetlist.add(data.max_error);
    }
    data.reducer.wait(data.np);

    if(err < data.max_error)
      break;
    
    if ( processor == 0 )
      data.niter++;

    // Simple Preconditioning...
    ColumnMatrix& diag=*data.diag;
    ColumnMatrix& R=*data.R;
    Mult(Z, R, diag, stats->flop, stats->memref, beg, end);
    // BiCG
    ColumnMatrix& R1=*data.R1;
    Mult(Z1, R1, diag, stats->flop, stats->memref, beg, end);
    
    // Calculate coefficient bk and direction vectors p and pp
    // BiCG - change R->R1
    double my_bknum=Dot(Z, R1, stats->flop, stats->memref, beg, end);
    double bknum=data.reducer.sum(processor, data.np, my_bknum);
    
    // BiCG
    if ( bknum == 0 ) {
      //tol = 
      // max_iter = 
      // return 2
      break;
    }
    
    if(data.niter==1){
      Copy(P, Z, stats->flop, stats->memref, beg, end);
      // BiCG
      Copy(P1, Z1, stats->flop, stats->memref, beg, end);
    } else {
      double bk=bknum/bkden;
      ScMult_Add(P, bk, P, Z, stats->flop, stats->memref, beg, end);
      // BiCG
      ScMult_Add(P1, bk, P1, Z1, stats->flop, stats->memref, beg, end);
    }

    data.reducer.wait(data.np);

    // Calculate coefficient ak, new iterate x and new residuals r and rr
    matrix->mult(P, Z, stats->flop, stats->memref, beg, end);
    bkden=bknum;

    // BiCG
//     matrix->mult_transpose(P1, Z1, stats->flop, stats->memref, beg, end);
    data.trans->mult(P1, Z1, stats->flop, stats->memref, beg, end);

    // BiCG = change P -> P1
    double my_akden=Dot(Z, P1, stats->flop, stats->memref, beg, end);
    double akden=data.reducer.sum(processor, data.np, my_akden);

    double ak=bknum/akden;
    ColumnMatrix& lhs=*data.lhs;
    ScMult_Add(lhs, ak, P, lhs, stats->flop, stats->memref, beg, end);
    //        ColumnMatrix& rhs=*data.rhs;
    ScMult_Add(R, -ak, Z, R, stats->flop, stats->memref, beg, end);
    // BiCG
    ScMult_Add(R1, -ak, Z1, R1, stats->flop, stats->memref, beg, end);
    
    double my_err=R.vector_norm(stats->flop, stats->memref, beg, end)/data.bnorm;
    err=data.reducer.sum(processor, data.np, my_err);

    int ev=(err<1000000);
    if (!ev) err=1000000;

    stats->gflop+=stats->flop/1000000000;
    stats->flop=stats->flop%1000000000;
    stats->grefs+=stats->memref/1000000000;
    stats->memref=stats->memref%1000000000;
    
    if(processor == 0){
      errlist.add(err);
      
      stats->gflop+=stats->flop/1000000000;
      stats->flop=stats->flop%1000000000;
      stats->grefs+=stats->memref/1000000000;
      stats->memref=stats->memref%1000000000;
      
      if(data.niter == 1 || data.niter == 10 || data.niter%20 == 0){
	if(data.niter <= 60 || data.niter%60 == 0){
	  iteration.set(data.niter);
	  current_error.set(to_string(err));
	  double time=timer.time();
	  flops.set(14*stats->gflop*1.e9+stats->flop);
	  floprate.set(14*(stats->gflop*1.e3+stats->flop*1.e-6)/time);                    memrefs.set(14*stats->grefs*1.e9+stats->memref);
	  memrate.set(14*(stats->grefs*1.e3+stats->memref*1.e-6)/time);
	  append_values(data.niter, errlist, last_update, targetidx,
			targetlist, last_errupdate);
	  
	  if(err > 0){
	    double progress=(log_orig-log(err))/(log_orig-log_targ);                        
	    warning("err=" + to_string(err));
	    update_progress(progress);
	  }
	}
#ifdef yarden
	if(data.niter%60 == 0)
	  solport->send_intermediate(lhs.clone());
#endif
      }
    }
  }

  if(processor == 0){
    data.niter++;
    
    iteration.set(data.niter);
    current_error.set(to_string(err));
    data.timer->stop();
    double time=data.timer->time();
    flops.set(14*stats->gflop*1.e9+stats->flop);
    floprate.set(14*(stats->gflop*1.e3+stats->flop*1.e-6)/time);
    memrefs.set(14*stats->grefs*1.e9+stats->memref);
    memrate.set(14*(stats->grefs*1.e3+stats->memref*1.e-6)/time);
    remark("Done in " + to_string(time) + " seconds.");
    
    TCL::execute(id+" finish_graph");
    append_values(data.niter, errlist, last_update, targetidx, targetlist,
		  last_errupdate);
    
  }
  data.reducer.wait(data.np);
}

} // End namespace SCIRun
