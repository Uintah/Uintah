//static char *id="@(#) $Id$";

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

#ifdef SCI_SPARSELIB
#include "comprow_double.h"        //compressed row matrix storage
#include "iotext_double.h"        //matrix, vector input- output


#include "mvm.h"           //matrix definitions
#include "mvv.h"           //vector definitions
#include "mvblasd.h"       //vector multiplication BLASD


#include "icpre_double.h"         //preconditionars
#include "diagpre_double.h"
#include "ilupre_double.h"


#include "cg.h"                  //iterative IML  methods
#include "bicg.h"
#include "qmr.h"
#include "cgs.h"
#include "bicgstab.h"
#include "ir.h"
#include "gmres.h"
#endif



#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/ColumnMatrixPort.h>
#include <SCICore/Datatypes/SparseRowMatrix.h>
#include <PSECore/Datatypes/MatrixPort.h>
#include <PSECore/Datatypes/SurfacePort.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <SCICore/Thread/Parallel.h>
#include <SCICore/Thread/SimpleReducer.h>
#include <SCICore/Thread/Thread.h>
#include <iostream>
using std::cerr;
using std::endl;
#include <sstream>

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::GeomSpace;
using SCICore::Containers::to_string;
using SCICore::Thread::Parallel;
using SCICore::Thread::SimpleReducer;
using SCICore::Thread::Thread;

struct CGData;

class SolveMatrix : public Module {
  MatrixIPort* matrixport;
    ColumnMatrixIPort* rhsport;
    ColumnMatrixOPort* solport;
    ColumnMatrixHandle solution;
  
#ifdef SCI_SPARSELIB
    void conjugate_gradient(Matrix*, ColumnMatrix&, ColumnMatrix&,int flag);
    void quasi_minimal_res(Matrix*, ColumnMatrix&, ColumnMatrix&,int flag);
    void bi_conjugate_gradient(Matrix*, ColumnMatrix&, ColumnMatrix&,int flag);
    void bi_conjugate_gradient_stab(Matrix*, ColumnMatrix&, ColumnMatrix&,int flag);
    void conj_grad_squared(Matrix*, ColumnMatrix&, ColumnMatrix&,int flag);
    void gen_min_res_iter(Matrix*, ColumnMatrix&, ColumnMatrix&,int flag);
    void richardson_iter(Matrix*, ColumnMatrix&, ColumnMatrix&,int flag);
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
    SolveMatrix(const clString& id);
    virtual ~SolveMatrix();
    virtual void execute();
//     virtual void do_execute();

    TCLdouble target_error;
    TCLdouble flops;
    TCLdouble floprate;
    TCLdouble memrefs;
    TCLdouble memrate;
    TCLdouble orig_error;
    TCLstring current_error;
    TCLstring method;
    TCLstring precond;
    TCLint iteration;
    TCLint maxiter;
    TCLint use_previous_soln;
    TCLint emit_partial;
    int ep;
    TCLstring status;
    TCLint tcl_np;
    CGData* data;
};

Module* make_SolveMatrix(const clString& id) {
  return new SolveMatrix(id);
}


SolveMatrix::SolveMatrix(const clString& id)
: Module("SolveMatrix", id, Filter),
    target_error("target_error", id, this),
  flops("flops", id, this), floprate("floprate", id, this),
  memrefs("memrefs", id, this), memrate("memrate", id, this),
  orig_error("orig_error", id, this), current_error("current_error", id, this),
  method("method", id, this),precond("precond",id,this), iteration("iteration", id, this),
  maxiter("maxiter", id, this),
  use_previous_soln("use_previous_soln", id, this),
  emit_partial("emit_partial", id, this),status("status",id,this),
  tcl_np("np", id, this)
{
    matrixport=scinew MatrixIPort(this, "Matrix", MatrixIPort::Atomic);
    add_iport(matrixport);
    rhsport=scinew ColumnMatrixIPort(this, "RHS", ColumnMatrixIPort::Atomic);
    add_iport(rhsport);

    solport=scinew ColumnMatrixOPort(this, "Solution", ColumnMatrixIPort::Atomic);
    add_oport(solport);
}

SolveMatrix::~SolveMatrix()
{
}

void SolveMatrix::execute()
{
#ifdef SCI_SPARSELIB
 int flag = 1;
#endif
  MatrixHandle matrix;
  ColumnMatrixHandle rhs;
  
  int m = matrixport->get(matrix);
  int r = rhsport->get(rhs);
  
  if ( !r || !m ) {
    return;
  }
  
  if ( !matrix.get_rep() || !rhs.get_rep() ) {
    cerr << "Solve: no input\n";
    solport->send(ColumnMatrixHandle(0));
    return;
  }
  
  if(use_previous_soln.get() && solution.get_rep() && solution->nrows() == rhs->nrows()){
    solution.detach();
  } else {
    solution=scinew ColumnMatrix(rhs->nrows());
    solution->zero();
  }

  int size=matrix->nrows();
  if(matrix->ncols() != size){
    error(clString("Matrix should be square, but is ")
	  +to_string(size)+" x "+to_string(matrix->ncols()));
    return;
  }
  if(rhs->nrows() != size){
    error(clString("Matrix size mismatch"));
    return;
  }
  
#ifdef SCI_SPARSELIB
  clString pre=precond.get();
  if(pre == "Diag_P") flag = 1;
  else if(pre == "IC_P") flag = 2;
  else if(pre == "ILU_P") flag = 3;
#endif
  
  ep=emit_partial.get();
  cerr << "emit_partial="<<ep<<"\n";
  clString meth=method.get();
#ifdef SCI_SPARSELIB
    if(meth == "conjugate_gradient"){
      rhs.detach();
	conjugate_gradient(matrix.get_rep(),
			   *solution.get_rep(), *rhs.get_rep(),flag);
	solport->send(solution);
    } else if(meth == "quasi_minimal_res"){
	quasi_minimal_res(matrix.get_rep(),
	       *solution.get_rep(), *rhs.get_rep(),flag);
	solport->send(solution);
    } else if(meth == "bi_conjugate_gradient"){
	bi_conjugate_gradient(matrix.get_rep(),
	       *solution.get_rep(), *rhs.get_rep(),flag);
	solport->send(solution);
    } else if(meth == "bi_conjugate_gradient_stab"){
	bi_conjugate_gradient_stab(matrix.get_rep(),
	       *solution.get_rep(), *rhs.get_rep(),flag);
	solport->send(solution);

   } else if(meth == "conj_grad_squared"){
	conj_grad_squared(matrix.get_rep(),
	       *solution.get_rep(), *rhs.get_rep(),flag);
        solport->send(solution);
  } else if(meth == "gen_min_res_iter"){
	gen_min_res_iter(matrix.get_rep(),
	       *solution.get_rep(), *rhs.get_rep(),flag);
	solport->send(solution);

   } else if(meth == "richardson_iter"){
	richardson_iter(matrix.get_rep(),
	       *solution.get_rep(), *rhs.get_rep(),flag);
        solport->send(solution);
   } else 
#endif
   if(meth == "conjugate_gradient_sci"){
     conjugate_gradient_sci(matrix.get_rep(),
			    *solution.get_rep(), *rhs.get_rep());
//     if (ep)
//	 solport->send_intermediate(solution);
//     else
	 solport->send(solution);
     
   } else if(meth == "bi_conjugate_gradient_sci"){
     bi_conjugate_gradient_sci(matrix.get_rep(),
			       *solution.get_rep(), *rhs.get_rep());
     solport->send_intermediate(solution);
     
   } else if(meth == "jacoby_sci"){
     jacobi_sci(matrix.get_rep(),
		*solution.get_rep(), *rhs.get_rep());
     solport->send(solution);
     
     
     
   }else {
     error(clString("Unknown solution method: ")+meth);
   }
}


#if 0
void SolveMatrix::append_values(int, const Array1<double>&,
				int&, const Array1<int>&,
				const Array1<double>&,
				int&)
{
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

#ifdef SCI_SPARSELIB

//********************** IML++  **************************************************

void SolveMatrix::conjugate_gradient(Matrix* matrix,
				    ColumnMatrix& lhs, ColumnMatrix& rhs,int flag)
{
int result;  
int size = matrix->nrows();
int non_zero =  matrix->get_row()[size];


int maxit = maxiter.get();
double tol = target_error.get();
double x_init = 0.0;


status.set("Running"); 
  TCL::execute("update idletasks");
iteration.set(0); 
current_error.set(clString("0"));
  TCL::execute("update idletasks");


VECTOR_double b(rhs.get_rhs(),size);


VECTOR_double  x(size,x_init);

for(int i=0;i<size;i++)
   x[i] = lhs[i];



CompRow_Mat_double A(size,size,non_zero,matrix->get_val(),matrix->get_row(),matrix->get_col()); 
 

if (flag == 1){
     DiagPreconditioner_double D(A);
     result = CG(A,x,b,D,maxit,tol);  //IML++ solver 
}

else if(flag == 2){
     ICPreconditioner_double D(A);
     result = CG(A,x,b,D,maxit,tol);  //IML++ solver  
}

else if(flag ==3){
   CompRow_ILUPreconditioner_double D(A);
    result = CG(A,x,b,D,maxit,tol);  //IML++ solver
}


for(i=0;i<size;i++)
  lhs[i] = x[i];

//lhs.put_lhs(&x[0]);



if(result == 0)
  status.set("Done");
else
  status.set("Failed to Converge");

iteration.set(maxit);
current_error.set(to_string(tol));	      
 TCL::execute("update idletasks");

}


void SolveMatrix::bi_conjugate_gradient(Matrix* matrix,
					ColumnMatrix& lhs, ColumnMatrix& rhs,int flag)
{
  int result; 
  int size = matrix->nrows();
  
  int non_zero =  matrix->get_row()[size];
  
  int maxit = maxiter.get();
  double tol = target_error.get();
  double x_init = 0.0;
  
  status.set("Running");
  TCL::execute("update idletasks");
  iteration.set(0);
  current_error.set(to_string(0));
  TCL::execute("update idletasks");
  
  
  VECTOR_double b(rhs.get_rhs(),size);
  
  VECTOR_double  x(size,x_init);
  
  for(int i=0;i<size;i++)
    x[i] = lhs[i];
  
  
  CompRow_Mat_double A(size,size,non_zero,matrix->get_val(),matrix->get_row(),matrix->get_col()); 
  
  
  if (flag == 1){
    DiagPreconditioner_double D(A);
    result = BiCG(A,x,b,D,maxit,tol);  //IML++ solver 
  }
  
  else if(flag == 2){
    ICPreconditioner_double D(A);
    result = BiCG(A,x,b,D,maxit,tol);  //IML++ solver  
  }
  
  else if(flag ==3){
    CompRow_ILUPreconditioner_double D(A);
    result = BiCG(A,x,b,D,maxit,tol);  //IML++ solver
  }
  
  //lhs.put_lhs(&x[0]);
  
  for(i=0;i<size;i++)
    lhs[i] = x[i];
  
  if(result == 0)
    status.set("Done");
  else
    status.set("Failed to Converge");
  
  iteration.set(maxit);
  current_error.set(to_string(tol));	   
  TCL::execute("update idletasks");
  
}


void SolveMatrix::quasi_minimal_res(Matrix* matrix,
				    ColumnMatrix& lhs, ColumnMatrix& rhs,int flag)
{
int result; 
int size = matrix->nrows();

int non_zero =  matrix->get_row()[size];

int maxit = maxiter.get();
double tol = target_error.get();
double x_init = 0.0;


status.set("Running");
  TCL::execute("update idletasks");
iteration.set(0);
current_error.set(to_string(0));
 TCL::execute("update idletasks");

VECTOR_double b(rhs.get_rhs(),size);

VECTOR_double  x(size,x_init);

for(int i=0;i<size;i++)
   x[i] = lhs[i];

CompRow_Mat_double A(size,size,non_zero,matrix->get_val(),matrix->get_row(),matrix->get_col()); 


if (flag == 1){
     DiagPreconditioner_double D(A);
     result = QMR(A,x,b,D,D,maxit,tol);  //IML++ solver 
}

else if(flag == 2){
     ICPreconditioner_double D(A);
     result = QMR(A,x,b,D,D,maxit,tol);  //IML++ solver  
}

else if(flag ==3){
   CompRow_ILUPreconditioner_double D(A);
    result = QMR(A,x,b,D,D,maxit,tol);  //IML++ solver
}

//lhs.put_lhs(&x[0]);

for(i=0;i<size;i++)
   lhs[i] = x[i];


if(result == 0)
  status.set("Done");
else
  status.set("Failed to Converge");

iteration.set(maxit);
current_error.set(to_string(tol));	   

 TCL::execute("update idletasks");

}



void SolveMatrix::bi_conjugate_gradient_stab(Matrix* matrix,
					     ColumnMatrix& lhs, ColumnMatrix& rhs,int flag)
{
  int result; 
  int size = matrix->nrows();
  
  int non_zero =  matrix->get_row()[size];
  int maxit = maxiter.get();
  double tol = target_error.get();
  double x_init = 0.0;
  
  status.set("Running");
  TCL::execute("update idletasks");
  iteration.set(0);
  current_error.set(to_string(0));
  TCL::execute("update idletasks");
  
  VECTOR_double b(rhs.get_rhs(),size);
  
  VECTOR_double  x(size,x_init);
  
  for(int i=0;i<size;i++)
    x[i] = lhs[i];
  
  CompRow_Mat_double A(size,size,non_zero,matrix->get_val(),matrix->get_row(),matrix->get_col()); 
  
  
  
  if (flag == 1){
    DiagPreconditioner_double D(A);
    result = BiCGSTAB(A,x,b,D,maxit,tol);  //IML++ solver 
  }
  
  else if(flag == 2){
    ICPreconditioner_double D(A);
     result = BiCGSTAB(A,x,b,D,maxit,tol);  //IML++ solver  
}

else if(flag ==3){
   CompRow_ILUPreconditioner_double D(A);
    result = BiCGSTAB(A,x,b,D,maxit,tol);  //IML++ solver
}

//lhs.put_lhs(&x[0]);

for(i=0;i<size;i++)
   lhs[i] = x[i];


if(result == 0)
  status.set("Done");
else
  status.set("Failed to Converge");

iteration.set(maxit);
current_error.set(to_string(tol));	   

 TCL::execute("update idletasks");

}


void SolveMatrix::conj_grad_squared(Matrix* matrix,
				    ColumnMatrix& lhs, ColumnMatrix& rhs,int flag)
{
int result; 
int size = matrix->nrows();

int non_zero =  matrix->get_row()[size];
int maxit = maxiter.get();
double tol = target_error.get();
double x_init = 0.0;

status.set("Running");
TCL::execute("update idletasks");
iteration.set(0);
current_error.set(to_string(0));
 TCL::execute("update idletasks");

VECTOR_double b(rhs.get_rhs(),size);
VECTOR_double  x(size,x_init);

for(int i=0;i<size;i++)
   x[i] = lhs[i];

CompRow_Mat_double A(size,size,non_zero,matrix->get_val(),matrix->get_row(),matrix->get_col()); 


if (flag == 1){
     DiagPreconditioner_double D(A);
     result = CGS(A,x,b,D,maxit,tol);  //IML++ solver 
}

else if(flag == 2){
     ICPreconditioner_double D(A);
     result = CGS(A,x,b,D,maxit,tol);  //IML++ solver  
}

else if(flag ==3){
   CompRow_ILUPreconditioner_double D(A);
    result = CGS(A,x,b,D,maxit,tol);  //IML++ solver
}

//lhs.put_lhs(&x[0]);

for(i=0;i<size;i++)
   lhs[i] = x[i];

if(result == 0)
  status.set("Done");
else
  status.set("Failed to Converge");

iteration.set(maxit);
current_error.set(to_string(tol));	   
 TCL::execute("update idletasks");
}

 
void SolveMatrix::gen_min_res_iter(Matrix* matrix,
				    ColumnMatrix& lhs, ColumnMatrix& rhs,int flag)
{
int result; 
int restart = 32;
int size = matrix->nrows();

int non_zero =  matrix->get_row()[size];
int maxit = maxiter.get();
double tol = target_error.get();
double x_init = 0.0;

status.set("Running");
TCL::execute("update idletasks");
iteration.set(0);
current_error.set(to_string(0));
 TCL::execute("update idletasks");

VECTOR_double b(rhs.get_rhs(),size);
VECTOR_double  x(size,x_init);

MATRIX_double H(restart+1,restart,0.0);

for(int i=0;i<size;i++)
   x[i] = lhs[i];

CompRow_Mat_double A(size,size,non_zero,matrix->get_val(),matrix->get_row(),matrix->get_col()); 


if (flag == 1){
     DiagPreconditioner_double D(A);
     result = GMRES(A,x,b,D,H,restart,maxit,tol);  //IML++ solver 
}

else if(flag == 2){
     ICPreconditioner_double D(A);
     result = GMRES(A,x,b,D,H,restart,maxit,tol);  //IML++ solver  
}

else if(flag ==3){
   CompRow_ILUPreconditioner_double D(A);
    result = GMRES(A,x,b,D,H,restart,maxit,tol);  //IML++ solver
}

//lhs.put_lhs(&x[0]);

for(i=0;i<size;i++)
   lhs[i] = x[i];

if(result == 0)
  status.set("Done");
else
  status.set("Failed to Converge");

iteration.set(maxit);
current_error.set(to_string(tol));	   
 TCL::execute("update idletasks");

}


void SolveMatrix::richardson_iter(Matrix* matrix,
				    ColumnMatrix& lhs, ColumnMatrix& rhs,int flag)
{
int result; 
int size = matrix->nrows();

int non_zero =  matrix->get_row()[size];
int maxit = maxiter.get();
double tol = target_error.get();
double x_init = 0.0;

status.set("Running");
TCL::execute("update idletasks");
iteration.set(0);
current_error.set(to_string(0));
 TCL::execute("update idletasks");

VECTOR_double b(rhs.get_rhs(),size);
VECTOR_double  x(size,x_init);

for(int i=0;i<size;i++)
   x[i] = lhs[i];

CompRow_Mat_double A(size,size,non_zero,matrix->get_val(),matrix->get_row(),matrix->get_col()); 


if (flag == 1){
     DiagPreconditioner_double D(A);
     result = IR(A,x,b,D,maxit,tol);  //IML++ solver 
}

else if(flag == 2){
     ICPreconditioner_double D(A);
     result = IR(A,x,b,D,maxit,tol);  //IML++ solver  
}

else if(flag ==3){
   CompRow_ILUPreconditioner_double D(A);
    result = IR(A,x,b,D,maxit,tol);  //IML++ solver
}

//lhs.put_lhs(&x[0]);

for(i=0;i<size;i++)
   lhs[i] = x[i];

if(result == 0)
  status.set("Done");
else
  status.set("Failed to Converge");

iteration.set(maxit);
current_error.set(to_string(tol));	   
 TCL::execute("update idletasks");

}
//**********************End IML++ *********************************************


#endif 



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
	invdiag[i]=1./matrix->get(i,i);
    }
    flop+=size;
    memref=2*size*sizeof(double);

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
	if(get_tcl_doublevar(id, "target_error", new_error)
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

struct PStats {
    int flop;
    int memref;
    int gflop;
    int grefs;
    int pad[28];
};

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
  //
  double max_error;
  SimpleReducer reducer;
  int np;
  PStats* stats;
  double err;
  double bnorm;
  CGData();
};

CGData::CGData()
    : reducer("SolveMatrix reduction barrier")
{
}

void SolveMatrix::conjugate_gradient_sci(Matrix* matrix,
					 ColumnMatrix& lhs, ColumnMatrix& rhs)
{
  cerr << "cg started\n";
  CPUTimer timer;
  timer.start();
  int np = tcl_np.get();
  //     int np=Task::nprocessors();
  cerr << "np=" << np << endl;
  data=new CGData;
  data->module=this;
  data->np=np;
  data->rhs=&rhs;
  data->lhs=&lhs;
  data->mat=matrix;
  data->timer=new WallClockTimer;
  data->stats=new PStats[data->np];
  Thread::parallel(Parallel<SolveMatrix>(this, &SolveMatrix::parallel_conjugate_gradient),
		   data->np, true);
  delete data->timer;
  delete data->stats;
  delete data;
  timer.stop();
  cerr << "cg done: " << timer.time() << " seconds\n";
}

void SolveMatrix::conjugate_gradient_netsolve(Matrix* matrix,
					 ColumnMatrix& lhs, ColumnMatrix& rhs)
{
  cerr << "netsolve-enabled cg started\n";
  CPUTimer timer;
  timer.start();
  int np = tcl_np.get();
  //     int np=Task::nprocessors();
  cerr << "np=" << np << endl;
  data=new CGData;
  data->module=this;
  data->np=np;
  data->rhs=&rhs;
  data->lhs=&lhs;
  data->mat=matrix;
  data->timer=new WallClockTimer;
  data->stats=new PStats[data->np];
  Thread::parallel(Parallel<SolveMatrix>(this, &SolveMatrix::parallel_conjugate_gradient),
		   data->np, true);
  delete data->timer;
  delete data->stats;
  delete data;
  timer.stop();
  cerr << "cg done: " << timer.time() << " seconds\n";
}

void SolveMatrix::parallel_conjugate_gradient(int processor)
{
  Matrix* matrix=data->mat;
  PStats* stats=&data->stats[processor];
  int size=matrix->nrows();
  
  int beg=processor*size/data->np;
  int end=(processor+1)*size/data->np;
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
    data->timer->clear();
    data->timer->start();
    flops.set(0);
    floprate.set(0);
    memrefs.set(0);
    memrate.set(0);
    iteration.set(0);

    if (data->rhs->vector_norm(stats->flop, stats->memref) < 0.0000001) {
	*data->lhs=*data->rhs;
	return;
    }
        
    data->diag=new ColumnMatrix(size);
    // We should try to do a better job at preconditioning...
    int i;
    
    for(i=0;i<size;i++){
      ColumnMatrix& diag=*data->diag;
      diag[i]=1./matrix->get(i,i);
    }
    stats->flop+=size;
    stats->memref+=2*size*sizeof(double);
    data->R=new ColumnMatrix(size);
    ColumnMatrix& R=*data->R;
    ColumnMatrix& lhs=*data->lhs;

    matrix->mult(lhs, R, stats->flop, stats->memref);    
    
    ColumnMatrix& rhs=*data->rhs;
    Sub(R, rhs, R, stats->flop, stats->memref);
    data->bnorm=rhs.vector_norm(stats->flop, stats->memref);
    
    data->Z=new ColumnMatrix(size);
    ColumnMatrix& Z=*data->Z;
    matrix->mult(R, Z, stats->flop, stats->memref);
    
    data->P=new ColumnMatrix(size);
//     ColumnMatrix& P=*data->P;
    data->err=R.vector_norm(stats->flop, stats->memref)/data->bnorm;

    if(data->err == 0){
      lhs=rhs;
      stats->memref+=2*size*sizeof(double);
      return;
    } else {
	int ev=(data->err<1000000);
//	cerr << "EVALUATING "<<ev<<"\n";
	if (!ev) data->err=1000000;
    }

    data->niter=0;
    data->toomany=maxiter.get();
    if(data->toomany == 0)
      data->toomany=2*size;
    data->max_error=target_error.get();
    
    stats->gflop+=stats->flop/1000000000;
    stats->flop=stats->flop%1000000000;
    stats->grefs+=stats->memref/1000000000;
    stats->memref=stats->memref%1000000000;
    orig_error.set(data->err);
    current_error.set(to_string(data->err));
    
    double time=data->timer->time();
    flops.set(stats->gflop*1.e9+stats->flop);
    floprate.set((stats->gflop*1.e3+stats->flop*1.e-6)/time);
    memrefs.set(stats->grefs*1.e9+stats->memref);
    memrate.set((stats->grefs*1.e3+stats->memref*1.e-6)/time);
    
    TCL::execute(id+" reset_graph");
    errlist.add(data->err);
    targetidx.add(0);
    targetlist.add(data->max_error);
    
    append_values(1, errlist, last_update, targetidx, targetlist, last_errupdate);
  }
  double log_orig=log(data->err);
  double log_targ=log(data->max_error);
  data->reducer.wait(data->np);
  double err=data->err;
  double bkden=0;
  while(data->niter < data->toomany){
//     if(err < data->max_error)
//       break;
    
    ColumnMatrix& Z=*data->Z;
    ColumnMatrix& P=*data->P;
    if(processor==0){
//       data->niter++;
      double new_error;
     if(get_tcl_doublevar(id, "target_error", new_error)
	 && new_error != data->max_error){
	targetidx.add(data->niter+1);
	targetlist.add(data->max_error);
	data->max_error=new_error;
      }
      targetidx.add(data->niter);
      targetlist.add(data->max_error);
    }
    data->reducer.wait(data->np);
    if(err < data->max_error)
      break;

    if (processor == 0 )
      data->niter++;
    
    // Simple Preconditioning...
    ColumnMatrix& diag=*data->diag;
    ColumnMatrix& R=*data->R;
    Mult(Z, R, diag, stats->flop, stats->memref, beg, end);
    
    // Calculate coefficient bk and direction vectors p and pp
    double my_bknum=Dot(Z, R, stats->flop, stats->memref, beg, end);
    double bknum=data->reducer.sum(processor, data->np, my_bknum);

//    if (processor==0) cerr << "bknum="<<bknum<<"\n";
    
    if(data->niter==1){
      Copy(P, Z, stats->flop, stats->memref, beg, end);
    } else {
      double bk=bknum/bkden;
      ScMult_Add(P, bk, P, Z, stats->flop, stats->memref, beg, end);
    }
    data->reducer.wait(data->np);
    // Calculate coefficient ak, new iterate x and new residuals r and rr

#if 0
    if (processor==2) {
	cerr << "P=";
	for (int iii=beg; iii<end; iii++) cerr << " "<<P[iii];
	cerr << "\n";
    }
#endif

    matrix->mult(P, Z, stats->flop, stats->memref, beg, end);
    bkden=bknum;
    double my_akden=Dot(Z, P, stats->flop, stats->memref, beg, end);
//    cerr << "p="<<processor<<" my_akden="<<my_akden<<"\n";

    double akden=data->reducer.sum(processor, data->np, my_akden);
    
    double ak=bknum/akden;
//    if (processor == 0) cerr << "ak="<<ak<<"  akden="<<akden<<"\n";
    ColumnMatrix& lhs=*data->lhs;
    ScMult_Add(lhs, ak, P, lhs, stats->flop, stats->memref, beg, end);
//     ColumnMatrix& rhs=*data->rhs;
    ScMult_Add(R, -ak, Z, R, stats->flop, stats->memref, beg, end);
    
    double my_err=R.vector_norm(stats->flop, stats->memref, beg, end)/data->bnorm;
    err=data->reducer.sum(processor, data->np, my_err);
//    if (processor==0) cerr << "err="<<err<<"\n";
    int ev=(err<1000000);
//    cerr << "EVALUATING2 "<<ev<<"\n";
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
      
      if(data->niter == 1 || data->niter == 10 || data->niter%20 == 0){
	if(data->niter <= 60 || data->niter%60 == 0){
	  iteration.set(data->niter);
	  current_error.set(to_string(err));
	  double time=timer.time();
	  flops.set(14*stats->gflop*1.e9+stats->flop);
	  floprate.set(14*(stats->gflop*1.e3+stats->flop*1.e-6)/time);                    memrefs.set(14*stats->grefs*1.e9+stats->memref);
	  memrate.set(14*(stats->grefs*1.e3+stats->memref*1.e-6)/time);
	  append_values(data->niter, errlist, last_update, targetidx,
			targetlist, last_errupdate);
	  
	  if(err > 0){
	    double progress=(log_orig-log(err))/(log_orig-log_targ);                        
	    cerr << "err=" << err << endl;
	    //                         cerr << "log_orig=" << log_orig << endl;
	    update_progress(progress);
	  }
	}

	if(ep && data->niter%60 == 0)
	  solport->send_intermediate(lhs.clone());

      }
    }
  }
  if(processor == 0){
    data->niter++;
    
    iteration.set(data->niter);
    current_error.set(to_string(err));
    data->timer->stop();
    double time=data->timer->time();
    flops.set(14*stats->gflop*1.e9+stats->flop);
    floprate.set(14*(stats->gflop*1.e3+stats->flop*1.e-6)/time);
    memrefs.set(14*stats->grefs*1.e9+stats->memref);
    memrate.set(14*(stats->grefs*1.e3+stats->memref*1.e-6)/time);
    cerr << "Done in " << time << " seconds\n";
    
    TCL::execute(id+" finish_graph");
    append_values(data->niter, errlist, last_update, targetidx, targetlist,
		  last_errupdate);
    
  }
}

void 
SolveMatrix::bi_conjugate_gradient_sci(Matrix* matrix,
				       ColumnMatrix& lhs, ColumnMatrix& rhs)
{
  cerr << "bi_cg started\n";
  CPUTimer timer;
  timer.start();
  int np = tcl_np.get();
  SparseRowMatrix *trans = scinew SparseRowMatrix;
  //  trans->transpose( *matrix->getSparseRow() );

  data=new CGData;
  data->module=this;
  data->np=np;
  data->rhs=&rhs;
  data->lhs=&lhs;
  data->mat=matrix;
  data->timer=new WallClockTimer;
  data->stats=new PStats[data->np];
  data->trans = trans;
  
//   int i,p;
  Thread::parallel(Parallel<SolveMatrix>(this, &SolveMatrix::parallel_bi_conjugate_gradient),
		   data->np, true);
  delete data->timer;
  delete data->stats;
  delete data;
  timer.stop();
  cerr << "bi_cg done: " << timer.time() << " seconds\n";
}


void SolveMatrix::parallel_bi_conjugate_gradient(int processor)
{
#ifdef PRINT
  //if ( processor == 0)
    printf("BiCG[%d]: %d\n",processor, getpid());
#endif
  Matrix* matrix=data->mat;
  PStats* stats=&data->stats[processor];
  int size=matrix->nrows();
  
  int beg=processor*size/data->np;
  int end=(processor+1)*size/data->np;
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
    data->timer->clear();
    data->timer->start();
    flops.set(0);
    floprate.set(0);
    memrefs.set(0);
    memrate.set(0);
    iteration.set(0);
    
    data->diag=new ColumnMatrix(size);
    // We should try to do a better job at preconditioning...
    int i;
    
    ColumnMatrix& diag=*data->diag;
    for(i=0;i<size;i++){
      diag[i]=1./matrix->get(i,i);
    }
    stats->flop+=size;
    stats->memref+=2*size*sizeof(double);
    data->R=new ColumnMatrix(size);
    ColumnMatrix& R=*data->R;
    ColumnMatrix& lhs=*data->lhs;
    matrix->mult(lhs, R, stats->flop, stats->memref);
    
    ColumnMatrix& rhs=*data->rhs;
    Sub(R, rhs, R, stats->flop, stats->memref);
    data->bnorm=rhs.vector_norm(stats->flop, stats->memref);
    
    // BiCG
    data->R1=new ColumnMatrix(size);
    ColumnMatrix& R1=*data->R1;
    Copy(R1, R, stats->flop, stats->memref, 0, size);
    //
    
    data->Z=new ColumnMatrix(size);
    //         ColumnMatrix& Z=*data->Z;
    //         matrix->mult(R, Z, stats->flop, stats->memref);
    
    // BiCG ??
    data->Z1=new ColumnMatrix(size);
    //         ColumnMatrix& Z1=*data->Z1;
    //         matrix->mult(R, Z, stats->flop, stats->memref);
	//

    data->P=new ColumnMatrix(size);
    //         ColumnMatrix& P=*data->P;
    
    // BiCG
    data->P1=new ColumnMatrix(size);
    //         ColumnMatrix& P1=*data->P1;
    //
    
    data->err=R.vector_norm(stats->flop, stats->memref)/data->bnorm;
    if(data->err == 0){
      lhs=rhs;
      stats->memref+=2*size*sizeof(double);
      return;
    } else {
	int ev=(data->err<1000000);
//	cerr << "EVALUATING "<<ev<<"\n";
	if (!ev) data->err=1000000;
    }
    
    data->niter=0;
    data->toomany=maxiter.get();
    if(data->toomany == 0)
      data->toomany=2*size;
    data->max_error=target_error.get();
    
    stats->gflop+=stats->flop/1000000000;
    stats->flop=stats->flop%1000000000;
    stats->grefs+=stats->memref/1000000000;
    stats->memref=stats->memref%1000000000;
    orig_error.set(data->err);
    current_error.set(to_string(data->err));
    
    double time=data->timer->time();
    flops.set(stats->gflop*1.e9+stats->flop);
    floprate.set((stats->gflop*1.e3+stats->flop*1.e-6)/time);
    memrefs.set(stats->grefs*1.e9+stats->memref);
    memrate.set((stats->grefs*1.e3+stats->memref*1.e-6)/time);
    
    TCL::execute(id+" reset_graph");
    errlist.add(data->err);
    targetidx.add(0);
    targetlist.add(data->max_error);
    
    append_values(1, errlist, last_update, targetidx, targetlist, last_errupdate);
  }
  double log_orig=log(data->err);
  double log_targ=log(data->max_error);
  data->reducer.wait(data->np);
  double err=data->err;
  double bkden=0;


  while(data->niter < data->toomany){
    ColumnMatrix& Z=*data->Z;
    ColumnMatrix& P=*data->P;
    // BiCG
    ColumnMatrix& Z1=*data->Z1;
    ColumnMatrix& P1=*data->P1;
    //
    
    if(processor==0){
      double new_error;
      if(get_tcl_doublevar(id, "target_error", new_error)
	 && new_error != data->max_error){
	targetidx.add(data->niter+1);
	targetlist.add(data->max_error);
	data->max_error=new_error;
      }
      targetidx.add(data->niter);
      targetlist.add(data->max_error);
    }
    data->reducer.wait(data->np);

    if(err < data->max_error)
      break;
    
    if ( processor == 0 )
      data->niter++;

    // Simple Preconditioning...
    ColumnMatrix& diag=*data->diag;
    ColumnMatrix& R=*data->R;
    Mult(Z, R, diag, stats->flop, stats->memref, beg, end);
    // BiCG
    ColumnMatrix& R1=*data->R1;
    Mult(Z1, R1, diag, stats->flop, stats->memref, beg, end);
    
    // Calculate coefficient bk and direction vectors p and pp
    // BiCG - change R->R1
    double my_bknum=Dot(Z, R1, stats->flop, stats->memref, beg, end);
    double bknum=data->reducer.sum(processor, data->np, my_bknum);
    
    // BiCG
    if ( bknum == 0 ) {
      //tol = 
      // max_iter = 
      // return 2
      printf("BiCG[%d]: bknum == 0\n", processor);
      break;
    }
    //
    
    if(data->niter==1){
      Copy(P, Z, stats->flop, stats->memref, beg, end);
      // BiCG
      Copy(P1, Z1, stats->flop, stats->memref, beg, end);
      //
    } else {
      double bk=bknum/bkden;
      ScMult_Add(P, bk, P, Z, stats->flop, stats->memref, beg, end);
      // BiCG
      ScMult_Add(P1, bk, P1, Z1, stats->flop, stats->memref, beg, end);
      //
    }

    data->reducer.wait(data->np);

    // Calculate coefficient ak, new iterate x and new residuals r and rr
    matrix->mult(P, Z, stats->flop, stats->memref, beg, end);
    bkden=bknum;

    // BiCG
//     matrix->mult_transpose(P1, Z1, stats->flop, stats->memref, beg, end);
    data->trans->mult(P1, Z1, stats->flop, stats->memref, beg, end);

    // BiCG = change P -> P1
    double my_akden=Dot(Z, P1, stats->flop, stats->memref, beg, end);
    //
    double akden=data->reducer.sum(processor, data->np, my_akden);

    double ak=bknum/akden;
    ColumnMatrix& lhs=*data->lhs;
    ScMult_Add(lhs, ak, P, lhs, stats->flop, stats->memref, beg, end);
    //        ColumnMatrix& rhs=*data->rhs;
    ScMult_Add(R, -ak, Z, R, stats->flop, stats->memref, beg, end);
    // BiCG
    ScMult_Add(R1, -ak, Z1, R1, stats->flop, stats->memref, beg, end);
    //
    
    double my_err=R.vector_norm(stats->flop, stats->memref, beg, end)/data->bnorm;
    err=data->reducer.sum(processor, data->np, my_err);

    int ev=(err<1000000);
//    cerr << "EVALUATING2 "<<ev<<"\n";
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
      
      if(data->niter == 1 || data->niter == 10 || data->niter%20 == 0){
	if(data->niter <= 60 || data->niter%60 == 0){
	  iteration.set(data->niter);
	  current_error.set(to_string(err));
	  double time=timer.time();
	  flops.set(14*stats->gflop*1.e9+stats->flop);
	  floprate.set(14*(stats->gflop*1.e3+stats->flop*1.e-6)/time);                    memrefs.set(14*stats->grefs*1.e9+stats->memref);
	  memrate.set(14*(stats->grefs*1.e3+stats->memref*1.e-6)/time);
	  append_values(data->niter, errlist, last_update, targetidx,
			targetlist, last_errupdate);
	  
	  if(err > 0){
	    double progress=(log_orig-log(err))/(log_orig-log_targ);                        
	    cerr << "err=" << err << endl;
	    //                         cerr << "log_orig=" << log_orig << endl;
	    update_progress(progress);
	  }
	}
#ifdef yarden
	if(data->niter%60 == 0)
	  solport->send_intermediate(lhs.clone());
#endif
      }
    }
  }

  if(processor == 0){
    data->niter++;
    
    iteration.set(data->niter);
    current_error.set(to_string(err));
    data->timer->stop();
    double time=data->timer->time();
    flops.set(14*stats->gflop*1.e9+stats->flop);
    floprate.set(14*(stats->gflop*1.e3+stats->flop*1.e-6)/time);
    memrefs.set(14*stats->grefs*1.e9+stats->memref);
    memrate.set(14*(stats->grefs*1.e3+stats->memref*1.e-6)/time);
    cerr << "Done in " << time << " seconds\n";
    
    TCL::execute(id+" finish_graph");
    append_values(data->niter, errlist, last_update, targetidx, targetlist,
		  last_errupdate);
    
  }
}

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.1  2000/01/21 16:26:54  moulding
// initial import of netsolve stuff
//
// Revision 1.12  1999/10/07 02:06:52  sparker
// use standard iostreams and complex type
//
// Revision 1.11  1999/09/22 04:06:15  dmw
// removed debug info
//
// Revision 1.10  1999/09/21 23:20:10  dmw
// fixed it so it works for RHS==0
//
// Revision 1.9  1999/09/16 00:38:11  dmw
// fixed TCL files for SurfToGeom and SolveMatrix and added SurfToGeom to the Makefile
//
// Revision 1.8  1999/09/08 02:26:34  sparker
// Various #include cleanups
//
// Revision 1.7  1999/08/29 00:46:40  sparker
// Integrated new thread library
// using statement tweaks to compile with both MipsPRO and g++
// Thread library bug fixes
//
// Revision 1.6  1999/08/25 03:47:51  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.5  1999/08/19 23:17:48  sparker
// Removed a bunch of #include <SCICore/Util/NotFinished.h> statements
// from files that did not need them.
//
// Revision 1.4  1999/08/19 05:30:52  sparker
// Configuration updates:
//  - renamed config.h to sci_config.h
//  - also uses sci_defs.h, since I couldn't get it to substitute vars in
//    sci_config.h
//  - Added flags for --enable-scirun, --enable-uintah, and
//    --enable-davew, to build the specific package set.  More than one
//    can be specified, and at least one must be present.
//  - Added a --enable-parallel, to build the new parallel version.
//    Doesn't do much yet.
//  - Made construction of config.h a little bit more general
//
// Revision 1.3  1999/08/18 20:19:45  sparker
// Eliminated copy constructor and clone in all modules
// Added a private copy ctor and a private clone method to Module so
//  that future modules will not compile until they remvoe the copy ctor
//  and clone method
// Added an ASSERTFAIL macro to eliminate the "controlling expression is
//  constant" warnings.
// Eliminated other miscellaneous warnings
//
// Revision 1.2  1999/08/17 06:37:31  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:57:46  mcq
// Initial commit
//
// Revision 1.2  1999/04/27 22:57:50  dav
// updates in Modules for Datatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:31  dav
// Import sources
//
//
