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

#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Datatypes/ColumnMatrixPort.h>
#include <Datatypes/MatrixPort.h>
#include <Datatypes/SurfacePort.h>
#include <Geometry/Point.h>
#include <Malloc/Allocator.h>
#include <TCL/TCLvar.h>
#include <strstream.h>

#include <Multitask/Task.h>

class SolveMatrix : public Module {
    MatrixIPort* matrixport;
    ColumnMatrixIPort* rhsport;
    ColumnMatrixOPort* solport;
    ColumnMatrixHandle solution;
    void conjugate_gradient(Matrix*, ColumnMatrix&, ColumnMatrix&);
    void jacobi(Matrix*, ColumnMatrix&, ColumnMatrix&);

    void append_values(int niter, const Array1<double>& errlist,
		       int& last_update, const Array1<int>& targetidx,
		       const Array1<double>& targetlist,
		       int& last_errupdate);
public:
    SolveMatrix(const clString& id);
    SolveMatrix(const SolveMatrix&, int deep);
    virtual ~SolveMatrix();
    virtual Module* clone(int deep);
    virtual void execute();

    TCLdouble target_error;
    TCLdouble flops;
    TCLdouble floprate;
    TCLdouble memrefs;
    TCLdouble memrate;
    TCLdouble orig_error;
    TCLdouble current_error;
    TCLstring method;
    TCLint iteration;
    TCLint maxiter;
    TCLint use_previous_soln;
};

extern "C" {
Module* make_SolveMatrix(const clString& id)
{
    return scinew SolveMatrix(id);
}
};

SolveMatrix::SolveMatrix(const clString& id)
: Module("SolveMatrix", id, Filter), target_error("target_error", id, this),
  flops("flops", id, this), floprate("floprate", id, this),
  memrefs("memrefs", id, this), memrate("memrate", id, this),
  orig_error("orig_error", id, this), current_error("current_error", id, this),
  method("method", id, this), iteration("iteration", id, this),
  maxiter("maxiter", id, this), use_previous_soln("use_previous_soln", id, this)
{
    matrixport=scinew MatrixIPort(this, "Matrix", MatrixIPort::Atomic);
    add_iport(matrixport);
    rhsport=scinew ColumnMatrixIPort(this, "RHS", ColumnMatrixIPort::Atomic);
    add_iport(rhsport);

    solport=scinew ColumnMatrixOPort(this, "Solution", ColumnMatrixIPort::Atomic);
    add_oport(solport);
}

SolveMatrix::SolveMatrix(const SolveMatrix& copy, int deep)
: Module(copy, deep), target_error("target_error", id, this),
  flops("flops", id, this), floprate("floprate", id, this),
  memrefs("memrefs", id, this), memrate("memrate", id, this),
  orig_error("orig_error", id, this), current_error("current_error", id, this),
  method("method", id, this), iteration("iteration", id, this),
  maxiter("maxiter", id, this), use_previous_soln("use_previous_soln", id, this)
{
    NOT_FINISHED("SolveMatrix::SolveMatrix");
}

SolveMatrix::~SolveMatrix()
{
}

Module* SolveMatrix::clone(int deep)
{
    return scinew SolveMatrix(*this, deep);
}

void SolveMatrix::execute()
{
    MatrixHandle matrix;
    if(!matrixport->get(matrix))
	return;	
    ColumnMatrixHandle rhs;
    if(!rhsport->get(rhs))
	return;
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

    clString meth=method.get();
    if(meth == "conjugate_gradient"){
	conjugate_gradient(matrix.get_rep(),
			   *solution.get_rep(), *rhs.get_rep());
	solport->send(solution);
    } else if(meth == "jacobi"){
	jacobi(matrix.get_rep(),
	       *solution.get_rep(), *rhs.get_rep());
	solport->send(solution);
    } else {
	error(clString("Unknown solution method: ")+meth);
    }
}

void SolveMatrix::jacobi(Matrix* matrix,
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

    orig_error.set(err);
    current_error.set(err);

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
	Sub(lhs, lhs, Z, flop, memref);

	matrix->mult(lhs, Z, flop, memref);
	Sub(Z, Z, rhs, flop, memref);
	err=Z.vector_norm(flop, memref)/bnorm;

	errlist.add(err);

	gflop+=flop/1000000000;
	flop=flop%1000000000;
	grefs+=memref/1000000000;
	memref=memref%1000000000;

	if(niter == 1 || niter == 5 || niter%10 == 0){
	    iteration.set(niter);
	    current_error.set(err);
	    double time=timer.time();
	    flops.set(gflop*1.e9+flop);
	    floprate.set((gflop*1.e3+flop*1.e-6)/time);
	    memrefs.set(grefs*1.e9+memref);
	    memrate.set((grefs*1.e3+memref*1.e-6)/time);

	    append_values(niter, errlist, last_update, targetidx, targetlist, last_errupdate);

	    double progress=(log_orig-log(err))/(log_orig-log_targ);
	    update_progress(progress);

	    solport->send_intermediate(rhs.clone());
	}
    }
    iteration.set(niter);
    current_error.set(err);

    time=timer.time();
    flops.set(gflop*1.e9+flop);
    floprate.set((gflop*1.e3+flop*1.e-6)/time);
    memrefs.set(grefs*1.e9+memref);
    memrate.set((grefs*1.e3+memref*1.e-6)/time);

    TCL::execute(id+" finish_graph");
    append_values(niter, errlist, last_update, targetidx, targetlist, last_errupdate);
}

void SolveMatrix::conjugate_gradient(Matrix* matrix,
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
    
    ColumnMatrix diag(size);
    // We should try to do a better job at preconditioning...
    int i;

    for(i=0;i<size;i++){
	diag[i]=1./matrix->get(i,i);
    }
    flop+=size;
    memref=2*size*sizeof(double);

    ColumnMatrix R(size);
    matrix->mult(lhs, R, flop, memref);


    Sub(R, rhs, R, flop, memref);
    double bnorm=rhs.vector_norm(flop, memref);

    ColumnMatrix Z(size);
    matrix->mult(R, Z, flop, memref);

    ColumnMatrix P(size);
    double bkden=0;
    double err=R.vector_norm(flop, memref)/bnorm;

    orig_error.set(err);
    current_error.set(err);

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

	// Simple Preconditioning...
	Mult(Z, R, diag, flop, memref);	

	// Calculate coefficient bk and direction vectors p and pp
	double bknum=Dot(Z, R, flop, memref);

	if(niter==1){
	    P=Z;
	    memref+=2*sizeof(double);
	} else {
	    double bk=bknum/bkden;
	    ScMult_Add(P, bk, P, Z, flop, memref);
	}
	bkden=bknum;

	// Calculate coefficient ak, new iterate x and new residuals r and rr
	matrix->mult(P, Z, flop, memref);

	double akden=Dot(Z, P, flop, memref);

	double ak=bknum/akden;
	ScMult_Add(lhs, ak, P, lhs, flop, memref);
	ScMult_Add(R, -ak, Z, R, flop, memref);

	err=R.vector_norm(flop, memref)/bnorm;

	errlist.add(err);

	gflop+=flop/1000000000;
	flop=flop%1000000000;
	grefs+=memref/1000000000;
	memref=memref%1000000000;

	if(niter == 1 || niter == 5 || niter%10 == 0){
	    iteration.set(niter);
	    current_error.set(err);
	    double time=timer.time();
	    flops.set(gflop*1.e9+flop);
	    floprate.set((gflop*1.e3+flop*1.e-6)/time);
	    memrefs.set(grefs*1.e9+memref);
	    memrate.set((grefs*1.e3+memref*1.e-6)/time);

	    append_values(niter, errlist, last_update, targetidx, targetlist, last_errupdate);

	    double progress=(log_orig-log(err))/(log_orig-log_targ);
	    update_progress(progress);

	    solport->send_intermediate(lhs.clone());
	    for(int i=0;i<10000;i++){
		Task::yield();
	    }
	}
    }
    iteration.set(niter);
    current_error.set(err);

    time=timer.time();
    flops.set(gflop*1.e9+flop);
    floprate.set((gflop*1.e3+flop*1.e-6)/time);
    memrefs.set(grefs*1.e9+memref);
    memrate.set((grefs*1.e3+memref*1.e-6)/time);

    TCL::execute(id+" finish_graph");
    append_values(niter, errlist, last_update, targetidx, targetlist, last_errupdate);
}

void SolveMatrix::append_values(int niter, const Array1<double>& errlist,
				int& last_update,
				const Array1<int>& targetidx,
				const Array1<double>& targetlist,
				int& last_errupdate)
{
    char buf[10000];
    ostrstream str(buf, 1000);
    str << id << " append_graph " << niter << " \"";
    int i;
    for(i=last_update;i<errlist.size();i++){
	str << i << " " << errlist[i] << " ";
    }
    str << "\" \"";
    for(i=last_errupdate;i<targetidx.size();i++){
	str << targetidx[i] << " " << targetlist[i] << " ";
    }
    str << "\" ; update idletasks" << '\0';
    TCL::execute(str.str());
    last_update=errlist.size();
    last_errupdate=targetidx.size();
}
