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

class SolveMatrix : public Module {
    MatrixIPort* matrixport;
    ColumnMatrixIPort* rhsport;
    ColumnMatrixOPort* solport;
    ColumnMatrixHandle solution;
    void conjugate_gradient(Matrix*, ColumnMatrix&, ColumnMatrix&);

    void append_values(int niter, const Array1<double>& errlist,
		       int& last_update);
public:
    SolveMatrix(const clString& id);
    SolveMatrix(const SolveMatrix&, int deep);
    virtual ~SolveMatrix();
    virtual Module* clone(int deep);
    virtual void execute();

    TCLdouble target_error;
    TCLint flops;
    TCLdouble floprate;
    TCLdouble orig_error;
    TCLdouble current_error;
    TCLstring method;
    TCLint iteration;
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
  orig_error("orig_error", id, this), current_error("current_error", id, this),
  method("method", id, this), iteration("iteration", id, this)
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
  orig_error("orig_error", id, this), current_error("current_error", id, this),
  method("method", id, this), iteration("iteration", id, this)
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
    if(!solution.get_rep() || solution->nrows() != rhs->nrows()){
	solution=scinew ColumnMatrix(rhs->nrows());
	solution->zero();
    } else {
	solution.detach();
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
    } else {
	error(clString("Unknown solution method: ")+meth);
    }
}

void SolveMatrix::conjugate_gradient(Matrix* matrix,
				    ColumnMatrix& lhs, ColumnMatrix& rhs)
{
    int size=matrix->nrows();

    int flop=0;
    flops.set(flop);
    floprate.set(0);

    iteration.set(0);
    
    ColumnMatrix diag(size);
    // We should try to do a better job at preconditioning...
    int i;
    for(i=0;i<size;i++){
	diag[i]=1./matrix->get(i,i);
    }
    flop+=size;

    ColumnMatrix R(size);
    flop+=matrix->mult(lhs, R);

    for(i=0;i<size;i++){
	R[i]=rhs[i]-R[i];
    }
    flop+=size;

    double bnorm=rhs.vector_norm();
    flop+=2*size;

    ColumnMatrix Z(size);
    flop+=matrix->mult(R, Z);

    ColumnMatrix P(size);
    double bkden=0;
    double err=R.vector_norm();
    flop+=2*size;

    orig_error.set(err);
    current_error.set(err);

    int niter=0;
    int toomany=2*size;
    double max_error=target_error.get();

    flops.set(flop);
    double time=timer.time();
    floprate.set((double)flop*1.e-6/time);
    
    TCL::execute(id+" reset_graph");
    Array1<double> errlist;
    errlist.add(err);
    int last_update=0;
    append_values(1, errlist, last_update);

    double log_orig=log(err);
    double log_targ=log(max_error);
    while(niter < toomany && err > max_error){
	niter++;

	// Simple Preconditioning...
	for(int idd=0;idd<size;idd++){
	    Z[idd]=R[idd]*diag[idd];
	}
	flop+=size;
	
	// Calculate coefficient bk and direction vectors p and pp
	double bknum=0;
	int i;
	for(i=0;i<size;i++)
	    bknum+=Z[i]*R[i];
	flop+=size;

	if(niter==1){
	    P=Z;
	} else {
	    double bk=bknum/bkden;
	    for(int i=0;i<size;i++){
		P[i]=bk*P[i]+Z[i];
	    }
	    flop+=2*size+1;
	}
	bkden=bknum;

	// Calculate coefficient ak, new iterate x and new residuals r and rr
	flop+=matrix->mult(P, Z);

	double akden=0.0;
	for(i=0;i<size;i++)
	    akden+=Z[i]*P[i];
	flop+=2*size;

	double ak=bknum/akden;
	for(i=0;i<size;i++){
	    lhs[i]+=ak*P[i];
	    R[i]-=ak*Z[i];
	}
	flop+=4*size;

	err=R.vector_norm()/bnorm;
	flop+=2*size;

	errlist.add(err);

	if(niter%10 == 0){
	    iteration.set(niter);
	    current_error.set(err);
	    flops.set(flop);
	    double time=timer.time();
	    floprate.set((double)flop*1.e-6/time);

	    append_values(niter, errlist, last_update);

	    double progress=(log_orig-log(err))/(log_orig-log_targ);
	    update_progress(progress);
	}
    }
    iteration.set(niter);
    current_error.set(err);
    flops.set(flop);
    time=timer.time();
    floprate.set((double)flop*1.e-6/time);
    TCL::execute(id+" finish_graph");
    append_values(niter, errlist, last_update);
}

void SolveMatrix::append_values(int niter, const Array1<double>& errlist,
				int& last_update)
{
    char buf[10000];
    ostrstream str(buf, 1000);
    str << id << " append_graph " << niter << " \"";
    for(int i=last_update;i<errlist.size();i++){
	str << i << " " << errlist[i] << " ";
    }
    str << "\" ; update idletasks" << '\0';
    TCL::execute(str.str());
    last_update=errlist.size();
}
