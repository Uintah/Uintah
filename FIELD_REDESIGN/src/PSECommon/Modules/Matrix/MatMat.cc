
/*
 *  MatMat: Matrix - Matrix operation (e.g. addition, multiplication, ...)
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <PSECore/Datatypes/MatrixPort.h>
#include <SCICore/Datatypes/ColumnMatrix.h>
#include <SCICore/Datatypes/DenseMatrix.h>
#include <SCICore/Datatypes/SymSparseRowMatrix.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <SCICore/Thread/Parallel.h>
#include <SCICore/Thread/SimpleReducer.h>
#include <SCICore/Thread/Thread.h>
#include <iostream>
using std::cerr;
using std::endl;
#include <math.h>

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;
using SCICore::Thread::Parallel;
using SCICore::Thread::SimpleReducer;
using SCICore::Thread::Thread;

struct Result {
    double r;
    double pad[15];
};

struct PStats {
    int flop;
    int memref;
    int gflop;
    int grefs;
    int pad[28];
};

class MatMat;

struct CGData2 {
    MatMat *module;
    ColumnMatrix *rhs;
    ColumnMatrix *lhs;
    Matrix *mat;
    ColumnMatrix* diag;
    int niter;
    int toomany;
    ColumnMatrix *Z;
    ColumnMatrix *R;
    ColumnMatrix *P;
    SimpleReducer reducer;
    int np;
    Result* res1;
    Result* res2;
    Result* res3;
    double err;
    double max_error;
    double bnorm;
    PStats* stats;
    CGData2();
};

CGData2::CGData2() : reducer("MatMat reduction barrier") 
{
}

class MatMat : public Module {
    MatrixIPort* imat1;
    MatrixIPort* imat2;
    MatrixOPort* omat;
    TCLstring opTCL;
    MatrixHandle im1H_last;
    MatrixHandle im2H_last;
    MatrixHandle omH;
    clString opTCL_last;

    void conjugate_gradient_sci(Matrix* matrix,
			   ColumnMatrix& lhs, ColumnMatrix& rhs);
    int somethingChanged(MatrixHandle m1h, MatrixHandle m2h, clString opS);
    int AtimesBinv();
    CGData2* data;
public:
    void parallel_conjugate_gradient(int proc);
    MatMat(const clString& id);
    virtual ~MatMat();
    virtual void execute();
};

extern "C" Module* make_MatMat(const clString& id)
{
    return new MatMat(id);
}

MatMat::MatMat(const clString& id)
: Module("MatMat", id, Filter), opTCL("opTCL", id, this)
{
    imat1=new MatrixIPort(this, "A", MatrixIPort::Atomic);
    add_iport(imat1);
    imat2=new MatrixIPort(this, "B", MatrixIPort::Atomic);
    add_iport(imat2);

    // Create the output port
    omat=new MatrixOPort(this, "Output", MatrixIPort::Atomic);
    add_oport(omat);
}

MatMat::~MatMat()
{
}

int MatMat::somethingChanged(MatrixHandle im1H, MatrixHandle im2H, 
			    clString opTCL) {
    int changed=0;
    if (im1H.get_rep() != im1H_last.get_rep()) {im1H_last=im1H; changed=1;}
    if (im2H.get_rep() != im2H_last.get_rep()) {im2H_last=im2H; changed=1;}
    if (opTCL != opTCL_last) {opTCL_last=opTCL; changed=1;}
    return changed;
}

int MatMat::AtimesBinv() {
    Matrix *A=im1H_last.get_rep();
    Matrix *B=im2H_last.get_rep();
    if (!dynamic_cast<SymSparseRowMatrix*>(B)) {
	cerr << "Error - B must be a SymSparseRowMatrix to compute A x B^(-1).\n";
	return 0;
    }
    if (A->ncols() != B->nrows()) {
	cerr << "Error - A->ncols must equal B->nrows.\n";
	return 0;
    }
    DenseMatrix *C = scinew DenseMatrix(A->nrows(), B->ncols());
    Array1<double> nzeros;
    Array1<int> nzidx;
    ColumnMatrix lhs(A->ncols());
    ColumnMatrix rhs(B->nrows());
    lhs.zero();
    int i,j;
    for (i=0; i<A->nrows(); i++) {
	A->getRowNonzeros(i, nzidx, nzeros);
	rhs.zero();
	for (j=0; j<nzidx.size(); j++) {
	    if ((nzidx[j] < 0) || (nzidx[j] >= B->nrows())) {
		cerr << "Error - tried to access element "<<nzidx[j];
		cerr << " i="<<i<<" A->nrows="<<A->nrows();
		cerr << " B->ncols="<<B->ncols()<<"\n";
		return 0;
	    }
	    rhs[nzidx[j]]=nzeros[j];
	}
	conjugate_gradient_sci(B, lhs, rhs);
	for (j=0; j<B->ncols(); j++) (*C)[i][j]=lhs[j];
    }
    omH=C;
    return 1;
}

void MatMat::execute() {
    update_state(NeedData);

    MatrixHandle im1H;
    if (!imat1->get(im1H) || !im1H.get_rep()) return;
    MatrixHandle im2H;
    if (!imat2->get(im2H) || !im2H.get_rep()) return;
    clString opS=opTCL.get();

    update_state(JustStarted);

    if (!somethingChanged(im1H, im2H, opS)) omat->send(omH);
    if (opS == "AtimesBinv") {
	if (AtimesBinv()) omat->send(omH);
    } else {
	cerr << "MatMat: unknown operation "<<opS<<"\n";
    }
}    

void MatMat::conjugate_gradient_sci(Matrix* matrix,
					 ColumnMatrix& lhs, ColumnMatrix& rhs)
{
  cerr << "cg started\n";
  data=new CGData2;
  int np = Thread::numProcessors();
  if (np>4) np/=2;	// be nice - just use half the processors
  cerr << "np=" << np << endl;
  data->module=this;
  data->np=np;
  data->rhs=&rhs;
  data->lhs=&lhs;
  data->mat=matrix;
  data->stats=new PStats[data->np];
  data->max_error=1.0e-5;
  data->toomany=matrix->nrows()*2;
  Thread::parallel(Parallel<MatMat>(this, &MatMat::parallel_conjugate_gradient),
		   data->np, true);
  delete data->stats;
  delete data;
}

void MatMat::parallel_conjugate_gradient(int processor)
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
#if 0
  Array1<int> targetidx;
  Array1<double> targetlist;
  Array1<double> errlist;

  int last_update=0;
  
  int last_errupdate=0;
#endif
  
  if(processor == 0){
#if 0
    data->timer->clear();
    data->timer->start();
    flops.set(0);
    floprate.set(0);
    memrefs.set(0);
    memrate.set(0);
    iteration.set(0);
#endif    
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
#if 0
    data->toomany=maxiter.get();
    if(data->toomany == 0)
      data->toomany=2*size;
    data->max_error=target_error.get();
#endif
    data->res1=new Result[data->np];
    data->res2=new Result[data->np];
    data->res3=new Result[data->np];
    
    stats->gflop+=stats->flop/1000000000;
    stats->flop=stats->flop%1000000000;
    stats->grefs+=stats->memref/1000000000;
    stats->memref=stats->memref%1000000000;
#if 0
    orig_error.set(data->err);
    current_error.set(data->err);
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
#endif
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
#if 0
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
#endif
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
    data->res1[processor].r=Dot(Z, R, stats->flop, stats->memref, beg, end);
    data->reducer.wait(data->np);
    
    double  bknum=0;
    int ii;
    for(ii=0;ii<data->np;ii++)
      bknum+=data->res1[ii].r;
    
    if(data->niter==1){
      Copy(P, Z, stats->flop, stats->memref, beg, end);
    } else {
      double bk=bknum/bkden;
      ScMult_Add(P, bk, P, Z, stats->flop, stats->memref, beg, end);
    }
    data->reducer.wait(data->np);
    // Calculate coefficient ak, new iterate x and new residuals r and rr
    matrix->mult(P, Z, stats->flop, stats->memref, beg, end);
    bkden=bknum;
    data->res2[processor].r=Dot(Z, P, stats->flop, stats->memref, beg, end);
    data->reducer.wait(data->np);
    
    double akden=0;
    for(ii=0;ii<data->np;ii++)
      akden+=data->res2[ii].r;
    double ak=bknum/akden;
    ColumnMatrix& lhs=*data->lhs;
    ScMult_Add(lhs, ak, P, lhs, stats->flop, stats->memref, beg, end);
//     ColumnMatrix& rhs=*data->rhs;
    ScMult_Add(R, -ak, Z, R, stats->flop, stats->memref, beg, end);
    
    data->res3[processor].r=R.vector_norm(stats->flop, stats->memref, beg, end)/data->bnorm;
    data->reducer.wait(data->np);
    err=0;
    for(ii=0;ii<data->np;ii++)
      err+=data->res3[ii].r;

    int ev=(err<1000000);
//    cerr << "EVALUATING2 "<<ev<<"\n";
    if (!ev) err=1000000;


    stats->gflop+=stats->flop/1000000000;
    stats->flop=stats->flop%1000000000;
    stats->grefs+=stats->memref/1000000000;
    stats->memref=stats->memref%1000000000;
    
    if(processor == 0){
#if 0
      errlist.add(err);
#endif      
      stats->gflop+=stats->flop/1000000000;
      stats->flop=stats->flop%1000000000;
      stats->grefs+=stats->memref/1000000000;
      stats->memref=stats->memref%1000000000;
      
      if(data->niter == 1 || data->niter == 10 || data->niter%20 == 0){
	if(data->niter <= 60 || data->niter%60 == 0){
#if 0
	  iteration.set(data->niter);
	  current_error.set(err);
	  double time=timer.time();
	  flops.set(14*stats->gflop*1.e9+stats->flop);
	  floprate.set(14*(stats->gflop*1.e3+stats->flop*1.e-6)/time);                    memrefs.set(14*stats->grefs*1.e9+stats->memref);
	  memrate.set(14*(stats->grefs*1.e3+stats->memref*1.e-6)/time);
	  append_values(data->niter, errlist, last_update, targetidx,
			targetlist, last_errupdate);
	  
#endif
	  if(err > 0){
	    double progress=(log_orig-log(err))/(log_orig-log_targ);                        
	    cerr << "err=" << err << endl;
	    //                         cerr << "log_orig=" << log_orig << endl;
	    update_progress(progress);
	  }
	}
#if 0
	if(emit_partial.get() && data->niter%60 == 0)
	  solport->send_intermediate(lhs.clone());
#endif
      }
    }
  }
  if(processor == 0){
    data->niter++;
    
#if 0
    iteration.set(data->niter);
    current_error.set(err);
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
#endif    
  }
}
} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.4  2000/03/17 09:27:06  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.3  1999/10/07 02:06:52  sparker
// use standard iostreams and complex type
//
// Revision 1.2  1999/09/10 04:09:02  jmk
// Added & so it will compile on Linux
//
// Revision 1.1  1999/09/07 04:02:23  dmw
// more modules that were left behind...
//
