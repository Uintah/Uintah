/****************************************************************
 *  Least Squares Solve -- take a RHS (b) and a matrix (A);     *
 *                         return the least squares solution (x)*
 *  Written by:                                                 *
 *   David Weinstein                                            *
 *   Department of Computer Science                             *
 *   University of Utah                                         *
 *   December 1999                                              *
 *                                                              *
 *  Copyright (C) 1999 SCI Group                                *
 *                                                              *
 *                                                              *
 ****************************************************************/

#include <DaveW/ThirdParty/OldLinAlg/matrix.h>
#include <DaveW/ThirdParty/OldLinAlg/vector.h>
#include <DaveW/ThirdParty/NumRec/dsvdcmp.h>
#include <DaveW/ThirdParty/NumRec/dsvbksb.h>

#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/MatrixPort.h>
#include <PSECore/Datatypes/ColumnMatrixPort.h>
#include <SCICore/Datatypes/DenseMatrix.h>
#include <iostream>

namespace DaveW {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;
using std::cerr;
using std::endl;

class LeastSquaresSolve : public Module {
    MatrixIPort* A_port;
    ColumnMatrixIPort* b_port;
    ColumnMatrixOPort* x_port;    
public:
    LeastSquaresSolve(const clString& id);
    virtual ~LeastSquaresSolve();
    virtual void execute();
  
};

extern "C" Module* make_LeastSquaresSolve(const clString& id) {
    return new LeastSquaresSolve(id);
}

LeastSquaresSolve::LeastSquaresSolve(const clString& id)
  : Module("LeastSquaresSolve", id, Filter) {
    A_port = new MatrixIPort(this,"A", MatrixIPort::Atomic);
    add_iport(A_port);

    b_port = new ColumnMatrixIPort(this,"b",ColumnMatrixIPort::Atomic);
    add_iport(b_port);

    x_port = new ColumnMatrixOPort(this,"x",ColumnMatrixIPort::Atomic);
    add_oport(x_port);
}

LeastSquaresSolve::~LeastSquaresSolve(){}

void LeastSquaresSolve::execute()
{
    MatrixHandle AH;
    if (!A_port->get(AH)) return;
    
    ColumnMatrixHandle bH;
    if (!b_port->get(bH)) return;
   
    if (AH->nrows() != bH->nrows()) {
      cerr << "Error - matrix and RHS must have the same number of rows!\n";
      return;
    }
    double *b=bH->get_rhs();

    int nrows=AH->nrows();
    int ncols=AH->ncols();

    DenseMatrix *AA = new DenseMatrix(ncols, ncols);
    ColumnMatrix *bb = new ColumnMatrix(ncols);
    double *bbp=bb->get_rhs();

    AA->zero();
    bb->zero();

    int i, j, k;
    for (i=0; i<ncols; i++) {
	for (j=0; j<ncols; j++)
	    for (k=0; k<nrows; k++)
		(*AA)[i][j]+=AH->get(k,i)*AH->get(k,j);
	for (k=0; k<nrows; k++)
	    bbp[i]+=AH->get(k,i)*(b[k]);
    }

    if (!AA->solve(*bb)) {
	// copy the matrix and rhs into numerical recipes structures 
	//   and use SVD to solve this
	double **a0 = makeMatrix(ncols, ncols);
	double **v0 = makeMatrix(ncols, ncols);
	double *x0 = makeVector(ncols);
	double *w0 = makeVector(ncols);
	double *b0 = makeVector(ncols);
	for (i=0; i<ncols; i++) {
	    for (j=0; j<ncols; j++) 
		a0[i+1][j+1]=(*AA)[i][j];
	    b0[i+1]=bbp[i];
	}
	dsvdcmp(a0, ncols, ncols, w0, v0, 30);
	int trunc=0;
	for (i=1; i<ncols; i++) 
	    if (w0[i] < 0.00001) {
		w0[i]=0;
		trunc++;
	    }
	cerr << "LeastSquaresSolve truncated "<<trunc<<" terms.\n";
	dsvbksb(a0, w0, v0, ncols, ncols, b0, x0);
	for (i=0; i<ncols; i++) {
	    bbp[i]=x0[i+1];
	}
	freeMatrix(a0);
	freeMatrix(v0);
	freeVector(x0);
	freeVector(w0);
	freeVector(b0);
    }
    x_port->send(bb);
} 

} // End namespace Modules
} // End namespace DaveW

//
// $Log$
// Revision 1.4  2000/03/17 09:25:47  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.3  1999/12/10 07:00:58  dmw
// if DenseMatrix::solve() (Gaussian-Elimination) fails, use SVD and backsubstitution to get the least squares solution
//
// Revision 1.2  1999/12/09 09:56:26  dmw
// got this module working
//
// Revision 1.1  1999/12/09 00:10:04  dmw
// woops - wrong filename
//
