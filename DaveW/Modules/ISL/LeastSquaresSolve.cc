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

Module* make_LeastSquaresSolve(const clString& id) {
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

    AA->solve(*bb);
    x_port->send(bb);
} 

} // End namespace Modules
} // End namespace DaveW

//
// $Log$
// Revision 1.2  1999/12/09 09:56:26  dmw
// got this module working
//
// Revision 1.1  1999/12/09 00:10:04  dmw
// woops - wrong filename
//
