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

    ColumnMatrix *x=new ColumnMatrix(AH->ncols());
    ColumnMatrixHandle xH(x);

    // compute:   x = A^(-1) b

    x_port->send(xH);

} 

} // End namespace Modules
} // End namespace DaveW

//
// $Log$
// Revision 1.1  1999/12/09 00:10:04  dmw
// woops - wrong filename
//
