/*
 *  SplitMatVec.cc: Split an nx3 matrix into 3 nx1 column-matrices
 *
 *  Written by:
 *   David Weinstein
 *   University of Utah
 *   June 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/ColumnMatrixPort.h>
#include <PSECore/Datatypes/MatrixPort.h>
#include <SCICore/Malloc/Allocator.h>
#include <iostream>
using std::cerr;

namespace DaveW {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;

class VecSplit : public Module {
    MatrixIPort* imatP;
    ColumnMatrixOPort* ovec1P;
    ColumnMatrixOPort* ovec2P;
    ColumnMatrixOPort* ovec3P;
public:
    VecSplit(const clString& id);
    virtual ~VecSplit();
    virtual void execute();
};

Module* make_VecSplit(const clString& id)
{
    return scinew VecSplit(id);
}

VecSplit::VecSplit(const clString& id)
: Module("VecSplit", id, Filter)
{
    // Create the input port
    imatP=scinew MatrixIPort(this, "Input Matrix", MatrixIPort::Atomic);
    add_iport(imatP);
    ovec1P=scinew ColumnMatrixOPort(this, "Output Vector1",ColumnMatrixIPort::Atomic);
    add_oport(ovec1P);
    ovec2P=scinew ColumnMatrixOPort(this, "Output Vector2",ColumnMatrixIPort::Atomic);
    add_oport(ovec2P);
    ovec3P=scinew ColumnMatrixOPort(this, "Output Vector3",ColumnMatrixIPort::Atomic);
    add_oport(ovec3P);
}

VecSplit::~VecSplit() {
}

void VecSplit::execute()
{
     MatrixHandle imatH;
     Matrix* imat;
     if (!imatP->get(imatH) || !(imat=imatH.get_rep())) return;
     if (imat->ncols() != 3) {
	 cerr << "Error - this modules splits an nx3 matrix into 3 nx1 columns -- can't \noperate on a matrix with "<<imat->ncols()<<" columns.\n";
	 return;
     }
     int nr=imat->nrows();
     ColumnMatrix* ovec1, *ovec2, *ovec3;
     ovec1=new ColumnMatrix(nr);
     ovec2=new ColumnMatrix(nr);
     ovec3=new ColumnMatrix(nr);

     for (int i=0; i<nr; i++) {
	 (*ovec1)[i]=(*imat)[i][0];
	 (*ovec2)[i]=(*imat)[i][1];
	 (*ovec3)[i]=(*imat)[i][2];
     }

     ColumnMatrixHandle ov1H(ovec1);
     ColumnMatrixHandle ov2H(ovec2);
     ColumnMatrixHandle ov3H(ovec3);

     ovec1P->send(ov1H);
     ovec2P->send(ov2H);
     ovec3P->send(ov3H);
}
} // End namespace Modules
} // End namespace DaveW


//
// $Log$
// Revision 1.3  1999/10/07 02:06:35  sparker
// use standard iostreams and complex type
//
// Revision 1.2  1999/09/05 23:16:19  dmw
// build scalar field of error values from Basis Matrix
//
// Revision 1.1  1999/09/02 04:49:25  dmw
// more of Dave's modules
//
//
