
/*
 *  MatVec: Matrix - Matrix operation (e.g. addition, multiplication, ...)
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <PSECore/Datatypes/ColumnMatrixPort.h>
#include <PSECore/Datatypes/MatrixPort.h>
#include <SCICore/Datatypes/DenseMatrix.h>
#include <SCICore/Datatypes/SymSparseRowMatrix.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <iostream>
using std::cerr;

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;

class MatVec : public Module {
    ColumnMatrixIPort* icol;
    MatrixIPort* imat;
    ColumnMatrixOPort* ocol;
    TCLstring opTCL;
    ColumnMatrixHandle icolH_last;
    MatrixHandle imatH_last;
    ColumnMatrixHandle ocH;
    clString opTCL_last;

    void conjugate_gradient_sci(Matrix* matrix,
			   ColumnMatrix& lhs, ColumnMatrix& rhs);
    int somethingChanged(ColumnMatrixHandle m1h, MatrixHandle m2h, clString opS);
    int AtimesB();
public:
    MatVec(const clString& id);
    virtual ~MatVec();
    virtual void execute();
};

extern "C" Module* make_MatVec(const clString& id)
{
    return new MatVec(id);
}

MatVec::MatVec(const clString& id)
: Module("MatVec", id, Filter), opTCL("opTCL", id, this)
{
    icol=new ColumnMatrixIPort(this, "A", ColumnMatrixIPort::Atomic);
    add_iport(icol);
    imat=new MatrixIPort(this, "b", MatrixIPort::Atomic);
    add_iport(imat);

    // Create the output port
    ocol=new ColumnMatrixOPort(this, "Output", ColumnMatrixIPort::Atomic);
    add_oport(ocol);
}

MatVec::~MatVec()
{
}

int MatVec::somethingChanged(ColumnMatrixHandle icolH, MatrixHandle imatH, 
			    clString opTCL) {
    int changed=0;
    if (icolH.get_rep() != icolH_last.get_rep()) {icolH_last=icolH; changed=1;}
    if (imatH.get_rep() != imatH_last.get_rep()) {imatH_last=imatH; changed=1;}
    if (opTCL != opTCL_last) {opTCL_last=opTCL; changed=1;}
    return changed;
}

int MatVec::AtimesB() {
    ColumnMatrix *res = scinew ColumnMatrix(imatH_last->nrows());
    int dummy1, dummy2;
//    cerr << "Calling mult...\n";
    imatH_last->mult(*icolH_last.get_rep(), *res, dummy1, dummy2);
//    cerr << "Done!\n";
    ocH=res;
    return 1;
}

void MatVec::execute() {
    update_state(NeedData);

    ColumnMatrixHandle icolH;
    if (!icol->get(icolH) || !icolH.get_rep()) return;
    MatrixHandle imatH;
    if (!imat->get(imatH) || !imatH.get_rep()) return;
    clString opS=opTCL.get();

//    cerr << "Starting MatVec!\n";
    update_state(JustStarted);
//    cerr << "Here we go...\n";
    if (!somethingChanged(icolH, imatH, opS)) ocol->send(ocH);
//    cerr << "Doing great!\n";
    if (opS == "AtimesB") {
	if (AtimesB()) ocol->send(ocH);
    } else {
	cerr << "MatVec: unknown operation "<<opS<<"\n";
    }
}    
} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.4  2000/03/17 09:27:07  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.3  1999/12/09 09:54:19  dmw
// took out debug comments
//
// Revision 1.2  1999/10/07 02:06:52  sparker
// use standard iostreams and complex type
//
// Revision 1.1  1999/09/07 04:02:23  dmw
// more modules that were left behind...
//
