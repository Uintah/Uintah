
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

#include <Dataflow/Ports/ColumnMatrixPort.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/SymSparseRowMatrix.h>
#include <Core/Malloc/Allocator.h>
#include <Core/TclInterface/TCLvar.h>
#include <iostream>
using std::cerr;

namespace SCIRun {


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
    void AtimesB(int trans);
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

void MatVec::AtimesB(int trans) {
    ColumnMatrix *res;
    if (trans) res=scinew ColumnMatrix(imatH_last->ncols());
    else res=scinew ColumnMatrix(imatH_last->nrows());
    int dummy1, dummy2;
//    cerr << "Calling mult...\n";

    int cnt=0;
    for (int i=0; i<icolH_last->nrows(); i++) 
	if ((*icolH_last.get_rep())[i]) cnt++;
    int spVec=0;
    if (cnt < icolH_last->nrows()/10) spVec=1;
    if (trans) imatH_last->mult_transpose(*icolH_last.get_rep(), *res, 
					  dummy1, dummy2, -1, -1, spVec);
    else
	imatH_last->mult(*icolH_last.get_rep(), *res, dummy1, dummy2, 
			 -1, -1, spVec);
//    cerr << "Done!\n";
    ocH=res;
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
	AtimesB(0); 
	ocol->send(ocH);
    } else if (opS == "AtTimesB") {
	AtimesB(1);
	ocol->send(ocH);
    } else {
	cerr << "MatVec: unknown operation "<<opS<<"\n";
    }
}    
} // End namespace SCIRun
