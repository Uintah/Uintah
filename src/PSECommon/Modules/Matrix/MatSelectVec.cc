
/*
 *  MatSelectVec: Read in a surface, and output a .tri and .pts file
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   June 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <PSECore/Datatypes/ColumnMatrixPort.h>
#include <PSECore/Datatypes/MatrixPort.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCLvar.h>

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;

class MatSelectVec : public Module {
    MatrixIPort* imat;
    ColumnMatrixOPort* ovec;
    TCLstring rowOrColTCL;
    TCLint rowTCL;
    TCLint rowMaxTCL;
    TCLint colTCL;
    TCLint colMaxTCL;
    TCLint animateTCL;

public:
    MatSelectVec(const clString& id);
    virtual ~MatSelectVec();
    virtual void execute();
    virtual void tcl_command(TCLArgs&, void*);
    int stop;

};

Module* make_MatSelectVec(const clString& id)
{
    return new MatSelectVec(id);
}

MatSelectVec::MatSelectVec(const clString& id)
: Module("MatSelectVec", id, Filter), animateTCL("animateTCL", id, this),
  colTCL("colTCL", id, this), colMaxTCL("colMaxTCL", id, this),
  rowTCL("rowTCL", id, this), rowMaxTCL("rowMaxTCL", id, this),
  rowOrColTCL("rowOrColTCL", id, this)
{
    // Create the input port
    imat=new MatrixIPort(this, "Matrix", MatrixIPort::Atomic);
    add_iport(imat);

    // Create the output port
    ovec=new ColumnMatrixOPort(this,"ColumnMatrix", ColumnMatrixIPort::Atomic);
    add_oport(ovec);
    stop=0;
}

MatSelectVec::~MatSelectVec()
{
}

void MatSelectVec::execute() {
    stop=0;
    update_state(NeedData);

    MatrixHandle mh;
    if (!imat->get(mh))
	return;
    if (!mh.get_rep()) {
	cerr << "Error: empty matrix\n";
	return;
    }

    update_state(JustStarted);

    if (colMaxTCL.get() != mh->ncols()-1)
	colMaxTCL.set(mh->ncols()-1);
    if (rowMaxTCL.get() != mh->nrows()-1)
	rowMaxTCL.set(mh->nrows()-1);
    reset_vars();

    int which;
    int useRow=(rowOrColTCL.get() == "row");
    if (useRow) which=rowTCL.get();
    else which=colTCL.get();

    if (!animateTCL.get()) {
	ColumnMatrix *cm;
	if (useRow) {
	    ColumnMatrix* cm=new ColumnMatrix(mh->ncols());
	    double *data = cm->get_rhs();
	    for (int c=0; c<mh->ncols(); c++) data[c]=mh->get(which, c);
	} else {
	    ColumnMatrix* cm=new ColumnMatrix(mh->nrows());
	    double *data = cm->get_rhs();
	    for (int r=0; r<mh->nrows(); r++) data[r]=mh->get(r, which);
	}	    
	ColumnMatrixHandle cmh(cm);
	ovec->send(cmh);
    } else {
	ColumnMatrix *cm;
	if (useRow) {
	    for (; which<mh->nrows()-1; which++, rowTCL.set(which)) {
		if (stop) { stop=0; break; }
		cerr << which << "\n";
		ColumnMatrix* cm=new ColumnMatrix(mh->ncols());
		double *data = cm->get_rhs();
		for (int c=0; c<mh->ncols(); c++) data[c]=mh->get(which, c);
		ColumnMatrixHandle cmh(cm);
		ovec->send_intermediate(cmh);
	    }
	} else {
	    for (; which<mh->ncols()-1; which++, colTCL.set(which)) {
		if (stop) { stop=0; break; }
		cerr << which << "\n";
		ColumnMatrix* cm=new ColumnMatrix(mh->nrows());
		double *data = cm->get_rhs();
		for (int r=0; r<mh->nrows(); r++) data[r]=mh->get(r, which);
		ColumnMatrixHandle cmh(cm);
		ovec->send_intermediate(cmh);
	    }
	}	    
	if (useRow) {
	    ColumnMatrix* cm=new ColumnMatrix(mh->ncols());
	    double *data = cm->get_rhs();
	    for (int c=0; c<mh->ncols(); c++) data[c]=mh->get(which, c);
	} else {
	    ColumnMatrix* cm=new ColumnMatrix(mh->nrows());
	    double *data = cm->get_rhs();
	    for (int r=0; r<mh->nrows(); r++) data[r]=mh->get(r, which);
	}	    
	ColumnMatrixHandle cmh(cm);
	ovec->send(cmh);
    }
}    


void MatSelectVec::tcl_command(TCLArgs& args, void* userdata)
{
    if(args.count() < 2) {
	args.error("MatSelectVec needs a minor command");
	return;
    }
    if(args[1] == "stop") stop=1;
    else Module::tcl_command(args, userdata);
}
} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.1  1999/09/07 04:02:23  dmw
// more modules that were left behind...
//
