
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
#include <SCICore/TclInterface/TCLvar.h>
#include <iostream>
using std::cerr;
#include <sstream>

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;

class MatSelectVec : public Module {
    MatrixIPort* imat;
    ColumnMatrixIPort* info;
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

extern "C" Module* make_MatSelectVec(const clString& id)
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
    info=new ColumnMatrixIPort(this,"SelectInfo", ColumnMatrixIPort::Atomic);
    add_iport(info);

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
    int usetcl=1;

    ColumnMatrixHandle infoH;
    MatrixHandle mh;
    if (!imat->get(mh))
	return;
    if (!mh.get_rep()) {
	cerr << "Error: empty matrix\n";
	return;
    }
    if (info->get(infoH)) {
	if (infoH.get_rep() && infoH->nrows() && ((infoH->nrows()%2)==0))
	    usetcl=0;
	else {
	    cerr << "Error: bad info vector in MatSelectVec\n";
	    return;
	}
    }

    if (!usetcl) {
	int useRow=(rowOrColTCL.get() == "row");
	double *i=infoH->get_rhs();
	int nsel=infoH->nrows()/2;
	ColumnMatrix *cm;
	if (useRow) {	// grab rows
	    cm=new ColumnMatrix(mh->ncols());
	    cm->zero();
	    double *data = cm->get_rhs();
	    int r, c;
	    for (r=0; r<nsel; r++)
		if (i[r*2+1])
		    for (c=0; c<mh->ncols(); c++) 
			data[c]+=mh->get(i[r*2], c)*i[r*2+1];
	} else {	// grab columns
	    cm=new ColumnMatrix(mh->nrows());
	    cm->zero();
	    double *data = cm->get_rhs();
	    int r, c;
	    for (c=0; c<nsel; c++)
		if (i[c*2+1])
		    for (r=0; r<mh->nrows(); r++) 
			data[r]+=mh->get(r, i[c*2])*i[c*2+1];
	}
	ColumnMatrixHandle cmh(cm);
	ovec->send(cmh);
	return;
    }
	
    int changed=0;

    update_state(JustStarted);

    if (colMaxTCL.get() != mh->ncols()-1) {
	colMaxTCL.set(mh->ncols()-1);
	changed=1;
    }
    if (rowMaxTCL.get() != mh->nrows()-1) {
	rowMaxTCL.set(mh->nrows()-1);
	changed=1;
    }
    if (changed) {
	std::ostringstream str;
	str << id << " update";
	TCL::execute(str.str().c_str());
    }

	

    reset_vars();

    int which;
    int useRow=(rowOrColTCL.get() == "row");
    if (useRow) which=rowTCL.get();
    else which=colTCL.get();

    if (!animateTCL.get()) {
	ColumnMatrix *cm;
	if (useRow) {
	    cm=new ColumnMatrix(mh->ncols());
	    double *data = cm->get_rhs();
	    for (int c=0; c<mh->ncols(); c++) data[c]=mh->get(which, c);
	} else {
	    cm=new ColumnMatrix(mh->nrows());
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
		cm=new ColumnMatrix(mh->ncols());
		double *data = cm->get_rhs();
		for (int c=0; c<mh->ncols(); c++) data[c]=mh->get(which, c);
		ColumnMatrixHandle cmh(cm);
		ovec->send_intermediate(cmh);
	    }
	} else {
	    for (; which<mh->ncols()-1; which++, colTCL.set(which)) {
		if (stop) { stop=0; break; }
		cerr << which << "\n";
		cm=new ColumnMatrix(mh->nrows());
		double *data = cm->get_rhs();
		for (int r=0; r<mh->nrows(); r++) data[r]=mh->get(r, which);
		ColumnMatrixHandle cmh(cm);
		ovec->send_intermediate(cmh);
	    }
	}	    
	if (useRow) {
	    cm=new ColumnMatrix(mh->ncols());
	    double *data = cm->get_rhs();
	    for (int c=0; c<mh->ncols(); c++) data[c]=mh->get(which, c);
	} else {
	    cm=new ColumnMatrix(mh->nrows());
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
// Revision 1.5.2.3  2000/11/01 23:02:56  mcole
// Fix for previous merge from trunk
//
// Revision 1.5.2.1  2000/09/28 03:16:02  mcole
// merge trunk into FIELD_REDESIGN branch
//
// Revision 1.6  2000/08/04 18:09:06  dmw
// added widget-based transform generation
//
// Revision 1.5  2000/03/17 09:27:07  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.4  1999/10/07 02:06:52  sparker
// use standard iostreams and complex type
//
// Revision 1.3  1999/09/16 00:38:11  dmw
// fixed TCL files for SurfToGeom and SolveMatrix and added SurfToGeom to the Makefile
//
// Revision 1.2  1999/09/08 02:26:34  sparker
// Various #include cleanups
//
// Revision 1.1  1999/09/07 04:02:23  dmw
// more modules that were left behind...
//
