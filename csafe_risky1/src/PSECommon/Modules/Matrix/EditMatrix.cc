//static char *id="@(#) $Id$";

/*
 *  EditMatrix.cc:  Visual matrix editor
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   November 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#include <SCICore/Containers/String.h>
#include <PSECore/Dataflow/Module.h>
#include <SCICore/Datatypes/DenseMatrix.h>
#include <SCICore/Datatypes/Matrix.h>
#include <PSECore/Datatypes/MatrixPort.h>
#include <SCICore/Datatypes/SymSparseRowMatrix.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <SCICore/TclInterface/TCL.h>
#include <tcl.h>
#include <iostream>
using std::cerr;
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::GeomSpace;
using namespace SCICore::Containers;

class EditMatrix : public Module {
    TCLstring Matrixtype;
    TCLstring TclMat;
    TCLint nrow;
    TCLint ncol;
    MatrixIPort* iport;
    MatrixOPort* oport;
    char *word;
    int loadFlag;
    int sendFlag;
    MatrixHandle omat;
    MatrixHandle imat;
public:
    EditMatrix(const clString& id);
    virtual ~EditMatrix();
    virtual void execute();
    void tcl_command( TCLArgs&, void * );
};

extern "C" Module* make_EditMatrix(const clString& id) {
  return new EditMatrix(id);
}

EditMatrix::EditMatrix(const clString& id)
: Module("EditMatrix", id, Source), Matrixtype("Matrixtype", id, this),
  TclMat("TclMat", id, this), nrow("nrow", id, this), ncol("ncol", id, this)
{
    // Create the input port
    iport=scinew MatrixIPort(this, "Matrix", MatrixIPort::Atomic);
    add_iport(iport);
    // Create the output port
    oport=scinew MatrixOPort(this, "Matrix", MatrixIPort::Atomic);
    add_oport(oport);
    loadFlag=sendFlag=1;
    word=0;
}

EditMatrix::~EditMatrix()
{
}

void EditMatrix::execute()
{
    int have_it=iport->get(imat);
    if (loadFlag) {
//	cerr << "Caught load flag\n";
	loadFlag=0;
	if (have_it && imat.get_rep()) {
	    int r,c;
	    Array1<clString> a(imat->nrows()*imat->ncols());
	    for (r=0; r<imat->nrows(); r++) {
		for (c=0; c<imat->ncols(); c++) {
		    a[r*imat->ncols()+c] = to_string(imat->get(r,c));
		}
	    }
	    int space=0;
	    for (r=0; r<a.size(); r++) {
		space += a[r].len() + 1;
	    }
	    if (word) free(word);
//	    cerr << "Allocating " << space+1 << " chars.\n";
	    word = (char *) malloc (sizeof(char)*(space+2));
	    int curr=0;
	    for (r=0; r<a.size(); r++) {
		strncpy(&(word[curr]), (a[r])(), a[r].len());
		curr+=a[r].len();
		word[curr]=' ';
		curr++;
	    }
	    word[curr]='\0';
//	    cerr << "Here's the outgoing string: " << word << "\n";
	    TclMat.set(word);
	    nrow.set(imat->nrows());
	    ncol.set(imat->ncols());
	    cerr << "Loaded type = "<<imat->getType()<<"\n";
	    DenseMatrix *dm=imat->getDense();
	    if (dm){
		cerr << "Loaded matrix is dense\n";
		Matrixtype.set("dense");
	    } else {
		cerr << "Loaded matrix isn't dense\n";
		Matrixtype.set("symsparse");
	    }
cerr << "Done with c++, calling tcl_load...\n";
	    TCL::execute(id+" tcl_load");
	}
    }
    if (sendFlag) {
//	cerr << "Caught send flag\n";
	sendFlag=0;
	const char *m=(TclMat.get())();
	int nr = nrow.get();
	int nc = ncol.get();
	if (strlen(m)<nr*nc) return;
	clString st(Matrixtype.get());
	cerr << "st=" << st << "\n";
//	cerr << "m=" << m << "\n";
//	cerr << "Here's TclMat: "<<TclMat.get()<<"\n";
	if (word) free(word);
	char *word = (char *) malloc (sizeof(char) * strlen(m+1));
	// Dd: Using "word" instead of "m" below... should work...
	strcpy(word, m);
//	cerr << "r=" <<r<<"  c="<<c << "\n";
	char *next;
	if (st == "dense") {
	    DenseMatrix* dm = scinew DenseMatrix(nr, nc);
	    int r,c;
	    for (r=0; r<nr; r++) {
		for (c=0; c<nc; c++) {
		    if (r==0 && c==0) {
			next=strtok(word, " ");
		    } else {
			next=strtok((char *)0, " ");
		    }
//		    cerr << "Read matrix element("<<r<<","<<c<<"): "<<atof(next)<<"\n";
		    dm->put(r,c,atof(next));
		}
	    }
	    cerr << "Sending a dense matrix.\n";
	    omat = dm;
	} else {
	    Array1<double> val;
	    Array1<int> ridx;
	    Array1<int> cidx;
	    int r,c;
	    for (r=0; r<nr; r++) {
		for (c=0; c<nc; c++) {
		    if (r==0 && c==0) {
			next=strtok(word, " ");
		    } else {
			next=strtok((char *)0, " ");
		    }
		    double v=atof(next);
		    if (v!=0) {
			val.add(v);
			ridx.add(r);
			cidx.add(c);
		    }
		}
	    }
	    Array1<int> in_rows;
	    int z;
	    if (ridx.size()!=0) {
		int curr_r=-1;
		for (z=0; z<ridx.size(); z++) {
		    if (curr_r != ridx[z]) {
			curr_r++;
			in_rows.add(z);
			z--;
		    }
		}
		for (z=in_rows.size(); z<=nrow.get(); z++) {
		    in_rows.add(ridx.size());
		}
	    }
//	    cerr << "SSRM: nrow="<<nrow.get()<<" ncol="<<ncol.get()<<"\n";
//	    cerr << "  in_rows("<<in_rows.size()<<"): ";
//	    int i;
//	    for (i=0; i<in_rows.size(); i++) {
//		cerr <<in_rows[i]<<" ";
//	    }
//	    cerr << "\n  in_cols("<<cidx.size()<<"): ";
//	    for (i=0; i<cidx.size(); i++) {
//		cerr <<cidx[i]<<" ";
//	    }
//	    cerr << "\n\n";
	    SymSparseRowMatrix* ssm = 
		scinew SymSparseRowMatrix(nrow.get(), ncol.get(), in_rows, cidx);
	    for (z=0; z<val.size(); z++)
		ssm->put(ridx[z], cidx[z], val[z]);
	    cerr << "Sending a sparse matrix.\n";
	    omat = ssm;
	}
    }
    oport->send(omat);
}

void EditMatrix::tcl_command(TCLArgs& args, void* userdata) {
    if (args[1] == "load") {
	loadFlag=1;
	want_to_execute();
    } else if (args[1] == "send") {
	sendFlag=1;
	want_to_execute();
    } else {
        Module::tcl_command(args, userdata);
    }
}

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.7  2000/03/17 09:27:06  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.6  1999/10/07 02:06:51  sparker
// use standard iostreams and complex type
//
// Revision 1.5  1999/08/25 03:47:51  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.4  1999/08/19 23:17:47  sparker
// Removed a bunch of #include <SCICore/Util/NotFinished.h> statements
// from files that did not need them.
//
// Revision 1.3  1999/08/18 20:19:45  sparker
// Eliminated copy constructor and clone in all modules
// Added a private copy ctor and a private clone method to Module so
//  that future modules will not compile until they remvoe the copy ctor
//  and clone method
// Added an ASSERTFAIL macro to eliminate the "controlling expression is
//  constant" warnings.
// Eliminated other miscellaneous warnings
//
// Revision 1.2  1999/08/17 06:37:31  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:57:45  mcq
// Initial commit
//
// Revision 1.3  1999/07/07 21:10:23  dav
// added beginnings of support for g++ compilation
//
// Revision 1.2  1999/04/27 22:57:49  dav
// updates in Modules for Datatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:31  dav
// Import sources
//
//
