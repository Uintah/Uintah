//static char *id="@(#) $Id$";

/*
 *  FieldGainCorrect.cc:  Unfinished modules
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   February 1995
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/ScalarFieldPort.h>
#include <SCICore/Datatypes/ScalarField.h>
#include <SCICore/Datatypes/ScalarFieldRGBase.h>
#include <SCICore/Datatypes/ScalarFieldRGuchar.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <stdio.h>
#include <math.h>
#include <iostream.h>

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::GeomSpace;

class FieldGainCorrect : public Module {
    ScalarFieldIPort* iField;
    ScalarFieldIPort* iGain;
    ScalarFieldOPort* oField;
    ScalarFieldHandle oFldHandle;
    ScalarFieldRGuchar* osf;
    ScalarFieldRGuchar* isf;
    ScalarFieldRGuchar* igain;
    TCLstring filterType;
    TCLint Offset;
    clString lastFT;
    int lastOffset;
    int genIFld;
    int genGain;
public:
    FieldGainCorrect(const clString& id);
    void subtractFields();
    void divideFields();
    void hybridFields();
    virtual ~FieldGainCorrect();
    virtual void execute();
};

Module* make_FieldGainCorrect(const clString& id) {
   return new FieldGainCorrect(id);
}

static clString module_name("FieldGainCorrect");
FieldGainCorrect::FieldGainCorrect(const clString& id)
: Module("FieldGainCorrect", id, Filter), filterType("filterType", id, this),
  Offset("Offset", id, this)
{
    iField=new ScalarFieldIPort(this, "SF", ScalarFieldIPort::Atomic);
    iGain=new ScalarFieldIPort(this, "Gain", ScalarFieldIPort::Atomic);
    oFldHandle=osf=0;
    add_iport(iField);
    add_iport(iGain);
    // Create the output ports
    oField=new ScalarFieldOPort(this, "SF", ScalarFieldIPort::Atomic);
    add_oport(oField);
    genIFld=genGain=-1;
    lastOffset=0;
}

FieldGainCorrect::~FieldGainCorrect()
{
}

void FieldGainCorrect::subtractFields()
{
    for (int i=0; i<isf->nx; i++) {
	for (int j=0; j<isf->ny; j++) {
	    for (int k=0; k<isf->nz; k++) {
		int val=isf->grid(i,j,k)-igain->grid(i,j,k);
		if (val<0) osf->grid(i,j,k)=0;
		else if (val>255) osf->grid(i,j,k)=255;
		else osf->grid(i,j,k)=val;
	    }
	}
    }
}

void FieldGainCorrect::hybridFields() {
    Array3<double> g;
    g.newsize(isf->nx, isf->ny, isf->nz);
    int i,j,k;
    for (i=0; i<isf->nx; i++) {
	for (j=0; j<isf->ny; j++) {
	    for (k=0; k<isf->nz; k++) {
		g(i,j,k)=((double) isf->grid(i,j,k))/(igain->grid(i,j,k)+lastOffset);
	    }
	}
    }
    cerr << "Done building g (1/3)...\n";
    double min, max;
    min=max=g(0,0,0);
    for (i=0; i<isf->nx; i++) {
	for (j=0; j<isf->ny; j++) {
	    for (k=0; k<isf->nz; k++) {
		if (g(i,j,k)>max) max=g(i,j,k);
		else if (g(i,j,k)<min) min=g(i,j,k);
	    }
	}
    }
    cerr << "Done building g (2/3)...\n";
    double ispan=255./(max-min);
    for (i=0; i<isf->nx; i++) {
	for (j=0; j<isf->ny; j++) {
	    for (k=0; k<isf->nz; k++) {
		osf->grid(i,j,k)=(g(i,j,k)-min)*ispan;
	    }
	}
    }
    cerr << "Done building g (3/3)!\n";
}
    
void FieldGainCorrect::divideFields() {
    Array3<double> g;
    g.newsize(isf->nx, isf->ny, isf->nz);
    int i,j,k;
    for (i=0; i<isf->nx; i++) {
	for (j=0; j<isf->ny; j++) {
	    for (k=0; k<isf->nz; k++) {
		if (igain->grid(i,j,k) == 0)
		    g(i,j,k)=0;
		else 
		    g(i,j,k)=((double) isf->grid(i,j,k))/(igain->grid(i,j,k));
	    }
	}
    }
    cerr << "Done building g (1/3)...\n";
    double min, max;
    min=max=g(0,0,0);
    for (i=0; i<isf->nx; i++) {
	for (j=0; j<isf->ny; j++) {
	    for (k=0; k<isf->nz; k++) {
		if (g(i,j,k)>max) max=g(i,j,k);
		else if (g(i,j,k)<min) min=g(i,j,k);
	    }
	}
    }
    cerr << "Done building g (2/3)...\n";
    double ispan=255./(max-min);
    for (i=0; i<isf->nx; i++) {
	for (j=0; j<isf->ny; j++) {
	    for (k=0; k<isf->nz; k++) {
		osf->grid(i,j,k)=(g(i,j,k)-min)*ispan;
	    }
	}
    }
    cerr << "Done building g (3/3)!\n";
}
    
void FieldGainCorrect::execute() {
    ScalarFieldHandle ifh;
    if(!iField->get(ifh))
	return;
    ScalarFieldRGBase *rgb=ifh->getRGBase();
    if(!rgb){
	error("FieldGainCorrect can't deal with unstructured grids!");
	return;
    }
    isf=rgb->getRGUchar();
    if(!isf){
	error("FieldGainCorrect can only deal with uchars.\n");
	return;
    }
    ScalarFieldHandle igfh;
    if(!iGain->get(igfh))
	return;
    rgb = igfh->getRGBase();
    if(!rgb){
	error("FieldGainCorrect can't deal with unstructured grids!");
	return;
    }
    igain=rgb->getRGUchar();
    if(!igain){
	error("FieldGainCorrect can only deal with uchars.\n");
	return;
    }
    if (isf->nx != igain->nx || isf->ny != igain->ny || isf->nz != igain->nz) {
	error("Input fields must have the same dimensions.\n");
	return;
    }

    int nOffset=Offset.get();
    clString nFT=filterType.get();
    if (genIFld==isf->generation && genGain==igain->generation && lastFT==nFT
	&& lastOffset==nOffset){
	oField->send(oFldHandle);
	return;
    }
    genIFld=isf->generation;
    genGain=igain->generation;
    lastFT=nFT;
    lastOffset=nOffset;
    Point min, max;
    igain->get_bounds(min, max);
    osf = new ScalarFieldRGuchar();
    osf->set_bounds(min, max);
    osf->resize(igain->nx, igain->ny, igain->nz);
    osf->grid.initialize(0);
    oFldHandle = osf;
    if (lastFT == "Subtract") {
	subtractFields();
    } else if (lastFT == "Divide") {
	divideFields();
    } else if (lastFT == "Hybrid") {
	hybridFields();
    } else {
	cerr << "Don't know gain correction filter: "<<lastFT<<"\n";
    }
    oField->send(oFldHandle);
}

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.5  1999/08/25 03:47:46  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.4  1999/08/19 23:17:43  sparker
// Removed a bunch of #include <SCICore/Util/NotFinished.h> statements
// from files that did not need them.
//
// Revision 1.3  1999/08/18 20:19:39  sparker
// Eliminated copy constructor and clone in all modules
// Added a private copy ctor and a private clone method to Module so
//  that future modules will not compile until they remvoe the copy ctor
//  and clone method
// Added an ASSERTFAIL macro to eliminate the "controlling expression is
//  constant" warnings.
// Eliminated other miscellaneous warnings
//
// Revision 1.2  1999/08/17 06:37:27  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:57:41  mcq
// Initial commit
//
// Revision 1.2  1999/04/28 20:51:10  dav
// deleted some files that are dependent on DaveW files
//
// Revision 1.1.1.1  1999/04/24 23:12:32  dav
// Import sources
//
//
