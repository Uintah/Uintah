//static char *id="@(#) $Id$";

/*
 *  FieldMedianFilter.cc:  Unfinished modules
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   February 1995
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <SCICore/Containers/FLPQueue.h>
#include <PSECore/Dataflow/Module.h>
#include <PSECore/CommonDatatypes/ScalarFieldPort.h>
#include <SCICore/CoreDatatypes/ScalarField.h>
#include <SCICore/CoreDatatypes/ScalarFieldRGBase.h>
#include <SCICore/CoreDatatypes/ScalarFieldRGuchar.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <stdio.h>
#include <math.h>
#include <iostream.h>

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::CommonDatatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::GeomSpace;

class QNode {
    inline int isequal(const QNode&) const {return 0;}
};

class FieldMedianFilter : public Module {
    ScalarFieldIPort* iField;
    ScalarFieldOPort* oField;
    ScalarFieldHandle oFldHandle;
    ScalarFieldRGuchar* osf;
    ScalarFieldRGuchar* isf;
    TCLint kernel;
    int lastKernel;
    int genIFld;
public:
    FieldMedianFilter(const clString& id);
    void filter(int ksize);
    virtual ~FieldMedianFilter();
    virtual void execute();
};

Module* make_FieldMedianFilter(const clString& id) {
   return new FieldMedianFilter(id);
}

static clString module_name("FieldMedianFilter");
FieldMedianFilter::FieldMedianFilter(const clString& id)
: Module("FieldMedianFilter", id, Filter), kernel("kernel", id, this)
{
    iField=new ScalarFieldIPort(this, "SF", ScalarFieldIPort::Atomic);
    oFldHandle=osf=0;
    add_iport(iField);
    // Create the output ports
    oField=new ScalarFieldOPort(this, "SF", ScalarFieldIPort::Atomic);
    add_oport(oField);
    genIFld=lastKernel=-1;
}

FieldMedianFilter::~FieldMedianFilter()
{
}

inline int ucharCompare(const void* first, const void* second) {uchar f=*((const uchar *)first); uchar s=*((const uchar *)second); if (f<s) return -1; else if (f==s) return 0; else return 1;}

void FieldMedianFilter::filter(int ksize) {
    int kcubed=(ksize*2+1)*(ksize*2+1)*(ksize*2+1);
    int mid=kcubed/2;
    Array1<uchar> unsorted;
    cerr << "Kcubed = "<<kcubed<<"\n";
    unsorted.resize(kcubed);
    uchar *ucp;

    for (int i=ksize; i<isf->nx-ksize; i++) {
	for (int j=ksize; j<isf->ny-ksize; j++) {
	    for (int k=ksize; k<isf->nz-ksize; k++) {
		ucp=&(unsorted[0]);
		for (int ii=i-ksize; ii<=i+ksize; ii++) {
		    for (int jj=j-ksize; jj<=j+ksize; jj++) {
			for (int kk=k-ksize; kk<=k+ksize; kk++) {
			    isf->grid(ii,jj,kk)=*ucp;
			    ucp++;
			}
		    }
		}
		qsort(&(unsorted[0]), kcubed, sizeof(uchar), ucharCompare);
		osf->grid(i,j,k)=unsorted[mid];
	    }
	}
    }
}

void FieldMedianFilter::execute() {
    ScalarFieldHandle ifh;
    if(!iField->get(ifh))
	return;
    ScalarFieldRGBase *rgb=ifh->getRGBase();
    if(!rgb){
	error("FieldMedianFilter can't deal with unstructured grids!");
	return;
    }
    isf=rgb->getRGUchar();
    if(!isf){
	error("FieldMedianFilter can only deal with uchars.\n");
	return;
    }

    int nKernel=kernel.get();
    if (genIFld==isf->generation && lastKernel==nKernel)
	return;
    genIFld=isf->generation;
    lastKernel=nKernel;

    Point min, max;
    isf->get_bounds(min, max);
    osf = new ScalarFieldRGuchar();
    osf->set_bounds(min, max);
    osf->resize(isf->nx, isf->ny, isf->nz);
    osf->grid.initialize(0);
    oFldHandle = osf;
    filter(lastKernel);
    oField->send(oFldHandle);
}

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
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
