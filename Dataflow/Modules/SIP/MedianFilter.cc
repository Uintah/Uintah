
/*
 *  MedianFilter.cc:  Unfinished modules
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   February 1995
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Core/Containers/FLPQueue.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/ScalarFieldPort.h>
#include <Core/Datatypes/ScalarField.h>
#include <Core/Datatypes/ScalarFieldRG.h>
#include <Core/TclInterface/TCLvar.h>
#include <stdio.h>
#include <math.h>
#include <iostream>
using std::cerr;

namespace SCIRun {


class QNode {
    inline int isequal(const QNode&) const {return 0;}
};

class MedianFilter : public Module {
    ScalarFieldIPort* iField;
    ScalarFieldOPort* oField;
    ScalarFieldHandle oFldHandle;
    ScalarFieldRGuchar* osf;
    ScalarFieldRGuchar* isf;
    TCLint kernel;
    int lastKernel;
    int genIFld;
public:
    MedianFilter(const clString& id);
    void filter(int ksize);
    virtual ~MedianFilter();
    virtual void execute();
};

extern "C" Module* make_MedianFilter(const clString& id) {
   return new MedianFilter(id);
}

static clString module_name("MedianFilter");
MedianFilter::MedianFilter(const clString& id)
: Module("MedianFilter", id, Filter), kernel("kernel", id, this)
{
    iField=new ScalarFieldIPort(this, "SF", ScalarFieldIPort::Atomic);
    oFldHandle=osf=0;
    add_iport(iField);
    // Create the output ports
    oField=new ScalarFieldOPort(this, "SF", ScalarFieldIPort::Atomic);
    add_oport(oField);
    genIFld=lastKernel=-1;
}

MedianFilter::~MedianFilter()
{
}

inline int ucharCompare(const void* first, const void* second) {uchar f=*((const uchar *)first); uchar s=*((const uchar *)second); if (f<s) return -1; else if (f==s) return 0; else return 1;}

void MedianFilter::filter(int ksize) {
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

void MedianFilter::execute() {
    ScalarFieldHandle ifh;
    if(!iField->get(ifh))
	return;
    ScalarFieldRGBase *rgb=ifh->getRGBase();
    if(!rgb){
	error("MedianFilter can't deal with unstructured grids!");
	return;
    }
    isf=rgb->getRGUchar();
    if(!isf){
	error("MedianFilter can only deal with uchars.\n");
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

} // End namespace SCIRun

