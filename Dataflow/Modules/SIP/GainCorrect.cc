
/*
 *  GainCorrect.cc:  Unfinished modules
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   February 1995
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/ScalarFieldPort.h>
#include <Core/Datatypes/ScalarField.h>
#include <Core/Datatypes/ScalarFieldRGBase.h>
#include <Core/Datatypes/ScalarFieldRGuchar.h>
#include <Core/TclInterface/TCLvar.h>
#include <stdio.h>
#include <math.h>
#include <iostream>
using std::cerr;

namespace SCIRun {


class GainCorrect : public Module {
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
    GainCorrect(const clString& id);
    void subtractFields();
    void divideFields();
    void hybridFields();
    virtual ~GainCorrect();
    virtual void execute();
};

extern "C" Module* make_GainCorrect(const clString& id) {
   return new GainCorrect(id);
}

static clString module_name("GainCorrect");
GainCorrect::GainCorrect(const clString& id)
: Module("GainCorrect", id, Filter), filterType("filterType", id, this),
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

GainCorrect::~GainCorrect()
{
}

void GainCorrect::subtractFields()
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

void GainCorrect::hybridFields() {
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
    
void GainCorrect::divideFields() {
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
    
void GainCorrect::execute() {
    ScalarFieldHandle ifh;
    if(!iField->get(ifh))
	return;
    ScalarFieldRGBase *rgb=ifh->getRGBase();
    if(!rgb){
	error("GainCorrect can't deal with unstructured grids!");
	return;
    }
    isf=rgb->getRGUchar();
    if(!isf){
	error("GainCorrect can only deal with uchars.\n");
	return;
    }
    ScalarFieldHandle igfh;
    if(!iGain->get(igfh))
	return;
    rgb = igfh->getRGBase();
    if(!rgb){
	error("GainCorrect can't deal with unstructured grids!");
	return;
    }
    igain=rgb->getRGUchar();
    if(!igain){
	error("GainCorrect can only deal with uchars.\n");
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

} // End namespace SCIRun

