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

#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Datatypes/ScalarFieldPort.h>
#include <Datatypes/ScalarField.h>
#include <Datatypes/ScalarFieldRGBase.h>
#include <Datatypes/ScalarFieldRGuchar.h>
#include <TCL/TCLvar.h>
#include <stdio.h>
#include <math.h>
#include <iostream.h>

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
    FieldGainCorrect(const FieldGainCorrect&, int deep);
    void subtractFields();
    void divideFields();
    void hybridFields();
    virtual ~FieldGainCorrect();
    virtual Module* clone(int deep);
    virtual void execute();
};

extern "C" {
Module* make_FieldGainCorrect(const clString& id)
{
    return new FieldGainCorrect(id);
}
};

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

FieldGainCorrect::FieldGainCorrect(const FieldGainCorrect& copy, int deep)
: Module(copy, deep), filterType("filterType", id, this),
  Offset("Offset", id, this)
{
}

FieldGainCorrect::~FieldGainCorrect()
{
}

Module* FieldGainCorrect::clone(int deep)
{
    return new FieldGainCorrect(*this, deep);
}

void FieldGainCorrect::subtractFields() {
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

