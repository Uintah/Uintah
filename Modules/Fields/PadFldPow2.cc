/*
 *  PadFldPow2.cc:  Unfinished modules
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   February 1995
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Classlib/Array2.h>
#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Datatypes/ScalarFieldPort.h>
#include <TCL/TCLvar.h>
#include <Classlib/Pstreams.h>
#include <Datatypes/ScalarFieldPort.h>
#include <Datatypes/ScalarField.h>
#include <Datatypes/ScalarFieldRGBase.h>
#include <Datatypes/ScalarFieldRG.h>
#include <Geometry/BBox.h>
#include <Geometry/Point.h>
#include <Geometry/Vector.h>
#include <Math/MinMax.h>
#include <stdio.h>
#include <math.h>
#include <iostream.h>
#include <fstream.h>

class PadFldPow2 : public Module {
    ScalarFieldIPort* iField;
    ScalarFieldOPort* oField;
public:
    PadFldPow2(const clString& id);
    PadFldPow2(const PadFldPow2&, int deep);
    virtual ~PadFldPow2();
    virtual Module* clone(int deep);
    virtual void execute();
    TCLdouble padvalTCL;
};

extern "C" {
Module* make_PadFldPow2(const clString& id)
{
    return new PadFldPow2(id);
}
}

static clString module_name("PadFldPow2");
PadFldPow2::PadFldPow2(const clString& id)
: Module("PadFldPow2", id, Filter), padvalTCL("padvalTCL", id, this)
{
    iField=new ScalarFieldIPort(this, "Geometry", ScalarFieldIPort::Atomic);
    add_iport(iField);
    // Create the output ports
    oField=new ScalarFieldOPort(this, "Geometry", ScalarFieldIPort::Atomic);
    add_oport(oField);
}

PadFldPow2::PadFldPow2(const PadFldPow2& copy, int deep)
: Module(copy, deep), padvalTCL("padvalTCL", id, this)
{
}

PadFldPow2::~PadFldPow2()
{
}

Module* PadFldPow2::clone(int deep)
{
    return new PadFldPow2(*this, deep);
}

int nextPowerOfTwo(int v) {
    int z=1;
    v=v-1;

    while(v) { z=z<<1; v=v>>1; }
    return z;
}

void PadFldPow2::execute() {
    ScalarFieldHandle ifh;
    if(!iField->get(ifh))
	return;
    ScalarFieldRG*osf=ifh->getRG();
    if(!osf){
	error("PadFldPow2 can only deal with doubles (use SFRGfile to cast).");
	return;
    }
    Point min, max;
    osf->get_bounds(min,max);

    int ox=osf->nx;
    int oy=osf->ny;
    int oz=osf->nz;

    int nx, ny, nz;
    nx=nextPowerOfTwo(ox);
    ny=nextPowerOfTwo(oy);
    nz=nextPowerOfTwo(oz);
    ScalarFieldRG* nsf = new ScalarFieldRG;
    nsf->resize(nx,ny,nz);

    Vector d(max-min);
    d.x(d.x()/(ox-1));
    d.y(d.y()/(oy-1));
    d.z(d.z()/(oz-1));

    
    int px=(nx-ox)/2;
    int py=(ny-oy)/2;
    int pz=(nz-oz)/2;

    Point newMin(min);
    newMin.x(newMin.x()-px*d.x());
    newMin.y(newMin.y()-py*d.y());
    newMin.z(newMin.z()-pz*d.z());

    Point newMax(newMin);
    newMax.x(newMax.x()+(nx-1)*d.x());
    newMax.y(newMax.y()+(ny-1)*d.y());
    newMax.z(newMax.z()+(nz-1)*d.z());

    cerr << "Going from field ("<<ox<<","<<oy<<","<<oz<<") to ("<<nx<<","<<ny<<","<<nz<<")\n";
    cerr << "  and from "<<min<<"-"<<max<<"  to  "<<newMin<<"-"<<newMax<<"\n";

    nsf->set_bounds(newMin, newMax);

    double padval=padvalTCL.get();

    int i,j,k;
    for (i=0; i<nx; i++)
	for (j=0; j<ny; j++)
	    for (k=0; k<nz; k++) 
		nsf->grid(i,j,k)=padval;

    for (i=px; i<px+ox; i++)
	for (j=py; j<py+oy; j++)
	    for (k=pz; k<pz+oz; k++)
		nsf->grid(i,j,k)=osf->grid(i-px, j-py, k-pz);

    ScalarFieldHandle nfh(nsf);
    oField->send(nfh);
}
