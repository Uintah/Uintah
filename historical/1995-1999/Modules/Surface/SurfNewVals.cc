
/*
 *  STreeToJAS: Read in a surface, and output a .tri and .pts file
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   October 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Classlib/Array1.h>
#include <Classlib/NotFinished.h>
#include <Classlib/String.h>
#include <Classlib/Pstreams.h>
#include <Dataflow/Module.h>
#include <Datatypes/ColumnMatrixPort.h>
#include <Datatypes/ColumnMatrix.h>
#include <Datatypes/SurfacePort.h>
#include <Datatypes/TriSurface.h>
#include <Geometry/Point.h>
#include <Malloc/Allocator.h>
#include <Math/Expon.h>

#include <TCL/TCLvar.h>

#include <iostream.h>
#include <stdio.h>

class SurfNewVals : public Module {
    SurfaceIPort* isurf;
    ColumnMatrixIPort* imat;
    SurfaceOPort* osurf;
    TCLstring surfid;
public:
    SurfNewVals(const clString& id);
    SurfNewVals(const SurfNewVals&, int deep);
    virtual ~SurfNewVals();
    virtual Module* clone(int deep);
    virtual void execute();
};

extern "C" {
Module* make_SurfNewVals(const clString& id)
{
    return new SurfNewVals(id);
}
};

SurfNewVals::SurfNewVals(const clString& id)
: Module("SurfNewVals", id, Filter), surfid("surfid", id, this)
{
    isurf=new SurfaceIPort(this, "SurfIn", SurfaceIPort::Atomic);
    add_iport(isurf);
    imat=new ColumnMatrixIPort(this, "MatIn", ColumnMatrixIPort::Atomic);
    add_iport(imat);
    // Create the output port
    osurf=new SurfaceOPort(this, "SurfOut", SurfaceIPort::Atomic);
    add_oport(osurf);
}

SurfNewVals::SurfNewVals(const SurfNewVals& copy, int deep)
: Module(copy, deep), surfid("surfid", id, this)
{
}

SurfNewVals::~SurfNewVals()
{
}

Module* SurfNewVals::clone(int deep)
{
    return new SurfNewVals(*this, deep);
}

void SurfNewVals::execute() {

    update_state(NeedData);

    SurfaceHandle sh;
    if (!isurf->get(sh))
	return;
    if (!sh.get_rep()) {
	cerr << "Error: empty surface\n";
	return;
    }
    TriSurface *ts=sh->getTriSurface();
    if (!ts) {
	cerr << "Error: surface isn't a trisurface\n";
	return;
    }

    update_state(JustStarted);
    
    ColumnMatrixHandle cmh;
    if (!imat->get(cmh)) return;
    if (!cmh.get_rep()) {
	cerr << "Error: empty columnmatrix\n";
	return;
    }

    TriSurface *nts = new TriSurface(*ts);
    double total2=0;
    double total=0;
    double error=0;
//    cerr << "Comparing new to old...\n";

#if 0
    for (int j=0; j<ts->bcIdx.size(); j++) {
#endif

    for (int j=0; j<64; j++) {
	nts->bcVal[j]=(*(cmh.get_rep()))[j];
	double tt=ts->bcVal[j];
	total += fabs(tt);
	total2 += fabs(nts->bcVal[j]);
	error += fabs(tt-nts->bcVal[j]);
//	cerr << j<<": "<< ts->bcVal[j]<<" -> "<<nts->bcVal[j]<<"\n";
    }

    cerr << "Total = "<<total<<"  total2 = "<<total2<<" error = "<<error<<"\n";
    cerr << "j="<<j<<"\n";
    cerr << "RMS Percent Error = "<<error*100/total<<" percent.\n";
    nts->name=nts->name+clString("Fwd");
    SurfaceHandle sh2(nts);
    osurf->send(sh2);
}    
