
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

#include <Core/Containers/Array1.h>
#include <Core/Containers/String.h>
#include <Core/Persistent/Pstreams.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/ColumnMatrixPort.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Dataflow/Ports/SurfacePort.h>
#include <Core/Datatypes/TriSurface.h>
#include <Core/Geometry/Point.h>
#include <Core/Math/Expon.h>

#include <Core/TclInterface/TCLvar.h>

#include <iostream>
using std::cerr;
#include <stdio.h>

namespace SCIRun {


class SurfNewVals : public Module {
    SurfaceIPort* isurf;
    ColumnMatrixIPort* imat;
    SurfaceOPort* osurf;
    TCLstring surfid;
public:
    SurfNewVals(const clString& id);
    virtual ~SurfNewVals();
    virtual void execute();
};

extern "C" Module* make_SurfNewVals(const clString& id) {
  return new SurfNewVals(id);
}

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

SurfNewVals::~SurfNewVals()
{
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

#if 1
    TriSurface *nts = new TriSurface;
    int i;
    nts->points=ts->points;
    nts->normals=ts->normals;
    nts->normType=ts->normType;
    for (i=0; i<ts->points.size(); i++) {
	nts->bcVal.add((*(cmh.get_rep()))[i]);
	nts->bcIdx.add(i);
    }
    nts->elements.resize(ts->elements.size());
    for (i=0; i<ts->elements.size(); i++) {
	nts->elements[i]=new TSElement(*(ts->elements[i]));
    }
#else
    TriSurface *nts=new TriSurface(*ts);
    nts->bcIdx.resize(0);
    nts->bcVal.resize(0);
    for (int i=0; i<cmh->nrows(); i++) {
	nts->bcIdx.add(i);
	nts->bcVal.add((*(cmh.get_rep()))[i]);
    }
#endif

    nts->name=nts->name+clString("Fwd");
    SurfaceHandle sh2(nts);
    osurf->send(sh2);
}
} // End namespace SCIRun

