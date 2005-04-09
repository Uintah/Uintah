
/*
 *  RescaleSegFld.cc:  Rescale a SegFld
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
#include <Dataflow/ModuleList.h>
#include <Datatypes/SegFldPort.h>
#include <Datatypes/SegFld.h>
#include <Geometry/BBox.h>
#include <Math/Expon.h>
#include <Math/MusilRNG.h>
#include <TCL/TCLvar.h>
#include <stdio.h>
#include <Malloc/Allocator.h>

class RescaleSegFld : public Module {
    SegFldIPort* iSegFld;
    SegFldOPort* oSegFld;
    TCLint coreg;
    TCLdouble scale;
    SegFldHandle osh;
    double last_scale;
    int generation;
public:
    RescaleSegFld(const clString& id);
    RescaleSegFld(const RescaleSegFld&, int deep);
    virtual ~RescaleSegFld();
    virtual Module* clone(int deep);
    virtual void execute();
};

extern "C" {
Module* make_RescaleSegFld(const clString& id)
{
    return new RescaleSegFld(id);
}
};

static clString module_name("RescaleSegFld");

RescaleSegFld::RescaleSegFld(const clString& id)
: Module("RescaleSegFld", id, Filter),
  scale("scale", id, this), coreg("coreg", id, this),
  generation(-1), last_scale(0)
{
    iSegFld=scinew SegFldIPort(this, "SegFld", SegFldIPort::Atomic);
    add_iport(iSegFld);
    // Create the output port
    oSegFld=scinew SegFldOPort(this, "SegFld", SegFldIPort::Atomic);
    add_oport(oSegFld);
}

RescaleSegFld::RescaleSegFld(const RescaleSegFld& copy, int deep)
: Module(copy, deep),
  scale("scale", id, this), coreg("coreg", id, this),
  generation(-1), last_scale(0)
{
    NOT_FINISHED("RescaleSegFld::RescaleSegFld");
}

RescaleSegFld::~RescaleSegFld()
{
}

Module* RescaleSegFld::clone(int deep)
{
    return new RescaleSegFld(*this, deep);
}

void RescaleSegFld::execute()
{
    SegFldHandle isurf;
    if(!iSegFld->get(isurf))
	return;

    double new_scale=scale.get();
    double s=pow(10.,new_scale);
	
    Point min, max;
    isurf->get_bounds(min, max);
    if (coreg.get()) {
	BBox b;
	b.extend(min);
	b.extend(max);
	s=1./b.longest_edge()*1.7;
	cerr << "b.min="<<b.min()<<" b.max="<<b.max()<<"   s="<<s<<"\n";
	scale.set(log10(s));
	reset_vars();
	new_scale=scale.get();
    }

    if (generation == isurf->generation && new_scale == last_scale) {
	oSegFld->send(isurf);
	return;
    }
    
    last_scale = new_scale;
    max.x(max.x()*s);
    max.y(max.y()*s);
    max.z(max.z()*s);
    isurf->set_bounds(min, max);
    oSegFld->send(isurf);
    return;
}
