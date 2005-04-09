
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
#include <Datatypes/SurfTree.h>
#include <Datatypes/SurfacePort.h>
#include <Geometry/Point.h>
#include <Malloc/Allocator.h>
#include <TCL/TCLvar.h>

#include <iostream.h>
#include <stdio.h>

class STreeExtractSurf : public Module {
    SurfaceIPort* istree;
    SurfaceOPort* osurf;
    TCLstring surfid;
public:
    STreeExtractSurf(const clString& id);
    STreeExtractSurf(const STreeExtractSurf&, int deep);
    virtual ~STreeExtractSurf();
    virtual Module* clone(int deep);
    virtual void execute();
};

extern "C" {
Module* make_STreeExtractSurf(const clString& id)
{
    return new STreeExtractSurf(id);
}
}

STreeExtractSurf::STreeExtractSurf(const clString& id)
: Module("STreeExtractSurf", id, Filter), surfid("surfid", id, this)
{
    istree=new SurfaceIPort(this, "SurfTreeIn", SurfaceIPort::Atomic);
    add_iport(istree);

    // Create the output port
    osurf=new SurfaceOPort(this, "SurfOut", SurfaceIPort::Atomic);
    add_oport(osurf);
}

STreeExtractSurf::STreeExtractSurf(const STreeExtractSurf& copy, int deep)
: Module(copy, deep), surfid("surfid", id, this)
{
}

STreeExtractSurf::~STreeExtractSurf()
{
}

Module* STreeExtractSurf::clone(int deep)
{
    return new STreeExtractSurf(*this, deep);
}

void STreeExtractSurf::execute() {

    update_state(NeedData);

    SurfaceHandle sh;
    if (!istree->get(sh))
	return;
    if (!sh.get_rep()) {
	cerr << "Error: empty surftree\n";
	return;
    }
    SurfTree *st=sh->getSurfTree();
    if (!st) {
	cerr << "Error: surface isn't a surftree\n";
	return;
    }

    TriSurface *ts=0;
    Array1<int> map;	// not used
    Array1<int> imap;	// not used

    update_state(JustStarted);

    int comp;
    clString cls=surfid.get();
    int ok;
    ok = cls.get_int(comp);
    if (!ok) {
	for (comp=0; comp<st->surfI.size(); comp++) {
	    if (st->surfI[comp].name == cls) {
		break;
	    }
	}
	if (comp == st->surfI.size()) {
	    cerr << "Error: bad surface name "<<cls<<"\n";
	    return;
	}
    }

//    cerr << "ST has "<<st->bcIdx.size()<<" vals...\n";
//    for (int i=0; i<st->bcIdx.size(); i++)
//	 cerr <<"  "<<i<<"  "<<st->bcVal[i]<<"  "<<st->points[st->bcIdx[i]]<<"\n";

    ts = new TriSurface;
    if (!st->extractTriSurface(ts, map, imap, comp)) {
	cerr << "Error, couldn't extract triSurface.\n";
	return;
    }

//    cerr << "surface11 "<<ts->name<<" has "<<ts->points.size()<<" points, "<<ts->elements.size()<<" elements and "<<ts->bcVal.size()<<" known vals.\n";

//    cerr << "TS has "<<ts->bcIdx.size()<<" vals...\n";
//    for (i=0; i<ts->bcIdx.size(); i++)
//	 cerr <<"  "<<i<<"  "<<ts->bcVal[i]<<"  "<<ts->points[ts->bcIdx[i]]<<"\n";

    SurfaceHandle sh2(ts);
    osurf->send(sh2);
}    
