
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

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/SurfacePort.h>
#include <Core/Containers/String.h>
#include <Core/Containers/Array1.h>
#include <Core/Datatypes/SurfTree.h>
#include <Core/Geometry/Point.h>
#include <Core/Persistent/Pstreams.h>
#include <Core/GuiInterface/GuiVar.h>

#include <iostream>
using std::cerr;
#include <stdio.h>

namespace DaveW {
using namespace SCIRun;

class STreeExtractSurf : public Module {
    SurfaceIPort* istree;
    SurfaceOPort* osurf;
    GuiString surfid;
    GuiInt remapTCL;
public:
    STreeExtractSurf(const clString& id);
    virtual ~STreeExtractSurf();
    virtual void execute();
};

extern "C" Module* make_STreeExtractSurf(const clString& id)
{
    return new STreeExtractSurf(id);
}

STreeExtractSurf::STreeExtractSurf(const clString& id)
: Module("STreeExtractSurf", id, Filter), surfid("surfid", id, this),
  remapTCL("remapTCL", id, this)
{
    istree=new SurfaceIPort(this, "SurfTreeIn", SurfaceIPort::Atomic);
    add_iport(istree);

    // Create the output port
    osurf=new SurfaceOPort(this, "SurfOut", SurfaceIPort::Atomic);
    add_oport(osurf);
}

STreeExtractSurf::~STreeExtractSurf()
{
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

    TriSurfFieldace *ts=0;
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

    ts = new TriSurfFieldace;
    if (!st->extractTriSurfFieldace(ts, map, imap, comp, remapTCL.get())) {
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
} // End namespace DaveW



