//static char *id="@(#) $Id$";

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

#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/SurfacePort.h>
#include <SCICore/Containers/String.h>
#include <SCICore/Containers/Array1.h>
#include <SCICore/Datatypes/SurfTree.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Persistent/Pstreams.h>
#include <SCICore/TclInterface/TCLvar.h>

#include <iostream>
using std::cerr;
#include <stdio.h>

namespace DaveW {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::Datatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::Containers;
using namespace SCICore::Geometry;

class STreeExtractSurf : public Module {
    SurfaceIPort* istree;
    SurfaceOPort* osurf;
    TCLstring surfid;
    TCLint remapTCL;
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
    if (!st->extractTriSurface(ts, map, imap, comp, remapTCL.get())) {
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

} // End namespace Modules
} // End namespace DaveW


//
// $Log$
// Revision 1.6  2000/03/17 09:25:34  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.5  1999/11/17 00:32:00  dmw
// fixed a bug in Taubin (nrows has to equal ncols) and added a flag to STreeExtractSurf so the node numbers dont change
//
// Revision 1.4  1999/10/07 02:06:28  sparker
// use standard iostreams and complex type
//
// Revision 1.3  1999/09/08 02:26:23  sparker
// Various #include cleanups
//
// Revision 1.2  1999/08/25 03:47:39  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.1  1999/08/24 06:23:02  dmw
// Added in everything for the DaveW branch
//
// Revision 1.2  1999/05/03 04:52:13  dmw
// Added and updated DaveW Datatypes/Modules
//
//
