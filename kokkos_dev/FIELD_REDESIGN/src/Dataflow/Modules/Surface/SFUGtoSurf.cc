//static char *id="@(#) $Id$";

/*
 *  SFUGtoSurf.cc:  Dumb module -- blow away elements and make nodes into
 *			a trisurface with no triangles
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1998
 *
 *  Copyright (C) 1998 SCI Group
 */

#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/ScalarFieldPort.h>
#include <SCICore/Datatypes/ScalarFieldUG.h>
#include <PSECore/Datatypes/SurfacePort.h>
#include <SCICore/Datatypes/TriSurface.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <stdio.h>
#include <SCICore/Malloc/Allocator.h>

namespace PSECommon {
namespace Modules {

using PSECore::Dataflow::Module;
using PSECore::Datatypes::SurfaceIPort;
using PSECore::Datatypes::SurfaceOPort;
using PSECore::Datatypes::SurfaceHandle;
using PSECore::Datatypes::TriSurface;
using PSECore::Datatypes::ScalarFieldIPort;
using PSECore::Datatypes::ScalarFieldHandle;
using PSECore::Datatypes::ScalarFieldUG;

using SCICore::Containers::clString;

class SFUGtoSurf : public Module {
    ScalarFieldIPort* isf;
    SurfaceOPort* osurf;
public:
    SFUGtoSurf(const clString& id);
    virtual ~SFUGtoSurf();
    virtual void execute();
};

extern "C" Module* make_SFUGtoSurf(const clString& id) {
  return new SFUGtoSurf(id);
}

static clString module_name("SFUGtoSurf");

SFUGtoSurf::SFUGtoSurf(const clString& id)
: Module("SFUGtoSurf", id, Filter)
{
    isf=scinew ScalarFieldIPort(this, "SF", ScalarFieldIPort::Atomic);
    add_iport(isf);
    // Create the output port
    osurf=scinew SurfaceOPort(this, "Surface", SurfaceIPort::Atomic);
    add_oport(osurf);
}

SFUGtoSurf::~SFUGtoSurf()
{
}

void SFUGtoSurf::execute() {
    ScalarFieldHandle isfh;
    if(!isf->get(isfh))
	return;
    ScalarFieldUG* sfug=isfh->getUG();
    if (!sfug) return;
    TriSurface *ts=scinew TriSurface;
    int npts=sfug->mesh->nodes.size();
    ts->points.resize(npts);
    for (int i=0; i<npts; i++) {
	ts->points[i]=sfug->mesh->nodes[i]->p;
	if (sfug->mesh->nodes[i]->bc) {
	    ts->bcIdx.add(i);
	    ts->bcVal.add(sfug->mesh->nodes[i]->bc->value);
	}
    }
    SurfaceHandle osh(ts);
    osurf->send(osh);
    return;
}

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.6  2000/03/17 09:27:21  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.5  1999/08/25 03:48:00  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.4  1999/08/19 23:17:54  sparker
// Removed a bunch of #include <SCICore/Util/NotFinished.h> statements
// from files that did not need them.
//
// Revision 1.3  1999/08/18 20:19:57  sparker
// Eliminated copy constructor and clone in all modules
// Added a private copy ctor and a private clone method to Module so
//  that future modules will not compile until they remvoe the copy ctor
//  and clone method
// Added an ASSERTFAIL macro to eliminate the "controlling expression is
//  constant" warnings.
// Eliminated other miscellaneous warnings
//
// Revision 1.2  1999/08/17 06:37:43  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:57:58  mcq
// Initial commit
//
// Revision 1.2  1999/04/29 03:19:27  dav
// updates
//
// Revision 1.1.1.1  1999/04/24 23:12:31  dav
// Import sources
//
//
