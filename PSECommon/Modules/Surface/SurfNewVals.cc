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

#include <SCICore/Containers/Array1.h>
#include <SCICore/Containers/String.h>
#include <SCICore/Persistent/Pstreams.h>
#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/ColumnMatrixPort.h>
#include <SCICore/Datatypes/ColumnMatrix.h>
#include <PSECore/Datatypes/SurfacePort.h>
#include <SCICore/Datatypes/TriSurface.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/Math/Expon.h>

#include <SCICore/TclInterface/TCLvar.h>

#include <iostream.h>
#include <stdio.h>

namespace PSECommon {
namespace Modules {

using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::Containers;

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

Module* make_SurfNewVals(const clString& id) {
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
} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.6  1999/09/05 05:32:26  dmw
// updated and added Modules from old tree to new
//
// Revision 1.5  1999/08/25 03:48:01  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.4  1999/08/19 23:17:54  sparker
// Removed a bunch of #include <SCICore/Util/NotFinished.h> statements
// from files that did not need them.
//
// Revision 1.3  1999/08/18 20:19:59  sparker
// Eliminated copy constructor and clone in all modules
// Added a private copy ctor and a private clone method to Module so
//  that future modules will not compile until they remvoe the copy ctor
//  and clone method
// Added an ASSERTFAIL macro to eliminate the "controlling expression is
//  constant" warnings.
// Eliminated other miscellaneous warnings
//
// Revision 1.2  1999/08/17 06:37:44  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:57:59  mcq
// Initial commit
//
// Revision 1.2  1999/04/29 03:19:29  dav
// updates
//
// Revision 1.1.1.1  1999/04/24 23:12:31  dav
// Import sources
//
//
