//static char *id="@(#) $Id$";

/*
 *  ApplyBC.cc:  Unfinished modules
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <SCICore/Util/NotFinished.h>
#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/SurfacePort.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Malloc/Allocator.h>

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::GeomSpace;

class ApplyBC : public Module {
    SurfaceIPort* iport;
    SurfaceOPort* oport;
public:
    ApplyBC(const clString& id);
    virtual ~ApplyBC();
    virtual void execute();
};

Module* make_ApplyBC(const clString& id) {
  return new ApplyBC(id);
}

ApplyBC::ApplyBC(const clString& id)
: Module("ApplyBC", id, Filter)
{
    iport=scinew SurfaceIPort(this, "Geometry", SurfaceIPort::Atomic);
    add_iport(iport);
    // Create the output port
    oport=scinew SurfaceOPort(this, "Geometry", SurfaceIPort::Atomic);
    add_oport(oport);
}

ApplyBC::~ApplyBC()
{
}

void ApplyBC::execute()
{
    SurfaceHandle surf;
    if(!iport->get(surf))
	return;
    NOT_FINISHED("ApplyBC::execute");
    oport->send(surf);
}

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.4  1999/08/25 03:47:44  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.3  1999/08/18 20:19:36  sparker
// Eliminated copy constructor and clone in all modules
// Added a private copy ctor and a private clone method to Module so
//  that future modules will not compile until they remvoe the copy ctor
//  and clone method
// Added an ASSERTFAIL macro to eliminate the "controlling expression is
//  constant" warnings.
// Eliminated other miscellaneous warnings
//
// Revision 1.2  1999/08/17 06:37:24  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:57:39  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:30  dav
// Import sources
//
//
