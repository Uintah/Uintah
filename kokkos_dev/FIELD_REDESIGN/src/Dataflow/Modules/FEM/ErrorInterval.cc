//static char *id="@(#) $Id$";

/*
 *  ErrorInterval.cc: Evaluate the error in a finite element solution
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   May 1996
 *
 *  Copyright (C) 1996 SCI Group
 */

#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/ColorMapPort.h>
#include <PSECore/Datatypes/IntervalPort.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/TclInterface/TCLvar.h>

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::GeomSpace;

class ErrorInterval : public Module {
    ColorMapIPort* icmap;
    IntervalOPort* ointerval;
    ColorMapOPort* ocmap;
    TCLdouble low;
    TCLdouble high;
public:
    ErrorInterval(const clString& id);
    virtual ~ErrorInterval();
    virtual void execute();
};

extern "C" Module* make_ErrorInterval(const clString& id) {
  return new ErrorInterval(id);
}

ErrorInterval::ErrorInterval(const clString& id)
: Module("ErrorInterval", id, Filter), low("low", id, this),
  high("high", id, this)
{
    icmap=new ColorMapIPort(this, "ColorMap", ColorMapIPort::Atomic);
    add_iport(icmap);
    ointerval=new IntervalOPort(this, "Interval", IntervalIPort::Atomic);
    add_oport(ointerval);
    ocmap=new ColorMapOPort(this, "ColorMap", ColorMapIPort::Atomic);
    add_oport(ocmap);
}

ErrorInterval::~ErrorInterval()
{
}

void ErrorInterval::execute()
{
    Interval* interval=new Interval(low.get(), high.get());
    ointerval->send(interval);
    ColorMapHandle cmap;
    if(icmap->get(cmap)){
        cmap.detach();
	cmap->min=low.get();
	cmap->max=high.get();
        ocmap->send(cmap);
    }
}

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.7  2000/03/17 09:26:52  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.6  1999/09/08 02:26:33  sparker
// Various #include cleanups
//
// Revision 1.5  1999/08/25 03:47:45  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.4  1999/08/19 23:17:42  sparker
// Removed a bunch of #include <SCICore/Util/NotFinished.h> statements
// from files that did not need them.
//
// Revision 1.3  1999/08/18 20:19:37  sparker
// Eliminated copy constructor and clone in all modules
// Added a private copy ctor and a private clone method to Module so
//  that future modules will not compile until they remvoe the copy ctor
//  and clone method
// Added an ASSERTFAIL macro to eliminate the "controlling expression is
//  constant" warnings.
// Eliminated other miscellaneous warnings
//
// Revision 1.2  1999/08/17 06:37:25  sparker
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
