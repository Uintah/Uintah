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

#include <SCICore/Util/NotFinished.h>
#include <PSECore/Dataflow/Module.h>
#include <PSECore/CommonDatatypes/ColorMapPort.h>
#include <PSECore/CommonDatatypes/IntervalPort.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCLvar.h>

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::CommonDatatypes;
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
    ErrorInterval(const ErrorInterval&, int deep);
    virtual ~ErrorInterval();
    virtual Module* clone(int deep);
    virtual void execute();
};

Module* make_ErrorInterval(const clString& id) {
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

ErrorInterval::ErrorInterval(const ErrorInterval& copy, int deep)
: Module(copy, deep), low("low", id, this),
  high("high", id, this)
{
}

ErrorInterval::~ErrorInterval()
{
}

Module* ErrorInterval::clone(int deep)
{
    return new ErrorInterval(*this, deep);
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
