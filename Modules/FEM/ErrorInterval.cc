
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

#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Datatypes/ColormapPort.h>
#include <Datatypes/IntervalPort.h>
#include <Geometry/Point.h>
#include <Malloc/Allocator.h>
#include <TCL/TCLvar.h>

class ErrorInterval : public Module {
    ColormapIPort* icmap;
    IntervalOPort* ointerval;
    ColormapOPort* ocmap;
    TCLdouble low;
    TCLdouble high;
public:
    ErrorInterval(const clString& id);
    ErrorInterval(const ErrorInterval&, int deep);
    virtual ~ErrorInterval();
    virtual Module* clone(int deep);
    virtual void execute();
};

extern "C" {
Module* make_ErrorInterval(const clString& id)
{
    return new ErrorInterval(id);
}
};

ErrorInterval::ErrorInterval(const clString& id)
: Module("ErrorInterval", id, Filter), low("low", id, this),
  high("high", id, this)
{
    icmap=new ColormapIPort(this, "Colormap", ColormapIPort::Atomic);
    add_iport(icmap);
    ointerval=new IntervalOPort(this, "Interval", IntervalIPort::Atomic);
    add_oport(ointerval);
    ocmap=new ColormapOPort(this, "Colormap", ColormapIPort::Atomic);
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
    ColormapHandle cmap;
    if(icmap->get(cmap)){
        cmap.detach();
	cmap->min=low.get();
	cmap->max=high.get();
        ocmap->send(cmap);
    }
}


  

