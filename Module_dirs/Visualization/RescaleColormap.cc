
/*
 *  RescaleColormap.cc:  Generate Color maps
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   November 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Dataflow/Module.h>
#include <Classlib/NotFinished.h>
#include <Dataflow/ModuleList.h>
#include <Datatypes/ColormapPort.h>
#include <Datatypes/Colormap.h>
#include <Datatypes/ScalarFieldPort.h>
#include <Datatypes/ScalarField.h>
#include <TCL/TCLvar.h>

class RescaleColormap : public Module {
    ColormapOPort* omap;
    ScalarFieldIPort* ifield;
    ColormapIPort* imap;
public:
    RescaleColormap(const clString& id);
    RescaleColormap(const RescaleColormap&, int deep);
    virtual ~RescaleColormap();
    virtual Module* clone(int deep);
    virtual void execute();
};

static Module* make_RescaleColormap(const clString& id)
{
    return new RescaleColormap(id);
}

static RegisterModule db1("Visualization", "RescaleColormap", make_RescaleColormap);

RescaleColormap::RescaleColormap(const clString& id)
: Module("RescaleColormap", id, Filter)
{
    // Create the output port
    omap=new ColormapOPort(this, "Colormap", ColormapIPort::Atomic);
    add_oport(omap);

    // Create the input ports
    imap=new ColormapIPort(this, "Colormap", ColormapIPort::Atomic);
    add_iport(imap);
    ifield=new ScalarFieldIPort(this, "ScalarField", ScalarFieldIPort::Atomic);
    add_iport(ifield);
}

RescaleColormap::RescaleColormap(const RescaleColormap& copy, int deep)
: Module(copy, deep)
{
    NOT_FINISHED("RescaleColormap::RescaleColormap");
}

RescaleColormap::~RescaleColormap()
{
}

Module* RescaleColormap::clone(int deep)
{
    return new RescaleColormap(*this, deep);
}

void RescaleColormap::execute()
{
    ColormapHandle cmap;
    if(!imap->get(cmap))
	return;
    ScalarFieldHandle sfield;
    if(ifield->get(sfield)){
	double min;
	double max;
	sfield->get_minmax(min, max);
	cmap.detach();
	cmap->min=min;
	cmap->max=max;
    }
    omap->send(cmap);
}
