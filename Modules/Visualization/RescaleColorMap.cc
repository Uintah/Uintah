
/*
 *  RescaleColorMap.cc:  Generate Color maps
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
#include <Datatypes/ColorMapPort.h>
#include <Datatypes/ColorMap.h>
#include <Datatypes/ScalarFieldPort.h>
#include <Datatypes/ScalarField.h>
#include <Malloc/Allocator.h>
#include <TCL/TCLvar.h>

class RescaleColorMap : public Module {
    ColorMapOPort* omap;
    Array1<ScalarFieldIPort*> fieldports;
    ColorMapIPort* imap;
public:
    RescaleColorMap(const clString& id);
    RescaleColorMap(const RescaleColorMap&, int deep);
    virtual ~RescaleColorMap();
    virtual Module* clone(int deep);
    virtual void execute();
    virtual void connection(ConnectionMode mode, int which_port, int);
};

extern "C" {
Module* make_RescaleColorMap(const clString& id)
{
    return scinew RescaleColorMap(id);
}
};

RescaleColorMap::RescaleColorMap(const clString& id)
: Module("RescaleColorMap", id, Filter)
{
    // Create the output port
    omap=scinew ColorMapOPort(this, "ColorMap", ColorMapIPort::Atomic);
    add_oport(omap);

    // Create the input ports
    imap=scinew ColorMapIPort(this, "ColorMap", ColorMapIPort::Atomic);
    add_iport(imap);
    ScalarFieldIPort* ifield=scinew ScalarFieldIPort(this, "ScalarField",
						     ScalarFieldIPort::Atomic);
    add_iport(ifield);
    fieldports.add(ifield);
}

RescaleColorMap::RescaleColorMap(const RescaleColorMap& copy, int deep)
: Module(copy, deep)
{
    NOT_FINISHED("RescaleColorMap::RescaleColorMap");
}

RescaleColorMap::~RescaleColorMap()
{
}

Module* RescaleColorMap::clone(int deep)
{
    return scinew RescaleColorMap(*this, deep);
}

void RescaleColorMap::execute()
{
    ColorMapHandle cmap;
    if(!imap->get(cmap))
	return;
    for(int i=0;i<fieldports.size()-1;i++){
        ScalarFieldHandle sfield;
        if(fieldports[i]->get(sfield)){
	    double min;
	    double max;
	    sfield->get_minmax(min, max);
	    cmap.detach();
	    cmap->min=min;
	    cmap->max=max;
	}
    }
    omap->send(cmap);
}

void RescaleColorMap::connection(ConnectionMode mode, int which_port, int)
{
    if(which_port > 0){
        if(mode==Disconnected){
	    remove_iport(which_port);
	    fieldports.remove(which_port-1);
	} else {
	    ScalarFieldIPort* p=scinew ScalarFieldIPort(this, "Field",
							ScalarFieldIPort::Atomic);
	    fieldports.add(p);
	    add_iport(p);
	}
    }
}
