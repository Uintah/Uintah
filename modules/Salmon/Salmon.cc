/*
 *  Salmon.cc:  The Geometry Viewer
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

// Someday, we should delete these four lines, when the
// compiler stops griping about const cast away...
#include <X11/Intrinsic.h>
#include "myStringDefs.h"
#include "myXmStrDefs.h"
#include "myShell.h"

#include <Salmon/Salmon.h>
#include <Salmon/Roe.h>
#include <Connection.h>
#include <GeometryPort.h>
#include <ModuleList.h>
#include <MotifCallback.h>
#include <MtXEventLoop.h>
#include <NetworkEditor.h>
#include <NotFinished.h>
#include <XQColor.h>
#include <Mt/DrawingArea.h>
#include <iostream.h>

class GeomObj;

extern MtXEventLoop* evl;

static Module* make_Salmon()
{
    return new Salmon;
}

static RegisterModule db1("Geometry", "Salmon", make_Salmon);

Salmon::Salmon()
: Module("Salmon", Sink)
{
    // Create the input port
    iports.add(new GeometryIPort(this, "Geometry", GeometryIPort::Atomic));
    add_iport(iports[0]);
}

Salmon::Salmon(const Salmon& copy, int deep)
: Module(copy, deep)
{
    NOT_FINISHED("Salmon::Salmon");
}

Salmon::~Salmon()
{
}

Module* Salmon::clone(int deep)
{
    return new Salmon(*this, deep);
}

void Salmon::do_execute()
{
    NOT_FINISHED("Salmon::do_execute");
}

void Salmon::create_interface()
{
    // Create the module icon
    evl->lock(); // Lock just once - for efficiency
    bgcolor=new XQColor(netedit->color_manager, "salmon");
    drawing_a=new DrawingAreaC;
    drawing_a->SetUnitType(XmPIXELS);
    drawing_a->SetX(xpos);
    drawing_a->SetY(ypos);
    drawing_a->SetWidth(100);
    drawing_a->SetHeight(100);
    drawing_a->SetMarginHeight(0);
    drawing_a->SetMarginWidth(0);
    drawing_a->SetShadowThickness(0);
    drawing_a->SetBackground(bgcolor->pixel());
    drawing_a->SetResizePolicy(XmRESIZE_NONE);
    // Add redraw callback...
    new MotifCallback<Salmon>FIXCB(drawing_a, XmNexposeCallback,
				   &netedit->mailbox, this,
				   &Salmon::redraw_widget, 0, 0);

    drawing_a->Create(*netedit->drawing_a, "usermodule");


    // Create the viewer window...
    topRoe.add(new Roe(this));
    evl->unlock();
}

void Salmon::redraw_widget(CallbackData*, void*)
{
}

int Salmon::should_execute()
{
    // See if there is new data upstream...
    int changed=0;
    for(int i=0;i<iports.size();i++){
	IPort* port=iports[i];
	for(int c=0;c<port->nconnections();c++){
	    Module* mod=port->connection(c)->iport->get_module();
	    if(mod->sched_state == SchedNewData){
		sched_state=SchedNewData;
		changed=1;
		break;
	    }
	}
    }
    return changed;
}

void Salmon::reconfigure_iports()
{
    NOT_FINISHED("Salmon::reconfigure_iports");
}

void Salmon::reconfigure_oports()
{
    NOT_FINISHED("Salmon::reconfigure_oports");
}

void Salmon::addObj(int serial, GeomObj *obj)
{
    NOT_FINISHED("Salmon::addObj");
}

void Salmon::delObj(int serial)
{
    NOT_FINISHED("Salmon::delObj");
}

void Salmon::addTopRoe(Roe *r)
{
    NOT_FINISHED("Salmon::addTopRoe");
}

void Salmon::makeTopRoe()
{
    NOT_FINISHED("Salmon::makeTopRoe");
}

void Salmon::delTopRoe(Roe *r)
{
    NOT_FINISHED("Salmon::delTopRoe");
}
