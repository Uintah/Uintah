
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
#include <Connection.h>
#include <GeometryPort.h>
#include <MotifCallback.h>
#include <MtXEventLoop.h>
#include <NetworkEditor.h>
#include <NotFinished.h>
#include <XQColor.h>
#include <Mt/DialogShell.h>
#include <Mt/DrawingArea.h>
#include <Mt/Form.h>
#include <Mt/Frame.h>
#include <Mt/GLwMDraw.h>
#include <iostream.h>

extern MtXEventLoop* evl;

Salmon::Salmon()
: Module("Salmon", Sink)
{
    // Create the input port
    iports.add(new GeometryIPort(this, "Geometry", GeometryIPort::Atomic));
    add_iport(iports[0]);

    // Create the User Interface...
    dialog=new DialogShellC;
    dialog->Create("sci", "sci", evl->get_display());

#if 0
    form=new FormC;
    form->Create(*dialog, "viewer_form");
#endif

    gr_frame=new FrameC;
    gr_frame->SetShadowType(XmSHADOW_IN);
#if 0
    gr_frame->SetLeftAttachment(XmATTACH_FORM);
    gr_frame->SetRightAttachment(XmATTACH_POSITION);
    gr_frame->SetRightPosition(100);
    gr_frame->SetTopAttachment(XmATTACH_FORM);
#endif
    gr_frame->Create(*dialog, "frame");

    graphics=new GLwMDrawC;
    graphics->SetWidth(600);
    graphics->SetHeight(500);
#if 0
    graphics->SetNavigationType(XmSTICKY_TAB_GROUP);
    graphics->SetTraversalOn(True);
#endif
    graphics->Create(*gr_frame, "opengl_viewer");

}

Salmon::Salmon(const Salmon& copy, int deep)
: Module(copy, deep)
{
    NOT_FINISHED("Salmon::Salmon");
}

Salmon::~Salmon()
{
}

Module* make_Salmon()
{
    return new Salmon;
}

Module* Salmon::clone(int deep)
{
    return new Salmon(*this, deep);
}

void Salmon::do_execute()
{
    NOT_FINISHED("Salmon::do_execute");
}

void Salmon::create_widget()
{
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
}

void Salmon::redraw_widget(CallbackData*, void*)
{
    // This doesn't belong here!!!
    evl->lock();
    XtPopup(*dialog, XtGrabNone);
    evl->unlock();
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

