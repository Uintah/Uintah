
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
#include <GeometryPort.h>
#include <MotifCallback.h>
#include <NetworkEditor.h>
#include <NotFinished.h>
#include <XQColor.h>
#include <Mt/DrawingArea.h>
#include <iostream.h>

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
    NOT_FINISHED("Salmon::execute");
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
}

int Salmon::should_execute()
{
    NOT_FINISHED("Salmon::should_execute");
    return 0;
}

