/*
 *  Salmon.cc:  The Geometry Viewer Window
 *
 *  Written by:
 *   David Weinstein
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

#include <Salmon/Roe.h>
#include <MotifCallback.h>
#include <MtXEventLoop.h>
#include <NetworkEditor.h>
#include <NotFinished.h>
#include <Mt/DialogShell.h>
#include <Mt/DrawingArea.h>
#include <Mt/Form.h>
#include <Mt/Frame.h>
#include <Mt/GLwMDraw.h>
#include <iostream.h>
#include <Geometry/Vector.h>

extern MtXEventLoop* evl;

Roe::Roe(Salmon* s) 
{
    evl->lock();
    manager=s;
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
    evl->unlock();
}


Roe::Roe(const Roe& copy)
{
    NOT_FINISHED("Roe::Roe");
}

Roe::~Roe()
{
}

void Roe::SetParent(Roe *r)
{
    NOT_FINISHED("Roe::SetParent(Roe *)");
}

void Roe::SetParent(Salmon *s)
{
    NOT_FINISHED("Roe::SetParent(Salmon *)");
}

void Roe::rotate(double angle, Vector v)
{
    NOT_FINISHED("Roe::rotate");
}

void Roe::translate(Vector v)
{
    NOT_FINISHED("Roe:translate");
}

void Roe::scale(Vector v)
{
    NOT_FINISHED("Roe::scale");
}

void Roe::redraw()
{
    NOT_FINISHED("Roe::redraw");
}

void Roe::addChild(Roe *r)
{
    NOT_FINISHED("Roe::addChild");
}

void Roe::deleteChild(Roe *r)
{
    NOT_FINISHED("Roe::deleteChild");
}

