
/*
 *  Connection.cc: A Connection between two modules
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

// Someday, we should delete these four lines, when the
// compiler stops griping about const cast away...
#include <X11/Intrinsic.h>
#include "myStringDefs.h"
#include "myXmStrDefs.h"
#include "myShell.h"

#include <Connection.h>
#include <Module.h>
#include <ModuleShape.h>
#include <MotifCallback.h>
#include <MtXEventLoop.h>
#include <NetworkEditor.h>
#include <NotFinished.h>
#include <Port.h>
#include <XQColor.h>
#include <Math/MinMax.h>
#include <Math/MiscMath.h>
#include <Mt/DrawingArea.h>

extern MtXEventLoop* evl;

Connection::Connection(Module* m1, int p1, Module* m2, int p2)
{
    oport=m1->oport(p1);
    iport=m2->iport(p2);
    local=1;
    connected=0;
    netedit=0;
}

void Connection::connect()
{
    oport->attach(this);
    iport->attach(this);
    connected=1;
}

Connection::~Connection()
{
    if(netedit){
	evl->lock();
	for(int i=0;i<5;i++){
	    delete drawing_a[i];
	}
	XFreeGC(netedit->display, gc);
	evl->unlock();
    }
    if(connected){
	NOT_FINISHED("detach...");
    }
}


void Connection::calc_portwindow_size(int which, int& x, int& y,
				      int& w, int& h)
{
    Module* imod=iport->get_module();
    Module* omod=oport->get_module();
    int iwhich=iport->get_which_port();
    int owhich=oport->get_which_port();
    int ix, iy;
    imod->get_iport_coords(iwhich, ix, iy);
    int ox, oy;
    omod->get_oport_coords(owhich, ox, oy);
    int cx=(ix+ox)/2;
    int cy=(iy+oy)/2;
    int width=PIPE_WIDTH+2*PIPE_SHADOW_WIDTH;
    int ly=oy+MIN_WIRE_EXTEND;
    int uy=iy-MIN_WIRE_EXTEND;
    if(ox < ix){
        if(cx >= imod->xpos-MODULE_PORT_SPACE){
            cx=imod->xpos-MODULE_PORT_SPACE;
        }
        if(cx <= omod->xpos+omod->width+MODULE_PORT_SPACE){
            cx=omod->xpos+omod->width+MODULE_PORT_SPACE;
        }
    } else {
        if(cx >= omod->xpos-MODULE_PORT_SPACE){
            cx=omod->xpos-MODULE_PORT_SPACE;
        }
        if(cx <= imod->xpos+imod->width+MODULE_PORT_SPACE){
            cx=imod->xpos+imod->width+MODULE_PORT_SPACE;
        }
    }
    int x1, x2;
    int y1, y2;
    if(ly > uy){
        switch(which){
	case 0:
            // Vertical segment from output port...
            x1=ox;
            y1=ly;
            x2=ox+width;
            y2=oy;
            break;
        case 1:
            // Horizontal segment to center...
            x1=ox;
            y1=ly;
            x2=cx+width;
            y2=ly+width;
            if(x1 > x2-width){
                x1+=width;
                x2-=width;
            }
            break;
        case 2:
            // Vertical segment...
            x1=cx;
            y1=ly;
            x2=cx+width;
            y2=uy;
            break;
        case 3:
            // Horizontal segment...
            x1=cx;
            y1=uy-width;
            x2=ix+width;
            y2=uy;
            if(x1 > x2-width){
                x1+=width;
                x2-=width;
            }
            break;
        case 4:
            // Vertical segment to input port...
            x1=ix;
            y1=iy;
            x2=ix+width;
            y2=uy;
            break;
        }
    } else {
        switch(which){
        case 0:
            // Vertical segment from output port...
            x1=ox;
            y1=oy;
            x2=ox+width;
            y2=cy;
            break;
        case 1:
            // Horizontal segment...
            x1=ox;
            y1=cy;
            x2=ix+width;
            y2=cy+width;
            if(x1 > x2-width){
                x1+=width;
                x2-=width;
            }
            break;
        case 2:
        case 3:
            // Empty...
            x1=x2=y1=y2=0;
            break;
        case 4:
            // Vertical segment from input port...
            x1=ix;
            y1=iy;
            x2=ix+width;
            y2=cy+width;
            break;
        }
    }
    x=Min(x1, x2);
    y=Min(y1, y2);
    w=Abs(x2-x1);
    h=Abs(y2-y1);
    if(w==0)w=1;
    if(h==0)h=1;
}

void Connection::set_context(NetworkEditor* netedit_)
{
    netedit=netedit_;
    evl->lock();
    iport->get_colors(netedit->color_manager);
    for(int i=0;i<5;i++){
	int x,y,w,h;
	calc_portwindow_size(i, x, y, w, h);
	DrawingAreaC* da=drawing_a[i]=new DrawingAreaC;
	da->SetUnitType(XmPIXELS);
	da->SetX(x);
	da->SetY(y);
	da->SetWidth(w);
	da->SetHeight(h);
	da->SetResizePolicy(XmNONE);
	da->SetMarginHeight(0);
	da->SetMarginWidth(0);
	da->SetShadowThickness(0);
	da->SetBackground(iport->bgcolor->pixel());
	new MotifCallback<Connection>FIXCB(da, XmNexposeCallback,
					   &netedit->mailbox, this,
					   &Connection::redraw, (void*)i, 0);
	da->Create(*netedit->drawing_a, "connection");
    }
    gc=XCreateGC(netedit->display, XtWindow(*netedit->drawing_a), 0, 0);
    evl->unlock();
}

void Connection::redraw(CallbackData*, void* ud)
{
    evl->lock();
    int which_seg=(int)ud;
    Dimension w, h;
    Display* dpy=XtDisplay(*drawing_a[which_seg]);
    Drawable win=XtWindow(*drawing_a[which_seg]);
    XClearWindow(dpy, win);
    drawing_a[which_seg]->GetWidth(&w);
    drawing_a[which_seg]->GetHeight(&h);
    drawing_a[which_seg]->GetValues();
    Pixel bottom_shadow=iport->bottom_shadow->pixel();
    Pixel top_shadow=iport->top_shadow->pixel();
    if(which_seg==0 || which_seg==2 || which_seg==4){
        // Vertical segment...
        XSetForeground(dpy, gc, top_shadow);
        for(int i=0;i<PIPE_SHADOW_WIDTH;i++){
            XDrawLine(dpy, win, gc, i, 0, i, h);
        }
        XSetForeground(dpy, gc, bottom_shadow);
        for(i=0;i<PIPE_SHADOW_WIDTH;i++){
            XDrawLine(dpy, win, gc, w-i-1, 0, w-i-1, h);
        }
    } else {
        Module* imod=iport->get_module();
        Module* omod=oport->get_module();
        int iwhich=iport->get_which_port();
        int owhich=oport->get_which_port();
        int ix, iy;
        imod->get_iport_coords(iwhich, ix, iy);
        int ox, oy;
        omod->get_oport_coords(owhich, ox, oy);
        int cx=(ix+ox)/2;
        int cy=(iy+oy)/2;
        int width=PIPE_WIDTH+2*PIPE_SHADOW_WIDTH;
        int ly=oy+MIN_WIRE_EXTEND;
        int uy=iy-MIN_WIRE_EXTEND;
        if(ly > uy){
            if(which_seg == 1){
                XSetForeground(dpy, gc, top_shadow);
                for(int i=0;i<PIPE_SHADOW_WIDTH;i++){
                    XDrawLine(dpy, win, gc, h-i-1, i, w-h+i, i);
                    XDrawLine(dpy, win, gc, w-h+i, 0, w-h+1, i);
                    XDrawLine(dpy, win, gc, i, 0, i, h-i-1);
                }
                XSetForeground(dpy, gc, bottom_shadow);
                for(i=0;i<PIPE_SHADOW_WIDTH;i++){
                    XDrawLine(dpy, win, gc, i+1, h-i-1, w-i-1, h-i-1);
                    XDrawLine(dpy, win, gc, w-i-1, 0, w-i-1, h-i-1);
                    XDrawLine(dpy, win, gc, h-i-1, 0, h-i-1, i);
                }
            } else {
                // seg 3
                XSetForeground(dpy, gc, top_shadow);
                for(int i=0;i<PIPE_SHADOW_WIDTH;i++){
                    XDrawLine(dpy, win, gc, i+1, i, w-i-1, i);
                    XDrawLine(dpy, win, gc, i, i, i, h-1);
                    XDrawLine(dpy, win, gc, w-h+i, h-i, w-h+i, h-1);
                }
                XSetForeground(dpy, gc, bottom_shadow);
                for(i=0;i<PIPE_SHADOW_WIDTH;i++){
                    XDrawLine(dpy, win, gc, w-i-1, i+1, w-i-1, h-1);
                    XDrawLine(dpy, win, gc, h-i-1, h-i-1, h-i-1, h-1);
                    XDrawLine(dpy, win, gc, h-i-1, h-i-1, w-h+i, h-i-1);
                }
            }
        } else {
            // Just one seg (1)
            if(ix > ox){
                XSetForeground(dpy, gc, top_shadow);
                for(int i=0;i<PIPE_SHADOW_WIDTH;i++){
                    XDrawLine(dpy, win, gc, i, 0, i, h-i-1);
                    XDrawLine(dpy, win, gc, w-h+i, h-i-1, w-h+i, h-1);
                    XDrawLine(dpy, win, gc, h-i-1, i, w-i-1, i);
                }
                XSetForeground(dpy, gc, bottom_shadow);
                for(i=0;i<PIPE_SHADOW_WIDTH;i++){
                    XDrawLine(dpy, win, gc, i+1, h-i-1, w-h+i, h-i-1);
                    XDrawLine(dpy, win, gc, w-i-1, i+1, w-i-1, h-1);
                    XDrawLine(dpy, win, gc, h-i-1, 0, h-i-1, i);
                }
            } else {
                XSetForeground(dpy, gc, top_shadow);
                for(int i=0;i<PIPE_SHADOW_WIDTH;i++){
                    XDrawLine(dpy, win, gc, i, i, i, h-1);
                    XDrawLine(dpy, win, gc, i, i, w-h+i, i);
                    XDrawLine(dpy, win, gc, w-h+i, 0, w-h+i, i);
                }
                XSetForeground(dpy, gc, bottom_shadow);
                for(i=0;i<PIPE_SHADOW_WIDTH;i++){
                    XDrawLine(dpy, win, gc, h-i-1, h-i-1, h-i-1, h-1);
                    XDrawLine(dpy, win, gc, h-i-1, h-i-1, w-i, h-i-1);
                    XDrawLine(dpy, win, gc, w-i-1, 0, w-i-1, h-i);
                }
            }
        }
    }
    evl->unlock();
}

void Connection::move()
{
    for(int i=0;i<5;i++){
	int x,y,w,h;
	calc_portwindow_size(i, x, y, w, h);
	DrawingAreaC* da=drawing_a[i];
	da->SetX(x);
	da->SetY(y);
	da->SetWidth(w);
	da->SetHeight(h);
	da->SetValues();
	redraw(0, (void*)i);
    }
}

