
/*
 *  Dialbox.cc: Dialbox manager thread...
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   June 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

// Someday, we should delete these four lines, when the
// compiler stops griping about const cast away...
#include <X11/Intrinsic.h>
#include "myStringDefs.h"
#include "myXmStrDefs.h"
#include "myShell.h"

#include <Dialbox.h>
#include <CallbackCloners.h>
#include <MotifCallback.h>
#include <MtXEventLoop.h>
#include <NotFinished.h>
#include <XQColor.h>
#include <Math/Trig.h>
#include <Mt/DialogShell.h>
#include <Mt/DrawingArea.h>
#include <iostream.h>
#include <stdio.h>
#include <X11/Xlib.h>
#include <X11/extensions/XInput.h>

#define DIAL_WIDTH 80
#define DIAL_HEIGHT 110
#define TITLE_HEIGHT 20
#define DIAL_CENTER 10
#define DIAL_BGCOLOR "steve1"
#define DIAL_BORDER 5
#define DIAL_BEVEL 2
#define DIAL_FONT "-*-lucida-medium-r-*-*-12-*-*-*-*-*-*-*"


static Dialbox* the_dialbox;
int have_dials;
extern MtXEventLoop* evl;

int ndials;
int* dial_min;
int* dial_max;
int* dial_value;
int* dial_res;
int* dial_first;
int absolute_mode;

#define DIALBOX_DEVICE_NAME "dial+buttons"

Dialbox::Dialbox(ColorManager* color_manager)
: Task("Dialbox", 1), context(0), color_manager(color_manager),
  mailbox(100)
{
}

Dialbox::~Dialbox()
{
}

int Dialbox::body(int)
{
    if(the_dialbox){
	cerr << "There should be only one Dialbox thread!\n";
	return 0;
    }
    the_dialbox=this;
    while(1){
	MessageBase* msg=mailbox.receive();
	switch(msg->type){
	case MessageTypes::DoCallback:
	    {
		Callback_Message* cmsg=(Callback_Message*)msg;
		cmsg->mcb->perform(cmsg->cbdata);
		delete cmsg->cbdata;
	    }
	    break;
	case MessageTypes::AttachDialbox:
	    {
		DialMsg* dmsg=(DialMsg*)msg;
		context=dmsg->context;
		popup_ui();
		for(int i=0;i<8;i++)
		    redraw_dial(0, (void*)i);
		redraw_title(0, 0);
	    }
	    break;
	case MessageTypes::DialMoved:
	    {
		DialMsg* dmsg=(DialMsg*)msg;
		int which=dmsg->which;
		double delta=double(dmsg->info)/double(dial_res[which]);
		if(context && context->knobs[which].callback){
		    DBContext::Knob* knob=&context->knobs[which];
		    double oldvalue=knob->value;
		    switch(knob->rangetype){
		    case DBContext::Bounded:
			knob->value+=(knob->max-knob->min)*delta;
			if(knob->value > knob->max)
			    knob->value=knob->max;
			else if(knob->value < knob->min)
			    knob->value=knob->min;
			break;
		    case DBContext::Unbounded:
			knob->value+=knob->scale*delta;
			break;
		    case DBContext::Wrapped:
			knob->value+=(knob->max-knob->min)*delta;
			while(knob->value > knob->max)
			    knob->value-=(knob->max-knob->min);
			while(knob->value < knob->min)
			    knob->value+=(knob->max-knob->min);
			break;
		    }
		    if(oldvalue != knob->value){
			redraw_dial(0, (void*)which);
			context->knobs[which].callback->dispatch(context,
								 which,
								 knob->value,
								 knob->value-oldvalue);
		    }
		}
	    }
	    break;
	default:
	    cerr << "Bad message sent to dialbox...\n";
	    break;
	}
	delete msg;
    }
    return 0;
}

void Dialbox::handle_event(void* vevent)
{
    // Send the event to the Dialbox body, which
    // will update the interface and pass it on...
    XEvent* event=(XEvent*)vevent;
    XDeviceMotionEvent* me=(XDeviceMotionEvent*)&event->xany;
    if(absolute_mode){
	int axis=me->first_axis;
	for(int i=0;i<me->axes_count;i++){
	    if(dial_first[axis]){
		dial_value[axis]=me->axis_data[i];
		dial_first[axis]=0;
	    } else {
		int d=me->axis_data[i]-dial_value[axis];
		if(d)
		    the_dialbox->mailbox.send(new DialMsg(axis, d));
		dial_value[axis]=me->axis_data[i];
	    }
	    axis++;
	}
    } else {
	int axis=me->first_axis;
	for(int i=0;i<me->axes_count;i++){
	    the_dialbox->mailbox.send(new DialMsg(axis, me->axis_data[i]));
	    axis++;
	}
    }
}

int Dialbox::get_event_type()
{
    // Create the device...
    // Called by event loop, so we don't have to lock...
    have_dials=0;
    int ndev;
    Display* display=evl->get_display();
    XDeviceInfo* devinfo=XListInputDevices(display, &ndev);
    for(int i=0;i<ndev;i++){
	if(!strcmp(devinfo[i].name, DIALBOX_DEVICE_NAME)){
	    // Found it...
	    break;
	}
    }
    if(i==ndev){
	cerr << "No dialbox found!\n";
	return -1;
    }
    XDevice* dev=XOpenDevice(display, devinfo[i].id);
    if(!dev){
	cerr << "Cannot open dials...\n";
	return -1;
    }
    XEventClass events[1];
    int type;
    DeviceMotionNotify(dev, type, events[0]);
    XSelectExtensionEvent(display, DefaultRootWindow(display),
			  &events[0], 1);
    char* p=(char*)(devinfo[i].inputclassinfo);
    XValuatorInfo* vi;
    for(int j=0;j<devinfo[i].num_classes;j++){
	vi=(XValuatorInfo*)p;
	if(vi->c_class == ValuatorClass)
	    break;
	p+=vi->length;
    }
    if(j==devinfo[i].num_classes){
	cerr << "I am very confused...\n";
	return -1;
    }
    absolute_mode=(vi->mode == Absolute);
    if(absolute_mode)
	cerr << "Dialbox in absolute mode...\n";
    else
	cerr << "Dialbox in Relative mode...\n";
    ndials=vi->num_axes;
    dial_min=new int[ndials];
    dial_max=new int[ndials];
    dial_value=new int[ndials];
    dial_res=new int[ndials];
    for(i=0;i<ndials;i++){
	dial_min[i]=dial_max[i]=dial_value[i]=0;
	dial_res[i]=vi->axes[i].resolution;
    }
    if(absolute_mode){
	dial_first=new int[ndials];
	for(int i=0;i<ndials;i++){
	    dial_min[i]=vi->axes[i].min_value;
	    dial_max[i]=vi->axes[i].max_value;
	    dial_first[i]=1;
	}
    }
    return type;
}

void Dialbox::attach_dials(DBContext* context)
{
    if(!the_dialbox){
	cerr << "The dialbox hasn't been created yet!!!" << endl;
	return;
    }
    the_dialbox->mailbox.send(new DialMsg(context));
}

DialMsg::DialMsg(DBContext* context)
: MessageBase(MessageTypes::AttachDialbox), context(context)
{
}

DialMsg::DialMsg(int which, int info)
: MessageBase(MessageTypes::DialMoved), info(info), which(which)
{
}

DialMsg::~DialMsg()
{
}

void Dialbox::popup_ui()
{
    if(!window){
	evl->lock();
	window=new DialogShellC;
	window->SetWidth(2*DIAL_WIDTH);
	window->SetHeight(4*DIAL_HEIGHT+TITLE_HEIGHT);
	window->SetAllowShellResize(False);
	window->Create("sci", "sci", evl->get_display());
	main_da=new DrawingAreaC;
	main_da->SetMarginWidth(0);
	main_da->SetMarginHeight(0);
	main_da->SetWidth(2*DIAL_WIDTH);
	main_da->SetHeight(4*DIAL_HEIGHT+TITLE_HEIGHT);
	main_da->Create(*window, "drawing_a");
	title_da=new DrawingAreaC;
	title_da->SetWidth(2*DIAL_WIDTH);
	title_da->SetHeight(TITLE_HEIGHT);
	title_da->SetX(0);
	title_da->SetY(0);
	new MotifCallback<Dialbox>FIXCB(title_da, XmNexposeCallback,
					&mailbox, this,
					&Dialbox::redraw_title,
					0, 0);
	title_da->Create(*main_da, "drawing_a");
	for(int i=0;i<8;i++){
	    int r=3-i/2;
	    int c=i%2;
	    dial_da[i]=new DrawingAreaC;
	    dial_da[i]->SetWidth(DIAL_WIDTH);
	    dial_da[i]->SetHeight(DIAL_HEIGHT);
	    dial_da[i]->SetX(c*DIAL_WIDTH);
	    dial_da[i]->SetY(r*DIAL_HEIGHT+TITLE_HEIGHT);
	    new MotifCallback<Dialbox>FIXCB(dial_da[i], XmNexposeCallback,
					    &mailbox, this,
					    &Dialbox::redraw_dial,
					    (void*)i, 0);
	    dial_da[i]->Create(*main_da, "drawing_a");
	}
	dpy=evl->get_display();
	gc=XCreateGC(dpy, XtWindow(*title_da), 0, 0);
	font=XLoadQueryFont(dpy, DIAL_FONT);
	if(!font){
	    cerr << "Can't load font: " << DIAL_FONT << endl;
	}
	XSetFont(dpy, gc, font->fid);
	bgcolor=new XQColor(color_manager, DIAL_BGCOLOR);
	top_shadow=bgcolor->top_shadow();
	bottom_shadow=bgcolor->bottom_shadow();
	fgcolor=bgcolor->fg_color();
	inset_color=bgcolor->select_color();
	evl->unlock();
    } else {
	evl->lock();
	XtPopup(*window, XtGrabNone);
	evl->unlock();
    }
}

void Dialbox::redraw_title(CallbackData*, void*)
{
    evl->lock();
    XClearWindow(dpy, XtWindow(*title_da));
    XSetForeground(dpy, gc, fgcolor->pixel());
    int x=DIAL_BORDER;
    int y=DIAL_BORDER+10;
    XDrawString(dpy, XtWindow(*title_da), gc, x, y,
		context->name(), context->name.len());
    evl->unlock();
}

static void draw_inset_circle(Display* dpy, GC gc, Window win,
			      int x, int y, int w, int h,
			      int bw,
			      XQColor* top,
			      XQColor* bot,
			      XQColor* inset)
{
    XGCValues gcv;
    gcv.line_width=bw;
    XChangeGC(dpy, gc, GCLineWidth, &gcv);
    int bw2=bw/2;
    XSetForeground(dpy, gc, top->pixel());
    XDrawArc(dpy, win, gc, x+bw2, y+bw2, w-bw, h-bw,
	     225*64, -180*64);
    XSetForeground(dpy, gc, bot->pixel());
    XDrawArc(dpy, win, gc, x+bw2, y+bw2, w-bw, h-bw,
	     45*64, -180*64);
    XSetForeground(dpy, gc, inset->pixel());
    XFillArc(dpy, win, gc, x+bw, y+bw, w-2*bw, h-2*bw,
	     0, 360*64);
}

void Dialbox::redraw_dial(CallbackData*, void* vwhich)
{
    evl->lock();
    int which=(int)vwhich;
    XClearWindow(dpy, XtWindow(*dial_da[which]));
    if(!context || !context->knobs[which].callback){
	evl->unlock();
	return;
    }
    draw_inset_circle(dpy, gc, XtWindow(*dial_da[which]),
		      DIAL_BORDER, DIAL_BORDER,
		      DIAL_WIDTH-2*DIAL_BORDER, DIAL_WIDTH-2*DIAL_BORDER,
		      DIAL_BEVEL,
		      bottom_shadow, top_shadow, inset_color);
    int c=DIAL_WIDTH/2-DIAL_CENTER;
    draw_inset_circle(dpy, gc, XtWindow(*dial_da[which]),
		      c, c,
		      2*DIAL_CENTER, 2*DIAL_CENTER,
		      DIAL_BEVEL,
		      top_shadow, bottom_shadow, bgcolor);
    double ad;
    DBContext::Knob* knob=&context->knobs[which];
    switch(knob->rangetype){
    case DBContext::Bounded:
	ad=45+270*(knob->value-knob->min)/(knob->max-knob->min);
	break;
    case DBContext::Unbounded:
	ad=360*(knob->value/knob->scale);
	break;
    case DBContext::Wrapped:
	ad=360*(knob->value-knob->min)/(knob->max-knob->min);
	break;
    }
    int r2=DIAL_WIDTH/2-DIAL_BORDER-DIAL_BEVEL;
    int r1=DIAL_CENTER;
    double add=DtoR(ad);
    int x1=(int)(-Sin(add)*r1)+DIAL_WIDTH/2;
    int y1=(int)(Cos(add)*r1)+DIAL_WIDTH/2;
    int x2=(int)(-Sin(add)*r2)+DIAL_WIDTH/2;
    int y2=(int)(Cos(add)*r2)+DIAL_WIDTH/2;
    XSetForeground(dpy, gc, fgcolor->pixel());
    XDrawLine(dpy, XtWindow(*dial_da[which]), gc, x1, y1, x2, y2);

    if(knob->rangetype == DBContext::Bounded){
	// Draw tickmarks
	int r2=DIAL_WIDTH/2-DIAL_BORDER-DIAL_BEVEL;
	r1=(int)(0.85*r2);
	ad=45;
	double add=DtoR(ad);
	int x1=(int)(-Sin(add)*r1)+DIAL_WIDTH/2;
	int y1=(int)(Cos(add)*r1)+DIAL_WIDTH/2;
	int x2=(int)(-Sin(add)*r2)+DIAL_WIDTH/2;
	int y2=(int)(Cos(add)*r2)+DIAL_WIDTH/2;
	XDrawLine(dpy, XtWindow(*dial_da[which]), gc, x1, y1, x2, y2);
	ad=315;
	add=DtoR(ad);
	x1=(int)(-Sin(add)*r1)+DIAL_WIDTH/2;
	y1=(int)(Cos(add)*r1)+DIAL_WIDTH/2;
	x2=(int)(-Sin(add)*r2)+DIAL_WIDTH/2;
	y2=(int)(Cos(add)*r2)+DIAL_WIDTH/2;
	XDrawLine(dpy, XtWindow(*dial_da[which]), gc, x1, y1, x2, y2);	

	// Draw Min/Max numbers...
    }
    // Draw value
    char buf[100];
    sprintf(buf, "%f", knob->value);
    int ascent;
    int descent;
    int dir;
    XCharStruct dim;
    if(!XTextExtents(font, buf, strlen(buf), &dir, &ascent, &descent, &dim)){
	cerr << "XTextExtents failed???\n";
    }
    int x=DIAL_WIDTH/2-dim.width/2;
    int y=DIAL_WIDTH+ascent;
    XDrawString(dpy, XtWindow(*dial_da[which]), gc, x, y, buf, strlen(buf));

    // Draw label
    if(!XTextExtents(font, knob->name(), knob->name.len(),
		     &dir, &ascent, &descent, &dim)){
	cerr << "XTextExtents failed???\n";
    }
    y+=ascent+descent;
    x=DIAL_WIDTH/2-dim.width/2;
    XDrawString(dpy, XtWindow(*dial_da[which]), gc, x, y,
		knob->name(), knob->name.len());
    evl->unlock();
}
