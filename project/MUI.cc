
/*
 *  MUI.cc: Module User Interface classes
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

#include <MUI.h>
#include <CallbackCloners.h>
#include <ModuleShape.h>
#include <MotifCallback.h>
#include <MtXEventLoop.h>
#include <NetworkEditor.h>
#include <NotFinished.h>
#include <UserModule.h>
#include <XQColor.h>
#include <Mt/DialogShell.h>
#include <Mt/DrawingArea.h>
#include <Mt/FileSelectionBox.h>
#include <Mt/RowColumn.h>
#include <Mt/Scale.h>
extern MtXEventLoop* evl;
#define MUI_FONT "screen14"
#define ONOFF_NSTEPS 10
#define ONOFF_INSET 2
#define ONOFF_BORDER 4
#define ONOFF_BEVEL 2

struct MUI_window_private {
    DialogShellC shell;
    RowColumnC rc;
};

MUI_window::MUI_window(UserModule* module)
: module(module), activated(0), popup_on_activate(0), priv(0)
{
}

MUI_window::~MUI_window()
{
    delete priv;
}

void MUI_window::attach(MUI_widget* widget)
{
    if(priv)
	widget->attach(this, &priv->rc);
    widgets.add(widget);
}

void MUI_window::detach(MUI_widget*)
{
    NOT_FINISHED("MUI_window::detach");
}

void MUI_window::activate()
{
    activated=1;
    if(popup_on_activate)
	popup();
}

void MUI_window::reconfigure()
{
    NOT_FINISHED("MUI_window::reconfigure");
}

void MUI_window::popup()
{
    if(!activated){
	popup_on_activate=1;
	return;
    }
    if(!priv){
	evl->lock(); // Optimization...
	priv=new MUI_window_private;
	priv->shell.SetAllowShellResize(True);
	priv->shell.Create("sci", "sci", evl->get_display());
	priv->rc.Create(priv->shell, "rc");
	for(int i=0;i<widgets.size();i++){
	    widgets[i]->attach(this, &priv->rc);
	}
	evl->unlock();
    } else {
	evl->lock();
	XtPopup(priv->shell, XtGrabNone);
	evl->unlock();
    }
}

void MUI_window::popdown()
{
    evl->lock();
    XtPopdown(priv->shell);
    evl->unlock();
}

UserModule* MUI_window::get_module()
{
    return module;
}

MUI_widget::MUI_widget(const clString& name, void* cbdata,
		       DispatchPolicy dispatch_policy)
: name(name), window(0), cbdata(cbdata), dispatch_policy(dispatch_policy)
{
}

MUI_widget::~MUI_widget()
{
}

void MUI_widget::set_title(const clString&)
{
    NOT_FINISHED("MUI_widget::set_title");
}

void MUI_widget::dispatch(const clString& newstr, clString* str, 
			  int info)
{
    if(dispatch_policy==Immediate){
	*str=newstr;
	window->get_module()->mui_callback(cbdata, info);
    } else {
	// Send it to the module...
	MUI_Module_Message* msg=new MUI_Module_Message(window->get_module(),
						       newstr, str,
						       cbdata, info);
	window->get_module()->mailbox.send(msg);
    }
}

void MUI_widget::dispatch(double newdata, double* data,
			  int info)
{
    if(dispatch_policy==Immediate){
	*data=newdata;
	window->get_module()->mui_callback(cbdata, info);
    } else {
	// Send it to the module...
	MUI_Module_Message* msg=new MUI_Module_Message(window->get_module(),
						       newdata, data,
						       cbdata, info);
	window->get_module()->mailbox.send(msg);
    }
}

void MUI_widget::dispatch(int newdata, int* data,
			  int info)
{
    if(dispatch_policy==Immediate){
	*data=newdata;
	window->get_module()->mui_callback(cbdata, info);
    } else {
	// Send it to the module...
	MUI_Module_Message* msg=new MUI_Module_Message(window->get_module(),
						       newdata, data,
						       cbdata, info);
	window->get_module()->mailbox.send(msg);
    }
}

MUI_slider_real::MUI_slider_real(const clString& name, double* data,
				 DispatchPolicy dp, int dispatch_drag,
				 Style style, Orientation orient,
				 void* cbdata)
: MUI_widget(name, cbdata, dp), data(data), dispatch_drag(dispatch_drag)
{
    scale=new ScaleC;
    switch(orient){
    case Horizontal:
	scale->SetOrientation(XmHORIZONTAL);
	break;
    case Vertical:
	scale->SetOrientation(XmVERTICAL);
	break;
    }
    scale->SetShowValue(True);
    XmString nstring=XmStringCreateSimple(name());
    scale->SetTitleString(nstring);
    scale->SetHighlightThickness(0);
    scale->SetDecimalPoints(2);
    int val=(int)(*data)*100;
    scale->SetValue(val);
    base=0.01;
}

MUI_slider_real::~MUI_slider_real()
{
}

void MUI_slider_real::set_minmax(double min, double max)
{
    int imin=(int)(min/base);
    int imax=(int)(max/base);
    scale->SetMinimum(imin);
    scale->SetMaximum(imax);
    if(window)
	scale->SetValues();
}

void MUI_slider_real::set_value(double val)
{
    int ival=(int)(val/base);
    scale->SetValue(ival);
    if(window)
	scale->SetValues();
}

void MUI_slider_real::attach(MUI_window* _window, EncapsulatorC* parent)
{
    window=_window;
    NetworkEditor* netedit=window->get_module()->netedit;
    new MotifCallback<MUI_slider_real>FIXCB(scale, XmNdragCallback,
					    &netedit->mailbox, this,
					    &MUI_slider_real::drag_callback,
					    cbdata,
					    &CallbackCloners::scale_clone);
    new MotifCallback<MUI_slider_real>FIXCB(scale, XmNvalueChangedCallback,
					    &netedit->mailbox, this,
					    &MUI_slider_real::value_callback,
					    cbdata,
					    &CallbackCloners::scale_clone);
    scale->Create(*parent, "scale");
}

void MUI_slider_real::drag_callback(CallbackData* cbdata, void*)
{
    double newdata=cbdata->get_int()*base;
    if(dispatch_drag)
	dispatch(newdata, data, Drag);
}

void MUI_slider_real::value_callback(CallbackData* cbdata, void*)
{
    double newdata=cbdata->get_int()*base;
    dispatch(newdata, data, Value);
}

MUI_file_selection::MUI_file_selection(const clString& name,
				       clString* filename,
				       DispatchPolicy dp, void* cbdata)
: MUI_widget(name, cbdata, dp), filename(filename)
{
}

MUI_file_selection::~MUI_file_selection()
{
}

void MUI_file_selection::attach(MUI_window* _window, EncapsulatorC* parent)
{
    window=_window;
    NetworkEditor* netedit=window->get_module()->netedit;
    sel=new FileSelectionBoxC;
    new MotifCallback<MUI_file_selection>FIXCB(sel, XmNokCallback,
					       &netedit->mailbox, this,
					       &MUI_file_selection::ok_callback,
					       cbdata,
					       &CallbackCloners::selection_clone);
    new MotifCallback<MUI_file_selection>FIXCB(sel, XmNcancelCallback,
					       &netedit->mailbox, this,
					       &MUI_file_selection::cancel_callback,
					       cbdata, 0);
    sel->Create(*parent, "file_selection");
}

void MUI_file_selection::ok_callback(CallbackData* cbdata, void*)
{
    clString new_filename=cbdata->get_string();
    dispatch(new_filename, filename, 0);
}

void MUI_file_selection::cancel_callback(CallbackData*, void*)
{
    window->popdown();
}

MUI_onoff_switch::MUI_onoff_switch(const clString& name, int* data,
				   DispatchPolicy dp, void* cbdata)
: MUI_widget(name, cbdata, dp), data(data)
{
}

MUI_onoff_switch::~MUI_onoff_switch()
{
}

void MUI_onoff_switch::attach(MUI_window* _window, EncapsulatorC* parent)
{
    evl->lock();
    Display* dpy=evl->get_display();
    XFontStruct* font;
    
    if( (font = XLoadQueryFont(dpy, MUI_FONT)) == 0){
	cerr << "Error loading font: " << MUI_FONT << endl;
	exit(-1);
    }
    int dir;
    int ascent;
    XCharStruct dim;
    if(!XTextExtents(font, name(), name.len(), &dir,
		     &ascent, &descent, &dim)){
	cerr << "XTextExtents failed...\n";
	exit(-1);
    }
    fh=ascent+descent;
    height=3*fh+2*ONOFF_INSET+3*ONOFF_BORDER;
    width=dim.width+2*ONOFF_BORDER;
    window=_window;
    NetworkEditor* netedit=window->get_module()->netedit;
    sw=new DrawingAreaC;
    sw->SetWidth(width);
    sw->SetHeight(height);
    sw->SetResizePolicy(XmRESIZE_NONE);
    new MotifCallback<MUI_onoff_switch>FIXCB(sw, XmNexposeCallback,
					     &netedit->mailbox, this,
					     &MUI_onoff_switch::expose_callback,
					     0, 0);
    sw->Create(*parent, "onoff");
    new MotifCallback<MUI_onoff_switch>FIXCB(sw,
					     "<Btn1Down>",
					     &netedit->mailbox, this,
					     &MUI_onoff_switch::event_callback, 0,
					     &CallbackCloners::event_clone);
    new MotifCallback<MUI_onoff_switch>FIXCB(sw,
					     "<Btn1Up>",
					     &netedit->mailbox, this,
					     &MUI_onoff_switch::event_callback, 0,
					     &CallbackCloners::event_clone);
    if(*data)
	anim=ONOFF_NSTEPS;
    else
	anim=0;
    bgcolor=new XQColor(netedit->color_manager, MODULE_BGCOLOR);
    top_shadow=bgcolor->top_shadow();
    bot_shadow=bgcolor->bottom_shadow();
    text_color=bgcolor->fg_color();
    inset_color=bgcolor->select_color();
    GC gc=XCreateGC(dpy, XtWindow(*sw), 0, 0);
    XSetFont(dpy, gc, font->fid);
    vgc=(void*)gc;
    evl->unlock();
}

void MUI_onoff_switch::event_callback(CallbackData*, void*)
{
    int newdata;
    if(anim){
	// It's on... turn it off...
	evl->lock();
	Display* dpy=XtDisplay(*sw);
	for(anim=ONOFF_NSTEPS;anim>=0;anim--){
	    XClearWindow(dpy, XtWindow(*sw));
	    expose_callback(0, 0);
	    XFlush(dpy);
	    Task::sleep(0.2);
	}
	evl->unlock();
	anim=0;
	newdata=0;
    } else {
	// It's off... turn it on...
	evl->lock();
	Display* dpy=XtDisplay(*sw);
	for(anim=0;anim<=ONOFF_NSTEPS;anim++){
	    XClearWindow(dpy, XtWindow(*sw));
	    expose_callback(0, 0);
	    XFlush(dpy);
	    Task::sleep(0.2);
	}
	evl->unlock();
	newdata=1;
	anim=ONOFF_NSTEPS;
    }
    dispatch(newdata, data, Value);
}

static void draw_shadow(Display* dpy, Window win, GC gc,
			int xmin, int ymin, int xmax, int ymax,
			int width, Pixel top, Pixel bot)
{
    XSetForeground(dpy, gc, top);
    for(int i=0;i<width;i++){
	XDrawLine(dpy, win, gc, xmin, ymin+i, xmax-i, ymin+i);
	XDrawLine(dpy, win, gc, xmin+i, ymin, xmin+i, ymax-i);
    }
    XSetForeground(dpy, gc, bot);
    for(i=0;i<width;i++){
	XDrawLine(dpy, win, gc, xmax-i, ymin+i+1, xmax-i, ymax);
	XDrawLine(dpy, win, gc, xmin+i+1, ymax-i, xmax, ymax-i);
    }
}

void MUI_onoff_switch::expose_callback(CallbackData*, void*)
{
    evl->lock();
    int left=ONOFF_BORDER;
    Display* dpy=XtDisplay(*sw);
    GC gc=(GC)vgc;
    Window win=XtWindow(*sw);
    XSetForeground(dpy, gc, text_color->pixel());
    XDrawString(dpy, win, gc, ONOFF_BORDER, height-ONOFF_BORDER-descent,
		name(), name.len());
    draw_shadow(dpy, win, gc, left, ONOFF_BORDER,
		left+fh+2*ONOFF_INSET, ONOFF_BORDER+2*fh+2*ONOFF_INSET,
		ONOFF_INSET, bot_shadow->pixel(), top_shadow->pixel());
    XSetForeground(dpy, gc, inset_color->pixel());
    XFillRectangle(dpy, win, gc, left+ONOFF_INSET, ONOFF_BORDER+ONOFF_INSET,
		   fh+1, 2*fh+1);
    int top=ONOFF_BORDER+ONOFF_INSET+((ONOFF_NSTEPS-anim)*fh)/ONOFF_NSTEPS;
    draw_shadow(dpy, win, gc, left+ONOFF_INSET, top,
		left+ONOFF_INSET+fh, top+fh, ONOFF_BEVEL,
		top_shadow->pixel(), bot_shadow->pixel());
    XSetForeground(dpy, gc, bgcolor->pixel());
    XFillRectangle(dpy, win, gc, left+ONOFF_INSET+ONOFF_BEVEL,
		   top+ONOFF_BEVEL, fh-2*ONOFF_BEVEL+1, fh-2*ONOFF_BEVEL+1);
    XSetForeground(dpy, gc, text_color->pixel());
    if(anim==0){
	XDrawString(dpy, win, gc, left+2*ONOFF_INSET+fh+3,
		    height-ONOFF_BORDER-fh-descent-ONOFF_INSET, "Off", 3);
    } else if(anim==ONOFF_NSTEPS){
	XDrawString(dpy, win, gc, left+2*ONOFF_INSET+fh+3,
		    height-ONOFF_BORDER-2*fh-descent-ONOFF_INSET, "On", 2);
    }
    evl->unlock();
}

MUI_Module_Message::MUI_Module_Message(UserModule* module,
				       const clString& newstr,
				       clString* str, void* cbdata,
				       int flags)
: MessageBase(MessageTypes::MUIDispatch), type(StringData),
  module(module), newstr(newstr), str(str), cbdata(cbdata), flags(flags)
{
}

MUI_Module_Message::MUI_Module_Message(UserModule* module, double newddata,
				       double* ddata, void* cbdata, int flags)
: MessageBase(MessageTypes::MUIDispatch), type(DoubleData),
  module(module), newddata(newddata), ddata(ddata), cbdata(cbdata),
  flags(flags)
{
}

MUI_Module_Message::MUI_Module_Message(UserModule* module, int newidata,
				       int* idata, void* cbdata, int flags)
: MessageBase(MessageTypes::MUIDispatch), type(IntData),
  module(module), newidata(newidata), idata(idata), cbdata(cbdata),
  flags(flags)
{
}

MUI_Module_Message::~MUI_Module_Message()
{
}

void MUI_Module_Message::do_it()
{
    switch(type){
    case StringData:
	*str=newstr;
	break;
    case DoubleData:
	*ddata=newddata;
	break;
    case IntData:
	*idata=newidata;
	break;
    }
}

MUI_point::MUI_point(const clString& name, Point* data,
		     DispatchPolicy dp, int dispatch_drag,
		     void* cbdata)
: MUI_widget(name, cbdata, dp), data(data), dispatch_drag(dispatch_drag)
{
    NOT_FINISHED("MUI_point::MUI_point");
}

MUI_point::~MUI_point()
{
}

void MUI_point::attach(MUI_window* _window, EncapsulatorC* parent)
{
    NOT_FINISHED("MUI_point::attach");
}

