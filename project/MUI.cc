
/*
 *  MUI.h: Module User Interface classes
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
#include <MotifCallback.h>
#include <MtXEventLoop.h>
#include <NetworkEditor.h>
#include <NotFinished.h>
#include <UserModule.h>
#include <Mt/DialogShell.h>
#include <Mt/FileSelectionBox.h>
#include <Mt/RowColumn.h>
#include <Mt/Scale.h>
extern MtXEventLoop* evl;

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
    NOT_FINISHED("MUI");
}

void MUI_window::activate()
{
    activated=1;
    if(popup_on_activate)
	popup();
}

void MUI_window::reconfigure()
{
    NOT_FINISHED("MUI");
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
    NOT_FINISHED("MUI");
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

MUI_onoff_switch::MUI_onoff_switch(const clString& name, int*,
				   DispatchPolicy dp, void* cbdata)
: MUI_widget(name, cbdata, dp)
{
    NOT_FINISHED("MUI");
}

MUI_onoff_switch::~MUI_onoff_switch()
{
}

void MUI_onoff_switch::attach(MUI_window*, EncapsulatorC*)
{
    NOT_FINISHED("onoff switch");
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
    }
}
