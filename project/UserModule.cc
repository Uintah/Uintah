
/*
 *  UserModule.cc: Base class for defined modules
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

#include <UserModule.h>
#include <CallbackCloners.h>
#include <Connection.h>
#include <HelpUI.h>
#include <MUI.h>
#include <ModuleHelper.h>
#include <ModuleShape.h>
#include <MotifCallback.h>
#include <MtXEventLoop.h>
#include <NetworkEditor.h>
#include <NotFinished.h>
#include <Port.h>
#include <XFont.h>
#include <XQColor.h>
#include <Math/MinMax.h>
#include <Mt/DrawingArea.h>
#include <Mt/PushButton.h>
#include <PopupMenu.h>

#include <stdio.h>

extern MtXEventLoop* evl;

UserModule::UserModule(const clString& name, SchedClass sched_class)
: Module(name, sched_class), window(0), popup_on_create(0),
  drawing_a(0), need_reconfig(0), popup_menu(0), btn(0)
{
    ui_button_enabled=0;
}

UserModule::~UserModule()
{
    if(window)delete window;
}

UserModule::UserModule(const UserModule& copy, int deep)
: Module(copy, deep)
{
    NOT_FINISHED("UserModule::UserModule");
}

// User interface stuff...
void UserModule::add_ui(MUI_widget* widget)
{
    if(!window){
	window=new MUI_window(this);
	if(popup_on_create)
	    window->popup();
	enable_ui_button();
    }
    window->attach(widget);
}

void UserModule::enable_ui_button()
{
    ui_button_enabled=1;
    if(btn){
	btn->SetSensitive(True);
	btn->SetValues();
    }
}

void UserModule::remove_ui(MUI_widget* widget)
{
    ASSERT(window != 0);
    window->detach(widget);
}

void UserModule::reconfigure_ui()
{
    ASSERT(window != 0);
    window->reconfigure();
}


// Error conditions
void UserModule::error(const clString& string)
{
    netedit->add_text(name+": "+string);
}

void UserModule::create_interface()
{
    // Make the network icon..
    evl->lock();
    // Create the GC
    gc=XCreateGC(netedit->display, XtWindow(*netedit->drawing_a), 0, 0);

    int dir;
    int title_ascent;
    int title_descent;
    XCharStruct dim_title;
    if(!XTextExtents(netedit->name_font->font, name(), name.len(), &dir,
		     &title_ascent, &title_descent, &dim_title)){
	cerr << "XTextExtents failed...\n";
	exit(-1);
    }
    title_width=dim_title.width;
    int time_ascent;
    int time_descent;
    XCharStruct dim_time;
    static char* timestr="88:88";
    if(!XTextExtents(netedit->time_font->font, timestr, strlen(timestr), &dir,
		     &time_ascent, &time_descent, &dim_time)){
	cerr << "XTextExtents failed...\n";
	exit(-1);
    }
    int widget_ytop=MODULE_EDGE_WIDTH+MODULE_PORT_SIZE+MODULE_PORTLIGHT_HEIGHT
	+MODULE_PORT_SPACE;
    widget_ytitle=widget_ytop+title_ascent;
    widget_ygraphtop=widget_ytitle+title_descent;
    time_ascent=dim_time.ascent;
    time_descent=dim_time.descent;
    widget_ytime=widget_ygraphtop+MODULE_GRAPH_INSET+time_ascent;
    widget_ygraphbot=widget_ytime+time_descent;
    int bbot=widget_ytop+2*MODULE_BUTTON_SHADOW+MODULE_BUTTON_SIZE;
    int b=Min(widget_ygraphbot, bbot);
    height=widget_ygraphbot+MODULE_GRAPH_INSET+MODULE_PORT_SPACE
	+MODULE_PORTLIGHT_HEIGHT+MODULE_PORT_SIZE+MODULE_EDGE_WIDTH;

    int btn_left=MODULE_EDGE_WIDTH+MODULE_SIDE_BORDER;
    title_left=btn_left+2*MODULE_BUTTON_SHADOW+MODULE_BUTTON_SIZE
	+MODULE_TITLE_LEFT_SPACE;

    width=compute_width();
    widget_xgraphleft=title_left+dim_time.width+MODULE_GRAPH_TEXT_SPACE
	+MODULE_GRAPH_INSET;
    int gheight=widget_ygraphbot-widget_ygraphtop;
    widget_xgraphright=width-MODULE_EDGE_WIDTH-MODULE_SIDE_BORDER
	-MODULE_GRAPH_INSET;
    bgcolor=new XQColor(netedit->color_manager, MODULE_BGCOLOR);
    drawing_a=new DrawingAreaC;
    drawing_a->SetUnitType(XmPIXELS);
    drawing_a->SetX(xpos);
    drawing_a->SetY(ypos);
    drawing_a->SetWidth(width);
    drawing_a->SetHeight(height);
    drawing_a->SetMarginHeight(0);
    drawing_a->SetMarginWidth(0);
    drawing_a->SetShadowThickness(0);
    drawing_a->SetBackground(bgcolor->pixel());
    drawing_a->SetResizePolicy(XmRESIZE_NONE);
    drawing_a->SetTranslations(XtParseTranslationTable(""));
    // Add redraw callback...
    new MotifCallback<UserModule>FIXCB(drawing_a, XmNexposeCallback,
				       &netedit->mailbox, this,
				       &UserModule::redraw_widget, 0, 0);
    drawing_a->Create(*netedit->drawing_a, "usermodule");
    // Add button action callbacks.  These must be done after Create()
    new MotifCallback<UserModule>FIXCB(drawing_a,
				       "<Btn1Down>",
				       &netedit->mailbox, this,
				       &UserModule::move_widget, 0,
				       &CallbackCloners::event_clone);
    new MotifCallback<UserModule>FIXCB(drawing_a,
				       "<Btn1Up>",
				       &netedit->mailbox, this,
				       &UserModule::move_widget, 0,
				       &CallbackCloners::event_clone);
    new MotifCallback<UserModule>FIXCB(drawing_a,
				       "<Btn1Motion>",
				       &netedit->mailbox, this,
				       &UserModule::move_widget, 0,
				       &CallbackCloners::event_clone);
    new MotifCallback<NetworkEditor>FIXCB(drawing_a,
					  "<Btn2Down>",
					  &netedit->mailbox, netedit,
					  &NetworkEditor::connection_cb,
					  this,
					  &CallbackCloners::event_clone);
    new MotifCallback<UserModule>FIXCB(drawing_a,
				       "<Btn3Down>",
				       &netedit->mailbox, this,
				       &UserModule::post_menu, 0,
				       &CallbackCloners::event_clone);

    // Create the button...
    int bsize=MODULE_BUTTON_SIZE+2*MODULE_BUTTON_SHADOW;
    btn=new PushButtonC;
    btn->SetUnitType(XmPIXELS);
    btn->SetShadowThickness(MODULE_BUTTON_SHADOW);
    btn->SetX(btn_left);
    btn->SetY(widget_ytop);
    btn->SetWidth(bsize);
    btn->SetHeight(bsize);
    btn->SetBackground(bgcolor->pixel());
    btn->SetHighlightThickness(0);
    if(!ui_button_enabled)
	btn->SetSensitive(False);
    new MotifCallback<UserModule>FIXCB(btn, XmNactivateCallback,
				       &netedit->mailbox, this,
				       &UserModule::widget_button,
				       0, 0);
    btn->Create(*drawing_a, "UI");

    // Allocate colors...
    top_shadow=bgcolor->top_shadow();
    bottom_shadow=bgcolor->bottom_shadow();
    fgcolor=bgcolor->fg_color();
    select_color=bgcolor->select_color();
    executing_color=new XQColor(netedit->color_manager, MODULE_EXECCOLOR);
    executing_color_top=executing_color->top_shadow();
    executing_color_bot=executing_color->bottom_shadow();
    completed_color=new XQColor(netedit->color_manager, MODULE_COMPLETEDCOLOR);
    completed_color_top=completed_color->top_shadow();
    completed_color_bot=completed_color->bottom_shadow();

    // Create the window
    if(window)
	window->activate();
    evl->unlock();

    // Start up the event loop
    helper=new ModuleHelper(this, 0);
    helper->activate(0);
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

void UserModule::redraw_widget(CallbackData*, void*)
{
    if(need_reconfig){
	need_reconfig=0;
	reconfigure_iports();
	reconfigure_oports();
    }
    Display* dpy=netedit->display;
    Drawable win=XtWindow(*drawing_a);

    // Draw base
    evl->lock();
    draw_shadow(dpy, win, gc, 0, 0, width-1, height-1,
		MODULE_EDGE_WIDTH, top_shadow->pixel(), bottom_shadow->pixel());

    // Draw Input ports
    int port_spacing=MODULE_PORTPAD_WIDTH+MODULE_PORTPAD_SPACE;
    for(int p=0;p<iports.size();p++){
	IPort* iport=iports[p];
	iport->get_colors(netedit->color_manager);
	iport->update_light();
	XSetForeground(dpy, gc, iport->top_shadow->pixel());
	int left=p*port_spacing+MODULE_EDGE_WIDTH+MODULE_SIDE_BORDER;
	int right=left+MODULE_PORTPAD_WIDTH-1;
	XFillRectangle(dpy, win, gc, left, 0, right-left+1, MODULE_EDGE_WIDTH);
	XSetForeground(dpy, gc, iport->bgcolor->pixel());
	int t=MODULE_EDGE_WIDTH;
	XFillRectangle(dpy, win, gc, left, t, right-left+1, MODULE_PORT_SIZE);
	if(iport->nconnections() > 0){
	    // Draw tab...
	    int p2=(MODULE_PORTPAD_WIDTH-PIPE_WIDTH-2*PIPE_SHADOW_WIDTH)/2;
	    int l=left+p2+PIPE_SHADOW_WIDTH;
	    XSetForeground(dpy, gc, iport->bgcolor->pixel());
	    for(int i=0;i<PIPE_WIDTH;i++){
		XDrawLine(dpy, win, gc, l+i, 0, l+i, MODULE_EDGE_WIDTH-1);
	    }
	    XSetForeground(dpy, gc, iport->bottom_shadow->pixel());
	    l+=PIPE_WIDTH;
	    for(i=0;i<PIPE_SHADOW_WIDTH;i++){
		XDrawLine(dpy, win, gc, l+i, 0, l+i, MODULE_EDGE_WIDTH-i-1);
	    }
	}
    }

    // Draw Output ports
    for(p=0;p<oports.size();p++){
	OPort* oport=oports[p];
	oport->get_colors(netedit->color_manager);
	oport->update_light();
	int h=height;
	int left=p*port_spacing+MODULE_EDGE_WIDTH+MODULE_SIDE_BORDER;
	int right=left+MODULE_PORTPAD_WIDTH-1;
	XSetForeground(dpy, gc, oport->bottom_shadow->pixel());
	XFillRectangle(dpy, win, gc, left, h-MODULE_EDGE_WIDTH,
		       right-left+1, MODULE_EDGE_WIDTH);
	XSetForeground(dpy, gc, oport->bgcolor->pixel());
	int t=MODULE_EDGE_WIDTH+MODULE_PORT_SIZE;
	XFillRectangle(dpy, win, gc, left, h-t, right-left+1, MODULE_PORT_SIZE);
	if(oport->nconnections() > 0){
	    // Draw tab...
	    int p2=(MODULE_PORTPAD_WIDTH-PIPE_WIDTH-2*PIPE_SHADOW_WIDTH)/2;
	    int l=left+p2;
	    XSetForeground(dpy, gc, oport->top_shadow->pixel());
	    for(int i=0;i<PIPE_SHADOW_WIDTH;i++){
		XDrawLine(dpy, win, gc, l+i, h-i-1, l+i, h-1);
	    }
	    l+=PIPE_SHADOW_WIDTH;
	    XSetForeground(dpy, gc, oport->bgcolor->pixel());
	    for(i=0;i<PIPE_WIDTH;i++){
		XDrawLine(dpy, win, gc, l+i, h-MODULE_EDGE_WIDTH-1, l+i, h-1);
	    }
	}
    }

    // Draw title
    XSetFont(dpy, gc, netedit->name_font->font->fid);
    XSetForeground(dpy, gc, fgcolor->pixel());
    XDrawString(dpy, win, gc, title_left, widget_ytitle, name(), name.len());

    // Draw time and graph...
    update_module(0);

    evl->unlock();
}

void UserModule::update_module(int clear_first)
{
    evl->lock();
    Widget da=*drawing_a;
    Display* dpy=XtDisplay(da);
    Drawable win=XtWindow(da);
    int yginsettop=widget_ygraphtop-MODULE_GRAPH_INSET;
    int xginsetleft=widget_xgraphleft-MODULE_GRAPH_INSET;
    int xginsetright=widget_xgraphright+MODULE_GRAPH_INSET;
    int yginsetbot=widget_ygraphbot+MODULE_GRAPH_INSET;
    int ginsetheight=yginsetbot-yginsettop+1;
    int xleft=MODULE_EDGE_WIDTH+MODULE_SIDE_BORDER;
    if(clear_first){
	XSetForeground(dpy, gc, bgcolor->pixel());
	XFillRectangle(dpy, win, gc, xleft, yginsettop,
		       xginsetright-xleft+1, ginsetheight+1);
    }
    // Draw time/graph
    double time=timer.time();
    int secs=(int)time;
    int mins=(int)(secs/60);
    int hrs=(int)(secs/60);
    char timebuf[12];
    if(hrs > 0){
	sprintf(timebuf, "%d:%02d", hrs, mins);
    } else if(mins > 0){
	sprintf(timebuf, "%d:%02d", mins, secs);
    } else {
	int frac=(int)((time-secs)*100);
	sprintf(timebuf, "%d.%02d", secs, frac);
    }
    int timelen=strlen(timebuf);
    XSetFont(dpy, gc, netedit->time_font->font->fid);
    XSetForeground(dpy, gc, fgcolor->pixel());
    XDrawString(dpy, win, gc, title_left, widget_ytime, timebuf, timelen);

    // Draw indent for graph
    XSetLineAttributes(dpy, gc, 0, LineSolid, CapButt, JoinMiter);
    XSetForeground(dpy, gc, bottom_shadow->pixel());
    draw_shadow(dpy, win, gc, 
		xginsetleft, yginsettop, xginsetright, yginsetbot,
		MODULE_GRAPH_INSET, bottom_shadow->pixel(), top_shadow->pixel());

    // Draw Graph
    XSetForeground(dpy, gc, select_color->pixel());
    int total_gwidth=widget_xgraphright-widget_xgraphleft+1;
    int gheight=widget_ygraphbot-widget_ygraphtop+1;
    XFillRectangle(dpy, win, gc, widget_xgraphleft, widget_ygraphtop,
		   total_gwidth, gheight);
    double completed;
    Pixel gtop;
    Pixel gbot;
    switch(get_state()){
    case Module::NeedData:
	completed=0;
	gtop=gbot=0;
	break;
    case Module::Executing:
	completed=get_progress();
	completed=completed<0?0:completed>1?1:completed;
	XSetForeground(dpy, gc, executing_color->pixel());
	gtop=executing_color_top->pixel();
	gbot=executing_color_bot->pixel();
	break;
    case Module::Completed:
	completed=1;
	XSetForeground(dpy, gc, completed_color->pixel());
	gtop=completed_color_top->pixel();
	gbot=completed_color_bot->pixel();
	break;
    }
    int gwidth=(int)(completed*total_gwidth);
    old_gwidth=gwidth;
    if(gwidth==0){
	// Do nothing...
    } else if(gwidth <= 2*MODULE_GRAPH_SHADOW+1){
	XFillRectangle(dpy, win, gc, widget_xgraphleft, widget_ygraphtop,
		       gwidth+1, gheight);
    } else {
	XFillRectangle(dpy, win, gc,
		       widget_xgraphleft+MODULE_GRAPH_SHADOW,
		       widget_ygraphtop+MODULE_GRAPH_SHADOW,
		       gwidth-2*MODULE_GRAPH_SHADOW, gheight-2*MODULE_GRAPH_SHADOW);
	draw_shadow(dpy, win, gc, widget_xgraphleft, widget_ygraphtop,
		    widget_xgraphleft+gwidth-1, widget_ygraphbot,
		    MODULE_GRAPH_SHADOW,
		    gtop, gbot);
    }
    evl->unlock();
}

void UserModule::update_progress(double p)
{
    progress=p;
    int total_gwidth=widget_xgraphright-widget_xgraphleft+1;
    int gwidth=(int)(p*total_gwidth);
    if(gwidth != old_gwidth)update_module(1);
}

void UserModule::update_progress(int n, int max)
{
    update_progress(double(n)/double(max));
}

int UserModule::should_execute()
{
    if(sched_state == SchedNewData)
	return 0; // Already maxed out...
    int changed=0;
    if(sched_class != Sink){
	// See if any outputs are connected...
	int have_outputs;
	for(int i=0;i<oports.size();i++){
	    if(oports[i]->nconnections() > 0){
		have_outputs=1;
		break;
	    }
	}
	if(!have_outputs)cerr << "Not executing - not hooked up...\n";
	if(!have_outputs)return 0; // Don't bother checking stuff...
    }
    if(sched_state == SchedDormant){
	// See if we should be in the regen state
	for(int i=0;i<oports.size();i++){
	    OPort* port=oports[i];
	    for(int c=0;c<port->nconnections();c++){
		Module* mod=port->connection(c)->iport->get_module();
		if(mod->sched_state == SchedNewData
		   || mod->sched_state == SchedRegenData){
		    sched_state=SchedRegenData;
		    changed=1;
		    break;
		}
	    }
	}
    }

    // See if there is new data upstream...
    if(sched_class != Source){
	for(int i=0;i<iports.size();i++){
	    IPort* port=iports[i];
	    for(int c=0;c<port->nconnections();c++){
		Module* mod=port->connection(c)->oport->get_module();
		if(mod->sched_state == SchedNewData){
		    sched_state=SchedNewData;
		    changed=1;
		    break;
		}
	    }
	}
    }
    return changed;
}

void UserModule::do_execute()
{
    // Reset all of the ports...
    for(int i=0;i<oports.size();i++){
	OPort* port=oports[i];
	port->reset();
    }
    for(i=0;i<iports.size();i++){
	IPort* port=iports[i];
	port->reset();
    }
    // Call the User's execute function...
    state=Executing;
    update_progress(0.0);
    timer.clear();
    timer.start();
    execute();
    timer.stop();
    state=Completed;
    update_progress(1.0);

    // Call finish on all ports...
    for(i=0;i<iports.size();i++){
	IPort* port=iports[i];
	port->finish();
    }
    for(i=0;i<oports.size();i++){
	OPort* port=oports[i];
	port->finish();
    }
}

void UserModule::widget_button(CallbackData*, void*)
{
    // User interface...
    ui_button();
    if(window)
	window->popup();
    else
	popup_on_create=1;
}

void UserModule::ui_button()
{
}

void UserModule::mui_callback(void*, int)
{
    // Default - do nothing...
}

void UserModule::get_iport_coords(int which, int& x, int& y)
{
    int port_spacing=MODULE_PORTPAD_WIDTH+MODULE_PORTPAD_SPACE;
    int p2=(MODULE_PORTPAD_WIDTH-PIPE_WIDTH-2*PIPE_SHADOW_WIDTH)/2;
    x=xpos+which*port_spacing+MODULE_EDGE_WIDTH+MODULE_SIDE_BORDER+p2;
    y=ypos;
}

void UserModule::get_oport_coords(int which, int& x, int& y)
{
    get_iport_coords(which, x, y);
    y+=height;
}

void UserModule::reconfigure_iports()
{
    if(!drawing_a){
	need_reconfig=1;
	return;
    }
    need_reconfig=0;
    int port_spacing=MODULE_PORTPAD_WIDTH+MODULE_PORTPAD_SPACE;
    for(int p=0;p<iports.size();p++){
	int x=p*port_spacing+MODULE_EDGE_WIDTH+MODULE_SIDE_BORDER;
	int y=MODULE_EDGE_WIDTH+MODULE_PORT_SIZE;
	IPort* iport=iports[p];
	iport->set_context(x, y, drawing_a, gc);
    }
    if(compute_width() != width){
	width=compute_width();
	widget_xgraphright=width-MODULE_EDGE_WIDTH-MODULE_SIDE_BORDER
	    -MODULE_GRAPH_INSET;
	drawing_a->SetWidth(width);
	drawing_a->SetValues();
	evl->lock();
	drawing_a->SetWidth(width);
	drawing_a->SetValues();
	XClearWindow(netedit->display, XtWindow(*drawing_a));
	evl->unlock();
    }
    redraw_widget(0, 0);
}

void UserModule::reconfigure_oports()
{
    if(!drawing_a){
	need_reconfig=1;
	return;
    }
    need_reconfig=0;
    int port_spacing=MODULE_PORTPAD_WIDTH+MODULE_PORTPAD_SPACE;
    for(int p=0;p<oports.size();p++){
	int x=p*port_spacing+MODULE_EDGE_WIDTH+MODULE_SIDE_BORDER;
	int y=height-MODULE_EDGE_WIDTH-MODULE_PORT_SIZE-MODULE_PORTLIGHT_HEIGHT;
	OPort* oport=oports[p];
	oport->set_context(x, y, drawing_a, gc);
    }
    if(compute_width() != width){
	width=compute_width();
	widget_xgraphright=width-MODULE_EDGE_WIDTH-MODULE_SIDE_BORDER
	    -MODULE_GRAPH_INSET;
	evl->lock();
	drawing_a->SetWidth(width);
	drawing_a->SetValues();
	XClearWindow(netedit->display, XtWindow(*drawing_a));
	evl->unlock();
    }
    redraw_widget(0, 0);
}

void UserModule::move_widget(CallbackData* cbdata, void*)
{
    XEvent* event=cbdata->get_event();
    int i;
    evl->lock();
    switch(event->type){
    case ButtonPress:
	last_x=event->xbutton.x_root;
	last_y=event->xbutton.y_root;
	break;
    case ButtonRelease:
	break;
    case MotionNotify:
	xpos+=event->xmotion.x_root-last_x;
	ypos+=event->xmotion.y_root-last_y;
	drawing_a->SetX(xpos);
	drawing_a->SetY(ypos);
	drawing_a->SetValues();
	last_x=event->xmotion.x_root;
	last_y=event->xmotion.y_root;
	for(i=0;i<iports.size();i++)
	    iports[i]->move();
	for(i=0;i<oports.size();i++)
	    oports[i]->move();
	break;
    };
    evl->unlock();
}

void UserModule::post_menu(CallbackData* cbdata, void*)
{
    if(netedit->check_cancel())
	return;
    evl->lock();
    if(!popup_menu){
	popup_menu=new PopupMenuC;
	popup_menu->Create(*drawing_a, "popup");
	PushButtonC* pb=new PushButtonC;
	new MotifCallback<UserModule>FIXCB(pb, XmNactivateCallback,
					   &netedit->mailbox, this,
					   &UserModule::destroy, 0, 0);
	pb->Create(*popup_menu, "Destroy");
	pb=new PushButtonC;
	new MotifCallback<UserModule>FIXCB(pb, XmNactivateCallback,
					   &netedit->mailbox, this,
					   &UserModule::interrupt, 0, 0);
	pb->Create(*popup_menu, "Interrupt");
	pb=new PushButtonC;
	new MotifCallback<UserModule>FIXCB(pb, XmNactivateCallback,
					   &netedit->mailbox, this,
					   &UserModule::popup_help, 0, 0);
	pb->Create(*popup_menu, "Help...");
    }
    XmMenuPosition(*popup_menu, (XButtonPressedEvent*)cbdata->get_event());
    XtManageChild(*popup_menu);
    evl->unlock();
}

void UserModule::destroy(CallbackData*, void*)
{
    NOT_FINISHED("UserModule::destroy");
}

void UserModule::interrupt(CallbackData*, void*)
{
    NOT_FINISHED("UserModule::interrupt");
}

void UserModule::popup_help(CallbackData*, void*)
{
    HelpUI::load(name);
}

int UserModule::compute_width()
{
    int w=title_left+title_width+MODULE_SIDE_BORDER+MODULE_EDGE_WIDTH;
    int port_spacing=MODULE_PORTPAD_WIDTH+MODULE_PORTPAD_SPACE;
    int p2=(MODULE_PORTPAD_WIDTH-PIPE_WIDTH-2*PIPE_SHADOW_WIDTH)/2;
    int np=Max(iports.size(), oports.size());
    int x=np*port_spacing+2*(MODULE_EDGE_WIDTH+MODULE_SIDE_BORDER);
    return Max(x, w);
}

