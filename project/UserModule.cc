
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
#include <MotifCallback.h>
#include <MtXEventLoop.h>
#include <NetworkEditor.h>
#include <NotFinished.h>
#include <Port.h>
#include <XQColor.h>
#include <Mt/DrawingArea.h>
#include <Mt/PushButton.h>

#include <stdio.h>

extern MtXEventLoop* evl;

#define MODULE_BGCOLOR "pink4"
#define MODULE_EXECCOLOR "red"
#define MODULE_COMPLETEDCOLOR "green"
#define MODULE_EDGE_WIDTH 3
#define MODULE_PORT_SIZE 3
#define MODULE_PORT_SPACE 3
#define MODULE_BUTTON_EDGE 2
#define MODULE_BUTTON_SIZE 28
#define MODULE_BUTTON_BORDER 1
#define MODULE_TITLE_TOP_SPACE 2
#define MODULE_TITLE_BOT_SPACE 2
#define MODULE_GRAPH_INSET 1
#define MODULE_GRAPH_SHADOW 2
#define MODULE_NBUTTONS 5
#define MODULE_GRAPH_TEXT_SPACE 3
#define MODULE_GRAPH_BUTT_SPACE 3
#define MODULE_SIDE_BORDER 5
#define MODULE_PORTPAD_WIDTH 13
#define MODULE_PORTPAD_SPACE 3
#define PIPE_WIDTH 3
#define PIPE_SHADOW_WIDTH 2

UserModule::UserModule(const clString& name, SchedClass sched_class)
: Module(name, sched_class), window(0), mapped(0), popup_on_create(0)
{
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
    }
    window->attach(widget);
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
    cerr << string << endl;
}


void UserModule::create_interface()
{
    // Make the network icon..
    evl->lock();
    // Create the GC
    gc=XCreateGC(netedit->display, XtWindow(*netedit->drawing_a), 0, 0);

    int dir;
    int ascent;
    int descent;
    XCharStruct dim_title;
    if(!XTextExtents(netedit->name_font, name(), name.len(), &dir, &ascent, &descent,
		     &dim_title)){
	cerr << "XTextExtents failed...\n";
	exit(-1);
    }
    XCharStruct dim_time;
    static char* timestr="88:88";
    if(!XTextExtents(netedit->time_font, timestr, strlen(timestr), &dir, &ascent, &descent,
		     &dim_time)){
	cerr << "XTextExtents failed...\n";
	exit(-1);
    }
    widget_ytitle=MODULE_EDGE_WIDTH+MODULE_PORT_SIZE+MODULE_PORT_SPACE
	+MODULE_BUTTON_BORDER+MODULE_BUTTON_EDGE+MODULE_BUTTON_SIZE
	    + MODULE_BUTTON_EDGE+MODULE_BUTTON_BORDER+MODULE_TITLE_TOP_SPACE
		+dim_title.ascent;
    widget_ygraphtop=widget_ytitle+dim_title.descent
	+MODULE_TITLE_BOT_SPACE+MODULE_GRAPH_INSET;
    widget_ytime=widget_ygraphtop+dim_time.ascent;
    widget_ygraphbot=widget_ytime+dim_time.descent;
    widget_height=widget_ygraphbot+MODULE_GRAPH_INSET+MODULE_PORT_SPACE
	+MODULE_PORT_SIZE+MODULE_EDGE_WIDTH;
    int twidth=MODULE_EDGE_WIDTH+MODULE_SIDE_BORDER+ dim_title.width
	+MODULE_SIDE_BORDER+MODULE_EDGE_WIDTH;
    int bwidth=MODULE_EDGE_WIDTH+MODULE_SIDE_BORDER
	+ (MODULE_NBUTTONS+1)*MODULE_BUTTON_BORDER
	+ MODULE_NBUTTONS*(MODULE_BUTTON_SIZE+2*MODULE_BUTTON_EDGE)
	+ MODULE_SIDE_BORDER+MODULE_EDGE_WIDTH;
    widget_width=bwidth;
    if(twidth > widget_width)widget_width=twidth;
    widget_xgraphleft=MODULE_EDGE_WIDTH+MODULE_SIDE_BORDER+dim_time.width
	+MODULE_GRAPH_TEXT_SPACE+MODULE_GRAPH_INSET;
    int gheight=widget_ygraphbot-widget_ygraphtop;
    widget_xgraphright=widget_width-MODULE_EDGE_WIDTH-MODULE_SIDE_BORDER
	-gheight-MODULE_GRAPH_BUTT_SPACE-MODULE_GRAPH_INSET;
    bgcolor=new XQColor(netedit->color_manager, MODULE_BGCOLOR);
    drawing_a=new DrawingAreaC;
    drawing_a->SetUnitType(XmPIXELS);
    drawing_a->SetX(xpos);
    drawing_a->SetY(ypos);
    drawing_a->SetWidth(widget_width);
    drawing_a->SetHeight(widget_height);
    drawing_a->SetMarginHeight(0);
    drawing_a->SetMarginWidth(0);
    drawing_a->SetShadowThickness(0);
    drawing_a->SetBackground(bgcolor->pixel());
    drawing_a->SetResizePolicy(XmRESIZE_NONE);
    // Add redraw callback...
    new MotifCallback<UserModule>FIXCB(drawing_a, XmNexposeCallback,
				       &netedit->mailbox, this,
				       &UserModule::redraw_widget, 0, 0);

    new MotifCallback<UserModule>FIXCB(drawing_a, XmNinputCallback,
				       &netedit->mailbox, this,
				       &UserModule::input_widget, 0, 0);
    drawing_a->Create(*netedit->drawing_a, "usermodule");

    // Create the buttons...
    btn=new PushButtonC*[MODULE_NBUTTONS];
    int xbleft=MODULE_EDGE_WIDTH+MODULE_SIDE_BORDER+MODULE_BUTTON_BORDER;
    int xbsize=MODULE_BUTTON_BORDER+MODULE_BUTTON_SIZE+2*MODULE_BUTTON_EDGE;
    int xbsize2=xbsize-MODULE_BUTTON_BORDER;
    int ybtop=MODULE_EDGE_WIDTH+MODULE_PORT_SIZE+MODULE_PORT_SPACE+MODULE_BUTTON_BORDER;
    for(int i=0;i<MODULE_NBUTTONS;i++){
	btn[i]=new PushButtonC;
	btn[i]->SetShadowThickness(MODULE_BUTTON_EDGE);
	btn[i]->SetUnitType(XmPIXELS);
	int x=xbleft+i*xbsize;
	btn[i]->SetX(x);
	btn[i]->SetY(ybtop);
	btn[i]->SetWidth(xbsize2);
	btn[i]->SetHeight(xbsize2);
	btn[i]->SetBackground(bgcolor->pixel());
	btn[i]->SetHighlightThickness(0);
	new MotifCallback<UserModule>FIXCB(btn[i], XmNactivateCallback,
					   &netedit->mailbox, this,
					   &UserModule::widget_button,
					   (void*)i, 0);
	btn[i]->Create(*drawing_a, "button");
    }

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
    Display* dpy=netedit->display;
    Drawable win=XtWindow(*drawing_a);

    // Draw base
    evl->lock();
    draw_shadow(dpy, win, gc, 0, 0, widget_width-1, widget_height-1,
		MODULE_EDGE_WIDTH, top_shadow->pixel(), bottom_shadow->pixel());

    // Draw Input ports
    int port_spacing=MODULE_PORTPAD_WIDTH+MODULE_PORTPAD_SPACE;
    for(int p=0;p<iports.size();p++){
	IPort* iport=iports[p];
	iport->get_colors(netedit->color_manager);
	XSetForeground(dpy, gc, iport->top_shadow->pixel());
	int left=p*port_spacing+MODULE_EDGE_WIDTH+MODULE_SIDE_BORDER;
	int right=left+MODULE_PORTPAD_WIDTH-1;
	for(int i=0;i<MODULE_EDGE_WIDTH;i++){
	    XDrawLine(dpy, win, gc, left, i, right, i);
	}
	XSetForeground(dpy, gc, iport->bgcolor->pixel());
	int t=MODULE_EDGE_WIDTH;
	for(i=0;i<MODULE_PORT_SIZE;i++){
	    XDrawLine(dpy, win, gc, left, i+t, right, i+t);
	}
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
	int h=widget_height;
	int left=p*port_spacing+MODULE_EDGE_WIDTH+MODULE_SIDE_BORDER;
	int right=left+MODULE_PORTPAD_WIDTH-1;
	XSetForeground(dpy, gc, oport->bottom_shadow->pixel());
	for(int i=0;i<MODULE_EDGE_WIDTH;i++){
	    XDrawLine(dpy, win, gc, left, h-i-1, right, h-i-1);
	}
	XSetForeground(dpy, gc, oport->bgcolor->pixel());
	int t=MODULE_EDGE_WIDTH+1;
	for(i=0;i<MODULE_PORT_SIZE;i++){
	    XDrawLine(dpy, win, gc, left, h-i-t, right, h-i-t);
	}
	if(oport->nconnections() > 0){
	    // Draw tab...
	    int p2=(MODULE_PORTPAD_WIDTH-PIPE_WIDTH-2*PIPE_SHADOW_WIDTH)/2;
	    int l=left+p2;
	    XSetForeground(dpy, gc, oport->top_shadow->pixel());
	    for(i=0;i<PIPE_SHADOW_WIDTH;i++){
		XDrawLine(dpy, win, gc, l+i, h-i-1, l+i, h-1);
	    }
	    l+=PIPE_SHADOW_WIDTH;
	    XSetForeground(dpy, gc, oport->bgcolor->pixel());
	    for(int i=0;i<PIPE_WIDTH;i++){
		XDrawLine(dpy, win, gc, l+i, h-MODULE_EDGE_WIDTH-1, l+i, h-1);
	    }
	}
    }

    // Draw buttons
    int ybtop=MODULE_EDGE_WIDTH+MODULE_PORT_SIZE+MODULE_PORT_SPACE;
    int ybbot=ybtop+MODULE_BUTTON_BORDER+MODULE_BUTTON_EDGE+MODULE_BUTTON_SIZE
	+MODULE_BUTTON_EDGE;
    int xbleft=MODULE_EDGE_WIDTH+MODULE_SIDE_BORDER;
    int xbsize=MODULE_BUTTON_BORDER+MODULE_BUTTON_SIZE
	+2*MODULE_BUTTON_EDGE;
    int xbright=xbleft+MODULE_BUTTON_BORDER+MODULE_NBUTTONS*xbsize;
    // Draw border..
    XSetForeground(dpy, gc, fgcolor->pixel());
    XDrawLine(dpy, win, gc, xbleft, ybtop, xbright-1, ybtop);
    XDrawLine(dpy, win, gc, xbleft, ybbot, xbright-1, ybbot);
    int x=xbleft;
    ybtop+=MODULE_BUTTON_BORDER;
    ybbot-=MODULE_BUTTON_BORDER;
    xbleft+=MODULE_BUTTON_BORDER;
    for(int i=0;i<MODULE_NBUTTONS+1;i++){
	XDrawLine(dpy, win, gc, x, ybtop, x, ybbot);
	x+=xbsize;
    }

    // Draw title
    XSetFont(dpy, gc, netedit->name_font->fid);
    XSetForeground(dpy, gc, fgcolor->pixel());
    int xleft=MODULE_EDGE_WIDTH+MODULE_SIDE_BORDER;
    XDrawString(dpy, win, gc, xleft, widget_ytitle, name(), name.len());

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
    XSetFont(dpy, gc, netedit->time_font->fid);
    XSetForeground(dpy, gc, fgcolor->pixel());
    XDrawString(dpy, win, gc, xleft, widget_ytime, timebuf, timelen);

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
	if(!have_outputs)return 0; // Don't bother checking stuff...
    }
    if(sched_state == SchedDormant){
	// See if we should be in the regen state
	for(int i=0;i<oports.size();i++){
	    OPort* port=oports[i];
	    for(int c=0;c<port->nconnections();c++){
		Module* mod=port->connection(c)->oport->get_module();
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
		Module* mod=port->connection(c)->iport->get_module();
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
    // Reset all of the output ports...
    for(int i=0;i<oports.size();i++){
	OPort* port=oports[i];
	port->reset();
    }
    for(i=0;i<iports.size();i++){
	IPort* port=iports[i];
	port->reset();
    }
    // Call the User's execute function...
    execute();

    // Call finish on all output ports...
    for(i=0;i<oports.size();i++){
	OPort* port=oports[i];
	port->finish();
    }
}

void UserModule::input_widget(CallbackData*, void*)
{
    NOT_FINISHED("UserModule::input_widget");
}

void UserModule::widget_button(CallbackData*, void* data)
{
    int which=(int)data;
    switch(which){
    case 0:
	// User interface...
	if(window)
	    window->popup();
	else
	    popup_on_create=1;
	break;
    case 3:
	// Help...
	HelpUI::load(name);
	break;
    default:
	cerr << "Button " << which << " pushed...\n";
	break;
    }
}
