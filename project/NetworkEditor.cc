
/*
 *  NetworkEditor.cc: The network editor...
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <NetworkEditor.h>

#include <Connection.h>
#include <Datatype.h>
#include <Module.h>
#include <Network.h>
#include <NotFinished.h>
#include <Port.h>

#include <Classlib/ArgProcessor.h>
#include <Math/MinMax.h>
#include <Math/MiscMath.h>

#include <Xm/Xm.h>
#include <Xm/DrawingA.h>
#include <Xm/MainW.h>
#include <Xm/ScrollBar.h>

#include <iostream.h>
#include <stdlib.h>
#include <stdio.h>

// The look of things...
#define SCI_BACKGROUND_COLOR "#224488"
#define EXECUTING_COLOR "red"
#define COMPLETED_COLOR "green"
#define MOD_EDGE_WIDTH 3
#define MOD_PORT_SIZE 3
#define MOD_PORT_SPACE 3
#define MOD_BUTTON_EDGE 2
#define MOD_BUTTON_SIZE 28
#define MOD_BUTTON_BORDER 1
#define MOD_TITLE_TOP_SPACE 2
#define MOD_TITLE_BOT_SPACE 2
#define MOD_GRAPH_INSET 1
#define MOD_GRAPH_SHADOW 2
#define MOD_NBUTTONS 5
#define MOD_GRAPH_TEXT_SPACE 3
#define MOD_GRAPH_BUTT_SPACE 3
#define MOD_SIDE_BORDER 5
#define MOD_PORTPAD_WIDTH 13
#define MOD_PORTPAD_SPACE 3
#define PIPE_WIDTH 3
#define PIPE_SHADOW_WIDTH 2
#define CANVAS_SIZE 2000
#define STAY_FROM_EDGE 2
#define MIN_WIRE_EXTEND 8
// #define MOD_NAME_FONT "-*-lucida-medium-r-*-*-17-*-*-*-*-*-*-*"
#define MOD_NAME_FONT "-*-lucida-bold-r-*-*-14-*-*-*-*-*-*-*"
#define MOD_TIME_FONT "-*-lucida-medium-r-*-*-11-*-*-*-*-*-*-*"
#define DEFAULT_INTERVAL_TIME 100L

// These are too messy to put in the .h file...
static Pixel background_color;
static Widget toplevel;
static Widget main_w;
static XtAppContext app;
static String fallback_resources[] = {
    "*fontList: screen14",
    "*background: pink3",
    NULL
    };
static XFontStruct* name_font;
static XFontStruct* time_font;
static GC gc;
static int have_gc=0;
static Pixel bg_color;
static Pixel fg_color;
static Pixel top_shadow;
static Pixel bottom_shadow;
static Pixel select_color;
static Pixel executing_color;
static Pixel executing_color_top;
static Pixel executing_color_bot;
static Pixel completed_color;
static Pixel completed_color_top;
static Pixel completed_color_bot;
static long interval_time=DEFAULT_INTERVAL_TIME;
static XtIntervalId timer_id;
static int redrawn_once=0;
static String module_translations =
	"<Btn1Down>:   module_move(down) ManagerGadgetArm()\n"
	"<Btn1Up>:     module_move(up) ManagerGadgetActivate()\n"
	"<Btn1Motion>: module_move(motion) ManagerGadgetButtonMotion()";
static String network_translations =
	"<Btn1Down>:   network_scroll(down) ManagerGadgetArm()\n"
	"<Btn1Up>:     network_scroll(up) ManagerGadgetActivate()\n"
	"<Btn1Motion>: network_scroll(motion) ManagerGadgetButtonMotion()";
static String connection_translations =
	"<Btn1Down>:   connection_move(down) ManagerGadgetArm()\n"
	"<Btn1Up>:     connection_move(up) ManagerGadgetActivate()\n"
	"<Btn1Motion>: connection_move(motion) ManagerGadgetButtonMotion()";

#define MOVE_NONE 0
#define MOVE_WIDGET 1
static int dragmode=MOVE_NONE;
static int drag_sx;
static int drag_sy;
static int drag_slop_x;
static int drag_slop_y;

class ModuleWidgetCallbackData {
public:
    NetworkEditor* ne;
    Module* mod;
};

class ConnectionWidgetCallbackData {
public:
    NetworkEditor* ne;
    Connection* conn;
    int which_seg;
};

// Callbacks for motif...
void do_timer(XtPointer ud, XtIntervalId*)
{
    NetworkEditor* ne=(NetworkEditor*)ud;
    ne->timer();
}

void do_redraw(Widget, XtPointer ud, XtPointer xcbdata)
{
    NetworkEditor* ne=(NetworkEditor*)ud;
    ne->redraw(xcbdata);
}

void do_mod_redraw(Widget, XtPointer ud, XtPointer)
{
    ModuleWidgetCallbackData* cb=(ModuleWidgetCallbackData*)ud;
    cb->ne->draw_module(cb->mod);
}

void do_con_redraw(Widget, XtPointer ud, XtPointer)
{
    ConnectionWidgetCallbackData* cb=(ConnectionWidgetCallbackData*)ud;
    cb->ne->draw_connection(cb->conn, cb->which_seg);
}

void do_module_move(Widget w, XButtonEvent* event, String* args, int* num_args)
{
    if(*num_args != 1)
	XtError("Wrong number of args!");

    XtPointer ud;
    XtVaGetValues(w, XmNuserData, &ud, NULL);
    ModuleWidgetCallbackData* cb=(ModuleWidgetCallbackData*)ud;
    cb->ne->module_move(cb->mod, event, args[0]);
}

void do_connection_move(Widget w, XButtonEvent* event, String* args, int* num_args)
{
    if(*num_args != 1)
	XtError("Wrong number of args!");

    XtPointer ud;
    XtVaGetValues(w, XmNuserData, &ud, NULL);
    ModuleWidgetCallbackData* cb=(ModuleWidgetCallbackData*)ud;
    cb->ne->connection_move(cb->mod, event, args[0]);
}

void do_network_scroll(Widget, XButtonEvent* event, String* args,
		       int* num_args)
{
    if(*num_args != 1)
	XtError("Wrong number of args!");
    static int nsx, nsy;
    static int slopx=0, slopy=0;
    if(strcmp(args[0], "down") == 0){
	nsx=event->x_root;
	nsy=event->y_root;
	slopx=slopy=0;
    } else {
	int dx=event->x_root-nsx;
	int dy=event->y_root-nsy;
	if(slopx < 0){
	    slopx+=Abs(dx);
	    if(slopx > 0){
		dx=Sign(dx)*slopx;
		slopx=0;
	    }
	}
	if(slopy < 0){
	    slopy+=Abs(dy);
	    if(slopy > 0){
		dy=Sign(dy)*slopy;
		slopy=0;
	    }
	}
	// Move the sliders...
	Widget hbar;
	XtVaGetValues(main_w, XmNhorizontalScrollBar, &hbar, NULL);
	Widget vbar;
	XtVaGetValues(main_w, XmNverticalScrollBar, &vbar, NULL);
	int x,y;
	int xsize, ysize;
	int xinc, yinc;
	int xpinc, ypinc;
	XmScrollBarGetValues(hbar, &x, &xsize, &xinc, &xpinc);
	XmScrollBarGetValues(vbar, &y, &ysize, &yinc, &ypinc);
	int xmax=CANVAS_SIZE-xsize-1;
	int ymax=CANVAS_SIZE-ysize-1;
	x-=dx;
	y-=dy;
	int nx=x;
	int ny=y;
	if(nx<0)nx=0;
	if(ny<0)ny=0;
	if(nx>xmax)nx=xmax;
	if(ny>ymax)ny=ymax;
	XmScrollBarSetValues(hbar, nx, xsize, xinc, xpinc, True);
	XmScrollBarSetValues(vbar, ny, ysize, yinc, ypinc, True);
	if(x != nx || y != ny){
	    slopx-=Abs(nx-x);
	    slopy-=Abs(ny-y);
	}
	nsx=event->x_root+x-nx;
	nsy=event->y_root+y-ny;
    }
}

NetworkEditor::NetworkEditor(Network* net)
: Task("Network Editor", 1), net(net)
{
}

NetworkEditor::~NetworkEditor()
{
}

void NetworkEditor::set_sched(Scheduler* s_sched)
{
    sched=s_sched;
}

int NetworkEditor::body(int)
{
    // Rendezvous with scheduler...

    // Create User interface...
    build_ui();

    initialized=1;
    update_needed=0;

    // Install timeout
    // Go into Main loop...
    // Substitute for XtAppMainLoop(app);
    while(toplevel){
	XEvent event;
	XtAppNextEvent(app, &event);
	XtDispatchEvent(&event);
	if(update_needed){
	    update_display();
	}
    }
    XtDestroyApplicationContext(app);

    // Signal scheduler to shut down...


    return 0;
}

void NetworkEditor::build_ui()
{
    // Build the window...
    int x_argc;
    char** x_argv;
    ArgProcessor::get_x_args(x_argc, x_argv);
    toplevel=XtAppInitialize(&app, "sci", NULL, 0,
			     &x_argc, x_argv,
			     fallback_resources,
			     NULL, 0);
    // Load fonts...
    if( (name_font = XLoadQueryFont(XtDisplay(toplevel), MOD_NAME_FONT)) == 0){
	cerr << "Error loading font: " << MOD_NAME_FONT << endl;
	exit(-1);
    }
    if( (time_font = XLoadQueryFont(XtDisplay(toplevel), MOD_TIME_FONT)) == 0){
	cerr << "Error loading font: " << MOD_TIME_FONT << endl;
	exit(-1);
    }

    // Allocate Colors...
    XColor col, unused;
    if(!XAllocNamedColor(XtDisplay(toplevel),
			 DefaultColormapOfScreen(XtScreen(toplevel)),
			 SCI_BACKGROUND_COLOR, &col, &unused)){
	cerr << "Warning: Can't allocate color" << endl;
	background_color=BlackPixelOfScreen(XtScreen(toplevel));
    } else {
	background_color=col.pixel;
    }
    if(!XAllocNamedColor(XtDisplay(toplevel),
			 DefaultColormapOfScreen(XtScreen(toplevel)),
			 EXECUTING_COLOR, &col, &unused)){
	cerr << "Warning: Can't allocate color" << endl;
	executing_color=WhitePixelOfScreen(XtScreen(toplevel));
    } else {
	executing_color=col.pixel;
    }
    XmGetColors(XtScreen(toplevel), DefaultColormapOfScreen(XtScreen(toplevel)),
		executing_color, NULL, &executing_color_top,
		&executing_color_bot, NULL);
    if(!XAllocNamedColor(XtDisplay(toplevel),
			 DefaultColormapOfScreen(XtScreen(toplevel)),
			 COMPLETED_COLOR, &col, &unused)){
	cerr << "Warning: Can't allocate color" << endl;
	completed_color=WhitePixelOfScreen(XtScreen(toplevel));
    } else {
	completed_color=col.pixel;
    }
    XmGetColors(XtScreen(toplevel), DefaultColormapOfScreen(XtScreen(toplevel)),
		completed_color, NULL, &completed_color_top,
		&completed_color_bot, NULL);

    // Create the rest...
    main_w = XtVaCreateManagedWidget("SCI Window",
				     xmMainWindowWidgetClass, toplevel,
				     XmNscrollingPolicy, XmAUTOMATIC,
				     XmNwidth, 800,
				     XmNheight, 600,
				     NULL);

    drawing_a = XtVaCreateManagedWidget("drawing_a",
					xmDrawingAreaWidgetClass,
					main_w,
					XmNunitType, XmPIXELS,
					XmNwidth, CANVAS_SIZE,
					XmNheight, CANVAS_SIZE,
					XmNmarginHeight, STAY_FROM_EDGE,
					XmNmarginWidth, STAY_FROM_EDGE,
					XmNresizePolicy, XmNONE,
					XmNbackground, background_color,
					XmNtranslations,
				          XtParseTranslationTable(network_translations),
					XmNuserData, this,

					NULL);
    XtAddCallback(drawing_a, XmNexposeCallback, do_redraw, (XtPointer)this);

    XtActionsRec actions[3];
    actions[0].string="module_move";
    actions[0].proc=(XtActionProc)do_module_move;
    actions[1].string="network_scroll";
    actions[1].proc=(XtActionProc)do_network_scroll;
    actions[2].string="connection_move";
    actions[2].proc=(XtActionProc)do_connection_move;
    XtAppAddActions(app, &actions[0], 3);

    XtVaGetValues(main_w, XmNbackground, &bg_color, NULL);
    XmGetColors(XtScreen(drawing_a), DefaultColormapOfScreen(XtScreen(drawing_a)),
		bg_color, &fg_color, &top_shadow, &bottom_shadow, &select_color);

    XtRealizeWidget(toplevel);

    // Initialize connections and modules...
    // We do some writing to the network, but it is all
    // either our own private data, or the update flags, where
    // locking is not necessary
    net->read_lock();
    int nconnections=net->nconnections();
    for(int i=0;i<nconnections;i++){
	Connection* conn=net->connection(i);
	initialize(conn);
    }
    int nmodules=net->nmodules();
    for(i=0;i<nmodules;i++){
	Module* mod=net->module(i);
	initialize(mod);
    }
    net->read_unlock();

    // Finally... Make the timer...
    timer_id=XtAppAddTimeOut(app, interval_time, do_timer, this);
}

void NetworkEditor::redraw(XtPointer cbdata)
{
    if(!have_gc){
	// Create the GC
	gc=XCreateGC(XtDisplay(drawing_a), XtWindow(drawing_a), 0, 0);
	have_gc=1;
    }
    redrawn_once=1;

    XmDrawingAreaCallbackStruct* cbs=(XmDrawingAreaCallbackStruct*)cbdata;
}

void NetworkEditor::initialize(Datatype* dt)
{
    XColor col, unused;
    if(!XAllocNamedColor(XtDisplay(toplevel),
			 DefaultColormapOfScreen(XtScreen(toplevel)),
			 dt->color_name(), &col, &unused)){
	cerr << "Warning: Can't allocate color" << endl;
	col.pixel=BlackPixelOfScreen(XtScreen(toplevel));
    }
    dt->color=col.pixel;
    XmGetColors(XtScreen(toplevel), DefaultColormapOfScreen(XtScreen(toplevel)),
		dt->color, NULL, &dt->top_shadow, &dt->bottom_shadow, NULL);
    dt->interface_initialized=1;
}

void NetworkEditor::initialize(Connection* conn)
{
    Datatype* dt=conn->oport->datatype;
    if(!dt->interface_initialized)
	initialize(dt);

    for(int i=0;i<5;i++){
	ConnectionWidgetCallbackData* cbdata=new ConnectionWidgetCallbackData;
	int x, y, w, h;
	calc_portwindow_size(conn, i, x, y, w, h);
	Widget da=XtVaCreateManagedWidget("drawing_a",
					  xmDrawingAreaWidgetClass, drawing_a,
					  XmNunitType, XmPIXELS,
					  XmNx, x,
					  XmNy, y,
					  XmNwidth, w,
					  XmNheight, h,
					  XmNresizePolicy, XmNONE,
					  XmNmarginHeight, 0,
					  XmNmarginWidth, 0,
					  XmNshadowThickness, 0,
					  XmNbackground, dt->color,
					  XmNtranslations,
					    XtParseTranslationTable(connection_translations),
					  XmNuserData, cbdata,
					  NULL);
	conn->drawing_a[i]=(void*)da;
	cbdata->conn=conn;
	cbdata->ne=this;
	cbdata->which_seg=i;
	XtAddCallback(da, XmNexposeCallback, do_con_redraw, (XtPointer)cbdata);
    }
}
    

void NetworkEditor::initialize(Module* mod)
{
    clString name(mod->get_name());
    int dir;
    int ascent;
    int descent;
    XCharStruct dim_title;
    if(!XTextExtents(name_font, name(), name.len(), &dir, &ascent, &descent,
		     &dim_title)){
	cerr << "XTextExtents failed...\n";
	exit(-1);
    }
    XCharStruct dim_time;
    static char* timestr="88:88";
    if(!XTextExtents(time_font, timestr, strlen(timestr), &dir, &ascent, &descent,
		     &dim_time)){
	cerr << "XTextExtents failed...\n";
	exit(-1);
    }
    mod->ytitle=MOD_EDGE_WIDTH+MOD_PORT_SIZE+MOD_PORT_SPACE
	+MOD_BUTTON_BORDER+MOD_BUTTON_EDGE+MOD_BUTTON_SIZE
	    + MOD_BUTTON_EDGE+MOD_BUTTON_BORDER+MOD_TITLE_TOP_SPACE
		+dim_title.ascent;
    mod->ygraphtop=mod->ytitle+dim_title.descent
	+MOD_TITLE_BOT_SPACE+MOD_GRAPH_INSET;
    mod->ytime=mod->ygraphtop+dim_time.ascent;
    mod->ygraphbot=mod->ytime+dim_time.descent;
    mod->height=mod->ygraphbot+MOD_GRAPH_INSET+MOD_PORT_SPACE
	+MOD_PORT_SIZE+MOD_EDGE_WIDTH;
    int twidth=MOD_EDGE_WIDTH+MOD_SIDE_BORDER+ dim_title.width
	+MOD_SIDE_BORDER+MOD_EDGE_WIDTH;
    int bwidth=MOD_EDGE_WIDTH+MOD_SIDE_BORDER
	+ (MOD_NBUTTONS+1)*MOD_BUTTON_BORDER
	+ MOD_NBUTTONS*(MOD_BUTTON_SIZE+2*MOD_BUTTON_EDGE)
	+ MOD_SIDE_BORDER+MOD_EDGE_WIDTH;
    mod->width=bwidth;
    if(twidth > mod->width)mod->width=twidth;
    mod->xgraphleft=MOD_EDGE_WIDTH+MOD_SIDE_BORDER+dim_time.width
	+MOD_GRAPH_TEXT_SPACE+MOD_GRAPH_INSET;
    int gheight=mod->ygraphbot-mod->ygraphtop;
    mod->xgraphright=mod->width-MOD_EDGE_WIDTH-MOD_SIDE_BORDER-gheight
	-MOD_GRAPH_BUTT_SPACE-MOD_GRAPH_INSET;
    mod->wcbdata=new ModuleWidgetCallbackData;
    Widget da=XtVaCreateManagedWidget("drawing_a",
				      xmDrawingAreaWidgetClass, drawing_a,
				      XmNunitType, XmPIXELS,
				      XmNx, mod->xpos,
				      XmNy, mod->ypos,
				      XmNwidth, mod->width,
				      XmNheight, mod->height,
				      XmNresizePolicy, XmNONE,
				      XmNmarginHeight, 0,
				      XmNmarginWidth, 0,
				      XmNshadowThickness, 0,
				      XmNbackground, bg_color,
				      XmNtranslations,
				        XtParseTranslationTable(module_translations),
				      XmNuserData, mod->wcbdata,
				      NULL);
    mod->wcbdata->ne=this;
    mod->wcbdata->mod=mod;
    XtAddCallback(da, XmNexposeCallback, do_mod_redraw, (XtPointer)mod->wcbdata);
    mod->drawing_a=(void*)da;
    mod->interface_initialized=1;
}

void NetworkEditor::update_display()
{
    if(!redrawn_once)return;
    // We do some writing to the network, but it is all
    // either our own private data, or the update flags, where
    // locking is not necessary
    net->read_lock();
    int nmodules=net->nmodules();
    for(int i=0;i<nmodules;i++){
	Module* mod=net->module(i);
	if(!mod->interface_initialized)
	    initialize(mod);
	if(mod->needs_update()
	   || mod->get_state() == Module::Executing){
	    update_module(mod, 1);
	    mod->updated();
	}
    }
    net->read_unlock();
}


void NetworkEditor::draw_module(Module* mod)
{
    Widget da=(Widget)mod->drawing_a;
    Display* dpy=XtDisplay(da);
    Drawable win=XtWindow(da);

    // Draw base
    //XClearWindow(dpy, win);
    XSetForeground(dpy, gc, top_shadow);
    draw_shadow(dpy, win, gc, 0, 0, mod->width-1, mod->height-1,
		MOD_EDGE_WIDTH, top_shadow, bottom_shadow);

    // Draw Input ports
    int port_spacing=MOD_PORTPAD_WIDTH+MOD_PORTPAD_SPACE;
    int niports=mod->niports();
    for(int p=0;p<niports;p++){
	IPort* iport=mod->iport(p);
	Datatype* dt=iport->datatype;
	if(!dt->interface_initialized)
	    initialize(dt);
	XSetForeground(dpy, gc, dt->top_shadow);
	int left=p*port_spacing+MOD_EDGE_WIDTH+MOD_SIDE_BORDER;
	int right=left+MOD_PORTPAD_WIDTH-1;
	for(int i=0;i<MOD_EDGE_WIDTH;i++){
	    XDrawLine(dpy, win, gc, left, i, right, i);
	}
	XSetForeground(dpy, gc, dt->color);
	int t=MOD_EDGE_WIDTH;
	for(i=0;i<MOD_PORT_SIZE;i++){
	    XDrawLine(dpy, win, gc, left, i+t, right, i+t);
	}
	if(iport->nconnections() > 0){
	    // Draw tab...
	    int p2=(MOD_PORTPAD_WIDTH-PIPE_WIDTH-2*PIPE_SHADOW_WIDTH)/2;
	    int l=left+p2+PIPE_SHADOW_WIDTH;
	    XSetForeground(dpy, gc, dt->color);
	    for(int i=0;i<PIPE_WIDTH;i++){
		XDrawLine(dpy, win, gc, l+i, 0, l+i, MOD_EDGE_WIDTH-1);
	    }
	    XSetForeground(dpy, gc, dt->bottom_shadow);
	    l+=PIPE_WIDTH;
	    for(i=0;i<PIPE_SHADOW_WIDTH;i++){
		XDrawLine(dpy, win, gc, l+i, 0, l+i, MOD_EDGE_WIDTH-i-1);
	    }
	}
    }

    // Draw Output ports
    int noports=mod->noports();
    for(p=0;p<noports;p++){
	OPort* oport=mod->oport(p);
	Datatype* dt=oport->datatype;
	if(!dt->interface_initialized)
	    initialize(dt);
	int h=mod->height;
	int left=p*port_spacing+MOD_EDGE_WIDTH+MOD_SIDE_BORDER;
	int right=left+MOD_PORTPAD_WIDTH-1;
	XSetForeground(dpy, gc, dt->bottom_shadow);
	for(int i=0;i<MOD_EDGE_WIDTH;i++){
	    XDrawLine(dpy, win, gc, left, h-i-1, right, h-i-1);
	}
	XSetForeground(dpy, gc, dt->color);
	int t=MOD_EDGE_WIDTH+1;
	for(i=0;i<MOD_PORT_SIZE;i++){
	    XDrawLine(dpy, win, gc, left, h-i-t, right, h-i-t);
	}
	if(oport->nconnections() > 0){
	    // Draw tab...
	    int p2=(MOD_PORTPAD_WIDTH-PIPE_WIDTH-2*PIPE_SHADOW_WIDTH)/2;
	    int l=left+p2;
	    XSetForeground(dpy, gc, dt->top_shadow);
	    for(i=0;i<PIPE_SHADOW_WIDTH;i++){
		XDrawLine(dpy, win, gc, l+i, h-i-1, l+i, h-1);
	    }
	    l+=PIPE_SHADOW_WIDTH;
	    XSetForeground(dpy, gc, dt->color);
	    for(int i=0;i<PIPE_WIDTH;i++){
		XDrawLine(dpy, win, gc, l+i, h-MOD_EDGE_WIDTH-1, l+i, h-1);
	    }
	}
    }

    // Draw buttons
    int ybtop=MOD_EDGE_WIDTH+MOD_PORT_SIZE+MOD_PORT_SPACE;
    int ybbot=ybtop+MOD_BUTTON_BORDER+MOD_BUTTON_EDGE+MOD_BUTTON_SIZE
	+MOD_BUTTON_EDGE;
    int xbleft=MOD_EDGE_WIDTH+MOD_SIDE_BORDER;
    int xbsize=MOD_BUTTON_BORDER+MOD_BUTTON_SIZE
	+2*MOD_BUTTON_EDGE;
    int xbright=xbleft+MOD_BUTTON_BORDER+MOD_NBUTTONS*xbsize;
    // Draw border..
    XSetForeground(dpy, gc, fg_color);
    XDrawLine(dpy, win, gc, xbleft, ybtop, xbright-1, ybtop);
    XDrawLine(dpy, win, gc, xbleft, ybbot, xbright-1, ybbot);
    int x=xbleft;
    ybtop+=MOD_BUTTON_BORDER;
    ybbot-=MOD_BUTTON_BORDER;
    xbleft+=MOD_BUTTON_BORDER;
    for(int i=0;i<MOD_NBUTTONS+1;i++){
	XDrawLine(dpy, win, gc, x, ybtop, x, ybbot);
	x+=xbsize;
    }
    x=xbleft;
    int xbsize2=xbsize-MOD_BUTTON_BORDER;
    for(i=0;i<MOD_NBUTTONS;i++){
	draw_shadow(dpy, win, gc, x, ybtop, x+xbsize2-1, ybbot,
		    MOD_BUTTON_EDGE, top_shadow, bottom_shadow);
	x+=xbsize;
    }

    // Draw title
    clString name(mod->get_name());
    XSetFont(dpy, gc, name_font->fid);
    XSetForeground(dpy, gc, fg_color);
    int xleft=MOD_EDGE_WIDTH+MOD_SIDE_BORDER;
    XDrawString(dpy, win, gc, xleft, mod->ytitle, name(), name.len());

    // Draw time and graph...
    update_module(mod, 0);
}

void NetworkEditor::update_module(Module* mod, int clear_first)
{
    Widget da=(Widget)mod->drawing_a;
    Display* dpy=XtDisplay(da);
    Drawable win=XtWindow(da);
    int yginsettop=mod->ygraphtop-MOD_GRAPH_INSET;
    int xginsetleft=mod->xgraphleft-MOD_GRAPH_INSET;
    int xginsetright=mod->xgraphright+MOD_GRAPH_INSET;
    int yginsetbot=mod->ygraphbot+MOD_GRAPH_INSET;
    int ginsetheight=yginsetbot-yginsettop+1;
    int xleft=MOD_EDGE_WIDTH+MOD_SIDE_BORDER;
    if(clear_first){
	XSetForeground(dpy, gc, bg_color);
	XFillRectangle(dpy, win, gc, xleft, yginsettop,
		       xginsetright-xleft+1, ginsetheight+1);
    }
    // Draw time/graph
    double time=mod->get_execute_time();
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
    XSetFont(dpy, gc, time_font->fid);
    XSetForeground(dpy, gc, fg_color);
    XDrawString(dpy, win, gc, xleft, mod->ytime, timebuf, timelen);

    // Draw indent for graph
    XSetLineAttributes(dpy, gc, 0, LineSolid, CapButt, JoinMiter);
    XSetForeground(dpy, gc, bottom_shadow);
    draw_shadow(dpy, win, gc, 
		xginsetleft, yginsettop, xginsetright, yginsetbot,
		MOD_GRAPH_INSET, bottom_shadow, top_shadow);

    // Draw Graph
    XSetForeground(dpy, gc, select_color);
    int total_gwidth=mod->xgraphright-mod->xgraphleft+1;
    int gheight=mod->ygraphbot-mod->ygraphtop+1;
    XFillRectangle(dpy, win, gc, mod->xgraphleft, mod->ygraphtop,
		   total_gwidth, gheight);
    double completed;
    Pixel gtop;
    Pixel gbot;
    switch(mod->get_state()){
    case Module::NeedData:
	completed=0;
	gtop=gbot=0;
	break;
    case Module::Executing:
	completed=mod->get_progress();
	completed=completed<0?0:completed>1?1:completed;
	XSetForeground(dpy, gc, executing_color);
	gtop=executing_color_top;
	gbot=executing_color_bot;
	break;
    case Module::Completed:
	completed=1;
	XSetForeground(dpy, gc, completed_color);
	gtop=completed_color_top;
	gbot=completed_color_bot;
	break;
    }
    int gwidth=(int)(completed*total_gwidth);
    if(gwidth==0){
	// Do nothing...
    } else if(gwidth <= 2*MOD_GRAPH_SHADOW+1){
	XFillRectangle(dpy, win, gc, mod->xgraphleft, mod->ygraphtop,
		       gwidth+1, gheight);
    } else {
	XFillRectangle(dpy, win, gc,
		       mod->xgraphleft+MOD_GRAPH_SHADOW,
		       mod->ygraphtop+MOD_GRAPH_SHADOW,
		       gwidth-2*MOD_GRAPH_SHADOW, gheight-2*MOD_GRAPH_SHADOW);
	draw_shadow(dpy, win, gc, mod->xgraphleft, mod->ygraphtop,
		    mod->xgraphleft+gwidth-1, mod->ygraphbot,
		    MOD_GRAPH_SHADOW,
		    gtop, gbot);
    }
}

void NetworkEditor::module_move(Module* mod, XButtonEvent* event, String arg)
{
    if(strcmp(arg, "down") == 0){
	drag_sx=event->x_root;
	drag_sy=event->y_root;
	dragmode = MOVE_WIDGET;
    } else if(strcmp(arg, "up") == 0
	      || strcmp(arg, "motion") == 0){
	switch(dragmode){
	case MOVE_NONE:
	    break;
	case MOVE_WIDGET:
	    // New position...
	    {
		int dx=event->x_root-drag_sx;
		int dy=event->y_root-drag_sy;
		if(drag_slop_x < 0){
		    drag_slop_x+=Abs(dx);
		    if(drag_slop_x > 0){
			dx=Sign(dx)*drag_slop_x;
			drag_slop_x=0;
		    }
		}
		if(drag_slop_y < 0){
		    drag_slop_y+=Abs(dy);
		    if(drag_slop_y > 0){
			dy=Sign(dy)*drag_slop_y;
			drag_slop_y=0;
		    }
		}
		       
		mod->xpos+=dx;
		mod->ypos+=dy;

		XtVaSetValues((Widget)mod->drawing_a,
			      XmNx, mod->xpos,
			      XmNy, mod->ypos,
			      NULL);
		Position nx, ny;
		XtVaGetValues((Widget)mod->drawing_a,
			      XmNx, &nx,
			      XmNy, &ny,
			      NULL);
		if(nx != mod->xpos || ny!=mod->ypos){
		    drag_slop_x-=Abs(nx-mod->xpos);
		    drag_slop_y-=Abs(ny-mod->ypos);
		    mod->xpos=nx;
		    mod->ypos=ny;
		}
		drag_sx=event->x_root;
		drag_sy=event->y_root;

		// Update connections...
		for(int p=0;p<mod->niports();p++){
		    Port* ip=mod->iport(p);
		    for(int c=0;c<ip->nconnections();c++){
			Connection* conn=ip->connection(c);
			for(int i=0;i<5;i++){
			    int x,y,w,h;
			    calc_portwindow_size(conn, i, x, y, w, h);
			    Widget da=(Widget)conn->drawing_a[i];
			    XtVaSetValues(da, XmNx, x, XmNy, y,
					  XmNwidth, w, XmNheight, h, NULL);
			    XClearWindow(XtDisplay(da), XtWindow(da));
			    draw_connection(conn, i);
			}
		    }
		}
		for(p=0;p<mod->noports();p++){
		    Port* op=mod->oport(p);
		    for(int c=0;c<op->nconnections();c++){
			Connection* conn=op->connection(c);
			for(int i=0;i<5;i++){
			    int x,y,w,h;
			    calc_portwindow_size(conn, i, x, y, w, h);
			    Widget da=(Widget)conn->drawing_a[i];
			    XtVaSetValues(da, XmNx, x, XmNy, y,
					  XmNwidth, w, XmNheight, h, NULL);
			    XClearWindow(XtDisplay(da), XtWindow(da));
			    draw_connection(conn, i);
			}
		    }
		}
	    }
	    break;
	}
	if(strcmp(arg, "up") == 0)
	    dragmode=MOVE_NONE;
    } else {
	dragmode=MOVE_NONE;
    }
}

void NetworkEditor::connection_move(Module* mod, XButtonEvent* event, String arg)
{
    if(strcmp(arg, "down") == 0){
    } else if(strcmp(arg, "up") == 0
	      || strcmp(arg, "motion") == 0){
    }
}

void NetworkEditor::timer()
{
    // Reset the timer...
    timer_id=XtAppAddTimeOut(app, interval_time, do_timer, this);
    update_display();
}

void NetworkEditor::draw_shadow(Display* dpy, Window win, GC gc,
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

void NetworkEditor::get_iport_coords(Module* mod, int which, int& x, int& y)
{
    int port_spacing=MOD_PORTPAD_WIDTH+MOD_PORTPAD_SPACE;
    int p2=(MOD_PORTPAD_WIDTH-PIPE_WIDTH-2*PIPE_SHADOW_WIDTH)/2;
    x=mod->xpos+which*port_spacing+MOD_EDGE_WIDTH+MOD_SIDE_BORDER+p2;
    y=mod->ypos;
}

void NetworkEditor::get_oport_coords(Module* mod, int which, int& x, int& y)
{
    get_iport_coords(mod, which, x, y);
    y+=mod->height;
}

void NetworkEditor::draw_connection(Connection* conn, int which_seg)
{
    Dimension w, h;
    Widget wg=(Widget)conn->drawing_a[which_seg];
    Display* dpy=XtDisplay(wg);
    Drawable win=XtWindow(wg);
    XtVaGetValues(wg, XmNwidth, &w, XmNheight, &h, NULL);
    Datatype* dt=conn->oport->datatype;
    if(!dt->interface_initialized)
	initialize(dt);
    Pixel bottom_shadow=dt->bottom_shadow;
    Pixel top_shadow=dt->top_shadow;
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
	IPort* iport=conn->iport;
	OPort* oport=conn->oport;
	Module* imod=iport->module;
	Module* omod=oport->module;
	int iwhich=iport->which_port;
	int owhich=oport->which_port;
	int ix, iy;
	get_iport_coords(imod, iwhich, ix, iy);
	int ox, oy;
	get_oport_coords(omod, owhich, ox, oy);
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
}

void NetworkEditor::calc_portwindow_size(Connection* conn, int which,
					 int& x, int& y, int& w, int& h)
{
    IPort* iport=conn->iport;
    OPort* oport=conn->oport;
    Module* imod=iport->module;
    Module* omod=oport->module;
    int iwhich=iport->which_port;
    int owhich=oport->which_port;
    int ix, iy;
    get_iport_coords(imod, iwhich, ix, iy);
    int ox, oy;
    get_oport_coords(omod, owhich, ox, oy);
    int cx=(ix+ox)/2;
    int cy=(iy+oy)/2;
    int width=PIPE_WIDTH+2*PIPE_SHADOW_WIDTH;
    int ly=oy+MIN_WIRE_EXTEND;
    int uy=iy-MIN_WIRE_EXTEND;
    if(ox < ix){
	if(cx >= imod->xpos-MOD_PORT_SPACE){
	    cx=imod->xpos-MOD_PORT_SPACE;
	}
	if(cx <= omod->xpos+omod->width+MOD_PORT_SPACE){
	    cx=omod->xpos+omod->width+MOD_PORT_SPACE;
	}
    } else {
	if(cx >= omod->xpos-MOD_PORT_SPACE){
	    cx=omod->xpos-MOD_PORT_SPACE;
	}
	if(cx <= imod->xpos+imod->width+MOD_PORT_SPACE){
	    cx=imod->xpos+imod->width+MOD_PORT_SPACE;
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
