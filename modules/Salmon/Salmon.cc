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
#include <stdio.h>
#include <Salmon/Salmon.h>
#include <Salmon/Roe.h>
#include <CallbackCloners.h>
#include <Connection.h>
#include <HelpUI.h>
#include <MessageTypes.h>
#include <ModuleHelper.h>
#include <ModuleList.h>
#include <ModuleShape.h>
#include <MotifCallback.h>
#include <MtXEventLoop.h>
#include <NetworkEditor.h>
#include <NotFinished.h>
#include <XQColor.h>
#include <Mt/DrawingArea.h>
#include <PopupMenu.h>
#include <Mt/PushButton.h>
#include <iostream.h>
#include <Geom.h>
#include <Classlib/HashTable.h>

extern MtXEventLoop* evl;

static Module* make_Salmon()
{
    return new Salmon;
}

static RegisterModule db1("Geometry", "Salmon", make_Salmon);

Salmon::Salmon()
: Module("Salmon", Sink), max_portno(0), need_reconfig(0)
{
    // Create the input port
    add_iport(new GeometryIPort(this, "Geometry", GeometryIPort::Atomic));
cerr << &mailbox << "\n";
    default_matl=new MaterialProp(Color(.1,.1,.1),
				  Color(.6,0,0),
				  Color(.7,.7,.7),
				  10);
}

Salmon::~Salmon()
{
    delete default_matl;
}

Module* Salmon::clone(int deep)
{
    return new Salmon(*this, deep);
}

void Salmon::do_execute()
{
    while(1){
	MessageBase* msg=mailbox.receive();
	GeometryComm* gmsg=(GeometryComm*)msg;
	switch(msg->type){
	case MessageTypes::DoCallback:
	    {
		Callback_Message* cmsg=(Callback_Message*)msg;
		cmsg->mcb->perform(cmsg->cbdata);
		if(cmsg->cbdata)delete cmsg->cbdata;
	    }
	    break;
	case MessageTypes::GeometryInit:
	    initPort(gmsg->reply);
	    break;	
	case MessageTypes::GeometryAddObj:
	    addObj(gmsg->portno, gmsg->serial, gmsg->obj);
	    break;
	case MessageTypes::GeometryDelObj:
	    delObj(gmsg->portno, gmsg->serial);
	    break;
	case MessageTypes::GeometryDelAll:
	    delAll(gmsg->portno);
	    break;
	case MessageTypes::GeometryFlush:
	    flushViews();
	    break;
	default:
	    cerr << "Salomon: Illegal Message type: " << msg->type << endl;
	    break;
	}
	delete msg;
    }
}

void Salmon::create_interface()
{
    // Create the module icon
    evl->lock(); // Lock just once - for efficiency

    int dir;
    int title_ascent;
    int title_descent;
    XCharStruct dim_title;
    if(!XTextExtents(netedit->name_font, name(), name.len(), &dir,
		     &title_ascent, &title_descent, &dim_title)){
	cerr << "XTextExtents failed...\n";
	exit(-1);
    }
    title_width=dim_title.width;
    int time_ascent;
    int time_descent;
    XCharStruct dim_time;
    static char* timestr="88:88";
    if(!XTextExtents(netedit->time_font, timestr, strlen(timestr), &dir,
		     &time_ascent, &time_descent, &dim_time)){
	cerr << "XTextExtents failed...\n";
	exit(-1);
    }
    int widget_ytop=MODULE_EDGE_WIDTH+MODULE_PORT_SIZE+MODULE_PORTLIGHT_HEIGHT
	+MODULE_PORT_SPACE;
    widget_ytitle=widget_ytop+title_ascent;
    int widget_ygraphtop=widget_ytitle+title_descent;
    time_ascent=dim_time.ascent;
    time_descent=dim_time.descent;
    int widget_ytime=widget_ygraphtop+MODULE_GRAPH_INSET+time_ascent;
    int widget_ygraphbot=widget_ytime+time_descent;
    int bbot=widget_ytop+2*MODULE_BUTTON_SHADOW+MODULE_BUTTON_SIZE;
    int b=Min(widget_ygraphbot, bbot);
    height=widget_ygraphbot+MODULE_GRAPH_INSET+MODULE_PORT_SPACE
	+MODULE_PORTLIGHT_HEIGHT+MODULE_PORT_SIZE+MODULE_EDGE_WIDTH;
    int btn_left=MODULE_EDGE_WIDTH+MODULE_SIDE_BORDER;
    title_left=btn_left+2*MODULE_BUTTON_SHADOW+MODULE_BUTTON_SIZE
	+MODULE_TITLE_LEFT_SPACE;

    bgcolor=new XQColor(netedit->color_manager, "salmon");
    fgcolor=bgcolor->fg_color();
    top_shadow=bgcolor->top_shadow();
    bottom_shadow=bgcolor->bottom_shadow();
    drawing_a=new DrawingAreaC;
    drawing_a->SetUnitType(XmPIXELS);
    drawing_a->SetX(xpos);
    drawing_a->SetY(ypos);
    width=compute_width();
    drawing_a->SetWidth(width);
    drawing_a->SetHeight(height);
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
    // Add button action callbacks.  These must be done after Create()
    new MotifCallback<Salmon>FIXCB(drawing_a,
				   "<Btn1Down>",
				   &netedit->mailbox, this,
				   &Salmon::move_widget, 0,
				   &CallbackCloners::event_clone);
    new MotifCallback<Salmon>FIXCB(drawing_a,
				   "<Btn1Up>",
				   &netedit->mailbox, this,
				   &Salmon::move_widget, 0,
				   &CallbackCloners::event_clone);
    new MotifCallback<Salmon>FIXCB(drawing_a,
				   "<Btn1Motion>",
				   &netedit->mailbox, this,
				   &Salmon::move_widget, 0,
				   &CallbackCloners::event_clone);
    new MotifCallback<NetworkEditor>FIXCB(drawing_a,
					  "<Btn2Down>",
					  &netedit->mailbox, netedit,
					  &NetworkEditor::connection_cb,
					  this,
					  &CallbackCloners::event_clone);
    new MotifCallback<Salmon>FIXCB(drawing_a,
				   "<Btn3Down>",
				   &netedit->mailbox, this,
				   &Salmon::post_menu, 0,
				   &CallbackCloners::event_clone);
    gc=XCreateGC(XtDisplay(*drawing_a), XtWindow(*drawing_a), 0, 0);

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
    new MotifCallback<Salmon>FIXCB(btn, XmNactivateCallback,
				   &netedit->mailbox, this,
				   &Salmon::widget_button,
				   0, 0);
    btn->Create(*drawing_a, "UI");
    evl->unlock();

    // Create the viewer window...
    topRoe.add(new Roe(this));
    topRoe[topRoe.size()-1]->SetTop();
    
//    printFamilyTree();

    // Start up the event loop thread...
    helper=new ModuleHelper(this, 1);
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

void Salmon::redraw_widget(CallbackData*, void*)
{
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
    XSetFont(dpy, gc, netedit->name_font->fid);
    XSetForeground(dpy, gc, fgcolor->pixel());
    XDrawString(dpy, win, gc, title_left, widget_ytitle, name(), name.len());
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

void Salmon::initPort(Mailbox<int>* reply)
{
    reply->send(max_portno++);
}

void Salmon::flushViews()
{
    for (int i=0; i<topRoe.size(); i++) {
	topRoe[i]->redrawAll();
    }
}

void Salmon::addObj(int portno, int serial, GeomObj *obj)
{
//    cerr << "I'm adding an Object!\n";
    HashTable<int, GeomObj*>* serHash;
    if (!portHash.lookup(portno, serHash)) {
	// need to make this table
	serHash = new HashTable<int, GeomObj*>;
	portHash.insert(portno, serHash);
    }
    serHash->insert(serial, obj);
    char nm[30];
    sprintf(nm, "Item %d", serial);
    for (int i=0; i<topRoe.size(); i++) {
	topRoe[i]->itemAdded(obj, nm);
    }
}

void Salmon::delObj(int portno, int serial)
{
    HashTable<int, GeomObj*>* serHash;
    if (portHash.lookup(portno, serHash)) {
	GeomObj *g;
	serHash->lookup(serial, g);
	serHash->remove(serial);
	for (int i=0; i<topRoe.size(); i++) {
	    topRoe[i]->itemDeleted(g);
	}
    }
}

void Salmon::printFamilyTree()
{
    cerr << "\nSalmon Family Tree\n";
    for (int i=0, flag=1; flag!=0; i++) {
	flag=0;
	for (int j=0; j<topRoe.size(); j++) {
	    topRoe[j]->printLevel(i, flag);
	}
	cerr << "\n";
    }
}

void Salmon::delAll(int portno)
{

    HashTable<int, GeomObj*>* serHash;
    if (portHash.lookup(portno, serHash)) {
	HashTableIter<int, GeomObj*> iter(serHash);
	for (iter.first(); iter.ok(); ++iter) {
	    GeomObj* g=iter.get_data();
	    int serial=iter.get_key();
	    serHash->lookup(serial, g);
	    serHash->remove(serial);
	    for (int i=0; i<topRoe.size(); i++) {
		topRoe[i]->itemDeleted(g);
	    }
	}
    }
}

void Salmon::addTopRoe(Roe *r)
{
    topRoe.add(r);
}

void Salmon::delTopRoe(Roe *r)
{
    for (int i=0; i<topRoe.size(); i++) {
	if (r==topRoe[i]) topRoe.remove(i);
    }
} 

void Salmon::spawnIndCB(CallbackData*, void*)
{
  topRoe.add(new Roe(this));
  topRoe[topRoe.size()-1]->SetTop();
  GeomItem *item;
  for (int i=0; i<topRoe[0]->geomItemA.size(); i++) {
      item=topRoe[0]->geomItemA[i];
      topRoe[topRoe.size()-1]->itemAdded(item->geom, item->name);
  }
  topRoe[topRoe.size()-1]->redrawAll();
//  printFamilyTree();
}

Salmon::Salmon(const Salmon& copy, int deep)
: Module(copy, deep)
{
    NOT_FINISHED("Salmon::Salmon");
}

void Salmon::reconfigure_iports()
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

void Salmon::reconfigure_oports()
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
	evl->lock();
	drawing_a->SetWidth(width);
	drawing_a->SetValues();
	XClearWindow(netedit->display, XtWindow(*drawing_a));
	evl->unlock();
    }
    redraw_widget(0, 0);
}

void Salmon::widget_button(CallbackData*, void*)
{
    NOT_FINISHED("Salmon::widget_button");
}

void Salmon::move_widget(CallbackData* cbdata, void*)
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


void Salmon::post_menu(CallbackData* cbdata, void*)
{
    if(netedit->check_cancel())
	return;
    evl->lock();
    if(!popup_menu){
	popup_menu=new PopupMenuC;
	popup_menu->Create(*drawing_a, "popup");
	PushButtonC* pb=new PushButtonC;
	new MotifCallback<Salmon>FIXCB(pb, XmNactivateCallback,
				       &netedit->mailbox, this,
				       &Salmon::destroy, 0, 0);
	pb->Create(*popup_menu, "Destroy");
	pb=new PushButtonC;
	new MotifCallback<Salmon>FIXCB(pb, XmNactivateCallback,
				       &netedit->mailbox, this,
				       &Salmon::popup_help, 0, 0);
	pb->Create(*popup_menu, "Help...");
    }
    XmMenuPosition(*popup_menu, (XButtonPressedEvent*)cbdata->get_event());
    XtManageChild(*popup_menu);
    evl->unlock();
}

void Salmon::destroy(CallbackData*, void*)
{
    NOT_FINISHED("Salmon::destroy");
}

void Salmon::popup_help(CallbackData*, void*)
{
    HelpUI::load(name);
}

int Salmon::compute_width()
{
    int w=title_left+title_width+MODULE_SIDE_BORDER+MODULE_EDGE_WIDTH;
    int port_spacing=MODULE_PORTPAD_WIDTH+MODULE_PORTPAD_SPACE;
    int p2=(MODULE_PORTPAD_WIDTH-PIPE_WIDTH-2*PIPE_SHADOW_WIDTH)/2;
    int np=Max(iports.size(), oports.size());
    int x=np*port_spacing+2*(MODULE_EDGE_WIDTH+MODULE_SIDE_BORDER);
    return Max(x, w);
}

void Salmon::connection(ConnectionMode mode, int which_port,
			int output)
{
    if(mode==Disconnected){
	remove_iport(which_port);
    } else {
	add_iport(new GeometryIPort(this, "Geometry", GeometryIPort::Atomic));
    }
}
