
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

// Someday, we should delete these four lines, when the
// compiler stops griping about const cast away...
#include <X11/Intrinsic.h>
#include "myStringDefs.h"
#include "myXmStrDefs.h"
#include "myShell.h"

#include <NetworkEditor.h>
#include <CallbackCloners.h>
#include <Connection.h>
#include <MemStats.h>
#include <MessageBase.h>
#include <Module.h>
#include <ModuleList.h>
#include <MotifCallbackBase.h>
#include <MotifCallback.h>
#include <MtXEventLoop.h>
#include <Network.h>
#include <NotFinished.h>
#include <Port.h>
#include <XQColor.h>
#include <Math/MiscMath.h>
#include <Mt/ApplicationShell.h>
#include <Mt/CascadeButton.h>
#include <Mt/DrawingArea.h>
#include <Mt/Label.h>
#include <Mt/List.h>
#include <Mt/MainWindow.h>
#include <MenuBar.h>
#include <Mt/PanedWindow.h>
#include <Mt/PushButton.h>
#include <Mt/RowColumn.h>
#include <Mt/ScrolledWindow.h>
#include <Mt/Separator.h>
#include <Mt/Text.h>

extern MtXEventLoop* evl;

// Someday, we should make these resources...
#define NETEDIT_WINDOW_WIDTH 800
#define NETEDIT_WINDOW_HEIGHT 600
#define NETEDIT_CANVAS_SIZE 2000
#define NETEDIT_STAY_FROM_EDGE 2
#define NETEDIT_BACKGROUND_COLOR "#224488"
#define MODULE_NAME_FONT "-*-lucida-bold-r-*-*-14-*-*-*-*-*-*-*"
#define MODULE_TIME_FONT "-*-lucida-medium-r-*-*-11-*-*-*-*-*-*-*"
#define LIST_VISIBLE 4

NetworkEditor::NetworkEditor(Network* net, Display* display,
			     ColorManager* color_manager)
: Task("Network Editor", 1), net(net), display(display),
  color_manager(color_manager), mailbox(100), first_schedule(1),
  memstats(0), making_connection(0)
{
}

NetworkEditor::~NetworkEditor()
{
}

int NetworkEditor::body(int)
{
    // Create User interface...
    window=new ApplicationShellC;
    window->SetTitle("Project with no name");
    window->Create("sci", "sci", display);

    // Allocate fonts...
    evl->lock();
    if( (name_font = XLoadQueryFont(display, MODULE_NAME_FONT)) == 0){
	cerr << "Error loading font: " << MODULE_NAME_FONT << endl;
	exit(-1);
    }
    if( (time_font = XLoadQueryFont(display, MODULE_TIME_FONT)) == 0){
	cerr << "Error loading font: " << MODULE_TIME_FONT << endl;
	exit(-1);
    }
    
    // Allocate Colors...
    XQColor background(color_manager, NETEDIT_BACKGROUND_COLOR);

    MainWindowC mainw;
    mainw.Create(*window, "main_window");

    MenuBarC menubar(mainw);
    MenuC* file=menubar.AddMenu("File");
    PushButtonC* quitbutton=file->AddButton("Quit");
    new MotifCallback<NetworkEditor>FIXCB(quitbutton, XmNactivateCallback,
					  &mailbox, this,
					  &NetworkEditor::quit,
					  0, 0);
    MenuC* stats=menubar.AddMenu("Statistics");
    PushButtonC* membutton=stats->AddButton("Memory");
    new MotifCallback<NetworkEditor>FIXCB(membutton, XmNactivateCallback,
					  &mailbox, this,
					  &NetworkEditor::popup_memstats,
					  0, 0);

    PanedWindowC pane;
    pane.Create(mainw, "pane");

    RowColumnC lists;
    lists.SetOrientation(XmHORIZONTAL);
    lists.Create(pane, "lists");

    RowColumnC rc1;
    rc1.Create(lists, "rc");
    LabelC label1;
    label1.Create(rc1, "Category:");
    ScrolledWindowC slist1;
    slist1.Create(rc1, "slist1");
    list1=new ListC;
    list1->SetVisibleItemCount(LIST_VISIBLE);
    new MotifCallback<NetworkEditor>FIXCB(list1, XmNbrowseSelectionCallback,
					  &mailbox, this,
					  &NetworkEditor::list1_cb, 0,
					  &CallbackCloners::list_clone);
    list1->Create(slist1, "list1");
    // Fill in the top level list...
    current_db=ModuleList::get_db();
    ModuleDBIter iter(current_db);
    int item=1;
    for(iter.first();iter.ok();++iter){
	clString k(iter.get_key());
	XmString s1=XmStringCreateSimple(k());
	XmListAddItem(*list1, s1, item++);
	XmStringFree(s1);
    }

    RowColumnC rc2;
    rc2.Create(lists, "rc");
    LabelC label2;
    label2.Create(rc2, "SubCategory:");
    ScrolledWindowC slist2;
    slist2.Create(rc2, "slist2");
    list2=new ListC;
    list2->SetVisibleItemCount(LIST_VISIBLE);
    new MotifCallback<NetworkEditor>FIXCB(list2, XmNbrowseSelectionCallback,
					  &mailbox, this,
					  &NetworkEditor::list2_cb, 0,
					  &CallbackCloners::list_clone);
    list2->Create(slist2, "list2");

    RowColumnC rc3;
    rc3.Create(lists, "rc");
    LabelC label3;
    label3.Create(rc3, "Module:");
    ScrolledWindowC slist3;
    slist3.Create(rc3, "slist3");
    list3=new ListC;
    list3->SetVisibleItemCount(LIST_VISIBLE);
    new MotifCallback<NetworkEditor>FIXCB(list3, XmNdefaultActionCallback,
					  &mailbox, this,
					  &NetworkEditor::list3_cb, 0,
					  &CallbackCloners::list_clone);
    list3->Create(slist3, "list3");

    iter.first();
    if(iter.ok()){
	update_list(iter.get_data());
    } else {
	current_cat=0;
	current_subcat=0;
    }

    SeparatorC sep1;
    sep1.SetOrientation(XmVERTICAL);
    sep1.Create(lists, "sep");

    RowColumnC all_rc;
    all_rc.Create(lists, "rc");
    LabelC all_label;
    all_label.Create(all_rc, "Complete List:");
    ScrolledWindowC all_slist;
    all_slist.Create(all_rc, "slist");
    ListC all_list;
    all_list.SetVisibleItemCount(LIST_VISIBLE);
    new MotifCallback<NetworkEditor>FIXCB(&all_list, XmNdefaultActionCallback,
					  &mailbox, this,
					  &NetworkEditor::list3_cb, 0,
					  &CallbackCloners::list_clone);
    all_list.Create(all_slist, "list3");
    ModuleSubCategory* all_modules=ModuleList::get_all();
    ModuleSubCategoryIter all_iter(all_modules);
    item=1;
    for(all_iter.first();all_iter.ok();++all_iter){
	clString k(all_iter.get_key());
	XmString s1=XmStringCreateSimple(k());
	XmListAddItem(all_list, s1, item++);
	XmStringFree(s1);
    }
    ScrolledWindowC stext;
    stext.SetScrollBarDisplayPolicy(XmSTATIC);
    stext.SetScrollingPolicy(XmAPPLICATION_DEFINED);
    stext.SetShadowThickness(0);
    stext.SetVisualPolicy(XmVARIABLE);
    stext.Create(pane, "stext");
    text=new TextC;
    text->SetEditable(False);
    text->SetCursorPositionVisible(False);
    text->SetWordWrap(True);
    text->SetScrollHorizontal(False);
    text->SetScrollVertical(True);
    text->SetEditMode(XmMULTI_LINE_EDIT);
    text->SetRows(2);
    text->Create(stext, "text");

    ScrolledWindowC scroller;
    scroller.SetWidth(NETEDIT_WINDOW_WIDTH);
    scroller.SetHeight(NETEDIT_WINDOW_HEIGHT);
    scroller.SetScrollingPolicy(XmAUTOMATIC);
    scroller.Create(pane, "scroller");

    drawing_a=new DrawingAreaC;
    drawing_a->SetUnitType(XmPIXELS);
    drawing_a->SetWidth(NETEDIT_CANVAS_SIZE);
    drawing_a->SetHeight(NETEDIT_CANVAS_SIZE);
    drawing_a->SetMarginHeight(NETEDIT_STAY_FROM_EDGE);
    drawing_a->SetMarginWidth(NETEDIT_STAY_FROM_EDGE);
    drawing_a->SetResizePolicy(XmRESIZE_NONE);
    drawing_a->SetBackground(background.pixel());
    new MotifCallback<NetworkEditor>FIXCB(drawing_a, XmNexposeCallback,
					  &mailbox, this,
					  &NetworkEditor::redraw, 0, 0);
    drawing_a->Create(scroller, "drawing_a");
    new MotifCallback<NetworkEditor>FIXCB(drawing_a,
					  "<Btn2Down>",
					  &mailbox, this,
					  &NetworkEditor::connection_cb, 0,
					  &CallbackCloners::event_clone);
    new MotifCallback<NetworkEditor>FIXCB(drawing_a,
					  "<Btn3Down>",
					  &mailbox, this,
					  &NetworkEditor::rightmouse, 0,
					  &CallbackCloners::event_clone);
    window->Realize();

    // Initialize the network
    net->initialize(this);
    evl->unlock();

    // Go into Main loop...
    do_scheduling();
    main_loop();
    return 0;
}

void NetworkEditor::main_loop()
{
    // Dispatch events...
    int done=0;
    while(!done){
	MessageBase* msg=mailbox.receive();
	// Dispatch message....
	int need_sched=0;
	switch(msg->type){
	case MessageTypes::DoCallback:
	    {
		Callback_Message* cmsg=(Callback_Message*)msg;
		cmsg->mcb->perform(cmsg->cbdata);
		delete cmsg->cbdata;
	    }
	    break;
	case MessageTypes::ReSchedule:
	    do_scheduling();
	    break;
	default:
	    cerr << "Unknown message type: " << msg->type << endl;
	    break;
	};
	delete msg;
    };
}

void NetworkEditor::do_scheduling()
{
    // Each Stream (Connection) has one of three properties
    // Dormant - not needed this time
    // New - Contains new data
    // Repeat - Another copy of old data
    //
    // New's are propogated downstream - never blocked
    // Repeats are propogated upstream - blocked by "sources"
    //
    // A module is considered 'New' if any of it's upstream modules are
    // receiving new data
    // A module is considered 'Repeat' if any of it's downstream modules
    // are 'Repeat'
    //
    int changed=1;
    int nmodules=net->nmodules();
    int any_changed=0;
    while(changed){
	changed=0;
	int nmodules=net->nmodules();
	for(int i=0;i<nmodules;i++){
	    Module* mod=net->module(i);
	    changed |= mod->should_execute();
	}
	any_changed|=changed;
    }
    if(first_schedule || any_changed){
	// Do the scheduling...
	for(int i=0;i<nmodules;i++){
	    Module* module=net->module(i);

	    // Tell it to trigger...
	    if(module->sched_state != Module::SchedDormant){
		module->mailbox.send(new Scheduler_Module_Message);

		// Reset the state...
		module->sched_state=Module::SchedDormant;
	    }
	}
	first_schedule=0;
    }
}

Scheduler_Module_Message::Scheduler_Module_Message()
: MessageBase(MessageTypes::ExecuteModule)
{
}

Scheduler_Module_Message::~Scheduler_Module_Message()
{
}

Module_Scheduler_Message::Module_Scheduler_Message()
: MessageBase(MessageTypes::ReSchedule)
{
}

Module_Scheduler_Message::~Module_Scheduler_Message()
{
}

void NetworkEditor::popup_memstats(CallbackData*, void*)
{
    if(!memstats)
	memstats=new MemStats(this);
    else
	memstats->popup();
}

void NetworkEditor::list1_cb(CallbackData* cbdata, void*)
{
    ModuleCategory* cat;
    if(!current_db->lookup(cbdata->get_string(), cat)){
	cerr << "Category list inconsistency!!!\n";
	return;
    }
    update_list(cat);
}

void NetworkEditor::list2_cb(CallbackData* cbdata, void*)
{
    ModuleSubCategory* subcat;
    if(!current_cat->lookup(cbdata->get_string(), subcat)){
	cerr << "SubCategory list inconsistency!!!\n";
	return;
    }
    update_list(subcat);
}

void NetworkEditor::list3_cb(CallbackData* cbdata, void*)
{
    net->add_module(cbdata->get_string());
}

void NetworkEditor::update_list(ModuleCategory* cat)
{
    current_cat=cat;
    ModuleCategoryIter iter(cat);
    int item=1;
    evl->lock();
    XmListDeleteAllItems(*list2);
    for(iter.first();iter.ok();++iter){
	clString k(iter.get_key());
	XmString s1=XmStringCreateSimple(k());
	XmListAddItem(*list2, s1, item++);
	XmStringFree(s1);
    }
    iter.first();
    if(iter.ok())
	update_list(iter.get_data());
    else
	current_subcat=0;
    evl->unlock();
}

void NetworkEditor::update_list(ModuleSubCategory* subcat)
{
    current_subcat=subcat;
    ModuleSubCategoryIter iter(subcat);
    int item=1;
    evl->lock();
    XmListDeleteAllItems(*list3);
    for(iter.first();iter.ok();++iter){
	clString k(iter.get_key());
	XmString s1=XmStringCreateSimple(k());
	XmListAddItem(*list3, s1, item++);
	XmStringFree(s1);
    }
    evl->unlock();
}

void NetworkEditor::connection_cb(CallbackData* cbdata, void* vmodule)
{
    Module* module=(Module*)vmodule;
    XEvent* event=cbdata->get_event();
    int x=event->xbutton.x;
    int y=event->xbutton.y;
    if(module){
	x+=module->xpos;
	y+=module->ypos;
    }
    // Find the closest port...
    Module* cmodule=0;
    int which_port=0;
    int oport=0;
    
    if(making_connection){
	// Complete it...
	conn_in_progress->connect();
	net->connect(conn_in_progress);
	if(from_oport){
	    from_module->reconfigure_oports();
	    to_module->reconfigure_iports();
	} else {
	    from_module->reconfigure_iports();
	    to_module->reconfigure_oports();
	}
	conn_in_progress=0;
	check_cancel();
	add_text(clString("Connection made from ")
		 +from_module->name+" (port "+to_string(from_which)+") to "
		 +to_module->name+" (port "+to_string(to_which)+")");
	do_scheduling();
    } else {
	// Check the callback module first..
	if(module){
	    closeness(module, x, y, cmodule, which_port, oport);
	}
	if(!cmodule){
	    int n=net->nmodules();
	    for(int i=0;i<n;i++){
		Module* m=net->module(i);
		closeness(m, x, y, cmodule, which_port, oport);
		if(cmodule)
		    break;
	    }	
	}
	if(cmodule){
	    Port* port=oport?cmodule->oport(which_port):cmodule->iport(which_port);
	    if(port->nconnections() == 0){
		// Connect...
		from_module=cmodule;
		from_which=which_port;
		from_oport=oport;
		add_text(clString("Making connection from ")
			 +from_module->name+" (port "+to_string(from_which)
			 +") : "+port->get_portname()+" ("
			 +port->get_typename()
			 +") - hit right mouse button to cancel");
		update_to(x, y);
		if(to_module){
		    if(from_oport){
			conn_in_progress=new Connection(from_module, from_which,
							to_module, to_which);
		    } else {
			conn_in_progress=new Connection(to_module, to_which,
							from_module, from_which);
		    }
		    conn_in_progress->set_context(this);
		    making_connection=1;
		    draw_temp_portlines();
		} else {
		    add_text(clString("Cannot connect ")+from_module->name
			     +" (port "+to_string(from_which)
			     +") - no match found");
		}
	    } else {
		// Disconnect
		NOT_FINISHED("disconnect...\n");
	    }
	} else {
	    add_text("No port found");
	}
    }
}

void NetworkEditor::closeness(Module* mod, int xm, int ym, Module*& ret_mod,
			      int& which, int& output)
{
    int n=mod->niports();
    for(int i=0;i<n;i++){
	int x,y;
	mod->get_iport_coords(i, x, y);
	int dist=Abs(x-xm)+Abs(y-ym);
	if(dist < 10){
	    // Found it...
	    ret_mod=mod;
	    which=i;
	    output=0;
	    return;
	}
    }
    n=mod->noports();
    for(i=0;i<n;i++){
	int x,y;
	mod->get_oport_coords(i, x, y);
	int dist=Abs(x-xm)+Abs(y-ym);
	if(dist < 10){
	    // Found it...
	    ret_mod=mod;
	    which=i;
	    output=1;
	    return;
	}
    }
}

int NetworkEditor::check_cancel()
{
    if(making_connection){
	if(conn_in_progress){
	    delete conn_in_progress;
	    add_text("Connection cancelled");
	}
	making_connection=0;
	return 1;
    } else {
	return 0;
    }
}

void NetworkEditor::update_to(int x, int y)
{
    // Find the closest port to x,y...
    to_module=0;
    int nmodules=net->nmodules();
    int mindist=100000;
    if(from_oport){
	// Look for a matching iport...
	OPort* oport=from_module->oport(from_which);
	clString oport_name(oport->get_typename());
	for(int i=0;i<nmodules;i++){
	    Module* mod=net->module(i);
	    for(int i=0;i<mod->niports();i++){
		IPort* iport=mod->iport(i);
		if(iport->nconnections() == 0
		   && iport->get_typename() == oport_name){
		    // Maybe...
		    int px, py;
		    mod->get_iport_coords(i, px, py);
		    int dist=Abs(x-px)+Abs(y-py);
		    if(dist < mindist){
			to_module=mod;
			to_which=i;
			mindist=dist;
		    }
		}
	    }
	}
    } else {
	// Look for a matching oport...
	IPort* iport=from_module->iport(from_which);
	clString iport_name(iport->get_typename());
	for(int i=0;i<nmodules;i++){
	    Module* mod=net->module(i);
	    for(int i=0;i<mod->noports();i++){
		OPort* oport=mod->oport(i);
		if(oport->nconnections() == 0
		   && oport->get_typename() == iport_name){
		    // Maybe...
		    int px, py;
		    mod->get_oport_coords(i, px, py);
		    int dist=Abs(x-px)+Abs(y-py);
		    if(dist < mindist){
			to_module=mod;
			to_which=i;
			mindist=dist;
		    }
		}
	    }
	}
    }
}

void NetworkEditor::add_text(const clString& str)
{
    evl->lock();
    clString ins(clString("\n")+str);
    XmTextInsert(*text, XmTextGetLastPosition(*text), ins());
    evl->unlock();
}

void NetworkEditor::redraw(CallbackData*, void*)
{
    if(making_connection)
	draw_temp_portlines();
}

void NetworkEditor::draw_temp_portlines()
{
    NOT_FINISHED("NetworkEditor::draw_temp_portlines");
}

void NetworkEditor::rightmouse(CallbackData*, void*)
{
    if(check_cancel())
	return;
}

void NetworkEditor::quit(CallbackData*, void*)
{
    TaskManager::exit_all(0);
}
