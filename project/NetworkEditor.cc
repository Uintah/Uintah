
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
#include <MemStats.h>
#include <MessageBase.h>
#include <Module.h>
#include <MotifCallbackBase.h>
#include <MotifCallback.h>
#include <CallbackCloners.h>
#include <MtXEventLoop.h>
#include <Network.h>
#include <XQColor.h>
#include <Mt/ApplicationShell.h>
#include <Mt/CascadeButton.h>
#include <Mt/DrawingArea.h>
#include <Mt/MainWindow.h>
#include <MenuBar.h>
#include <Mt/PushButton.h>
#include <Mt/RowColumn.h>
#include <Mt/ScrolledWindow.h>

extern MtXEventLoop* evl;

// Someday, we should make these resources...
#define NETEDIT_WINDOW_WIDTH 800
#define NETEDIT_WINDOW_HEIGHT 600
#define NETEDIT_CANVAS_SIZE 2000
#define NETEDIT_STAY_FROM_EDGE 2
#define NETEDIT_BACKGROUND_COLOR "#224488"
#define MODULE_NAME_FONT "-*-lucida-bold-r-*-*-14-*-*-*-*-*-*-*"
#define MODULE_TIME_FONT "-*-lucida-medium-r-*-*-11-*-*-*-*-*-*-*"

NetworkEditor::NetworkEditor(Network* net, Display* display,
			     ColorManager* color_manager)
: Task("Network Editor", 1), net(net), display(display),
  color_manager(color_manager), mailbox(10), first_schedule(1),
  memstats(0)
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
    MenuC* stats=menubar.AddMenu("Statistics");
    PushButtonC* membutton=stats->AddButton("Memory");
    new MotifCallback<NetworkEditor>FIXCB(membutton, XmNactivateCallback,
					  &mailbox, this,
					  &NetworkEditor::popup_memstats,
					  0, 0);

    ScrolledWindowC scroller;
    scroller.SetWidth(NETEDIT_WINDOW_WIDTH);
    scroller.SetHeight(NETEDIT_WINDOW_HEIGHT);
    scroller.SetScrollingPolicy(XmAUTOMATIC);
    scroller.Create(mainw, "scroller");

    drawing_a=new DrawingAreaC;
    drawing_a->SetUnitType(XmPIXELS);
    drawing_a->SetWidth(NETEDIT_CANVAS_SIZE);
    drawing_a->SetHeight(NETEDIT_CANVAS_SIZE);
    drawing_a->SetMarginHeight(NETEDIT_STAY_FROM_EDGE);
    drawing_a->SetMarginWidth(NETEDIT_STAY_FROM_EDGE);
    drawing_a->SetResizePolicy(XmRESIZE_NONE);
    drawing_a->SetBackground(background.pixel());
    drawing_a->Create(scroller, "drawing_a");
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
    cerr << "Scheduler started...\n";
    int changed=1;
    int nmodules=net->nmodules();
    int any_changed=0;
    while(changed){
	changed=0;
	int nmodules=net->nmodules();
	for(int i=0;i<nmodules;i++){
	    Module* mod=net->module(i);
	    changed |= mod->should_execute();
	    cerr << "Module (" << mod << ") " << i << " state=" << mod->sched_state << endl;
	}
	any_changed|=changed;
    }
    if(first_schedule || any_changed){
	// Do the scheduling...
	cerr << "Executing Modules...\n";
	for(int i=0;i<nmodules;i++){
	    Module* module=net->module(i);

	    // Tell it to trigger...
	    module->mailbox.send(new Scheduler_Module_Message);

	    // Reset the state...
	    module->sched_state=Module::SchedDormant;
	}
	first_schedule=0;
    } else {
	cerr << "Scheduler decided not to execute\n";
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
