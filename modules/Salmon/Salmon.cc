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
#include <MessageTypes.h>
#include <ModuleHelper.h>
#include <ModuleList.h>
#include <MotifCallback.h>
#include <MtXEventLoop.h>
#include <NetworkEditor.h>
#include <NotFinished.h>
#include <XQColor.h>
#include <Mt/DrawingArea.h>
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
: Module("Salmon", Sink), max_portno(0)
{
    // Create the input port
    iports.add(new GeometryIPort(this, "Geometry", GeometryIPort::Atomic));
    add_iport(iports[0]);

}

Salmon::~Salmon()
{
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
    bgcolor=new XQColor(netedit->color_manager, "salmon");
    drawing_a=new DrawingAreaC;
    drawing_a->SetUnitType(XmPIXELS);
    drawing_a->SetX(xpos);
    drawing_a->SetY(ypos);
    drawing_a->SetWidth(100);
    drawing_a->SetHeight(100);
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
    evl->unlock();

    // Create the viewer window...
    topRoe.add(new Roe(this));
    topRoe[topRoe.size()-1]->SetTop();
    
//    printFamilyTree();

    // Start up the event loop thread...
    helper=new ModuleHelper(this, 1);
    helper->activate(0);
}

void Salmon::redraw_widget(CallbackData*, void*)
{
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
//  printFamilyTree();
}

Salmon::Salmon(const Salmon& copy, int deep)
: Module(copy, deep)
{
    NOT_FINISHED("Salmon::Salmon");
}

void Salmon::reconfigure_iports()
{
    NOT_FINISHED("Salmon::reconfigure_iports");
}

void Salmon::reconfigure_oports()
{
    NOT_FINISHED("Salmon::reconfigure_oports");
}

