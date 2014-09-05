/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

/*
 *  Viewer.cc:  The Geometry Viewer
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Dataflow/Modules/Render/Viewer.h>
#include <Dataflow/Modules/Render/Renderer.h>
#include <Dataflow/Modules/Render/ViewWindow.h>
#include <Dataflow/Comm/MessageTypes.h>
#include <Dataflow/Network/Connection.h>
#include <Dataflow/Network/ModuleHelper.h>
#include <Core/Geom/GeomObj.h>
#include <Dataflow/Modules/Render/ViewGeom.h>
#include <Core/Geom/HeadLight.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Thread/FutureValue.h>

#include <iostream>
using std::cerr;
using std::endl;
using std::ostream;

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

namespace SCIRun {



//----------------------------------------------------------------------
extern "C" Module* make_Viewer(const clString& id) {
  return new Viewer(id);
}

//----------------------------------------------------------------------
Viewer::Viewer(const clString& id)
  : Module("Viewer", id, ViewerSpecial,"Render","SCIRun"), max_portno(0), geomlock("Viewer geometry lock")
{
				// Add a headlight
    lighting.lights.add(scinew HeadLight("Headlight", Color(1,1,1)));
    
				// Create the input port
    //add_iport(scinew GeometryIPort(this, "Geometry", GeometryIPort::Atomic));
    //add_iport(scinew GeometryIPort(this, "0", GeometryIPort::Atomic));
    default_matl=scinew Material(Color(.1,.1,.1), Color(.6,0,0),
			      Color(.7,.7,.7), 50);
    busy_bit=1;
    have_own_dispatch=1;

				// Create port 0 - we use this for
				// global objects such as cameras,
				// light source icons, etc.
    int portid=max_portno++;
    
				// Create the port
    GeomViewerPort *pi = new GeomViewerPort(portid);

#if 0
    PortInfo* pi=scinew PortInfo;
    
    portHash[portid] = pi;
    
    pi->msg_head=pi->msg_tail=0;
    pi->portno=portid;
    
    pi->objs = scinew MapIntSceneItem;
    
#endif

    ports.addObj(pi,portid);
}

//----------------------------------------------------------------------
Viewer::Viewer(const clString& id, const clString& moduleName):
  Module(moduleName, id, ViewerSpecial,"Render","SCIRun"), max_portno(0),
  geomlock((moduleName + clString(" geometry lock"))())
{

				// Add a headlight
    lighting.lights.add(scinew HeadLight("Headlight", Color(1,1,1)));
    
				// Create the input port
    //add_iport(scinew GeometryIPort(this, "Geometry", GeometryIPort::Atomic));
    //add_iport(scinew GeometryIPort(this, "0", GeometryIPort::Atomic));
    default_matl=scinew Material(Color(.1,.1,.1), Color(.6,0,0),
			      Color(.7,.7,.7), 50);
    busy_bit=1;
    have_own_dispatch=1;

				// Create port 0 - we use this for
				// global objects such as cameras,
				// light source icons, etc.
    int portid=max_portno++;
    
				// Create the port
    GeomViewerPort *pi = new GeomViewerPort(portid);

#if 0
    PortInfo* pi=scinew PortInfo;
    
    portHash[portid] = pi;
    
    pi->msg_head=pi->msg_tail=0;
    pi->portno=portid;
    
    pi->objs = scinew MapIntSceneItem;
    
#endif

    ports.addObj(pi,portid);
}

//----------------------------------------------------------------------
Viewer::~Viewer()
{
}

//----------------------------------------------------------------------
void Viewer::do_execute()
{
    for(;;){
	if(mailbox.numItems() == 0){
	    // See if anything needs to be redrawn...
	    int did_some=1;
	    while(did_some){
		did_some=0;
		for(int i=0;i<viewwindow.size();i++){
		    did_some+=viewwindow[i]->need_redraw;
		    viewwindow[i]->redraw_if_needed();
		}
	    }
	}
	process_event(1);
    }
}

//----------------------------------------------------------------------
int Viewer::process_event(int block)
{
    int ni=mailbox.numItems();
    if(!block && ni==0)return 0;
    if(!ni)busy_bit=0;
    MessageBase* msg=mailbox.receive();
    busy_bit=1;
    GeometryComm* gmsg=(GeometryComm*)msg;
    switch(msg->type){
    case MessageTypes::ExecuteModule:
	// We ignore these messages...
	break;
    case MessageTypes::ViewWindowRedraw:
	{
	    ViewerMessage* rmsg=(ViewerMessage*)msg;
	    ViewWindow* r=0;
	    int i;
	    for(i=0;i<viewwindow.size();i++){
		r=viewwindow[i];
		if(r->id == rmsg->rid)
		    break;
	    }
	    if(i==viewwindow.size()){
		cerr << "Warning: ViewWindow not found for redraw! (id=" << rmsg->rid << "\n";
	    } else if(rmsg->nframes == 0){
		// Normal redraw (lazy)
		r->need_redraw=1;
	    } else {
		// Do animation...
		r->redraw(rmsg->tbeg, rmsg->tend, rmsg->nframes,
			  rmsg->framerate);
	    }
	}
	break;
    case MessageTypes::ViewWindowDumpImage:
	{
	    ViewerMessage* rmsg=(ViewerMessage*)msg;
	    for(int i=0;i<viewwindow.size();i++){
		ViewWindow* r=viewwindow[i];
		if(r->id == rmsg->rid){
                 r->current_renderer->saveImage(rmsg->filename, rmsg->format);
		    break;
		}
	    }
	}
	break;
    case MessageTypes::ViewWindowDumpObjects:
	{
	    geomlock.readLock();
	    ViewerMessage* rmsg=(ViewerMessage*)msg;
	    for(int i=0;i<viewwindow.size();i++){
		ViewWindow* r=viewwindow[i];
		if(r->id == rmsg->rid){
		    r->dump_objects(rmsg->filename, rmsg->format);
		    break;
		}
	    }
	    geomlock.readUnlock();
	}
	break;
    case MessageTypes::ViewWindowMouse:
	{
	    ViewWindowMouseMessage* rmsg=(ViewWindowMouseMessage*)msg;
	    for(int i=0;i<viewwindow.size();i++){
		ViewWindow* r=viewwindow[i];
		if(r->id == rmsg->rid){
		    (r->*(rmsg->handler))(rmsg->action, rmsg->x, rmsg->y, rmsg->state, rmsg->btn, rmsg->time);
		    break;
		}
	    }
	}
	break;
    case MessageTypes::GeometryInit:
	geomlock.writeLock();
	initPort(gmsg->reply);
	geomlock.writeUnlock();
	break;	
    case MessageTypes::GeometryAddObj:
    case MessageTypes::GeometryDelObj:
    case MessageTypes::GeometryDelAll:
	append_port_msg(gmsg);
	msg=0; // Don't delete it yet...
	break;
    case MessageTypes::GeometryFlush:
	geomlock.writeLock();
	flushPort(gmsg->portno);
	geomlock.writeUnlock();
	break;
    case MessageTypes::GeometryFlushViews:
	geomlock.writeLock();
	flushPort(gmsg->portno);
	geomlock.writeUnlock();
	flushViews();
	if(gmsg->wait){
	    // Synchronized redraw - do it now and signal them...
	    for(int i=0;i<viewwindow.size();i++)
		viewwindow[i]->redraw_if_needed();
	    gmsg->wait->up();
	}
	break;
    case MessageTypes::GeometryGetNViewWindows:
	gmsg->nreply->send(viewwindow.size());
	break;
    case MessageTypes::GeometryGetData:
	if(gmsg->which_viewwindow >= viewwindow.size()){
	    gmsg->datareply->send(0);
	} else {
	    cerr << "Calling viewwindow->getData\n";
	    viewwindow[gmsg->which_viewwindow]->getData(gmsg->datamask, gmsg->datareply);
	    cerr << "getDat done\n";
	}
	break;
    case MessageTypes::GeometrySetView:
	if(gmsg->which_viewwindow < viewwindow.size()){
	    cerr << "Calling viewwindow->setView\n";
	    viewwindow[gmsg->which_viewwindow]->setView(gmsg->view);
	    cerr << "setView done\n";
	}
	break;
#if 0
    case MessageTypes::TrackerMoved:
	{
	    TrackerMessage* tmsg=(TrackerMessage*)msg;
	    ViewWindow* viewwindow=(ViewWindow*)tmsg->clientdata;
	    if(tmsg->data.head_moved)
		viewwindow->head_moved(tmsg->data.head_pos);
	    if(tmsg->data.mouse_moved)
		viewwindow->flyingmouse_moved(tmsg->data.mouse_pos);
	}
	break;
#endif
    default:
	cerr << "Viewer: Illegal Message type: " << msg->type << endl;
	break;
    }
    if(msg)
	delete msg;
    return 1;
}

//----------------------------------------------------------------------
void Viewer::initPort(Mailbox<GeomReply>* reply)
{
    int portid=max_portno++;
				// Create the port
    // PortInfo* pi=scinew PortInfo;
    GeomViewerPort *pi = new GeomViewerPort(portid);
    ports.addObj(pi,portid);
#if 0
    portHash[portid] = pi;
    pi->msg_head=pi->msg_tail=0;
    pi->portno=portid;
    //pi->objs=scinew MapIntSceneItem;
#endif
    reply->send(GeomReply(portid, &busy_bit));
}

//----------------------------------------------------------------------
void Viewer::flushViews()
{
  for (int i=0; i<viewwindow.size(); i++)
    viewwindow[i]->force_redraw();
}

//----------------------------------------------------------------------
void Viewer::addObj(GeomViewerPort* port, int serial, GeomObj *obj,
		    const clString& name, CrowdMonitor* lock)
{
    clString pname(name+" ("+to_string(port->portno)+")");
    // SceneItem* si=scinew SceneItem(obj, pname, lock);
    GeomViewerItem* si = scinew GeomViewerItem(obj, pname, lock);
    port->addObj(si,serial);
    // port->objs->insert(serial, si);
    for (int i=0; i<viewwindow.size(); i++) {
	viewwindow[i]->itemAdded(si);
    }
}

//----------------------------------------------------------------------
void Viewer::delObj(GeomViewerPort* port, int serial, int del)
{
    GeomViewerItem* si;
    if((si = ((GeomViewerItem*)port->getObj(serial)))){
	for (int i=0; i<viewwindow.size(); i++)
	    viewwindow[i]->itemDeleted(si);
	port->delObj(serial, del);
    } else {
	cerr << "Error deleting object, object not in database...(serial=" << serial << ")" << endl;
    }
}

//----------------------------------------------------------------------
void Viewer::delAll(GeomViewerPort* port)
{
  GeomIndexedGroup::IterIntGeomObj iter = port->getIter();
  
  for ( ; iter.first != iter.second; iter.first++) {
    GeomViewerItem* si =
      (GeomViewerItem*)((*iter.first).second);
    for (int i=0; i<viewwindow.size(); i++)
      viewwindow[i]->itemDeleted(si);
  }
  
  port->delAll();
}

void Viewer::addTopViewWindow(ViewWindow *r)
{
    topViewWindow.add(r);
}

void Viewer::delTopViewWindow(ViewWindow *r)
{
    for (int i=0; i<topViewWindow.size(); i++) {
	if (r==topViewWindow[i]) topViewWindow.remove(i);
    }
} 

//----------------------------------------------------------------------
#ifdef OLDUI
void Viewer::spawnIndCB(CallbackData*, void*)
{
  topViewWindow.add(scinew ViewWindow(this));
  topViewWindow[topViewWindow.size()-1]->SetTop();
  GeomItem *item;
  for (int i=0; i<topViewWindow[0]->geomItemA.size(); i++) {
      item=topViewWindow[0]->geomItemA[i];
      topViewWindow[topViewWindow.size()-1]->itemAdded(item->geom, item->name);
  }
//  printFamilyTree();
}
#endif

//----------------------------------------------------------------------
//void Viewer::connection(ConnectionMode mode, int which_port, int)
//{
//    if(mode==Disconnected){
//	remove_iport(which_port);
//    } else {
//	add_iport(scinew GeometryIPort(this, "Geometry", GeometryIPort::Atomic));
//    }
//}
//----------------------------------------------------------------------
void Viewer::delete_viewwindow(ViewWindow* delviewwindow)
{
  for(int i=0;i<viewwindow.size();i++){
    if(viewwindow[i] == delviewwindow){
      viewwindow.remove(i);
      delete delviewwindow;
    }
  }
}

//----------------------------------------------------------------------
void Viewer::tcl_command(TCLArgs& args, void* userdata)
{
    if(args.count() < 2){
	args.error("Viewer needs a minor command");
	return;
    }
    if(args[1] == "addviewwindow"){
	if(args.count() != 3){
	    args.error("addviewwindow must have a RID");
	    return;
	}
	ViewWindow* r=scinew ViewWindow(this, args[2]);
	viewwindow.add(r);
    } else if(args[1] == "listrenderers"){
	Array1<clString> list;
	AVLTreeIter<clString, RegisterRenderer*> iter(Renderer::get_db());
	for(iter.first();iter.ok();++iter){
	    list.add(iter.get_key());
	}
	args.result(args.make_list(list));
    } else {
	Module::tcl_command(args, userdata);
    }
}

//----------------------------------------------------------------------
void Viewer::execute()
{
    // Never gets called...
}

//----------------------------------------------------------------------
ViewerMessage::ViewerMessage(const clString& rid)
: MessageBase(MessageTypes::ViewWindowRedraw), rid(rid), nframes(0)
{
}

//----------------------------------------------------------------------
ViewerMessage::ViewerMessage(const clString& rid, double tbeg, double tend,
			     int nframes, double framerate)
: MessageBase(MessageTypes::ViewWindowRedraw), rid(rid), tbeg(tbeg), tend(tend),
  nframes(nframes), framerate(framerate)
{
    if(nframes == 0)
	cerr << "Warning - nframes shouldn't be zero for animation\n";
}

//----------------------------------------------------------------------
ViewerMessage::ViewerMessage(MessageTypes::MessageType type,
			     const clString& rid, const clString& filename)
: MessageBase(type), rid(rid), filename(filename)
{
}

//----------------------------------------------------------------------
ViewerMessage::ViewerMessage(MessageTypes::MessageType type,
			     const clString& rid, const clString& filename,
			     const clString& format)
: MessageBase(type), rid(rid), filename(filename), format(format)
{
}

//----------------------------------------------------------------------
ViewerMessage::~ViewerMessage()
{
}

//----------------------------------------------------------------------
#if 0
SceneItem::SceneItem(GeomObj* obj, const clString& name, CrowdMonitor* lock)
: obj(obj), name(name), lock(lock)
{
}

//----------------------------------------------------------------------
SceneItem::~SceneItem()
{
}
#endif

//----------------------------------------------------------------------
void Viewer::append_port_msg(GeometryComm* gmsg)
{
				// Look up the right port number
  // PortInfo* pi;
  GeomViewerPort *pi;
  
  if (!(pi = ((GeomViewerPort*)ports.getObj(gmsg->portno)))) {
    // if(portHash.find(gmsg->portno) == portHash.end()){
    cerr << "Geometry message sent to bad port!!!: " << gmsg->portno << "\n";
    return;
  }
  
				// Queue up the messages until the
				// flush...
  if(pi->msg_tail){
    pi->msg_tail->next=gmsg;
    pi->msg_tail=gmsg;
  } else {
    pi->msg_head=pi->msg_tail=gmsg;
  }
  gmsg->next=0;
}

//----------------------------------------------------------------------
void Viewer::flushPort(int portid)
{
    // Look up the right port number
    GeomViewerPort* pi;
    if(!(pi = ((GeomViewerPort*)ports.getObj(portid)))){
	cerr << "Geometry message sent to bad port!!!\n";
	return;
    }
    GeometryComm* gmsg=pi->msg_head;
    while(gmsg){
	switch(gmsg->type){
	case MessageTypes::GeometryAddObj:
	    addObj(pi, gmsg->serial, gmsg->obj, gmsg->name, gmsg->lock);
	    break;
	case MessageTypes::GeometryDelObj:
	    delObj(pi, gmsg->serial, gmsg->del);
	    break;
	case MessageTypes::GeometryDelAll:
	    delAll(pi);
	    break;
	default:
	    cerr << "How did this message get in here???\n";
	}
	GeometryComm* next=gmsg->next;
	delete gmsg;
	gmsg=next;
    }
    if(pi->msg_head){
	flushViews();
	pi->msg_head=pi->msg_tail=0;
    }
}

//----------------------------------------------------------------------
int Viewer::lookup_specific(const clString& key, void*& data)
{
    //return specific.lookup(key, data);
    
    MapClStringVoid::iterator result = specific.find(key);
    if (result != specific.end()) {
	data = (*result).second;
	return 1;
    }
    return 0;
}

//----------------------------------------------------------------------
void Viewer::insert_specific(const clString& key, void* data)
{
    //specific.insert(key, data);
    specific[key] = data;
}

//----------------------------------------------------------------------
void Viewer::emit_vars(ostream& out, clString& midx)
{
//  cerr << "Viewer emitvars" << endl;
  TCL::emit_vars(out, midx);
  clString viewwindowstr;
  for(int i=0;i<viewwindow.size();i++){
    viewwindowstr=midx+clString("-ViewWindow_")+to_string(i);
    out << midx << " ui\n";
    viewwindow[i]->emit_vars(out, viewwindowstr);
  }
}

} // End namespace SCIRun
