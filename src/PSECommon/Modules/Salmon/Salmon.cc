//static char *id="@(#) $Id$";

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

#include <PSECommon/Modules/Salmon/Salmon.h>
#include <PSECommon/Modules/Salmon/Renderer.h>
#include <PSECommon/Modules/Salmon/Roe.h>
#include <SCICore/Containers/HashTable.h>
#include <PSECore/Comm/MessageTypes.h>
#include <PSECore/Dataflow/Connection.h>
#include <PSECore/Dataflow/ModuleHelper.h>
#include <SCICore/Geom/GeomObj.h>
#include <PSECommon/Modules/Salmon/SalmonGeom.h>
#include <SCICore/Geom/HeadLight.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/Multitask/AsyncReply.h>

#include <iostream.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

namespace PSECommon {
namespace Modules {

using PSECore::Dataflow::Module;
using PSECore::Datatypes::GeometryIPort;
using PSECore::Datatypes::GeometryComm;

using SCICore::GeomSpace::HeadLight;
using SCICore::Containers::to_string;
using SCICore::Containers::HashTableIter;
using SCICore::Containers::AVLTreeIter;

Module* make_Salmon(const clString& id) {
  return new Salmon(id);
}

Salmon::Salmon(const clString& id)
: Module("Salmon", id, SalmonSpecial), max_portno(0)
{
    // Add a headlight
    lighting.lights.add(scinew HeadLight("Headlight", Color(1,1,1)));
    // Create the input port
    add_iport(scinew GeometryIPort(this, "Geometry", GeometryIPort::Atomic));
    default_matl=scinew Material(Color(.1,.1,.1), Color(.6,0,0),
			      Color(.7,.7,.7), 50);
    busy_bit=1;
    have_own_dispatch=1;

    // Create port 0 - we use this for global objects such as cameras,
    // light source icons, etc.
    int portid=max_portno++;
    // Create the port
    GeomSalmonPort *pi = new GeomSalmonPort(portid);

#if 0
    PortInfo* pi=scinew PortInfo;
    portHash.insert(portid, pi);
    pi->msg_head=pi->msg_tail=0;
    pi->portno=portid;
    pi->objs=scinew HashTable<int, SceneItem*>;
#endif

    ports.addObj(pi,portid);
}

Salmon::~Salmon()
{
}

void Salmon::do_execute()
{
    for(;;){
	if(mailbox.nitems() == 0){
	    // See if anything needs to be redrawn...
	    int did_some=1;
	    while(did_some){
		did_some=0;
		for(int i=0;i<roe.size();i++){
		    did_some+=roe[i]->need_redraw;
		    roe[i]->redraw_if_needed();
		}
	    }
	}
	process_event(1);
    }
}

int Salmon::process_event(int block)
{
    int ni=mailbox.nitems();
    if(!block && ni==0)return 0;
    if(!ni)busy_bit=0;
    MessageBase* msg=mailbox.receive();
    busy_bit=1;
    GeometryComm* gmsg=(GeometryComm*)msg;
    switch(msg->type){
    case MessageTypes::ExecuteModule:
	// We ignore these messages...
	break;
    case MessageTypes::RoeRedraw:
	{
	    SalmonMessage* rmsg=(SalmonMessage*)msg;
	    Roe* r=0;
	    int i;
	    for(i=0;i<roe.size();i++){
		r=roe[i];
		if(r->id == rmsg->rid)
		    break;
	    }
	    if(i==roe.size()){
		cerr << "Warning: Roe not found for redraw! (id=" << rmsg->rid << "\n";
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
    case MessageTypes::RoeDumpImage:
	{
	    SalmonMessage* rmsg=(SalmonMessage*)msg;
	    for(int i=0;i<roe.size();i++){
		Roe* r=roe[i];
		if(r->id == rmsg->rid){
		    r->current_renderer->dump_image(rmsg->filename);
		    break;
		}
	    }
	}
	break;
    case MessageTypes::RoeDumpObjects:
	{
	    geomlock.read_lock();
	    SalmonMessage* rmsg=(SalmonMessage*)msg;
	    for(int i=0;i<roe.size();i++){
		Roe* r=roe[i];
		if(r->id == rmsg->rid){
		    r->dump_objects(rmsg->filename, rmsg->format);
		    break;
		}
	    }
	    geomlock.read_unlock();
	}
	break;
    case MessageTypes::RoeMouse:
	{
	    RoeMouseMessage* rmsg=(RoeMouseMessage*)msg;
	    for(int i=0;i<roe.size();i++){
		Roe* r=roe[i];
		if(r->id == rmsg->rid){
		    (r->*(rmsg->handler))(rmsg->action, rmsg->x, rmsg->y, rmsg->state, rmsg->btn, rmsg->time);
		    break;
		}
	    }
	}
	break;
    case MessageTypes::GeometryInit:
	geomlock.write_lock();
	initPort(gmsg->reply);
	geomlock.write_unlock();
	break;	
    case MessageTypes::GeometryAddObj:
    case MessageTypes::GeometryDelObj:
    case MessageTypes::GeometryDelAll:
	append_port_msg(gmsg);
	msg=0; // Don't delete it yet...
	break;
    case MessageTypes::GeometryFlush:
	geomlock.write_lock();
	flushPort(gmsg->portno);
	geomlock.write_unlock();
	break;
    case MessageTypes::GeometryFlushViews:
	geomlock.write_lock();
	flushPort(gmsg->portno);
	geomlock.write_unlock();
	flushViews();
	if(gmsg->wait){
	    // Synchronized redraw - do it now and signal them...
	    for(int i=0;i<roe.size();i++)
		roe[i]->redraw_if_needed();
	    gmsg->wait->up();
	}
	break;
    case MessageTypes::GeometryGetNRoe:
	gmsg->nreply->reply(roe.size());
	break;
    case MessageTypes::GeometryGetData:
	if(gmsg->which_roe >= roe.size()){
	    gmsg->datareply->reply(0);
	} else {
	    cerr << "Calling roe->getData\n";
	    roe[gmsg->which_roe]->getData(gmsg->datamask, gmsg->datareply);
	    cerr << "getDat done\n";
	}
	break;
#if 0
    case MessageTypes::TrackerMoved:
	{
	    TrackerMessage* tmsg=(TrackerMessage*)msg;
	    Roe* roe=(Roe*)tmsg->clientdata;
	    if(tmsg->data.head_moved)
		roe->head_moved(tmsg->data.head_pos);
	    if(tmsg->data.mouse_moved)
		roe->flyingmouse_moved(tmsg->data.mouse_pos);
	}
	break;
#endif
    default:
	cerr << "Salmon: Illegal Message type: " << msg->type << endl;
	break;
    }
    if(msg)
	delete msg;
    return 1;
}

void Salmon::initPort(Mailbox<GeomReply>* reply)
{
    int portid=max_portno++;
    // Create the port
//    PortInfo* pi=scinew PortInfo;
    GeomSalmonPort *pi = new GeomSalmonPort(portid);
    ports.addObj(pi,portid);
#if 0
    portHash.insert(portid, pi);
    pi->msg_head=pi->msg_tail=0;
    pi->portno=portid;
    pi->objs=scinew HashTable<int, SceneItem*>;
#endif
    reply->send(GeomReply(portid, &busy_bit));
}

void Salmon::flushViews()
{
    for (int i=0; i<roe.size(); i++)
	roe[i]->force_redraw();
}

void Salmon::addObj(GeomSalmonPort* port, int serial, GeomObj *obj,
		    const clString& name, CrowdMonitor* lock)
{
    clString pname(name+" ("+to_string(port->portno)+")");
//    SceneItem* si=scinew SceneItem(obj, pname, lock);
    GeomSalmonItem* si = scinew GeomSalmonItem(obj, pname, lock);
    port->addObj(si,serial);
//    port->objs->insert(serial, si);
    for (int i=0; i<roe.size(); i++)
	roe[i]->itemAdded(si);
}

void Salmon::delObj(GeomSalmonPort* port, int serial, int del)
{
    GeomSalmonItem* si;
    if(si = ((GeomSalmonItem*)port->getObj(serial))){
	for (int i=0; i<roe.size(); i++)
	    roe[i]->itemDeleted(si);
	port->delObj(serial, del);
    } else {
	cerr << "Error deleting object, object not in database...(serial=" << serial << ")" << endl;
    }
}

void Salmon::delAll(GeomSalmonPort* port)
{
    HashTableIter<int, GeomObj*> iter = port->getIter();
    for (iter.first(); iter.ok();++iter) {
	GeomSalmonItem* si=(GeomSalmonItem*)iter.get_data();
	for (int i=0; i<roe.size(); i++)
	    roe[i]->itemDeleted(si);
    }
    port->delAll();
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

#ifdef OLDUI
void Salmon::spawnIndCB(CallbackData*, void*)
{
  topRoe.add(scinew Roe(this));
  topRoe[topRoe.size()-1]->SetTop();
  GeomItem *item;
  for (int i=0; i<topRoe[0]->geomItemA.size(); i++) {
      item=topRoe[0]->geomItemA[i];
      topRoe[topRoe.size()-1]->itemAdded(item->geom, item->name);
  }
//  printFamilyTree();
}
#endif

void Salmon::connection(ConnectionMode mode, int which_port, int)
{
    if(mode==Disconnected){
	remove_iport(which_port);
    } else {
	add_iport(scinew GeometryIPort(this, "Geometry", GeometryIPort::Atomic));
    }
}

void Salmon::delete_roe(Roe* delroe)
{
  for(int i=0;i<roe.size();i++){
    if(roe[i] == delroe){
      roe.remove(i);
      delete delroe;
    }
  }
}

void Salmon::tcl_command(TCLArgs& args, void* userdata)
{
    if(args.count() < 2){
	args.error("Salmon needs a minor command");
	return;
    }
    if(args[1] == "addroe"){
	if(args.count() != 3){
	    args.error("addroe must have a RID");
	    return;
	}
	Roe* r=scinew Roe(this, args[2]);
	roe.add(r);
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

void Salmon::execute()
{
    // Never gets called...
}

SalmonMessage::SalmonMessage(const clString& rid)
: MessageBase(MessageTypes::RoeRedraw), rid(rid), nframes(0)
{
}

SalmonMessage::SalmonMessage(const clString& rid, double tbeg, double tend,
			     int nframes, double framerate)
: MessageBase(MessageTypes::RoeRedraw), rid(rid), tbeg(tbeg), tend(tend),
  nframes(nframes), framerate(framerate)
{
    if(nframes == 0)
	cerr << "Warning - nframes shouldn't be zero for animation\n";
}

SalmonMessage::SalmonMessage(MessageTypes::MessageType type,
			     const clString& rid, const clString& filename)
: MessageBase(type), rid(rid), filename(filename)
{
}

SalmonMessage::SalmonMessage(MessageTypes::MessageType type,
			     const clString& rid, const clString& filename,
			     const clString& format)
: MessageBase(type), rid(rid), filename(filename), format(format)
{
}

SalmonMessage::~SalmonMessage()
{
}

#if 0
SceneItem::SceneItem(GeomObj* obj, const clString& name, CrowdMonitor* lock)
: obj(obj), name(name), lock(lock)
{
}

SceneItem::~SceneItem()
{
}
#endif

void Salmon::append_port_msg(GeometryComm* gmsg)
{
    // Look up the right port number
//    PortInfo* pi;
    GeomSalmonPort *pi;

    if (!(pi = ((GeomSalmonPort*)ports.getObj(gmsg->portno)))) {
//    if(!portHash.lookup(gmsg->portno, pi)){
	cerr << "Geometry message sent to bad port!!!: " << gmsg->portno << "\n";
	return;
    }

    // Queue up the messages until the flush...
    if(pi->msg_tail){
	pi->msg_tail->next=gmsg;
	pi->msg_tail=gmsg;
    } else {
	pi->msg_head=pi->msg_tail=gmsg;
    }
    gmsg->next=0;
}

void Salmon::flushPort(int portid)
{
    // Look up the right port number
    GeomSalmonPort* pi;
    if(!(pi = ((GeomSalmonPort*)ports.getObj(portid)))){
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

int Salmon::lookup_specific(const clString& key, void*& data)
{
    return specific.lookup(key, data);
}

void Salmon::insert_specific(const clString& key, void* data)
{
    specific.insert(key, data);
}

void Salmon::emit_vars(ostream& out)
{
  cerr << "Salmon emitvars" << endl;
  TCL::emit_vars(out);
  for(int i=0;i<roe.size();i++){
    out << id << " ui " << roe[i]->id << "\n";
    roe[i]->emit_vars(out);
  }
}

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.6  1999/08/25 03:47:58  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.5  1999/08/19 23:17:53  sparker
// Removed a bunch of #include <SCICore/Util/NotFinished.h> statements
// from files that did not need them.
//
// Revision 1.4  1999/08/18 20:19:53  sparker
// Eliminated copy constructor and clone in all modules
// Added a private copy ctor and a private clone method to Module so
//  that future modules will not compile until they remvoe the copy ctor
//  and clone method
// Added an ASSERTFAIL macro to eliminate the "controlling expression is
//  constant" warnings.
// Eliminated other miscellaneous warnings
//
// Revision 1.3  1999/08/17 23:50:15  sparker
// Removed all traces of the old Raytracer and X11 renderers.
// Also removed a .o and .d file
//
// Revision 1.2  1999/08/17 06:37:39  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:57:53  mcq
// Initial commit
//
// Revision 1.3  1999/07/07 21:10:28  dav
// added beginnings of support for g++ compilation
//
// Revision 1.2  1999/06/21 23:52:52  dav
// updated makefiles.main
//
// Revision 1.1.1.1  1999/04/24 23:12:31  dav
// Import sources
//
//
