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

#include <Modules/Salmon/Salmon.h>
#include <Modules/Salmon/Renderer.h>
#include <Modules/Salmon/Roe.h>
#include <Classlib/HashTable.h>
#include <Classlib/NotFinished.h>
#include <Comm/MessageTypes.h>
#include <Dataflow/Connection.h>
#include <Dataflow/HelpUI.h>
#include <Dataflow/ModuleHelper.h>
#include <Dataflow/ModuleList.h>
#include <Datatypes/GeometryComm.h>
#include <Devices/DBContext.h>
#include <Geom/Geom.h>
#include <Geom/HeadLight.h>
#include <iostream.h>

static Module* make_Salmon(const clString& id)
{
    return new Salmon(id);
}

static RegisterModule db1("Geometry", "Salmon", make_Salmon);
static RegisterModule db2("Dave", "Salmon", make_Salmon);

Salmon::Salmon(const clString& id)
: Module("Salmon", id, Sink), max_portno(0)
{
    // Add a headlight
    lighting.lights.add(new HeadLight("Headlight", Color(1,1,1)));
    // Create the input port
    add_iport(new GeometryIPort(this, "Geometry", GeometryIPort::Atomic));
    default_matl=new Material(Color(.1,.1,.1), Color(.6,0,0),
			      Color(.7,.7,.7), 10);
    busy_bit=1;
    have_own_dispatch=1;

    // Create port 0 - we use this for global objects such as cameras,
    // light source icons, etc.
    int portid=max_portno++;
    // Create the port
    PortInfo* pi=new PortInfo;
    portHash.insert(portid, pi);
    pi->msg_head=pi->msg_tail=0;
    pi->portno=portid;
    pi->objs=new HashTable<int, SceneItem*>;

    // Fill it up with the defaults...
    for(int i=0;i<lighting.lights.size();i++){
	GeomObj* geom=lighting.lights[i]->geom();
	if(geom)
	    addObj(pi, i, geom, lighting.lights[i]->name, 0);
    }
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
#ifdef OLDUI
    case MessageTypes::DoDBCallback:
	{
	    DBCallback_Message* cmsg=(DBCallback_Message*)msg;
	    cmsg->mcb->perform(cmsg->context, cmsg->which,
			       cmsg->value, cmsg->delta, cmsg->cbdata);
	    }	
	break;
#endif
    case MessageTypes::ExecuteModule:
	// We ignore these messages...
	break;
    case MessageTypes::RoeRedraw:
	{
	    SalmonMessage* rmsg=(SalmonMessage*)msg;
	    for(int i=0;i<roe.size();i++){
		Roe* r=roe[i];
		if(r->id == rmsg->rid){
		    r->need_redraw=1;
		    break;
		}
	    }
	    if(i==roe.size()){
		cerr << "Warning: Roe not found for redraw! (id=" << rmsg->rid << "\n";
	    }
	}
	break;
    case MessageTypes::RoeDump:
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
    case MessageTypes::RoeMouse:
	{
	    RoeMouseMessage* rmsg=(RoeMouseMessage*)msg;
	    for(int i=0;i<roe.size();i++){
		Roe* r=roe[i];
		if(r->id == rmsg->rid){
		    (r->*(rmsg->handler))(rmsg->action, rmsg->x, rmsg->y);
		    break;
		}
	    }
	}
	break;
    case MessageTypes::GeometryInit:
	initPort(gmsg->reply);
	break;	
    case MessageTypes::GeometryAddObj:
    case MessageTypes::GeometryDelObj:
    case MessageTypes::GeometryDelAll:
	append_port_msg(gmsg);
	msg=0; // Don't delete it yet...
	break;
    case MessageTypes::GeometryFlush:
	flushPort(gmsg->portno);
	break;
    case MessageTypes::GeometryFlushViews:
	flushPort(gmsg->portno);
	flushViews();
	break;
    default:
	cerr << "Salmon: Illegal Message type: " << msg->type << endl;
	break;
    }
    if(msg)
	delete msg;
    return 1;
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

void Salmon::initPort(Mailbox<GeomReply>* reply)
{
    int portid=max_portno++;
    // Create the port
    PortInfo* pi=new PortInfo;
    portHash.insert(portid, pi);
    pi->msg_head=pi->msg_tail=0;
    pi->portno=portid;
    pi->objs=new HashTable<int, SceneItem*>;
    reply->send(GeomReply(portid, &busy_bit));
}

void Salmon::flushViews()
{
    for (int i=0; i<roe.size(); i++)
	roe[i]->force_redraw();
}

void Salmon::addObj(PortInfo* port, int serial, GeomObj *obj,
		    const clString& name, CrowdMonitor* lock)
{
    clString pname(name+" ("+to_string(port->portno)+")");
    SceneItem* si=new SceneItem(obj, pname, lock);
    port->objs->insert(serial, si);
    for (int i=0; i<roe.size(); i++)
	roe[i]->itemAdded(si);
}

void Salmon::delObj(PortInfo* port, int serial)
{
    SceneItem* si;
    if(port->objs->lookup(serial, si)){
	port->objs->remove(serial);
	for (int i=0; i<roe.size(); i++)
	    roe[i]->itemDeleted(si);
	delete si->obj;
	delete si;
    } else {
	cerr << "Error deleting object, object not in database...(serial=" << serial << ")" << endl;
    }
}

void Salmon::delAll(PortInfo* port)
{
    HashTableIter<int, SceneItem*> iter(port->objs);
    for (iter.first(); iter.ok();++iter) {
	SceneItem* si=iter.get_data();
	for (int i=0; i<roe.size(); i++)
	    roe[i]->itemDeleted(si);
	delete si->obj;
	delete si;
    }
    port->objs->remove_all();
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
  topRoe.add(new Roe(this));
  topRoe[topRoe.size()-1]->SetTop();
  GeomItem *item;
  for (int i=0; i<topRoe[0]->geomItemA.size(); i++) {
      item=topRoe[0]->geomItemA[i];
      topRoe[topRoe.size()-1]->itemAdded(item->geom, item->name);
  }
//  printFamilyTree();
}
#endif

Salmon::Salmon(const Salmon& copy, int deep)
: Module(copy, deep)
{
    NOT_FINISHED("Salmon::Salmon");
}

void Salmon::connection(ConnectionMode mode, int which_port, int)
{
    if(mode==Disconnected){
	remove_iport(which_port);
    } else {
	add_iport(new GeometryIPort(this, "Geometry", GeometryIPort::Atomic));
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
	Roe* r=new Roe(this, args[2]);
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
: MessageBase(MessageTypes::RoeRedraw), rid(rid)
{
}

SalmonMessage::SalmonMessage(const clString& rid, const clString& filename)
: MessageBase(MessageTypes::RoeDump), rid(rid), filename(filename)
{
}

SalmonMessage::~SalmonMessage()
{
}

SceneItem::SceneItem(GeomObj* obj, const clString& name, CrowdMonitor* lock)
: obj(obj), name(name), lock(lock)
{
}

SceneItem::~SceneItem()
{
}

void Salmon::append_port_msg(GeometryComm* gmsg)
{
    // Look up the right port number
    PortInfo* pi;
    if(!portHash.lookup(gmsg->portno, pi)){
	cerr << "Geometry message sent to bad port!!!\n";
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
    PortInfo* pi;
    if(!portHash.lookup(portid, pi)){
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
	    delObj(pi, gmsg->serial);
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
