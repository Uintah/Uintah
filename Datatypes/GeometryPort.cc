
/*
 *  GeometryPort.cc: Handle to the Geometry Data type
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Datatypes/GeometryPort.h>
#include <Datatypes/GeometryComm.h>

#include <Classlib/Assert.h>
#include <Classlib/NotFinished.h>
#include <Classlib/String.h>
#include <Dataflow/Connection.h>
#include <Dataflow/Module.h>
#include <Dataflow/Port.h>
#include <Malloc/Allocator.h>
#include <Multitask/AsyncReply.h>

#include <iostream.h>

static clString Geometry_type("Geometry");
static clString Geometry_color("magenta3");

GeometryIPort::GeometryIPort(Module* module, const clString& portname, int protocol)
: IPort(module, Geometry_type, portname, Geometry_color, protocol)
{
}

GeometryIPort::~GeometryIPort()
{
}

GeometryOPort::GeometryOPort(Module* module, const clString& portname, int protocol)
: OPort(module, Geometry_type, portname, Geometry_color, protocol),
  save_msgs(0), outbox(0)
{
    serial=1;
}

GeometryOPort::~GeometryOPort()
{
}

void GeometryIPort::reset()
{
}

void GeometryIPort::finish()
{
}

void GeometryOPort::reset()
{
    if(nconnections() == 0)
	return;
    if(!outbox){
	turn_on(Resetting);
	Connection* connection=connections[0];
	Module* mod=connection->iport->get_module();
	outbox=&mod->mailbox;
	// Send the registration message...
	Mailbox<GeomReply> tmp(1);
	outbox->send(scinew GeometryComm(&tmp));
	GeomReply reply=tmp.receive();
	portid=reply.portid;
	busy_bit=reply.busy_bit;
	turn_off();
    }
    dirty=0;
}

void GeometryOPort::flush()
{
  GeometryComm* msg=scinew GeometryComm(MessageTypes::GeometryFlush, portid);
  if(outbox){
    outbox->send(msg);
  } else {
    save_msg(msg);
  } 
}
  
void GeometryOPort::finish()
{
    if(dirty){
	GeometryComm* msg=scinew GeometryComm(MessageTypes::GeometryFlush, portid);
	if(outbox){
	    turn_on(Finishing);
	    outbox->send(msg);
	    turn_off();
	} else {
	    save_msg(msg);
	}
    }
}

GeomID GeometryOPort::addObj(GeomObj* obj, const clString& name,
			     CrowdMonitor* lock)
{
    turn_on();
    GeomID id=serial++;
    GeometryComm* msg=scinew GeometryComm(portid, id, obj, name, lock);
    if(outbox){
	outbox->send(msg);
    } else {
	save_msg(msg);
    }
    dirty=1;
    turn_off();
    return id;
}

void GeometryOPort::forward(GeometryComm* msg)
{
    /*turn_on();*/
    if(outbox){
	msg->portno=portid;
	outbox->send(msg);
    } else {
	save_msg(msg);
    }
    /*turn_off();*/
}

void GeometryOPort::delObj(GeomID id, int del)
{
    turn_on();
    GeometryComm* msg=scinew GeometryComm(portid, id, del);
    if(outbox)
	outbox->send(msg);
    else
	save_msg(msg);
    dirty=1;
    turn_off();
}

void GeometryOPort::delAll()
{
    turn_on();
    GeometryComm* msg=scinew GeometryComm(MessageTypes::GeometryDelAll, portid);
    if(outbox)
	outbox->send(msg);
    else
	save_msg(msg);
    dirty=1;
    turn_off();
}

void GeometryOPort::flushViews()
{
    turn_on();
    GeometryComm* msg=scinew GeometryComm(MessageTypes::GeometryFlushViews, portid, (Semaphore*)0);
    if(outbox)
	outbox->send(msg);
    else
	save_msg(msg);
    dirty=0;
    turn_off();
}

void GeometryOPort::flushViewsAndWait()
{
    turn_on();
    Semaphore waiter(0);
    GeometryComm* msg=scinew GeometryComm(MessageTypes::GeometryFlushViews, portid, &waiter);
    if(outbox)
	outbox->send(msg);
    else
	save_msg(msg);
    waiter.down();
    dirty=0;
    turn_off();
}

int GeometryOPort::busy()
{
    return *busy_bit;
}

void GeometryOPort::save_msg(GeometryComm* msg)
{
    if(save_msgs){
	save_msgs_tail->next=msg;
	save_msgs_tail=msg;
    } else {
	save_msgs=save_msgs_tail=msg;
    }
    msg->next=0;
}

void GeometryOPort::attach(Connection* c)
{
    OPort::attach(c);
    reset();
    turn_on();
    GeometryComm* p=save_msgs;
    while(p){
	GeometryComm* next=p->next;
	p->portno=portid;
	outbox->send(p);
	p=next;
    }
    save_msgs=0;
    turn_off();
}

int GeometryOPort::have_data()
{
    return 0;
}

void GeometryOPort::resend(Connection*)
{
    cerr << "GeometryOPort can't resend and shouldn't need to!\n";
}

int GeometryOPort::getNRoe()
{
    if(nconnections() == 0)
	return 0;
    AsyncReply<int> reply;
    outbox->send(new GeometryComm(MessageTypes::GeometryGetNRoe, portid, &reply));
    return reply.wait();
}

GeometryData* GeometryOPort::getData(int which_roe, int datamask)
{
    if(nconnections() == 0)
	return 0;
    AsyncReply<GeometryData*> reply;
    outbox->send(new GeometryComm(MessageTypes::GeometryGetData, portid, &reply, which_roe, datamask));
    return reply.wait();
}

GeometryComm::GeometryComm(Mailbox<GeomReply>* reply)
: MessageBase(MessageTypes::GeometryInit), reply(reply)
{
}

GeometryComm::GeometryComm(int portno, GeomID serial, GeomObj* obj,
			   const clString& name, CrowdMonitor* lock)
: MessageBase(MessageTypes::GeometryAddObj),
  portno(portno), serial(serial), obj(obj), name(name), lock(lock)
{
}

GeometryComm::GeometryComm(int portno, GeomID serial, int del)
: MessageBase(MessageTypes::GeometryDelObj),
  portno(portno), serial(serial), del(del)
{
}

GeometryComm::GeometryComm(MessageTypes::MessageType type, int portno, Semaphore* wait)
: MessageBase(type), portno(portno), wait(wait)
{
}

GeometryComm::GeometryComm(MessageTypes::MessageType type, int portno)
: MessageBase(type), portno(portno)
{
}

GeometryComm::GeometryComm(MessageTypes::MessageType type, int portno,
			   AsyncReply<GeometryData*>* datareply,
			   int which_roe, int datamask)
: MessageBase(type), portno(portno), datareply(datareply),
  which_roe(which_roe), datamask(datamask)
{
}

GeometryComm::GeometryComm(MessageTypes::MessageType type, int portno,
			   AsyncReply<int>* nreply)
: MessageBase(type), portno(portno), nreply(nreply)
{
}

GeometryComm::~GeometryComm()
{
}

GeomReply::GeomReply()
{
}

GeomReply::GeomReply(int portid, int* busy_bit)
: portid(portid), busy_bit(busy_bit)
{
}

GeometryData::GeometryData()
{
    view=0;
    colorbuffer=0;
    depthbuffer=0;
}

void
GeometryData::Print()
{
  cerr << "GEOMETRY data review\n\n";
  cerr << "X resolution: " << xres << " Y resolution: " << yres << endl;
  cerr << "Clipping planes.  Near = " << znear << " Far = " << zfar << endl;

  if ( depthbuffer == NULL )
    cerr << "depthbuffer has nothing\n";

  if ( colorbuffer == NULL )
    cerr << "colorbuffer has nothing\n";

  if ( view == NULL )
    cerr << "view has nothing\n";

  cerr << endl;
}

#ifdef __GNUG__

#include <Multitask/Mailbox.cc>
template class Mailbox<GeomReply>;

#endif
