
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
	outbox->send(new GeometryComm(&tmp));
	GeomReply reply=tmp.receive();
	portid=reply.portid;
	busy_bit=reply.busy_bit;
	serial=1;
	turn_off();
    }
    dirty=0;
}

void GeometryOPort::finish()
{
    if(dirty){
	GeometryComm* msg=new GeometryComm(MessageTypes::GeometryFlush, portid);
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
    GeometryComm* msg=new GeometryComm(portid, id, obj, name, lock);
    if(outbox)
	outbox->send(msg);
    else
	save_msg(msg);
    dirty=1;
    turn_off();
    return id;
}

void GeometryOPort::delObj(GeomID id)
{
    turn_on();
    GeometryComm* msg=new GeometryComm(portid, id);
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
    GeometryComm* msg=new GeometryComm(MessageTypes::GeometryDelAll, portid);
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
    GeometryComm* msg=new GeometryComm(MessageTypes::GeometryFlushViews, portid);
    if(outbox)
	outbox->send(msg);
    else
	save_msg(msg);
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
    cerr << "saving message: " << msg << endl;
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
	cerr << "sending message: " << p << endl;
	outbox->send(p);
	p=next;
    }
    save_msgs=0;
    turn_off();
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

GeometryComm::GeometryComm(int portno, GeomID serial)
: MessageBase(MessageTypes::GeometryDelObj),
  portno(portno), serial(serial)
{
}

GeometryComm::GeometryComm(MessageTypes::MessageType type, int portno)
: MessageBase(type), portno(portno)
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
