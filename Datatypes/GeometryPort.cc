
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
  outbox(0)
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
	turn_on(Finishing);
	outbox->send(new GeometryComm);
	turn_off();
    }
}

GeomID GeometryOPort::addObj(GeomObj* obj, const clString& name)
{
    turn_on();
    GeomID id=serial++;
    outbox->send(new GeometryComm(portid, id, obj, name));
    dirty=1;
    turn_off();
    return id;
}

void GeometryOPort::delObj(GeomID id)
{
    turn_on();
    outbox->send(new GeometryComm(portid, id));
    dirty=1;
    turn_off();
}

void GeometryOPort::delAll()
{
    turn_on();
    outbox->send(new GeometryComm(portid));
    dirty=1;
    turn_off();
}

void GeometryOPort::flushViews()
{
    turn_on();
    outbox->send(new GeometryComm);
    dirty=0;
    turn_off();
}

int GeometryOPort::busy()
{
    return *busy_bit;
}

GeometryComm::GeometryComm(Mailbox<GeomReply>* reply)
: MessageBase(MessageTypes::GeometryInit), reply(reply)
{
}

GeometryComm::GeometryComm(int portno, GeomID serial, GeomObj* obj,
			   const clString& name)
: MessageBase(MessageTypes::GeometryAddObj),
  portno(portno), serial(serial), obj(obj), name(name)
{
}

GeometryComm::GeometryComm(int portno, GeomID serial)
: MessageBase(MessageTypes::GeometryDelObj),
  portno(portno), serial(serial)
{
}

GeometryComm::GeometryComm(int portno)
: MessageBase(MessageTypes::GeometryDelAll),
  portno(portno)
{
}

GeometryComm::GeometryComm()
: MessageBase(MessageTypes::GeometryFlush)
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
