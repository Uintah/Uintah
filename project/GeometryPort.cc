
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

#include <GeometryPort.h>
#include <Connection.h>
#include <Module.h>
#include <NotFinished.h>
#include <Port.h>
#include <Classlib/Assert.h>
#include <Classlib/String.h>
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
: OPort(module, Geometry_type, portname, Geometry_color, protocol)
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
	Connection* connection=connections[0];
	Module* mod=connection->iport->get_module();
	outbox=&mod->mailbox;
	// Send the registration message...
	Mailbox<int> tmp(1);
	outbox->send(new GeometryComm(&tmp));
	portid=tmp.receive();
	serial=1;
    }
}

void GeometryOPort::finish()
{
}

GeomID GeometryOPort::addObj(GeomObj* obj)
{
    GeomID id=serial++;
    outbox->send(new GeometryComm(portid, id, obj));
    return id;
}

void GeometryOPort::delObj(GeomID id)
{
    outbox->send(new GeometryComm(portid, id));
}

void GeometryOPort::delAll()
{
    outbox->send(new GeometryComm(portid));
}

GeometryComm::GeometryComm(Mailbox<int>* reply)
: MessageBase(MessageTypes::GeometryInit), reply(reply)
{
}

GeometryComm::GeometryComm(int portno, GeomID serial, GeomObj* obj)
: MessageBase(MessageTypes::GeometryAddObj),
  portno(portno), serial(serial), obj(obj)
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

GeometryComm::~GeometryComm()
{
}
