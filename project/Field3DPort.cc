
/*
 *  Field3DPort.cc: Handle to the Field3D Data type
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Field3DPort.h>
#include <Connection.h>
#include <Field3D.h>
#include <NotFinished.h>
#include <Port.h>
#include <Classlib/Assert.h>
#include <Classlib/String.h>
#include <iostream.h>

static clString Field3D_type("Field3D");
static clString Field3D_color("VioletRed2");

Field3DIPort::Field3DIPort(Module* module, const clString& portname, int protocol)
: IPort(module, Field3D_type, portname, Field3D_color, protocol),
  mailbox(2)
{
}

Field3DIPort::~Field3DIPort()
{
}

Field3DOPort::Field3DOPort(Module* module, const clString& portname, int protocol)
: OPort(module, Field3D_type, portname, Field3D_color, protocol), in(0)
{
}

Field3DOPort::~Field3DOPort()
{
}

void Field3DIPort::reset()
{
    recvd=0;
}

void Field3DIPort::finish()
{
    if(!recvd){
	turn_on();
	Field3DComm* msg=mailbox.receive();
	delete msg;
	turn_off();
    }
}

void Field3DOPort::reset()
{
    sent_something=0;
}

void Field3DOPort::finish()
{
    if(!sent_something){
	// Tell them that we didn't send anything...
	turn_on();
	if(!in){
	    Connection* connection=connections[0];
	    in=(Field3DIPort*)connection->iport;
	}
	Field3DComm* msg=new Field3DComm();
	in->mailbox.send(msg);
	turn_off();
    }
}

void Field3DOPort::send_field(const Field3DHandle& field)
{
    if(!in){
	Connection* connection=connections[0];
	in=(Field3DIPort*)connection->iport;
    }
    if(sent_something){
	cerr << "The field got sent twice - ignoring second one...\n";
	return;
    }
    turn_on();
    Field3DComm* msg=new Field3DComm(field);
    in->mailbox.send(msg);
    sent_something=1;
    turn_off();
}

int Field3DIPort::get_field(Field3DHandle& f)
{
    turn_on();
    Field3DComm* comm=mailbox.receive();
    if(comm->has_field){
       f=comm->field;
       recvd=1;
       delete comm;
       turn_off();
       return 1;
   } else {
       delete comm;
       turn_off();
       return 0;
   }
}

Field3DComm::Field3DComm()
: has_field(0)
{
}

Field3DComm::Field3DComm(const Field3DHandle& field)
: field(field), has_field(1)
{
}
