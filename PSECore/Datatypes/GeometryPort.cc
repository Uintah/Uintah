//static char *id="@(#) $Id$";

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

#include <PSECore/Datatypes/GeometryPort.h>
#include <PSECore/Datatypes/GeometryComm.h>

#include <SCICore/Util/Assert.h>
#include <SCICore/Containers/String.h>
#include <PSECore/Dataflow/Connection.h>
#include <PSECore/Dataflow/Module.h>
#include <PSECore/Dataflow/Port.h>
#include <SCICore/Geom/View.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/Thread/FutureValue.h>

#include <iostream>
using std::cerr;
using std::endl;

namespace PSECore {
namespace Datatypes {

extern "C" {
PSECORESHARE IPort* make_GeometryIPort(Module* module, const clString& name) {
  return new GeometryIPort(module,name);
}
PSECORESHARE OPort* make_GeometryOPort(Module* module, const clString& name) {
  return new GeometryOPort(module,name);
}
}

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
	if (module->show_stat) turn_on(Resetting);
	Connection* connection=connections[0];
	Module* mod=connection->iport->get_module();
	outbox=&mod->mailbox;
	// Send the registration message...
	Mailbox<GeomReply> tmp("Temporary GeometryOPort mailbox", 1);
	outbox->send(scinew GeometryComm(&tmp));
	GeomReply reply=tmp.receive();
	portid=reply.portid;
	busy_bit=reply.busy_bit;
	if (module->show_stat) turn_off();
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
	    if (module->show_stat) turn_on(Finishing);
	    outbox->send(msg);
	    if (module->show_stat) turn_off();
	} else {
	    save_msg(msg);
	}
    }
}

GeomID GeometryOPort::addObj(GeomObj* obj, const clString& name,
			     CrowdMonitor* lock)
{
    if (module->show_stat) turn_on();
    GeomID id=serial++;
    GeometryComm* msg=scinew GeometryComm(portid, id, obj, name, lock);
    if(outbox){
	outbox->send(msg);
    } else {
	save_msg(msg);
    }
    dirty=1;
    if (module->show_stat) turn_off();
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
    if (module->show_stat) turn_on();
    GeometryComm* msg=scinew GeometryComm(portid, id, del);
    if(outbox)
	outbox->send(msg);
    else
	save_msg(msg);
    dirty=1;
    if (module->show_stat) turn_off();
}

void GeometryOPort::delAll()
{
    if (module->show_stat) turn_on();
    GeometryComm* msg=scinew GeometryComm(MessageTypes::GeometryDelAll, portid);
    if(outbox)
	outbox->send(msg);
    else
	save_msg(msg);
    dirty=1;
    if (module->show_stat) turn_off();
}

void GeometryOPort::flushViews()
{
    if (module->show_stat) turn_on();
    GeometryComm* msg=scinew GeometryComm(MessageTypes::GeometryFlushViews, portid, (Semaphore*)0);
    if(outbox)
	outbox->send(msg);
    else
	save_msg(msg);
    dirty=0;
    if (module->show_stat) turn_off();
}

void GeometryOPort::flushViewsAndWait()
{
    if (module->show_stat) turn_on();
    Semaphore waiter("flushViewsAndWait wait semaphore", 0);
    GeometryComm* msg=scinew GeometryComm(MessageTypes::GeometryFlushViews, portid, &waiter);
    if(outbox)
	outbox->send(msg);
    else
	save_msg(msg);
    waiter.down();
    dirty=0;
    if (module->show_stat) turn_off();
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
    if (module->show_stat) turn_on();
    GeometryComm* p=save_msgs;
    while(p){
	GeometryComm* next=p->next;
	p->portno=portid;
	outbox->send(p);
	p=next;
    }
    save_msgs=0;
    if (module->show_stat) turn_off();
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
    FutureValue<int> reply("Geometry getNRoe reply");
    outbox->send(new GeometryComm(MessageTypes::GeometryGetNRoe, portid, &reply));
    return reply.receive();
}

GeometryData* GeometryOPort::getData(int which_roe, int datamask)
{
    if(nconnections() == 0)
	return 0;
    FutureValue<GeometryData*> reply("Geometry getData reply");
    outbox->send(new GeometryComm(MessageTypes::GeometryGetData, portid, &reply, which_roe, datamask));
    return reply.receive();
}

void GeometryOPort::setView(int which_roe, View view)
{
    if(nconnections() == 0)
	return;
    
    GeometryComm* msg=scinew GeometryComm(MessageTypes::GeometrySetView, portid, which_roe, view);

    if(outbox)
	outbox->send(msg);
    else
	save_msg(msg);
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
			   FutureValue<GeometryData*>* datareply,
			   int which_roe, int datamask)
: MessageBase(type), portno(portno), datareply(datareply),
  which_roe(which_roe), datamask(datamask)
{
}

GeometryComm::GeometryComm(MessageTypes::MessageType type, int portno,
			   int which_roe, View view)
: MessageBase(type), portno(portno), which_roe(which_roe), view(view)
{
}

GeometryComm::GeometryComm(MessageTypes::MessageType type, int portno,
			   FutureValue<int>* nreply)
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

} // End namespace Datatypes
} // End namespace PSECore

//
// $Log$
// Revision 1.10  2000/11/22 17:14:41  moulding
// added extern "C" make functions for input and output ports (to be used
// by the auto-port facility).
//
// Revision 1.9  1999/12/07 02:53:34  dmw
// made show_status variable persistent with network maps
//
// Revision 1.8  1999/12/03 00:36:08  dmw
// more files for the setView message
//
// Revision 1.7  1999/11/11 19:56:37  dmw
// added show_status check for GeometryPort and SoundPort
//
// Revision 1.6  1999/10/07 02:07:21  sparker
// use standard iostreams and complex type
//
// Revision 1.5  1999/08/28 17:54:31  sparker
// Integrated new Thread library
//
// Revision 1.4  1999/08/25 03:48:20  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.3  1999/08/19 23:18:03  sparker
// Removed a bunch of #include <SCICore/Util/NotFinished.h> statements
// from files that did not need them.
//
// Revision 1.2  1999/08/17 06:38:08  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:55:47  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:49  dav
// Import sources
//
//
