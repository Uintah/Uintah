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

#include <Dataflow/Ports/GeometryPort.h>
#include <Dataflow/Ports/GeometryComm.h>

#include <Core/Util/Assert.h>
#include <Dataflow/Network/Connection.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Network/Port.h>
#include <Core/Geom/View.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Thread/FutureValue.h>

#include <iostream>
using std::cerr;
//using std::endl;

namespace SCIRun {

extern "C" {
PSECORESHARE IPort* make_GeometryIPort(Module* module, const string& name) {
  return scinew GeometryIPort(module,name);
}
PSECORESHARE OPort* make_GeometryOPort(Module* module, const string& name) {
  return scinew GeometryOPort(module,name);
}
}

static string Geometry_type("Geometry");
static string Geometry_color("magenta3");

GeometryIPort::GeometryIPort(Module* module, const string& portname, int protocol)
: IPort(module, Geometry_type, portname, Geometry_color, protocol)
{
}

GeometryIPort::~GeometryIPort()
{
}

GeometryOPort::GeometryOPort(Module* module, const string& portname, int protocol)
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

GeomID GeometryOPort::addObj(GeomObj* obj, const string& name,
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

int GeometryOPort::getNViewWindows()
{
    if(nconnections() == 0)
	return 0;
    FutureValue<int> reply("Geometry getNViewWindows reply");
    outbox->send(new GeometryComm(MessageTypes::GeometryGetNViewWindows, portid, &reply));
    return reply.receive();
}

GeometryData* GeometryOPort::getData(int which_viewwindow, int datamask)
{
    if(nconnections() == 0)
	return 0;
    FutureValue<GeometryData*> reply("Geometry getData reply");
    outbox->send(new GeometryComm(MessageTypes::GeometryGetData, portid, &reply, which_viewwindow, datamask));
    return reply.receive();
}

void GeometryOPort::setView(int which_viewwindow, View view)
{
    if(nconnections() == 0)
	return;
    
    GeometryComm* msg=scinew GeometryComm(MessageTypes::GeometrySetView, portid, which_viewwindow, view);

    if(outbox)
	outbox->send(msg);
    else
	save_msg(msg);
}

GeometryComm::GeometryComm(Mailbox<GeomReply>* reply)
  : MessageBase(MessageTypes::GeometryInit),
    reply(reply)
{
}

GeometryComm::GeometryComm(int portno, GeomID serial, GeomObj* obj,
			   const string& name, CrowdMonitor* lock)
  : MessageBase(MessageTypes::GeometryAddObj),
    portno(portno),
    serial(serial),
    obj(obj),
    name(name),
    lock(lock)
{
}

GeometryComm::GeometryComm(int portno, GeomID serial, int del)
  : MessageBase(MessageTypes::GeometryDelObj),
    portno(portno),
    serial(serial),
    del(del)
{
}

GeometryComm::GeometryComm(MessageTypes::MessageType type,
			   int portno, Semaphore* wait)
  : MessageBase(type),
    portno(portno),
    wait(wait)
{
}

GeometryComm::GeometryComm(MessageTypes::MessageType type, int portno)
  : MessageBase(type),
    portno(portno)
{
}

GeometryComm::GeometryComm(MessageTypes::MessageType type, int portno,
			   FutureValue<GeometryData*>* datareply,
			   int which_viewwindow, int datamask)
  : MessageBase(type),
    portno(portno),
    which_viewwindow(which_viewwindow),
    datamask(datamask),
    datareply(datareply)
{
}

GeometryComm::GeometryComm(MessageTypes::MessageType type, int portno,
			   int which_viewwindow, View view)
: MessageBase(type),
  portno(portno),
  view(view),
  which_viewwindow(which_viewwindow)
{
}

GeometryComm::GeometryComm(MessageTypes::MessageType type, int portno,
			   FutureValue<int>* nreply)
  : MessageBase(type),
    portno(portno),
    nreply(nreply)
{
}

GeometryComm::~GeometryComm()
{
}

GeomReply::GeomReply()
{
}

GeomReply::GeomReply(int portid, int* busy_bit)
  : portid(portid),
    busy_bit(busy_bit)
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
  cerr << "X resolution: " << xres << " Y resolution: " << yres << std::endl;
  cerr << "Clipping planes.  Near = " << znear << " Far = " << zfar << std::endl;

  if ( depthbuffer == NULL )
    cerr << "depthbuffer has nothing\n";

  if ( colorbuffer == NULL )
    cerr << "colorbuffer has nothing\n";

  if ( view == NULL )
    cerr << "view has nothing\n";

  cerr << std::endl;
}

} // End namespace SCIRun

