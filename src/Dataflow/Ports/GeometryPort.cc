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

GeometryIPort::GeometryIPort(Module* module, const string& portname)
  : IPort(module, Geometry_type, portname, Geometry_color)
{
}

GeometryIPort::~GeometryIPort()
{
}


void GeometryIPort::reset()
{
}

void GeometryIPort::finish()
{
}



GeometryOPort::GeometryOPort(Module* module, const string& portname)
  : OPort(module, Geometry_type, portname, Geometry_color),
    serial_(1), lserial_(4),
    dirty_(false)
{
}


GeometryOPort::~GeometryOPort()
{
  list<GeometryComm *>::iterator itr = saved_msgs_.begin();
  while (itr != saved_msgs_.end())
  {
    delete *itr;
    ++itr;
  }
  saved_msgs_.clear();
}


void
GeometryOPort::reset()
{
  dirty_ = false;
}


void
GeometryOPort::flush()
{
  for (unsigned int i=0; i<outbox_.size(); i++)
  {
    GeometryComm* msg = scinew GeometryComm(MessageTypes::GeometryFlush,
					    portid_[i]);
    outbox_[i]->send(msg);
  }
  GeometryComm* msg = scinew GeometryComm(MessageTypes::GeometryFlush, 0);
  save_msg(msg);
}
  

void
GeometryOPort::finish()
{
  if (dirty_)
  {
    if (module->showStats()) turn_on(Finishing);
    flush();
    if (module->showStats()) turn_off();
  }
}



GeomID
GeometryOPort::addObj(GeomHandle obj, const string& name, CrowdMonitor* lock)
{
  if (module->showStats()) turn_on();
  GeomID id = serial_++;
  for (unsigned int i = 0; i < outbox_.size(); i++)
  {
    GeometryComm* msg = scinew GeometryComm(portid_[i], id, obj, name, lock);
    outbox_[i]->send(msg);
  }
  GeometryComm* msg = scinew GeometryComm(0, id, obj, name, lock);
  save_msg(msg);
  dirty_ = true;
  if (module->showStats()) turn_off();
  return id;
}

LightID
GeometryOPort::addLight(LightHandle obj, 
			const string& name, CrowdMonitor* lock)
{
  //static LightID next_id = 1;
  if (module->showStats()) turn_on();
  LightID id = lserial_++;;
//   if( next_id > lserial_ ){
//     id = next_id++;
//     lserial_ = next_id;
//   } else {
//     id = lserial_++;
//     next_id = lserial_;
//   }

  for (unsigned int i = 0; i < outbox_.size(); i++)
  {
    GeometryComm* msg = scinew GeometryComm(portid_[i], id, obj, name, lock);
    outbox_[i]->send(msg);
  }
  GeometryComm* msg = scinew GeometryComm(0, id, obj, name, lock);
  save_msg(msg);
  dirty_ = true;
  if (module->showStats()) turn_off();
  return id;
}



bool
GeometryOPort::direct_forward(GeometryComm* msg)
{
  for (unsigned int i = 0; i < outbox_.size(); i++)
  {
    GeometryComm *cpy = scinew GeometryComm(*msg);
//    cpy->portno = portid_[i];
    outbox_[i]->send(cpy);
  }
  return outbox_.size() > 0;
}



void
GeometryOPort::forward(GeometryComm* msg0)
{
  for (unsigned int i = 0; i < outbox_.size(); i++)
  {
    GeometryComm *msg = scinew GeometryComm(*msg0);
    msg->portno = portid_[i];
    outbox_[i]->send(msg);
  }
  save_msg(msg0);
}



void
GeometryOPort::delObj(GeomID id, int del)
{
  if (module->showStats()) turn_on();

  for (unsigned int i=0; i < outbox_.size(); i++)
  {
    GeometryComm* msg = scinew GeometryComm(portid_[i], id);
    outbox_[i]->send(msg);
  }
  GeometryComm* msg = scinew GeometryComm(0, id);
  save_msg(msg);
  dirty_ = true;
  if (module->showStats()) turn_off();
}

void
GeometryOPort::delLight(LightID id, int del)
{
  if (module->showStats()) turn_on();

  for (unsigned int i=0; i < outbox_.size(); i++)
  {
    GeometryComm* msg = scinew GeometryComm(portid_[i], id);
    outbox_[i]->send(msg);
  }
  GeometryComm* msg = scinew GeometryComm(0, id);
  save_msg(msg);
  dirty_ = true;
  if (module->showStats()) turn_off();
}



void
GeometryOPort::delAll()
{
  if (module->showStats()) turn_on();
  for (unsigned int i = 0; i < outbox_.size(); i++)
  {
    GeometryComm* msg = scinew GeometryComm(MessageTypes::GeometryDelAll,
					    portid_[i]);
    outbox_[i]->send(msg);
  }
  GeometryComm* msg = scinew GeometryComm(MessageTypes::GeometryDelAll, 0);
  save_msg(msg);
  dirty_ = true;
  if (module->showStats()) turn_off();
}



void
GeometryOPort::flushViews()
{
  if (module->showStats()) turn_on();
  for (unsigned int i = 0; i < outbox_.size(); i++)
  {
    GeometryComm* msg = scinew GeometryComm(MessageTypes::GeometryFlushViews,
					    portid_[i], (Semaphore*)0);
    outbox_[i]->send(msg);
  }
  GeometryComm* msg = scinew GeometryComm(MessageTypes::GeometryFlushViews,
					  0, (Semaphore*)0);
  save_msg(msg);
  dirty_ = false;
  if (module->showStats()) turn_off();
}


void
GeometryOPort::flushViewsAndWait()
{
  if (module->showStats()) turn_on();
  Semaphore waiter("flushViewsAndWait wait semaphore", 0);
  for (unsigned int i = 0; i < outbox_.size(); i++)
  {
    GeometryComm* msg = scinew GeometryComm(MessageTypes::GeometryFlushViews,
					    portid_[i], &waiter);
    outbox_[i]->send(msg);
  }
  GeometryComm* msg = scinew GeometryComm(MessageTypes::GeometryFlushViews,
					  0, &waiter);
  save_msg(msg); // TODO:  Should a synchronized primitive be queued?

  // Wait on everyone.
  for (unsigned int i = 0; i < outbox_.size(); i++)
  {
    waiter.down();
  }
  dirty_ = false;
  if (module->showStats()) turn_off();
}


void
GeometryOPort::save_msg(GeometryComm* msg)
{
  switch(msg->type)
  {
  case MessageTypes::GeometryDelObj:
    // Delete the object from the message queue.
    {
      list<GeometryComm *>::iterator itr0 = saved_msgs_.begin();
      while (itr0 != saved_msgs_.end())
      {
	list<GeometryComm *>::iterator itr  = itr0;
	++itr0;
	if ((*itr)->type == MessageTypes::GeometryAddObj &&
	    (*itr)->serial == msg->serial)
	{
	  delete *itr;
	  saved_msgs_.erase(itr);
	}
      }
      delete msg;
    }
    break;
  case MessageTypes::GeometryDelLight:
    // Delete the object from the message queue.
    {
      list<GeometryComm *>::iterator itr0 = saved_msgs_.begin();
      while (itr0 != saved_msgs_.end())
      {
	list<GeometryComm *>::iterator itr  = itr0;
	++itr0;
	if ((*itr)->type == MessageTypes::GeometryAddLight &&
	    (*itr)->lserial == msg->lserial)
	{
	  delete *itr;
	  saved_msgs_.erase(itr);
	}
      }
      delete msg;
    }
    break;

  case MessageTypes::GeometryDelAll:
    // Delete all AddObj messages from the queue.
    {
      list<GeometryComm *>::iterator itr0 = saved_msgs_.begin();
      while (itr0 != saved_msgs_.end())
      {
	list<GeometryComm *>::iterator itr  = itr0;
	++itr0;
	if ((*itr)->type == MessageTypes::GeometryAddObj)
	{
	  delete *itr;
	  saved_msgs_.erase(itr);
	}
      }
      delete msg;
    }
    break;

  case MessageTypes::GeometryFlush:
    // Delete the object from the message queue.
    {
      list<GeometryComm *>::iterator itr0 = saved_msgs_.begin();
      while (itr0 != saved_msgs_.end())
      {
	list<GeometryComm *>::iterator itr  = itr0;
	++itr0;
	if ((*itr)->type == MessageTypes::GeometryFlush)
	{
	  delete *itr;
	  saved_msgs_.erase(itr);
	}
      }
      saved_msgs_.push_back(msg);
    }
    break;

  case MessageTypes::GeometryFlushViews:
    // Delete the object from the message queue.
    {
      list<GeometryComm *>::iterator itr0 = saved_msgs_.begin();
      while (itr0 != saved_msgs_.end())
      {
	list<GeometryComm *>::iterator itr  = itr0;
	++itr0;
	if ((*itr)->type == MessageTypes::GeometryFlushViews)
	{
	  delete *itr;
	  saved_msgs_.erase(itr);
	}
      }
      saved_msgs_.push_back(msg);
    }
    break;

  case MessageTypes::GeometrySetView:
    // Delete the object from the message queue.
    {
      list<GeometryComm *>::iterator itr0 = saved_msgs_.begin();
      while (itr0 != saved_msgs_.end())
      {
	list<GeometryComm *>::iterator itr  = itr0;
	++itr0;
	if ((*itr)->type == MessageTypes::GeometrySetView)
	{
	  delete *itr;
	  saved_msgs_.erase(itr);
	}
      }
      saved_msgs_.push_back(msg);
    }
    break;

  default:
    saved_msgs_.push_back(msg);
  }
}


void
GeometryOPort::attach(Connection* c)
{
  OPort::attach(c);

  int which = outbox_.size();

  // Set up the outbox_ and portid_ variables.
  if (module->showStats()) turn_on(Resetting);
  Module* mod = c->iport->get_module();
  outbox_.push_back(&mod->mailbox);
  // Send the registration message.
  
  Mailbox<GeomReply> *tmp = 
    new Mailbox<GeomReply> ("Temporary GeometryOPort mailbox", 1);
  outbox_[which]->send(scinew GeometryComm(tmp));
  GeomReply reply = tmp->receive();
  portid_.push_back(reply.portid);
  if (module->showStats()) turn_off();

  // Forward all of the queued up messages.
  if (module->showStats()) turn_on();
  list<GeometryComm *>::iterator itr = saved_msgs_.begin();
  while (itr != saved_msgs_.end())
  {
    GeometryComm *msg = scinew GeometryComm(**itr);
    msg->portno = portid_[which];
    outbox_[which]->send(msg);
    ++itr;
  }
  if (module->showStats()) turn_off();
}


void
GeometryOPort::detach(Connection* c)
{
  // Determine which connection gets it.
  unsigned int i;
  for (i = 0; i < connections.size(); i++)
  {
    if (connections[i] == c)
    {
      break;
    }
  }
  
  if (i < connections.size())
  {
    // Let the Viewer know that the port is shutting down.
    GeometryComm *msg =
      scinew GeometryComm(MessageTypes::GeometryDetach, portid_[i]);
    outbox_[i]->send(msg);

    // Clean up the outbox_ and portid_ vectors.
    outbox_.erase(outbox_.begin() + i);
    portid_.erase(portid_.begin() + i);
  }

  OPort::detach(c);
}


bool
GeometryOPort::have_data()
{
  return saved_msgs_.size();
}


void
GeometryOPort::resend(Connection*)
{
  cerr << "GeometryOPort can't resend and shouldn't need to!\n";
}

int
GeometryOPort::getNViewers()
{
  return outbox_.size();
}


int
GeometryOPort::getNViewWindows(int viewer)
{
  if (viewer < 0 || viewer >= (int)outbox_.size()) return 0;

  FutureValue<int> reply("Geometry getNViewWindows reply");
  GeometryComm *msg =
    scinew GeometryComm(MessageTypes::GeometryGetNViewWindows,
			portid_[viewer], &reply);
  outbox_[viewer]->send(msg);

  return reply.receive();
}


GeometryData *
GeometryOPort::getData(int which_viewer, int which_viewwindow, int datamask)
{
  if (which_viewer >= outbox_.size() || which_viewer < 0) return 0;

  FutureValue<GeometryData*> reply("Geometry getData reply");
  GeometryComm *msg = scinew GeometryComm(MessageTypes::GeometryGetData,
					  portid_[which_viewer],
					  &reply, which_viewwindow,
					  datamask);
  outbox_[which_viewer]->send(msg);
  return reply.receive();
}


void
GeometryOPort::setView(int which_viewer, int which_viewwindow, View view)
{
  if (which_viewer >= outbox_.size() || which_viewer < 0) return;

  GeometryComm* msg = scinew GeometryComm(MessageTypes::GeometrySetView,
					  portid_[which_viewer],
					  which_viewwindow,
					  view);
  outbox_[which_viewer]->send(msg);
}


GeometryComm::GeometryComm(Mailbox<GeomReply>* reply)
  : MessageBase(MessageTypes::GeometryInit),
    reply(reply)
{
}


GeometryComm::GeometryComm(int portno, GeomID serial, GeomHandle obj,
			   const string& name, CrowdMonitor* lock)
  : MessageBase(MessageTypes::GeometryAddObj),
    portno(portno),
    serial(serial),
    obj(obj),
    name(name),
    lock(lock)
{
}
GeometryComm::GeometryComm(int portno, LightID serial, LightHandle light,
			   const string& name, CrowdMonitor* lock)
  : MessageBase(MessageTypes::GeometryAddLight),
    portno(portno),
    lserial(serial),
    light(light),
    name(name),
    lock(lock)
{
}


GeometryComm::GeometryComm(int portno, GeomID serial)
  : MessageBase(MessageTypes::GeometryDelObj),
    portno(portno),
    serial(serial)
{
}

GeometryComm::GeometryComm(int portno, LightID serial)
  : MessageBase(MessageTypes::GeometryDelLight),
    portno(portno),
    lserial(serial)
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


GeometryComm::GeometryComm(const GeometryComm &copy)
  : MessageBase(copy),
    reply(copy.reply),
    portno(copy.portno),
    serial(copy.serial),
    obj(copy.obj),
    lserial(copy.lserial),
    light(copy.light),
    name(copy.name),
    lock(0),
    wait(0),
    view(copy.view),
    next(0),
    which_viewwindow(copy.which_viewwindow),
    datamask(copy.datamask),
    datareply(0),
    nreply(0)
{
}


GeometryComm::~GeometryComm()
{
}



GeomReply::GeomReply()
{
}


GeomReply::GeomReply(int portid)
  : portid(portid)
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

