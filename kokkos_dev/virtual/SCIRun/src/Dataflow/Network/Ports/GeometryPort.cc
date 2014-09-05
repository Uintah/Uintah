/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
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

#include <Dataflow/Network/Ports/GeometryPort.h>
#include <Dataflow/Network/Ports/GeometryComm.h>

#include <Dataflow/Network/Connection.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Network/Ports/Port.h>
#include <Core/Geom/View.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Thread/FutureValue.h>
#include <Core/Util/Assert.h>

#include <iostream>
using std::cerr;

#undef SCISHARE
#if defined(_WIN32) && !defined(BUILD_DATAFLOW_STATIC)
#  define SCISHARE __declspec(dllexport)
#else
#  define SCISHARE
#endif

namespace SCIRun {

extern "C" {
  SCISHARE IPort* make_GeometryIPort(Module* module, const string& name) {
    return scinew GeometryIPort(module,name);
  }
  SCISHARE OPort* make_GeometryOPort(Module* module, const string& name) {
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
    turn_on_light(Finishing);
    flush();
    turn_off_light();
  }
}


void
GeometryOPort::synchronize()
{
  for (unsigned int i=0; i<outbox_.size(); i++)
  {
    GeometryComm* msg =
      scinew GeometryComm(MessageTypes::GeometrySynchronize, portid_[i]);
    outbox_[i]->send(msg);
  }
}


GeomID
GeometryOPort::addObj(GeomHandle obj, const string& name, CrowdMonitor* lock)
{
  turn_on_light();
  GeomID id = serial_++;
  for (unsigned int i = 0; i < outbox_.size(); i++)
  {
    GeometryComm* msg = scinew GeometryComm(portid_[i], id, obj, name, lock);
    outbox_[i]->send(msg);
  }
  GeometryComm* msg = scinew GeometryComm(0, id, obj, name, lock);
  save_msg(msg);
  dirty_ = true;
  turn_off_light();
  return id;
}

LightID
GeometryOPort::addLight(LightHandle obj, 
			const string& name, CrowdMonitor* lock)
{
  //static LightID next_id = 1;
  turn_on_light();
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
  turn_off_light();
  return id;
}



bool
GeometryOPort::direct_forward(GeometryComm* msg)
{
  if (outbox_.size() == 0) { return false; }

  // Send msg to last port directly, but copy to all of the prior ones.
  // Note that msg is not deleted if this function returns false.
  unsigned int i;
  for (i = 0; i < outbox_.size()-1; i++)
  {
    GeometryComm *cpy = scinew GeometryComm(*msg);
    outbox_[i]->send(cpy);
  }
  outbox_[i]->send(msg);

  return true;
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
  turn_on_light();

  for (unsigned int i=0; i < outbox_.size(); i++)
  {
    GeometryComm* msg = scinew GeometryComm(portid_[i], id);
    outbox_[i]->send(msg);
  }
  GeometryComm* msg = scinew GeometryComm(0, id);
  save_msg(msg);
  dirty_ = true;
  turn_off_light();
}

void
GeometryOPort::delLight(LightID id, int del)
{
  turn_on_light();

  for (unsigned int i=0; i < outbox_.size(); i++)
  {
    GeometryComm* msg = scinew GeometryComm(portid_[i], id);
    outbox_[i]->send(msg);
  }
  GeometryComm* msg = scinew GeometryComm(0, id);
  save_msg(msg);
  dirty_ = true;
  turn_off_light();
}



void
GeometryOPort::delAll()
{
  turn_on_light();
  for (unsigned int i = 0; i < outbox_.size(); i++)
  {
    GeometryComm* msg = scinew GeometryComm(MessageTypes::GeometryDelAll,
					    portid_[i]);
    outbox_[i]->send(msg);
  }
  GeometryComm* msg = scinew GeometryComm(MessageTypes::GeometryDelAll, 0);
  save_msg(msg);
  dirty_ = true;
  turn_off_light();
}



void
GeometryOPort::flushViews()
{
  turn_on_light();
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
  turn_off_light();
}

bool
GeometryOPort::get_view_bounds(BBox &bbox)
{

  GeometryData *data;
  data = getData(0, 0, GEOM_VIEW_BOUNDS);
  if (data) {
    bbox = data->view_bounds_;
    return true;
  }
  return false;
}


void
GeometryOPort::flushViewsAndWait()
{
  turn_on_light();
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
  turn_off_light();
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
  turn_on_light(Resetting);
  Module* mod = c->iport->get_module();
  outbox_.push_back(&mod->mailbox_);
  // Send the registration message.
  Mailbox<GeomReply> *tmp = new Mailbox<GeomReply>("Temporary GeometryOPort mailbox", 1);
  outbox_[which]->send(scinew GeometryComm(tmp));
  GeomReply reply = tmp->receive();
  portid_.push_back(reply.portid);
  turn_off_light();

  // Forward all of the queued up messages.
  turn_on_light();
  list<GeometryComm *>::iterator itr = saved_msgs_.begin();
  while (itr != saved_msgs_.end())
  {
    GeometryComm *msg = scinew GeometryComm(**itr);
    msg->portno = portid_[which];
    outbox_[which]->send(msg);
    ++itr;
  }
  turn_off_light();
}


void
GeometryOPort::detach(Connection* c, bool blocked)
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

  OPort::detach(c, blocked);
}


bool
GeometryOPort::have_data()
{
  return saved_msgs_.size();
}


void
GeometryOPort::resend(Connection*)
{
  //  cerr << "GeometryOPort can't resend and shouldn't need to!\n";
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
  if (which_viewer >= (int)outbox_.size() || which_viewer < 0) return 0;

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
  if (which_viewer >= (int)outbox_.size() || which_viewer < 0) return;

  GeometryComm* msg = scinew GeometryComm(MessageTypes::GeometrySetView,
					  portid_[which_viewer],
					  which_viewwindow,
					  view);
  outbox_[which_viewer]->send(msg);
}

//! The memory for reply is not owned by this.
GeometryComm::GeometryComm(Mailbox<GeomReply> *reply)
  : MessageBase(MessageTypes::GeometryInit),
    reply(reply)
{
}


GeometryComm::GeometryComm(int portno, GeomID serial, GeomHandle obj,
			   const string& name, CrowdMonitor* lock)
  : MessageBase(MessageTypes::GeometryAddObj),
    reply(0),
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
    reply(0),
    portno(portno),
    lserial(serial),
    light(light),
    name(name),
    lock(lock)
{
}


GeometryComm::GeometryComm(int portno, GeomID serial)
  : MessageBase(MessageTypes::GeometryDelObj),
    reply(0),
    portno(portno),
    serial(serial)
{
}

GeometryComm::GeometryComm(int portno, LightID serial)
  : MessageBase(MessageTypes::GeometryDelLight),
    reply(0),
    portno(portno),
    lserial(serial)
{
}


GeometryComm::GeometryComm(MessageTypes::MessageType type,
			   int portno, Semaphore* wait)
  : MessageBase(type),
    reply(0),
    portno(portno),
    wait(wait)
{
}


GeometryComm::GeometryComm(MessageTypes::MessageType type, int portno)
  : MessageBase(type),
    reply(0),
    portno(portno)
{
}


GeometryComm::GeometryComm(MessageTypes::MessageType type, int portno,
			   FutureValue<GeometryData*>* datareply,
			   int which_viewwindow, int datamask)
  : MessageBase(type),
    reply(0),
    portno(portno),
    which_viewwindow(which_viewwindow),
    datamask(datamask),
    datareply(datareply)
{
}


GeometryComm::GeometryComm(MessageTypes::MessageType type, int portno,
			   int which_viewwindow, View view)
  : MessageBase(type),
    reply(0),
    portno(portno),
    view(view),
    which_viewwindow(which_viewwindow)
{
}


GeometryComm::GeometryComm(MessageTypes::MessageType type, int portno,
			   FutureValue<int>* nreply)
  : MessageBase(type),
    reply(0),
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

GeometryData::~GeometryData()
{
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

