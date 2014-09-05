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
 *  Viewer.cc:  The Geometry Viewer
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Dataflow/Modules/Render/Viewer.h>
#include <Dataflow/Modules/Render/ViewWindow.h>
#include <Dataflow/Comm/MessageTypes.h>
#include <Dataflow/Network/Connection.h>
#include <Dataflow/Network/ModuleHelper.h>
#include <Core/Geom/GeomObj.h>
#include <Dataflow/Modules/Render/ViewGeom.h>
#include <Dataflow/Modules/Render/OpenGL.h>
#include <Core/Geom/HeadLight.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Thread/FutureValue.h>
#include <Core/Containers/StringUtil.h>

#include <iostream>
using std::cerr;
using std::endl;
using std::ostream;

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

namespace SCIRun {



//----------------------------------------------------------------------
DECLARE_MAKER(Viewer)
//----------------------------------------------------------------------
Viewer::Viewer(GuiContext* ctx)
  : Module("Viewer", ctx, ViewerSpecial,"Render","SCIRun"),
    geomlock_("Viewer geometry lock"),
    max_portno_(0)
{
				// Add a headlight
  lighting_.lights.add(scinew HeadLight("Headlight", Color(1,1,1)));
    
  default_material_ = scinew Material(Color(.1,.1,.1),
				      Color(.6,0,0),
				      Color(.7,.7,.7),
				      50);
  have_own_dispatch=true;

				// Create port 0 - we use this for
				// global objects such as cameras,
				// light source icons, etc.
  int portid=max_portno_++;
    
				// Create the port
  GeomViewerPort *pi = new GeomViewerPort(portid);

  ports_.addObj(pi,portid);
}


//----------------------------------------------------------------------
Viewer::~Viewer()
{
}


//----------------------------------------------------------------------
void Viewer::do_execute()
{
  for(;;)
  {
    if(mailbox.numItems() == 0)
    {
      // See if anything needs to be redrawn...
      int did_some=1;
      while(did_some)
      {
	did_some=0;
	for(unsigned int i=0;i<view_window_.size();i++)
	{
	  if (view_window_[i]->need_redraw)
	  {
	    did_some++;
	    view_window_[i]->redraw_if_needed();
	  }
	}
      }
    }
    if (process_event() == 86)
    {
      //for(unsigned int i=0;i<view_window_.size();i++)
      //{
      //	View_Window_* r=view_window_[i];
      //if (r && r->current_renderer)
      //{
      //r->current_renderer->kill_helper();
      //	}
      //      }
      
      helper_done.send(1);
      return;
    }
  }
}

//----------------------------------------------------------------------
int Viewer::process_event()
{
  MessageBase* msg=mailbox.receive();
  GeometryComm* gmsg=(GeometryComm*)msg;
  switch(msg->type)
  {
  case MessageTypes::GoAway:
    return 86;

  case MessageTypes::ExecuteModule:
    // We ignore these messages...
    break;

  case MessageTypes::ViewWindowRedraw:
    {
      ViewerMessage* rmsg=(ViewerMessage*)msg;
      ViewWindow* r=0;
      unsigned int i;
      for(i=0;i<view_window_.size();i++)
      {
	r=view_window_[i];
	if(r->id == rmsg->rid)
	  break;
      }
      if(i==view_window_.size())
      {
	warning("ViewWindow not found for redraw! (id=" + rmsg->rid + ").");
      }
      else if(rmsg->nframes == 0)
      {
	// Normal redraw (lazy)
	r->need_redraw=1;
      }
      else
      {
	// Do animation...
	r->redraw(rmsg->tbeg, rmsg->tend, rmsg->nframes,
		  rmsg->framerate);
      }
    }
    break;

  case MessageTypes::ViewWindowDumpImage:
    {
      ViewerMessage* rmsg=(ViewerMessage*)msg;
      for(unsigned int i=0;i<view_window_.size();i++)
      {
	ViewWindow* r=view_window_[i];
	if(r->id == rmsg->rid)
	{
	  r->current_renderer->saveImage(rmsg->filename, rmsg->format, 
					 rmsg->resx,rmsg->resy);
	  break;
	}
      }
    }
    break;

  case MessageTypes::ViewWindowDumpObjects:
    {
      geomlock_.readLock();
      ViewerMessage* rmsg=(ViewerMessage*)msg;
      for(unsigned int i=0;i<view_window_.size();i++)
      {
	ViewWindow* r=view_window_[i];
	if(r->id == rmsg->rid)
	{
	  r->dump_objects(rmsg->filename, rmsg->format);
	  break;
	}
      }
      geomlock_.readUnlock();
    }
    break;

  case MessageTypes::ViewWindowMouse:
    {
      ViewWindowMouseMessage* rmsg=(ViewWindowMouseMessage*)msg;
      for(unsigned int i=0;i<view_window_.size();i++)
      {
	ViewWindow* r=view_window_[i];
	if(r->id == rmsg->rid)
	{
	  (r->*(rmsg->handler))(rmsg->action, rmsg->x, rmsg->y, rmsg->state, rmsg->btn, rmsg->time);
	  break;
	}
      }
    }
    break;

  case MessageTypes::GeometryInit:
    geomlock_.writeLock();
    initPort(gmsg->reply);
    geomlock_.writeUnlock();
    break;	

  case MessageTypes::GeometryAddObj:
  case MessageTypes::GeometryDelObj:
  case MessageTypes::GeometryDelAll:
    append_port_msg(gmsg);
    msg=0; // Don't delete it yet...
    break;

  case MessageTypes::GeometryFlush:
    geomlock_.writeLock();
    flushPort(gmsg->portno);
    geomlock_.writeUnlock();
    break;

  case MessageTypes::GeometryFlushViews:
    geomlock_.writeLock();
    flushPort(gmsg->portno);
    geomlock_.writeUnlock();
    flushViews();
    if(gmsg->wait)
    {
      // Synchronized redraw - do it now and signal them...
      for(unsigned int i=0;i<view_window_.size();i++)
	view_window_[i]->redraw_if_needed();
      gmsg->wait->up();
    }
    break;

  case MessageTypes::GeometryGetNViewWindows:
    gmsg->nreply->send(view_window_.size());
    break;

  case MessageTypes::GeometryGetData:
    if((unsigned int)(gmsg->which_viewwindow) >= view_window_.size())
    {
      gmsg->datareply->send(0);
    }
    else
    {
      view_window_[gmsg->which_viewwindow]->
	getData(gmsg->datamask, gmsg->datareply);
    }
    break;

  case MessageTypes::GeometrySetView:
    if((unsigned int)(gmsg->which_viewwindow) < view_window_.size())
    {
      view_window_[gmsg->which_viewwindow]->setView(gmsg->view);
    }
    break;

#if 0
  case MessageTypes::TrackerMoved:
    {
      TrackerMessage* tmsg=(TrackerMessage*)msg;
      ViewWindow* view_window_=(ViewWindow*)tmsg->clientdata;
      if(tmsg->data.head_moved)
	view_window_->head_moved(tmsg->data.head_pos);
      if(tmsg->data.mouse_moved)
	view_window_->flyingmouse_moved(tmsg->data.mouse_pos);
    }
    break;
#endif

  default:
    error("Illegal Message type: " + to_string(msg->type));
    break;
  }

  if(msg) { delete msg; }

  return 1;
}

//----------------------------------------------------------------------
void Viewer::initPort(Mailbox<GeomReply>* reply)
{
  int portid=max_portno_++;
				// Create the port
  // PortInfo* pi=scinew PortInfo;
  GeomViewerPort *pi = new GeomViewerPort(portid);
  ports_.addObj(pi,portid);
  reply->send(GeomReply(portid));
}

//----------------------------------------------------------------------
void Viewer::flushViews()
{
  for (unsigned int i=0; i<view_window_.size(); i++)
    view_window_[i]->force_redraw();
}

//----------------------------------------------------------------------
void Viewer::addObj(GeomViewerPort* port, int serial, GeomObj *obj,
		    const string& name, CrowdMonitor* lock)
{
  string pname(name+" ("+to_string(port->portno)+")");
  GeomViewerItem* si = scinew GeomViewerItem(obj, pname, lock);
  port->addObj(si,serial);
  // port->objs->insert(serial, si);
  for (unsigned int i=0; i<view_window_.size(); i++)
  {
    view_window_[i]->itemAdded(si);
  }
}

//----------------------------------------------------------------------
void Viewer::delObj(GeomViewerPort* port, int serial, int del)
{
  GeomViewerItem* si;
  if ((si = ((GeomViewerItem*)port->getObj(serial))))
  {
    for (unsigned int i=0; i<view_window_.size(); i++)
      view_window_[i]->itemDeleted(si);
    port->delObj(serial, del);
  }
  else
  {
    error("Error deleting object, object not in database...(serial=" +
	  to_string(serial));
  }
}

//----------------------------------------------------------------------
void Viewer::delAll(GeomViewerPort* port)
{
  GeomIndexedGroup::IterIntGeomObj iter = port->getIter();
  
  for ( ; iter.first != iter.second; iter.first++)
  {
    GeomViewerItem* si =
      (GeomViewerItem*)((*iter.first).second);
    for (unsigned int i=0; i<view_window_.size(); i++)
      view_window_[i]->itemDeleted(si);
  }
  
  port->delAll();
}

void Viewer::addTopViewWindow(ViewWindow *r)
{
  top_view_window_.push_back(r);
}

void Viewer::delTopViewWindow(ViewWindow *r)
{
  for (unsigned int i=0; i<top_view_window_.size(); i++)
  {
    if (r==top_view_window_[i])
    {
      top_view_window_.erase(top_view_window_.begin()+i);
    }
  }
} 

//----------------------------------------------------------------------
#ifdef OLDUI
void Viewer::spawnIndCB(CallbackData*, void*)
{
  top_view_window_.push_back(scinew ViewWindow(this));
  top_view_window_[top_view_window_.size()-1]->SetTop();
  GeomItem *item;
  for (int i=0; i<top_view_window_[0]->geomItemA.size(); i++)
  {
    item=top_view_window_[0]->geomItemA[i];
    top_view_window_[top_view_window_.size()-1]->itemAdded(item->geom, item->name);
  }
  //  printFamilyTree();
}
#endif

//----------------------------------------------------------------------
//void Viewer::connection(ConnectionMode mode, int which_port, int)
//{
//    if(mode==Disconnected){
//	remove_iport(which_port);
//    } else {
//	add_iport(scinew GeometryIPort(this, "Geometry", GeometryIPort::Atomic));
//    }
//}
//----------------------------------------------------------------------
void Viewer::delete_viewwindow(ViewWindow* delviewwindow)
{
  for(unsigned int i=0;i<view_window_.size();i++)
  {
    if(view_window_[i] == delviewwindow)
    {
      view_window_.erase(view_window_.begin() + i);
      delete delviewwindow;
    }
  }
}

//----------------------------------------------------------------------
void Viewer::tcl_command(GuiArgs& args, void* userdata)
{
  if(args.count() < 2)
  {
    args.error("Viewer needs a minor command");
    return;
  }
  if(args[1] == "addviewwindow")
  {
    if(args.count() != 3)
    {
      args.error("addviewwindow must have a RID");
      return;
    }
    view_window_.push_back(scinew ViewWindow(this, gui, gui->createContext(args[2])));
  }
  else
  {
    Module::tcl_command(args, userdata);
  }
}
//----------------------------------------------------------------------
void Viewer::execute()
{
  // Never gets called...
}

//----------------------------------------------------------------------
ViewerMessage::ViewerMessage(const string& rid)
  : MessageBase(MessageTypes::ViewWindowRedraw), rid(rid), nframes(0)
{
}

//----------------------------------------------------------------------
ViewerMessage::ViewerMessage(const string& rid, double tbeg, double tend,
			     int nframes, double framerate)
  : MessageBase(MessageTypes::ViewWindowRedraw),
    rid(rid),
    tbeg(tbeg),
    tend(tend),
    nframes(nframes),
    framerate(framerate)
{
  if (nframes <= 0)
  {
    std::cerr << "nframes shouldn't be zero for animation.\n";
    nframes = 1;
  }
}

//----------------------------------------------------------------------
ViewerMessage::ViewerMessage(MessageTypes::MessageType type,
			     const string& rid, const string& filename)
  : MessageBase(type), rid(rid), filename(filename)
{
}

//----------------------------------------------------------------------
ViewerMessage::ViewerMessage(MessageTypes::MessageType type,
			     const string& rid,
			     const string& filename,
			     const string& format,
			     const string& resx_string,
			     const string& resy_string)
  : MessageBase(type), rid(rid), filename(filename), format(format)
{
  resx = atoi(resx_string.c_str());
  resy = atoi(resy_string.c_str());
}

//----------------------------------------------------------------------
ViewerMessage::~ViewerMessage()
{
}


//----------------------------------------------------------------------
void Viewer::append_port_msg(GeometryComm* gmsg)
{
				// Look up the right port number
  // PortInfo* pi;
  GeomViewerPort *pi;
  
  if (!(pi = ((GeomViewerPort*)ports_.getObj(gmsg->portno))))
  {
    warning("Geometry message sent to bad port!!!: " +
	    to_string(gmsg->portno));
    return;
  }
  
				// Queue up the messages until the
				// flush...
  if(pi->msg_tail)
  {
    pi->msg_tail->next=gmsg;
    pi->msg_tail=gmsg;
  } else
  {
    pi->msg_head=pi->msg_tail=gmsg;
  }
  gmsg->next=0;
}

//----------------------------------------------------------------------
void Viewer::flushPort(int portid)
{
  // Look up the right port number
  GeomViewerPort* pi;
  if(!(pi = ((GeomViewerPort*)ports_.getObj(portid))))
  {
    warning("Geometry message sent to bad port!!!\n");
    return;
  }
  GeometryComm* gmsg=pi->msg_head;
  while(gmsg)
  {
    switch(gmsg->type)
    {
    case MessageTypes::GeometryAddObj:
      addObj(pi, gmsg->serial, gmsg->obj, gmsg->name, gmsg->lock);
      break;
    case MessageTypes::GeometryDelObj:
      delObj(pi, gmsg->serial, gmsg->del);
      break;
    case MessageTypes::GeometryDelAll:
      delAll(pi);
      break;
    default:
      error("How did this message get in here???");
    }
    GeometryComm* next=gmsg->next;
    delete gmsg;
    gmsg=next;
  }
  if(pi->msg_head)
  {
    flushViews();
    pi->msg_head=pi->msg_tail=0;
  }
}


//----------------------------------------------------------------------
void Viewer::emit_vars(ostream& out, const string& midx)
{
  ctx->emit(out, midx);
  string viewwindowstr;
  for(unsigned int i=0;i<view_window_.size();i++)
  {
    viewwindowstr=midx+string("-ViewWindow_")+to_string(i);
    out << midx << " ui\n";
    view_window_[i]->emit_vars(out, viewwindowstr);
  }
}

} // End namespace SCIRun
