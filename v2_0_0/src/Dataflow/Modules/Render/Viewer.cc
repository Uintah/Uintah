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
#include <Core/Geom/DirectionalLight.h>
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
    // CollabVis code begin
#ifdef HAVE_COLLAB_VIS
    newViewWindowMailbox( "NewViewWindowMailbox", 10 ),
#endif
    // CollabVis code end
    max_portno_(0)
{

  map<LightID, int> li;
  // Add a headlight
  lighting_.lights.add(scinew HeadLight("Headlight", Color(1,1,1)));
  li[0] = 0;
  for(int i = 1; i < 4; i++){ // only set up 3 more lights
    char l[8];
    sprintf( l, "Light%d", i );
    lighting_.lights.add(scinew DirectionalLight(string(l), 
						       Vector(0,0,1),
						       Color(1,1,1), 
						       false, false));
    li[i] = i;
  }
  pli_[0] = li;

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
 
  case MessageTypes::ViewWindowEditLight:
    {
      ViewerMessage* rmsg=(ViewerMessage*)msg;
      for(unsigned int i=0;i<view_window_.size();i++)
      {
	ViewWindow* r=view_window_[i];
	if(r->id == rmsg->rid)
	{
	  ((lighting_.lights)[rmsg->lightNo])->on = rmsg->on;
	  if( rmsg->on ){
	    if(DirectionalLight *dl = dynamic_cast<DirectionalLight *>
	       (((lighting_.lights)[rmsg->lightNo]).get_rep())) {
	      dl->move( rmsg->lightDir );
	      dl->setColor( rmsg->lightColor );
	    }
	  }
	  r->need_redraw = 1;
	  break;
	}
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

  case MessageTypes::GeometryAddLight:
    {
      // Add a light to the light list
      lighting_.lights.add(gmsg->light);

      // Now associate the port, and LightID to the index
      map<LightID, int> li;
      map<int, map<LightID, int> >::iterator it = 
	pli_.find( gmsg->portno );
      if( it == pli_.end() ){
	li[gmsg->lserial] =  lighting_.lights.size() - 1;
	pli_[ gmsg->portno ] = li;
      } else {
	((*it).second)[gmsg->lserial] = lighting_.lights.size() - 1;
      }
      break;
    }
    break;
  case MessageTypes::GeometryDelLight:
    {
      map<LightID, int>::iterator li;
      map<int, map<LightID, int> >::iterator it = 
	pli_.find( gmsg->portno );
      if( it == pli_.end() ){
	error("Error while deleting a light: no data base for port number " +
	      to_string( gmsg->portno ) );
      } else {
	li = ((*it).second).find(gmsg->lserial);
	if( li == (*it).second.end() ){
	  error("Error while deleting a light: no light with id " +
		to_string(gmsg->lserial) + "in database for port number" +
		to_string( gmsg->portno));
	} else {
	  int idx = (*li).second;
	  int i;
	  for(i = 0; i < lighting_.lights.size(); i++){
	    if( i == idx ){
	      lighting_.lights[i] = 0;
	      lighting_.lights.remove(i);
	      ((*it).second).erase( li );
	      break;
	    }
	    if( i == lighting_.lights.size() )
	      error("Error deleting light, light not in database...(lserial=" +
		    to_string(gmsg->lserial));
	  }
	}
      }
    }
    break;
  
  case MessageTypes::GeometryInit:
    geomlock_.writeLock();
    initPort(gmsg->reply);
    geomlock_.writeUnlock();
    break;

  case MessageTypes::GeometryDetach:
    geomlock_.writeLock();
    detachPort(gmsg->portno);
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
  portno_map_.push_back(portid);
  syncronized_map_.push_back(false);

  // Create the port
  GeomViewerPort *pi = new GeomViewerPort(portid);
  ports_.addObj(pi,portid);
  reply->send(GeomReply(portid));
}

//----------------------------------------------------------------------
int Viewer::real_portno(int portno)
{
  for (unsigned int i=0; i < portno_map_.size(); i++)
  {
    if (portno == portno_map_[i])
    {
      return i + 1;
    }
  }
  ASSERTFAIL("PORTNO NOT FOUND");
  return 0;
}

//----------------------------------------------------------------------
void Viewer::delete_patch_portnos(int portid)
{
  int found = -1;
  for (unsigned int i=0; i < portno_map_.size(); i++)
  {
    if (found >= 0)
    {
      GeomViewerPort *pi;
      if(!(pi = ((GeomViewerPort*)ports_.getObj(portno_map_[i]).get_rep())))
      {
	warning("Geometry message sent to bad port!!!\n");
	continue;
      }
      GeomIndexedGroup::IterIntGeomObj iter = pi->getIter();
      for (; iter.first != iter.second; iter.first++)
      {
	GeomViewerItem* si = 
	  dynamic_cast<GeomViewerItem*>((*iter.first).second.get_rep());
	if (si)
	{
	  const string::size_type loc = si->getString().find_last_of('(');
	  string newname =
	    si->getString().substr(0, loc+1) + to_string(i) + ")";

	  // Do a rename here.
	  for (unsigned int j = 0; j < view_window_.size(); j++)
	  {
	    view_window_[j]->itemRenamed(si, newname);
	  }
	}
      }
    }
    else if (portid == portno_map_[i])
    {
      found = i;
    }
  }

  if (found >= 0)
  {
    portno_map_.erase(portno_map_.begin() + found);
    syncronized_map_.erase(syncronized_map_.begin() + found);
  }

}

//----------------------------------------------------------------------
void Viewer::detachPort(int portid)
{
  GeomViewerPort* pi;
  if(!(pi = ((GeomViewerPort*)ports_.getObj(portid).get_rep())))
  {
    warning("Geometry message sent to bad port!!!\n");
    return;
  }
  delAll(pi);
  ports_.delObj(portid);

  delete_patch_portnos(portid);

  flushViews();
}

//----------------------------------------------------------------------
void Viewer::flushViews()
{
  for (unsigned int i=0; i<view_window_.size(); i++)
    view_window_[i]->force_redraw();
}


//----------------------------------------------------------------------
void Viewer::addObj(GeomViewerPort* port, int serial, GeomHandle obj,
		    const string& name, CrowdMonitor* lock)
{
  string pname(name + " ("+to_string(real_portno(port->portno))+")");
  GeomViewerItem* si = scinew GeomViewerItem(obj, pname, lock);
  port->addObj(si,serial);
  // port->objs->insert(serial, si);
  for (unsigned int i=0; i<view_window_.size(); i++)
  {
    view_window_[i]->itemAdded(si);
  }
}

//----------------------------------------------------------------------
void Viewer::delObj(GeomViewerPort* port, int serial)
{
  GeomViewerItem* si;
  if ((si = ((GeomViewerItem*)port->getObj(serial).get_rep())))
  {
    for (unsigned int i=0; i<view_window_.size(); i++)
      view_window_[i]->itemDeleted(si);
    port->delObj(serial);
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
      (GeomViewerItem*)((*iter.first).second.get_rep());
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
    // CollabVis code begin
#ifdef HAVE_COLLAB_VIS
    //std::cerr << "[HAVE_COLLAB_VIS] (Viewer::tcl_command) 0\n";
    if ( args.count() == 4 ) {
      ViewWindow* r=scinew ViewWindow(this, gui, gui->createContext(args[2]));
      view_window_.push_back(r);
      newViewWindowMailbox.send(r);
      return;
    }
#endif
    // CollabVis code end
    
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

ViewerMessage::ViewerMessage(MessageTypes::MessageType type,
			     const string& rid, int lightNo, 
			     bool on, const Vector& dir,
			     const Color& color)
  : MessageBase(type), rid(rid), lightDir(dir), lightColor(color), lightNo(lightNo), on(on)
{}


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
  
  if (!(pi = ((GeomViewerPort*)ports_.getObj(gmsg->portno).get_rep())))
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
  if(!(pi = ((GeomViewerPort*)ports_.getObj(portid).get_rep())))
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
      delObj(pi, gmsg->serial);
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

  syncronized_map_[real_portno(portid)-1] = true;
  bool all = true;
  
  for (unsigned int i=0; i+1 < (unsigned int)numIPorts(); i++)
  {
    if (syncronized_map_[i] == false)
    {
      all = false;
      break;
    }
  }
  if (all)
  {
    if (getenv("SCI_REGRESSION_TESTING"))
    {
      geomlock_.writeUnlock();
      for (unsigned int i = 0; i < view_window_.size(); i++)
      {
	const string name = string("snapshot") + to_string(i) + ".ppm";
	view_window_[i]->redraw_if_needed();
	view_window_[i]->current_renderer->saveImage(name, "ppm", 640, 512);
      }
      geomlock_.writeLock();
      flushViews();
    }

    for (unsigned int i = 0; i < syncronized_map_.size(); i++)
    {
      syncronized_map_[i] = false;
    }
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
