/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
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
#include <Dataflow/Network/Scheduler.h>
#include <Core/Geom/GeomObj.h>
#include <Dataflow/Modules/Render/ViewGeom.h>
#include <Dataflow/Modules/Render/OpenGL.h>
#include <Core/Geom/HeadLight.h>
#include <Core/Geom/DirectionalLight.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Thread/FutureValue.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Util/Environment.h>
#include <Core/Math/MiscMath.h>
#include <Core/Thread/CleanupManager.h>
#include <iostream>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

using std::cerr;
using std::endl;
using std::ostream;

namespace SCIRun {


#ifdef __linux
// This is a workaround for an unusual crash on exit bug on newer
// linux systems.  It appears that if a viewer is created at SCIRun
// start from within a command line network that SCIRun will crash on
// exit in the tcl thread unless the viewer is deleted first.  So we
// just add them to the cleanup manager and have it destroy the
// viewers before it goes away.  Once the bug is fixed we should
// revert this back to just deleting the viewwindow directly.
static void delete_viewwindow_callback(void *vwptr)
{
  if (sci_getenv("SCI_REGRESSION_TESTING")) return;
  ViewWindow *vw = (ViewWindow *)vwptr;
  delete vw;
}
#endif

//----------------------------------------------------------------------
DECLARE_MAKER(Viewer)
//----------------------------------------------------------------------

Viewer::Viewer(GuiContext* ctx)
  : Module("Viewer", ctx, ViewerSpecial,"Render","SCIRun"),
    geomlock_("Viewer geometry lock"), 
    view_window_lock_("Viewer view window lock"),
    max_portno_(0),
    stop_rendering_(false),
    synchronized_debt_(0)
{

  map<LightID, int> li;
  // Add a headlight
  lighting_.lights.add(scinew HeadLight("Headlight", Color(1,1,1)));
  li[0] = 0;
  for(int i = 1; i < 4; i++){ // only set up 3 more lights
    char l[8];
    sprintf(l, "Light%d", i);
    lighting_.lights.add
      (scinew DirectionalLight(l, Vector(0,0,1), Color(1,1,1), false, false));
    li[i] = i;
  }
  pli_[0] = li;

  default_material_ =
    scinew Material(Color(.1,.1,.1), Color(.6,0,0), Color(.7,.7,.7), 50);
  have_own_dispatch=true;

  // Create port 0 - we use this for global objects such as cameras,
  // light source icons, etc.
  ports_.addObj(new GeomViewerPort(0),0);
  max_portno_ = 1;
}

//----------------------------------------------------------------------
Viewer::~Viewer()
{
  for(unsigned int i=0;i<view_window_.size();i++)
  {
    view_window_lock_.lock();
#ifdef __linux    
    CleanupManager::invoke_remove_callback(delete_viewwindow_callback,
                                           (void *)view_window_[i]);
#else
    delete view_window_[i];
#endif
    view_window_lock_.unlock();
  }

}

//----------------------------------------------------------------------
void
Viewer::do_execute()
{
  for(;;)
  {
    if(!stop_rendering_ && mailbox.numItems() == 0)
    {
      // See if anything needs to be redrawn.
      int did_some=1;
      while(did_some)
      {
	did_some=0;
	for(unsigned int i=0;i<view_window_.size();i++)
	{
	  if (view_window_[i]) {
	    if (view_window_[i]->need_redraw_)
	    {
	      did_some++;
	      view_window_[i]->redraw_if_needed();
	    }
	  }
	}
      }
    }
    if (process_event() == 86)  
    {
      // Doesn't this get handled in the destructor of the viewwindow?
      for(unsigned int i=0;i<view_window_.size();i++)
      {
      	ViewWindow* r=view_window_[i];
	if (r && r->renderer_)
	{
	  r->renderer_->kill_helper();
	  r->viewer_ = 0;
      	}
      }
      return;
    }
  }
}


//----------------------------------------------------------------------
int 
Viewer::process_event()
{
  MessageBase* msg=mailbox.receive();
  GeometryComm* gmsg=(GeometryComm*)msg;
  switch(msg->type)
  {
  case MessageTypes::GoAway:
    return 86;

  case MessageTypes::GoAwayWarn:
    stop_rendering_ = true;
    //stop spinning windows.
    for(unsigned i = 0; i < view_window_.size(); i++) {
      view_window_[i]->inertia_mode_ = 0;
    }    
    break;

  case MessageTypes::ExecuteModule:
    if (synchronized_debt_ < 0)
    {
      synchronized_debt_++;
      sched->report_execution_finished(msg);
    }
    else
    {
      Scheduler_Module_Message *smmsg = (Scheduler_Module_Message *)msg;
      synchronized_serials_.push_back(smmsg->serial);
    }
    break;

  case MessageTypes::SynchronizeModule:
    // We (mostly) ignore these messages.
    sched->report_execution_finished(msg);
    break;

  case MessageTypes::ViewWindowRedraw:
    {
      ViewerMessage* rmsg=(ViewerMessage*)msg;
      ViewWindow* r=0;
      unsigned int i;
      for(i=0;i<view_window_.size();i++)
      {
	r=view_window_[i];
	if(r->id_ == rmsg->rid)
	  break;
      }
      if(i==view_window_.size())
      {
	warning("ViewWindow not found for redraw! (id=" + rmsg->rid + ").");
      }
      else if(rmsg->nframes == 0)
      {
	// Normal redraw (lazy)
	r->need_redraw_=1;
      }
      else
      {
	// Do animation.
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
	if(r->id_ == rmsg->rid)
	{
	  ((lighting_.lights)[rmsg->lightNo])->on = rmsg->on;
	  if( rmsg->on ){
	    if(DirectionalLight *dl = dynamic_cast<DirectionalLight *>
	       (((lighting_.lights)[rmsg->lightNo]).get_rep())) {
	      dl->move( rmsg->lightDir );
	      dl->setColor( rmsg->lightColor );
	    } else if( HeadLight *hl = dynamic_cast<HeadLight *>
		       (((lighting_.lights)[rmsg->lightNo]).get_rep())) {
	      hl->setColor( rmsg->lightColor );
	    }
	  }
	  r->need_redraw_ = 1;
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
	if(r->id_ == rmsg->rid)
	{
	  r->renderer_->saveImage(rmsg->filename, rmsg->format, 
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
	if(r->id_ == rmsg->rid)
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
      float NX = 1.0, NY = 1.0;
      bool tracking = false;
      for(unsigned int i=0;i<view_window_.size();i++)
      {
	ViewWindow* r=view_window_[i];
	if(r->id_ == rmsg->rid)
	{
	  (r->*(rmsg->handler))(rmsg->action, rmsg->x, rmsg->y, 
				rmsg->state, rmsg->btn, rmsg->time);
	  if (i == 0) {
	    tracking = true;
	    r->NormalizeMouseXY(rmsg->x, rmsg->y, &NX, &NY);
	  }
	}
      }
      if (tracking) {
	for(unsigned int i=0;i<view_window_.size();i++)
	{
	  ViewWindow* r=view_window_[i];
	  r->gui_track_view_window_0_.reset();
	  if(r->id_ != rmsg->rid && r->gui_track_view_window_0_.get())
	  {
	    int xx,yy;
	    r->UnNormalizeMouseXY(NX,NY,&xx,&yy);
	    (r->*(rmsg->handler))(rmsg->action, xx, yy, 
				  rmsg->state, rmsg->btn, rmsg->time);
	  }
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
	      to_string(gmsg->portno));
      } else {
	li = ((*it).second).find(gmsg->lserial);
	if( li == (*it).second.end() ){
	  error("Error while deleting a light: no light with id " +
		to_string(gmsg->lserial) + "in database for port number" +
		to_string(gmsg->portno));
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
	      error("Error deleting light, light not in database.(lserial=" +
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
    msg=0; // Don't delete it yet.
    break;

  case MessageTypes::GeometryFlush:
    geomlock_.writeLock();
    flushPort(gmsg->portno);
    geomlock_.writeUnlock();
    break;

  case MessageTypes::GeometrySynchronize:
    // Port finish message.  Synchronize viewer.
    //geomlock_.writeLock();
    finishPort(gmsg->portno);
    //geomlock_.writeUnlock();
    break;

  case MessageTypes::GeometryFlushViews:
    geomlock_.writeLock();
    flushPort(gmsg->portno);
    geomlock_.writeUnlock();
    flushViews();
    if(gmsg->wait)
    {
      // Synchronized redraw - do it now and signal them.
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


  default:
    error("Illegal Message type: " + to_string(msg->type));
    break;
  }

  if(msg) { delete msg; }

  return 1;
}

//----------------------------------------------------------------------
void
Viewer::initPort(Mailbox<GeomReply>* reply)
{
  int portid=max_portno_++;
  portno_map_.push_back(portid);
  synchronized_map_.push_back(0);
  ports_.addObj(new GeomViewerPort(portid), portid);   // Create the port
  reply->send(GeomReply(portid));
}

//----------------------------------------------------------------------
int
Viewer::real_portno(int portno)
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
void
Viewer::delete_patch_portnos(int portid)
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
	  string cached_name = si->getString();
	  // Do a rename here.
	  for (unsigned int j = 0; j < view_window_.size(); j++)
	  {
	    // itemRenamed will set si->name_ = newname
	    // so we need to reset it for other windows
	    si->getString() = cached_name;
	    view_window_[j]->itemRenamed(si, newname);
	  }
	  si->getString() = newname;
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
    synchronized_map_.erase(synchronized_map_.begin() + found);
  }

}

//----------------------------------------------------------------------
void
Viewer::detachPort(int portid)
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
void
Viewer::flushViews()
{
  for (unsigned int i=0; i<view_window_.size(); i++)
    view_window_[i]->need_redraw_ = 1;
}


//----------------------------------------------------------------------
void
Viewer::addObj(GeomViewerPort* port, int serial, GeomHandle obj,
		    const string& name, CrowdMonitor* lock)
{
  string pname(name + " ("+to_string(real_portno(port->portno))+")");
  GeomViewerItem* si = scinew GeomViewerItem(obj, pname, lock);
  port->addObj(si,serial);
  for (unsigned int i=0; i<view_window_.size(); i++)
  {
    view_window_[i]->itemAdded(si);
  }
}

//----------------------------------------------------------------------
void
Viewer::delObj(GeomViewerPort* port, int serial)
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
    error("Error deleting object, object not in database.(serial=" +
	  to_string(serial) + ")" );
  }
}

//----------------------------------------------------------------------
void
Viewer::delAll(GeomViewerPort* port)
{
  GeomIndexedGroup::IterIntGeomObj iter = port->getIter();
  if (!stop_rendering_) {
    for ( ; iter.first != iter.second; iter.first++)
    {
      GeomViewerItem* si = (GeomViewerItem*)((*iter.first).second.get_rep());
      for (unsigned int i=0; i<view_window_.size(); i++)
	view_window_[i]->itemDeleted(si);
    }
  }
  port->delAll();
}


//----------------------------------------------------------------------
void
Viewer::delete_viewwindow(const string &id)
{
  for(unsigned int i=0;i<view_window_.size();i++)
  {
    if(view_window_[i]->id_ == id)
    {
      view_window_lock_.lock();
#ifdef __linux
      CleanupManager::invoke_remove_callback(delete_viewwindow_callback,
                                             (void *)view_window_[i]);
#else
      delete view_window_[i];
#endif
      view_window_.erase(view_window_.begin() + i);
      view_window_lock_.unlock();
      return;
    }
  }
  cerr << "ERROR in delete_viewwindow, cannot find ID: " << id << std::endl;
}

//----------------------------------------------------------------------
void
Viewer::tcl_command(GuiArgs& args, void* userdata)
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
      args.error(args[1]+" must have a RID");
      return;
    }
    view_window_lock_.lock();
    ViewWindow* r=scinew ViewWindow(this, gui, gui->createContext(args[2]));
#ifdef __linux
    CleanupManager::add_callback(delete_viewwindow_callback, (void *)r);
#endif
    view_window_.push_back(r);
    view_window_lock_.unlock();
  } else if (args[1] == "deleteviewwindow") {
    if(args.count() != 3)
    {
      args.error(args[1]+" must have a RID");
      return;
    }
    gui->unlock();
    delete_viewwindow(args[2]);
    gui->lock();
  } else {
    Module::tcl_command(args, userdata);
  }

}
//----------------------------------------------------------------------
void
Viewer::execute()
{
  // Never gets called.
  ASSERTFAIL("Viewer::execute() should not ever be called.");
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
  : MessageBase(type), rid(rid), lightDir(dir), lightColor(color), 
    lightNo(lightNo), on(on)
{}


//----------------------------------------------------------------------
ViewerMessage::~ViewerMessage()
{
}


//----------------------------------------------------------------------
void
Viewer::append_port_msg(GeometryComm* gmsg)
{
  // Look up the right port number.
  GeomViewerPort *pi;
  if (!(pi = ((GeomViewerPort*)ports_.getObj(gmsg->portno).get_rep())))
  {
    warning("Geometry message sent to bad port!!!: "+to_string(gmsg->portno));
    return;
  }
  
  // Queue up the messages until the flush.
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
void
Viewer::flushPort(int portid)
{
  // Look up the right port number.
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
}


//----------------------------------------------------------------------
void
Viewer::finishPort(int portid)
{
  synchronized_map_[real_portno(portid)-1]++;

#if 0
  // Debugging junk.  Whole block not needed.
  if (synchronized_serials_.size())
  {
    unsigned int serial = synchronized_serials_.front();
    cout << "   finishPort " << real_portno(portid) <<
      " (" << serial << ")   :";
    for (unsigned int k = 0; k < synchronized_map_.size(); k++)
    {
      cout << " " << synchronized_map_[k];
    }
    cout << "\n";
  }
  else
  {
    cout << "   finishPort " << real_portno(portid) << " (...)   :";
    for (unsigned int k = 0; k < synchronized_map_.size(); k++)
    {
      cout << " " << synchronized_map_[k];
    }
    cout << "\n";
  }
#endif

  bool all = true;
  for (unsigned int i=0; i < synchronized_map_.size(); i++)
  {
    if (synchronized_map_[i] == 0)
    {
      all = false;
      break;
    }
  }
  
  if (all)
  {
    // Clear the entries from the map.
    for (unsigned int i = 0; i < synchronized_map_.size(); i++)
    {
      synchronized_map_[i]--;
    }

    // Push the serial number back to the scheduler.
    if (synchronized_serials_.size())
    {
      unsigned int serial = synchronized_serials_.front();
      synchronized_serials_.pop_front();

#if 0      
      // Debugging.
      cout << " Finished, sending " << serial << "   :";
      for (unsigned int k = 0; k < synchronized_map_.size(); k++)
      {
        cout << " " << synchronized_map_[k]-1;
      }
      cout << "\n";
#endif
      
      sched->report_execution_finished(serial);

#if 0
      // This turns on synchronous movie making.  It's very useful for
      // making movies that are driven by an event loop (such as
      // send-intermediate).  The movie frames are not taken during
      // user interaction but only on execute.  This is only
      // synchronous in one direction.  If a module executes too fast
      // then the latest one is used.  That is, it waits for the slow
      // modules but does not throttle the fast ones.
      char buff[1024];
      static int moviecounter = 0;
      sprintf(buff, "movie%04d.ppm", moviecounter++);
      view_window_[0]->redraw_if_needed();
      view_window_[0]->renderer_->saveImage(buff, "ppm", 640, 480);
      view_window_[0]->redraw();
#endif

    }
    else
    {
      // Serial number hasn't arrived yet, defer until we get it.
      synchronized_debt_--;
    }
  }
}


void
Viewer::set_context(Scheduler* sched, Network* network)
{
  Module::set_context(sched, network);
  if (sci_getenv("SCI_REGRESSION_TESTING"))
  {
    sched->add_callback(save_image_callback, this, -1);
  }
}


bool
Viewer::save_image_callback(void *voidstuff)
{
  Viewer *viewer = (Viewer *)voidstuff;
  for (unsigned int i = 0; i < viewer->view_window_.size(); i++)
  {
    const string name = string("snapshot") + to_string(i) + ".ppm";
    viewer->view_window_[i]->redraw_if_needed();
    // Make sure that the 640x480 here matches up with ViewWindow.cc defaults.
    viewer->view_window_[i]->renderer_->saveImage(name, "ppm", 640, 480);
    viewer->view_window_[i]->redraw(); // flushes saveImage.
  }
  return true;
}


} // End namespace SCIRun
