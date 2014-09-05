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
 *  SynchronizeGeometry.cc:
 *
 *  Written by:
 *   Kai Li
 *   Jan, 2003
 *
 */
#include <Dataflow/Network/Module.h>
#include <Dataflow/Network/Scheduler.h>
#include <Core/Malloc/Allocator.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Dataflow/Ports/GeometryComm.h>
#include <Dataflow/Comm/MessageTypes.h>
#include <Dataflow/Comm/MessageBase.h>
#include <Core/Containers/StringUtil.h>

#include <vector>

namespace SCIRun {

using std::vector;

class SynchronizeGeometry : public Module {
public:
  SynchronizeGeometry(GuiContext*);

  virtual ~SynchronizeGeometry();

  virtual void execute();
  virtual void do_execute();

private:
  vector<GeometryComm*> msg_heads_;
  vector<GeometryComm*> msg_tails_;
  vector<int> physical_portno_;
  vector<map<GeomID, GeomID, less<GeomID> > > geom_ids_;

  int max_portno_;

  GeometryOPort *ogeom_;

  GuiInt gui_enforce_;

  int process_event(MessageBase* message);
  void forward_saved_msg();
  void flush_all_msgs();
  void append_msg(GeometryComm* gmsg);
  bool flush_port(int portno, int count);
};



DECLARE_MAKER(SynchronizeGeometry)
SynchronizeGeometry::SynchronizeGeometry(GuiContext* ctx)
  : Module("SynchronizeGeometry", ctx, Filter, "Render", "SCIRun"),
    max_portno_(0),
    gui_enforce_(ctx->subVar("enforce"))
{
  have_own_dispatch = true;
}



SynchronizeGeometry::~SynchronizeGeometry()
{
}



void
SynchronizeGeometry::execute()
{
}



void
SynchronizeGeometry::do_execute()
{
  ogeom_ = (GeometryOPort*)getOPort("Output Geometry");
  for (;;)
  {
    MessageBase *msg = mailbox.receive();
    if (process_event(msg) == 86)
    {
      return;
    }
  }
}



int
SynchronizeGeometry::process_event(MessageBase* msg)
{
  GeometryComm* gmsg = (GeometryComm*)msg;

  switch (msg->type)
  {
  case MessageTypes::GoAway:
    return 86;

  case MessageTypes::GeometryInit:
    gmsg->reply->send(GeomReply(max_portno_));
    physical_portno_.push_back(numIPorts()-1);
    max_portno_++;
    msg_heads_.push_back(NULL);
    msg_tails_.push_back(NULL);
    geom_ids_.push_back(map<GeomID, GeomID, less<GeomID> >());
    break;

  case MessageTypes::GeometryDetach:
    {
      // Remove all of the gmsg->portno objects
      map<GeomID, GeomID, less<GeomID> >::iterator itr;
      itr = geom_ids_[gmsg->portno].begin();
      int counter = 0;
      while (itr != geom_ids_[gmsg->portno].end())
      {
	ogeom_->delObj((*itr).second);
	++itr;
	counter++;
      }
      geom_ids_[gmsg->portno].clear();
      if (counter) { ogeom_->flush(); }

      // Maybe still connected geometries are now in sync?
      gui_enforce_.reset();
      if (gui_enforce_.get())
      {
	forward_saved_msg();
      }
      else
      {
	flush_all_msgs();
      }

      // Fix the portnos.
      for (unsigned int i=gmsg->portno + 1; i < physical_portno_.size(); i++)
      {
	if (physical_portno_[i] != -1) { physical_portno_[i]--; }
      }
      physical_portno_[gmsg->portno] = -1;

      // Push changed portno strings to output port?
    }
    break;

  case MessageTypes::GeometryDelAll:
    gui_enforce_.reset();
    if (gui_enforce_.get())
    {
      append_msg(gmsg);
      msg = 0;
    }
    else
    {
      flush_all_msgs();
      map<GeomID, GeomID, less<GeomID> >::iterator itr;
      itr = geom_ids_[gmsg->portno].begin();
      while (itr != geom_ids_[gmsg->portno].end())
      {
	ogeom_->delObj((*itr).second);
	++itr;
      }
      geom_ids_[gmsg->portno].clear();
    }
    break;

  case MessageTypes::GeometryDelObj:
    gui_enforce_.reset();
    if (gui_enforce_.get())
    {
      append_msg(gmsg);
      msg = 0;
    }
    else
    {
      flush_all_msgs();
      // TODO: verify id found in map.
      ogeom_->delObj(geom_ids_[gmsg->portno][gmsg->serial]);
      geom_ids_[gmsg->portno].erase(gmsg->serial);
    }
    break;
    break;

  case MessageTypes::GeometryAddObj:
    gui_enforce_.reset();
    if (gui_enforce_.get())
    {
      append_msg(gmsg);
      msg = 0;
    }
    else
    {
      flush_all_msgs();
      const string pnum = to_string(physical_portno_[gmsg->portno]);
      const string newname =  gmsg->name + " (" + pnum + ")";
      const GeomID id = ogeom_->addObj(gmsg->obj, newname);
      geom_ids_[gmsg->portno][gmsg->serial] = id;
    }
    break;

  case MessageTypes::GeometryFlush:
    gui_enforce_.reset();
    if (gui_enforce_.get())
    {
      append_msg(gmsg);
      forward_saved_msg();
      msg = 0;
    }
    else
    {
      flush_all_msgs();
      ogeom_->flush();
    }
    break;

  case MessageTypes::GeometryFlushViews:
    gui_enforce_.reset();
    if (gui_enforce_.get())
    {
      append_msg(gmsg);
      forward_saved_msg();
      msg = 0;
    }
    else
    {
      flush_all_msgs();
      ogeom_->flushViews();
    }
    break;

  case MessageTypes::ExecuteModule:
    gui_enforce_.reset();
    if (!gui_enforce_.get())
    {
      flush_all_msgs();
    }
    sched->report_execution_finished(msg);
    break;

  default:
    break;
  }

  if (msg)
  {
    delete msg;
  }

  return 0;
}



void
SynchronizeGeometry::append_msg(GeometryComm* gmsg)
{
  int portno = gmsg->portno;

  gmsg->next = NULL;
  if (msg_heads_[portno])
  {
    msg_tails_[portno]->next = gmsg;
    msg_tails_[portno] = gmsg;
  }
  else
  {
    msg_heads_[portno] = msg_tails_[portno] = gmsg;
  }
}



void
SynchronizeGeometry::forward_saved_msg()
{
  ostringstream str;
  str << " Checking " << max_portno_ << " ports.";
  remark( str.str() );

  int i, num_flush, valid;

  num_flush = 0;
  valid = 0;
  for (i = 0; i < max_portno_; i++)
  {
    if (physical_portno_[i] != -1)
    {
      valid++;
      GeometryComm *tmp_gmsg = msg_heads_[i];
      while (tmp_gmsg)
      {
	if (tmp_gmsg->type == MessageTypes::GeometryFlush ||
	    tmp_gmsg->type == MessageTypes::GeometryFlushViews)
	{
	  num_flush++;

	  ostringstream str;
	  str << "  port " << i << " is ready.";
	  remark( str.str() );
	  break;
	}
	tmp_gmsg = tmp_gmsg->next;
      }
    }
  }

  if (num_flush == valid)
  {
    remark( " All were ready, flushing." );
    bool some = false;
    for (i = 0; i < max_portno_; i++)
    {
      if (physical_portno_[i] != -1)
      {
	some |= flush_port(i, 1);
      }
    }
    if (some) { ogeom_->flush(); }

    update_progress(1.0);
    update_state(Completed);
  }
  else
  {
    update_progress(num_flush, numIPorts() - 1);
  }
}



bool
SynchronizeGeometry::flush_port(int portno, int count)
{
  GeometryComm *gmsg = msg_heads_[portno];
  const bool some = gmsg;
  while (gmsg)
  {
    // Process messages here.
    // GeometryDelAll
    if (gmsg->type == MessageTypes::GeometryDelAll)
    {
      map<GeomID, GeomID, less<GeomID> >::iterator itr;
      itr = geom_ids_[gmsg->portno].begin();
      while (itr != geom_ids_[gmsg->portno].end())
      {
	ogeom_->delObj((*itr).second);
	++itr;
      }
      geom_ids_[gmsg->portno].clear();
    }

    // GeometryDelObj
    else if (gmsg->type == MessageTypes::GeometryDelObj)
    {
      // TODO: verify id found in map.
      ogeom_->delObj(geom_ids_[gmsg->portno][gmsg->serial]);
      geom_ids_[gmsg->portno].erase(gmsg->serial);
    }

    // GeometryAddObj
    else if (gmsg->type == MessageTypes::GeometryAddObj)
    {
      const string pnum = to_string(physical_portno_[gmsg->portno]);
      const string newname =  gmsg->name + " (" + pnum + ")";
      const GeomID id = ogeom_->addObj(gmsg->obj, newname);
      geom_ids_[gmsg->portno][gmsg->serial] = id;
    }

    // Eat up the flushes.
    else if (gmsg->type == MessageTypes::GeometryFlush ||
	     gmsg->type == MessageTypes::GeometryFlushViews)
    {
      count--;
      if (count == 0)
      {
	msg_heads_[portno] = gmsg->next;
	if (gmsg->next == NULL)
	{
	  msg_tails_[portno] = NULL;
	}
	delete gmsg;
	break;
      }
    }
    else
    {
      // Unprocessed message.
    }

    GeometryComm *next = gmsg->next;
    delete gmsg;
    gmsg = next;
  }

  if (gmsg == NULL)
  {
    msg_heads_[portno] = NULL;
    msg_tails_[portno] = NULL;
  }

  return some;
}



void
SynchronizeGeometry::flush_all_msgs()
{
  bool some = false;
  for (int i = 0; i < max_portno_; i++)
  {
    if (physical_portno_[i] != -1)
    {
      some |= flush_port(i, -1);
    }
  }
  if (some)
  {
    ogeom_->flush();
  }

  update_progress(1.0);
  update_state(Completed);
}


} // End namespace SCIRun
