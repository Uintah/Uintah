/*
 *  SynchronizeGeometry.cc:
 *
 *  Written by:
 *   Kai Li
 *   Jan, 2003
 *
 */
#include <vector>
#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Dataflow/Ports/GeometryComm.h>
#include <Dataflow/Comm/MessageTypes.h>
#include <Dataflow/Comm/MessageBase.h>
#include <Core/Containers/StringUtil.h>

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
  vector<int> portno_map_;

  int max_portno_;

  GeometryOPort *ogeom_;

  GuiInt gui_enforce_;

  int enforce_barrier(MessageBase* message);
  void forward_saved_msg();
  void flush_all_msgs();
  void append_msg(GeometryComm* gmsg);
  void flush_port(int portno);
  bool init_ports(int nports);
};



DECLARE_MAKER(SynchronizeGeometry)
SynchronizeGeometry::SynchronizeGeometry(GuiContext* ctx)
  : Module("SynchronizeGeometry", ctx, Filter, "Render", "SCIRun"),
    gui_enforce_(ctx->subVar("enforce"))
{
  have_own_dispatch=true;
  max_portno_ = 0;
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
  MessageBase* msg;

  ogeom_ = (GeometryOPort*)getOPort("Output Geometry");

  if (ogeom_ == NULL)
  {
    error("Unable to initialize iport 'Output Geometry");
  }

  for (;;)
  {
    msg = mailbox.receive();
    if (enforce_barrier(msg) == 86)
    {
      return;
    }
  }
}



int
SynchronizeGeometry::enforce_barrier(MessageBase* msg)
{
  const bool enforce = gui_enforce_.get();
  GeometryComm* gmsg = (GeometryComm*)msg;

  switch (msg->type)
  {
  case MessageTypes::GoAway:
    return 86;

  case MessageTypes::GeometryInit:
    gmsg->reply->send(GeomReply(max_portno_));
    portno_map_.push_back(-1);
    msg_heads_.push_back(NULL);
    msg_tails_.push_back(NULL);
    max_portno_++;
    init_ports(1);
    break;

  case MessageTypes::GeometryDelObj:
  case MessageTypes::GeometryDelAll:
  case MessageTypes::GeometryAddObj:
    if (enforce)
    {
      append_msg(gmsg);
    }
    else
    {
      gmsg->portno = portno_map_[gmsg->portno];
      flush_all_msgs();
      if (!(ogeom_->direct_forward(gmsg)))
      {
	delete gmsg;
      }
    }
    msg = 0;
    break;

  case MessageTypes::GeometryFlush:
  case MessageTypes::GeometryFlushViews:
    if (enforce)
    {
      append_msg(gmsg);
      forward_saved_msg();
    }
    else
    {
      gmsg->portno = portno_map_[gmsg->portno];
      flush_all_msgs();
      if (!(ogeom_->direct_forward(gmsg)))
      {
	delete gmsg;
      }
    }
    msg = 0;
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



bool
SynchronizeGeometry::init_ports(int nports)
{
  if (nports == 0)
  {
    return true;
  }

  if (ogeom_->nconnections() == 0)
  {
    return false;
  }

  if (ogeom_->connection(0) == 0)
  {
    return false;
  }

  for (int i=0; i < nports; i++)
  {
    Mailbox<GeomReply> *tmp =
      scinew Mailbox<GeomReply>("Temporary GeometryOPort mailbox", 1);
    ogeom_->forward(scinew GeometryComm(tmp));
    GeomReply reply = tmp->receive();
    portno_map_[max_portno_ - nports + i] = reply.portid;
  }

  return true;
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

  gmsg->portno = portno_map_[portno];
}



void
SynchronizeGeometry::forward_saved_msg()
{
  int i, num_flush;

  cout << "SynchronizeGeometry::forward_saved_msg called, checking " <<
    max_portno_ << " ports.\n";

  num_flush = 0;
  for (i = 0; i < max_portno_; i++)
  {
    GeometryComm *tmp_gmsg = msg_heads_[i];
    while (tmp_gmsg)
    {
      if (tmp_gmsg->type == MessageTypes::GeometryFlush ||
	  tmp_gmsg->type == MessageTypes::GeometryFlushViews)
      {
	num_flush++;
	cout << "  port " << i << " is ready.\n";
	break;
      }
      tmp_gmsg = tmp_gmsg->next;
    }
  }

  if (num_flush == numIPorts() - 1)
  {
    cout << " All were ready, flushing.\n";
    for (i = 0; i < max_portno_; i++)
    {
      flush_port(i);

#if 0
      GeometryComm *tmp_gmsg = msg_heads_[i];
      while (tmp_gmsg &&
	     (tmp_gmsg->type == MessageTypes::GeometryFlush ||
	      tmp_gmsg->type == MessageTypes::GeometryFlushViews))
      {
	flush_port(i);
	tmp_gmsg = msg_heads_[i];
      }
#endif
    }
  }
}



void
SynchronizeGeometry::flush_port(int portno)
{
  bool delete_msg;
  GeometryComm *gmsg = msg_heads_[portno];
  while (gmsg)
  {
    delete_msg = false;
    if (!(ogeom_->direct_forward(gmsg)))
    {
      delete_msg = true;
    }

    if (gmsg->type == MessageTypes::GeometryFlush ||
	gmsg->type == MessageTypes::GeometryFlushViews)
    {
      msg_heads_[portno] = gmsg->next;
      if (gmsg->next == NULL)
      {
	msg_tails_[portno] = NULL;
      }

      if (delete_msg)
      {
	delete gmsg;
      }

      break;
    }

    GeometryComm *next = gmsg->next;
    if (delete_msg)
    {
      delete gmsg;
    }
    gmsg = next;
  }
}



void
SynchronizeGeometry::flush_all_msgs()
{
  for (int i = 0; i < max_portno_; i++)
  {
    GeometryComm *gmsg = msg_heads_[i];
    while (gmsg)
    {
      GeometryComm *next = gmsg->next;
      if (!(ogeom_->direct_forward(gmsg)))
      {
	delete gmsg;
      }
      gmsg = next;
    }

    msg_heads_[i] = msg_tails_[i] = NULL;
  }
}


} // End namespace SCIRun
