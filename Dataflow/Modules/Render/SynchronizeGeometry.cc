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
  virtual void tcl_command(GuiArgs&, void*);
private:
  vector<GeometryComm*> msg_heads_;
  vector<GeometryComm*> msg_tails_;
  vector<int> portno_map_;

  int num_of_IPorts_;
  int max_portno_;
  int init_ports_;

  GeometryOPort * ogeom_;

  GuiInt gui_enforce_;

  int enforce_barrier(MessageBase* message);
  void forward_saved_msg();
  void flush_all_msgs();
  inline void append_msg(GeometryComm* gmsg);
  inline void flush_port(int portno);
  inline int get_enforce();
  inline bool init_ports();
};


DECLARE_MAKER(SynchronizeGeometry)
SynchronizeGeometry::SynchronizeGeometry(GuiContext* ctx)
  : Module("SynchronizeGeometry", ctx, Filter, "Render", "SCIRun"),
    gui_enforce_(ctx->subVar("enforce"))
{
  have_own_dispatch=true;
  max_portno_ = 0;
  init_ports_ = 0;
}

SynchronizeGeometry::~SynchronizeGeometry(){
}

void SynchronizeGeometry::execute() {
}

void SynchronizeGeometry::do_execute() {
  MessageBase* msg;

  ogeom_ = (GeometryOPort*)getOPort("OutputGeometry");

  if(ogeom_ == NULL) 
    cout << "ogeom_ is NULL" << std::endl; cout.flush();

  for(;;){
    msg = mailbox.receive();
    if(enforce_barrier(msg) == 86) {
      return;
    }
  }

}

int SynchronizeGeometry::enforce_barrier(MessageBase* message) {
  int enforce = 1;
  int new_enforce;
  int portno;
  MessageBase* msg;
  GeometryComm* gmsg;

  msg = message;
  gmsg = (GeometryComm*)msg;

  switch(msg->type) {
  case MessageTypes::GoAway:
    return 86;
  case MessageTypes::GeometryInit:
    gmsg->reply->send(GeomReply(max_portno_));
    portno_map_.push_back(-1);
    msg_heads_.push_back(NULL);
    msg_tails_.push_back(NULL);
    max_portno_++;
    init_ports_++;
    init_ports();
    break;
  case MessageTypes::GeometryDelObj:
  case MessageTypes::GeometryDelAll:
  case MessageTypes::GeometryAddObj:
    if(!init_ports())
      break;
    portno = gmsg->portno;

    new_enforce = get_enforce();
    if(new_enforce) 
      append_msg(gmsg);
    else {
      gmsg->portno = portno_map_[portno];
      if(enforce != new_enforce)
	flush_all_msgs();
      if(!(ogeom_->direct_forward(gmsg)))
	delete gmsg;
    }
    enforce = new_enforce;
    msg = 0;
    break;
  case MessageTypes::GeometryFlush:
  case MessageTypes::GeometryFlushViews:
    if(!init_ports())
      break;
    portno = gmsg->portno;

    new_enforce = get_enforce();
    if(new_enforce) {
      append_msg(gmsg);
      forward_saved_msg();
    } else {
      gmsg->portno = portno_map_[portno];      
      if(enforce != new_enforce)
	flush_all_msgs();
      if(!(ogeom_->direct_forward(gmsg)))
	delete gmsg;
    }
    enforce = new_enforce;
    msg = 0;
    break;

  default:
    break;
  }

  if(msg)
    delete msg;

  return 0;
}

inline 
bool 
SynchronizeGeometry::init_ports() 
{
  if(init_ports_ == 0)
    return true;

  Connection* connection;
  if(ogeom_->nconnections()==0)
    return false;
  else
    connection = ogeom_->connection(0);

  if(connection == 0)
    return false;

  int i, portno;
  for(i=0; i<init_ports_; i++) {
    Mailbox<GeomReply> tmp("Temporary GeometryOPort mailbox", 0);    
    ogeom_->forward(scinew GeometryComm(&tmp));
    GeomReply reply = tmp.receive();
    portno=reply.portid;
    portno_map_[max_portno_ - init_ports_ + i] = portno;
  }

  init_ports_ = 0;

  return true;
}

inline int SynchronizeGeometry::get_enforce() {
  string enforce_string;
  gui->get(id+"-enforce",enforce_string);
  return (atoi(enforce_string.c_str()));
}

inline void SynchronizeGeometry::append_msg(GeometryComm* gmsg) {
  int portno = gmsg->portno;
    
  gmsg->next = NULL;
  if(msg_heads_[portno]) {
    msg_tails_[portno]->next = gmsg;
    msg_tails_[portno] = gmsg;
  }
  else {
    msg_heads_[portno] = msg_tails_[portno] = gmsg;
  }

  gmsg->portno = portno_map_[portno];
    
}

void SynchronizeGeometry::forward_saved_msg() {
  int i,num_flush;
  GeometryComm* tmp_gmsg;

  num_flush = 0;
  for(i = 0; i < max_portno_; i++){
    tmp_gmsg = msg_heads_[i];
    while(tmp_gmsg) {
      if(tmp_gmsg->type == MessageTypes::GeometryFlush ||
	 tmp_gmsg->type == MessageTypes::GeometryFlushViews){
	num_flush++;
	break;
      }
      tmp_gmsg = tmp_gmsg->next;
    }
  }

  num_of_IPorts_ = this->numIPorts();
  if(num_flush == num_of_IPorts_ - 1) {
    for(i = 0; i < max_portno_; i++){
      tmp_gmsg = msg_heads_[i];
      bool flushport = false;
      while(tmp_gmsg){	
	if(tmp_gmsg->type == MessageTypes::GeometryFlush ||
	   tmp_gmsg->type == MessageTypes::GeometryFlushViews) {
	  flushport = true;
	  break;
	} else
	  tmp_gmsg = tmp_gmsg->next;
      }
      if(flushport)
	flush_port(i);

      tmp_gmsg=msg_heads_[i];
      while(tmp_gmsg &&
	    (tmp_gmsg->type == MessageTypes::GeometryFlush ||
	     tmp_gmsg->type == MessageTypes::GeometryFlushViews)) {
	flush_port(i);
	tmp_gmsg = msg_heads_[i];
      }
    }
  } 
}

inline void SynchronizeGeometry::flush_port(int portno) {
  bool delete_msg;
  GeometryComm* gmsg,*gmsg2;
  gmsg = msg_heads_[portno];
  while(gmsg){
    delete_msg = false;
    if(!(ogeom_->direct_forward(gmsg)))
      delete_msg  = true;

    if(gmsg->type == MessageTypes::GeometryFlush ||
       gmsg->type == MessageTypes::GeometryFlushViews)  {

      msg_heads_[portno] = gmsg->next;
      if(gmsg->next == NULL)
	msg_tails_[portno] = NULL;
      
      if(delete_msg)
	delete gmsg;

      break;
    } else {
      gmsg2 = gmsg;
      gmsg = gmsg->next;
      if(delete_msg)
	delete gmsg2;

    }
  }
}

void SynchronizeGeometry::flush_all_msgs() {
  int i;
  GeometryComm* gmsg;
  for(i = 0; i < max_portno_; i++){
    gmsg = msg_heads_[i];
    while(gmsg) {
      if(!(ogeom_->direct_forward(gmsg)))
	delete gmsg;
      gmsg = gmsg->next;
    }

    msg_heads_[i] = msg_tails_[i] = NULL;
  }
}

void
SynchronizeGeometry::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace SCIRun
