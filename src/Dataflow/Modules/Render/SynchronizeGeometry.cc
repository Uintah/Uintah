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
  int num_of_IPorts_;
  int max_iportno_ ;
  GeometryOPort * ogeom_;
  void append_msg(GeometryComm* gmsg);
  void forward_saved_msg();
  void flush_port(int portno);
};


DECLARE_MAKER(SynchronizeGeometry)
SynchronizeGeometry::SynchronizeGeometry(GuiContext* ctx)
  : Module("SynchronizeGeometry", ctx, Filter, "Render", "SCIRun")
{
  have_own_dispatch=true;
  max_iportno_ = -1;
}

SynchronizeGeometry::~SynchronizeGeometry(){
}

  void SynchronizeGeometry::execute() {
  }

void SynchronizeGeometry::do_execute() {
  int i, portno;
  MessageBase* msg;
  GeometryComm* gmsg;

  ogeom_ = (GeometryOPort*)getOPort("OutputGeometry");

  for(;;){
    msg = mailbox.receive();

    bool num_of_IPorts_changed = false;
    num_of_IPorts_ = this->numIPorts();

    gmsg = (GeometryComm*)msg;

    if (msg->type == MessageTypes::GeometryInit ||
	msg->type == MessageTypes::GeometryDelObj ||
	msg->type == MessageTypes::GeometryDelAll ||
	msg->type == MessageTypes::GeometryAddObj ||
	msg->type == MessageTypes::GeometryFlush ||
	msg->type == MessageTypes::GeometryFlushViews) {

      gmsg->next = NULL;

      portno = gmsg->portno;

      if(portno > max_iportno_) {
	num_of_IPorts_changed = true;
      }
    }

    if (num_of_IPorts_changed) {
      // resize msg_heads_ and msg_tails_
      // if they got bigger (num_of_IPorts_ < this->numIPorts),
      //    then initilialize new entries to NULL
      msg_heads_.resize(portno + 1);
      msg_tails_.resize(portno + 1);
      for(i=max_iportno_ + 1; i<= portno; i++){
	msg_heads_[i] = NULL;
	msg_tails_[i] = NULL;
      }
      max_iportno_ = portno;
    }

    switch(msg->type) {
    case MessageTypes::GoAway:
      helper_done.send(1);
      return;
    case MessageTypes::ExecuteModule:
      break; 
    case MessageTypes::ViewWindowRedraw:
    case MessageTypes::ViewWindowDumpImage:
    case MessageTypes::ViewWindowMouse:
      break;
    case MessageTypes::GeometryDelObj:
    case MessageTypes::GeometryDelAll:
    case MessageTypes::GeometryAddObj:
      append_msg(gmsg);
      msg = 0;
      break;
    case MessageTypes::GeometryInit:
      ogeom_->forward(gmsg);
      msg = 0;
      break;
    case MessageTypes::GeometryFlush:
    case MessageTypes::GeometryFlushViews:
      if(msg_heads_[portno] == NULL) {
	if(!(ogeom_->direct_forward(gmsg)))
	  delete gmsg;
      } else {
	append_msg(gmsg);
	forward_saved_msg();
      }
      msg = 0;
      break;
    case MessageTypes::GeometryGetNViewWindows:
    case MessageTypes::GeometryGetData:
    case MessageTypes::GeometrySetView:
      break;
    default:
      break;
    }
    if(msg)
      delete msg;
  }  
}

void SynchronizeGeometry::append_msg(GeometryComm* gmsg) {
  int portno = gmsg->portno;
    
  if(msg_heads_[portno]) {
    msg_tails_[portno]->next = gmsg;
    msg_tails_[portno] = gmsg;
  }
  else {
    msg_heads_[portno] = msg_tails_[portno] = gmsg;
  }
}

void SynchronizeGeometry::forward_saved_msg() {
  int i,num_flush;
  GeometryComm* tmp_gmsg;
  
  num_flush = 0;
  for(i = 0; i <= max_iportno_; i++){
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

  if(num_flush == num_of_IPorts_ - 1) {
    for(i = max_iportno_; i >= 0; i--){
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

void SynchronizeGeometry::flush_port(int portno) {
  GeometryComm* gmsg;
  gmsg = msg_heads_[portno];
  while(gmsg){
    if(!(ogeom_->direct_forward(gmsg)))
      delete gmsg;
    if(gmsg->type == MessageTypes::GeometryFlush ||
       gmsg->type == MessageTypes::GeometryFlushViews)  {

      msg_heads_[portno] = gmsg->next;
      if(gmsg->next == NULL)
	msg_tails_[portno] = NULL;
      
      break;
    } else {
      gmsg = gmsg->next;
    }
  }
}

void
 SynchronizeGeometry::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace SCIRun


