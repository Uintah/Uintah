/*
 * GatherTimeSteps.cc:  Collect geometry objects and make them time-dependent
 *
 * Written by:
 *  Philip Sutton
 *   Department of Computer Science
 *   University of Utah
 *   September 1998
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <Dataflow/Module.h>
#include <Geom/Geom.h>
#include <Geom/Sphere.h>
#include <Geom/TimeGroup.h>
#include <Geom/Group.h>
#include <Geometry/BBox.h>
#include <Datatypes/GeometryPort.h>
#include <Datatypes/GeometryComm.h>
#include <Malloc/Allocator.h>
#include <TCL/TCLvar.h>
#include <Geom/IndexedGroup.h>
#include <Classlib/Array1.h>

class GatherTimeSteps : public Module {
private:

  GeometryOPort *outport;
  Array1<int> portids;
  
  int busy;
  int portid;
  virtual void do_execute();
  void process_event();

  GeomTimeGroup *tgroup;
  Array1<GeomObj *> lastobjs;
  int currtime;
  TCLint timelimit;
  TCLint tsp;
 
public:
  GatherTimeSteps( const clString &id );
  GatherTimeSteps( const GatherTimeSteps&, int deep );
  virtual ~GatherTimeSteps();
  virtual Module* clone(int deep);
  virtual void execute();
  virtual void connection(ConnectionMode, int, int);
  virtual void tcl_command(TCLArgs& args, void* userdata);
};

extern "C" {
  Module* make_GatherTimeSteps(const clString& id)
  {
    return new GatherTimeSteps(id);
  }
	   }

GatherTimeSteps::GatherTimeSteps(const clString& id) :
  Module("GatherTimeSteps", id, Filter), timelimit("timelimit", id, this),
  tsp("tsp", id, this)
{
  // create ports
  outport = scinew GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
  add_oport( outport );
  add_iport( scinew GeometryIPort(this, "Geometry", GeometryIPort::Atomic) );
  
  have_own_dispatch = 1;
  busy = 0;
  
  // set up timed group
  tgroup = new GeomTimeGroup();

  // set up variables
  currtime = 1;
  timelimit.set(255);
  tsp.set(0);
}

GatherTimeSteps::GatherTimeSteps(const GatherTimeSteps& copy, int deep)
  : Module( copy, deep ), timelimit("timelimit", id, this),
    tsp("tsp", id, this)
{
}

GatherTimeSteps::~GatherTimeSteps() {
}

Module* GatherTimeSteps::clone(int deep){
  return new GatherTimeSteps(*this, deep);
}

void GatherTimeSteps::do_execute() {
  update_state( Completed );
  for(;;) {
    process_event();
  }
}

void GatherTimeSteps::process_event()
{
  MessageBase* msg=mailbox.receive();
  GeometryComm* gmsg=(GeometryComm*)msg;

  cout << "GTS: gmsg->type = " << gmsg->type << ", name="
       << ((gmsg->type==MessageTypes::GeometryAddObj)?gmsg->name():"<none>")
       << " port = " << gmsg->portno << endl;
  switch(gmsg->type){
  case MessageTypes::ExecuteModule:
    // We ignore these messages...
    break;
  case MessageTypes::GeometryAddObj:
    {
      // don't accept any geometry after timelimit is reached
      if( currtime > timelimit.get() ) {
	break;
      }

      // replace lastobj with the message's object
      cout << "portids.size() = " << portids.size() << " gmsg->portno = "
	   << gmsg->portno << endl;
      for( int i = 0; i < portids.size(); i++ ) {
	cout << "portids[" << i << "] = " << portids[i] << endl;
	if( portids[i] == gmsg->portno ) {
	  cout << "found it at i = " << i << " lastobjs.size() = "
	       << lastobjs.size() << endl;
	  if( lastobjs.size() < i+1 ) 
	    lastobjs.grow( i - lastobjs.size() + 1 );
	  cout << "now lastobjs.size() = " << lastobjs.size() << endl;
	  lastobjs[i] = gmsg->obj;
	  break;
	}
      }
      ASSERT( i <= portids.size() );
    }
    break;
  case MessageTypes::GeometryDelObj:
  case MessageTypes::GeometryDelAll:
    // I don't let upstream modules delete anything because I haven't
    // added anything to Salmon, just to the timegroup.
    break;
  case MessageTypes::GeometryInit:
    gmsg->reply->send(GeomReply(portid++, &busy));
    break;	
  case MessageTypes::GeometryFlush:
    outport->forward(gmsg);
    break;
  case MessageTypes::GeometryFlushViews:
    outport->forward(gmsg);
    break;
  case MessageTypes::GeometryGetNRoe:
    outport->forward(gmsg);
    break;
  case MessageTypes::GeometryGetData:
    outport->forward(gmsg);
    break;
  case MessageTypes::ModuleGeneric1:
    {
      // this block is reached by sending the "update" message
      // (see tcl_command below)
      
      timelimit.reset();
      tsp.reset();
      
      // each timestep is represented by one group
      GeomGroup *grp = new GeomGroup();

      // add the last objects on each port to the non-timed-group
      for( int i = 0; i < lastobjs.size(); i++ ) {
	grp->add( lastobjs[i] );
      }
      tsp.set( tsp.get() + 1 );
      tsp.reset();
      
      // add the non-timed group to the timegroup
      tgroup->add( grp, (double)(currtime-1) / (double)timelimit.get() );
      
      // if we've gotten the appropriate number of timesteps, forward the
      // timegroup to Salmon
      if( currtime == timelimit.get() ) {
	BBox b;
	tgroup->get_bounds(b);
	tgroup->setbbox(b);

	cout << "GatherTimeSteps: forwarding message" << endl;

	outport->addObj( tgroup, "TimeGroup" );
	outport->flush();
      }
      currtime++;
    }

    break;
  default:
    cerr << "GatherTimeSteps: Illegal Message type: " << gmsg->type << endl;
    break;
  }
}

void GatherTimeSteps::execute() {
}


void GatherTimeSteps::tcl_command(TCLArgs& args, void* userdata)
{
    if(args.count() < 2){
      args.error("GatherTimeSteps needs a minor command");
      return;
    }
    if( args[1] == "update" ){
      mailbox.send(new MessageBase(MessageTypes::ModuleGeneric1));
    } else {
      Module::tcl_command(args, userdata);
    }
}

void GatherTimeSteps::connection(ConnectionMode mode, int which_port, int)
{
  if(mode==Disconnected){
    remove_iport(which_port);
  } else {
    add_iport(scinew GeometryIPort(this, "Geometry", GeometryIPort::Atomic));
    portids.add( portid );
  }
}
