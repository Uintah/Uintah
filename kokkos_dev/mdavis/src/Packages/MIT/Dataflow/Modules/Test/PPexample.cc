/*
 *  PPexample.cc:
 *
 *  Written by:
 *   moulding
 *   TODAY'S DATE HERE
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Parts/GraphPart.h>
#include <Packages/MIT/Dataflow/Modules/Test/PPInterface.h>
#include <Packages/MIT/Dataflow/Modules/Test/PPexampleGui.h>
#include <Core/2d/LockedPolyline.h>
#include <Core/2d/ParametricPolyline.h>

#include <Packages/MIT/share/share.h>

#include <math.h>

#include <vector>

namespace MIT {

using namespace SCIRun;
using std::vector;

class MITSHARE PPexample : public Module {

  PPInterface *interface_;
  GraphPart *graph_;

public:
  PPexample(const string& id);

  virtual ~PPexample();

  virtual void execute();

  virtual void tcl_command(TCLArgs&, void*);
};

extern "C" MITSHARE Module* make_PPexample(const string& id) {
  return scinew PPexample(id);
}

PPexample::PPexample(const string& id)
  : Module("PPexample", id, Source, "Test", "MIT")
{
  interface_ = new PPInterface( (Part*)this, 0 );
  graph_ = new GraphPart( interface_, "example");
  graph_->set_num_lines(2);
}

PPexample::~PPexample(){
}

void PPexample::execute()
{
  vector<double> myv;
  vector<DrawObj*> v;
  myv.resize(30000);

  v.resize(2);
  ParametricPolyline p;
  LockedPolyline p2;
  v[0]=&p;
  v[1]=&p2;
  graph_->reset(v);
  
  for (int i=0; i<29997; i+=3) {
    myv[i]=i/300.;
    myv[i+1]=myv[i]+cos(2*myv[i]);
    myv[i+2]=cos(myv[i]/30.)+sin(.5*myv[i]);
  }
  graph_->add_values(0,myv);

  myv.resize(100);
  for (int i=0; i<100; ++i) {
      myv[i]=sin(i/3.);
  }
  graph_->add_values(1,myv);
}

void PPexample::tcl_command(TCLArgs& args, void* data)
{
  if ( args[1] == "set-window" ) {
    PPexampleGui *gui = new PPexampleGui( id+"-gui" );
    gui->set_window( args[2] );

    connect( gui->burning, interface_, &PPInterface::burning );
    connect( gui->monitor, interface_, &PPInterface::monitor );
    connect( gui->thin, interface_, &PPInterface::thin );
    connect( gui->go, interface_, &PPInterface::go);
    connect( interface_->has_child, (PartGui* )gui, &PartGui::add_child );

    interface_->report_children( (PartGui* )gui, &PartGui::add_child );
  } else Module::tcl_command( args, data );
}

} // End namespace MIT


