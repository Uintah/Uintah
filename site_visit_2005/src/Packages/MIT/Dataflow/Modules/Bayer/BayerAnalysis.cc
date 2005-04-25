/*
 *  BayerAnalysis.cc:
 *
 *  Written by:
 *   Yarden Livnat
 *   Sep 2001
 *
 */

#include <Core/Containers/Array2.h>
#include <Core/Containers/StringUtil.h>
#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
// #include <Core/GuiInterface/TCLTask.h>
// #include <Core/GuiInterface/TCL.h>
#include <Core/2d/Graph.h>
#include <Core/2d/Diagram.h>
#include <Core/2d/HistObj.h>

#include <Packages/MIT/Core/Datatypes/MetropolisData.h>
#include <Packages/MIT/Dataflow/Ports/MetropolisPorts.h>

#include <Packages/MIT/share/share.h>

namespace MIT {

using namespace SCIRun;

class MITSHARE BayerAnalysis : public Module {
public:
  ResultsHandle data_;
  
  // 
  Graph *graph_;
  Diagram *diagram_;
  Array1<HistObj *> hist_;
  int current_generation;
  
public:
  BayerAnalysis(const string& id);

  virtual ~BayerAnalysis();

  virtual void execute();
  
  virtual void tcl_command(TCLArgs&, void*);

private:
  void compute_statistics();
};

extern "C" MITSHARE Module* make_BayerAnalysis(const string& id) {
  return scinew BayerAnalysis(id);
}

BayerAnalysis::BayerAnalysis(const string& id)
  : Module("BayerAnalysis", id, Source, "Bayer", "MIT"),
    current_generation(-1)
{
  graph_ = scinew Graph( id+"-Graph" );
  diagram_ = scinew Diagram( "Histogram");
  graph_->add( "Histogram", diagram_);
}

BayerAnalysis::~BayerAnalysis()
{
}

void 
BayerAnalysis::execute()
{
  update_state(NeedData);

  ResultsIPort *input = (ResultsIPort *) get_iport("Posterior");
  input->get(data_);

  if ( !data_.get_rep() )
    return;

  update_state(JustStarted);

  compute_statistics();
}

void
BayerAnalysis::compute_statistics()
{
  int n = data_->size();
  if ( hist_.size() != n ) {
    for ( int i=hist_.size(); i<n; i++ ) {
      hist_.add( new HistObj( string("hist-") + to_string(i) ) );
      hist_[i]->set_color( data_->color_[i] ); 
      diagram_->add( hist_[i] );
    }
  }      

  for (int i=0; i<n; i++ ) {
    hist_[i]->set_data( data_->data_[i] );
  }

  graph_->need_redraw();
}

void 
BayerAnalysis::tcl_command(TCLArgs& args, void* userdata)
{
  if ( args[1] == "graph-window" ) {
    graph_->set_window( args[2] ); 
  } 
  else
    Module::tcl_command(args, userdata);

}


} // namespace MIT


