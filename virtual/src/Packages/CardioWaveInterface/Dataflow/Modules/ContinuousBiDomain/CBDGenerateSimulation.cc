#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

#include <Core/Bundle/Bundle.h>
#include <Core/Datatypes/String.h>
#include <Dataflow/Network/Ports/BundlePort.h>
#include <Dataflow/Network/Ports/StringPort.h>
#include <Packages/CardioWaveInterface/Core/Model/ModelAlgo.h>

#include <sgi_stl_warnings_off.h>
#include <vector>
#include <string>
#include <sgi_stl_warnings_on.h> 

namespace CardioWaveInterface {

using namespace SCIRun;

class CBDGenerateSimulation : public Module {
public:
  CBDGenerateSimulation(GuiContext*);

  virtual void execute();
  
private:
  GuiString gui_filename_;
  GuiInt    gui_enable_debug_;
  GuiInt    gui_build_visualization_bundle_;

};


DECLARE_MAKER(CBDGenerateSimulation)
CBDGenerateSimulation::CBDGenerateSimulation(GuiContext* ctx)
  : Module("CBDGenerateSimulation", ctx, Source, "ContinuousBiDomain", "CardioWaveInterface"),
    gui_filename_(ctx->subVar("filename")),
    gui_enable_debug_(ctx->subVar("enabledebug")),
    gui_build_visualization_bundle_(ctx->subVar("buildvisbundle"))
{
}

void CBDGenerateSimulation::execute()
{
  BundleHandle SimulationBundle;
  StringHandle FileName;
  BundleHandle VisualizationBundle;
  StringHandle SimulationScript;
  
  if (!(get_input_handle("SimulationBundle",SimulationBundle,true))) return;
  get_input_handle("FileName",FileName,false);
  
  if (FileName.get_rep())
  {
    gui_filename_.set(FileName->get());
    get_ctx()->reset();
  }
  
  std::string filename = gui_filename_.get();
  FileName = scinew String(filename);
  bool enable_debug = gui_enable_debug_.get();
  bool build_visualization_bundle = gui_build_visualization_bundle_.get();
  
  SimulationBundle = SimulationBundle->clone();
  SimulationBundle->set_property("enable_debug",enable_debug,false);
  SimulationBundle->set_property("build_visualization_bundle",build_visualization_bundle,false);
  
  ModelAlgo algo(this);  
  if(!(algo.DMDBuildSimulation(SimulationBundle,FileName,VisualizationBundle,SimulationScript))) return;
  
  send_output_handle("SimulationScript",SimulationScript,true);
  if (build_visualization_bundle)
  send_output_handle("VisualizationBundle",VisualizationBundle,true);
}


} // End namespace CardioWave


