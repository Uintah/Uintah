/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
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

#include <Dataflow/Network/Module.h>

#include <Core/Bundle/Bundle.h>
#include <Core/Datatypes/String.h>
#include <Dataflow/Network/Ports/BundlePort.h>
#include <Dataflow/Network/Ports/StringPort.h>
#include <Packages/CardioWaveInterface/Core/Model/ModelAlgo.h>
#include <Core/OS/FullFileName.h>

#include <sgi_stl_warnings_off.h>
#include <vector>
#include <string>
#include <sgi_stl_warnings_on.h> 

namespace CardioWaveInterface {

using namespace SCIRun;

class DMDGenerateSimulation : public Module {
public:
  DMDGenerateSimulation(GuiContext*);

  virtual void execute();
  
private:
  GuiString gui_filename_;
  GuiInt    gui_enable_debug_;
  GuiInt    gui_build_visualization_bundle_;
  GuiInt    gui_optimize_;

};


DECLARE_MAKER(DMDGenerateSimulation)
DMDGenerateSimulation::DMDGenerateSimulation(GuiContext* ctx)
  : Module("DMDGenerateSimulation", ctx, Source, "DiscreteMultiDomain", "CardioWaveInterface"),
    gui_filename_(ctx->subVar("filename")),
    gui_enable_debug_(ctx->subVar("usedebug")),
    gui_build_visualization_bundle_(ctx->subVar("buildvisbundle")),
    gui_optimize_(ctx->subVar("optimize"))
{
}

void DMDGenerateSimulation::execute()
{
  // Define dataflow handles:
  BundleHandle SimulationBundle;
  StringHandle FileName;
  BundleHandle VisualizationBundle;
  StringHandle SimulationScript;

  // Get data from ports:
  if (!(get_input_handle("SimulationBundle",SimulationBundle,true))) return;
  get_input_handle("FileName",FileName,false);

  // See whether we need to do any work:
  if (inputs_changed_ ||  gui_filename_.changed() ||
      gui_enable_debug_.changed() || 
      gui_build_visualization_bundle_.changed() || 
      !oport_cached("SimulationScript") || 
      (!oport_cached("VisualizationBundle") && gui_build_visualization_bundle_.get()) )
  {
    // Send data to widget:
    if (FileName.get_rep())
    {
      gui_filename_.set(FileName->get());
      get_ctx()->reset();
    }
    
    std::string filename = gui_filename_.get();
    FullFileName ffn(filename);
		if (!(ffn.create_file_path()))
		{
			error("Could not generate path to file");
			return;
		}
		
		filename = ffn.get_abs_filename();
		
		gui_filename_.set(filename);
    get_ctx()->reset();
					
		FileName = scinew String(filename);
    bool enable_debug = gui_enable_debug_.get();
    bool build_visualization_bundle = gui_build_visualization_bundle_.get();
    bool optimize_system = gui_optimize_.get();
    
    // Add the last details to the simulation bundle:
    SimulationBundle = SimulationBundle->clone();
    SimulationBundle->set_property("enable_debug",enable_debug,false);
    SimulationBundle->set_property("build_visualization_bundle",build_visualization_bundle,false);
    SimulationBundle->set_property("optimize-system",optimize_system,false);
    
    
    // Actual algorithm:
    ModelAlgo algo(this);  
    if(!(algo.DMDBuildSimulation(SimulationBundle,FileName,VisualizationBundle,SimulationScript))) return;
    
    // Send output downstream:
    send_output_handle("SimulationScript",SimulationScript,false);
    if (build_visualization_bundle) send_output_handle("VisualizationBundle",VisualizationBundle,false);
  }
}


} // End namespace CardioWave


