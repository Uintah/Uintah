//
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2004 Scientific Computing and Imaging Institute,
//  University of Utah.
//  
//  License for the specific language governing rights and limitations under
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//  
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
//  
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//  
//    File   : main.cc
//    Author : McKay Davis
//    Date   : Tue May 30 21:38:23 MDT 2006



#include <main/sci_version.h>
#include <Core/Events/EventManager.h>

#include <Core/Skinner/Skinner.h>
#include <Core/Skinner/XMLIO.h>
#include <Core/Skinner/Window.h>

#include <Core/Util/Environment.h>
#include <Core/Bundle/Bundle.h>

#include <Core/Events/Tools/QuitMainWindowTool.h>
#include <Core/Events/Tools/FilterRedrawEventsTool.h>
#include <Core/Volume/VolumeRenderer.h>
#include <Core/Events/SceneGraphEvent.h>
#include <Core/Geom/ShaderProgramARB.h>
#include <Core/Volume/ColorMap2.h>

#include <Core/Util/DynamicCompilation.h>
#include <Core/Algorithms/Visualization/NrrdTextureBuilderAlgo.h>

#include <StandAlone/Apps/Painter/Painter.h>

#include <sgi_stl_warnings_off.h>
#include <string>
#include <iostream>
#include <sgi_stl_warnings_on.h>
using std::cout;

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma set woff 1424
#pragma set woff 1209 
#endif


using namespace SCIRun;

int
main(int argc, char *argv[], char **environment) {

  if (argc < 4) return 0;
    create_sci_environment(environment, 0);

  //  try {
  const char *trailfile = sci_getenv("SCIRUN_TRAIL_FILE");
  const char *trailmode = sci_getenv("SCIRUN_TRAIL_MODE");
  if (trailmode && trailfile) {
    if (trailmode[0] == 'R') {
      if (EventManager::record_trail_file(trailfile)) {
        cerr << "Recording trail file: ";
      } else {
        cerr << "ERROR recording trail file ";
      } 
      cerr << trailfile << std::endl;
    }

    if (trailmode[0] == 'P') {
      if (EventManager::play_trail_file(trailfile)) {
        cerr << "Playing trail file: ";
      } else {
        cerr << "ERROR playinging trail file ";
      } 
      cerr << trailfile << std::endl;
    }
  }
  


  ShaderProgramARB::init_shaders_supported();


  BundleHandle bundle = new Bundle();
  NrrdDataHandle nrrd_handle = new NrrdData();
  Nrrd *nrrd = nrrd_handle->nrrd_;
  string filename = string(argv[2]);
  nrrdLoad(nrrd, filename.c_str(), 0);
  
  
  event_handle_t scene_event = 0;
  
  CompileInfoHandle ci =
    NrrdTextureBuilderAlgo::get_compile_info(nrrd->type,nrrd->type);
  
  
  const int card_mem = 128;
  cerr << "nrrd texture\n";
  TextureHandle texture = new Texture;
  NrrdTextureBuilderAlgo::build_static(texture, 
				       nrrd_handle, 0, 255,
				       0, 0, 255, card_mem);
  vector<Plane *> *planes = new vector<Plane *>;
  
  
  string fn = string(argv[3]);
  Piostream *stream = auto_istream(fn, 0);
  if (!stream) {
    cerr << "Error reading file '" + fn + "'." << std::endl;
    return -1;
  }  
  // read the file.
  ColorMap2 *cmap2 = new ColorMap2();
  ColorMap2Handle icmap = cmap2;
  try {
    Pio(*stream, icmap);
  } catch (...) {
    cerr << "Error loading "+fn << std::endl;
    icmap = 0;
  }
  delete stream;
  ColorMapHandle cmap;
  vector<ColorMap2Handle> *cmap2v = new vector<ColorMap2Handle>(0);
  cmap2v->push_back(icmap);
  string enabled("111111");
  if (sci_getenv("CMAP_WIDGETS")) 
    enabled = sci_getenv("CMAP_WIDGETS");
  for (unsigned int i = 0; i < icmap->widgets().size(); ++ i) {
    if (i < enabled.size() && enabled[i] == '1') {
      icmap->widgets()[i]->set_onState(1); 
    } else {
      icmap->widgets()[i]->set_onState(0); 
    }
  }

  VolumeRenderer *vol = new VolumeRenderer(texture, 
					   cmap, 
					   *cmap2v, 
					   *planes,
					   Round(card_mem*1024*1024*0.8));
  vol->set_slice_alpha(-0.5);
  vol->set_interactive_rate(4.0);
  vol->set_sampling_rate(4.0);
  vol->set_material(0.322, 0.868, 1.0, 18);
  scene_event = new SceneGraphEvent(vol, "FOO");
  
  
  bundle->setNrrd(string(argv[2]), nrrd_handle);
  
    Painter *painter = new Painter(0);
    painter->add_bundle(bundle);

    Skinner::init_skinner();
    Skinner::XMLIO::register_maker<Painter::SliceWindow>((void *)painter);    
    Skinner::Drawables_t drawables = Skinner::XMLIO::load(argv[1]);

    //    Skinner::ThrottledToolManager *skinner_non_draw_event_manager = 
    //      new Skinner::ThrottledToolManager("FourView", 120.0);
    BaseTool *non_draw_tool = new FilterRedrawEventsTool("Skinner Events", 1);
    //    skinner_non_draw_event_manager->add_tool(non_draw_tool,1);

    for (unsigned int d = 0; d < drawables.size(); ++d) {
      //      drawables[d]->process_event(new WindowEvent(WindowEvent::REDRAW_E));
      ASSERT(dynamic_cast<Skinner::GLWindow *>(drawables[d]));
      Skinner::ThrottledRunnableToolManager *runner = 
        new Skinner::ThrottledRunnableToolManager(drawables[d]->get_id(), 120.0);
      runner->add_tool(non_draw_tool,1);
      runner->add_tool(drawables[d], 2);
      string tname = drawables[d]->get_id()+" Throttled Tool Manager";
      Thread *thread = new Thread(runner, tname.c_str());
      thread->detach();
      //      thread->add_tool(new FilterRedrawEventsTool("filter"), 1);
      
    }


//     Skinner::Runner *skinner = new Skinner::Runner(;
//     Thread *skinner_thread = new Thread(skinner, "Skinner Runner");
//     skinner_thread->detach();
    
    if (scene_event.get_rep()) {

    }

    EventManager *em = new EventManager();
    EventManager::add_event(new WindowEvent(WindowEvent::REDRAW_E));
    if (!sci_getenv_p("PAINTER_NOSCENE")) 
      EventManager::add_event(scene_event);    
    em->tm().add_tool(new QuitMainWindowTool(drawables[0]->get_id()), 1);
    Thread *em_thread = new Thread(em, "Event Manager");
    //    em_thread->detach();

    //    em->run();
    //    delete em;
    if (trailmode && trailmode[0] == 'P') {
      EventManager::play_trail();
    }
      
    em_thread->join();
    cerr << argv[0] << " exited.\n";
    if (trailmode && trailmode[0] == 'R') {
      EventManager::stop_trail_file();
    }
    delete painter;

    Thread::exitAll(0);

//   } catch (string &err) {
//     cerr << "ERROR: " << err;
//   } catch (...) {
//     cerr << "Unhandled exception in Painter\n";
//   }
  return 0;
}


#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1424
#pragma reset woff 1209 
#endif
