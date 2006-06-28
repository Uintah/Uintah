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

#include <Core/Events/EventManager.h>
#include <Core/Skinner/Skinner.h>
#include <Core/Skinner/XMLIO.h>
#include <Core/Util/Environment.h>
#include <Core/Bundle/Bundle.h>
#include <Core/Events/Tools/QuitMainWindowTool.h>
#include <StandAlone/Apps/Painter/Painter.h>

#include <sgi_stl_warnings_off.h>
#include <string>
#include <iostream>
#include <sgi_stl_warnings_on.h>

using std::cout;
using namespace SCIRun;

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma set woff 1424
#pragma set woff 1209 
#endif


void
start_trail_file() {
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
}  


Painter *
create_painter(const char *filename) {
  BundleHandle bundle = new Bundle();

  if (filename) {
    NrrdDataHandle nrrd_handle = new NrrdData();
    Nrrd *nrrd = nrrd_handle->nrrd_;
    nrrdLoad(nrrd, filename, 0); 
    bundle->setNrrd(filename, nrrd_handle);
  }

  Painter *painter = new Painter(0);
  painter->add_bundle(bundle); 
  Skinner::XMLIO::register_maker<Painter::SliceWindow>((void *)painter); 
  return painter;
}  

void
listen_for_events(const string &main_window_name) {
  if (!main_window_name.empty()) {
    EventManager::add_event(new WindowEvent(WindowEvent::REDRAW_E));
    EventManager *em = new EventManager();
    em->tm().add_tool(new QuitMainWindowTool(main_window_name), 1);
    Thread *em_thread = new Thread(em, "Event Manager");
    start_trail_file();
    em_thread->join();
    EventManager::stop_trail_file();
  }
}  
  

string
get_skin_filename() {
  if (sci_getenv("SKINNER_SKIN")) {
    return sci_getenv("SKINNER_SKIN");
  } 
  const char *srcdir = sci_getenv("SCIRUN_SRCDIR");
  if (srcdir) {
    return string(srcdir) + "/StandAlone/Apps/Painter/Painter.skin";
  } 
  return "";
}

int
main(int argc, char *argv[], char **environment) {
  create_sci_environment(environment, argv[0]);
  ShaderProgramARB::init_shaders_supported();

  Painter *painter = create_painter(argv[1]);

  listen_for_events(Skinner::load_skin(get_skin_filename()));

  delete painter;

  Thread::exitAll(0);
  return 0;
}


#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1424
#pragma reset woff 1209 
#endif
