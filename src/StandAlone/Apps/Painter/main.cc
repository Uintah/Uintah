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
#include <Core/Skinner/XMLIO.h>
#include <Core/Util/Environment.h>
#include <Core/Util/FileUtils.h>
#include <StandAlone/Apps/Painter/Painter.h>

#include <sgi_stl_warnings_off.h>
#include <string>
#include <iostream>
#include <sgi_stl_warnings_on.h>
#include <sci_defs/x11_defs.h>

#if defined(__APPLE__) && !defined(HAVE_X11)
  namespace Carbon {
#  include <Carbon/Carbon.h>
  }
#endif

using std::cout;
using namespace SCIRun;

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma set woff 1424
#pragma set woff 1209 
#endif


void
start_trail_file() {
  string default_trailfile(string("/tmp/")+
                           sci_getenv("EXECUTABLE_NAME")+".trail");
  string default_trailmode("R");

  const char *trailfile = sci_getenv("SCIRUN_TRAIL_FILE");
  const char *trailmode = sci_getenv("SCIRUN_TRAIL_MODE");

  if (!trailfile) {
    trailfile = default_trailfile.c_str();
  }

  if (!trailmode) {
    trailmode = default_trailmode.c_str();
  }

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
        cerr << "Playing trail file: " << trailfile << std::endl;
        EventManager::play_trail();
        cerr << "Trail file completed.\n";
      } else {
        cerr << "ERROR playing trail file " << trailfile << std::endl;
      } 
    }
  }
}  
  

int
main(int argc, char *argv[], char **environment) {
  create_sci_environment(environment, argv[0]);  
  
  string default_skin = sci_getenv("SCIRUN_OBJDIR")+string("data");
  string skinner_path = default_skin;
  const char *path_ptr = sci_getenv("SKINNER_PATH");
  if (path_ptr) {
    skinner_path = string(path_ptr) + ":" + default_skin;
  }
  sci_putenv("SKINNER_PATH", skinner_path);
  sci_putenv("SCIRUN_FONT_PATH",skinner_path);
  string filename = "main.skin";
  string path = findFileInPath(filename, skinner_path);
  if (path.empty()) {
    std::cerr << "Cannot find main.skin in SKINNER_PATH.\n";
    std::cerr << "SKINNER_PATH=" << skinner_path << std::endl;;
    return 0;
  }

  Skinner::XMLIO::register_maker<Painter>();
  Skinner::load_skin(path+filename);

  EventManager::add_event(new WindowEvent(WindowEvent::REDRAW_E));
  EventManager *em = new EventManager();
  Thread *em_thread = new Thread(em, "Event Manager");
  start_trail_file();

#if defined(__APPLE__) && !defined(HAVE_X11)
  // Apples version of event management
  Carbon::RunApplicationEventLoop();
#endif

  em_thread->join();
  EventManager::stop_trail_file();

  return 0;
}


#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1424
#pragma reset woff 1209 
#endif

