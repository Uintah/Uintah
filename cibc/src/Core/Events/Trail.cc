//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2006 Scientific Computing and Imaging Institute,
//  University of Utah.
//  
//  
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
//    File   : Trail.cc
//    Author : McKay Davis
//    Date   : Sun Oct  1 19:54:39 2006

#include <Core/Events/Trail.h>
#include <Core/Events/EventManager.h>
#include <Core/Util/FileUtils.h>
#include <Core/Util/Environment.h>

namespace SCIRun {


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

}
