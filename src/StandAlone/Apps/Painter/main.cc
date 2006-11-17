//
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2004 Scientific Computing and Imaging Institute,
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
//    File   : main.cc
//    Author : McKay Davis
//    Date   : Tue May 30 21:38:23 MDT 2006


#include <Core/Util/Environment.h>
#include <Core/Skinner/XMLIO.h>
#include <StandAlone/Apps/Painter/Painter.h>
#include <Core/Events/EventManager.h>

#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>

#include <sci_defs/x11_defs.h>

#if defined(__APPLE__) && !defined(HAVE_X11)
namespace Carbon {
#  include <Carbon/Carbon.h>
}
#endif

using namespace SCIRun;

int
main(int argc, char *argv[], char **environment) {
  create_sci_environment(environment, argv[0]);  
  
  Skinner::XMLIO::register_maker<Painter>();
  if (!Skinner::load_default_skin()) {
    cerr << "Errors encounted loading default skin.\n";
    return 1;
  }

  EventManager *em = new EventManager();
  Thread *em_thread = new Thread(em, "Event Manager");
  EventManager::do_trails();

#if defined(__APPLE__) && !defined(HAVE_X11)
  // Apple's version of event management
  Carbon::RunApplicationEventLoop();
  EventManager::add_event(new QuitEvent());
#endif

  em_thread->join();

  return 0;
}
