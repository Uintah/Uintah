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
//    File   : Win32EventSpawner.h
//    Author : McKay Davis
//    Date   : Thu Jun  1 19:28 MDT 2006

#ifndef CORE_EVENTS_WIN32EVENTSPAWNER_H
#define CORE_EVENTS_WIN32EVENTSPAWNER_H

#include <Core/Thread/Runnable.h>
#include <Core/Thread/Semaphore.h>
#include <Core/Events/BaseEvent.h>
#include <Core/Events/EventSpawner.h>
#include <Core/Util/ThrottledRunnable.h>

#include <windows.h>
#include <string>

using std::string;

#include <Core/Events/share.h>

namespace SCIRun {
  
  struct Win32Event;
  class GLWindow;
  class Win32OpenGLContext;

  // TODO destroy context
  class SCISHARE Win32GLContextRunnable : public EventSpawner {
  public:
    Win32GLContextRunnable(const string &target, int visual, int x, int y, int width, int height, bool border);
    virtual void run(); // override to create the window, then call parent's run
    virtual bool iterate();

    // will lock until it gets created
    Win32OpenGLContext* getContext();
  private:
    int visual_, x_, y_, width_, height_;
    bool border_;
    bool mouse_in_window;
    Win32OpenGLContext* context_;
    Semaphore created_;

    event_handle_t              win_KeyEvent(MSG, bool pressed);
    event_handle_t              win_ButtonReleaseEvent(MSG, int button);
    event_handle_t              win_ButtonPressEvent(MSG, int button);
    event_handle_t              win_PointerMotion(MSG);
    event_handle_t              win_Expose(MSG);
    event_handle_t              win_Enter(MSG);
    event_handle_t              win_Leave(MSG);
    event_handle_t              win_Configure(MSG);
    event_handle_t              win_Client(MSG);
    event_handle_t              win_FocusIn(MSG);
    event_handle_t              win_FocusOut(MSG);
    event_handle_t              win_Visibilty(MSG);
    event_handle_t              win_Map(MSG);
  };
}

#endif
