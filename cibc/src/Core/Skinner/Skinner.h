//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2006 Scientific Computing and Imaging Institute,
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
//    File   : Skinner.h
//    Author : McKay Davis
//    Date   : Tue Jun 27 13:03:19 2006

#ifndef SKINNER_SKINNER_H
#define SKINNER_SKINNER_H

#include <Core/Events/EventManager.h>
#include <Core/Thread/Runnable.h>
#include <Core/Util/ThrottledRunnable.h>
#include <teem/air.h>
#include <string>
#include <deque>
using std::deque;


using std::string;

#include <Core/Skinner/share.h>

namespace SCIRun {
  class BaseTool;
  class Root;
  namespace Skinner {
    class Drawable;

    SCISHARE Drawable *         load_skin(const string&filename);
    SCISHARE bool               load_default_skin();
  

    class ThrottledRunnableToolManager : public ThrottledRunnable{
    public:
      ThrottledRunnableToolManager(const string &name,
                                   double hertz = 30.0);

      virtual ~ThrottledRunnableToolManager();
      void                                    add_tool(tool_handle_t,
                                                       unsigned int);
    protected:
      virtual bool                            iterate();
      
    private:
      SCIRun::EventManager::event_mailbox_t * mailbox_;
      SCIRun::ToolManager *                   tool_manager_;
      string                                  name_;
      Mutex *                                 tm_lock_;
    };

    class ThrottledRedraw : public ThrottledRunnable {
    public:
      ThrottledRedraw(Drawable *drawable, double hertz);

      virtual ~ThrottledRedraw();
    private:
      virtual bool                            iterate();
      
    private:
      Drawable *                              drawable_;
      bool                                    redraw_;
      SCIRun::EventManager::event_mailbox_t * mailbox_;
    };

  }
}



#endif
