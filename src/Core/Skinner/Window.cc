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
//  
//    File   : Window.cc
//    Author : McKay Davis
//    Date   : Tue Jun 27 13:01:52 2006

#include <Core/Skinner/Skinner.h>
#include <Core/Skinner/Window.h>
#include <Core/Skinner/Variables.h>
#include <Core/Util/ThrottledRunnable.h>
#include <Core/Util/Assert.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Geom/X11OpenGLContext.h>
#include <Core/Geom/X11Lock.h>
#include <Core/Events/X11EventSpawner.h>
#include <Core/Util/Environment.h>
#include <Core/Geom/ShaderProgramARB.h>
#include <sci_gl.h>
#include <sci_glu.h>
#include <sci_glx.h>
#include <iostream>

namespace SCIRun {
  namespace Skinner {
    
   GLWindow::GLWindow(Variables *variables) :
      Parent(variables),
      width_(640),
      height_(640),
      posx_(0),
      posy_(0),
      border_(true),
      context_(0)
    {
      variables->maybe_get_int("width", width_);
      variables->maybe_get_int("height", height_);
      variables->maybe_get_int("posx", posx_);
      variables->maybe_get_int("posy", posy_);
      variables->maybe_get_bool("border", border_);


      ShaderProgramARB::init_shaders_supported();
      X11OpenGLContext *x11_context = 
        new X11OpenGLContext(0, posx_, posy_, 
                             (unsigned)width_,(unsigned)height_,
                             border_);
      context_ = x11_context;

      string tname = get_id()+" Event Spawner";
      spawner_runnable_ = new X11EventSpawner(get_id(), 
                                              x11_context->display_,
                                              x11_context->window_);
      spawner_thread_ = new Thread(spawner_runnable_, tname.c_str());


      tname = get_id()+" Redraw";
      draw_runnable_ = new ThrottledRedraw(this, 120.0);
      draw_thread_ = new Thread(draw_runnable_, tname.c_str());

      catcher_functions_["GLWindow::close"] = 
        static_cast<SignalCatcher::CatcherFunctionPtr>(&GLWindow::close);

    }

    GLWindow::~GLWindow() 
    {
      spawner_runnable_->quit();
      spawner_thread_->join();
      draw_runnable_->quit();
      draw_thread_->join();
      // context must be deleted after event spawer is done
      delete context_;
    }

    MinMax
    GLWindow::get_minmax(unsigned int ltype)
    {
      return SPRING_MINMAX;
    }
    
    BaseTool::propagation_state_e
    GLWindow::process_event(event_handle_t event) {
      ASSERT(context_);

      PointerEvent *pointer = dynamic_cast<PointerEvent *>(event.get_rep());
      if (pointer) {
# if 1
        if ((pointer->get_pointer_state() & PointerEvent::MOTION_E) &&
            sci_getenv_p("SCIRUN_TRAIL_PLAYBACK")) {
          X11OpenGLContext *x11 = dynamic_cast<X11OpenGLContext *>(context_);
          if (x11) {
            XWarpPointer(x11->display_, x11->window_, x11->window_,
                         0,0, x11->width(), x11->height(),
                         pointer->get_x(), pointer->get_y());
          }
        }
#endif
          
        pointer->set_y(context_->height() - 1 - pointer->get_y());
      }


      WindowEvent *window = dynamic_cast<WindowEvent *>(event.get_rep());
      bool redraw = (window && 
                     window->get_window_state() & WindowEvent::REDRAW_E);
      
      if (redraw) {
        if (!context_->make_current()) {
          return CONTINUE_E;
        }
        X11Lock::lock();
        int vpw = context_->width();
        int vph = context_->height();
        set_region(RectRegion(0.0, 0.0, double(vpw), double(vph)));
        
        glViewport(0, 0, vpw, vph);
        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadIdentity();
        
        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glLoadIdentity();
        
        glScaled(2.0, 2.0, 2.0);
        glTranslated(-.5, -.5, -.5);
        glScaled(1.0/double(vpw), 1.0/double(vph), 1.0);
        CHECK_OPENGL_ERROR();
        
        glDisable(GL_CULL_FACE);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glDrawBuffer(GL_BACK);
        
        glClearColor(0,0,0,0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        CHECK_OPENGL_ERROR();
      }

      for (Drawables_t::iterator child = children_.begin(); 
           child != children_.end(); ++child) {
        (*child)->set_region(get_region());
        (*child)->process_event(event);
      }
      
      if (redraw){ 
        glMatrixMode(GL_MODELVIEW);
        glPopMatrix();
        glMatrixMode(GL_PROJECTION);
        glPopMatrix();
        CHECK_OPENGL_ERROR();
        context_->swap();
        context_->release();        
        X11Lock::unlock();
      }
      return CONTINUE_E;
    }

    BaseTool::propagation_state_e
    GLWindow::close(event_handle_t) {
      cerr << "Close Window\n";
    }

  }
}
