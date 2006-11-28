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
#include <Core/Thread/Semaphore.h>
#include <Core/Geom/X11Lock.h>
#include <Core/Util/Environment.h>
#include <Core/Events/BaseEvent.h>
#include <iostream>
#include <sci_gl.h>


#if defined(_WIN32)
#  include <Core/Geom/Win32OpenGLContext.h>
#  include <Core/Events/Win32EventSpawner.h>
#elif defined(__linux)
#  include <Core/Geom/X11OpenGLContext.h>
#  include <Core/Events/X11EventSpawner.h>
#elif defined(__APPLE__)
#  include <Core/Geom/OSXOpenGLContext.h>
#  include <Core/Events/OSXEventSpawner.h>
#endif



namespace SCIRun {
  namespace Skinner {
    
   GLWindow::GLWindow(Variables *vars) :
      Parent(vars),
      lock_(vars->get_id().c_str()),
      width_(Var<int>(vars,"width",640)()),
      height_(Var<int>(vars,"height",480)()),
      posx_(Var<int>(vars,"posx",50)()),
      posy_(Var<int>(vars,"posy",50)()),
      border_(Var<bool>(vars,"border",true)()),
      context_(0),
      spawner_runnable_(0),
      spawner_thread_(0),
      draw_runnable_(0),
      draw_thread_(0),
      redrawables_(),
      force_redraw_(false)
#ifdef HAVE_PNG
      ,
      png_buf_(0),
      png_rows_(0),
      png_num_(0),
      do_png_(sci_getenv("SKINNER_MOVIE_BASE_FILENAME"))
#endif
    {
#if defined(_WIN32)
        
      Win32GLContextRunnable* cr = 
        scinew Win32GLContextRunnable(get_id(), 0, posx_, posy_,
                                      (unsigned)width_,(unsigned)height_,
                                      border_);
      spawner_runnable_ = cr;
#elif defined(__linux)
      X11OpenGLContext* context =
        new X11OpenGLContext(0, posx_, posy_, 
                             (unsigned)width_,(unsigned)height_,
                             border_);
      ASSERT(context);
      spawner_runnable_ = 
        new X11EventSpawner(get_id(), context->display_, context->window_);
      context_ = context;
#elif defined(__APPLE__)
      OSXOpenGLContext *context =  
        new OSXOpenGLContext(posx_, posy_, 
                             (unsigned)width_,(unsigned)height_,
                             border_);
      ASSERT(context);
      //      spawner_runnable_ =
      new OSXEventSpawner(get_id(), context->window_);
      context_ = context;
#endif

      string tname = get_id()+" Event Spawner";
      if (spawner_runnable_) {
        spawner_thread_ = new Thread(spawner_runnable_, tname.c_str());
      }

#ifdef _WIN32
      // this waits for the context to be created, and needs to happen after the thread spawns
      context_ = cr->getContext();;
      width_ = context_->width();
      height_ = context_->height();         
#endif

      tname = get_id()+" Redraw";
      draw_runnable_ = new ThrottledRedraw(this, 120.0);
      draw_thread_ = new Thread(draw_runnable_, tname.c_str());


      REGISTER_CATCHER_TARGET(GLWindow::close);
      REGISTER_CATCHER_TARGET(GLWindow::mark_redraw);
      REGISTER_CATCHER_TARGET(GLWindow::redraw_drawable);
    }

    GLWindow::~GLWindow() 
    {
      if (spawner_runnable_) spawner_runnable_->quit();
      if (spawner_thread_) spawner_thread_->join();
      if (draw_runnable_) draw_runnable_->quit();
      if (draw_thread_) draw_thread_->join();
      // context must be deleted after event spawer is done
      delete context_;
      context_ = 0;
      throw_signal("GLWindow::Destructor");
    }

    int
    GLWindow::get_signal_id(const string &name) const {
      if (name == "GLWindow::Destructor") return 1;
      return 0;
    }

    MinMax
    GLWindow::get_minmax(unsigned int ltype)
    {
      return SPRING_MINMAX;
    }
    
    BaseTool::propagation_state_e
    GLWindow::process_event(event_handle_t event) {
      lock_.lock();
      PointerEvent *pointer = dynamic_cast<PointerEvent *>(event.get_rep());
      if (pointer) {

#if defined(__linux)
        if ((pointer->get_pointer_state() & PointerEvent::MOTION_E) &&
            EventManager::trailfile_is_playing())
        {
          const char *trailmode = sci_getenv("SCIRUN_TRAIL_MODE");
          X11OpenGLContext *x11 = dynamic_cast<X11OpenGLContext *>(context_);
          if (x11 && trailmode) {
            Window src_win = x11->window_;
            
            // if the second letter of SCIRUN_TRAIL_MODE is G, 
            // Grab the pointer from the root window from the user
            if (trailmode[1] == 'G') {
              src_win = None;
            }
            
            XWarpPointer(x11->display_, src_win, x11->window_,
                         0,0, x11->width(), x11->height(),
                         pointer->get_x(), pointer->get_y());
          }
        }
        pointer->set_y(context_->height() - 1 - pointer->get_y());
#endif
        
      }


      WindowEvent *window = dynamic_cast<WindowEvent *>(event.get_rep());
      bool redraw = (window && 
                     window->get_window_state() & WindowEvent::REDRAW_E);
      bool subdraw = redraw && redrawables_.size() && !force_redraw_;

      int vpw;
      int vph;

      if (redraw) {
        ASSERT(context_);
        if (!context_->make_current()) {
	  lock_.unlock();
          return CONTINUE_E;
        }
        X11Lock::lock();
        vpw = context_->width();
        vph = context_->height();

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

        if (!subdraw) {
          glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        }
        
        CHECK_OPENGL_ERROR();
      }

      if (subdraw) {
        // todo: thread unsafe, needs lock
        for (int c = redrawables_.size()-1; c >=0 ; --c) {
          redrawables_[c]->process_event(event);
        }
        redrawables_.clear();

      } else {
        if (redraw && force_redraw_) {
          redrawables_.clear();
          force_redraw_ = false;
        }
        for (Drawables_t::iterator child = children_.begin(); 
             child != children_.end(); ++child) {
          (*child)->set_region(get_region());
          (*child)->process_event(event);
        }
      }
      
      if (redraw){ 
        glMatrixMode(GL_MODELVIEW);
        glPopMatrix();
        glMatrixMode(GL_PROJECTION);
        glPopMatrix();
        CHECK_OPENGL_ERROR();
        context_->swap();
        save_png();

        context_->release();        
        X11Lock::unlock();
      }

      lock_.unlock();

      return CONTINUE_E;
    }

    BaseTool::propagation_state_e
    GLWindow::close(event_handle_t) {
      QuitEvent *qe = new QuitEvent();
      qe->set_target(get_id());
      EventManager::add_event(qe);
      return STOP_E;
    }

    BaseTool::propagation_state_e
    GLWindow::mark_redraw(event_handle_t) {
      force_redraw_ = true;
      EventManager::add_event(new WindowEvent(WindowEvent::REDRAW_E, get_id()));
      return CONTINUE_E;
    }

    BaseTool::propagation_state_e
    GLWindow::redraw_drawable(event_handle_t signalh) {
      Signal *signal = dynamic_cast<Signal *>(signalh.get_rep());
      Drawable *drawable = 
        dynamic_cast<Drawable *>(signal->get_signal_thrower());
      ASSERT(drawable);
      Var<bool> visible(signal->get_vars(), "visible",true);
      if (visible()) {
        redrawables_.push_back(drawable);
        EventManager::add_event(new WindowEvent(WindowEvent::REDRAW_E, get_id()));
      }
      return CONTINUE_E;
    }
      

    void
    GLWindow::save_png() {
#ifdef HAVE_PNG
      if (!do_png_) return;
      const int bpp = 3;
      const int vpw = context_->width();
      const int vph = context_->height();
      const int num = vpw*vph*bpp;

      if (num != png_num_) {
        if (png_buf_) delete png_buf_;
        if (png_rows_) delete png_rows_;
        png_buf_ = new unsigned char[num];
        png_rows_ = new png_bytep[vph];
        for(int y = 0; y < vph; y++) {
          png_rows_[vph - y - 1] = (png_bytep)(png_buf_ + y * vpw * bpp);
        }
        png_num_ = num;
      }

      ASSERT(png_buf_);
      ASSERT(png_rows_);
      ASSERT(png_num_);

      png_structp png_write = png_create_write_struct(PNG_LIBPNG_VER_STRING, 
                                                      0,0,0);
      png_set_compression_level(png_write, 4);
      png_infop png_info = png_create_info_struct(png_write);
      bool pngerror = setjmp(png_jmpbuf(png_write));
      ASSERT(!pngerror);      

      const char *fname = sci_getenv("SKINNER_MOVIE_BASE_FILENAME");
      static int i = 0;
      char name[strlen(fname)+32];
      sprintf (name, fname, ++i);
      FILE *fp = fopen(name, "wb");
      png_init_io(png_write, fp);
        
      png_set_IHDR(png_write, png_info, vpw, vph,
                   8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                   PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);
      /* write header */
      png_write_info(png_write, png_info);      

      glPixelStorei(GL_PACK_ALIGNMENT, 1);
      glReadBuffer(GL_BACK);
      glReadPixels(0, 0, vpw, vph, GL_RGB, GL_UNSIGNED_BYTE, png_buf_);
      
      png_set_rows(png_write, png_info, png_rows_);
      png_write_image(png_write, png_rows_);
      png_write_end(png_write, 0);
      png_destroy_write_struct(&png_write, &png_info);
      fclose(fp);
#endif
    }

  }
}
