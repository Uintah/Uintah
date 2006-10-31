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
//    File   : ColorMap2D.cc
//    Author : Milan Ikits
//    Author : Michael Callahan
//    Author : McKay Davis (Skinner Drawable conversion)
//    Date   : Thu Jul  8 01:50:58 2004

#include <Core/Skinner/ColorMap2D.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Geom/ShaderProgramARB.h>
#include <Core/Geom/ColorMap.h>

#include <sci_gl.h>

#include <stack>
#include <set>
#include <sstream>
#include <iostream>
#include <fstream>
#include <string>

#if defined(_WIN32) && !defined(BUILD_STATIC)
#define SCISHARE __declspec(dllimport)
#else // _WIN32
#define SCISHARE
#endif

namespace SCIRun {
  namespace Skinner {
    ColorMap2D::ColorMap2D(Variables *variables)
      : Parent(variables),        
        button_(0),
        widgets_(),
        shader_factory_(0),
        colormap_texture_(256, 512, 4),
        histo_(0), 
        histo_dirty_(true), 
        histogram_texture_id_(0),
        cmap_dirty_(true), 
        mouse_widget_(-1),
        mouse_object_(0),
        paint_widget_(0),
        first_motion_(true),
        mouse_last_x_(0),
        mouse_last_y_(0),
        pan_x_(0),
        pan_y_(0),
        scale_(1),
        updating_(false),
        value_range_(0.0, -1.0)
    {
    }


    ColorMap2D::~ColorMap2D()
    {
      if (shader_factory_) 
        delete shader_factory_;
    }

    void
    ColorMap2D::translate_start(int x, int y)
    {
      mouse_last_x_ = x;
      mouse_last_y_ = y;
    }

    void
    ColorMap2D::translate_motion(int x, int y)
    {
      float xmtn = float(mouse_last_x_ - x) / float(width_);
      float ymtn = -float(mouse_last_y_ - y) / float(height_);
      mouse_last_x_ = x;
      mouse_last_y_ = y;

      pan_x_.set(pan_x_.get() + xmtn / scale_.get());
      pan_y_.set(pan_y_.get() + ymtn / scale_.get());
    }

    void
    ColorMap2D::scale_start(int x, int y)
    {
      mouse_last_y_ = y;
    }

    void
    ColorMap2D::scale_motion(int x, int y)
    {
      float ymtn = -float(mouse_last_y_ - y) / float(height_);
      mouse_last_y_ = y;
      scale_.set(scale_.get() + -ymtn);

      if (scale_.get() < 0.0) scale_.set(0.0);
    }

    int
    do_round(float d)
    {
      if (d > 0.0) return (int)(d + 0.5);
      else return -(int)(-d + 0.5); 
    }

    void
    ColorMap2D::screen_val(int &x, int &y)
    {
      const float cx = width_ * 0.5;
      const float cy = height_ * 0.5;
      const float sf_inv = 1.0 / scale_.get();

      x = do_round((x - cx) * sf_inv + cx + (pan_x_.get() * width_));
      y = do_round((y - cy) * sf_inv + cy - (pan_y_.get() * height_));
    }


    pair<double, double>
    ColorMap2D::rescaled_val(int x, int y)
    {
      double xx = x/double(width_);
      double range = value_range_.second - value_range_.first;
      if (range > 0.0)
        xx = xx * range + value_range_.first;
      y = height_ - y - 1;
      double yy = y/double(height_);
      return make_pair(xx, yy);
    }

    void
    ColorMap2D::add_widget(CM2WidgetHandle &widget)
    {
      widgets_.push_back(widget);
      widgets_.back()->set_faux(false);
      widgets_.back()->set_value_range(value_range_);
      select_widget(widgets_.size()-1, 1);
    }

    BaseTool::propagation_state_e
    ColorMap2D::add_triangle_widget(event_handle_t) 
    {
      add_widget(new TriangleCM2Widget());
      return CONTINUE_E;
    }

    BaseTool::propagation_state_e
    ColorMap2D::add_rectangle_widget(event_handle_t) 
    {
      add_widget(new RectangleCM2Widget());
      return CONTINUE_E;
    }

    BaseTool::propagation_state_e
    ColorMap2D::add_paint_widget(event_handle_t) 
    {
      add_widget(new TriangleCM2Widget());
      return CONTINUE_E;
    }

    BaseTool::propagation_state_e
    ColorMap2D::delete_selected_widget(event_handle_t) 
    {
      gui_selected_widget_.reset();
      const int widget = gui_selected_widget_.get();
  
      if (widget < 0 || widget >= (int)widgets_.size()) return;
      // Delete widget.
      widgets_.erase(widgets_.begin() + widget);

      update_to_gui();
      if (gui_selected_widget_.get() >= (int)widgets_.size())
        select_widget(widgets_.size()-1, 1);
      else 
        select_widget(gui_selected_widget_.get(), 1);
      redraw(true);
      force_execute();
    }


    void
    ColorMap2D::save_file(bool save_ppm)
    {
#if 0
      filename_.reset();
      const string filename = filename_.get();
      if (filename == "") {
        error("Warning;  No filename provided to ColorMap2D");
        return;
      }
  
      redraw(true, save_ppm);
  
      // Open ostream
      Piostream* stream;
      stream = scinew BinaryPiostream(filename, Piostream::Write);
      if (stream->error())
        error("Could not open file for writing" + filename);
      else {
        Pio(*stream, sent_cmap2_);
        delete stream;
        remark ("Saved ColorMap2 to file: "+filename);
      }
#endif
    }


    void
    ColorMap2D::load_file()
    {
#if 0
      // The implementation of this was taken almost directly from
      // NrrdReader Module.  
      filename_.reset();
      string fn(filename_.get());
      if(fn == "") {
        error("Please Specify a Transfer Function filename.");
        return;
      }
  
      struct stat buf;
      if(stat(fn.c_str(), &buf) == -1) {
        error(string("ColorMap2D error - file not found: '")+fn+"'");
        return;
      }

      const int len = fn.size();
      const string suffix(".cmap2");
      // Return if the suffix is wrong
      if (fn.substr(len - suffix.size(), suffix.size()) != suffix) return;

      Piostream *stream = auto_istream(fn, this);
      if (!stream) {
        error("Error reading file '" + fn + "'.");
        return;
      }  
      // read the file.
      ColorMap2Handle icmap = scinew ColorMap2();
      try {
        Pio(*stream, icmap);
      } catch (...) {
        error("Error loading "+fn);
        icmap = 0;
      }
      delete stream;
      if (icmap.get_rep()) {
        widgets_ = icmap->widgets();
        icmap_generation_ = icmap->generation;
        while (!undo_stack_.empty()) undo_stack_.pop();
      }
  
      update_to_gui();
      select_widget(-1,0); 
      redraw(true);
      colormap_widgets_.clear();
      force_execute();
#endif
    }

    bool
    ColorMap2D::select_widget(int widget, int object)
    {
      int changed = false;
      if (widget == -1 && object == -1) {
        changed = gui_selected_widget_.changed() || gui_selected_object_.changed();
        widget = gui_selected_widget_.get();
        object = gui_selected_object_.get();
      } else {
        changed = gui_selected_widget_.get() != widget;
        gui_selected_widget_.set(widget);
        gui_selected_object_.set(object);
      }

      for (int i = 0; i < (int)widgets_.size(); i++)
        widgets_[i]->select(i == widget ? object : 0);
      return changed;
    }
  


    void
    ColorMap2D::push(int x, int y, int button)
    {
      button_ = button;
      first_motion_ = true;
      paint_widget_ = 0;
      int old_widget = gui_selected_widget_.get();
      int old_object = mouse_object_;
      mouse_pick(x,y,button);
      if (mouse_widget_ == -1 && old_widget != -1) {
        mouse_widget_ = old_widget;
        mouse_object_ = old_object;
      }
      select_widget(mouse_widget_, mouse_object_);

      // If the currently selected widget is a paint layer, start a new stroke
      if (mouse_widget_ >= 0 && mouse_widget_ < int(widgets_.size())) {
        paint_widget_ = 
          dynamic_cast<PaintCM2Widget *>(widgets_[mouse_widget_].get_rep());
        if (paint_widget_) {
          double range = 1.0;
          if (value_range_.first < value_range_.second)
            range = value_range_.second - value_range_.first;
          range /= scale_.get();
          paint_widget_->add_stroke(range/35.0);
          paint_widget_->add_coordinate(rescaled_val(x,y));
        }
      }

      redraw();
    }


    void
    ColorMap2D::mouse_pick(int x, int y, int b)
    {
      const bool right_button_down = (b==3);
      if (!right_button_down)
        for (mouse_widget_ = widgets_.size()-1; mouse_widget_>=0; mouse_widget_--)
          if (widgets_[mouse_widget_]->get_onState() &&
              (mouse_object_ = widgets_[mouse_widget_]->pick1
               (x, height_-1-y, width_, height_))) break;
  
      if (!mouse_object_)
        for (mouse_widget_ = widgets_.size()-1; mouse_widget_>=0; mouse_widget_--)
          if (widgets_[mouse_widget_]->get_onState() &&
              (mouse_object_ = widgets_[mouse_widget_]->pick2
               (x, height_-1-y, width_, height_, right_button_down))) break;
    }


    void
    ColorMap2D::motion(int x, int y)
    {
      if (button_ == 0) {
        set_window_cursor(x,y);
        return;
      }

      const int selected = gui_selected_widget_.get();
      if (selected < 0 || selected >= (int)widgets_.size()) return;

      if (button_ == 1 && paint_widget_) {
        paint_widget_->add_coordinate(rescaled_val(x,y));
      } else {
        if (!gui_selected_object_.get()) return;
        if (first_motion_)
        {
          undo_stack_.push(UndoItem(UndoItem::UNDO_CHANGE, selected,
                                    widgets_[selected]->clone()));
          first_motion_ = false;
        }
        widgets_[selected]->move(x, height_-1-y, width_, height_);
      }
      redraw(true);
      updating_ = true;
      if (execute_count_ == 0) {
        execute_count_ = 1;
        force_execute();
      }
    }



    void
    ColorMap2D::release(int x, int y)
    {
      button_ = 0;
      set_window_cursor(x,y);
      const int selected = gui_selected_widget_.get();
      if (selected >= 0 && selected < (int)widgets_.size())
      {
        if (!paint_widget_ && !gui_selected_object_.get()) return;
        widgets_[selected]->release(x, height_-1-y, width_, height_);
        updating_ = false;
        force_execute();
      }
      paint_widget_ = 0;
    }

    void
    ColorMap2D::init_shader_factory() 
    {
      if (!use_back_buffer_ || shader_factory_) return;

      if (sci_getenv_p("SCIRUN_SOFTWARE_CM2")) {
        remark("SCIRUN_SOFWARE_CM2 set. Rasterizing Colormap in software.");
        use_back_buffer_ = false;
      } else if (!shader_factory_ && ShaderProgramARB::shaders_supported()) {
        shader_factory_ = new CM2ShaderFactory();
        remark ("ARB Shaders are supported.");
        remark ("Using hardware rasterization for widgets.");
      } else {
        remark ("Shaders not supported.");
        remark ("Using software rasterization for widgets.");
        use_back_buffer_ = false;
      }
    }


    void
    ColorMap2D::build_colormap_texture()
    {
      glMatrixMode(GL_PROJECTION);
      glPushMatrix();
      glLoadIdentity();
      glMatrixMode(GL_MODELVIEW);
      glPushMatrix();
      glLoadIdentity();

      glScalef(2.0, 2.0, 2.0);
      glTranslatef(-0.5 , -0.5, 0.0);

      glDrawBuffer(GL_BACK);
      glReadBuffer(GL_BACK);

      bool rebuild_texture = (!use_back_buffer_ && 
                              (width_ != colormap_texture_.dim2() || 
                               height_ != colormap_texture_.dim1()));

      if (!glIsTexture(colormap_texture_id_)) rebuild_texture = true;

      if (cmap_dirty_ || rebuild_texture) {
        // update texture
        if(rebuild_texture) {
          if(glIsTexture(colormap_texture_id_)) {
            glDeleteTextures(1, &colormap_texture_id_);
            colormap_texture_id_ = 0;
          }
          glGenTextures(1, &colormap_texture_id_);
        }
    
        glBindTexture(GL_TEXTURE_2D, colormap_texture_id_);

        if (rebuild_texture) {
          glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
          glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
#ifndef GL_CLAMP_TO_EDGE
          glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
          glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
#else
          glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
          glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
#endif
        }
        
        if (use_back_buffer_) 
        {
          // Back Buffer /w Fragment shader rendering of colormap2 texture
          glEnable(GL_BLEND);
          glDrawBuffer(GL_BACK);
          glReadBuffer(GL_BACK);
          glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA); 
          for(unsigned int i=0; i<widgets_.size(); i++) {
            widgets_[i]->set_value_range(value_range_);
            widgets_[i]->rasterize(*shader_factory_, 0);
          }

          if (rebuild_texture)
            glCopyTexImage2D(GL_TEXTURE_2D,0,GL_RGBA, 0,0,width_, height_,0);
          else
            glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0,width_, height_);
          glClearColor(0.0, 0.0, 0.0, 0.0);
          glClear(GL_COLOR_BUFFER_BIT);
        } else {
          // Software Rendering of Colormap Texture
          if (rebuild_texture) // realloc cmap
            colormap_texture_.resize(height_, width_, 4);
          colormap_texture_.initialize(0.0);
          for (unsigned int i=0; i<widgets_.size(); i++) {
            widgets_[i]->set_value_range(value_range_);
            widgets_[i]->rasterize(colormap_texture_);
          }

          if (rebuild_texture)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 
                         colormap_texture_.dim2(), colormap_texture_.dim1(),
                         0, GL_RGBA, GL_FLOAT, &colormap_texture_(0,0,0));
          else
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 
                            colormap_texture_.dim2(), colormap_texture_.dim1(),
                            GL_RGBA, GL_FLOAT, &colormap_texture_(0,0,0));
        }
    
        glBindTexture(GL_TEXTURE_2D, 0);
        cmap_dirty_ = false;
      }
      glMatrixMode(GL_MODELVIEW);
      glPopMatrix();

      glMatrixMode(GL_PROJECTION);
      glPopMatrix();

    } 


    void
    ColorMap2D::build_histogram_texture()
    {
      if (!histo_dirty_) return;
  
      histo_dirty_ = false;
  
      if(glIsTexture(histogram_texture_id_)) {
        glDeleteTextures(1, &histogram_texture_id_);
        histogram_texture_id_ = 0;
      }
  
      glGenTextures(1, &histogram_texture_id_);
      glBindTexture(GL_TEXTURE_2D, histogram_texture_id_);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
#ifndef GL_CLAMP_TO_EDGE 
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
#else
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
#endif
      size_t axis_size[3];

      if (!histo_) {
        unsigned char zero = 0;
        glTexImage2D(GL_TEXTURE_2D, 0, 1, 1,1,
                     0, GL_LUMINANCE, GL_UNSIGNED_BYTE,  &zero);
        glBindTexture(GL_TEXTURE_2D, 0);
      } else {
        nrrdAxisInfoGet_nva(histo_, nrrdAxisInfoSize, axis_size);
        glTexImage2D(GL_TEXTURE_2D, 0, 1, axis_size[histo_->dim-2],
                     axis_size[histo_->dim-1], 0, GL_LUMINANCE, GL_UNSIGNED_BYTE,
                     histo_->data);
        glBindTexture(GL_TEXTURE_2D, 0);
      }
    }


    void
    ColorMap2D::draw_texture(GLuint &texture_id)
    {
      glEnable(GL_TEXTURE_2D);
      glBindTexture(GL_TEXTURE_2D, texture_id);

      glBegin(GL_QUADS);
      {
        glTexCoord2f( 0.0,  0.0);
        glVertex2f( 0.0,  0.0);
        glTexCoord2f( 1.0,  0.0);
        glVertex2f( 1.0,  0.0);
        glTexCoord2f( 1.0,  1.0);
        glVertex2f( 1.0,  1.0);
        glTexCoord2f( 0.0,  1.0);
        glVertex2f( 0.0,  1.0);
      }
      glEnd();

      glBindTexture(GL_TEXTURE_2D, 0);
      glDisable(GL_TEXTURE_2D);
    }


    BaseTool::propagation_state_e
    ColorMap2D::redraw(event_handle_t)
    {
      if (!ctx_) return;
      get_gui()->lock();

      if(ctx_->width()<3 || ctx_->height()<3 || !ctx_->make_current()) {
        get_gui()->unlock(); 
        return; 
      }
      if (force_cmap_dirty) cmap_dirty_ = true;
      if (select_widget()) cmap_dirty_ = true;

      init_shader_factory();

      glDrawBuffer(GL_BACK);
      glClearColor(0.0, 0.0, 0.0, 0.0);
      glClear(GL_COLOR_BUFFER_BIT);
  
      glViewport(0, 0, width_, height_);

      build_histogram_texture();
      build_colormap_texture();

      glMatrixMode(GL_PROJECTION);
      glLoadIdentity();
      glMatrixMode(GL_MODELVIEW);
      glLoadIdentity();

      const float scale_factor = 2.0 * scale_.get();
      glScalef(scale_factor, scale_factor, scale_factor);
      glTranslatef(-0.5 - pan_x_.get(), -0.5 - pan_y_.get() , 0.0);

      glEnable(GL_BLEND);
      glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);  

      glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
      glColor4f(gui_histo_.get(), gui_histo_.get(), gui_histo_.get(), 1.0);
      draw_texture(histogram_texture_id_);

      // Draw Colormap
      glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
      draw_texture(colormap_texture_id_);

      glDisable(GL_BLEND);
      // Draw Colormap Widget Frames
      for(unsigned int i=0; i<widgets_.size(); i++)
        widgets_[i]->draw();

      CHECK_OPENGL_ERROR("dummy 3")
        // draw outline of the image space.
        glColor4f(0.25, 0.35, 0.25, 1.0); 
      glBegin(GL_LINES);
      {
        glVertex2f( 0.0,  0.0);
        glVertex2f( 1.0,  0.0);

        glVertex2f( 1.0,  0.0);
        glVertex2f( 1.0,  1.0);

        glVertex2f( 1.0,  1.0);
        glVertex2f( 0.0,  1.0);

        glVertex2f( 0.0,  1.0);
        glVertex2f( 0.0,  0.0);
      }
      glEnd();

      if (save_ppm) {
        unsigned int* FrameBuffer = scinew unsigned int[width_*height_];
        glFlush();
        glReadBuffer(GL_BACK);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 8);
#ifndef _WIN32
        glReadPixels(0, 0,width_, height_,
                     GL_RGBA, GL_UNSIGNED_INT_8_8_8_8, FrameBuffer);
#else
        glReadPixels(0, 0,width_, height_,
                     GL_RGBA, GL_UNSIGNED_INT, FrameBuffer);
#endif
        string fn = filename_.get()+".ppm";
        remark("Writing PPM to file: "+fn);
        save_ppm_file(fn, width_, height_, 4,(const unsigned char *)(FrameBuffer));
        delete FrameBuffer;
      }
  
      ctx_->swap();
      // check for errors before the release, as opengl will barf (at least in
      // windows) if you do it after
      CHECK_OPENGL_ERROR("dummy")
        ctx_->release();
      get_gui()->unlock();
    }

  }
} // end namespace SCIRun



