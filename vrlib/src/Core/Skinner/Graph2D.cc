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
//    File   : Graph2D.cc
//    Author : McKay Davis
//    Date   : Thu Jul 20 15:51:10 2006

#include <Core/Skinner/Graph2D.h>
#include <Core/Skinner/Variables.h>
#include <Core/Geom/FontManager.h>
#include <Core/Math/MiscMath.h>
#include <Core/Math/MinMax.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Events/EventManager.h>
#include <Core/Events/keysyms.h>
#include <Core/Events/MatrixEvent.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <sci_gl.h>
#include <sci_glu.h>

#ifdef _WIN32
#include <Core/OS/Rand.h>
#endif


namespace SCIRun {
  namespace Skinner{
    Graph2D::Graph2D(Variables *variables) 
      : Parent(variables),
        data_(0),
        font_size_(variables,"font_size", 35.0)
    {
      for (int n = 0; n < 20; ++n) {

        int rows = Ceil(drand48() * 50);
        
        DenseMatrix *mat = new DenseMatrix(rows, 2);

        double x = 0.0;

        mat->put(0, 0, 0);
        mat->put(0, 1, 50);
        
        for (int r = 1 ; r < rows; ++r ) {
          if (r == rows-1) x = Round(x);
          mat->put(r, 0, x);
          mat->put(r, 1, mat->get(r-1, 1) + (50 - drand48() * 100.0));
          x += drand48() * 2.0 + 0.5;
        }

        add_matrix(mat);

      }
    }


    Graph2D::~Graph2D() {
    }


    Point
    Graph2D::screen_to_world(double x, double y) {

      GLdouble gl_modelview_matrix[16];
      GLdouble gl_projection_matrix[16];
      GLint    gl_viewport[4];
      
      glGetIntegerv(GL_VIEWPORT, gl_viewport);
      glGetDoublev(GL_MODELVIEW_MATRIX, gl_modelview_matrix);
      glGetDoublev(GL_PROJECTION_MATRIX, gl_projection_matrix);
      CHECK_OPENGL_ERROR();

      GLdouble xyz[3];
      gluUnProject(double(x)+0.5, double(y)+0.5, 0,
                   gl_modelview_matrix, 
                   gl_projection_matrix,
                   gl_viewport,
                   xyz+0, xyz+1, xyz+2);
      return Point(xyz[0], xyz[1], xyz[2]);
    }


    Point
    Graph2D::world_to_screen(const Point &world) {

      GLdouble gl_modelview_matrix[16];
      GLdouble gl_projection_matrix[16];
      GLint    gl_viewport[4];
      
      glGetIntegerv(GL_VIEWPORT, gl_viewport);
      glGetDoublev(GL_MODELVIEW_MATRIX, gl_modelview_matrix);
      glGetDoublev(GL_PROJECTION_MATRIX, gl_projection_matrix);
      CHECK_OPENGL_ERROR();

      GLdouble xyz[3];
      gluProject(world(0), world(1), world(2),
                 gl_modelview_matrix, 
                 gl_projection_matrix,
                 gl_viewport,
                 xyz+0, xyz+1, xyz+2);
      xyz[0] -= gl_viewport[0];
      xyz[1] -= gl_viewport[1];

      return Point(xyz[0], xyz[1], xyz[2]);
    }

      


    void
    Graph2D::calculate_data_range() {
      data_x_range_ = make_pair(AIR_POS_INF, AIR_NEG_INF);
      data_y_range_ = make_pair(AIR_POS_INF, AIR_NEG_INF);
      
      for (unsigned int n = 0; n < data_.size(); ++n) {
        for (int r = 0; r < data_[n]->nrows(); ++r) {
          const double x = data_[n]->get(r,0);
          data_x_range_.first = Min(data_x_range_.first, x);
          data_x_range_.second = Max(data_x_range_.second, x);
          
         const double y = data_[n]->get(r,1);
          data_y_range_.first = Min(data_y_range_.first, y);
          data_y_range_.second = Max(data_y_range_.second, y);
        }
      }
    }

    void
    Graph2D::add_matrix(const MatrixHandle &mh) {
      data_.push_back(mh);
      calculate_data_range();
      reset_window();
    }
    
    void
    Graph2D::reset_window() {
      ww_ = (data_x_range_.second - data_x_range_.first);
      wl_ = data_x_range_.first + ww_ / 2.0;
      set_window();
    }

    void
    Graph2D::set_window() {      
      window_y_range_ = data_y_range_;
      window_x_range_.first = wl_ - ww_/2.0;
      window_x_range_.second = wl_ + ww_/2.0;
    }        


    void
    Graph2D::render_axes()
    {
      const RectRegion &region = get_region();

      const int x = Floor(region.x1());
      const int y = Floor(region.y1());
      const int x2 = Floor(region.x2());
      const int y2 = Floor(region.y2());
      const double font_size = font_size_();

      gapx_ = font_size*3+5;
      gapy_ = font_size+25;

      TextRenderer *renderer = FontManager::get_renderer(font_size);

      Color grid_line_color = Var<Color>(get_vars(),"grid_line_color")();
      double grid_line_width = Var<double>(get_vars(), "grid_line_width")();
      glLineWidth(grid_line_width);

      int divx = Var<int>(get_vars(),"grid_divisions_x")();
      for (int i = 0 ; i < divx; ++i) {
        double px = gapx_ + i * (x2 - x - gapx_)/double(divx);

        glColor4dv(&grid_line_color.r);
        glBegin(GL_LINES);      
        glVertex2d(px, y);
        glVertex2d(px, y2);
        glEnd();

        Point p = world_to_screen(Point(px, gapy_,0));
        double val = wl_ - ww_/2.0 + ww_ * i / divx;
        renderer->render(to_string(val), p.x(), p.y()-5,
                         TextRenderer::N);
      }
      CHECK_OPENGL_ERROR();

      int divy = Var<int>(get_vars(),"grid_divisions_y")();

      double yww = window_y_range_.second - window_y_range_.first;
      double ywl = window_y_range_.first + yww/2.0;

      for (int i = 0 ; i < divy; ++i) {
        double py = gapy_ + i * (y2 - y - gapy_)/double(divy);

        glColor4dv(&grid_line_color.r);
        glBegin(GL_LINES);
        glVertex2d(x, py);
        glVertex2d(x2, py);
        glEnd();

        Point p = world_to_screen(Point(gapx_, py, 0));
        double val = ywl - yww/2.0 + yww * i / divy;
        renderer->render(to_string(val), p.x()-5, p.y(),
                         TextRenderer::E);

      }

      CHECK_OPENGL_ERROR();


      Color axis_line_color = Var<Color>(get_vars(),"axis_line_color")();
      double axis_line_width = Var<double>(get_vars(), "axis_line_width")();
      glColor4dv(&axis_line_color.r);
      glLineWidth(axis_line_width);
      glBegin(GL_LINES);
      glVertex2d(x+gapx_, gapy_);
      glVertex2d(x2, gapy_);
      glVertex2d(gapx_, y+gapy_);
      glVertex2d(gapx_, y2);
      glEnd();
      CHECK_OPENGL_ERROR();

      glPointSize(axis_line_width);
      glBegin(GL_POINTS);
      glVertex2d(gapx_, gapy_);
      glEnd();
      CHECK_OPENGL_ERROR();
      
      glGetIntegerv(GL_VIEWPORT, (GLint*)viewport_);
      glViewport(x + Ceil(gapx_), y + Ceil(gapy_),
                 x2 - x - Ceil(gapx_), y2 - y - Ceil(gapy_));
      CHECK_OPENGL_ERROR();
    }

    void
    Graph2D::set_color(unsigned int n) {
      double r = ((n+4) %  5 + 1.0) /  5.0;
      double g = ((n*5) % 11 + 1.0) / 11.0;
      double b = ((n*5) %  7 + 1.0) /  7.0;
      glColor4d(r,g,b,1.0);
    }

    void
    Graph2D::opp_color(unsigned int n) {
      double r = ((n+4) %  5 + 1.0) /  5.0;
      double g = ((n*5) % 11 + 1.0) / 11.0;
      double b = ((n*5) %  7 + 1.0) /  7.0;
      glColor4d(1.0-r,1.0-g,1.0-b,1.0);
    }
      
      

    void
    Graph2D::render_gl()
    {
      glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
      glEnable(GL_BLEND);

      glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
      glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);
      glEnable(GL_LINE_SMOOTH);
      glEnable(GL_POINT_SMOOTH);
     
      render_axes();

      glMatrixMode(GL_MODELVIEW);
      glPushMatrix();
      glLoadIdentity();
      glScaled(2.0, 2.0, 2.0);
      glTranslated(-.5, -.5, -.5);

      glScaled(1.0 / (window_x_range_.second - window_x_range_.first),
               1.0 / (window_y_range_.second - window_y_range_.first),
               1.0);

      glTranslated(-window_x_range_.first,
                   -window_y_range_.first,
                   0.0);

      Var<double> data_line_width(get_vars(), "data_line_width");
      Var<double> data_point_size(get_vars(), "data_point_size");
      glLineWidth(data_line_width());
      glPointSize(data_point_size());

      for (unsigned int n = 0; n < data_.size(); ++n) {
        set_color(n);

        glBegin(GL_LINE_STRIP);
        for (int r = 0; r < data_[n]->nrows(); ++r) {
          glVertex2d(data_[n]->get(r, 0), 
                     data_[n]->get(r, 1));

        }
        glEnd();
        
       
        glBegin(GL_POINTS);
        for (int r = 0; r < data_[n]->nrows(); ++r) {
          glVertex2d(data_[n]->get(r, 0), 
                     data_[n]->get(r, 1));
        }
        glEnd();
      }

      render_closest_coordinates();

      glLineWidth(1.0);
      glPointSize(1.0);
      glDisable(GL_LINE_SMOOTH);
      glDisable(GL_POINT_SMOOTH);
      CHECK_OPENGL_ERROR();



      glMatrixMode(GL_MODELVIEW);
      glPopMatrix();
      glViewport(viewport_[0],viewport_[1],viewport_[2],viewport_[3]);
      CHECK_OPENGL_ERROR();

    }


    void
    Graph2D::render_closest_coordinates() {
      int minn = -1;
      int minr = -1;
      double mindist = AIR_POS_INF;

      double font_size = font_size_();
      
            
      Point p = screen_to_world(mx_, my_);

      for (unsigned int n = 0; n < data_.size(); ++n) {
        for (int r = 0; r < data_[n]->nrows(); ++r) {
          double x = data_[n]->get(r, 0);
          double y = data_[n]->get(r, 1);
          double dx = x - p.x();
          double dy = y - p.y();
          double dist = Abs(dx*dx+dy*dy);
          if (dist < mindist) {
            minn = n;
            minr = r;
            mindist = dist;
          }
        }
      }
      
      if (minn != -1) {

        double x = data_[minn]->get(minr, 0);
        double y = data_[minn]->get(minr, 1);
        Point tp = world_to_screen(Point(x,y,0));

        double dx = tp.x() - mx_ + gapx_;
        double dy = tp.y() - my_ + gapy_;
        
        
        double ps = Var<double>(get_vars(), "data_point_size")();

        if (sqrt(dx*dx+dy*dy) > (25.0 + ps)) {
          return;
        }
        
        opp_color(minn);
        glPointSize(ps+6);
        glBegin(GL_POINTS);
        glVertex2d(x,y);
        glEnd();

        set_color(minn);
        glPointSize(ps);
        glBegin(GL_POINTS);
        glVertex2d(x,y);
        glEnd();
        


 
        TextRenderer *renderer = FontManager::get_renderer(font_size);        
        string str = "("+to_string(x)+","+to_string(y)+")";
        double wid = renderer->width(str);

        
        bool close_to_N = Abs(get_region().y2() - gapy_-tp.y()) < font_size/2;
        bool close_to_S = Abs(get_region().y1() - tp.y()) < font_size/2;
        bool close_to_E = Abs(get_region().x2() - gapx_ - tp.x()) < wid/2;
        bool close_to_W = Abs(get_region().x1() - tp.x()) < wid/2;

        int anchor = TextRenderer::C;

        if (close_to_N) {
          if (close_to_W) {
            anchor = TextRenderer::NW;
          } else if (close_to_E) {
            anchor = TextRenderer::NE;
          } else {
            anchor = TextRenderer::N;
          }
        } else if (close_to_S) {
          if (close_to_W) {
            anchor = TextRenderer::SW;
          } else if (close_to_E) {
            anchor = TextRenderer::SE;
          } else {
            anchor = TextRenderer::S;
          }
        } else if (close_to_W) {
          anchor = TextRenderer::W;
        } else if (close_to_E) {
          anchor = TextRenderer::E;
        }
            
        renderer->render(str, tp.x(), tp.y(), anchor);
        
        glPointSize(ps*3);
      }
    }
                                       

    BaseTool::propagation_state_e
    Graph2D::process_event(event_handle_t event) {

      MatrixEvent *matrix = dynamic_cast<MatrixEvent *>(event.get_rep());
      if (matrix) {
        data_ = matrix->get_data();
        calculate_data_range();
        reset_window();
        EventManager::add_event(new WindowEvent(WindowEvent::REDRAW_E));
      }

      WindowEvent *window = dynamic_cast<WindowEvent *>(event.get_rep());
      if (window && window->get_window_state() == WindowEvent::REDRAW_E)
      {
        render_gl();
      }

      KeyEvent *key = dynamic_cast<KeyEvent *>(event.get_rep());
      if (key && key->get_keyval() == SCIRun_space) {
        reset_window();
        EventManager::add_event(new WindowEvent(WindowEvent::REDRAW_E));
      }

      PointerEvent *pointer = dynamic_cast<PointerEvent *>(event.get_rep());

      if (pointer && 
          pointer->get_pointer_state() & PointerEvent::BUTTON_PRESS_E &&
          pointer->get_pointer_state() & PointerEvent::BUTTON_1_E) 
      {
        bx_ = pointer->get_x();
        by_ = pointer->get_y();
        bww_ = ww_;
        bwl_ = wl_;
      }

      if (pointer && 
          pointer->get_pointer_state() & PointerEvent::BUTTON_RELEASE_E &&
          pointer->get_pointer_state() & PointerEvent::BUTTON_1_E) 
      {
      }

      if (pointer && 
          pointer->get_pointer_state() & PointerEvent::MOTION_E) {
        mx_ = pointer->get_x();
        my_ = pointer->get_y();

        if (pointer->get_pointer_state() & PointerEvent::BUTTON_1_E) {
          double scale = 0.5/sqrt(get_region().width()*get_region().width() +
                                  get_region().height()*get_region().height());
          scale *= data_x_range_.second - data_y_range_.first;
          ww_ = bww_ - (my_ - by_)*scale;
          wl_ = bwl_ - (mx_ - bx_)*scale;
          set_window();
        }

        EventManager::add_event(new WindowEvent(WindowEvent::REDRAW_E));
      }
  
      return Parent::process_event(event);
    }
  }
}
  
