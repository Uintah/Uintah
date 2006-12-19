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
//    File   : Graph2D.h
//    Author : McKay Davis
//    Date   : Thu Jul 20 15:51:01 2006


#ifndef SKINNER_GRAPH2D_H
#define SKINNER_GRAPH2D_H

#include <Core/Skinner/Parent.h>
#include <Core/Skinner/Color.h>
#include <Core/Datatypes/Matrix.h>
#include <Core/Geometry/Point.h>

namespace SCIRun {
  namespace Skinner {  
    class Graph2D : public Parent {
    public:
      Graph2D(Variables *variables);
      ~Graph2D();
      propagation_state_e               process_event(event_handle_t event);
    private:
      void                              render_gl();
      void                              render_axes();
      void                              calculate_data_range();
      void                              add_matrix(const MatrixHandle &);
      Point                             screen_to_world(double, double);
      Point                             world_to_screen(const Point &);
      void                              render_closest_coordinates();
      void                              set_color(unsigned int);
      void                              opp_color(unsigned int);
      void                              set_window();
      void                              reset_window();

      vector<MatrixHandle>              data_;
      typedef pair<double, double>      range_t;
      Var<double>                       font_size_;
      range_t                           window_x_range_;
      range_t                           window_y_range_;
      range_t                           data_x_range_;
      range_t                           data_y_range_;
      int                               viewport_[4];
      int                               mx_;
      int                               my_;
      int                               bx_;
      int                               by_;

      double                            wl_;
      double                            ww_;
      double                            bwl_;
      double                            bww_;
      double                            gapx_;
      double                            gapy_;
    };
  }
} // End namespace SCIRun

#endif
