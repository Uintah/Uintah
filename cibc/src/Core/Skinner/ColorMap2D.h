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
//    File   : ColorMap2D.h
//    Author : Milan Ikits
//    Author : Michael Callahan
//    Author : McKay Davis
//    Date   : Thu Jul  8 01:50:58 2004

#include <Core/Skinner/Parent.h>
#include <Core/Geom/ColorMap.h>
#include <Core/Volume/CM2Widget.h>
#include <Core/Volume/CM2Shader.h>
#include <sci_gl.h>

#include <stack>
#include <set>
#include <sstream>
#include <iostream>
#include <fstream>
#include <string>

#ifdef _WIN32
#define SCISHARE __declspec(dllimport)
#else // _WIN32
#define SCISHARE
#endif

namespace SCIRun {
  namespace Skinner {
    class ColorMap2D : public Parent {
    public:
      ColorMap2D(Variables *variables);
      virtual ~ColorMap2D();      
    private:
      CatcherFunction_t                 redraw;
      CatcherFunction_t                 add_triangle_widget;
      CatcherFunction_t                 add_rectangle_widget;
      CatcherFunction_t                 add_paint_widget;
      CatcherFunction_t                 delete_selected_widget;
      void				init_shader_factory();
      void				build_colormap_texture();
      void				build_histogram_texture();
      void				draw_texture(GLuint &);

      void				push(int x, int y, int button);
      void				motion(int x, int y);
      void				release(int x, int y);
      void				mouse_pick(int x, int y, int b);
      void				set_window_cursor(int x, int y);
      bool				select_widget(int w=-1, int o=-1);
      void				screen_val(int &x, int &y);
      pair<double, double>		rescaled_val(int x, int y);
      //! functions for panning.
      void				translate_start(int x, int y);
      void				translate_motion(int x, int y);

      //! functions for zooming.
      void				scale_start(int x, int y);
      void				scale_motion(int x, int y);
      void				scale_end(int x, int y);
      
      int				button_;
      std::vector<CM2WidgetHandle>      widgets_;
      CM2ShaderFactory*                 shader_factory_;
      Array3<float>			colormap_texture_;
      bool				use_back_buffer_;
      Nrrd*				histo_;
      bool				histo_dirty_;
      GLuint                            histogram_texture_id_;

      bool				cmap_dirty_;
      int				mouse_widget_;
      int				mouse_object_;
      PaintCM2Widget *                  paint_widget_;
      // Push on undo when motion occurs, not on select.
      bool				first_motion_;
      
      int				mouse_last_x_;
      int				mouse_last_y_;
      double                            pan_x_;
      double                            pan_y_;
      double                            scale_;
      
      bool				updating_; // updating the tf or not
      double                            histogram_alpha_;
      
      int                               selected_widget_;
      // The currently selected widgets selected object
      int                               selected_object_;
      
      int                               num_entries_;
      int                               faux_;

      struct WidgetState {
        string                          name_;
        bool                            visible_;
        Color                           color_;
        int                             shading_;
      };
      
      vector<WidgetState>               widget_state_;
      pair<float,float>                 value_range_;
    };
  } // namespace Skinner
} // namespace SCIRun
