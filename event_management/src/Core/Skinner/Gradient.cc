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
//    File   : Gradient.cc
//    Author : McKay Davis
//    Date   : Tue Jun 27 13:00:43 2006
#include <Core/Skinner/Variables.h>
#include <Core/Skinner/Gradient.h>
#include <Core/Math/MiscMath.h>
#include <sci_gl.h>
#include <sci_glu.h>


namespace SCIRun {


Skinner::Gradient::Gradient(Variables *variables,
                            const Color &sw,
                            const Color &se,
                            const Color &ne,
                            const Color &nw)
  : Skinner::Drawable(variables)
{
  colors_[0] = sw;
  colors_[1] = se;
  colors_[2] = ne;
  colors_[3] = nw;
}


void
Skinner::Gradient::render_gl()
{
  const RectRegion &region = get_region();
  const double x = region.x1();
  const double y = region.y1();
  const double x2 = region.x2();
  const double y2 = region.y2();

  glShadeModel(GL_SMOOTH);
  glBegin(GL_QUADS);
  glColor4dv(&colors_[0].r);
  glVertex3d(x,y,0);
  glColor4dv(&colors_[1].r);
  glVertex3d(x2,y,0);
  glColor4dv(&colors_[2].r);
  glVertex3d(x2,y2,0);
  glColor4dv(&colors_[3].r);
  glVertex3d(x,y2,0);

  glEnd();
  CHECK_OPENGL_ERROR();

}



BaseTool::propagation_state_e
Skinner::Gradient::process_event(event_handle_t event) {
  WindowEvent *window = dynamic_cast<WindowEvent *>(event.get_rep());
  if (window && window->get_window_state() == WindowEvent::REDRAW_E) {
    render_gl();
  }
  return CONTINUE_E;
}


Skinner::Drawable *
Skinner::Gradient::maker(Variables *vars) 
{
  Color sw(1.0, 0.0, 0.0, 1.0);
  vars->maybe_get_color("bottom-color",sw);
  vars->maybe_get_color("left-color",sw);
  vars->maybe_get_color("sw-color",sw);

  Color se(0.0, 1.0, 0.0, 1.0);
  vars->maybe_get_color("bottom-color",se);
  vars->maybe_get_color("right-color",se);
  vars->maybe_get_color("sw-color",se);

  Color ne(0.0, 1.0, 0.0, 1.0);
  vars->maybe_get_color("top-color",ne);
  vars->maybe_get_color("right-color",ne);
  vars->maybe_get_color("ne-color",ne);

  Color nw(0.0, 0.0, 1.0, 1.0);
  vars->maybe_get_color("top-color",nw);
  vars->maybe_get_color("left-color",nw);
  vars->maybe_get_color("nw-color",nw);
    
  return new Gradient(vars, sw, se, ne, nw);
}


}
