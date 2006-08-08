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

#include <Core/Containers/StringUtil.h>
#include <Core/Skinner/Variables.h>
#include <Core/Skinner/Gradient.h>
#include <Core/Math/MiscMath.h>
#include <Core/Math/MinMax.h>

#include <sci_gl.h>


namespace SCIRun {


Skinner::Drawable *
Skinner::Gradient::maker(Variables *vars) 
{
  return new Gradient(vars);
}

Skinner::Gradient::Gradient(Variables *vars)
  : Skinner::Drawable(vars),
    backslash_(false),
    anchor_(-1)
{
  vars->maybe_get_bool("backslash", backslash_);
  string anchor = "";
  if (vars->maybe_get_string("anchor", anchor)) {
    if (string_toupper(anchor) == "SW") 
      anchor_ = SW;
    else if (string_toupper(anchor) == "SE") 
      anchor_ = SE;
    else if (string_toupper(anchor) == "NE") 
      anchor_ = NE;
    else if (string_toupper(anchor) == "NW") 
      anchor_ = NW;
    else
      throw "invalid anchor specifier";
  }

  colors_[SW] = Color(1.0, 0.0, 0.0, 1.0); // sw
  colors_[SE] = Color(0.0, 1.0, 0.0, .0); //se;
  colors_[NE] = Color(0.0, 1.0, 0.0, 1.0); // ne
  colors_[NW] = Color(0.0, 0.0, 1.0, 1.0); // nw

  vars->maybe_get_color("bottom-color",colors_[SW]);
  vars->maybe_get_color("left-color",colors_[SW]);
  vars->maybe_get_color("sw-color",colors_[SW]);

  vars->maybe_get_color("bottom-color", colors_[SE]);
  vars->maybe_get_color("right-color",colors_[SE]);
  vars->maybe_get_color("se-color",colors_[SE]);

  vars->maybe_get_color("top-color", colors_[NE]);
  vars->maybe_get_color("right-color", colors_[NE]);
  vars->maybe_get_color("ne-color", colors_[NE]);

  vars->maybe_get_color("top-color", colors_[NW]);
  vars->maybe_get_color("left-color", colors_[NW]);
  vars->maybe_get_color("nw-color", colors_[NW]);
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
  glBegin(GL_TRIANGLES);

  if (backslash_) {
    glColor4dv(&colors_[SW].r);
    glVertex3d(x,y,0);
    glColor4dv(&colors_[SE].r);
    glVertex3d(x2,y,0);
    glColor4dv(&colors_[NW].r);
    glVertex3d(x,y2,0);


    glColor4dv(&colors_[SE].r);
    glVertex3d(x2,y,0);
    glColor4dv(&colors_[NE].r);
    glVertex3d(x2,y2,0);
    glColor4dv(&colors_[NW].r);
    glVertex3d(x,y2,0);
  } else {

    glColor4dv(&colors_[SW].r);
    glVertex3d(x,y,0);
    glColor4dv(&colors_[NE].r);
    glVertex3d(x2,y2,0);
    glColor4dv(&colors_[NW].r);
    glVertex3d(x,y2,0);

    glColor4dv(&colors_[SW].r);
    glVertex3d(x,y,0);
    glColor4dv(&colors_[SE].r);
    glVertex3d(x2,y,0);
    glColor4dv(&colors_[NE].r);
    glVertex3d(x2,y2,0);
  }

  glEnd();
  CHECK_OPENGL_ERROR();

}
void
Skinner::Gradient::render_radial_gl()
{
  const RectRegion &region = get_region();
  const double width = region.width();
  const double height = region.height();

  double x;
  double y;
  double start;
  double end;

  switch (anchor_) {
  case SW: 
    x = region.x1();
    y = region.y1();
    start = 0.0;
    end = M_PI/2.0;
    break;

  case SE: 
    x = region.x2();
    y = region.y1();
    start = M_PI/2.0;
    end = M_PI;
    break;

  case NE: 
    x = region.x2();
    y = region.y2();
    start = M_PI;
    end = 3.0*M_PI/2.0;
    break;

  case NW: 
    x = region.x1();
    y = region.y2();
    start = 3.0*M_PI/2.0;
    end = 2.0*M_PI;
    break;

  default:
    throw "Unknown Gradient Anchor for rounded corner";
    break;
  }

  glShadeModel(GL_SMOOTH);
  glBegin(GL_TRIANGLE_FAN);

  // Anchor point for tri-fan
  glColor4dv(&colors_[anchor_].r);
  glVertex3d(x,y,0);

  // Need at least 2 radial points to produce tri-fan
  int divisions = Max(2, int(Min(region.width(), region.height())));
  double delta_rad = (end-start) / double(divisions-1);
  double rad = start;
  for (int i = 0; i < divisions; ++i) {

    glColor4dv(&colors_[(anchor_+1)%4].r);

    glVertex3d(x + cos(rad) * width,
               y + sin(rad) * height, 0);    
    rad += delta_rad;
  }
  glEnd();
  CHECK_OPENGL_ERROR();

}


BaseTool::propagation_state_e
Skinner::Gradient::process_event(event_handle_t event) {
  WindowEvent *window = dynamic_cast<WindowEvent *>(event.get_rep());
  if (window && window->get_window_state() == WindowEvent::REDRAW_E) {
    if (anchor_ == -1)
      render_gl();
    else 
      render_radial_gl();
  }
  return CONTINUE_E;
}



}
