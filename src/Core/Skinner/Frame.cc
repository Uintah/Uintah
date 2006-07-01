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
//    File   : Frame.cc
//    Author : McKay Davis
//    Date   : Tue Jun 27 13:03:11 2006


#include <Core/Skinner/Frame.h>
#include <Core/Skinner/Variables.h>
#include <Core/Math/MiscMath.h>
#include <sci_gl.h>
#include <sci_glu.h>


namespace SCIRun {

Skinner::Frame::Frame(Variables *variables,
                      double wid, 
                      double hei,
                      const Color &top,
                      const Color &bot,
                      const Color &left,
                      const Color &right) 
  : Skinner::Parent(variables),
    top_(top),
    bot_(bot),
    left_(left),
    right_(right),
    border_wid_(wid),
    border_hei_(hei)
{
}


Skinner::Frame::~Frame() {
}

void
Skinner::Frame::render_gl()
{
  const RectRegion &region = get_region();
  const double x = region.x1();
  const double y = region.y1();
  const double x2 = region.x2();
  const double y2 = region.y2();

  glBegin(GL_QUADS);
  glColor4dv(&bot_.r);
  glVertex3d(x,y,0);
  glVertex3d(x2,y,0);
  glVertex3d(x2,y+border_hei_,0);
  glVertex3d(x,y+border_hei_,0);

  glColor4dv(&right_.r);
  glVertex3d(x2-border_wid_,y+border_hei_,0);
  glVertex3d(x2,y+border_hei_,0);
  glVertex3d(x2,y2,0);
  glVertex3d(x2-border_wid_,y2,0);

  glColor4dv(&top_.r);
  glVertex3d(x,y2,0);
  glVertex3d(x2,y2,0);
  glVertex3d(x2-border_wid_,y2-border_hei_,0);
  glVertex3d(x,y2-border_hei_,0);

  glColor4dv(&left_.r);
  glVertex3d(x,y,0);
  glVertex3d(x+border_wid_,y+border_hei_,0);
  glVertex3d(x+border_wid_,y2-border_hei_,0);
  glVertex3d(x,y2-border_hei_,0);

  glEnd();
  CHECK_OPENGL_ERROR();

}



BaseTool::propagation_state_e
Skinner::Frame::process_event(event_handle_t event) {
  WindowEvent *window = dynamic_cast<WindowEvent *>(event.get_rep());
  if (window && window->get_window_state() == WindowEvent::REDRAW_E)
    render_gl();

  const RectRegion &region = get_region();
  const RectRegion subregion(region.x1()+border_wid_,
                             region.y1()+border_hei_,
                             region.x2()-border_wid_,
                             region.y2()-border_hei_);
  for (Drawables_t::iterator child = children_.begin(); 
       child != children_.end(); ++child) {
    (*child)->set_region(subregion);
    (*child)->process_event(event);
  }
  return CONTINUE_E;
}


Skinner::Drawable *
Skinner::Frame::maker(Variables *vars) 
{
  Color bottom(0.75, 0.75, 0.75, 1.0);
  vars->maybe_get_color("color1", bottom);
  vars->maybe_get_color("bottom-color", bottom);

  Color right(0.75, 0.75, 0.75, 1.0);
  vars->maybe_get_color("color1", right);
  vars->maybe_get_color("right-color", right);

  Color top(0.5, 0.5, 0.5, 1.0);
  vars->maybe_get_color("color2", top);
  vars->maybe_get_color("top-color", top);

  Color left(0.5, 0.5, 0.5, 1.0);
  vars->maybe_get_color("color2", left);
  vars->maybe_get_color("left-color", left);

  double width = 5;
  vars->maybe_get_double("width", width);

  double height = 5;
  vars->maybe_get_double("height", height);

  if (vars->get_bool("invert")) {
    SWAP (top, bottom);
    SWAP (left, right);
  }

  return new Frame(vars, width, height, top, bottom, left, right);
}


}
