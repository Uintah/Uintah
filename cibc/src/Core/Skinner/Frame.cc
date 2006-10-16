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
//    File   : Frame.cc
//    Author : McKay Davis
//    Date   : Tue Jun 27 13:03:11 2006


#include <Core/Skinner/Frame.h>
#include <Core/Skinner/Variables.h>
#include <Core/Math/MiscMath.h>
#include <sci_gl.h>
#include <sci_glu.h>


namespace SCIRun {


Skinner::Drawable *
Skinner::Frame::maker(Variables *vars) 
{
  return new Frame(vars);
}



Skinner::Frame::Frame(Variables *vars) 
  : Skinner::Parent(vars),
    top_(),
    bot_(),
    left_(),
    right_(),
    invert_(vars,"invert",0),
    border_wid_(vars, "width", 5),
    border_hei_(vars, "height", 5)
{
  top_  = Var<Color>(vars, "top-color");
  top_ |= Var<Color>(vars, "color2");
  top_ |= Color(0.5, 0.5, 0.5, 1.0);

  bot_  = Var<Color>(vars, "bottom-color");
  bot_ |= Var<Color>(vars, "color1");
  bot_ |= Color(0.75, 0.75, 0.75, 1.0);

  left_  = Var<Color>(vars, "left-color");
  left_ |= Var<Color>(vars, "color2");
  left_ |= Color(0.5, 0.5, 0.5, 1.0);

  right_  = Var<Color>(vars, "right-color");
  right_ |= Var<Color>(vars, "color1");
  right_ |= Color(0.75, 0.75, 0.75, 1.0);
}


Skinner::Frame::~Frame() {
}

BaseTool::propagation_state_e
Skinner::Frame::redraw(event_handle_t e)
{
  const RectRegion &region = get_region();
  const double x = region.x1();
  const double y = region.y1();
  const double x2 = region.x2();
  const double y2 = region.y2();

  double hei = border_hei_();
  double wid = border_wid_();
  bool invert = invert_();
  Color bot = invert ? top_() : bot_();
  Color top = invert ? bot_() : top_();

  Color left = invert ? right_() : left_();
  Color right = invert ? left_() : right_();

  
  glBegin(GL_QUADS);
  glColor4dv(&bot.r);
  glVertex3d(x,y,0);
  glVertex3d(x2,y,0);
  glVertex3d(x2,y+hei,0);
  glVertex3d(x,y+hei,0);

  glColor4dv(&right.r);
  glVertex3d(x2-wid,y+hei,0);
  glVertex3d(x2,y+hei,0);
  glVertex3d(x2,y2,0);
  glVertex3d(x2-wid,y2,0);

  glColor4dv(&top.r);
  glVertex3d(x,y2,0);
  glVertex3d(x2,y2,0);
  glVertex3d(x2-wid,y2-hei,0);
  glVertex3d(x,y2-hei,0);

  glColor4dv(&left.r);
  glVertex3d(x,y,0);
  glVertex3d(x+wid,y+hei,0);
  glVertex3d(x+wid,y2-hei,0);
  glVertex3d(x,y2-hei,0);

  glEnd();
  CHECK_OPENGL_ERROR();
  return CONTINUE_E;
  return CONTINUE_E;
}



BaseTool::propagation_state_e
Skinner::Frame::process_event(event_handle_t event) {
  WindowEvent *window = dynamic_cast<WindowEvent *>(event.get_rep());

  if (window && window->get_window_state() == WindowEvent::REDRAW_E)
  {
    if ((border_hei_()*2) > get_region().height() ||
        (border_wid_()*2) > get_region().width()) return CONTINUE_E;
    redraw(0);
  }

  const RectRegion &region = get_region();
  const RectRegion subregion(region.x1()+border_wid_(),
                             region.y1()+border_hei_(),
                             region.x2()-border_wid_(),
                             region.y2()-border_hei_());
  for (Drawables_t::iterator child = children_.begin(); 
       child != children_.end(); ++child) {
    (*child)->set_region(subregion);
    (*child)->process_event(event);
  }
  return CONTINUE_E;
}


}
