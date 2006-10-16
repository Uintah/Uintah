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
//    File   : Arrow.cc
//    Author : McKay Davis
//    Date   : Sat Aug 12 12:53:58 2006

#include <Core/Skinner/Variables.h>
#include <Core/Skinner/Arrow.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/BBox.h>
#include <Core/Math/MinMax.h>
#include <Core/Math/MiscMath.h>
#include <sci_gl.h>

namespace SCIRun {
  namespace Skinner {
    Arrow::Arrow(Variables *vars) :
      Drawable(vars),
      color_(vars, "color", Color(0,0,0,1)),
      degrees_(vars, "degrees", 0.0),
      point_angle_(vars, "point_angle", 90.0)
    {
      REGISTER_CATCHER_TARGET(Arrow::redraw);
    }

    Arrow::~Arrow() {}
  
    BaseTool::propagation_state_e
    Arrow::redraw(event_handle_t event)
    {
      const RectRegion &region = get_region();
      double dx = region.width() / 2.0;
      double dy = region.height() / 2.0;

      double cx = region.x1() + dx;
      double cy = region.y1() + dy;
      
      point_angle_ = Clamp(Abs(point_angle_()), 1.0, 180.0);
      const double ang = point_angle_()*2*M_PI / 360.0;
      const double rad = degrees_()*2*M_PI / 360.0;
      const double rad2 = rad+M_PI-ang;
      const double rad3 = rad+M_PI+ang;
      Point p1(cx + dx*cos(rad), cy + dy*sin(rad), 0.0);
      Point p2(cx + dx*cos(rad2), cy + dy*sin(rad2), 0.0);
      Point p3(cx + dx*cos(rad3), cy + dy*sin(rad3), 0.0);
      BBox bbox;
      bbox.extend(p1);
      bbox.extend(p2);
      bbox.extend(p3);
      Point center = bbox.min() + bbox.diagonal()/2.0;
      Vector delta(cx-center.x(), cy-center.y(), 0);
      p1 = p1 + delta;
      p2 = p2 + delta;
      p3 = p3 + delta;
      glColor4dv(&color_().r);
      glBegin(GL_TRIANGLES);
      glVertex3dv(&p1(0));
      glVertex3dv(&p2(0));
      glVertex3dv(&p3(0));
      glEnd();

      return CONTINUE_E;
    }
  }
}
