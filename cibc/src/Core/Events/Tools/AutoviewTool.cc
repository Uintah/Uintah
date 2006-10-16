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
//    File   : AutoviewTool.cc
//    Author : Martin Cole
//    Date   : Mon Jun  5 10:43:13 2006

#include <Core/Events/Tools/AutoviewTool.h>
#include <Core/Events/keysyms.h>
#include <Core/Math/Trig.h>
#include <Core/Geometry/Plane.h>

namespace SCIRun {

AutoviewToolInterface::AutoviewToolInterface(View &v) :
  view_(v)
{
}

AutoviewToolInterface::~AutoviewToolInterface()
{
}


AutoviewTool::AutoviewTool(string name, AutoviewToolInterface* i) :
  BaseTool(name),
  KeyTool(name),
  interface_(i)
{
}

AutoviewTool::~AutoviewTool()
{
}

BaseTool::propagation_state_e 
AutoviewTool::key_press(string key, int keyval, unsigned int modifiers, 
			unsigned int time)
{
  // Cnt-v for autoview
  if (keyval == SCIRun_v &&  (modifiers & KeyEvent::CONTROL_E))
  {
    calc_view();
    return STOP_E;
  }
  return CONTINUE_E;
}

BaseTool::propagation_state_e 
AutoviewTool::process_event(event_handle_t event) 
{
  if (dynamic_cast<AutoviewEvent *>(event.get_rep())) {
    calc_view();
    return STOP_E;
  }
  return CONTINUE_E;
}


bool
AutoviewTool::calc_view()
{
  BBox bbox;
  if (interface_->get_bounds(bbox))
  {
    View cv(interface_->view_);
    cv.lookat(bbox.center());
    interface_->view_ = cv;

    // Move forward/backwards until entire view is in scene.
    // change this a little, make it so that the FOV must be 20 deg.
    double myfov = 20.0;

    Vector diag(bbox.diagonal());
    
    double w = diag.length();
    if( w < 0.000001 ){
      BBox bb;
      bb.reset();
      Vector epsilon(0.001, 0.001, 0.001 );
      bb.extend( bbox.min() - epsilon );
      bb.extend( bbox.max() + epsilon );
      w = bb.diagonal().length();
    }
    Vector lookdir(cv.lookat() - cv.eyep()); 
    lookdir.safe_normalize();
    const double scale = 1.0 / (2 * Tan(DtoR(myfov / 2.0)));
    double length = w * scale;
    cv.fov(myfov);
    cv.eyep(cv.lookat() - lookdir * length);

    Plane upplane(cv.eyep(), lookdir);
    cv.up(upplane.project(cv.up()));

    interface_->view_ = cv;
    return true;
  }
  return false;
}

} // namespace SCIRun
