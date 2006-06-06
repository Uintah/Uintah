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
//    File   : ViewRotateTool.cc
//    Author : Martin Cole
//    Date   : Thu Jun  1 09:31:09 2006

#include <Core/Events/Tools/ViewRotateTool.h>

namespace SCIRun {

ViewToolInterface::ViewToolInterface(View &v) :
  view_(v)
{
}

ViewToolInterface::~ViewToolInterface()
{
}

ViewRotateTool::ViewRotateTool(string name, ViewToolInterface* i) :
  PointerTool(name),
  scene_interface_(i),
  ball_(new BallData()),
  rot_view_(),
  last_x_(0),
  last_y_(0),
  rotate_valid_p_(false),
  eye_dist_(0.0),
  last_time_(0)
{
}

ViewRotateTool::~ViewRotateTool()
{
}

BaseTool::propagation_state_e
ViewRotateTool::pointer_down(int which, int x, int y, int time)
{
  if (which != 2) return CONTINUE_E;
  scene_interface_->update_mode_string("rotate:");
  last_x_ = x;
  last_y_ = y;

  // Find the center of rotation...
  View tmpview = scene_interface_->view_;
  int xres = scene_interface_->width();
  int yres = scene_interface_->height();
  double znear, zfar;
  rotate_valid_p_ = false;
  if(!scene_interface_->compute_depth(tmpview, znear, zfar))
    return STOP_E;

  rot_view_ = tmpview;
  rotate_valid_p_ = true;

  double rad = 0.8;
  HVect center(0, 0, 0, 1.0);
	
  // we also want to keep the old transform information
  // around (so stuff correlates correctly)
  // OGL uses left handed coordinate system!
  Vector z_axis, y_axis, x_axis;

  y_axis = tmpview.up();
  z_axis = tmpview.eyep() - tmpview.lookat();
  x_axis = Cross(y_axis,z_axis);
  x_axis.normalize();
  y_axis.normalize();
  eye_dist_ = z_axis.normalize();

  prev_trans_.load_frame(Point(0.0, 0.0, 0.0), x_axis, y_axis, z_axis);

  ball_->Init();
  ball_->Place(center,rad);
  HVect mouse((2.0 * x) / xres - 1.0, 2.0 * (yres - y * 1.0) / yres - 1.0,
	      0.0, 1.0);
  ball_->Mouse(mouse);
  ball_->BeginDrag();

  prev_time_[0] = time;
  prev_quat_[0] = mouse;
  prev_time_[1] = prev_time_[2] = -100;
  ball_->Update();
  last_time_ = time;
  scene_interface_->need_redraw();
  return STOP_E;
}

BaseTool::propagation_state_e
ViewRotateTool::pointer_motion(int which, int x, int y, int time)
{
  if (which != 2) return CONTINUE_E;
  int xres = scene_interface_->width();
  int yres = scene_interface_->height();

  if(!rotate_valid_p_)
    return STOP_E;

  HVect mouse((2.0 * x) / xres - 1.0, 2.0 * (yres - y * 1.0) / yres - 1.0,
	      0.0, 1.0);
  prev_time_[2] = prev_time_[1];
  prev_time_[1] = prev_time_[0];
  prev_time_[0] = time;
  ball_->Mouse(mouse);
  ball_->Update();

  prev_quat_[2] = prev_quat_[1];
  prev_quat_[1] = prev_quat_[0];
  prev_quat_[0] = mouse;

  // now we should just send the view points through the rotation
  // (after centerd around the ball), eyep, lookat, and up
  View tmpview(rot_view_);

  Transform tmp_trans;
  HMatrix mNow;
  ball_->Value(mNow);
  tmp_trans.set(&mNow[0][0]);

  Transform prv = prev_trans_;
  prv.post_trans(tmp_trans);

  HMatrix vmat;
  prv.get(&vmat[0][0]);

  Point y_a(vmat[0][1], vmat[1][1], vmat[2][1]);
  Point z_a(vmat[0][2], vmat[1][2], vmat[2][2]);

  tmpview.up(y_a.vector());
  tmpview.eyep((z_a * (eye_dist_)) + tmpview.lookat().vector());

  scene_interface_->view_ = tmpview;
  scene_interface_->need_redraw();
  scene_interface_->update_mode_string("rotate:");

  last_time_ = time;

  return STOP_E;
}

BaseTool::propagation_state_e
ViewRotateTool::pointer_up(int which, int x, int y, int time)
{
  if (which != 2) return CONTINUE_E;
  if(time - last_time_ < 20){
    // now setup the normalized quaternion
    View tmpview(rot_view_);
	    
    Transform tmp_trans;
    HMatrix mNow;
    ball_->Value(mNow);
    tmp_trans.set(&mNow[0][0]);
	    
    Transform prv = prev_trans_;
    prv.post_trans(tmp_trans);
	    
    HMatrix vmat;
    prv.get(&vmat[0][0]);
	    
    Point y_a(vmat[0][1], vmat[1][1], vmat[2][1]);
    Point z_a(vmat[0][2], vmat[1][2], vmat[2][2]);
	    
    tmpview.up(y_a.vector());
    tmpview.eyep((z_a * (eye_dist_)) + tmpview.lookat().vector());
	    
    scene_interface_->view_ = tmpview;
    prev_trans_ = prv;

    // now you need to use the history to 
    // set up the arc you want to use...
    ball_->Init();
    double rad = 0.8;
    HVect center(0, 0, 0, 1.0);

    ball_->Place(center,rad);

    int index = 2;

    if (prev_time_[index] == -100)
      index = 1;

    ball_->vDown = prev_quat_[index];
    ball_->vNow  = prev_quat_[0];
    ball_->dragging = 1;
    ball_->Update();
	    
    ball_->qNorm = ball_->qNow.Conj();
    //double mag = ball_->qNow.VecMag();
    // Go into inertia mode...
//     if (mag > 0.00001) { // arbitrary ad-hoc threshold
//       gui_inertia_mode_.set(1);
//       double c = 1.0/mag;
//       ball_->qNorm.x *= c;
//       ball_->qNorm.y *= c;
//       ball_->qNorm.z *= c;
//       gui_inertia_y_.set(ball_->qNorm.x);
//       gui_inertia_x_.set(-ball_->qNorm.y);
//       angular_v_ = 2*acos(ball_->qNow.w)*1000.0/
// 	(prev_time_[0]-prev_time_[index]);
//       gui_inertia_mag_.set(angular_v_);
//     }
  }
  ball_->EndDrag();
  rotate_valid_p_ = false; // so we don't have to draw this...
  scene_interface_->need_redraw();     // always update this...
  scene_interface_->update_mode_string("");
  return STOP_E;
}

} // namespace SCIRun
