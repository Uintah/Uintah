/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/


/*
 *  BuildTransform.cc:  Build a 4x4 geometric transformation matrix
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   March 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <Dataflow/Ports/MatrixPort.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Geom/GeomSwitch.h>
#include <Core/Geometry/BBox.h>
#include <Core/Geometry/Plane.h>
#include <Core/Geometry/Transform.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/Expon.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Thread/CrowdMonitor.h>
#include <Dataflow/Widgets/BoxWidget.h>
#include <iostream>
#include <stdio.h>

namespace SCIRun {

class BuildTransform : public Module {
  MatrixIPort* imatrix_;
  MatrixOPort* omatrix_;
  GeometryOPort* ogeom_;

  GuiDouble rotate_x_gui_, rotate_y_gui_, rotate_z_gui_, rotate_theta_gui_;
  GuiDouble translate_x_gui_, translate_y_gui_, translate_z_gui_;
  GuiDouble scale_uniform_gui_, scale_x_gui_, scale_y_gui_, scale_z_gui_;
  GuiDouble shear_plane_a_gui_, shear_plane_b_gui_;
  GuiDouble shear_plane_c_gui_, shear_plane_d_gui_;
  GuiInt widget_resizable_gui_;
  GuiInt permute_x_gui_;
  GuiInt permute_y_gui_;
  GuiInt permute_z_gui_;
  GuiInt pre_transform_gui_;
  GuiString which_transform_gui_;
  GuiDouble widget_scale_gui_;
  GuiInt ignoring_widget_changes_gui_;

  MatrixHandle omatrixH_;
  BoxWidget *box_widget_;
  GeomHandle widget_switch_;
  CrowdMonitor widget_lock_;
  Transform composite_trans_, latest_trans_, latest_widget_trans_;
  Transform widget_trans_, widget_pose_inv_trans_;
  Point widget_pose_center_;
  int have_been_initialized_;
  int widgetid_;

public:
  BuildTransform(GuiContext* ctx);
  virtual ~BuildTransform();
  virtual void widget_moved(bool last, BaseWidget*);
  virtual void execute();
  void tcl_command( GuiArgs&, void * );
};

DECLARE_MAKER(BuildTransform)
  static string module_name("BuildTransform");
static string widget_name("TransformWidget");

BuildTransform::BuildTransform(GuiContext* ctx) : 
  Module("BuildTransform", ctx, Filter, "Math", "SCIRun"),
  imatrix_(0),
  omatrix_(0),
  ogeom_(0),
  rotate_x_gui_(ctx->subVar("rotate_x")),
  rotate_y_gui_(ctx->subVar("rotate_y")),
  rotate_z_gui_(ctx->subVar("rotate_z")), 
  rotate_theta_gui_(ctx->subVar("rotate_theta")),
  translate_x_gui_(ctx->subVar("translate_x")),
  translate_y_gui_(ctx->subVar("translate_y")),
  translate_z_gui_(ctx->subVar("translate_z")),
  scale_uniform_gui_(ctx->subVar("scale_uniform")),
  scale_x_gui_(ctx->subVar("scale_x")),
  scale_y_gui_(ctx->subVar("scale_y")), 
  scale_z_gui_(ctx->subVar("scale_z")), 
  shear_plane_a_gui_(ctx->subVar("shear_plane_a")),
  shear_plane_b_gui_(ctx->subVar("shear_plane_b")),
  shear_plane_c_gui_(ctx->subVar("shear_plane_c")),
  shear_plane_d_gui_(ctx->subVar("shear_plane_d")),
  widget_resizable_gui_(ctx->subVar("widget_resizable")),
  permute_x_gui_(ctx->subVar("permute_x")),
  permute_y_gui_(ctx->subVar("permute_y")), 
  permute_z_gui_(ctx->subVar("permute_z")),
  pre_transform_gui_(ctx->subVar("pre_transform")),
  which_transform_gui_(ctx->subVar("which_transform")),
  widget_scale_gui_(ctx->subVar("widget_scale")),
  ignoring_widget_changes_gui_(ctx->subVar("ignoring_widget_changes")),
  widget_lock_("BuildTransform widget lock"),
  have_been_initialized_(0),
  widgetid_(0)
{
  
  box_widget_ = scinew BoxWidget(this, &widget_lock_, 0.2, false, false);
  box_widget_->Connect((GeometryOPort*)get_oport("Geometry"));
  widget_switch_ = box_widget_->GetWidget();
}

BuildTransform::~BuildTransform()
{
  delete box_widget_;
}

void BuildTransform::execute()
{
  imatrix_ = (MatrixIPort *)get_iport("Matrix");
  omatrix_ = (MatrixOPort *)get_oport("Matrix");
  ogeom_ = (GeometryOPort *)get_oport("Geometry");
  string which_transform=which_transform_gui_.get();

  // create the widget
  if (!have_been_initialized_) {
    Point C, R, D, I;
    box_widget_->GetPosition(C,R,D,I);
    C=Point(0,0,0); R=Point(1,0,0); D=Point(0,1,0), I=Point(0,0,1);
    widget_pose_center_=C;
    box_widget_->SetPosition(C,R,D,I);
    box_widget_->SetCurrentMode(2);
    if (which_transform != "widget")
    {
      ((GeomSwitch *)(widget_switch_.get_rep()))->set_state(0);
    }
    widgetid_ = ogeom_->addObj(widget_switch_, widget_name, &widget_lock_);
    ogeom_->flushViews();
    have_been_initialized_=1;
  }

  // get the input matrix if there is one
  MatrixHandle input_matrix_H;
  Transform input_transform;
  if (imatrix_->get(input_matrix_H) && input_matrix_H.get_rep()) {
    input_transform = input_matrix_H->toTransform();
  }
  
  Transform local_transform;
  
  // get the "fixed point"
  Vector t(translate_x_gui_.get(), 
	   translate_y_gui_.get(), 
	   translate_z_gui_.get());
  
  // build the local transform
  if (which_transform == "translate") {
    local_transform.post_translate(t);
  } else if (which_transform == "scale") {
    double new_scale=scale_uniform_gui_.get();
    double s=pow(10.,new_scale);
    double new_scalex=scale_x_gui_.get();
    double sx=pow(10.,new_scalex)*s;
    double new_scaley=scale_y_gui_.get();
    double sy=pow(10.,new_scaley)*s;
    double new_scalez=scale_z_gui_.get();
    double sz=pow(10.,new_scalez)*s;
    Vector sc(sx, sy, sz);
    local_transform.post_translate(t);	
    local_transform.post_scale(sc);
    local_transform.post_translate(-t);
  } else if (which_transform == "rotate") {
    Vector axis(rotate_x_gui_.get(),rotate_y_gui_.get(),rotate_z_gui_.get());
    if (!axis.length2()) axis.x(1);
    axis.normalize();
    local_transform.post_translate(t);
    local_transform.post_rotate(rotate_theta_gui_.get()*M_PI/180., axis);
    local_transform.post_translate(-t);
  } else if (which_transform == "shear") {
    local_transform.post_shear(t, Plane(shear_plane_a_gui_.get(), 
					shear_plane_b_gui_.get(), 
					shear_plane_c_gui_.get(), 
					shear_plane_d_gui_.get()));
  } else if (which_transform == "permute") {
    local_transform.post_permute(permute_x_gui_.get(), 
				 permute_y_gui_.get(), 
				 permute_z_gui_.get());
  } else { // (which_transform == "widget")
    Point R, D, I, C;
    box_widget_->GetPosition(C, R, D, I);
    
    double ratio=widget_scale_gui_.get();
    widget_scale_gui_.set(1);
    R=C+(R-C)*ratio;
    D=C+(D-C)*ratio;
    I=C+(I-C)*ratio;
    box_widget_->SetPosition(C, R, D, I);
    
    // find the difference between widget_pose(_inv) and the current pose
    if (!ignoring_widget_changes_gui_.get()) {
      local_transform.load_frame(C,R-C,D-C,I-C);
      local_transform.post_trans(widget_pose_inv_trans_);
      local_transform.post_translate(-widget_pose_center_.vector());
      local_transform.pre_translate(C.vector());
    }
    local_transform.post_trans(latest_widget_trans_);
    latest_widget_trans_=local_transform;
    widget_pose_center_=C;
    widget_pose_inv_trans_.load_frame(C,R-C,D-C,I-C);
    widget_pose_inv_trans_.invert();
  }
  DenseMatrix *matrix_transform=scinew DenseMatrix(4,4);
  omatrixH_=matrix_transform;
  
  // now either pre- or post-multiply the transforms and store in matrix
  if (pre_transform_gui_.get()) {
    local_transform.post_trans(composite_trans_);
    latest_trans_=local_transform;
    local_transform.post_trans(input_transform);
  } else {
    local_transform.pre_trans(composite_trans_);
    latest_trans_=local_transform;
    local_transform.pre_trans(input_transform);
  }

  DenseMatrix *dm = scinew DenseMatrix(local_transform);
  omatrix_->send(MatrixHandle(dm));
}

void BuildTransform::widget_moved(bool last, BaseWidget*)
{
  // only re-execute if this was a widget-release event
  if (last) {
    want_to_execute();
  }
}

void BuildTransform::tcl_command(GuiArgs& args, void* userdata) {
  if (args[1] == "hide_widget") {
    ((GeomSwitch *)(widget_switch_.get_rep()))->set_state(0);
    if (ogeom_) ogeom_->flushViews();
  } else if (args[1] == "show_widget") {
    ((GeomSwitch *)(widget_switch_.get_rep()))->set_state(1);
    if (ogeom_) ogeom_->flushViews();
  } else if (args[1] == "reset_widget" || args[1] == "reset" || 
	     args[1] == "composite") {
    if (args[1] == "reset")
      composite_trans_.load_identity();
    else if (args[1] == "composite")
      composite_trans_=latest_trans_;
    latest_trans_.load_identity();
    latest_widget_trans_.load_identity();
    box_widget_->SetPosition(Point(0,0,0), 
			     Point(1,0,0), Point(0,1,0), Point(0,0,1));
    widget_pose_center_=Point(0,0,0);
    widget_pose_inv_trans_.load_identity();
    want_to_execute();
  } else if (args[1] == "change_handles") {
    if (args[2] == "1") {	// start showing resize handles
      box_widget_->SetCurrentMode(1);
      if (ogeom_) ogeom_->flushViews();
    } else {		        // stop showing resize handles
      box_widget_->SetCurrentMode(2);
      if (ogeom_) ogeom_->flushViews();
    }
  } else {
    Module::tcl_command(args, userdata);
  }
}

} // End namespace SCIRun
