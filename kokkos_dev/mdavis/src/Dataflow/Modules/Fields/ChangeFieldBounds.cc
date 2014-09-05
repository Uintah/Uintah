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



//    File   : ChangeFieldBounds.cc
//    Author : McKay Davis
//    Date   : July 2002


#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/FieldInterface.h>

#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Core/Geometry/Transform.h>
#include <Core/Thread/CrowdMonitor.h>
#include <Dataflow/Widgets/BoxWidget.h>
#include <Dataflow/Network/NetworkEditor.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Containers/StringUtil.h>
#include <map>
#include <iostream>

namespace SCIRun {

using std::endl;
using std::pair;

class ChangeFieldBounds : public Module {
public:
  ChangeFieldBounds(GuiContext* ctx);
  virtual ~ChangeFieldBounds();

  GuiDouble		outputcenterx_;	// the out geometry (center point and size)
  GuiDouble		outputcentery_;
  GuiDouble		outputcenterz_;
  GuiDouble		outputsizex_;
  GuiDouble		outputsizey_;
  GuiDouble		outputsizez_;
  GuiInt		useoutputcenter_;   // center checkbox
  GuiInt		useoutputsize_;   // size checkbox

  GuiString		inputcenterx_;
  GuiString		inputcentery_;
  GuiString		inputcenterz_;
  GuiString		inputsizex_;
  GuiString		inputsizey_;
  GuiString		inputsizez_;

  GuiDouble             box_scale_;
  GuiPoint              box_center_;
  GuiPoint              box_right_;
  GuiPoint              box_down_;
  GuiPoint              box_in_;

  GuiInt                resetting_;

  CrowdMonitor		widget_lock_;
  BoxWidget *		box_;
  Transform		box_initial_transform_;
  Transform		field_initial_transform_;
  BBox			box_initial_bounds_;
  int			generation_;
  int			widgetid_;

  void clear_vals();
  void update_input_attributes(FieldHandle);
  void build_widget(FieldHandle, bool reset);

  virtual void execute();
  virtual void widget_moved(bool, BaseWidget*);
};

  DECLARE_MAKER(ChangeFieldBounds)

ChangeFieldBounds::ChangeFieldBounds(GuiContext* ctx)
  : Module("ChangeFieldBounds", ctx, Filter, "FieldsGeometry", "SCIRun"),
    outputcenterx_(ctx->subVar("outputcenterx")),
    outputcentery_(ctx->subVar("outputcentery")),
    outputcenterz_(ctx->subVar("outputcenterz")),
    outputsizex_(ctx->subVar("outputsizex")),
    outputsizey_(ctx->subVar("outputsizey")),
    outputsizez_(ctx->subVar("outputsizez")),
    useoutputcenter_(ctx->subVar("useoutputcenter")),
    useoutputsize_(ctx->subVar("useoutputsize")),
    inputcenterx_(ctx->subVar("inputcenterx", false)),
    inputcentery_(ctx->subVar("inputcentery", false)),
    inputcenterz_(ctx->subVar("inputcenterz", false)),
    inputsizex_(ctx->subVar("inputsizex", false)),
    inputsizey_(ctx->subVar("inputsizey", false)),
    inputsizez_(ctx->subVar("inputsizez", false)),
    box_scale_(ctx->subVar("box-scale")),
    box_center_(ctx->subVar("box-center")),
    box_right_(ctx->subVar("box-right")),
    box_down_(ctx->subVar("box-down")),
    box_in_(ctx->subVar("box-in")),
    resetting_(ctx->subVar("resetting", false)),
    widget_lock_("ChangeFieldBounds widget lock"),
    generation_(-1),
    widgetid_(0)
{
  box_ = scinew BoxWidget(this, &widget_lock_, 1.0, false, false);
  box_->Connect((GeometryOPort*)get_oport("Transformation Widget"));

  inputcenterx_.set("---");
  inputcentery_.set("---");
  inputcenterz_.set("---");
  inputsizex_.set("---");
  inputsizey_.set("---");
  inputsizez_.set("---");
}

ChangeFieldBounds::~ChangeFieldBounds()
{
  delete box_;
}



void
ChangeFieldBounds::clear_vals() 
{
  inputcenterx_.set("---");
  inputcentery_.set("---");
  inputcenterz_.set("---");
  inputsizex_.set("---");
  inputsizey_.set("---");
  inputsizez_.set("---");
}


void
ChangeFieldBounds::update_input_attributes(FieldHandle f) 
{
  Point center;
  Vector size;
  
  BBox bbox = f->mesh()->get_bounding_box();

  if (!bbox.valid()) {
    warning("Input field is empty -- using unit cube.");
    bbox.extend(Point(0,0,0));
    bbox.extend(Point(1,1,1));
  }
  size = bbox.diagonal();
  center = bbox.center();

  inputcenterx_.set(to_string(center.x()));
  inputcentery_.set(to_string(center.y()));
  inputcenterz_.set(to_string(center.z()));
  inputsizex_.set(to_string(size.x()));
  inputsizey_.set(to_string(size.y()));
  inputsizez_.set(to_string(size.z()));
}


void
ChangeFieldBounds::build_widget(FieldHandle f, bool reset)
{
  if (reset || box_scale_.get() <= 0)
  {
    Point center;
    Vector size;
    BBox bbox = f->mesh()->get_bounding_box();
    if (!bbox.valid()) {
      warning("Input field is empty -- using unit cube.");
      bbox.extend(Point(0,0,0));
      bbox.extend(Point(1,1,1));
    }
    box_initial_bounds_ = bbox;

    // build a widget identical to the BBox
    size = Vector(bbox.max()-bbox.min());
    if (fabs(size.x())<1.e-6) {
      size.x(2.e-4); 
      bbox.extend(bbox.min()-Vector(1.0e-4, 0.0, 0.0));
      bbox.extend(bbox.max()+Vector(1.0e-4, 0.0, 0.0));
    }
    if (fabs(size.y())<1.e-6) {
      size.y(2.e-4); 
      bbox.extend(bbox.min()-Vector(0.0, 1.0e-4, 0.0));
      bbox.extend(bbox.max()+Vector(0.0, 1.0e-4, 0.0));
    }
    if (fabs(size.z())<1.e-6) {
      size.z(2.e-4); 
      bbox.extend(bbox.min()-Vector(0.0, 0.0, 1.0e-4));
      bbox.extend(bbox.max()+Vector(0.0, 0.0, 1.0e-4));
    }
    center = Point(bbox.min() + size/2.);

    Vector sizex(size.x(),0,0);
    Vector sizey(0,size.y(),0);
    Vector sizez(0,0,size.z());

    Point right(center + sizex/2.);
    Point down(center + sizey/2.);
    Point in(center +sizez/2.);

    const double l2norm = size.length();

    // Translate * Rotate * Scale.
    Transform r;
    Point unused;
    box_initial_transform_.load_identity();
    box_initial_transform_.pre_scale(Vector((right-center).length(),
					    (down-center).length(),
					    (in-center).length()));
    r.load_frame(unused, (right-center).normal(),
		 (down-center).normal(),
		 (in-center).normal());
    box_initial_transform_.pre_trans(r);
    box_initial_transform_.pre_translate(center.asVector());


    box_->SetScale(l2norm * 0.015);
    box_->SetPosition(center, right, down, in);
    box_scale_.set(-1.0);
  }
  else
  {
    const double l2norm = (box_right_.get().vector() +
			   box_down_.get().vector() +
			   box_in_.get().vector()).length();
    box_->SetScale(l2norm * 0.015);
    box_->SetPosition(box_center_.get(), box_right_.get(),
		      box_down_.get(), box_in_.get());
  }

  GeomGroup *widget_group = scinew GeomGroup;
  widget_group->add(box_->GetWidget());

  GeometryOPort *ogport = (GeometryOPort*)get_oport("Transformation Widget");
  widgetid_ = ogport->addObj(widget_group,"ChangeFieldBounds Transform widget",
			     &widget_lock_);
  ogport->flushViews();
}


void
ChangeFieldBounds::execute()
{
  FieldIPort *iport = (FieldIPort*)get_iport("Input Field"); 

  // The input port (with data) is required.
  FieldHandle fh;
  if (!iport->get(fh) || !fh.get_rep())
  {
    clear_vals();
    return;
  }

  // The output port is required.
  FieldOPort *oport = (FieldOPort*)get_oport("Output Field");
  update_state(Executing);

  // build the transform widget and set the the initial
  // field transform.
  if (generation_ != fh.get_rep()->generation || resetting_.get()) 
  {
    generation_ = fh.get_rep()->generation;
    // get and display the attributes of the input field
    update_input_attributes(fh);
    build_widget(fh, resetting_.get());
    BBox bbox = fh->mesh()->get_bounding_box();
    if (!bbox.valid()) {
      warning("Input field is empty -- using unit cube.");
      bbox.extend(Point(0,0,0));
      bbox.extend(Point(1,1,1));
    }
    Vector size(bbox.max()-bbox.min());
     if (fabs(size.x())<1.e-8) {
      size.x(2.e-8); 
      bbox.extend(bbox.min()-Vector(1.e-8,0,0));
    }
    if (fabs(size.y())<1.e-8) {
      size.y(2.e-8); 
      bbox.extend(bbox.min()-Vector(0,1.e-8,0));
    }
    if (fabs(size.z())<1.e-8) {
      size.z(2.e-8); 
      bbox.extend(bbox.min()-Vector(0,0,1.e-8));
    }
    Point center(bbox.min() + size/2.);
    Vector sizex(size.x(),0,0);
    Vector sizey(0,size.y(),0);
    Vector sizez(0,0,size.z());

    Point right(center + sizex/2.);
    Point down(center + sizey/2.);
    Point in(center +sizez/2.);

    Transform r;
    Point unused;
    field_initial_transform_.load_identity();
    field_initial_transform_.pre_scale(Vector((right-center).length(),
					      (down-center).length(),
					      (in-center).length()));
    r.load_frame(unused, (right-center).normal(),
		 (down-center).normal(),
		 (in-center).normal());
    field_initial_transform_.pre_trans(r);
    field_initial_transform_.pre_translate(center.asVector());

    resetting_.set(0);
  }

  if (useoutputsize_.get() || useoutputcenter_.get()) {
    Point center, right, down, in;
    outputcenterx_.reset(); outputcentery_.reset(); outputcenterz_.reset();
    outputsizex_.reset(); outputsizey_.reset(); outputsizez_.reset();
    if (outputsizex_.get() < 0 || 
	outputsizey_.get() < 0 || 
	outputsizez_.get() < 0) {
      error("Degenerate BBox requested.");
      return;                    // degenerate 
    }
    Vector sizex, sizey, sizez;
    box_->GetPosition(center,right,down,in);
    if (useoutputsize_.get()) {
      sizex=Vector(outputsizex_.get(),0,0);
      sizey=Vector(0,outputsizey_.get(),0);
      sizez=Vector(0,0,outputsizez_.get());
    } else {
      sizex=(right-center)*2;
      sizey=(down-center)*2;
      sizez=(in-center)*2;
    }
    if (useoutputcenter_.get()) {
      center = Point(outputcenterx_.get(),
		     outputcentery_.get(),
		     outputcenterz_.get());
    }
    right = Point(center + sizex/2.);
    down = Point(center + sizey/2.);
    in = Point(center + sizez/2.);
    box_->SetPosition(center,right,down,in);
  }

  // Transform the mesh if necessary.
  // Translate * Rotate * Scale.
  Point center, right, down, in;
  box_->GetPosition(center, right, down, in);
  Transform t, r;
  Point unused;
  t.load_identity();
  t.pre_scale(Vector((right-center).length(),
       (down-center).length(),
       (in-center).length()));
  r.load_frame(unused, (right-center).normal(),
	 (down-center).normal(),
	 (in-center).normal());
  t.pre_trans(r);
  t.pre_translate(center.asVector());

  Transform inv(field_initial_transform_);
  inv.invert();
  t.post_trans(inv);

  // Change the input field handle here.
  fh.detach();
  fh->mesh_detach();
  fh->mesh()->transform(t);

  oport->send(fh);

  // The output port is required.
  MatrixOPort *moport = (MatrixOPort*)get_oport("Transformation Matrix");

  // convert the transform into a matrix and send it out   
  DenseMatrix *matrix_transform = scinew DenseMatrix(t);
  MatrixHandle mh = matrix_transform;
  moport->send(mh);
}

    
void ChangeFieldBounds::widget_moved(bool last, BaseWidget*)
{
  if (last) {
    Point center, right, down, in;
    outputcenterx_.reset(); outputcentery_.reset(); outputcenterz_.reset();
    outputsizex_.reset(); outputsizey_.reset(); outputsizez_.reset();
    box_->GetPosition(center,right,down,in);
    outputcenterx_.set(center.x());
    outputcentery_.set(center.y());
    outputcenterz_.set(center.z());
    outputsizex_.set((right.x()-center.x())*2.);
    outputsizey_.set((down.y()-center.y())*2.);
    outputsizez_.set((in.z()-center.z())*2.);
    box_scale_.set(box_->GetScale());
    box_center_.set(center);
    box_right_.set(right);
    box_down_.set(down);
    box_in_.set(in);
    want_to_execute();
  }
}


} // End namespace SCIRun
