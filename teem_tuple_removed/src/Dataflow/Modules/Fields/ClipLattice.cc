/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/


/*
 *  ClipLattice.cc:  Rotate and flip field to get it into "standard" view
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   March 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Core/Thread/CrowdMonitor.h>
#include <Dataflow/Widgets/BoxWidget.h>
#include <Core/Datatypes/LatVolField.h>
#include <Dataflow/Modules/Fields/ClipLattice.h>
#include <iostream>

namespace SCIRun {

class ClipLattice : public Module
{
private:
  BoxWidget *widget_;
  CrowdMonitor widget_lock_;
  BBox last_bounds_;
  bool gui_exec_p_;
  int  last_input_generation_;
  GuiInt use_text_bbox_;
  GuiDouble text_min_x_;
  GuiDouble text_min_y_;
  GuiDouble text_min_z_;
  GuiDouble text_max_x_;
  GuiDouble text_max_y_;
  GuiDouble text_max_z_;
  bool init_;
  int widgetid_;
  bool bbox_similar_to(const BBox &a, const BBox &b);

public:
  ClipLattice(GuiContext* ctx);
  virtual ~ClipLattice();

  virtual void execute();
  virtual void tcl_command(GuiArgs&, void*);
  virtual void widget_moved(bool last);
};


DECLARE_MAKER(ClipLattice)

ClipLattice::ClipLattice(GuiContext* ctx)
  : Module("ClipLattice", ctx, Filter, "FieldsCreate", "SCIRun"),
    widget_lock_("ClipLattice widget lock"),
    gui_exec_p_(true),
    last_input_generation_(0),
    use_text_bbox_(ctx->subVar("use-text-bbox")),
    text_min_x_(ctx->subVar("text-min-x")),
    text_min_y_(ctx->subVar("text-min-y")),
    text_min_z_(ctx->subVar("text-min-z")),
    text_max_x_(ctx->subVar("text-max-x")),
    text_max_y_(ctx->subVar("text-max-y")),
    text_max_z_(ctx->subVar("text-max-z")),
    init_(false),
    widgetid_(0)
{
  widget_ = scinew BoxWidget(this, &widget_lock_, 1.0, true, false);
  widget_->Connect((GeometryOPort *)get_oport("Selection Widget"));
}


ClipLattice::~ClipLattice()
{
  delete widget_;
}


static bool
check_ratio(double x, double y, double lower, double upper)
{
  if (fabs(x) < 1e-6)
  {
    if (!(fabs(y) < 1e-6))
    {
      return false;
    }
  }
  else
  {
    const double ratio = y / x;
    if (ratio < lower || ratio > upper)
    {
      return false;
    }
  }
  return true;
}


bool
ClipLattice::bbox_similar_to(const BBox &a, const BBox &b)
{
  return 
    a.valid() &&
    b.valid() &&
    check_ratio(a.min().x(), b.min().x(), 0.5, 2.0) &&
    check_ratio(a.min().y(), b.min().y(), 0.5, 2.0) &&
    check_ratio(a.min().z(), b.min().z(), 0.5, 2.0) &&
    check_ratio(a.min().x(), b.min().x(), 0.5, 2.0) &&
    check_ratio(a.min().y(), b.min().y(), 0.5, 2.0) &&
    check_ratio(a.min().z(), b.min().z(), 0.5, 2.0);
}



void
ClipLattice::execute()
{
  // Get input field.
  FieldIPort *ifp = (FieldIPort *)get_iport("Input Field");
  FieldHandle ifieldhandle;
  if (!ifp) {
    error("Unable to initialize " +name + "'s iport.");
    return;
  }
  if (!(ifp->get(ifieldhandle) && ifieldhandle.get_rep()))
  {
    return;
  }
  if (ifieldhandle->mesh()->get_type_description()->get_name() !=
	"LatVolMesh")
  {
    error("Not a LatVolField field type.");
    return;
  }

  // Update the widget.
  const BBox bbox = ifieldhandle->mesh()->get_bounding_box();
  if (!bbox_similar_to(last_bounds_, bbox) || 
      use_text_bbox_.get() || !init_)
  {
    Point center, right, down, in, bmin, bmax;
    bmin = Point(text_min_x_.get(), text_min_y_.get(), text_min_z_.get());
    bmax = Point(text_max_x_.get(), text_max_y_.get(), text_max_z_.get());
    if (use_text_bbox_.get() || (!init_ && bmin!=bmax)) {
      center = bmin + Vector(bmax-bmin) * 0.5;
      right = center + Vector(bmax.x()-bmin.x()/2.0, 0, 0);
      down = center + Vector(0, bmax.x()-bmin.x()/2.0, 0);
      in = center + Vector(0, 0, bmax.x()-bmin.x()/2.0);
    } else {
      bmin = bbox.min();
      bmax = bbox.max();
      // Fix degenerate boxes.
      const double size_estimate = Max((bmax-bmin).length() * 0.01, 1.0e-5);
      if (fabs(bmax.x() - bmin.x()) < 1.0e-6)
      {
	bmin.x(bmin.x() - size_estimate);
	bmax.x(bmax.x() + size_estimate);
      }
      if (fabs(bmax.y() - bmin.y()) < 1.0e-6)
      {
	bmin.y(bmin.y() - size_estimate);
	bmax.y(bmax.y() + size_estimate);
      }
      if (fabs(bmax.z() - bmin.z()) < 1.0e-6)
      {
	bmin.z(bmin.z() - size_estimate);
	bmax.z(bmax.z() + size_estimate);
      }
      center = bmin + Vector(bmax - bmin) * 0.25;
      right = center + Vector((bmax.x()-bmin.x())/4.0, 0, 0);
      down = center + Vector(0, (bmax.y()-bmin.y())/4.0, 0);
      in = center + Vector(0, 0, (bmax.z()-bmin.z())/4.0);
      text_min_x_.set(right.x());
      text_min_y_.set(down.y());
      text_min_z_.set(in.z());
      text_max_x_.set(2.*center.x()-right.x());
      text_max_y_.set(2.*center.y()-down.y());
      text_max_z_.set(2.*center.z()-in.z());
    }

    const double l2norm = (bmax - bmin).length();

    widget_->SetScale(l2norm * 0.015);
    widget_->SetPosition(center, right, down, in);

    GeomGroup *widget_group = scinew GeomGroup;
    widget_group->add(widget_->GetWidget());

    GeometryOPort *ogport=0;
    ogport = (GeometryOPort*)get_oport("Selection Widget");
    if (!ogport) {
      error("Unable to initialize " + name + "'s oport.");
      return;
    }
    widgetid_ = ogport->addObj(widget_group, "ClipLattice Selection Widget",
			       &widget_lock_);
    ogport->flushViews();

    last_bounds_ = bbox;
  }

  if (gui_exec_p_ || ifieldhandle->generation != last_input_generation_)
  {
    last_input_generation_ = ifieldhandle->generation;
    gui_exec_p_ = false;

    // Dynamically compile the clipper.
    const TypeDescription *ftd = ifieldhandle->get_type_description();
    CompileInfoHandle ci = ClipLatticeAlgo::get_compile_info(ftd);
    Handle<ClipLatticeAlgo> algo;
    if (!module_dynamic_compile(ci, algo)) return;

    // Get widget bounds.
    Point center, r, d, i, top, bottom;
    if (use_text_bbox_.get() || !init_) {
      init_=true;
      top = Point(text_max_x_.get(), text_max_y_.get(), text_max_z_.get());
      bottom = Point(text_min_x_.get(), text_min_y_.get(), text_min_z_.get());
      center = bottom + Vector(top-bottom)/2.;
      r=d=i=center;
      r.x(bottom.x());
      d.y(bottom.y());
      i.z(bottom.z());
      widget_->SetPosition(center, r, d, i);
    } else {
      widget_->GetPosition(center, r, d, i);
      const Vector dx = r - center;
      const Vector dy = d - center;
      const Vector dz = i - center;
      text_min_x_.set(r.x());
      text_min_y_.set(d.y());
      text_min_z_.set(i.z());
      text_max_x_.set(2.*center.x()-r.x());
      text_max_y_.set(2.*center.y()-d.y());
      text_max_z_.set(2.*center.z()-i.z());      
      top = center + dx + dy + dz;
      bottom = center - dx - dy - dz;
    }

    // Execute the clip.
    FieldHandle ofield = algo->execute(ifieldhandle, top, bottom);

    FieldOPort *ofield_port = (FieldOPort *)get_oport("Output Field");
    if (!ofield_port) {
      error("Unable to initialize " + name + "'s oport.");
      return;
    }
    
    ofield_port->send(ofield);
  }
}


void
ClipLattice::widget_moved(bool last)
{
  if (last)
  {
    gui_exec_p_ = true;
    want_to_execute();
  }
}

void
ClipLattice::tcl_command(GuiArgs& args, void* userdata) {
  if (args.count() < 2) {
    args.error("ClipLattice needs a minor command");
    return;
  }
  if (args[1] == "execute") {
    gui_exec_p_ = true;
    want_to_execute();
  }
  else
  {
    Module::tcl_command(args, userdata);
  }
}

CompileInfoHandle
ClipLatticeAlgo::get_compile_info(const TypeDescription *fsrc)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("ClipLatticeAlgoT");
  static const string base_class_name("ClipLatticeAlgo");

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       fsrc->get_filename() + ".",
                       base_class_name, 
                       template_class_name, 
                       fsrc->get_name());

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  fsrc->fill_compile_info(rval);
  return rval;
}


} // End namespace SCIRun

