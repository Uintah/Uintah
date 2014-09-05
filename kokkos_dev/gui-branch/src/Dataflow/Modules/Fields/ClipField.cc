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
 *  ClipField.cc:  Rotate and flip field to get it into "standard" view
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
#include <Dataflow/Widgets/ScaledBoxWidget.h>
#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/Clipper.h>
#include <Dataflow/Modules/Fields/ClipField.h>
#include <iostream>
#include <stack>

namespace SCIRun {

using std::stack;

class ClipField : public Module
{
private:
  ScaledBoxWidget *box_;
  CrowdMonitor widget_lock_;
  BBox last_bounds_;
  GuiString clip_location_;
  GuiString clip_mode_;
  GuiInt    autoexec_;
  GuiInt    autoinvert_;
  GuiString exec_mode_;
  int  last_input_generation_;
  int  last_clip_generation_;
  ClipperHandle clipper_;
  stack<ClipperHandle> undo_stack_;

  bool bbox_similar_to(const BBox &a, const BBox &b);

public:
  ClipField(const string& id);
  virtual ~ClipField();

  virtual void execute();
  virtual void widget_moved(int);
};


extern "C" Module* make_ClipField(const string& id) {
  return new ClipField(id);
}


ClipField::ClipField(const string& id)
  : Module("ClipField", id, Source, "Fields", "SCIRun"),
    widget_lock_("ClipField widget lock"),
    clip_location_("clip-location", id, this),
    clip_mode_("clipmode", id, this),
    autoexec_("autoexec", id, this),
    autoinvert_("autoinvert", id, this),
    exec_mode_("execmode", id, this),
    last_input_generation_(0),
    last_clip_generation_(0)
{
  box_ = scinew ScaledBoxWidget(this, &widget_lock_, 1.0, false, false);
}


ClipField::~ClipField()
{
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
ClipField::bbox_similar_to(const BBox &a, const BBox &b)
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
ClipField::execute()
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
  if (!ifieldhandle->mesh()->is_editable())
  {
    error("Not an editable mesh type.");
    return;
  }

  bool do_clip_p = false;

  // Maybe get clip field.
  FieldIPort *cfp = (FieldIPort *)get_iport("Clip Field");
  if (!cfp) {
    error("Unable to initialize " + name + "'s iport\n");
    return;
  }
  FieldHandle cfieldhandle;
  if (cfp->get(cfieldhandle) && cfieldhandle.get_rep() &&
      cfieldhandle->generation != last_clip_generation_)
  {
    last_clip_generation_ = cfieldhandle->generation;

    const TypeDescription *ftd = cfieldhandle->mesh()->get_type_description();
    CompileInfo *ci = ClipFieldMeshAlgo::get_compile_info(ftd);
    DynamicAlgoHandle algo_handle;
    if (! DynamicLoader::scirun_loader().get(*ci, algo_handle))
    {
      error("Could not compile algorithm.");
      return;
    }
    ClipFieldMeshAlgo *algo =
      dynamic_cast<ClipFieldMeshAlgo *>(algo_handle.get_rep());
    if (algo == 0)
    {
      error("Could not get algorithm.");
      return;
    }
    clipper_ = algo->execute(cfieldhandle->mesh());
    do_clip_p = true;
  }

  // Update the widget.
  const BBox bbox = ifieldhandle->mesh()->get_bounding_box();
  if (!bbox_similar_to(last_bounds_, bbox))
  {
    Point bmin = bbox.min();
    Point bmax = bbox.max();

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

    const Point center = bmin + Vector(bmax - bmin) * 0.25;
    const Point right = center + Vector((bmax.x()-bmin.x())/4.0, 0, 0);
    const Point down = center + Vector(0, (bmax.y()-bmin.y())/4.0, 0);
    const Point in = center + Vector(0, 0, (bmax.z()-bmin.z())/4.0);

    const double l2norm = (bmax - bmin).length();

    box_->SetScale(l2norm * 0.015);
    box_->SetPosition(center, right, down, in);

    GeomGroup *widget_group = scinew GeomGroup;
    widget_group->add(box_->GetWidget());

    GeometryOPort *ogport=0;
    ogport = (GeometryOPort*)get_oport("Selection Widget");
    if (!ogport) {
      error("Unable to initialize " + name + "'s oport.");
      return;
    }
    ogport->addObj(widget_group, "ClipField Selection Widget",
		   &widget_lock_);
    ogport->flushViews();

    last_bounds_ = bbox;
    // Force clipper to sync with new widget.
    if (clipper_.get_rep() && !clipper_->mesh_p()) { clipper_ = 0; }
  }

  if (!clipper_.get_rep())
  {
    clipper_ = box_->get_clipper();
    do_clip_p = true;
  }
  else if (exec_mode_.get() == "execute")
  {
    undo_stack_.push(clipper_);
    ClipperHandle ctmp = box_->get_clipper();
    if (clip_mode_.get() == "intersect")
    {
      clipper_ = scinew IntersectionClipper(ctmp, clipper_);
    }
    else if (clip_mode_.get() == "union")
    {
      clipper_ = scinew UnionClipper(ctmp, clipper_);
    }
    else if (clip_mode_.get() == "remove")
    {
      ctmp = scinew InvertClipper(ctmp);
      clipper_ = scinew IntersectionClipper(ctmp, clipper_);
    }
    else
    {
      clipper_ = ctmp;
    }
    do_clip_p = true;
  }
  else if (exec_mode_.get() == "invert")
  {
    undo_stack_.push(clipper_);
    clipper_ = scinew InvertClipper(clipper_);
    do_clip_p = true;
  }
  else if (exec_mode_.get() == "undo")
  {
    if (!undo_stack_.empty())
    {
      clipper_ = undo_stack_.top();
      undo_stack_.pop();
      do_clip_p = true;
    }
  }
  else if (exec_mode_.get() == "location")
  {
    do_clip_p = true;
  }

  if (do_clip_p || ifieldhandle->generation != last_input_generation_)
  {
    last_input_generation_ = ifieldhandle->generation;
    exec_mode_.set("");

    const TypeDescription *ftd = ifieldhandle->get_type_description();
    CompileInfo *ci = ClipFieldAlgo::get_compile_info(ftd);
    DynamicAlgoHandle algo_handle;
    if (! DynamicLoader::scirun_loader().get(*ci, algo_handle))
    {
      error("Could not compile algorithm.");
      return;
    }
    ClipFieldAlgo *algo =
      dynamic_cast<ClipFieldAlgo *>(algo_handle.get_rep());
    if (algo == 0)
    {
      error("Could not get algorithm.");
      return;
    }

    // Maybe invert the clipper again.
    ClipperHandle clipper(clipper_);
    if (autoinvert_.get())
    {
      clipper = scinew InvertClipper(clipper_);
    }

    // Do the clip, dispatch based on which clip location test we are using.
    FieldHandle ofield = 0;
    if (clip_location_.get() == "nodeone")
    {
      ofield = algo->execute_node(ifieldhandle, clipper, true);
    }
    else if (clip_location_.get() == "nodeall")
    {
      ofield = algo->execute_node(ifieldhandle, clipper, false);
    }
    else // 'cell' and default
    {
      ofield = algo->execute_cell(ifieldhandle, clipper);
    }

    FieldOPort *ofield_port = (FieldOPort *)get_oport("Output Field");
    if (!ofield_port) {
      error("Unable to initialize " + name + "'s oport.");
      return;
    }
    
    ofield_port->send(ofield);
  }
}


void
ClipField::widget_moved(int i)
{
  if (i == 1 && autoexec_.get())
  {
    exec_mode_.set("execute");
    want_to_execute();
  }
}



CompileInfo *
ClipFieldAlgo::get_compile_info(const TypeDescription *fsrc)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("ClipFieldAlgoT");
  static const string base_class_name("ClipFieldAlgo");

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


CompileInfo *
ClipFieldMeshAlgo::get_compile_info(const TypeDescription *fsrc)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("ClipFieldMeshAlgoT");
  static const string base_class_name("ClipFieldMeshAlgo");

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

