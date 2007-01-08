/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
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


//    File   : SelectFieldROIWithBoxWidget.h
//    Author : Michael Callahan
//    Date   : August 2001

#include <Dataflow/Network/Module.h>
#include <Dataflow/Network/Ports/FieldPort.h>
#include <Dataflow/Network/Ports/GeometryPort.h>
#include <Dataflow/Modules/Fields/SelectFieldROIWithBoxWidget.h>
#include <Dataflow/Widgets/BoxWidget.h>
#include <Core/Datatypes/Clipper.h>
#include <Dataflow/GuiInterface/GuiVar.h>

#include <iostream>
#include <stdio.h>

namespace SCIRun {

class SelectFieldROIWithBoxWidget : public Module
{
private:
  FieldHandle output_field_;
  CrowdMonitor widget_lock_;
  BoxWidget *box_;

  GuiInt value_;
  GuiInt mode_;  // 0 nothing 1 accumulate 2 replace

  int  last_generation_;
  BBox last_bounds_;
  int  widgetid_;
public:
  SelectFieldROIWithBoxWidget(GuiContext* ctx);
  virtual ~SelectFieldROIWithBoxWidget();
  virtual void execute();
};


DECLARE_MAKER(SelectFieldROIWithBoxWidget)
SelectFieldROIWithBoxWidget::SelectFieldROIWithBoxWidget(GuiContext* ctx)
  : Module("SelectFieldROIWithBoxWidget", ctx, Filter, "MiscField", "SCIRun"),
    widget_lock_("SelectFieldROIWithBoxWidget widget lock"),
    value_(get_ctx()->subVar("stampvalue"), 100),
    mode_(get_ctx()->subVar("runmode"), 0),
    last_generation_(0),
    widgetid_(0)
{
  box_ = scinew BoxWidget(this, &widget_lock_, 1.0, false, false);
  box_->Connect((GeometryOPort*)get_oport("Selection Widget"));
}



SelectFieldROIWithBoxWidget::~SelectFieldROIWithBoxWidget()
{
  delete box_;
}



void
SelectFieldROIWithBoxWidget::execute()
{
  // Get input field.
  FieldHandle ifieldhandle;
  if (!get_input_handle("Input Field", ifieldhandle)) return;

  if (!ifieldhandle->query_scalar_interface(this).get_rep())
  {
    error("This module only works on scalar fields.");
    return;
  }

  bool forward_p = false;

  if (output_field_.get_rep() == NULL ||
      last_generation_ != ifieldhandle->generation)
  {
    const TypeDescription *mtd = ifieldhandle->mesh()->get_type_description();
    const TypeDescription *ftd = ifieldhandle->get_type_description();
    CompileInfoHandle ci = SelectFieldROIWithBoxWidgetCreateAlgo::get_compile_info(mtd, ftd);
    DynamicAlgoHandle algo_handle;
    Handle<SelectFieldROIWithBoxWidgetCreateAlgo> algo;
    if (!module_dynamic_compile(ci, algo)) return;    
    output_field_ =
      algo->execute(ifieldhandle->mesh(), ifieldhandle->basis_order());

    // Copy the properties.
    output_field_->copy_properties(ifieldhandle.get_rep());
    
    last_generation_ = ifieldhandle->generation;

    BBox obox = output_field_->mesh()->get_bounding_box();
    if (!(last_bounds_.valid() && obox.valid() &&
	  obox.min() == last_bounds_.min() &&
	  obox.max() == last_bounds_.max()))
    {
      // Update the widget.
      const BBox bbox = output_field_->mesh()->get_bounding_box();
      const Point &bmin = bbox.min();
      const Point &bmax = bbox.max();
#if 0
      const Point center = bmin + Vector(bmax - bmin) * 0.5;
      const Point right = center + Vector((bmax.x()-bmin.x())/2.,0,0);
      const Point down = center + Vector(0,(bmax.y()-bmin.y())/2.,0);
      const Point in = center + Vector(0,0,(bmax.z()-bmin.z())/2.);
#else
      const Point center = bmin + Vector(bmax - bmin) * 0.25;
      const Point right = center + Vector((bmax.x()-bmin.x())/4.0, 0, 0);
      const Point down = center + Vector(0, (bmax.y()-bmin.y())/4.0, 0);
      const Point in = center + Vector(0, 0, (bmax.z()-bmin.z())/4.0);
#endif
      const double l2norm = (bmax - bmin).length();

      box_->SetScale(l2norm * 0.015);
      box_->SetPosition(center, right, down, in);

      GeomGroup *widget_group = scinew GeomGroup;
      widget_group->add(box_->GetWidget());

      GeometryOPort *ogport=0;
      ogport = (GeometryOPort*)get_oport("Selection Widget");
      widgetid_ = ogport->addObj(widget_group, "SelectFieldROIWithBoxWidget Selection Widget",
				 &widget_lock_);
      ogport->flushViews();

      last_bounds_ = obox;
    }
    forward_p = true;
  }

  if (mode_.get() == 1 || mode_.get() == 2)
  {
    output_field_.detach();
    const TypeDescription *oftd = output_field_->get_type_description();
    const TypeDescription *oltd = output_field_->order_type_description();
    CompileInfoHandle ci = SelectFieldROIWithBoxWidgetFillAlgo::get_compile_info(oftd, oltd);
    Handle<SelectFieldROIWithBoxWidgetFillAlgo> algo;
    if (!module_dynamic_compile(ci, algo)) return;    

    bool replace_p = false;
    if (mode_.get() == 2) { replace_p = true; }

    ClipperHandle clipper = box_->get_clipper();
    algo->execute(output_field_, clipper, value_.get(), replace_p, 0);

    forward_p = true;
  }

  if (forward_p)
  {
    send_output_handle("Output Field", output_field_, true);
  }
}



CompileInfoHandle
SelectFieldROIWithBoxWidgetCreateAlgo::get_compile_info(const TypeDescription *msrc,
					const TypeDescription *fsrc)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("SelectFieldROIWithBoxWidgetCreateAlgoT");
  static const string base_class_name("SelectFieldROIWithBoxWidgetCreateAlgo");

  const string::size_type loc = fsrc->get_name().find_first_of('<');
  const string fout = fsrc->get_name().substr(0, loc) + "<int> ";

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       msrc->get_filename() + "." +
		       to_filename(fout) + ".",
                       base_class_name, 
                       template_class_name, 
                       msrc->get_name() + ", " + fout);

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  fsrc->fill_compile_info(rval);
  return rval;
}


CompileInfoHandle
SelectFieldROIWithBoxWidgetFillAlgo::get_compile_info(const TypeDescription *fsrc,
				      const TypeDescription *lsrc)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("SelectFieldROIWithBoxWidgetFillAlgoT");
  static const string base_class_name("SelectFieldROIWithBoxWidgetFillAlgo");

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       fsrc->get_filename() + "." +
		       lsrc->get_filename() + ".",
                       base_class_name, 
                       template_class_name, 
                       fsrc->get_name() + ", " + lsrc->get_name());

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  fsrc->fill_compile_info(rval);
  return rval;
}



} // End namespace SCIRun
