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
 *  Probe.cc:  Rotate and flip field to get it into "standard" view
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
#include <Dataflow/Widgets/PointWidget.h>
#include <Core/Datatypes/PointCloud.h>
#include <Core/Datatypes/Clipper.h>
#include <Dataflow/Modules/Fields/Probe.h>
#include <iostream>
#include <stack>

namespace SCIRun {

using std::stack;

class Probe : public Module
{
private:
  PointWidget *widget_;
  CrowdMonitor widget_lock_;
  int  last_input_generation_;
  BBox last_bounds_;

  GuiDouble gui_locx_;
  GuiDouble gui_locy_;
  GuiDouble gui_locz_;
  GuiString gui_value_;
  GuiString gui_node_;
  GuiString gui_edge_;
  GuiString gui_face_;
  GuiString gui_cell_;
  GuiString gui_moveto_;
  
  bool bbox_similar_to(const BBox &a, const BBox &b);

public:
  Probe(const string& id);
  virtual ~Probe();

  virtual void execute();
  virtual void widget_moved(int);
};


extern "C" Module* make_Probe(const string& id)
{
  return new Probe(id);
}


Probe::Probe(const string& id)
  : Module("Probe", id, Source, "Fields", "SCIRun"),
    widget_lock_("Probe widget lock"),
    last_input_generation_(0),
    gui_locx_("locx", id, this),
    gui_locy_("locy", id, this),
    gui_locz_("locz", id, this),
    gui_value_("value", id, this),
    gui_node_("node", id, this),
    gui_edge_("edge", id, this),
    gui_face_("face", id, this),
    gui_cell_("cell", id, this),
    gui_moveto_("moveto", id, this)
{
  widget_ = scinew PointWidget(this, &widget_lock_, 1.0);
}


Probe::~Probe()
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
Probe::bbox_similar_to(const BBox &a, const BBox &b)
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
Probe::execute()
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

  // Maybe update the widget.
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

    Point center = bmin + Vector(bmax - bmin) * 0.5;
    const double l2norm = (bmax - bmin).length();

    // If the current location looks reasonable, use that instead
    // of the center.
    Point location(gui_locx_.get(), gui_locy_.get(), gui_locz_.get());
    
    if (location.x() >= bmin.x() && location.x() <= bmax.x() && 
	location.y() >= bmin.y() && location.y() <= bmax.y() && 
	location.z() >= bmin.z() && location.z() <= bmax.z())
    {
      center = location;
    }

    widget_->SetScale(l2norm * 0.015);
    widget_->SetPosition(center);

    GeomGroup *widget_group = scinew GeomGroup;
    widget_group->add(widget_->GetWidget());

    GeometryOPort *ogport=0;
    ogport = (GeometryOPort*)get_oport("Probe Widget");
    if (!ogport)
    {
      error("Unable to initialize " + name + "'s oport.");
      return;
    }
    ogport->addObj(widget_group, "Probe Selection Widget",
		   &widget_lock_);
    ogport->flushViews();

    last_bounds_ = bbox;
  }

  const string &moveto = gui_moveto_.get();
  bool moved_p = false;
  if (moveto == "location")
  {
    const Point newloc(gui_locx_.get(), gui_locy_.get(), gui_locz_.get());
    widget_->SetPosition(newloc);
    moved_p = true;
  }
  else if (moveto == "center")
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

    Point center = bmin + Vector(bmax - bmin) * 0.5;
    widget_->SetPosition(center);
    moved_p = true;
  }
  else if (moveto != "")
  {
    const TypeDescription *mtd = ifieldhandle->mesh()->get_type_description();
    CompileInfo *ci = ProbeCenterAlgo::get_compile_info(mtd);
    DynamicAlgoHandle algo_handle;
    if (! DynamicLoader::scirun_loader().get(*ci, algo_handle))
    {
      error("Could not compile algorithm.");
      return;
    }
    ProbeCenterAlgo *algo =
      dynamic_cast<ProbeCenterAlgo *>(algo_handle.get_rep());
    if (algo == 0)
    {
      error("Could not get algorithm.");
      return;
    }
    if (moveto == "node")
    {
      Point newloc = widget_->GetPosition();
      if (algo->get_node(ifieldhandle->mesh(), gui_node_.get(), newloc))
      {
	widget_->SetPosition(newloc);
	moved_p = true;
      }
    }
    else if (moveto == "edge")
    {
      Point newloc = widget_->GetPosition();
      if (algo->get_edge(ifieldhandle->mesh(), gui_edge_.get(), newloc))
      {
	widget_->SetPosition(newloc);
	moved_p = true;
      }
    }
    else if (moveto == "face")
    {
      Point newloc = widget_->GetPosition();
      if (algo->get_face(ifieldhandle->mesh(), gui_face_.get(), newloc))
      {
	widget_->SetPosition(newloc);
	moved_p = true;
      }
    }
    else if (moveto == "cell")
    {
      Point newloc = widget_->GetPosition();
      if (algo->get_cell(ifieldhandle->mesh(), gui_cell_.get(), newloc))
      {
	widget_->SetPosition(newloc);
	moved_p = true;
      }
    }
  }
  if (moved_p)
  {
    GeometryOPort *ogport = (GeometryOPort*)get_oport("Probe Widget");
    if (!ogport)
    {
      error("Unable to initialize " + name + "'s oport.");
      return;
    }
    ogport->flushViews();
    gui_moveto_.set("");
  }

  const Point location = widget_->GetPosition();
  PointCloudMesh *mesh = scinew PointCloudMesh();
  PointCloudMesh::Node::index_type pcindex = mesh->add_point(location);
  FieldHandle ofield;
  std::ostringstream valstr;
  ScalarFieldInterface *sfi = ifieldhandle->query_scalar_interface();
  VectorFieldInterface *vfi = ifieldhandle->query_vector_interface();
  TensorFieldInterface *tfi = ifieldhandle->query_tensor_interface();
  if (sfi)
  {
    double result;
    if (!sfi->interpolate(result, location))
    {
      sfi->find_closest(result, location);
    }
    valstr << result;

    PointCloud<double> *field = scinew PointCloud<double>(mesh, Field::NODE);
    field->set_value(result, pcindex);
    ofield = field;
  }
  else if (vfi)
  {
    Vector result;
    if (!vfi->interpolate(result, location))
    {
      vfi->find_closest(result, location);
    }
    valstr << result;

    PointCloud<Vector> *field = scinew PointCloud<Vector>(mesh, Field::NODE);
    field->set_value(result, pcindex);
    ofield = field;
  }
  else if (tfi)
  {
    Tensor result;
    if (!tfi->interpolate(result, location))
    {
      tfi->find_closest(result, location);
    }
    valstr << result;

    PointCloud<Tensor> *field = scinew PointCloud<Tensor>(mesh, Field::NODE);
    field->set_value(result, pcindex);
    ofield = field;
  }
  gui_locx_.set(location.x());
  gui_locy_.set(location.y());
  gui_locz_.set(location.z());
  gui_value_.set(valstr.str());

  const TypeDescription *mtd = ifieldhandle->mesh()->get_type_description();
  CompileInfo *ci = ProbeLocateAlgo::get_compile_info(mtd);
  DynamicAlgoHandle algo_handle;
  if (! DynamicLoader::scirun_loader().get(*ci, algo_handle))
  {
    error("Could not compile algorithm.");
    return;
  }
  ProbeLocateAlgo *algo =
    dynamic_cast<ProbeLocateAlgo *>(algo_handle.get_rep());
  if (algo == 0)
  {
    error("Could not get algorithm.");
    return;
  }
  string nodestr, edgestr, facestr, cellstr;
  algo->execute(ifieldhandle->mesh(), location,
		nodestr, edgestr, facestr, cellstr);

  gui_node_.set(nodestr);
  gui_edge_.set(edgestr);
  gui_face_.set(facestr);
  gui_cell_.set(cellstr);
  //cout << "PROBE VALUE AT " << location << " IS " << valstr.str() << '\n';

  FieldOPort *ofp = (FieldOPort *)get_oport("Probe Point");
  if (!ofp) {
    postMessage("Unable to initialize " + name + "'s oport\n");
    return;
  }
  ofp->send(ofield);
}


void
Probe::widget_moved(int i)
{
  if (i == 1)
  {
    want_to_execute();
  }
}



CompileInfo *
ProbeLocateAlgo::get_compile_info(const TypeDescription *msrc)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("ProbeLocateAlgoT");
  static const string base_class_name("ProbeLocateAlgo");

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       msrc->get_filename() + ".",
                       base_class_name, 
                       template_class_name, 
                       msrc->get_name());

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  msrc->fill_compile_info(rval);
  return rval;
}


CompileInfo *
ProbeCenterAlgo::get_compile_info(const TypeDescription *msrc)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("ProbeCenterAlgoT");
  static const string base_class_name("ProbeCenterAlgo");

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       msrc->get_filename() + ".",
                       base_class_name, 
                       template_class_name, 
                       msrc->get_name());

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  msrc->fill_compile_info(rval);
  return rval;
}


} // End namespace SCIRun


