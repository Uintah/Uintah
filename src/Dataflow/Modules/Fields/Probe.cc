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
#include <Core/Datatypes/FieldInterface.h>
#include <Core/Geometry/Tensor.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Thread/CrowdMonitor.h>
#include <Dataflow/Widgets/PointWidget.h>
#include <Core/Datatypes/PointCloudField.h>
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
  GuiInt gui_show_value_;
  GuiInt gui_show_node_;
  GuiInt gui_show_edge_;
  GuiInt gui_show_face_;
  GuiInt gui_show_cell_;
  GuiString gui_moveto_;
  GuiDouble gui_probe_scale_;

  int widgetid_;

  bool bbox_similar_to(const BBox &a, const BBox &b);
  double l2norm_;

public:
  Probe(GuiContext* ctx);
  virtual ~Probe();

  virtual void execute();
  virtual void widget_moved(bool);
};


DECLARE_MAKER(Probe)

  Probe::Probe(GuiContext* ctx)
    : Module("Probe", ctx, Filter, "Fields", "SCIRun"),
      widget_lock_("Probe widget lock"),
      last_input_generation_(0),
      gui_locx_(ctx->subVar("locx")),
      gui_locy_(ctx->subVar("locy")),
      gui_locz_(ctx->subVar("locz")),
      gui_value_(ctx->subVar("value")),
      gui_node_(ctx->subVar("node")),
      gui_edge_(ctx->subVar("edge")),
      gui_face_(ctx->subVar("face")),
      gui_cell_(ctx->subVar("cell")),
      gui_show_value_(ctx->subVar("show-value")),
      gui_show_node_(ctx->subVar("show-node")),
      gui_show_edge_(ctx->subVar("show-edge")),
      gui_show_face_(ctx->subVar("show-face")),
      gui_show_cell_(ctx->subVar("show-cell")),
      gui_moveto_(ctx->subVar("moveto", false)),
      gui_probe_scale_(ctx->subVar("probe_scale")),
      widgetid_(0)
{
  widget_ = scinew PointWidget(this, &widget_lock_, 1.0);
  widget_->Connect((GeometryOPort*)get_oport("Probe Widget"));
}


Probe::~Probe()
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
  bool input_field_p = true;
  if (!(ifp->get(ifieldhandle) && ifieldhandle.get_rep()))
  {
    input_field_p = false;
  }

  // Maybe update the widget.
  BBox bbox;
  if (input_field_p)
  {
    bbox = ifieldhandle->mesh()->get_bounding_box();
  }
  else
  {
    bbox.extend(Point(-1.0, -1.0, -1.0));
    bbox.extend(Point(1.0, 1.0, 1.0));
  }

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
    l2norm_ = (bmax - bmin).length();

    // If the current location looks reasonable, use that instead
    // of the center.
    Point curloc(gui_locx_.get(), gui_locy_.get(), gui_locz_.get());
    
    if (curloc.x() >= bmin.x() && curloc.x() <= bmax.x() && 
	curloc.y() >= bmin.y() && curloc.y() <= bmax.y() && 
	curloc.z() >= bmin.z() && curloc.z() <= bmax.z())
    {
      center = curloc;
    }
    
    widget_->SetPosition(center);

    GeomGroup *widget_group = scinew GeomGroup;
    widget_group->add(widget_->GetWidget());

    GeometryOPort *ogport = (GeometryOPort*)get_oport("Probe Widget");
    if (!ogport)
    {
      error("Unable to initialize " + name + "'s oport.");
      return;
    }
    widgetid_ = ogport->addObj(widget_group, "Probe Selection Widget",
			       &widget_lock_);
    ogport->flushViews();

    last_bounds_ = bbox;
  }

  widget_->SetScale(gui_probe_scale_.get() * l2norm_ * 0.003);

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
  else if (moveto != "" && input_field_p)
  {
    const TypeDescription *mtd = ifieldhandle->mesh()->get_type_description();
    CompileInfoHandle ci = ProbeCenterAlgo::get_compile_info(mtd);
    Handle<ProbeCenterAlgo> algo;
    if (!module_dynamic_compile(ci, algo)) return;

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

  string nodestr, edgestr, facestr, cellstr;
  if (input_field_p)
  {
    const TypeDescription *mtd = ifieldhandle->mesh()->get_type_description();
    CompileInfoHandle ci = ProbeLocateAlgo::get_compile_info(mtd);
    Handle<ProbeLocateAlgo> algo;
    if (!module_dynamic_compile(ci, algo)) return;

    algo->execute(ifieldhandle->mesh(), location,
		  gui_show_node_.get(), nodestr,
		  gui_show_edge_.get(), edgestr,
		  gui_show_face_.get(), facestr,
		  gui_show_cell_.get(), cellstr);

    if (gui_show_node_.get()) { gui_node_.set(nodestr); }
    if (gui_show_edge_.get()) { gui_edge_.set(edgestr); }
    if (gui_show_face_.get()) { gui_face_.set(facestr); }
    if (gui_show_cell_.get()) { gui_cell_.set(cellstr); }
  }

  std::ostringstream valstr;
  ScalarFieldInterfaceHandle sfi = 0;
  VectorFieldInterfaceHandle vfi = 0;
  TensorFieldInterfaceHandle tfi = 0;
  if (!input_field_p ||
      ifieldhandle->data_at() == Field::NONE ||
      !gui_show_value_.get())
  {
    valstr << 0;
    PointCloudField<double> *field =
      scinew PointCloudField<double>(mesh, Field::NODE);
    field->set_value(0.0, pcindex);
    ofield = field;
  }
  else if ((sfi = ifieldhandle->query_scalar_interface(this)).get_rep())
  {
    double result;
    if (!sfi->interpolate(result, location))
    {
      sfi->find_closest(result, location);
    }
    valstr << result;

    PointCloudField<double> *field = scinew PointCloudField<double>(mesh, Field::NODE);
    field->set_value(result, pcindex);
    ofield = field;
  }
  else if ((vfi = ifieldhandle->query_vector_interface(this)).get_rep())
  {
    Vector result;
    if (!vfi->interpolate(result, location))
    {
      vfi->find_closest(result, location);
    }
    valstr << result;

    PointCloudField<Vector> *field = scinew PointCloudField<Vector>(mesh, Field::NODE);
    field->set_value(result, pcindex);
    ofield = field;
  }
  else if ((tfi = ifieldhandle->query_tensor_interface(this)).get_rep())
  {
    Tensor result;
    if (!tfi->interpolate(result, location))
    {
      tfi->find_closest(result, location);
    }
    valstr << result;

    PointCloudField<Tensor> *field = scinew PointCloudField<Tensor>(mesh, Field::NODE);
    field->set_value(result, pcindex);
    ofield = field;
  }
  gui_locx_.set(location.x());
  gui_locy_.set(location.y());
  gui_locz_.set(location.z());
  if (gui_show_value_.get()) { gui_value_.set(valstr.str()); }

  FieldOPort *ofp = (FieldOPort *)get_oport("Probe Point");
  if (!ofp) {
    error("Unable to initialize oport 'Probe Point'.");
    return;
  }
  ofp->send(ofield);

  if (input_field_p)
  {
    MatrixOPort *mp = (MatrixOPort *)get_oport("Element Index");
    if (!mp)
    {
      error("Unable to initialize oport 'Element Index'.");
      return;
    }
    unsigned int index = 0;
    switch (ifieldhandle->data_at())
    {
    case Field::NODE:
      index = atoi(nodestr.c_str());
      break;
    case Field::EDGE:
      index = atoi(edgestr.c_str());
      break;
    case Field::FACE:
      index = atoi(facestr.c_str());
      break;
    case Field::CELL:
      index = atoi(cellstr.c_str());
      break;
    case Field::NONE:
      break;
    }
    MatrixHandle cm = scinew ColumnMatrix(1);
    cm->get(0, 0) = (double)index;
    mp->send(cm);
  }
}


void
Probe::widget_moved(bool last)
{
  if (last)
  {
    want_to_execute();
  }
}


CompileInfoHandle
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


CompileInfoHandle
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


template <>
bool
probe_center_compute_index(LatVolMesh::Node::index_type &index,
			   LatVolMesh::Node::size_type &size,
			   const LatVolMesh *mesh, const string &indexstr)
{
  istringstream istr(indexstr);

  int i;
  unsigned int idx[3];
  for (i = 0; i < 3 && !istr.eof() && !istr.fail(); i++)
  {
    istr >> idx[i];
  }
  if (i == 1)
  {
    const int mij = (mesh->get_ni() * mesh->get_nj());
    idx[2] = idx[0] / mij;
    idx[1] = idx[0] % mij;
    idx[0] = idx[1] % mesh->get_ni();
    idx[1] = idx[1] / mesh->get_ni();
  }
  else if (i != 3)
  {
    return false;
  }
  mesh->size(size);
  if (idx[0] < size.i_ && idx[1] < size.j_ && idx[2] < size.k_)
  {
    index = LatVolMesh::Node::index_type(mesh, idx[0], idx[1], idx[2]);
    return true;
  }
  return false;
}

template <>
bool
probe_center_compute_index(LatVolMesh::Cell::index_type &index,
			   LatVolMesh::Cell::size_type &size,
			   const LatVolMesh *mesh, const string &indexstr)
{
  istringstream istr(indexstr);

  int i;
  unsigned int idx[3];
  for (i = 0; i < 3 && !istr.eof() && !istr.fail(); i++)
  {
    istr >> idx[i];
  }
  if (i == 1)
  {
    const int mij = ((mesh->get_ni()-1) * (mesh->get_nj()-1));
    idx[2] = idx[0] / mij;
    idx[1] = idx[0] % mij;
    idx[0] = idx[1] % (mesh->get_ni()-1);
    idx[1] = idx[1] / (mesh->get_ni()-1);
  }
  else if (i != 3)
  {
    return false;
  }
  mesh->size(size);
  if (idx[0] < size.i_ && idx[1] < size.j_ && idx[2] < size.k_)
  {
    index = LatVolMesh::Cell::index_type(mesh, idx[0], idx[1], idx[2]);
    return true;
  }
  return false;
}


template <>
bool
probe_center_compute_index(StructHexVolMesh::Node::index_type &index,
			   StructHexVolMesh::Node::size_type &size,
			   const StructHexVolMesh *mesh,
			   const string &indexstr)
{
  istringstream istr(indexstr);

  int i;
  unsigned int idx[3];
  for (i = 0; i < 3 && !istr.eof() && !istr.fail(); i++)
  {
    istr >> idx[i];
  }
  if (i == 1)
  {
    const int mij = (mesh->get_ni() * mesh->get_nj());
    idx[2] = idx[0] / mij;
    idx[1] = idx[0] % mij;
    idx[0] = idx[1] % mesh->get_ni();
    idx[1] = idx[1] / mesh->get_ni();
  }
  else if (i != 3)
  {
    return false;
  }
  mesh->size(size);
  if (idx[0] < size.i_ && idx[1] < size.j_ && idx[2] < size.k_)
  {
    index = StructHexVolMesh::Node::index_type(mesh, idx[0], idx[1], idx[2]);
    return true;
  }
  return false;
}


template <>
bool
probe_center_compute_index(StructHexVolMesh::Cell::index_type &index,
			   StructHexVolMesh::Cell::size_type &size,
			   const StructHexVolMesh *mesh,
			   const string &indexstr)
{
  istringstream istr(indexstr);

  int i;
  unsigned int idx[3];
  for (i = 0; i < 3 && !istr.eof() && !istr.fail(); i++)
  {
    istr >> idx[i];
  }
  if (i == 1)
  {
    const int mij = ((mesh->get_ni()-1) * (mesh->get_nj()-1));
    idx[2] = idx[0] / mij;
    idx[1] = idx[0] % mij;
    idx[0] = idx[1] % (mesh->get_ni()-1);
    idx[1] = idx[1] / (mesh->get_ni()-1);
  }
  else if (i != 3)
  {
    return false;
  }
  mesh->size(size);
  if (idx[0] < size.i_ && idx[1] < size.j_ && idx[2] < size.k_)
  {
    index = StructHexVolMesh::Cell::index_type(mesh, idx[0], idx[1], idx[2]);
    return true;
  }
  return false;
}



template <>
bool
probe_center_compute_index(ImageMesh::Node::index_type &index,
			   ImageMesh::Node::size_type &size,
			   const ImageMesh *mesh, const string &indexstr)
{
  istringstream istr(indexstr);

  int i;
  unsigned int idx[2];
  for (i = 0; i < 2 && !istr.eof() && !istr.fail(); i++)
  {
    istr >> idx[i];
  }
  if (i == 1)
  {
    idx[1] = idx[0] / mesh->get_ni();
    idx[0] = idx[0] % mesh->get_ni();
  }
  else if (i != 2)
  {
    return false;
  }
  mesh->size(size);
  if (idx[0] < size.i_ && idx[1] < size.j_)
  {
    index = ImageMesh::Node::index_type(mesh, idx[0], idx[1]);
    return true;
  }
  return false;
}


template <>
bool
probe_center_compute_index(ImageMesh::Face::index_type &index,
			   ImageMesh::Face::size_type &size,
			   const ImageMesh *mesh, const string &indexstr)
{
  istringstream istr(indexstr);

  int i;
  unsigned int idx[2];
  for (i = 0; i < 2 && !istr.eof() && !istr.fail(); i++)
  {
    istr >> idx[i];
  }
  if (i == 1)
  {
    idx[1] = idx[0] / (mesh->get_ni()-1);
    idx[0] = idx[0] % (mesh->get_ni()-1);
  }
  else if (i != 2)
  {
    return false;
  }
  mesh->size(size);
  if (idx[0] < size.i_ && idx[1] < size.j_)
  {
    index = ImageMesh::Face::index_type(mesh, idx[0], idx[1]);
    return true;
  }
  return false;
}


template <>
bool
probe_center_compute_index(StructQuadSurfMesh::Node::index_type &index,
			   StructQuadSurfMesh::Node::size_type &size,
			   const StructQuadSurfMesh *mesh,
			   const string &indexstr)
{
  istringstream istr(indexstr);

  int i;
  unsigned int idx[2];
  for (i = 0; i < 2 && !istr.eof() && !istr.fail(); i++)
  {
    istr >> idx[i];
  }
  if (i == 1)
  {
    idx[1] = idx[0] / mesh->get_ni();
    idx[0] = idx[0] % mesh->get_ni();
  }
  else if (i != 2)
  {
    return false;
  }
  mesh->size(size);
  if (idx[0] < size.i_ && idx[1] < size.j_)
  {
    index = StructQuadSurfMesh::Node::index_type(mesh, idx[0], idx[1]);
    return true;
  }
  return false;
}


template <>
bool
probe_center_compute_index(StructQuadSurfMesh::Face::index_type &index,
			   StructQuadSurfMesh::Face::size_type &size,
			   const StructQuadSurfMesh *mesh,
			   const string &indexstr)
{
  istringstream istr(indexstr);

  int i;
  unsigned int idx[2];
  for (i = 0; i < 2 && !istr.eof() && !istr.fail(); i++)
  {
    istr >> idx[i];
  }
  if (i == 1)
  {
    idx[1] = idx[0] / (mesh->get_ni()-1);
    idx[0] = idx[0] % (mesh->get_ni()-1);
  }
  else if (i != 2)
  {
    return false;
  }
  mesh->size(size);
  if (idx[0] < size.i_ && idx[1] < size.j_)
  {
    index = StructQuadSurfMesh::Face::index_type(mesh, idx[0], idx[1]);
    return true;
  }
  return false;
}


} // End namespace SCIRun


