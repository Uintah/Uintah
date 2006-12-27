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



/*
 *  GenerateSinglePointProbeFromField.cc:  Rotate and flip field to get it into "standard" view
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   March 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Dataflow/Modules/Fields/GenerateSinglePointProbeFromField.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Network/Ports/FieldPort.h>
#include <Dataflow/Network/Ports/GeometryPort.h>
#include <Dataflow/Network/Ports/MatrixPort.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Thread/CrowdMonitor.h>
#include <Dataflow/Widgets/PointWidget.h>

#include <Core/Datatypes/GenericField.h>

#include <Core/Datatypes/Clipper.h>
#include <iostream>
#include <stack>

namespace SCIRun {

using std::stack;

class GenerateSinglePointProbeFromField : public Module
{
private:
  PointWidget *widget_;
  CrowdMonitor widget_lock_;
  int  last_input_generation_;
  BBox last_bounds_;

  GuiString gui_frame_;
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
  typedef PointCloudMesh<ConstantBasis<Point> > PCMesh;
  GenerateSinglePointProbeFromField(GuiContext* ctx);
  virtual ~GenerateSinglePointProbeFromField();

  virtual void execute();
  virtual void widget_moved(bool, BaseWidget*);
};


DECLARE_MAKER(GenerateSinglePointProbeFromField)

GenerateSinglePointProbeFromField::GenerateSinglePointProbeFromField(GuiContext* ctx)
  : Module("GenerateSinglePointProbeFromField", ctx, Filter, "NewField", "SCIRun"),
    widget_lock_("GenerateSinglePointProbeFromField widget lock"),
    last_input_generation_(0),
    gui_frame_(get_ctx()->subVar("main_frame"), ""),
    gui_locx_(get_ctx()->subVar("locx"), 0.0),
    gui_locy_(get_ctx()->subVar("locy"), 0.0),
    gui_locz_(get_ctx()->subVar("locz"), 0.0),
    gui_value_(get_ctx()->subVar("value"), ""),
    gui_node_(get_ctx()->subVar("node"), ""),
    gui_edge_(get_ctx()->subVar("edge"), ""),
    gui_face_(get_ctx()->subVar("face"), ""),
    gui_cell_(get_ctx()->subVar("cell"), ""),
    gui_show_value_(get_ctx()->subVar("show-value"), 1),
    gui_show_node_(get_ctx()->subVar("show-node"), 1),
    gui_show_edge_(get_ctx()->subVar("show-edge"), 0),
    gui_show_face_(get_ctx()->subVar("show-face"), 0),
    gui_show_cell_(get_ctx()->subVar("show-cell"), 1),
    gui_moveto_(get_ctx()->subVar("moveto", false), ""),
    gui_probe_scale_(get_ctx()->subVar("probe_scale"), 5.0),
    widgetid_(0)
{
  widget_ = scinew PointWidget(this, &widget_lock_, 1.0);
  widget_->Connect((GeometryOPort*)get_oport("GenerateSinglePointProbeFromField Widget"));
}


GenerateSinglePointProbeFromField::~GenerateSinglePointProbeFromField()
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
GenerateSinglePointProbeFromField::bbox_similar_to(const BBox &a, const BBox &b)
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
GenerateSinglePointProbeFromField::execute()
{
  // Get input field.
  FieldHandle ifieldhandle;
  const bool input_field_p =
    get_input_handle("Input Field", ifieldhandle, false);

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

    // Invalidate current position if it's outside of our field.
    // Leave it alone if there was no field, as our bbox is arbitrary anyway.
    if (curloc.x() >= bmin.x() && curloc.x() <= bmax.x() && 
	curloc.y() >= bmin.y() && curloc.y() <= bmax.y() && 
	curloc.z() >= bmin.z() && curloc.z() <= bmax.z() ||
	!input_field_p)
    {
      center = curloc;
    }
    
    widget_->SetPosition(center);

    GeomGroup *widget_group = scinew GeomGroup;
    widget_group->add(widget_->GetWidget());

    GeometryOPort *ogport = (GeometryOPort*)get_oport("GenerateSinglePointProbeFromField Widget");
    widgetid_ = ogport->addObj(widget_group, "GenerateSinglePointProbeFromField Selection Widget",
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
    CompileInfoHandle ci = GenerateSinglePointProbeFromFieldCenterAlgo::get_compile_info(mtd);
    Handle<GenerateSinglePointProbeFromFieldCenterAlgo> algo;
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
    GeometryOPort *ogport = (GeometryOPort*)get_oport("GenerateSinglePointProbeFromField Widget");
    ogport->flushViews();
    gui_moveto_.set("");
  }

  const Point location = widget_->GetPosition();
  PCMesh *mesh = scinew PCMesh();
  PCMesh::Node::index_type pcindex = mesh->add_point(location);
  FieldHandle ofield;

  string nodestr, edgestr, facestr, cellstr;
  if (input_field_p)
  {
    const TypeDescription *mtd = ifieldhandle->mesh()->get_type_description();
    CompileInfoHandle ci = GenerateSinglePointProbeFromFieldLocateAlgo::get_compile_info(mtd);
    Handle<GenerateSinglePointProbeFromFieldLocateAlgo> algo;
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
  typedef ConstantBasis<double>                             DatBasis;
  typedef ConstantBasis<Tensor>                             DatTBasis;
  typedef ConstantBasis<Vector>                             DatVBasis;
  typedef GenericField<PCMesh, DatBasis, vector<double> >   PCField; 
  typedef GenericField<PCMesh, DatTBasis, vector<Tensor> >  PCFieldT;
  typedef GenericField<PCMesh, DatVBasis, vector<Vector> >  PCFieldV;


  std::ostringstream valstr;
  ScalarFieldInterfaceHandle sfi = 0;
  VectorFieldInterfaceHandle vfi = 0;
  TensorFieldInterfaceHandle tfi = 0;
  if (!input_field_p ||
      ifieldhandle->basis_order() == -1 ||
      !gui_show_value_.get())
  {
    valstr << 0;
    PCField *field = scinew PCField(mesh);
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

    PCField *field = scinew PCField(mesh);
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

    PCFieldV *field = scinew PCFieldV(mesh);
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

    PCFieldT *field = scinew PCFieldT(mesh);
    field->set_value(result, pcindex);
    ofield = field;
  }
  gui_locx_.set(location.x());
  gui_locy_.set(location.y());
  gui_locz_.set(location.z());
  if (gui_show_value_.get()) { gui_value_.set(valstr.str()); }

  send_output_handle("GenerateSinglePointProbeFromField Point", ofield);

  if (input_field_p)
  {
    unsigned int index = 0;
    switch (ifieldhandle->basis_order())
    {
    case 1:
      index = atoi(nodestr.c_str());
      break;
    case 0:
      {
	if (ifieldhandle->mesh()->dimensionality() == 1) {
	  index = atoi(edgestr.c_str());
	} else if (ifieldhandle->mesh()->dimensionality() == 2) {
	  index = atoi(facestr.c_str());
	} else if (ifieldhandle->mesh()->dimensionality() == 3) {
	  index = atoi(cellstr.c_str());
	}
      }
      break;
    case -1:
      break;
    }
    MatrixHandle cm = scinew ColumnMatrix(1);
    cm->put(0, 0, index);
    send_output_handle("Element Index", cm);
  }
}


void
GenerateSinglePointProbeFromField::widget_moved(bool last, BaseWidget*)
{
  if (last)
  {
    want_to_execute();
  }
}


CompileInfoHandle
GenerateSinglePointProbeFromFieldLocateAlgo::get_compile_info(const TypeDescription *msrc)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("GenerateSinglePointProbeFromFieldLocateAlgoT");
  static const string base_class_name("GenerateSinglePointProbeFromFieldLocateAlgo");

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
GenerateSinglePointProbeFromFieldCenterAlgo::get_compile_info(const TypeDescription *msrc)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("GenerateSinglePointProbeFromFieldCenterAlgoT");
  static const string base_class_name("GenerateSinglePointProbeFromFieldCenterAlgo");

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
probe_center_compute_index(LVMesh::Node::index_type &index,
			   LVMesh::Node::size_type &size,
			   const LVMesh *mesh, const string &indexstr)
{
  string tempstr;
  for (unsigned int pos = 0; pos < indexstr.size(); pos++)
    if ((int)indexstr[pos] >= (int)'0' &&
	(int)indexstr[pos] <= (int)'9') {
      tempstr += indexstr[pos];
    } else {
      tempstr += " ";
    }

  istringstream istr(tempstr);

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
    index = LVMesh::Node::index_type(mesh, idx[0], idx[1], idx[2]);
    return true;
  }
  return false;
}

template <>
bool
probe_center_compute_index(LVMesh::Cell::index_type &index,
			   LVMesh::Cell::size_type &size,
			   const LVMesh *mesh, const string &indexstr)
{
  string tempstr;
  for (unsigned int pos = 0; pos < indexstr.size(); pos++)
    if ((int)indexstr[pos] >= (int)'0' &&
	(int)indexstr[pos] <= (int)'9') {
      tempstr += indexstr[pos];
    } else {
      tempstr += " ";
    }

  istringstream istr(tempstr);

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
    index = LVMesh::Cell::index_type(mesh, idx[0], idx[1], idx[2]);
    return true;
  }
  return false;
}


template <>
bool
probe_center_compute_index(SHVMesh::Node::index_type &index,
			   SHVMesh::Node::size_type &size,
			   const SHVMesh *mesh,
			   const string &indexstr)
{
  string tempstr;
  for (unsigned int pos = 0; pos < indexstr.size(); pos++)
    if ((int)indexstr[pos] >= (int)'0' &&
	(int)indexstr[pos] <= (int)'9') {
      tempstr += indexstr[pos];
    } else {
      tempstr += " ";
    }

  istringstream istr(tempstr);


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
    index = SHVMesh::Node::index_type(mesh, idx[0], idx[1], idx[2]);
    return true;
  }
  return false;
}


template <>
bool
probe_center_compute_index(SHVMesh::Cell::index_type &index,
			   SHVMesh::Cell::size_type &size,
			   const SHVMesh *mesh,
			   const string &indexstr)
{
  string tempstr;
  for (unsigned int pos = 0; pos < indexstr.size(); pos++)
    if ((int)indexstr[pos] >= (int)'0' &&
	(int)indexstr[pos] <= (int)'9') {
      tempstr += indexstr[pos];
    } else {
      tempstr += " ";
    }

  istringstream istr(tempstr);
 
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
    index = SHVMesh::Cell::index_type(mesh, idx[0], idx[1], idx[2]);
    return true;
  }
  return false;
}



template <>
bool
probe_center_compute_index(IMesh::Node::index_type &index,
			   IMesh::Node::size_type &size,
			   const IMesh *mesh, const string &indexstr)
{
  string tempstr;
  for (unsigned int pos = 0; pos < indexstr.size(); pos++)
    if ((int)indexstr[pos] >= (int)'0' &&
	(int)indexstr[pos] <= (int)'9') {
      tempstr += indexstr[pos];
    } else {
      tempstr += " ";
    }

  istringstream istr(tempstr);

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
    index = IMesh::Node::index_type(mesh, idx[0], idx[1]);
    return true;
  }
  return false;
}


template <>
bool
probe_center_compute_index(IMesh::Face::index_type &index,
			   IMesh::Face::size_type &size,
			   const IMesh *mesh, const string &indexstr)
{
  string tempstr;
  for (unsigned int pos = 0; pos < indexstr.size(); pos++)
    if ((int)indexstr[pos] >= (int)'0' &&
	(int)indexstr[pos] <= (int)'9') {
      tempstr += indexstr[pos];
    } else {
      tempstr += " ";
    }

  istringstream istr(tempstr);

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
    index = IMesh::Face::index_type(mesh, idx[0], idx[1]);
    return true;
  }
  return false;
}


template <>
bool
probe_center_compute_index(SQSMesh::Node::index_type &index,
			   SQSMesh::Node::size_type &size,
			   const SQSMesh *mesh,
			   const string &indexstr)
{
  string tempstr;
  for (unsigned int pos = 0; pos < indexstr.size(); pos++)
    if ((int)indexstr[pos] >= (int)'0' &&
	(int)indexstr[pos] <= (int)'9') {
      tempstr += indexstr[pos];
    } else {
      tempstr += " ";
    }

  istringstream istr(tempstr);

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
    index = SQSMesh::Node::index_type(mesh, idx[0], idx[1]);
    return true;
  }
  return false;
}


template <>
bool
probe_center_compute_index(SQSMesh::Face::index_type &index,
			   SQSMesh::Face::size_type &size,
			   const SQSMesh *mesh,
			   const string &indexstr)
{
  string tempstr;
  for (unsigned int pos = 0; pos < indexstr.size(); pos++)
    if ((int)indexstr[pos] >= (int)'0' &&
	(int)indexstr[pos] <= (int)'9') {
      tempstr += indexstr[pos];
    } else {
      tempstr += " ";
    }

  istringstream istr(tempstr);

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
    index = SQSMesh::Face::index_type(mesh, idx[0], idx[1]);
    return true;
  }
  return false;
}


} // End namespace SCIRun


