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
#include <iostream>
#include <sstream>
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
    last_input_generation_(0)
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

    const Point center = bmin + Vector(bmax - bmin) * 0.5;
    const double l2norm = (bmax - bmin).length();

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

  std::ostringstream valstr;
  ScalarFieldInterface *sfi = ifieldhandle->query_scalar_interface();
  VectorFieldInterface *vfi = ifieldhandle->query_vector_interface();
  TensorFieldInterface *tfi = ifieldhandle->query_tensor_interface();
  const Point location = widget_->GetPosition();
  if (sfi)
  {
    double result;
    if (!sfi->interpolate(result, location))
    {
      sfi->find_closest(result, location);
    }
    valstr << result;
  }
  else if (vfi)
  {
    Vector result;
    if (!vfi->interpolate(result, location))
    {
      vfi->find_closest(result, location);
    }
    valstr << result;
  }
  else if (tfi)
  {
    Tensor result;
    if (!tfi->interpolate(result, location))
    {
      tfi->find_closest(result, location);
    }
    valstr << result;
  }
  cout << "PROBE VALUE AT " << location << " IS " << valstr.str() << '\n';


  PointCloudMesh *mesh = scinew PointCloudMesh();
  mesh->add_point(location);
  PointCloud<double> *ofield = scinew PointCloud<double>(mesh, Field::NODE);

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


} // End namespace SCIRun


