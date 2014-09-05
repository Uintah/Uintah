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
 *  SeedPoints.cc
 *
 *  Written by:
 *   Robert Van Uitert
 *   Diagnostic Radiology Department
 *   National Institutes of Health
 *   November 2004
 *
 */


#include <Dataflow/Network/Module.h>
#include <Core/Datatypes/FieldInterface.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Geometry/Tensor.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Thread/CrowdMonitor.h>
#include <Dataflow/Widgets/PointWidget.h>
#include <Dataflow/Widgets/ArrowWidget.h>
#include <Dataflow/Widgets/RingWidget.h>
#include <Core/Datatypes/PointCloudField.h>
#include <Core/Datatypes/Clipper.h>
#include <Core/Datatypes/LatVolField.h>
#include <Core/Datatypes/ImageField.h>
#include <iostream>
#include <stack>

namespace SCIRun {

using std::stack;

class SeedPoints : public Module
{
private:
  CrowdMonitor widget_lock_;
  BBox last_bounds_;

  vector<int>              widget_id_;
  vector<GeomHandle>       widget_switch_;
  vector<PointWidget*>     widget_;
  vector<ArrowWidget*>     widget_vec_;
  vector<RingWidget*>      widget_ring_;

  bool bbox_similar_to(const BBox &a, const BBox &b);
  double l2norm_;

public:
  SeedPoints(GuiContext* ctx);
  virtual ~SeedPoints();

  virtual void execute();
  virtual void widget_moved(bool, BaseWidget*);

  vector<GuiDouble*> seeds_;

  GuiInt gui_num_seeds_;
  GuiDouble gui_probe_scale_;
  GuiInt gui_send_;
  GuiInt gui_widget_;
  GuiDouble red_;
  GuiDouble green_;
  GuiDouble blue_;
  GuiInt    gui_auto_execute_;
};


DECLARE_MAKER(SeedPoints)

SeedPoints::SeedPoints(GuiContext* ctx)
  : Module("SeedPoints", ctx, Filter, "FieldsCreate", "SCIRun"),
    widget_lock_("SeedPoints widget lock"),
    gui_num_seeds_(ctx->subVar("num_seeds")),
    gui_probe_scale_(ctx->subVar("probe_scale")),
    gui_send_(ctx->subVar("send")),
    gui_widget_(ctx->subVar("widget")),
    red_(ctx->subVar("red")),
    green_(ctx->subVar("green")),
    blue_(ctx->subVar("blue")),
    gui_auto_execute_(ctx->subVar("auto_execute"))
{
}


SeedPoints::~SeedPoints()
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
SeedPoints::bbox_similar_to(const BBox &a, const BBox &b)
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
SeedPoints::execute()
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
    return;
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

  Point center;
  Point bmin = bbox.min();
  Point bmax = bbox.max();

  if (!bbox_similar_to(last_bounds_, bbox))
  {
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

    center = bmin + Vector(bmax - bmin) * 0.5;
    l2norm_ = (bmax - bmin).length();

    GeometryOPort *ogport = (GeometryOPort*)get_oport("SeedPoints Widget");
    if (!ogport)
    {
      error("Unable to initialize " + name + "'s oport.");
      return;
    }
    ogport->flushViews();

    last_bounds_ = bbox;
  }

  int numSeeds = gui_num_seeds_.get();
  
  if ((int)widget_id_.size() != numSeeds) 
  {

    GeometryOPort *ogport = (GeometryOPort*)get_oport("SeedPoints Widget");
    if (!ogport)
    {
      error("Unable to initialize " + name + "'s oport.");
      return;
    }

    if(numSeeds < (int)widget_id_.size()) 
    {

      for (int i = numSeeds; i < (int)widget_id_.size(); i++)
	{
	  ((GeomSwitch *)(widget_switch_[i].get_rep()))->set_state(0);
	  ogport->delObj(widget_id_[i]);
	  ogport->flushViews();

	  //delete in tcl side
	  ostringstream str;
	  str << i;
	  gui->execute(id.c_str() + string(" clear_seed " + str.str()));

	}
      widget_switch_.resize(numSeeds);
      widget_id_.resize(numSeeds);
      if (gui_widget_.get() == 0) 
      	widget_vec_.resize(numSeeds);
      else
      	widget_ring_.resize(numSeeds);


    } else {

      for (int i=widget_id_.size(); i <numSeeds; i++) 
      {
	if (gui_widget_.get() == 0) {
	  PointWidget *seed = scinew PointWidget(this, &widget_lock_, 1.0);
	  MaterialHandle redMatl = new Material(Color(red_.get(), green_.get(), blue_.get()));
	  seed->SetMaterial(0,redMatl);
	  seed->Connect(ogport);
	  widget_.push_back(seed);
	  
	  widget_switch_.push_back(widget_[i]->GetWidget());
	  ((GeomSwitch *)(widget_switch_[i].get_rep()))->set_state(1);
	  
	  widget_id_.push_back(ogport->addObj(widget_switch_[i], "SeedPoint" + to_string((int)i),
					      &widget_lock_));	

	  // input saved out positions
	  ostringstream strX, strY, strZ;
	  
	  strX << "seedX" << i;
	  GuiDouble* gui_locx = new GuiDouble(ctx->subVar(strX.str()));
	  seeds_.push_back(gui_locx);
	  
	  strY << "seedY" << i;
	  GuiDouble* gui_locy = new GuiDouble(ctx->subVar(strY.str()));
	  seeds_.push_back(gui_locy);
	  
	  strZ << "seedZ" << i;
	  GuiDouble* gui_locz = new GuiDouble(ctx->subVar(strZ.str()));
	  seeds_.push_back(gui_locz);
	  
	  Point curloc(gui_locx->get(),gui_locy->get(),gui_locz->get());
	  
	  if (curloc.x() >= bmin.x() && curloc.x() <= bmax.x() && 
	      curloc.y() >= bmin.y() && curloc.y() <= bmax.y() && 
	      curloc.z() >= bmin.z() && curloc.z() <= bmax.z() ||
	      !input_field_p)
	    {
	      center = curloc;
	    } else {
	      gui->execute(id.c_str() + string(" make_seed " + to_string((int)i)));
	    }

	  gui->execute(id.c_str() + string(" set_seed " + to_string((int)i) + " " + to_string((double)center.x()) + " " + to_string((double)center.y()) + " " + to_string((double)center.z())));
	  
	  seed->SetPosition(center);
	  seed->SetScale(gui_probe_scale_.get() * l2norm_ * 0.003);
	} else {
	  RingWidget *seed = scinew RingWidget(this, &widget_lock_, gui_probe_scale_.get(), false);

	  
	  MaterialHandle redMatl = new Material(Color(red_.get(), green_.get(), blue_.get()));
	  redMatl->specular.r(0.2);
	  redMatl->specular.g(0.2);
	  redMatl->specular.b(0.2);

	  MaterialHandle greyMatl = new Material(Color(0.5, 0.5, 0.5));
	  seed->SetMaterial(1,redMatl);
	  seed->SetDefaultMaterial(5,greyMatl);
	  seed->Connect(ogport);
	  widget_ring_.push_back(seed);
	  
	  widget_switch_.push_back(widget_ring_[i]->GetWidget());
	  ((GeomSwitch *)(widget_switch_[i].get_rep()))->set_state(1);
	  
	  widget_id_.push_back(ogport->addObj(widget_switch_[i], "SeedPoint" + to_string((int)i),
					      &widget_lock_));	

	  // input saved out positions
	  ostringstream strX, strY, strZ;
	  
	  strX << "seedX" << i;
	  GuiDouble* gui_locx = new GuiDouble(ctx->subVar(strX.str()));
	  seeds_.push_back(gui_locx);
	  
	  strY << "seedY" << i;
	  GuiDouble* gui_locy = new GuiDouble(ctx->subVar(strY.str()));
	  seeds_.push_back(gui_locy);
	  
	  strZ << "seedZ" << i;
	  GuiDouble* gui_locz = new GuiDouble(ctx->subVar(strZ.str()));
	  seeds_.push_back(gui_locz);
	  
	  Point curloc(gui_locx->get(),gui_locy->get(),gui_locz->get());
	  
	  if (curloc.x() >= bmin.x() && curloc.x() <= bmax.x() && 
	      curloc.y() >= bmin.y() && curloc.y() <= bmax.y() && 
	      curloc.z() >= bmin.z() && curloc.z() <= bmax.z() ||
	      !input_field_p)
	    {
	      center = curloc;
	    } else {
	      gui->execute(id.c_str() + string(" make_seed " + to_string((int)i)));
	    }

	  //	  gui->execute(id.c_str() + string(" set_seed " + to_string((int)i) + " " + to_string((double)center.x()) + " " + to_string((double)center.y()) + " " + to_string((double)center.z())));
	  gui->execute(id.c_str() + string(" set_seed " + to_string((int)i) + " " + to_string(135) + " " + to_string(293) + " " + to_string(0.1)));
	  
	  double r = gui_probe_scale_.get();
	  Vector normal(0.0, 0.0, 1.0);
	  center.x(0.5);
	  center.y(0.5);
	  center.z(0.1);
	  seed->SetPosition(center, normal, r);
	  seed->SetScale(gui_probe_scale_.get() * l2norm_ * 0.003);
	  seed->SetRadius(r);
	  seed->SetCurrentMode(3);
	}
      }
    }
  }

  // Find magnitude and base ring scale on that
  const BBox ibox = ifieldhandle->mesh()->get_bounding_box();
  Vector mag = ibox.max() - ibox.min();
  double max = 0.0;
  if (mag.x() > max) max = mag.x();
  if (mag.y() > max) max = mag.y();
  if (mag.z() > max) max = mag.z();
  


  for (int i=0; i <numSeeds; i++) 
  {
    if (gui_widget_.get() == 0) {
      widget_[i]->SetScale(gui_probe_scale_.get() * l2norm_ * 0.003);
      const Point location = widget_[i]->GetPosition();

      gui->execute(id.c_str() + string(" set_seed " + to_string((int)i) + " " + to_string((double)location.x()) + " " + to_string((double)location.y()) + " " + to_string((double)location.z())));

    } else {
      Point location;
      double r;
      Vector normal;
      widget_ring_[i]->GetPosition(center, normal, r);
      widget_ring_[i]->SetScale(max*0.02);

      // place rings at slight distance as seeds are added
      //      gui->execute(id.c_str() + string(" set_seed " + to_string((int)i) + " " + to_string((double)center.x()) + " " + to_string((double)center.y()) + " " + to_string((double)center.z())));
      gui->execute(id.c_str() + string(" set_seed " + to_string((int)i) + " " + to_string(135) + " " + to_string(293) + " " + to_string(0.1)));
      
    }
  }


  //when push send button
  if(gui_send_.get()) 
  {
    ostringstream strX, strY, strZ;
    
    PointCloudMesh *mesh = scinew PointCloudMesh();
    PointCloudField<double> *field = scinew PointCloudField<double>(mesh, 1);
    
    for (int i=0; i <numSeeds; i++)
    {
      if (gui_widget_.get() == 0) {
	const Point location = widget_[i]->GetPosition();
	
	PointCloudMesh::Node::index_type pcindex = mesh->add_point(location);
	field->resize_fdata();
	field->set_value(i, pcindex);
      } else {
	Point location;
	double r;
	Vector normal;
	widget_ring_[i]->GetPosition(location, normal, r);
	
	PointCloudMesh::Node::index_type pcindex = mesh->add_point(location);
	field->resize_fdata();
	field->set_value(r, pcindex); // FIX ME: get radius
      }
    }

    FieldHandle ofield = field;
    
    FieldOPort *ofp = (FieldOPort *)get_oport("SeedPoints Point");
    if (!ofp) 
    {
      error("Unable to initialize oport 'SeedPoints Point'.");
      return;
    }
    ofp->send(ofield);
    gui_send_.set(0);
  }
}


void
SeedPoints::widget_moved(bool last, BaseWidget*)
{
  if (last)
  {
    if (gui_auto_execute_.get() == 1) {
      want_to_execute();
    }
  }
}




} // End namespace SCIRun


