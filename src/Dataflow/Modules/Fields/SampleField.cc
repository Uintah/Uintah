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
 *  SampleField.cc:  From a mesh, seed some number of dipoles
 *
 *  Written by:
 *   David Weinstein
 *   University of Utah
 *   October 2000
 *
 *  Copyright (C) 2000 SCI Group
 */
 
#include <Dataflow/Modules/Fields/SampleField.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/Trig.h>
#include <Core/Geometry/Transform.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Thread/CrowdMonitor.h>
#include <Core/Containers/StringUtil.h>
#include <Dataflow/Widgets/GaugeWidget.h>
#include <Dataflow/Widgets/RingWidget.h>
#include <Dataflow/Widgets/FrameWidget.h>
#include <math.h>
#include <set>

#include <iostream>

using std::set;
using std::vector;
using std::pair;

namespace SCIRun {

class SampleField : public Module
{
  FieldIPort     *ifport_;
  FieldOPort     *ofport_;  
  GeometryOPort  *ogport_;

  BBox           ifield_bbox_;
  BBox           ring_bbox_;
  BBox           frame_bbox_;
  int            widgetid_;
  int            wtype_;   // 0 none, 1 rake, 2 ring, 3 frame
  Point          endpoint0_;
  Point          endpoint1_;

  GuiString gui_wtype_;
  GuiInt    endpoints_;
  GuiDouble endpoint0x_;
  GuiDouble endpoint0y_;
  GuiDouble endpoint0z_;
  GuiDouble endpoint1x_;
  GuiDouble endpoint1y_;
  GuiDouble endpoint1z_;
  GuiDouble widgetscale_;
  GuiString ringstate_;
  GuiString framestate_;

  GuiDouble maxSeeds_;
  GuiInt numSeeds_;
  GuiInt rngSeed_;
  GuiInt rngInc_;
  GuiInt clamp_;
  GuiInt autoexec_;
  GuiString randDistribution_;
  GuiString whichTab_;
  GuiInt force_rake_reset_;

  CrowdMonitor widget_lock_;
  GaugeWidget *rake_;
  RingWidget *ring_;
  FrameWidget *frame_;

  void execute_rake(FieldHandle ifield);
  void execute_ring(FieldHandle ifield);
  void execute_frame(FieldHandle ifield);
  void execute_random(FieldHandle ifield);
  bool bbox_similar_to(const BBox &a, const BBox &b);
public:
  SampleField(GuiContext* ctx);
  virtual ~SampleField();
  virtual void execute();
  virtual void widget_moved(bool last, BaseWidget*);
};


DECLARE_MAKER(SampleField)


SampleField::SampleField(GuiContext* ctx)
  : Module("SampleField", ctx, Filter, "FieldsCreate", "SCIRun"),
    widgetid_(0),
    wtype_(0),
    gui_wtype_(ctx->subVar("wtype")),
    endpoints_ (ctx->subVar("endpoints")),
    endpoint0x_(ctx->subVar("endpoint0x")),
    endpoint0y_(ctx->subVar("endpoint0y")),
    endpoint0z_(ctx->subVar("endpoint0z")),
    endpoint1x_(ctx->subVar("endpoint1x")),
    endpoint1y_(ctx->subVar("endpoint1y")),
    endpoint1z_(ctx->subVar("endpoint1z")),
    widgetscale_(ctx->subVar("widgetscale")),
    ringstate_(ctx->subVar("ringstate")),
    framestate_(ctx->subVar("framestate")),
    maxSeeds_(ctx->subVar("maxseeds")),
    numSeeds_(ctx->subVar("numseeds")),
    rngSeed_(ctx->subVar("rngseed")),
    rngInc_(ctx->subVar("rnginc")),
    clamp_(ctx->subVar("clamp")),
    autoexec_(ctx->subVar("autoexecute")),
    randDistribution_(ctx->subVar("dist")),
    whichTab_(ctx->subVar("whichtab")),
    force_rake_reset_(ctx->subVar("force-rake-reset", false)),
    widget_lock_("SampleField widget lock"),
    rake_(0),
    ring_(0),
    frame_(0)
{
  endpoints_.set( 0 );
}


SampleField::~SampleField()
{
  if (rake_) delete rake_;
  if (ring_) delete ring_;
  if (frame_) delete frame_;
}


void
SampleField::widget_moved(bool last, BaseWidget*)
{
  if (last)
  {
    if (rake_)
    {
      rake_->GetEndpoints(endpoint0_, endpoint1_);
      double ratio = rake_->GetRatio();
      if (ratio < 0.0001) ratio = 0.0001; // To avoid infinte loop
      double num_seeds = Max(0.0, 1.0/ratio+1.0);
      maxSeeds_.set(num_seeds);

      endpoint0x_.set( endpoint0_.x() );
      endpoint0y_.set( endpoint0_.y() );
      endpoint0z_.set( endpoint0_.z() );
      endpoint1x_.set( endpoint1_.x() );
      endpoint1y_.set( endpoint1_.y() );
      endpoint1z_.set( endpoint1_.z() );
    }

    autoexec_.reset();
    if (autoexec_.get())
    {
      want_to_execute();
    }
  } else { // rescaling the widget forces a "last=false" widget_moved event
    if (rake_)
      {
	widgetscale_.set(rake_->GetScale());
      }
  }
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
SampleField::bbox_similar_to(const BBox &a, const BBox &b)
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
SampleField::execute_rake(FieldHandle ifield)
{
  const BBox ibox = ifield->mesh()->get_bounding_box();
  if (!bbox_similar_to(ifield_bbox_, ibox) || force_rake_reset_.get())
  {
    if (!endpoints_.get() || ifield_bbox_.valid() || force_rake_reset_.get())
    {
      const Point &min = ibox.min();
      const Point &max = ibox.max();

      const Point center(((max.asVector() + min.asVector()) * 0.5).asPoint());
      const double dx  = max.x() - min.x();
      const double dy  = max.y() - min.y();
      const double dz  = max.z() - min.z();
  
      // This size seems empirically good.
      const double quarterl2norm = sqrt(dx * dx + dy * dy + dz * dz) / 4.0;
      widgetscale_.set( quarterl2norm * .06 );

      endpoint0x_.set( center.x() - quarterl2norm     );
      endpoint0y_.set( center.y() - quarterl2norm / 3 );
      endpoint0z_.set( center.z() - quarterl2norm / 4 );
      endpoint1x_.set( center.x() + quarterl2norm     );
      endpoint1y_.set( center.y() + quarterl2norm / 2 );
      endpoint1z_.set( center.z() + quarterl2norm / 3 );

      endpoints_.set( 1 );
    }

    endpoint0_ = Point(endpoint0x_.get(),endpoint0y_.get(),endpoint0z_.get()); 
    endpoint1_ = Point(endpoint1x_.get(),endpoint1y_.get(),endpoint1z_.get()); 

    if (rake_)
    {
      rake_->SetEndpoints(endpoint0_, endpoint1_);
      rake_->SetRatio(1/16.0);
      rake_->SetScale(widgetscale_.get());
      if (wtype_ == 1) { ogport_->flushViews(); }
    }

    ifield_bbox_ = ibox;
    force_rake_reset_.set(0);
  }

  if (!rake_)
  {
    rake_ = scinew GaugeWidget(this, &widget_lock_, widgetscale_.get(), true);
    rake_->Connect(ogport_);
    rake_->SetEndpoints(endpoint0_,endpoint1_);
    rake_->SetScale(widgetscale_.get());
    rake_->SetRatio(1/16.0);
  }

  if (wtype_ != 1)
  {
    if (widgetid_)  { ogport_->delObj(widgetid_); }
    GeomHandle widget = rake_->GetWidget();
    widgetid_ = ogport_->addObj(widget, "SampleField Rake", &widget_lock_);
    ogport_->flushViews();
    wtype_ = 1;
  }

  Point min, max;
  rake_->GetEndpoints(min, max);
  
  Vector dir(max-min);
  double num_seeds = Max(0.0, maxSeeds_.get());
  if (num_seeds <= 0)
  {
    remark("No seeds to send.");
    return;
  }
  remark("num_seeds = " + to_string(num_seeds));
  if (num_seeds > 1)
  {
    const double ratio = 1.0/(num_seeds-1.0);
    rake_->SetRatio(ratio);
    dir *= ratio;
  }

  PointCloudMesh* mesh = scinew PointCloudMesh;
  int loop;
  for (loop=0; loop<=(num_seeds-0.99999); ++loop)
  {
    mesh->add_node(min+dir*loop);
  }

  mesh->freeze();
  PointCloudField<double> *seeds =
    scinew PointCloudField<double>(mesh, 1);
  PointCloudField<double>::fdata_type &fdata = seeds->fdata();
  
  for (loop=0;loop<(num_seeds-0.99999);++loop)
  {
    fdata[loop]=loop;
  }
  seeds->freeze();
  ofport_->send(seeds);

  update_state(Completed);
}


void
SampleField::execute_ring(FieldHandle ifield)
{
  const BBox ibox = ifield->mesh()->get_bounding_box();
  bool reset = force_rake_reset_.get();
  force_rake_reset_.set(0);
  bool resize = ring_bbox_.valid() && !bbox_similar_to(ring_bbox_, ibox);
  if (!ring_)
  {
    ring_ = scinew RingWidget(this, &widget_lock_, widgetscale_.get(), false);
    ring_->Connect(ogport_);

    if (ringstate_.get() != "")
    {
      cout << "Setting state, reset = " << reset << "\n";
      ring_->SetStateString(ringstate_.get());

      // Check for state validity here.  If not valid then // reset = true;
    }
    else
    {
      reset = true;
    }
  }

  if (reset || resize)
  {
    Point c, nc;
    Vector n, nn;
    double r, nr;
    double s, ns;

    if (reset)
    {
      const Vector xaxis(0.0, 0.0, 0.2);
      const Vector yaxis(0.2, 0.0, 0.0);
      c = Point (0.5, 0.0, 0.0);
      n = Cross(xaxis, yaxis);
      r = 0.2;
      s = sqrt(3.0) * 0.03;

      ring_bbox_.reset();
      ring_bbox_.extend(Point(-1.0, -1.0, -1.0));
      ring_bbox_.extend(Point(1.0, 1.0, 1.0));
    }
    else
    {
      // Get the old coordinates.
      ring_->GetPosition(c, n, r);
      s = ring_->GetScale();
    }
    
    // Build a transform.
    Transform trans;
    trans.load_identity();
    const Vector scale =
      (ibox.max() - ibox.min()) / (ring_bbox_.max() - ring_bbox_.min());
    trans.pre_translate(-ring_bbox_.min().asVector());
    trans.pre_scale(scale);
    trans.pre_translate(ibox.min().asVector());
    
    // Do the transform.
    trans.project(c, nc);
    nn = n; //trans.project_normal(n, nn);
    nr = (r * scale).length() / sqrt(3.0);
    ns = (s * scale).length() / sqrt(3.0);

    // Apply the new coordinates.
    ring_->SetPosition(nc, nn, nr);
    ring_->SetRadius(nr);
    ring_->SetScale(ns);
    widgetscale_.set(ns);

    ring_bbox_ = ibox;
  }

  if (wtype_ != 2)
  {
    if (widgetid_)  { ogport_->delObj(widgetid_); }
    GeomHandle widget = ring_->GetWidget();
    widgetid_ = ogport_->addObj(widget, "SampleField Ring", &widget_lock_);
    ogport_->flushViews();
    wtype_ = 2;
  }
  
  double num_seeds = Max(0.0, maxSeeds_.get());
  if (num_seeds <= 0)
  {
    remark("No seeds to send.");
    return;
  }
  remark("num_seeds = " + to_string(num_seeds));

  PointCloudMesh* mesh = scinew PointCloudMesh;

  Point center;
  double r;
  Vector normal, xaxis, yaxis;
  ring_->GetPosition(center, normal, r);
  ring_->GetPlane(xaxis, yaxis);
  for (int i = 0; i < num_seeds; i++)
  {
    const double frac = 2.0 * M_PI * i / num_seeds;
    mesh->add_node(center + xaxis * r * cos(frac) + yaxis * r * sin(frac));
  }

  mesh->freeze();
  PointCloudField<double> *seeds =
    scinew PointCloudField<double>(mesh, 1);
  PointCloudField<double>::fdata_type &fdata = seeds->fdata();
  
  for (int loop=0; loop<num_seeds; ++loop)
  {
    fdata[loop]=loop;
  }
  seeds->freeze();
  ofport_->send(seeds);

  update_state(Completed);
}


void
SampleField::execute_frame(FieldHandle ifield)
{
  const BBox ibox = ifield->mesh()->get_bounding_box();
  bool reset = force_rake_reset_.get();
  force_rake_reset_.set(0);
  bool resize = frame_bbox_.valid() && !bbox_similar_to(frame_bbox_, ibox);
  if (!frame_)
  {
    frame_ = scinew FrameWidget(this, &widget_lock_, widgetscale_.get());
    frame_->Connect(ogport_);

    if (framestate_.get() != "")
    {
      frame_->SetStateString(framestate_.get());

      // Check for state validity here.  If not valid then // reset = true;
    }
    else
    {
      reset = true;
    }
  }

  if (reset || resize)
  {
    Point c, nc, r, nr, d, nd;
    double s, ns;

    if (reset)
    {
      c = Point(0.5, 0.0, 0.0);
      r = c + Vector(0.0, 0.0, 0.2);
      d = c + Vector(0.2, 0.0, 0.0);
      s = sqrt(3.0) * 0.03;

      frame_bbox_.reset();
      frame_bbox_.extend(Point(-1.0, -1.0, -1.0));
      frame_bbox_.extend(Point(1.0, 1.0, 1.0));
    }
    else
    {
      // Get the old coordinates.
      frame_->GetPosition(c, r, d);
      s = frame_->GetScale();
    }
    
    // Build a transform.
    Transform trans;
    trans.load_identity();
    const Vector scale =
      (ibox.max() - ibox.min()) / (frame_bbox_.max() - frame_bbox_.min());
    trans.pre_translate(-frame_bbox_.min().asVector());
    trans.pre_scale(scale);
    trans.pre_translate(ibox.min().asVector());
    
    // Do the transform.
    trans.project(c, nc);
    trans.project(r, nr);
    trans.project(d, nd);
    ns = (s * scale).length() / sqrt(3.0);

    // Apply the new coordinates.
    frame_->SetPosition(nc, nr, nd);
    frame_->SetScale(ns);
    widgetscale_.set(ns);

    frame_bbox_ = ibox;
  }

  if (wtype_ != 3)
  {
    if (widgetid_) { ogport_->delObj(widgetid_); }
    GeomHandle widget = frame_->GetWidget();
    widgetid_ = ogport_->addObj(widget, "SampleField Frame", &widget_lock_);
    ogport_->flushViews();
    wtype_ = 3;
  }

  double num_seeds = Max(0.0, maxSeeds_.get());
  if (num_seeds <= 0)
  {
    remark("No seeds to send.");
    return;
  }
  remark("num_seeds = " + to_string(num_seeds));

  PointCloudMesh* mesh = scinew PointCloudMesh;

  Point center, xloc, yloc;
  Point corner[4];
  Vector edge[4];
  frame_->GetPosition(center, xloc, yloc);
  const Vector xaxis = xloc - center;
  const Vector yaxis = yloc - center;
  corner[0] = center + xaxis + yaxis;
  corner[1] = center + xaxis - yaxis;
  corner[2] = center - xaxis - yaxis;
  corner[3] = center - xaxis + yaxis;
  edge[0] = corner[1] - corner[0];
  edge[1] = corner[2] - corner[1];
  edge[2] = corner[3] - corner[2];
  edge[3] = corner[0] - corner[3];
  for (int i = 0; i < num_seeds; i++)
  {
    const double frac =  4.0 * i / num_seeds;
    const int ei = (int)frac;
    const double eo = frac - ei;
    mesh->add_node(corner[ei] + edge[ei] * eo);
  }

  mesh->freeze();
  PointCloudField<double> *seeds =
    scinew PointCloudField<double>(mesh, 1);
  PointCloudField<double>::fdata_type &fdata = seeds->fdata();
  
  for (int loop=0; loop<num_seeds; ++loop)
  {
    fdata[loop]=loop;
  }
  seeds->freeze();
  ofport_->send(seeds);

  update_state(Completed);
}


void
SampleField::execute_random(FieldHandle ifield)
{
  const TypeDescription *mtd = ifield->mesh()->get_type_description();
  CompileInfoHandle ci = SampleFieldRandomAlgo::get_compile_info(mtd);
  Handle<SampleFieldRandomAlgo> algo;
  if (!module_dynamic_compile(ci, algo)) return;
  FieldHandle seedhandle(algo->execute(this, ifield, numSeeds_.get(),
				       rngSeed_.get(),
				       randDistribution_.get(), 
				       clamp_.get()));
  if (rngInc_.get())
  {
    rngSeed_.set(rngSeed_.get()+1);
  }

  if (widgetid_)
  {
    ogport_->delObj(widgetid_);
    ogport_->flushViews();
    widgetid_ = 0;
    wtype_ = 0;
  }

  ofport_->send(seedhandle);

  update_state(Completed);
}



void
SampleField::execute()
{
  ifport_ = (FieldIPort *)get_iport("Field to Sample");
  ofport_ = (FieldOPort *)get_oport("Samples");
  ogport_ = (GeometryOPort *)get_oport("Sampling Widget");

  if (!ifport_) {
    error("Unable to initialize iport 'Field to Sample'.");
    return;
  }
  if (!ofport_)  {
    error("Unable to initialize oport 'Samples'.");
    return;
  }
  if (!ogport_) {
    error("Unable to initialize oport 'Sampling Widget'.");
    return;
  }
  FieldHandle ifield;
  // The field input is required.
  if (!ifport_->get(ifield) || !ifield.get_rep())
  {
    error("Required input field is empty.");
    return;
  }

  if (ring_) { ringstate_.set(ring_->GetStateString()); }
  if (frame_) { framestate_.set(frame_->GetStateString()); }

  const string &tab = whichTab_.get();
  const string &wtype = gui_wtype_.get();
  if (tab == "Widget")
  {
    if (wtype == "rake")
    {
      execute_rake(ifield);
    }
    else if (wtype == "ring")
    {
      execute_ring(ifield);
    }
    else if (wtype == "frame")
    {
      execute_frame(ifield);
    }
  }
  else if (tab == "Random")
  {
    execute_random(ifield);
  }
}



CompileInfoHandle
SampleFieldRandomAlgo::get_compile_info(const TypeDescription *mesh_td)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("SampleFieldRandomAlgoT");
  static const string base_class_name("SampleFieldRandomAlgo");

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       mesh_td->get_filename() + ".",
                       base_class_name, 
                       template_class_name, 
                       mesh_td->get_name());

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  mesh_td->fill_compile_info(rval);
  return rval;
}


} // End namespace SCIRun

