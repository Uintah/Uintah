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
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Thread/CrowdMonitor.h>
#include <Core/Containers/StringUtil.h>
#include <Dataflow/Widgets/GaugeWidget.h>
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

  FieldHandle    vfhandle_;

  bool           firsttime_;
  int            widgetid_;
  Point          endpoint0_;
  Point          endpoint1_;

  GuiInt    endpoints_;
  GuiDouble endpoint0x_;
  GuiDouble endpoint0y_;
  GuiDouble endpoint0z_;
  GuiDouble endpoint1x_;
  GuiDouble endpoint1y_;
  GuiDouble endpoint1z_;
  GuiDouble widgetscale_;

  GuiInt maxSeeds_;
  GuiInt numSeeds_;
  GuiInt rngSeed_;
  GuiInt rngInc_;
  GuiInt clamp_;
  GuiInt autoexec_;
  GuiString widgetType_;
  GuiString randDist_;
  GuiString whichTab_;

  int vf_generation_;

  void execute_rake();
  void execute_random();

public:
  CrowdMonitor widget_lock_;
  GaugeWidget *rake_;
  SampleField(GuiContext* ctx);
  virtual ~SampleField();
  virtual void execute();
  virtual void widget_moved(bool last);
};


DECLARE_MAKER(SampleField)


SampleField::SampleField(GuiContext* ctx)
  : Module("SampleField", ctx, Filter, "FieldsCreate", "SCIRun"),
    
    firsttime_(true),
    widgetid_(0),
    endpoints_ (ctx->subVar("endpoints")),
    endpoint0x_(ctx->subVar("endpoint0x")),
    endpoint0y_(ctx->subVar("endpoint0y")),
    endpoint0z_(ctx->subVar("endpoint0z")),
    endpoint1x_(ctx->subVar("endpoint1x")),
    endpoint1y_(ctx->subVar("endpoint1y")),
    endpoint1z_(ctx->subVar("endpoint1z")),
    widgetscale_ (ctx->subVar("widgetscale")),

    maxSeeds_(ctx->subVar("maxseeds")),
    numSeeds_(ctx->subVar("numseeds")),
    rngSeed_(ctx->subVar("rngseed")),
    rngInc_(ctx->subVar("rnginc")),
    clamp_(ctx->subVar("clamp")),
    autoexec_(ctx->subVar("autoexecute")),
    widgetType_(ctx->subVar("type")),
    randDist_(ctx->subVar("dist")),
    whichTab_(ctx->subVar("whichtab")),
    vf_generation_(0),
    widget_lock_("StreamLines widget lock"),
    rake_(0)
{
  endpoints_.set( 0 );
}


SampleField::~SampleField()
{
  if (rake_) delete rake_;
}


void
SampleField::widget_moved(bool last)
{
  if (last)
  {
    if (rake_) {
      rake_->GetEndpoints(endpoint0_,endpoint1_);

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
  }
}



void
SampleField::execute_rake()
{
  const BBox bbox = vfhandle_->mesh()->get_bounding_box();
  Point min = bbox.min();
  Point max = bbox.max();
  double quarterl2norm;

  if (firsttime_)
  {
    firsttime_ = false;

    if(!endpoints_.get())
    {
      Point center(min.x()+(max.x()-min.x())/2.,
		   min.y()+(max.y()-min.y())/2.,
		   min.z()+(max.z()-min.z())/2.);

      double x  = max.x()-min.x();
      double x2 = x*x;
      double y  = max.y()-min.y();
      double y2 = y*y;
      double z  = max.z()-min.z();
      double z2 = z*z;
  
      quarterl2norm = sqrt(x2+y2+z2)/4.;
      widgetscale_.set( quarterl2norm*.06 );// this size seems empirically good

      endpoint0x_.set( center.x()-quarterl2norm   );
      endpoint0y_.set( center.y()-quarterl2norm/3 );
      endpoint0z_.set( center.z()-quarterl2norm/4 );
      endpoint1x_.set( center.x()+quarterl2norm   );
      endpoint1y_.set( center.y()+quarterl2norm/2 );
      endpoint1z_.set( center.z()+quarterl2norm/3 );

      endpoints_.set( 1 );
    }

    endpoint0_ = Point( endpoint0x_.get(),endpoint0y_.get(),endpoint0z_.get() ); 
    endpoint1_ = Point( endpoint1x_.get(),endpoint1y_.get(),endpoint1z_.get() ); 
  }

  if (!rake_)
  {
    rake_ = scinew GaugeWidget(this, &widget_lock_, widgetscale_.get(), false);
    rake_->Connect(ogport_);
    rake_->SetEndpoints(endpoint0_,endpoint1_);
    GeomGroup *widget_group = scinew GeomGroup;
    widget_group->add(rake_->GetWidget());
    widgetid_ = ogport_->addObj(widget_group,"StreamLines rake",&widget_lock_);
    ogport_->flushViews();
  }

  rake_->GetEndpoints(min, max);
  
  Vector dir(max-min);
  int num_seeds = Max(0, maxSeeds_.get());
  if (num_seeds <= 0)
  {
    remark("No seeds to send.");
    return;
  }
  remark("num_seeds = " + to_string(num_seeds));
  if (num_seeds > 1)
  {
    dir *= 1.0 / (num_seeds - 1.0);
  }

  PointCloudMesh* mesh = scinew PointCloudMesh;
  int loop;
  for (loop=0; loop<num_seeds; ++loop)
  {
    mesh->add_node(min+dir*loop);
  }

  mesh->freeze();
  PointCloudField<double> *seeds =
    scinew PointCloudField<double>(mesh, Field::NODE);
  PointCloudField<double>::fdata_type &fdata = seeds->fdata();
  
  for (loop=0;loop<num_seeds;++loop)
  {
    fdata[loop]=loop;
  }
  seeds->freeze();
  ofport_->send(seeds);

  update_state(Completed);
}




void
SampleField::execute_random()
{
  const TypeDescription *mtd = vfhandle_->mesh()->get_type_description();
  CompileInfoHandle ci = SampleFieldRandomAlgo::get_compile_info(mtd);
  Handle<SampleFieldRandomAlgo> algo;
  if (!module_dynamic_compile(ci, algo)) return;
  FieldHandle seedhandle(algo->execute(this, vfhandle_, numSeeds_.get(),
				       rngSeed_.get(), randDist_.get(), 
				       clamp_.get()));
  if (rngInc_.get())
  {
    rngSeed_.set(rngSeed_.get()+1);
  }

  if (widgetid_)
  {
    ogport_->delObj(widgetid_);
    ogport_->flushViews();
  }
  widgetid_ = 0;
  rake_ = 0;

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
  // The field input is required.
  if (!ifport_->get(vfhandle_) || !vfhandle_.get_rep())
  {
    error("Required input field is empty.");
    return;
  }

  const string &tab = whichTab_.get();
  if (tab == "Widget")
  {
    execute_rake();
  }
  else if (tab == "Random")
  {
    execute_random();
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

