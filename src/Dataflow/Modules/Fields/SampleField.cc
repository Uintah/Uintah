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
  Field          *vf_;

  bool           firsttime_;
  int            widgetid_;
  Point          endpoint0_,endpoint1_;
  double         widgetscale_;

  GuiInt maxSeeds_;
  GuiInt numSeeds_;
  GuiInt rngSeed_;
  GuiString widgetType_;
  GuiString randDist_;
  GuiString whichTab_;

  int vf_generation_;

  void generate_widget_seeds(Field *field);

public:
  CrowdMonitor widget_lock_;
  GaugeWidget *rake_;
  SampleField(const string& id);
  virtual ~SampleField();
  virtual void execute();
  virtual void tcl_command(TCLArgs&, void*);
  virtual void widget_moved(int);
};


extern "C" Module* make_SampleField(const string& id) {
  return new SampleField(id);
}


SampleField::SampleField(const string& id)
  : Module("SampleField", id, Filter, "Fields", "SCIRun"),
    maxSeeds_("maxseeds", id, this),
    numSeeds_("numseeds", id, this),
    rngSeed_("rngseed", id, this),
    widgetType_("type", id, this),
    randDist_("dist", id, this),
    whichTab_("whichtab", id, this),
    vf_generation_(0),
    widget_lock_("StreamLines widget lock")
{
  vf_ = 0;
  widgetid_=0;;
  rake_ = 0;

  firsttime_ = true;
}


SampleField::~SampleField()
{
}


void
SampleField::widget_moved(int i)
{
  if (rake_) 
    rake_->GetEndpoints(endpoint0_,endpoint1_);

  if (i==1) {
    want_to_execute();
  } else {
    Module::widget_moved(i);
  }
}



void
SampleField::generate_widget_seeds(Field *field)
{
  const BBox bbox = field->mesh()->get_bounding_box();
  Point min = bbox.min();
  Point max = bbox.max();
  double quarterl2norm;

  if (firsttime_) {
    firsttime_ = false;
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
    widgetscale_ = quarterl2norm*.06;// this size seems empirically good

    endpoint0_ = Point(center.x()-quarterl2norm,
		       center.y()-quarterl2norm/3,
		       center.z()-quarterl2norm/4);
    endpoint1_ = Point(center.x()+quarterl2norm,
		       center.y()+quarterl2norm/2,
		       center.z()+quarterl2norm/3);
  }

  if (!rake_)
  {
    rake_ = scinew GaugeWidget(this,&widget_lock_,1);
    rake_->SetScale(widgetscale_);
    
    rake_->SetEndpoints(endpoint0_,endpoint1_);
  }

  GeomGroup *widget_group = scinew GeomGroup;
  widget_group->add(rake_->GetWidget());
  
  rake_->GetEndpoints(min,max);
  
  int max_seeds = maxSeeds_.get();

  Vector dir(max-min);
  int num_seeds = (int)(rake_->GetRatio()*max_seeds);
  remark("num_seeds = " + to_string(num_seeds));
  dir*=1./(num_seeds-1);

  PointCloudMesh* mesh = scinew PointCloudMesh;
  int loop;
  for (int loop=0; loop<num_seeds; ++loop)
  {
    mesh->add_node(min+dir*loop);
  }

  PointCloud<double> *seeds = scinew PointCloud<double>(mesh, Field::NODE);
  PointCloud<double>::fdata_type &fdata = seeds->fdata();
  
  for (loop=0;loop<num_seeds;++loop)
  {
    fdata[loop]=1;
  }
  
  ofport_->send(seeds);
  widgetid_ = ogport_->addObj(widget_group,"StreamLines rake",&widget_lock_);
  ogport_->flushViews();
}

void
SampleField::execute()
{
  ifport_ = (FieldIPort *)get_iport("Field to Sample");
  ofport_ = (FieldOPort *)get_oport("Samples");
  ogport_ = (GeometryOPort *)get_oport("Sampling Widget");

  if (!ifport_) {
    postMessage("Unable to initialize "+name+"'s iport\n");
    return;
  }
  if (!ofport_)  {
    postMessage("Unable to initialize "+name+"'s oport\n");
    return;
  }
  if (!ogport_) {
    postMessage("Unable to initialize "+name+"'s oport\n");
    return;
  }
  // The field input is required.
  if (!ifport_->get(vfhandle_) || !(vf_ = vfhandle_.get_rep()))
  {
    return;
  }

  const string &tab = whichTab_.get();
  if (tab == "Random")
  {
    const TypeDescription *mtd = vfhandle_->mesh()->get_type_description();
    CompileInfo *ci = SampleFieldAlgo::get_compile_info(mtd);
    DynamicAlgoHandle algo_handle;
    if (! DynamicLoader::scirun_loader().get(*ci, algo_handle))
    {
      cout << "Could not compile algorithm." << std::endl;
      return;
    }
    SampleFieldAlgo *algo =
      dynamic_cast<SampleFieldAlgo *>(algo_handle.get_rep());
    if (algo == 0)
    {
      cout << "Could not get algorithm." << std::endl;
      return;
    }
    FieldHandle seedhandle(algo->execute(vfhandle_, numSeeds_.get(),
					 rngSeed_.get(), randDist_.get()));
    
    ofport_->send(seedhandle);
    if (widgetid_) { ogport_->delObj(widgetid_); }
    widgetid_=0;
    rake_ = 0;
    ogport_->flushViews();
  }
  else if (tab=="Widget")
  {
    generate_widget_seeds(vf_);
  }
}


void
SampleField::tcl_command(TCLArgs& args, void* userdata)
{
  if(args.count() < 2)
  {
    args.error("StreamLines needs a minor command");
    return;
  }
 
  if (args[1] == "execute")
  {
    want_to_execute();
  }
  else
  {
    Module::tcl_command(args, userdata);
  }
}


CompileInfo *
SampleFieldAlgo::get_compile_info(const TypeDescription *mesh_td)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("SampleFieldAlgoT");
  static const string base_class_name("SampleFieldAlgo");

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

