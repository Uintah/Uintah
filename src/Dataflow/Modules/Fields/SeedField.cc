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
 *  SeedField.cc:  From a mesh, seed some number of dipoles
 *
 *  Written by:
 *   David Weinstein
 *   University of Utah
 *   October 2000
 *
 *  Copyright (C) 2000 SCI Group
 */
 
#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MusilRNG.h>
#include <Core/Math/Trig.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Thread/CrowdMonitor.h>
#include <Dataflow/Widgets/GaugeWidget.h>
#include <Core/Datatypes/PointCloud.h>
#include <math.h>

#include <iostream>
using std::cerr;

namespace SCIRun {

class SeedField : public Module {
  FieldIPort     *ifport_;
  FieldOPort     *ofport_;  
  GeometryOPort  *ogport_;

  FieldHandle    vfhandle_;
  Field          *vf_;

  char           firsttime_;

  GuiString seedRandTCL_;
  GuiString numDipolesTCL_;
  GuiString dipoleMagnitudeTCL_;
public:
  CrowdMonitor widget_lock_;
  GaugeWidget rake_;
  SeedField(const clString& id);
  virtual ~SeedField();
  virtual void execute();
  virtual void tcl_command(TCLArgs&, void*);
};

extern "C" Module* make_SeedField(const clString& id) {
  return new SeedField(id);
}

SeedField::SeedField(const clString& id)
  : Module("SeedField", id, Filter, "Fields", "SCIRun"),
    seedRandTCL_("seedRandTCL", id, this),
    numDipolesTCL_("numDipolesTCL", id, this),
    dipoleMagnitudeTCL_("dipoleMagnitudeTCL", id, this),
    widget_lock_("StreamLines widget lock"),
    rake_(this,&widget_lock_,1)
{
  // Create the input port
  ifport_ = scinew FieldIPort(this, "Field to seed", FieldIPort::Atomic);
  add_iport(ifport_);
  
  // Create the output ports
  ofport_ = scinew FieldOPort(this,"Seeds", FieldIPort::Atomic);
  add_oport(ofport_);

  ogport_ = scinew GeometryOPort(this,"Seeding Widget", GeometryIPort::Atomic);
  add_oport(ogport_);

  vf_ = 0;

  firsttime_ = 1;
}

SeedField::~SeedField()
{
}

// FIX_ME upgrade to new fields.
void SeedField::execute()
{
  Point min,max;
  BBox bbox;
  

  // the field input is required
  if (!ifport_->get(vfhandle_) || !(vf_ = vfhandle_.get_rep()))
    return;

  bbox = vf_->mesh()->get_bounding_box();
  min = bbox.min();
  max = bbox.max();
  

  if (firsttime_) {
    firsttime_=0;
    Point center(min.x()+(max.x()-min.x())/2.,
		 min.y()+(max.y()-min.y())/2.,
		 min.z()+(max.z()-min.z())/2.);
    
    double x  = max.x()-min.x();
    double x2 = x*x;
    double y  = max.y()-min.y();
    double y2 = y*y;
    double z  = max.z()-min.z();
    double z2 = z*z;
  
    double quarterl2norm = sqrt(x2+y2+z2)/4.;
    
    rake_.SetScale(quarterl2norm*.06); // this size seems empirically good
    
    rake_.SetEndpoints(Point(center.x()-quarterl2norm,center.y(),center.z()),
		       Point(center.x()+quarterl2norm,center.y(),center.z()));
  }

  GeomGroup *widget_group = scinew GeomGroup;
  widget_group->add(rake_.GetWidget());
  
  rake_.GetEndpoints(min,max);
  
  Vector dir(max-min);
  int num_seeds = (int)(rake_.GetRatio()*15);
  cerr << "num_seeds = " << num_seeds << endl;
  dir*=1./(num_seeds-1);

  PointCloudMeshHandle mesh = scinew PointCloudMesh;
  int loop;
  for (loop=0;loop<num_seeds;++loop) {
    mesh->add_node(min+dir*loop);
  }

  PointCloud<double> *seeds = scinew PointCloud<double>(mesh, Field::NODE);
  PointCloud<double>::fdata_type &fdata = seeds->fdata();

  for (loop=0;loop<num_seeds;++loop) {
    fdata[loop]=1;
  }
  
  ofport_->send(seeds);
  ogport_->addObj(widget_group,"StreamLines rake",&widget_lock_);
}

void SeedField::tcl_command(TCLArgs& args, void* userdata)
{
  if(args.count() < 2){
    args.error("StreamLines needs a minor command");
    return;
  }
 
  if (args[1] == "execute") {
    want_to_execute();
  } else {
    Module::tcl_command(args, userdata);
  }
}

} // End namespace SCIRun

