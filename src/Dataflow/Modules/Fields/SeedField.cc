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

  GuiString seedRandTCL_;
  GuiString numDipolesTCL_;
  GuiString dipoleMagnitudeTCL_;
public:
  CrowdMonitor widget_lock_;
  GaugeWidget rake_;
  SeedField(const clString& id);
  virtual ~SeedField();
  virtual void execute();
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
}

SeedField::~SeedField()
{
}

// FIX_ME upgrade to new fields.
void SeedField::execute()
{
  // the field input is required
  if (!ifport_->get(vfhandle_) || !(vf_ = vfhandle_.get_rep()))
    return;

  BBox bbox = vf_->mesh()->get_bounding_box();
  Point min = bbox.min();
  Point max = bbox.max();

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

  rake_.SetEndpoints(Point(center.x()-quarterl2norm,center.y(),center.z()),
		     Point(center.x()+quarterl2norm,center.y(),center.z()));

  rake_.SetScale(quarterl2norm);

  GeomGroup *widget_group = scinew GeomGroup;
  widget_group->add(rake_.GetWidget());

  rake_.GetEndpoints(min,max);

  

  ogport_->addObj(widget_group,"StreamLines rake",&widget_lock_);
}

} // End namespace SCIRun






#if 0
  FieldHandle mesh;
  if (!imesh->get(mesh) || !mesh.get_rep()) return;
  int seedRand, numDipoles;
  double dipoleMagnitude;

  seedRandTCL.get().get_int(seedRand);
  numDipolesTCL.get().get_int(numDipoles);
  dipoleMagnitudeTCL.get().get_double(dipoleMagnitude);
  seedRandTCL.set(to_string(seedRand+1));
  cerr << "seedRand="<<seedRand<<"\n";
  MusilRNG mr(seedRand);
  mr();
//  cerr << "rand="<<mr()<<"\n";
  DenseMatrix *m=scinew DenseMatrix(numDipoles, 6);
  for (int i=0; i<numDipoles; i++) {
    int elem = mr() * mesh->elems.size();
    cerr << "elem["<<i<<"]="<<elem<<"\n";
    Point p(mesh->elems[elem]->centroid());
    (*m)[i][0]=p.x();
    (*m)[i][1]=p.y();
    (*m)[i][2]=p.z();
    (*m)[i][3]=2*(mr()-0.5)*dipoleMagnitude;
    (*m)[i][4]=2*(mr()-0.5)*dipoleMagnitude;
    (*m)[i][5]=2*(mr()-0.5)*dipoleMagnitude;
  }
  omat->send(MatrixHandle(m));
#endif
