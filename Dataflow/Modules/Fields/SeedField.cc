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
#include <Core/Datatypes/TetVol.h>
#include <math.h>
#include <set>

#include <iostream>

using std::set;

namespace SCIRun {

class SeedField : public Module
{
  FieldIPort     *ifport_;
  FieldOPort     *ofport_;  
  GeometryOPort  *ogport_;

  FieldHandle    vfhandle_;
  Field          *vf_;

  char           firsttime_;

  GuiInt random_seed_GUI_;
  GuiInt number_dipoles_GUI_;

  int random_seed_;
  int number_dipoles_;
  int vf_generation_;

  void execute_rake();
  void execute_gui();

  template <class M> void dispatch(M *mesh);

public:
  CrowdMonitor widget_lock_;
  GaugeWidget rake_;
  SeedField(const string& id);
  virtual ~SeedField();
  virtual void execute();
  virtual void tcl_command(TCLArgs&, void*);
  virtual void widget_moved(int);
};


extern "C" Module* make_SeedField(const string& id) {
  return new SeedField(id);
}


SeedField::SeedField(const string& id)
  : Module("SeedField", id, Filter, "Fields", "SCIRun"),
    random_seed_GUI_("random_seed", id, this),
    number_dipoles_GUI_("number_dipoles", id, this),
    random_seed_(0),
    number_dipoles_(0),
    vf_generation_(0),
    //dipoleMagnitudeTCL_("dipoleMagnitudeTCL", id, this),
    widget_lock_("StreamLines widget lock"),
    rake_(this,&widget_lock_,1)
{
  // Create the input port
  ifport_ = scinew FieldIPort(this, "Field to Seed", FieldIPort::Atomic);
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


void
SeedField::widget_moved(int i)
{
  if (i==1) 
  {
    want_to_execute();
  }
}


template <class M>
void
SeedField::dispatch(M *mesh)
{
  // Get size of mesh.
  unsigned int mesh_size = 0;
  typename M::cell_iterator itr = mesh->cell_begin();
  while (itr != mesh->cell_end())
  {
    mesh_size++;
    ++itr;
  }

  ASSERT((unsigned int)number_dipoles_ < mesh_size);

  set<unsigned int, less<unsigned int> > picks;

  // Pick a bunch of unique points.
  int i;
  srand(random_seed_);
  for (i=0; i < number_dipoles_; i++)
  {
    while (!(picks.insert(rand()%mesh_size).second));
  }

  PointCloudMesh *cloud_mesh = scinew PointCloudMesh;
  PointCloud<double> *cloud =
    scinew PointCloud<double>(cloud_mesh, Field::NODE);

  unsigned int counter = 0;
  itr = mesh->cell_begin();
  set<unsigned int, less<unsigned int> >::iterator pitr;
  for (pitr = picks.begin(); pitr != picks.end(); pitr++)
  {
    while (counter < *pitr)
    {
      ++counter;
      ++itr;
    }

    Point p;
    mesh->get_center(p, *itr);
    
    cloud_mesh->add_node(p);
  }

  PointCloud<double>::fdata_type &fdata = cloud->fdata();
  for (unsigned int i = 0; i < fdata.size(); i++)
  {
    fdata[i] = 1.0;
  }

  ofport_->send(cloud);
}
      
  

void
SeedField::execute_gui()
{
  // get gui variables;
  const int rand_seed = random_seed_GUI_.get();
  const int num_dipoles = number_dipoles_GUI_.get();

  //cout << "random_seed_ = " << random_seed_ << " " << rand_seed << endl;
  //cout << "number_dipoles = " << number_dipoles_ << " " << num_dipoles << endl;

  if (vf_generation_ != vfhandle_->generation ||
      random_seed_ != rand_seed ||
      number_dipoles_ != num_dipoles)
  {
    vf_generation_ = vfhandle_->generation;
    random_seed_ = rand_seed;
    number_dipoles_ = num_dipoles;

    const string geom_name = vf_->get_type_name(0);
    MeshBase *mesh = vf_->mesh().get_rep();
    if (geom_name == "TetVol")
    {
      dispatch((TetVolMesh *)mesh);
    }
    else if (geom_name == "LatticeVol")
    {
      dispatch((LatVolMesh *)mesh);
    }
    else
    {
      cout << "Unsupported input field type, no cells\n";
      return;
    }
  }
}

void
SeedField::execute_rake()
{
  const BBox bbox = vf_->mesh()->get_bounding_box();
  Point min = bbox.min();
  Point max = bbox.max();

  if (firsttime_)
  {
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
    
    rake_.SetEndpoints(Point(center.x()-quarterl2norm,
			     center.y()-quarterl2norm/3,
			     center.z()-quarterl2norm/4),
		       Point(center.x()+quarterl2norm,
			     center.y()+quarterl2norm/2,
			     center.z()+quarterl2norm/3));
  }

  GeomGroup *widget_group = scinew GeomGroup;
  widget_group->add(rake_.GetWidget());
  
  rake_.GetEndpoints(min,max);
  
  Vector dir(max-min);
  int num_seeds = (int)(rake_.GetRatio()*15);
  std::cerr << "num_seeds = " << num_seeds << endl;
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
  ogport_->addObj(widget_group,"StreamLines rake",&widget_lock_);
}


void
SeedField::execute()
{
  // The field input is required.
  if (!ifport_->get(vfhandle_) || !(vf_ = vfhandle_.get_rep()))
  {
    return;
  }

  if (ogport_->nconnections() > 0)
  {
    execute_rake();
  }
  else
  {
    execute_gui();
  }
}


void
SeedField::tcl_command(TCLArgs& args, void* userdata)
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

} // End namespace SCIRun

