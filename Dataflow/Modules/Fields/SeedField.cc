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
#include <Core/Math/MusilRNG.h>
#include <Dataflow/Widgets/GaugeWidget.h>
#include <Core/Datatypes/PointCloud.h>
#include <Core/Datatypes/TetVol.h>
#include <math.h>
#include <set>

#include <iostream>

using std::set;
using std::vector;

namespace SCIRun {

class SeedField : public Module
{
  FieldIPort     *ifport_;
  FieldOPort     *ofport_;  
  GeometryOPort  *ogport_;

  FieldHandle    vfhandle_;
  Field          *vf_;

  char           firsttime_;

  GuiInt maxSeeds_;
  GuiInt numSeeds_;
  GuiInt rngSeed_;
  GuiString widgetType_;
  GuiString randDist_;

  int vf_generation_;

  void execute_rake();
  void execute_gui();

  template <class Field> 
  bool build_weight_table(Field &, vector<double> &);
  template <class Field>
  PointCloud<double>* GenWidgetSeeds(Field &);
  template <class Field>
  PointCloud<double>* GenRandomSeeds(Field &);

  template <class M> 
  void dispatch(M *mesh);

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
    maxSeeds_("maxseeds", id, this),
    numSeeds_("numseeds", id, this),
    rngSeed_("rngseed", id, this),
    widgetType_("type", id, this),
    randDist_("dist", id, this),
    vf_generation_(0),
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

template <class Field>
bool 
SeedField::build_weight_table(Field &field, vector<double> &table)
{
  typedef typename Field::mesh_type         mesh_type;
  typedef typename Field::fdata_type        fdata_type;
  typedef typename mesh_type::elem_iterator elem_iterator;
  typedef typename mesh_type::elem_index    elem_index;

  string dist = randDist.get();

  mesh_type* mesh = field->mesh().get_rep();
  fdata_type fdata = field->fdata();

  elem_iterator ei = mesh->elem_begin();
  if (ei==mesh->elem_end()) // empty mesh
    return false;

  if (dist=="importance") { // size of element * data at element
    table.push_back(mesh->elems_size(ei)*fdata[*ei]);
    ++ei;
    while (ei != mesh.elem_end()) {
      table.push_back(mesh->elems_size(ei)*fdata[*ei]+table[table.size()-1]);
      ++ei;
    }
  } else if (dist=="uniform") { // size of element only
    table.push_back(mesh->elems_size(ei));
    ++ei;
    while (ei != mesh.elem_end()) {
      table.push_back(mesh->elems_size(ei)+table[table.size()-1]);
      ++ei;
    }
  } else if (dist=="scattered") { // element index; some strangely biased dist
    while (ei != mesh.elem_end()) {
      table.push_back(table.size());
      ++ei;
    }    
  } else { // unknown distribution type
    return false;
  } 

  return true;
}

template <class Field>
PointCloud<double>*
SeedField::GenWidgetSeeds(Field &field)
{
}

template <class Field>
PointCloud<double>*
SeedField::GenRandomSeeds(Field &field)
{
  typedef typename Field::mesh_type  mesh_type;
  typedef typename Field::elem_index elem_index;

  MusilRNG rng(rngSeed.get());
  vector<double> wt;

  rng();  // always discard first value

  if (!build_weight_table(field,wt))
    return 0;

  double max = wt[wt.size()-1];
  mesh_type *mesh = field.mesh().get_rep();
  PointCloud<double> *pc = scinew PointCloud<double>;

  unsigned int ns = numSeeds_.get();
  for (int loop=0;loop<ns;loop++) {
  }
}

template <class M>
void
SeedField::dispatch(M *mesh)
{
#if 0
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
#endif
}
      
  

void
SeedField::execute_gui()
{
#if 0
  // get gui variables;
  const int rand_seed = random_seed_GUI_.get();
  const int num_dipoles = number_dipoles_GUI_.get();

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
      error("Unsupported input field type, no cells.");
      return;
    }
  }
#endif
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
  
  int max_seeds = maxSeeds_.get();

  Vector dir(max-min);
  int num_seeds = (int)(rake_.GetRatio()*max_seeds);
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

