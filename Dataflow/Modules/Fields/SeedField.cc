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
#include <Core/Datatypes/LatticeVol.h>
#include <Core/Datatypes/TriSurf.h>
#include <math.h>
#include <set>

#include <iostream>

using std::set;
using std::vector;
using std::pair;

namespace SCIRun {

template <class Field>
class DistTable
{
public:
  typedef typename Field::mesh_type            mesh_type;
  typedef typename mesh_type::Elem::index_type elem_index;
  typedef pair<double,elem_index>              table_entry;

  vector<table_entry> table_;

  DistTable() {}
  ~DistTable() {}

  void push_back(double size,elem_index id) 
  { table_.push_back(table_entry(size,id)); }
  void push_back(table_entry entry) 
  { table_.push_back(entry); }

  const table_entry& operator[](unsigned idx) const
  { return table_[idx]; }
  table_entry& operator[](unsigned idx)
  { return table_[idx]; }

  double size() { return table_.size(); }

  bool search(table_entry&, double);
};

template <class Field>
bool
DistTable<Field>::search(table_entry &e, double d)
{
  int min=0,max=table_.size()-1;
  int cur = max/2;

  if ( (d<table_[0].first) || (d>table_[max].first) )
    return false; 

  // use binary search to find the bin holding the value d
  while ( (max-1>min) ) {
    if (table_[cur].first>=d) max = cur;
    if (table_[cur].first<d) min = cur;
    cur = (max-min)/2+min;
  }

  e = (table_[min].first>d)?table_[min]:table_[max];
  return true;
}

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
  GuiString whichTab_;

  int vf_generation_;

  template <class Field> 
  bool build_weight_table(Field *, DistTable<Field> &);
  template <class Field>
  PointCloud<double>* generate_widget_seeds(Field *);
  template <class Field>
  PointCloud<double>* generate_random_seeds(Field *);

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
    whichTab_("whichtab", id, this),
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
SeedField::build_weight_table(Field *field, DistTable<Field> &table)
{
  typedef typename Field::mesh_type            mesh_type;
  typedef typename Field::fdata_type           fdata_type;
  typedef typename mesh_type::Elem::iterator   elem_iterator;
  typedef typename mesh_type::Elem::index_type elem_index;

  string dist = randDist_.get();

  mesh_type* mesh = field->get_typed_mesh().get_rep();
  fdata_type fdata = field->fdata();

  elem_iterator ei = mesh->elem_begin();
  if (ei==mesh->elem_end()) // empty mesh
    return false;

  // the tables are to be filled with strictly increasing values.

  if (dist=="importance") { // size of element * data at element
    table.push_back(mesh->get_element_size(*ei)*fdata[*ei],*ei);
    ++ei;
    while (ei != mesh->elem_end()) {
      table.push_back(mesh->get_element_size(*ei)*fdata[*ei]+
		      table[table.size()-1].first,*ei);
      ++ei;
    }
  } else if (dist=="uniform") { // size of element only
    table.push_back(mesh->get_element_size(*ei),*ei);
    ++ei;
    while (ei != mesh->elem_end()) {
      table.push_back(mesh->get_element_size(*ei)+
		      table[table.size()-1].first,*ei);
      ++ei;
    }
  } else if (dist=="scattered") { // element index; not uniform!
    while (ei != mesh->elem_end()) {
      table.push_back(table.size(),*ei);
      ++ei;
    }    
  } else { // unknown distribution type
    return false;
  } 

  return true;
}

template <class Field>
PointCloud<double>*
SeedField::generate_widget_seeds(Field *field)
{
}

template <class Field>
PointCloud<double>*
SeedField::generate_random_seeds(Field *field)
{
  typedef typename Field::mesh_type              mesh_type;
  typedef typename DistTable<Field>::table_entry table_entry;

  DistTable<Field> table;

  if (!build_weight_table(field,table)) // unknown dist type
    return 0;

  MusilRNG rng(rngSeed_.get());

  double max = table[table.size()-1].first;
  mesh_type *mesh = field->get_typed_mesh().get_rep();
  PointCloudMesh *pcmesh = scinew PointCloudMesh;
  PointCloud<double> *pc = scinew PointCloud<double>(pcmesh,Field::NODE);

  unsigned int ns = numSeeds_.get();
  for (unsigned int loop=0;loop<ns;loop++) {
    Point p;
    table_entry e;
    table.search(e,rng()*max);           // find random cell
    mesh->get_random_point(p,e.second);  // find random point in that cell
    pcmesh->add_node(p);
  }

  return pc;
}

#if 0
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
#endif

void
SeedField::execute()
{
  // The field input is required.
  if (!ifport_->get(vfhandle_) || !(vf_ = vfhandle_.get_rep()))
  {
    return;
  }

  PointCloud<double> *seeds;

  string tab = whichTab_.get();

  if (tab=="random") {
    if (vf_->get_type_name(-1)=="LatticeVol<double>") {
      seeds = generate_random_seeds((LatticeVol<double>*)vf_);
    } else if (vf_->get_type_name(-1)=="TetVol<double>") {
      seeds = generate_random_seeds((TetVol<double>*)vf_);
    } else if (vf_->get_type_name(-1)=="TriSurf<double>") {
      seeds = generate_random_seeds((TriSurf<double>*)vf_);
    } else if (vf_->get_type_name(-1)=="LatticeVol<Vector>") {
      //pc = generate_random_seeds((LatticeVol<Vector>*)vf_);
    } else if (vf_->get_type_name(-1)=="TetVol<Vector>") {
      //pc = generate_random_seeds((TetVol<Vector>*)vf_);
    } else if (vf_->get_type_name(-1)=="TriSurf<Vector>") {
      //pc = generate_random_seeds((TriSurf<Vector>*)vf_);
    } else {
      // can't do this kind of field
      return;
    }
    ofport_->send(seeds);
  } else if (tab=="widget") {
    //pc = generate_widget_seeds(vf_);
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

