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
#include <Core/Datatypes/LatticeVol.h>
#include <Core/Datatypes/TriSurf.h>
#include <math.h>
#include <set>

#include <iostream>

using std::set;
using std::vector;
using std::pair;

namespace SCIRun {

template <class Mesh>
class DistTable
{
public:
  typedef typename Mesh::Elem::index_type      elem_index_type;
  typedef pair<long double, elem_index_type>   table_entry_type;

  vector<table_entry_type> table_;

  //DistTable() {}
  //~DistTable() {}

  void push_back(long double size, elem_index_type id) 
  { table_.push_back(table_entry_type(size,id)); }
  void push_back(table_entry_type entry) 
  { table_.push_back(entry); }

  const table_entry_type& operator[](unsigned idx) const
  { return table_[idx]; }
  table_entry_type& operator[](unsigned idx)
  { return table_[idx]; }

  int size() { return table_.size(); }
  void clear() { table_.clear(); }

  bool search(table_entry_type&, long double);
};

template <class Mesh>
bool
DistTable<Mesh>::search(table_entry_type &e, long double d)
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

  template <class Mesh> 
  bool build_weight_table_sfi(MeshBaseHandle,
			      ScalarFieldInterface *,
			      DistTable<Mesh> &);

  template <class Mesh> 
  bool build_weight_table_vfi(MeshBaseHandle,
			      VectorFieldInterface *,
			      DistTable<Mesh> &);

  template <class Mesh>
  void generate_random_seeds(FieldHandle, Mesh *);

  void generate_widget_seeds(Field *);

  template <class M> 
  void dispatch(M *mesh);

public:
  CrowdMonitor widget_lock_;
  GaugeWidget *rake_;
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
    widget_lock_("StreamLines widget lock")
{
  vf_ = 0;
  widgetid_=0;;
  rake_ = 0;

  firsttime_ = true;
}


SeedField::~SeedField()
{
}


void
SeedField::widget_moved(int i)
{
  if (rake_) 
    rake_->GetEndpoints(endpoint0_,endpoint1_);

  if (i==1) {
    want_to_execute();
  } else {
    Module::widget_moved(i);
  }
}



template <class Mesh>
bool 
SeedField::build_weight_table_sfi(MeshBaseHandle mesh_h,
				  ScalarFieldInterface *sfi,
				  DistTable<Mesh> &table)
{
  typedef typename Mesh::Elem::iterator   elem_iterator;
  typedef typename Mesh::Elem::index_type elem_index;
  
//  long double size = 2.e-300;
  long double size = 1;

  string dist = randDist_.get();

  Mesh *mesh = dynamic_cast<Mesh *>(mesh_h.get_rep());
  if (mesh == 0) { error("No mesh"); return false; }

  elem_iterator ei = mesh->tbegin((elem_iterator *)0);
  elem_iterator endi = mesh->tend((elem_iterator *)0);
  if (ei == endi) // empty mesh
    return false;

  // the tables are to be filled with increasing values.
  // degenerate elements (size<=0) will not be included in the table.
  // mag(data) <=0 means ignore the element (don't include in the table).
  // bin[n] = b[n-1]+newval;bin[0]=newval

  if (dist=="impuni") { // size of element * data at element
    double val;
    Point p;
    for(;;) {
      mesh->get_center(p,*ei);
      if (!sfi->interpolate(val,p)) continue;
      if ((val > 0) && (mesh->get_element_size(*ei)>0)) break;
      ++ei;
    }
    table.push_back(mesh->get_element_size(*ei) * val,*ei);
    ++ei;
    while (ei != endi) {
      mesh->get_center(p,*ei);
      if (!sfi->interpolate(val,p)) continue;
      if ( mesh->get_element_size(*ei)>0 && val > 0) {
	table.push_back(mesh->get_element_size(*ei) * val +
			table[table.size()-1].first,*ei);
      }
      ++ei;
    }
  } else if (dist == "impscat") { // standard size * data at element
    double val;
    Point p;
    for(;;) {
      mesh->get_center(p,*ei);
      if (!sfi->interpolate(val,p)) continue;
      if (val > 0) break;
      ++ei;
    }
    table.push_back(size * val, *ei);
    ++ei;
    while (ei != endi) {
      mesh->get_center(p,*ei);
      if (!sfi->interpolate(val,p)) continue;
      if (val > 0) {
	table.push_back(size * val +
			table[table.size()-1].first,*ei);
      }
      ++ei;
    }
  } else if (dist=="uniuni") { // size of element only
    for(;;) {
      if (mesh->get_element_size(*ei)>0) break;
      ++ei;
    }
    table.push_back(mesh->get_element_size(*ei),*ei);
    ++ei;
    while (ei != endi) {
      if (mesh->get_element_size(*ei)>0)
	table.push_back(mesh->get_element_size(*ei)+
			table[table.size()-1].first,*ei);
      ++ei;
    }
  } else if (dist=="uniscat") { // standard size only
    table.push_back(size,*ei);
    ++ei;
    while (ei != endi) {
      table.push_back(size+table[table.size()-1].first,*ei);
      ++ei;
    }    
  } else { // unknown distribution type
    return false;
  } 

  return true;
}


template <class Mesh>
bool 
SeedField::build_weight_table_vfi(MeshBaseHandle mesh_h,
				  VectorFieldInterface *vfi,
				  DistTable<Mesh> &table)
{
  typedef typename Mesh::Elem::iterator   elem_iterator;
  typedef typename Mesh::Elem::index_type elem_index;
  
//  long double size = 2.e-300;
  long double size = 1;

  string dist = randDist_.get();

  Mesh *mesh = dynamic_cast<Mesh *>(mesh_h.get_rep());
  if (mesh == 0) { error("No mesh"); return false; }

  elem_iterator ei = mesh->tbegin((elem_iterator *)0);
  elem_iterator endi = mesh->tend((elem_iterator *)0);
  if (ei == endi) // empty mesh
    return false;

  // the tables are to be filled with increasing values.
  // degenerate elements (size<=0) will not be included in the table.
  // mag(data) <=0 means ignore the element (don't include in the table).
  // bin[n] = b[n-1]+newval;bin[0]=newval

  if (dist=="impuni") { // size of element * data at element
    Vector val;
    Point p;
    for(;;) {
      mesh->get_center(p,*ei);
      if (!vfi->interpolate(val,p)) continue;
      if ((val.length()>0)&&(mesh->get_element_size(*ei)>0)) break;
      ++ei;
    }
    table.push_back(mesh->get_element_size(*ei) * val.length(),*ei);
    ++ei;
    while (ei != endi) {
      mesh->get_center(p,*ei);
      if (!vfi->interpolate(val,p)) continue;
      if ( mesh->get_element_size(*ei)>0 && val.length() > 0) {
	table.push_back(mesh->get_element_size(*ei) * val.length() +
			table[table.size()-1].first,*ei);
      }
      ++ei;
    }
  } else if (dist == "impscat") { // standard size * data at element
    Vector val;
    Point p;
    for(;;) {
      mesh->get_center(p,*ei);
      if (!vfi->interpolate(val,p)) continue;
      if (val.length() > 0) break;
      ++ei;
    }
    table.push_back(size * val.length(), *ei);
    ++ei;
    while (ei != endi) {
      mesh->get_center(p,*ei);
      if (!vfi->interpolate(val,p)) continue;
      if (val.length() > 0) {
	table.push_back(size * val.length() +
			table[table.size()-1].first,*ei);
      }
      ++ei;
    }
  } else if (dist=="uniuni") { // size of element only
    for(;;) {
      if (mesh->get_element_size(*ei)>0) break;
      ++ei;
    }
    table.push_back(mesh->get_element_size(*ei),*ei);
    ++ei;
    while (ei != endi) {
      if (mesh->get_element_size(*ei)>0)
	table.push_back(mesh->get_element_size(*ei)+
			table[table.size()-1].first,*ei);
      ++ei;
    }
  } else if (dist=="uniscat") { // standard size only
    table.push_back(size,*ei);
    ++ei;
    while (ei != endi) {
      table.push_back(size+table[table.size()-1].first,*ei);
      ++ei;
    }    
  } else { // unknown distribution type
    return false;
  } 

  return true;
}



template <class Mesh>
void
SeedField::generate_random_seeds(FieldHandle field, Mesh *)
{
  DistTable<Mesh> table;
  table.clear();

  ScalarFieldInterface *sfi = field->query_scalar_interface();
  VectorFieldInterface *vfi = field->query_vector_interface();
  if (sfi)
  {
    if (!build_weight_table_sfi(field->mesh(), sfi, table))
    {
      return;
    }
  }
  else if (vfi)
  {
    if (!build_weight_table_vfi(field->mesh(), vfi, table))
    {
      return;
    }
  }

  int rngSeed = rngSeed_.get();
  MusilRNG rng(rngSeed);
  rngSeed_.set(rngSeed+1);

  long double max = table[table.size()-1].first;
  Mesh *mesh = dynamic_cast<Mesh *>(field->mesh().get_rep());

  PointCloudMesh *pcmesh = scinew PointCloudMesh;

  unsigned int ns = numSeeds_.get();
  unsigned int loop;
  
  for (loop=0;loop<ns;loop++) {
    Point p;
    typename DistTable<Mesh>::table_entry_type e;
    table.search(e,rng() * max);             // find random cell
    // Find random point in that cell.
    mesh->get_random_point(p, e.second, rngSeed+loop);
    pcmesh->add_node(p);
  }

  PointCloud<double> *seeds = scinew PointCloud<double>(pcmesh,Field::NODE);
  PointCloud<double>::fdata_type &fdata = seeds->fdata();
  for (loop=0;loop<ns;++loop)
  {
    fdata[loop]=1;
  }

  ofport_->send(seeds);
  if (widgetid_) ogport_->delObj(widgetid_);
  widgetid_=0;
  rake_ = 0;
  ogport_->flushViews();
}


void
SeedField::generate_widget_seeds(Field *field)
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
SeedField::execute()
{
  ifport_ = (FieldIPort *)get_iport("Field to Seed");
  ofport_ = (FieldOPort *)get_oport("Seeds");
  ogport_ = (GeometryOPort *)get_oport("Seeding Widget");
  
  // The field input is required.
  if (!ifport_->get(vfhandle_) || !(vf_ = vfhandle_.get_rep()))
  {
    return;
  }

  string tab = whichTab_.get();

  if (tab=="Random") {
    if (vf_->get_type_name(-1)=="LatticeVol<double>") {
      generate_random_seeds(vfhandle_, (LatVolMesh *)0);
    } else if (vf_->get_type_name(-1)=="TetVol<double>") {
      generate_random_seeds(vfhandle_, (TetVolMesh *)0);
    } else if (vf_->get_type_name(-1)=="TriSurf<double>") {
      generate_random_seeds(vfhandle_, (TriSurfMesh *)0);
    } else if (vf_->get_type_name(-1)=="LatticeVol<Vector>") {
      generate_random_seeds(vfhandle_, (LatVolMesh *)0);
    } else if (vf_->get_type_name(-1)=="TetVol<Vector>") {
      generate_random_seeds(vfhandle_, (TetVolMesh *)0);
    } else if (vf_->get_type_name(-1)=="TriSurf<Vector>") {
      generate_random_seeds(vfhandle_, (TriSurfMesh *)0);
    } else if (vf_->get_type_name(-1)=="TetVol<int>") {
      generate_random_seeds(vfhandle_, (TetVolMesh *)0);
    } else {
      // can't do this kind of field
      return;
    }
  } else if (tab=="Widget")
    generate_widget_seeds(vf_);
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

