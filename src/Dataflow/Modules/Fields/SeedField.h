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

//    File   : SeedField.h
//    Author : Michael Callahan
//    Date   : June 2001

#if !defined(SeedField_h)
#define SeedField_h

#include <Core/Disclosure/TypeDescription.h>
#include <Core/Disclosure/DynamicLoader.h>
#include <Core/Math/MusilRNG.h>
#include <Core/Datatypes/PointCloud.h>

namespace SCIRun {

class SeedFieldAlgo : public DynamicAlgoBase
{
public:

  virtual FieldHandle execute(FieldHandle field, unsigned int num_seeds,
			      int rng_seed, const string &dist) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfo *get_compile_info(const TypeDescription *msrc);
};


template <class Mesh>
class SeedFieldAlgoT : public SeedFieldAlgo
{
private:
  class DistTable
  {
  public:
    typedef typename Mesh::Elem::index_type      elem_index_type;
    typedef pair<long double, elem_index_type>   table_entry_type;

    vector<table_entry_type> table_;

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

  bool build_weight_table_sfi(MeshHandle mesh_h,
			      ScalarFieldInterface *sfi,
			      DistTable &table,
			      const string &dist);

  bool build_weight_table_vfi(MeshHandle mesh_h,
			      VectorFieldInterface *vfi,
			      DistTable &table,
			      const string &dist);

public:

  virtual FieldHandle execute(FieldHandle field, unsigned int num_seeds,
			      int rng_seed, const string &dist);
};


template <class Mesh>
bool
SeedFieldAlgoT<Mesh>::DistTable::search(table_entry_type &e, long double d)
{
  int min = 0;
  int max = table_.size() - 1;
  int cur = max / 2;

  if ( (d < table_[0].first) || (d>table_[max].first) )
  {
    return false; 
  }

  // use binary search to find the bin holding the value d
  while (max - 1 > min)
  {
    if (table_[cur].first >= d) max = cur;
    if (table_[cur].first < d)  min = cur;
    cur = (max - min) / 2 + min;
  }

  e = (table_[min].first>d) ? table_[min] : table_[max];

  return true;
}


template <class Mesh>
bool 
SeedFieldAlgoT<Mesh>::build_weight_table_sfi(MeshHandle mesh_h,
					     ScalarFieldInterface *sfi,
					     DistTable &table,
					     const string &dist)
{
  typedef typename Mesh::Elem::iterator   elem_iterator;
  typedef typename Mesh::Elem::index_type elem_index;
  
  long double size = 1;

  Mesh *mesh = dynamic_cast<Mesh *>(mesh_h.get_rep());
  if (mesh == 0)
  {
    cout << "SeedFieldAlgo:: No mesh\n";
    return false;
  }

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
SeedFieldAlgoT<Mesh>::build_weight_table_vfi(MeshHandle mesh_h,
					     VectorFieldInterface *vfi,
					     DistTable &table,
					     const string &dist)
{
  typedef typename Mesh::Elem::iterator   elem_iterator;
  typedef typename Mesh::Elem::index_type elem_index;
  
//  long double size = 2.e-300;
  long double size = 1;

  Mesh *mesh = dynamic_cast<Mesh *>(mesh_h.get_rep());
  if (mesh == 0)
  {
    cout << "SeedFieldAlgo:: No mesh.\n";
    return false;
  }

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
FieldHandle
SeedFieldAlgoT<Mesh>::execute(FieldHandle field,
			      unsigned int num_seeds,
			      int rng_seed,
			      const string &dist)
{
  DistTable table;
  table.clear();

  ScalarFieldInterface *sfi = field->query_scalar_interface();
  VectorFieldInterface *vfi = field->query_vector_interface();
  if (sfi)
  {
    if (!build_weight_table_sfi(field->mesh(), sfi, table, dist))
    {
      return 0;
    }
  }
  else if (vfi)
  {
    if (!build_weight_table_vfi(field->mesh(), vfi, table, dist))
    {
      return 0;
    }
  }

  MusilRNG rng(rng_seed);

  long double max = table[table.size()-1].first;
  Mesh *mesh = dynamic_cast<Mesh *>(field->mesh().get_rep());

  PointCloudMesh *pcmesh = scinew PointCloudMesh;

  unsigned int loop;
  for (loop=0; loop < num_seeds; loop++) {
    Point p;
    typename DistTable::table_entry_type e;
    table.search(e,rng() * max);             // find random cell
    // Find random point in that cell.
    mesh->get_random_point(p, e.second, rng_seed + loop);
    pcmesh->add_node(p);
  }

  PointCloud<double> *seeds = scinew PointCloud<double>(pcmesh,Field::NODE);
  PointCloud<double>::fdata_type &fdata = seeds->fdata();
  for (loop=0; loop<num_seeds; ++loop)
  {
    fdata[loop]=1;
  }

  return seeds;
}


} // end namespace SCIRun

#endif // SeedField_h
