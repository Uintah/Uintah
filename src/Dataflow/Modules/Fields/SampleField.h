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

//    File   : SampleField.h
//    Author : Michael Callahan
//    Date   : June 2001

#if !defined(SampleField_h)
#define SampleField_h

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Math/MusilRNG.h>
#include <Core/Datatypes/PointCloudField.h>
#include <Core/Datatypes/FieldInterface.h>
#include <Core/Util/ProgressReporter.h>
#include <algorithm>

namespace SCIRun {

class SampleFieldRandomAlgo : public DynamicAlgoBase
{
public:

  virtual FieldHandle execute(ProgressReporter *mod,
			      FieldHandle field, unsigned int num_seeds,
			      int rng_seed, const string &dist, int clamp) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *msrc);
};


template <class Mesh>
class SampleFieldRandomAlgoT : public SampleFieldRandomAlgo
{
private:
  enum mode_e {IMPUNI, IMPSCAT, UNIUNI, UNISCAT};

  typedef pair<long double, typename Mesh::Elem::index_type> weight_type;

  bool build_table(Mesh *mesh,
		   ScalarFieldInterfaceHandle sfi,
		   VectorFieldInterfaceHandle vfi,
		   vector<weight_type> &table,
		   const mode_e dist);

  static bool
  weight_less(const weight_type &a, const weight_type &b)
  {
    return a.first < b.first;
  }

public:

  virtual FieldHandle execute(ProgressReporter *mod,
			      FieldHandle field, unsigned int num_seeds,
			      int rng_seed, const string &dist, int clamp);
};


template <class Mesh>
bool 
SampleFieldRandomAlgoT<Mesh>::build_table(Mesh *mesh,
					  ScalarFieldInterfaceHandle sfi,
					  VectorFieldInterfaceHandle vfi,
					  vector<weight_type> &table,
					  const mode_e dist)
{
  typename Mesh::Elem::iterator ei, ei_end;
  mesh->begin(ei);
  mesh->end(ei_end);
  long double sum = 0.0;
  while (ei != ei_end)
  {
    double elemsize = 0.0;
    if (dist == IMPUNI)
    { // Size of element * data at element.
      Point p;
      mesh->get_center(p, *ei);
      if (vfi.get_rep())
      {
	Vector v;
	if (vfi->interpolate(v, p))
	{
	  elemsize = v.length() * mesh->get_element_size(*ei);
	}
      }
      if (sfi.get_rep())
      {
	double d;
	if (sfi->interpolate(d, p) && d > 0.0)
	{
	  elemsize = d * mesh->get_element_size(*ei);
	}
      }
    }
    else if (dist == IMPSCAT)
    { // data at element
      Point p;
      mesh->get_center(p, *ei);
      if (vfi.get_rep())
      {
	Vector v;
	if (vfi->interpolate(v, p))
	{
	  elemsize = v.length();
	}
      }
      if (sfi.get_rep())
      {
	double d;
	if (sfi->interpolate(d, p) && d > 0.0)
	{
	  elemsize = d;
	}
      }
    }
    else if (dist == UNIUNI)
    { // size of element only
      elemsize = mesh->get_element_size(*ei);
    }
    else if (dist == UNISCAT)
    { 
      elemsize = 1.0;
    }
    if (elemsize > 0.0)
    {
      sum += elemsize;
      table.push_back(weight_type(sum, *ei));
    }
    ++ei;
  }
  if (table.size() > 0)
  {
    return true;
  }
  return false;
}



template <class Mesh>
FieldHandle
SampleFieldRandomAlgoT<Mesh>::execute(ProgressReporter *mod,
				      FieldHandle field,
				      unsigned int num_seeds,
				      int rng_seed,
				      const string &dist,
				      int clamp)
{
  vector<weight_type> table;
  Mesh *mesh = dynamic_cast<Mesh *>(field->mesh().get_rep());
  if (mesh == 0)
  {
    mod->error("Invalid input mesh.");
    return 0;
  }

  ScalarFieldInterfaceHandle sfi = 0;
  VectorFieldInterfaceHandle vfi = 0;
  mode_e distmode = IMPUNI;
  if (dist == "impscat")
  {
    distmode = IMPSCAT;
  }
  else if (dist == "uniuni")
  {
    distmode = UNIUNI;
  }
  else if (dist == "uniscat")
  {
    distmode = UNISCAT;
  }

  if (distmode == UNIUNI || distmode == UNISCAT)
  {
    if (!build_table(mesh, 0, 0, table, distmode))
    {
      mod->error("Unable to build unweighted weight table for this mesh.");
      mod->error("Mesh is likely to be empty.");
      return 0;
    }
  }
  else if ((sfi = field->query_scalar_interface(mod)).get_rep() ||
	   (vfi = field->query_vector_interface(mod)).get_rep())
  {
    mesh->synchronize(Mesh::LOCATE_E);
    if (!build_table(mesh, sfi, vfi, table, distmode))
    {
      mod->error("Invalid weights in mesh, probably all zero.");
      mod->error("Try using an unweighted option.");
      return 0;
    }
  }
  else
  {
    mod->error("Mesh contains non-weight data.");
    mod->error("Try using an unweighted option.");
    return 0;
  }

  MusilRNG rng(rng_seed);

  long double max = table[table.size()-1].first;

  PointCloudMesh *pcmesh = scinew PointCloudMesh;

  unsigned int i;
  for (i=0; i < num_seeds; i++)
  {
    Point p;
    typename vector<weight_type>::iterator loc;
    do {
      loc =
	std::lower_bound(table.begin(), table.end(),
			 weight_type(rng() * max,
				     typename Mesh::Elem::index_type()),
			 weight_less);
    } while (loc == table.end());

    if (clamp)
    {
      // Find a random node in that cell.
      typename Mesh::Node::array_type ra;
      mesh->get_nodes(ra, (*loc).second);
      mesh->get_center(p,ra[(int)(rng()*ra.size()+0.5)]);
    }
    else
    {
      // Find random point in that cell.
      mesh->get_random_point(p, (*loc).second, rng_seed + i);
    }
    pcmesh->add_node(p);
  }

  PointCloudField<double> *seeds =
    scinew PointCloudField<double>(pcmesh,Field::NODE);

  return seeds;
}


} // end namespace SCIRun

#endif // SampleField_h
