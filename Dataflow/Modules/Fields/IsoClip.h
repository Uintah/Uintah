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

//    File   : ClipField.h
//    Author : Michael Callahan
//    Date   : August 2001

#if !defined(ClipField_h)
#define ClipField_h

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Util/ProgressReporter.h>
#include <Core/Datatypes/Clipper.h>
#include <sci_hash_map.h>
#include <algorithm>

namespace SCIRun {


class GuiInterface;

class IsoClipAlgo : public DynamicAlgoBase
{
public:

  virtual FieldHandle execute(ProgressReporter *reporter, FieldHandle fieldh,
			      double isoval, bool lte) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *fsrc);

protected:
  static int permute_table[15][4];
};


template <class FIELD>
class IsoClipAlgoT : public IsoClipAlgo
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(ProgressReporter *reporter, FieldHandle fieldh,
			      double isoval, bool lte);
};

template <class FIELD>
FieldHandle
IsoClipAlgoT<FIELD>::execute(ProgressReporter *mod, FieldHandle fieldh,
			     double isoval, bool lte)
{
  FIELD *field = dynamic_cast<FIELD*>(fieldh.get_rep());
  typename FIELD::mesh_type *mesh =
    dynamic_cast<typename FIELD::mesh_type *>(fieldh->mesh().get_rep());
  typename FIELD::mesh_type *clipped = scinew typename FIELD::mesh_type();
  *(PropertyManager *)clipped = *(PropertyManager *)mesh;

#ifdef HAVE_HASH_MAP
  typedef hash_map<unsigned int,
    typename FIELD::mesh_type::Node::index_type,
    hash<unsigned int>,
    equal_to<unsigned int> > hash_type;
#else
  typedef map<unsigned int,
    typename FIELD::mesh_type::Node::index_type,
    equal_to<unsigned int> > hash_type;
#endif

  hash_type nodemap;

  typename FIELD::mesh_type::Node::array_type onodes;
  typename FIELD::value_type v[4];
  Point p[4];

  typename FIELD::mesh_type::Elem::iterator bi, ei;
  mesh->begin(bi); mesh->end(ei);
  while (bi != ei)
  {
    mesh->get_nodes(onodes, *bi);

    unsigned int inside = 0;
    for (unsigned int i = 0; i < onodes.size(); i++)
    {
      mesh->get_center(p[i], onodes[i]);
      field->value(v[i], onodes[i]);
      inside = inside << 1;
      if (v[i] > isoval)
      {
	inside |= 1;
      }
    }

    if (lte) { inside = ~inside & 0xf; }

    if (inside == 0)
    {
    }
    else if (inside == 0xf)
    {
      // Add this element to the new mesh.
      typename FIELD::mesh_type::Node::array_type nnodes(onodes.size());

      for (unsigned int i = 0; i<onodes.size(); i++)
      {
	if (nodemap.find((unsigned int)onodes[i]) == nodemap.end())
	{
	  const typename FIELD::mesh_type::Node::index_type nodeindex =
	    clipped->add_point(p[i]);
	  nodemap[(unsigned int)onodes[i]] = nodeindex;
	  nnodes[i] = nodeindex;
	}
	else
	{
	  nnodes[i] = nodemap[(unsigned int)onodes[i]];
	}
      }

      clipped->add_elem(nnodes);
    }
    else if (inside == 0x8 || inside == 0x4 || inside == 0x2 || inside == 0x1)
    {
      const int *perm = permute_table[inside];
      // Add this element to the new mesh.
      typename FIELD::mesh_type::Node::array_type nnodes(4);

      if (nodemap.find((unsigned int)onodes[perm[0]]) == nodemap.end())
      {
	const typename FIELD::mesh_type::Node::index_type nodeindex =
	  clipped->add_point(p[perm[0]]);
	nodemap[(unsigned int)onodes[perm[0]]] = nodeindex;
	nnodes[0] = nodeindex;
      }
      else
      {
	nnodes[0] = nodemap[(unsigned int)onodes[perm[0]]];
      }

      const double imv = isoval - v[perm[0]];
      const Point l1 = Interpolate(p[perm[0]], p[perm[1]],
				   imv / (v[perm[1]] - v[perm[0]]));
      const Point l2 = Interpolate(p[perm[0]], p[perm[2]],
				   imv / (v[perm[2]] - v[perm[0]]));
      const Point l3 = Interpolate(p[perm[0]], p[perm[3]],
				   imv / (v[perm[3]] - v[perm[0]]));

      nnodes[1] = clipped->add_point(l1);
      nnodes[2] = clipped->add_point(l2);
      nnodes[3] = clipped->add_point(l3);
      clipped->add_elem(nnodes);
    }
    else if (inside == 0x7 || inside == 0xb || inside == 0xd || inside == 0xe)
    {
      const int *perm = permute_table[inside];
      // Add this element to the new mesh.
      typename FIELD::mesh_type::Node::array_type nnodes(onodes.size());

      typename FIELD::mesh_type::Node::index_type inodes[9];
      for (unsigned int i = 1; i<onodes.size(); i++)
      {
	if (nodemap.find((unsigned int)onodes[perm[i]]) == nodemap.end())
	{
	  const typename FIELD::mesh_type::Node::index_type nodeindex =
	    clipped->add_point(p[perm[i]]);
	  nodemap[(unsigned int)onodes[perm[i]]] = nodeindex;
	  inodes[i-1] = nodeindex;
	}
	else
	{
	  inodes[i-1] = nodemap[(unsigned int)onodes[perm[i]]];
	}
      }
      const double imv = isoval - v[perm[0]];
      const Point l1 = Interpolate(p[perm[0]], p[perm[1]],
				   imv / (v[perm[1]] - v[perm[0]]));
      const Point l2 = Interpolate(p[perm[0]], p[perm[2]],
				   imv / (v[perm[2]] - v[perm[0]]));
      const Point l3 = Interpolate(p[perm[0]], p[perm[3]],
				   imv / (v[perm[3]] - v[perm[0]]));

      inodes[3] = clipped->add_point(l1);
      inodes[4] = clipped->add_point(l2);
      inodes[5] = clipped->add_point(l3);

      const Point c1 = Interpolate(l1, l2, 0.5);
      const Point c2 = Interpolate(l2, l3, 0.5);
      const Point c3 = Interpolate(l3, l1, 0.5);

      inodes[6] = clipped->add_point(c1);
      inodes[7] = clipped->add_point(c2);
      inodes[8] = clipped->add_point(c3);

      nnodes[0] = inodes[0];
      nnodes[1] = inodes[3];
      nnodes[2] = inodes[8];
      nnodes[3] = inodes[6];
      clipped->add_elem(nnodes);

      nnodes[0] = inodes[1];
      nnodes[1] = inodes[4];
      nnodes[2] = inodes[6];
      nnodes[3] = inodes[7];
      clipped->add_elem(nnodes);

      nnodes[0] = inodes[2];
      nnodes[1] = inodes[5];
      nnodes[2] = inodes[7];
      nnodes[3] = inodes[8];
      clipped->add_elem(nnodes);

      nnodes[0] = inodes[0];
      nnodes[1] = inodes[6];
      nnodes[2] = inodes[8];
      nnodes[3] = inodes[7];
      clipped->add_elem(nnodes);

      nnodes[0] = inodes[0];
      nnodes[1] = inodes[8];
      nnodes[2] = inodes[2];
      nnodes[3] = inodes[7];
      clipped->add_elem(nnodes);

      nnodes[0] = inodes[1];
      nnodes[1] = inodes[2];
      nnodes[2] = inodes[7];
      nnodes[3] = inodes[6];
      clipped->add_elem(nnodes);

      nnodes[0] = inodes[0];
      nnodes[1] = inodes[1];
      nnodes[2] = inodes[2];
      nnodes[3] = inodes[6];
      clipped->add_elem(nnodes);
    }

    ++bi;
  }

  FIELD *ofield = scinew FIELD(clipped, fieldh->data_at());
  *(PropertyManager *)ofield = *(PropertyManager *)(fieldh.get_rep());

  if (fieldh->data_at() == Field::NODE)
  {
    FIELD *field = dynamic_cast<FIELD *>(fieldh.get_rep());
    typename hash_type::iterator hitr = nodemap.begin();

    while (hitr != nodemap.end())
    {
      typename FIELD::value_type val;
      field->value(val, (typename FIELD::mesh_type::Node::index_type)((*hitr).first));
      ofield->set_value(val, (typename FIELD::mesh_type::Node::index_type)((*hitr).second));

      ++hitr;
    }
  }
  else
  {
    mod->warning("Unable to copy data at this field data location.");
  }

  return ofield;
}



} // end namespace SCIRun

#endif // ClipField_h
