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

#include <Core/Disclosure/TypeDescription.h>
#include <Core/Disclosure/DynamicLoader.h>
#include <Core/Datatypes/Clipper.h>
#include <sci_hash_map.h>
#include <algorithm>

namespace SCIRun {

using std::hash_map;

class ClipFieldAlgo : public DynamicAlgoBase
{
public:
  virtual FieldHandle execute(FieldHandle fieldh, ClipperHandle clipper) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfo *get_compile_info(const TypeDescription *fsrc);
};


template <class FIELD>
class ClipFieldAlgoT : public ClipFieldAlgo
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(FieldHandle fieldh, ClipperHandle clipper);
};


template <class FIELD>
FieldHandle
ClipFieldAlgoT<FIELD>::execute(FieldHandle fieldh, ClipperHandle clipper)
{
  typename FIELD::mesh_type *mesh =
    dynamic_cast<typename FIELD::mesh_type *>(fieldh->mesh().get_rep());
  typename FIELD::mesh_type *clipped = scinew typename FIELD::mesh_type();

  typedef hash_map<unsigned int,
    typename FIELD::mesh_type::Node::index_type,
    hash<unsigned int>,
    equal_to<unsigned int> > hash_type;

  hash_type nodemap;

  vector<typename FIELD::mesh_type::Elem::index_type> elemmap;

  typename FIELD::mesh_type::Elem::iterator bi, ei;
  mesh->begin(bi); mesh->end(ei);
  while (bi != ei)
  {
    Point p;
    mesh->get_center(p, *bi);
    if (clipper->inside_p(p))
    {
      // Add this element to the new mesh.
      typename FIELD::mesh_type::Node::array_type onodes;
      mesh->get_nodes(onodes, *bi);
      typename FIELD::mesh_type::Node::array_type nnodes(onodes.size());

      for (unsigned int i=0; i<onodes.size(); i++)
      {
	if (nodemap.find((unsigned int)onodes[i]) == nodemap.end())
	{
	  Point np;
	  mesh->get_center(np, onodes[i]);
	  nodemap[(unsigned int)onodes[i]] = clipped->add_point(np);
	}
	nnodes[i] = nodemap[(unsigned int)onodes[i]];
      }

      clipped->add_elem(nnodes);
      elemmap.push_back(*bi); // Assumes elements always added to end.
    }
    
    ++bi;
  }

  clipped->flush_changes();  // Really should copy normals

  FIELD *ofield = scinew FIELD(clipped, fieldh->data_at());

  if (fieldh->data_at() == Field::NODE)
  {
    FIELD *field = dynamic_cast<FIELD *>(fieldh.get_rep());
    hash_type::iterator hitr = nodemap.begin();

    while (hitr != nodemap.end())
    {
      typename FIELD::value_type val;
      field->value(val, (typename FIELD::mesh_type::Node::index_type)((*hitr).first));
      ofield->set_value(val, (typename FIELD::mesh_type::Node::index_type)((*hitr).second));

      ++hitr;
    }
  }
  else if (fieldh->data_at_type_description()->get_name() ==
	   get_type_description((typename FIELD::mesh_type::Elem *)0)->get_name())
  {
    FIELD *field = dynamic_cast<FIELD *>(fieldh.get_rep());
    for (unsigned int i=0; i < elemmap.size(); i++)
    {
      typename FIELD::value_type val;
      field->value(val,
		   (typename FIELD::mesh_type::Elem::index_type)elemmap[i]);
      ofield->set_value(val, (typename FIELD::mesh_type::Elem::index_type)i);
    }
  }
  else
  {
    cout << "Unable to copy data at this data locations, use DirectInterp.\n";
  }

  return ofield;
}



class ClipFieldMeshAlgo : public DynamicAlgoBase
{
public:
  virtual ClipperHandle execute(MeshHandle src) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfo *get_compile_info(const TypeDescription *msrc);
};


template <class MESH>
class ClipFieldMeshAlgoT : public ClipFieldMeshAlgo
{
public:
  //! virtual interface. 
  virtual ClipperHandle execute(MeshHandle src);
};


template <class MESH>
ClipperHandle
ClipFieldMeshAlgoT<MESH>::execute(MeshHandle mesh_h)
{
  MESH *msrc = dynamic_cast<MESH *>(mesh_h.get_rep());
  return scinew MeshClipper<MESH>(msrc);
}


} // end namespace SCIRun

#endif // ClipField_h
