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

//    File   : Unstructure.h
//    Author : Michael Callahan
//    Date   : September 2001

#if !defined(Unstructure_h)
#define Unstructure_h

#include <Core/Disclosure/TypeDescription.h>
#include <Core/Disclosure/DynamicLoader.h>
#include <Core/Datatypes/LatticeVol.h>
#include <Core/Datatypes/ImageField.h>
#include <Core/Datatypes/ScanlineField.h>
#include <Core/Datatypes/HexVol.h>
#include <Core/Datatypes/QuadSurf.h>
#include <Core/Datatypes/ContourField.h>

namespace SCIRun {

class UnstructureAlgo : public DynamicAlgoBase
{
public:
  virtual FieldHandle execute(FieldHandle src) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfo *get_compile_info(const TypeDescription *fsrc,
				       const string &partial_fdst);
};


template <class FSRC, class FDST>
class UnstructureAlgoT : public UnstructureAlgo
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(FieldHandle src);
};


struct SpecialUnstructuredHash
{
  size_t operator()(const LatVolMesh::Node::index_type &n) const
  { return n.i_ * ((1 << 20) - 1) + n.j_ * ((1 << 10) - 1) + n.k_; }

  size_t operator()(const ImageMesh::Node::index_type &n) const
  { return n.i_ * ((1 << 10) - 1) + n.j_; }

  size_t operator()(const ScanlineMesh::Node::index_type &n) const
  { return n; }

  size_t operator()(const LatVolMesh::Elem::index_type &n) const
  { return n.i_ * ((1 << 20) - 1) + n.j_ * ((1 << 10) - 1) + n.k_; }

  size_t operator()(const ImageMesh::Elem::index_type &n) const
  { return n.i_ * ((1 << 10) - 1) + n.j_; }

  size_t operator()(const ScanlineMesh::Elem::index_type &n) const
  { return n; }
};

struct SpecialUnstructuredEqual
{
  bool operator()(const LatVolMesh::Node::index_type &a,
		  const LatVolMesh::Node::index_type &b) const
  { return a.i_ == b.i_ && a.j_ == b.j_ && a.k_ == b.k_; }

  bool operator()(const ImageMesh::Node::index_type &a,
		  const ImageMesh::Node::index_type &b) const
  { return a.i_ == b.i_ && a.j_ == b.j_; }

  bool operator()(const ScanlineMesh::Node::index_type &a,
		  const ScanlineMesh::Node::index_type &b) const
  { return a == b; }

  bool operator()(const LatVolMesh::Elem::index_type &a,
		  const LatVolMesh::Elem::index_type &b) const
  { return a.i_ == b.i_ && a.j_ == b.j_ && a.k_ == b.k_; }

  bool operator()(const ImageMesh::Elem::index_type &a,
		  const ImageMesh::Elem::index_type &b) const
  { return a.i_ == b.i_ && a.j_ == b.j_; }

  bool operator()(const ScanlineMesh::Elem::index_type &a,
		  const ScanlineMesh::Elem::index_type &b) const
  { return a == b; }
};


template <class FSRC, class FDST>
FieldHandle
UnstructureAlgoT<FSRC, FDST>::execute(FieldHandle field_h)
{
  FSRC *ifield = dynamic_cast<FSRC *>(field_h.get_rep());
  typename FSRC::mesh_handle_type mesh = ifield->get_typed_mesh();

  typename FDST::mesh_handle_type outmesh = scinew typename FDST::mesh_type();

  typedef hash_map<typename FSRC::mesh_type::Node::index_type,
    typename FDST::mesh_type::Node::index_type,
    SpecialUnstructuredHash, SpecialUnstructuredEqual> node_hash_type;

  node_hash_type nodemap;

  typedef hash_map<typename FSRC::mesh_type::Elem::index_type,
    typename FDST::mesh_type::Elem::index_type,
    SpecialUnstructuredHash, SpecialUnstructuredEqual> elem_hash_type;
  
  elem_hash_type elemmap;

  typename FSRC::mesh_type::Elem::iterator bi, ei;
  mesh->begin(bi); mesh->end(ei);
  while (bi != ei)
  {
    // Add this element to the new mesh.
    typename FSRC::mesh_type::Node::array_type onodes;
    mesh->get_nodes(onodes, *bi);
    typename FDST::mesh_type::Node::array_type nnodes(onodes.size());

    for (unsigned int i=0; i<onodes.size(); i++)
    {
      if (nodemap.find(onodes[i]) == nodemap.end())
      {
	Point np;
	mesh->get_center(np, onodes[i]);
	nodemap[onodes[i]] = outmesh->add_point(np);
      }
      nnodes[i] = nodemap[onodes[i]];
    }

    elemmap[*bi] = outmesh->add_elem(nnodes);
    ++bi;
  }

  outmesh->flush_changes();  // Really should copy normals

  FDST *ofield = scinew FDST(outmesh, field_h->data_at());

  if (field_h->data_at() == Field::NODE)
  {
    node_hash_type::iterator hitr = nodemap.begin();

    while (hitr != nodemap.end())
    {
      typename FSRC::value_type val;
      ifield->value(val, (*hitr).first);
      ofield->set_value(val, (*hitr).second);

      ++hitr;
    }
  }
  else if (field_h->data_at_type_description()->get_name() ==
	   get_type_description((typename FSRC::mesh_type::Elem *)0)->get_name())
  {
    elem_hash_type::iterator hitr = elemmap.begin();

    while (hitr != elemmap.end())
    {
      typename FSRC::value_type val;
      ifield->value(val, (*hitr).first);
      ofield->set_value(val, (*hitr).second);

      ++hitr;
    }
  }
  else
  {
    cout << "Unable to copy data at this data locations, use DirectInterp.\n";
  }

  return ofield;
}


} // end namespace SCIRun

#endif // Unstructure_h
