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

//    File   : FieldInfo.h
//    Author : McKay Davis
//    Date   : July 2002

#if !defined(FieldInfo_h)
#define FieldInfo_h

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Datatypes/Field.h>

namespace SCIRun {

class FieldInfoAlgoCount : public DynamicAlgoBase
{
public:
  virtual void execute(MeshHandle src, int &num_nodes, int &num_elems,
		       int &dimension) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfo *get_compile_info(const TypeDescription *msrc);
};


template <class MESH>
class FieldInfoAlgoCountT : public FieldInfoAlgoCount
{
public:
  //! virtual interface. 
  virtual void execute(MeshHandle src, int &num_nodes, int &num_elems,
		       int &dimension);
};


template <class MESH>
void 
FieldInfoAlgoCountT<MESH>::execute(MeshHandle mesh_h,
				   int &num_nodes, int &num_elems,
				   int &dimension)
{
  typedef typename MESH::Node::iterator node_iter_type;
  typedef typename MESH::Elem::iterator elem_iter_type;

  MESH *mesh = dynamic_cast<MESH *>(mesh_h.get_rep());

  mesh->synchronize(Mesh::ALL_ELEMENTS_E);

  int count = 0;
  node_iter_type ni; mesh->begin(ni);
  node_iter_type nie; mesh->end(nie);
  while (ni != nie)
  {
    count++;
    ++ni;
  }
  num_nodes = count;

  count = 0;
  elem_iter_type ei; mesh->begin(ei);
  elem_iter_type eie; mesh->end(eie);
  while (ei != eie)
  {
    count++;
    ++ei;
  }
  num_elems = count;

  dimension = 0;
  
  typename MESH::Edge::iterator eb, ee; mesh->begin(eb); mesh->end(ee);
  if (eb != ee);
  {
    dimension = 1;
  }
  typename MESH::Face::iterator fb, fe; mesh->begin(fb); mesh->end(fe);
  if (fb != fe)
  {
    dimension = 2;
  }
  typename MESH::Cell::iterator cb, ce; mesh->begin(cb); mesh->end(ce);
  if (cb != ce)
  {
    dimension = 3;
  }
}






} // end namespace SCIRun

#endif // FieldInfo_h
