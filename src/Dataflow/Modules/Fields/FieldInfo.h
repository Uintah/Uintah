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
  virtual void execute(MeshHandle src, int &num_nodes, int &num_elems) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *msrc);
};


template <class MESH>
class FieldInfoAlgoCountT : public FieldInfoAlgoCount
{
public:
  //! virtual interface. 
  virtual void execute(MeshHandle src, int &num_nodes, int &num_elems);
};


template <class MESH>
void 
FieldInfoAlgoCountT<MESH>::execute(MeshHandle mesh_h, int &num_nodes, 
				   int &num_elems)
{
  MESH *mesh = dynamic_cast<MESH *>(mesh_h.get_rep());
  typename MESH::Node::size_type nnodes;
  typename MESH::Elem::size_type nelems;
  mesh->size(nnodes);
  mesh->size(nelems);
  num_nodes=nnodes;
  num_elems=nelems;
}

} // end namespace SCIRun

#endif // FieldInfo_h
