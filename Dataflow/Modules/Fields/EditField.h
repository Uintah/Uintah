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

//    File   : EditField.h
//    Author : Michael Callahan
//    Date   : June 2001

#if !defined(EditField_h)
#define EditField_h

#include <Core/Disclosure/TypeDescription.h>
#include <Core/Disclosure/DynamicLoader.h>

namespace SCIRun {

class EditFieldAlgoCN : public DynamicAlgoBase
{
public:
  virtual void execute(MeshBaseHandle src, int &num_nodes, int &num_elems) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfo *get_compile_info(const TypeDescription *msrc);
};


template <class MESH>
class EditFieldAlgoCNT : public EditFieldAlgoCN
{
public:
  //! virtual interface. 
  virtual void execute(MeshBaseHandle src, int &num_nodes, int &num_elems);
};


template <class MESH>
void 
EditFieldAlgoCNT<MESH>::execute(MeshBaseHandle mesh_h,
				int &num_nodes, int &num_elems)
{
  typedef typename MESH::Node::iterator node_iter_type;
  typedef typename MESH::Elem::iterator elem_iter_type;

  MESH *mesh = dynamic_cast<MESH *>(mesh_h.get_rep());

  int count = 0;
  node_iter_type ni = mesh->tbegin((node_iter_type *)0);
  node_iter_type nie = mesh->tend((node_iter_type *)0);
  while (ni != nie)
  {
    count++;
    ++ni;
  }
  num_nodes = count;

  count = 0;
  elem_iter_type ei = mesh->tbegin((elem_iter_type *)0);
  elem_iter_type eie = mesh->tend((elem_iter_type *)0);
  while (ei != eie)
  {
    count++;
    ++ei;
  }
  num_elems = count;
}


} // end namespace SCIRun

#endif // EditField_h
