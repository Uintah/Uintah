#ifndef SCIRun_SelectElements_H
#define SCIRun_SelectElements_H
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

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Datatypes/Field.h>
#include <Core/Containers/Array1.h>

namespace SCIRun {

class SelectElementsAlgo : public DynamicAlgoBase
{
public:
  virtual FieldHandle execute(FieldHandle field, Array1<int> &elem_valid,
			      Array1<int> &indices, int &count, 
			      const Array1<int> &values,
			      int keep_all_nodes)=0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *fsrc,
					    const TypeDescription *msrc);
};


template <class FIELD, class MESH>
class SelectElementsAlgoT : public SelectElementsAlgo
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(FieldHandle field, Array1<int> &elem_valid,
			      Array1<int> &indices, int &count, 
			      const Array1<int> &values,
			      int keep_all_nodes);
};

template <class FIELD, class MESH>
FieldHandle
SelectElementsAlgoT<FIELD, MESH>::execute(FieldHandle field, 
					  Array1<int> &elem_valid,
					  Array1<int> &indices,
					  int &count,
					  const Array1<int> &values,
					  int keep_all_nodes) {
  FIELD *fld = dynamic_cast<FIELD *>(field.get_rep());
  MESH *mesh = dynamic_cast<MESH *>(fld->get_typed_mesh().get_rep());
  typename MESH::Cell::size_type nelems;
  mesh->size(nelems);
  elem_valid.resize(nelems);
  elem_valid.initialize(0);
  typename MESH::Node::size_type nnodes;
  mesh->size(nnodes);
  Array1<bool> node_valid(nnodes);
  if (keep_all_nodes) 
    node_valid.initialize(1);
  else
    node_valid.initialize(0);

  typename MESH::Cell::iterator citer; mesh->begin(citer);
  typename MESH::Cell::iterator citere; mesh->end(citere);
  typename MESH::Node::array_type node_array;
  count=0;
  while (citer != citere) {
    typename MESH::Cell::index_type ci = *citer;
    ++citer;
    for (int ii=0; ii<values.size(); ii++) {
      if (fld->value(ci) == values[ii]) {
	elem_valid[count]=1;
	indices.add(count);
	mesh->get_nodes(node_array, ci);
	for (int jj=0; jj<node_array.size(); jj++) 
	  node_valid[(unsigned int)(node_array[jj])]=1;
      }
    }
    count++;
  }

  MESH *mesh_no_unattached_nodes = scinew MESH;
  Array1<typename MESH::Node::index_type> node_map(nnodes);
  typename MESH::Node::iterator niter; mesh->begin(niter);
  typename MESH::Node::iterator nitere; mesh->end(nitere);
  mesh->begin(niter);
  mesh->end(nitere);

  while(niter != nitere) {
    typename MESH::Node::index_type ni = *niter;
    if (node_valid[(unsigned int)(ni)]) {
      Point p;
      mesh->get_center(p, ni);
      node_map[(unsigned int)(ni)]=mesh_no_unattached_nodes->add_point(p);
    }
    ++niter;
  }

  mesh->begin(citer);
  mesh->end(citere);
  typename MESH::Node::array_type narr;
  while(citer != citere) {
    typename MESH::Cell::index_type ci = *citer;
    if (elem_valid[ci]) {
      mesh->get_nodes(narr, ci);
      for (int ii=0; ii<narr.size(); ii++) 
	narr[ii]=node_map[(unsigned int)(narr[ii])];
      mesh_no_unattached_nodes->add_elem(narr);
    }
    ++citer;
  }

  FIELD *ofld = 
    scinew FIELD(mesh_no_unattached_nodes, Field::CELL);
  
  mesh->begin(citer);
  mesh->end(citere);
  typename MESH::Cell::iterator citer_new;
  mesh_no_unattached_nodes->begin(citer_new);
  typename FIELD::value_type val;
  int c=0;
  while(citer != citere) {
    if (elem_valid[c]) {
      val=fld->value(*citer);
      ofld->set_value(val, *citer_new);
      ++citer_new;
    }
    ++citer;
    ++c;
  }
  return FieldHandle(ofld);
}

} // End namespace SCIRun

#endif
