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

//    File   : ManageFieldMesh.h
//    Author : Michael Callahan
//    Date   : July 2001

#if !defined(ManageFieldMesh_h)
#define ManageFieldMesh_h

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Util/ProgressReporter.h>
#include <Core/Geometry/Tensor.h>


namespace SCIRun {

class ManageFieldMeshAlgoExtract : public DynamicAlgoBase
{
public:
  virtual MatrixHandle execute(MeshHandle src) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *fsrc);
};


template <class MESH>
class ManageFieldMeshAlgoExtractT : public ManageFieldMeshAlgoExtract
{
public:
  //! virtual interface. 
  virtual MatrixHandle execute(MeshHandle src);
};


template <class MESH>
MatrixHandle
ManageFieldMeshAlgoExtractT<MESH>::execute(MeshHandle mesh_h)
{
  MESH *mesh = dynamic_cast<MESH *>(mesh_h.get_rep());

  typename MESH::Node::size_type nsize;
  mesh->size(nsize);
  DenseMatrix *omatrix = scinew DenseMatrix((int)nsize, 3);

  Point p;
  int index = 0;
  typename MESH::Node::iterator iter, eiter;
  mesh->begin(iter);
  mesh->end(eiter);
  while (iter != eiter)
  {
    mesh->get_center(p, *iter);
    omatrix->put(index, 0, p.x());
    omatrix->put(index, 1, p.y());
    omatrix->put(index, 2, p.z());
    ++index;
    ++iter;
  }

  return MatrixHandle(omatrix);
}


class ManageFieldMeshAlgoInsert : public DynamicAlgoBase
{
public:
  virtual FieldHandle execute(ProgressReporter *m,
			      FieldHandle src, MatrixHandle mat) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *fsrc);
};


template <class FIELD>
class ManageFieldMeshAlgoInsertT : public ManageFieldMeshAlgoInsert
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(ProgressReporter *m,
			      FieldHandle src, MatrixHandle mat);
};


template <class FIELD>
FieldHandle
ManageFieldMeshAlgoInsertT<FIELD>::execute(ProgressReporter *mod,
					   FieldHandle src,
					   MatrixHandle matrix)
{
  FIELD *copy = dynamic_cast<FIELD *>(src->clone());
  copy->mesh_detach();
  typename FIELD::mesh_handle_type mesh = copy->get_typed_mesh();

  if (matrix->ncols() < 3)
  {
    mod->error("Matrix must contain at least 3 columns for position data.");
    return 0;
  }
  if (matrix->ncols() > 3)
  {
    mod->remark("Matrix contains unused columns, only first three are used.");
  }

  typename FIELD::mesh_type::Node::size_type nsize;
  mesh->size(nsize);
  if (((int)nsize) != matrix->nrows())
  {
    mod->error("Matrix rows do not fit in this mesh.  May need transpose.");
    return 0;
  }

  unsigned int index = 0;
  typename FIELD::mesh_type::Node::iterator itr, eitr;
  mesh->begin(itr);
  mesh->end(eitr);
  while (itr != eitr)
  {
    const Point p(matrix->get(index, 0),
		  matrix->get(index, 1),
		  matrix->get(index, 2));
    mesh->set_point(p, *itr);
    ++index;
    ++itr;
  }

  return FieldHandle(copy);
}


} // end namespace SCIRun

#endif // ManageFieldMesh_h
