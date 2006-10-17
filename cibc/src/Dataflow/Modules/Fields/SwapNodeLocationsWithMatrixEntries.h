/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/


//    File   : SwapNodeLocationsWithMatrixEntries.h
//    Author : Michael Callahan
//    Date   : July 2001

#if !defined(SwapNodeLocationsWithMatrixEntries_h)
#define SwapNodeLocationsWithMatrixEntries_h

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Util/ProgressReporter.h>
#include <Core/Geometry/Tensor.h>


namespace SCIRun {

class SwapNodeLocationsWithMatrixEntriesAlgoExtract : public DynamicAlgoBase
{
public:
  virtual MatrixHandle execute(MeshHandle src) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *fsrc);
};


template <class MESH>
class SwapNodeLocationsWithMatrixEntriesAlgoExtractT : public SwapNodeLocationsWithMatrixEntriesAlgoExtract
{
public:
  //! virtual interface. 
  virtual MatrixHandle execute(MeshHandle src);
};


template <class MESH>
MatrixHandle
SwapNodeLocationsWithMatrixEntriesAlgoExtractT<MESH>::execute(MeshHandle mesh_h)
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


class SwapNodeLocationsWithMatrixEntriesAlgoInsert : public DynamicAlgoBase
{
public:
  virtual FieldHandle execute(ProgressReporter *reporter,
			      FieldHandle src, MatrixHandle mat) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *fsrc);
};


template <class FIELD>
class SwapNodeLocationsWithMatrixEntriesAlgoInsertT : public SwapNodeLocationsWithMatrixEntriesAlgoInsert
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(ProgressReporter *reporter,
			      FieldHandle src, MatrixHandle mat);
};


template <class FIELD>
FieldHandle
SwapNodeLocationsWithMatrixEntriesAlgoInsertT<FIELD>::execute(ProgressReporter *reporter,
					   FieldHandle src,
					   MatrixHandle matrix)
{
  FIELD *copy = dynamic_cast<FIELD *>(src->clone());
  copy->mesh_detach();
  typename FIELD::mesh_handle_type mesh = copy->get_typed_mesh();

  if (matrix->ncols() < 3)
  {
    reporter->error("Matrix must contain at least 3 columns for position data.");
    return 0;
  }
  if (matrix->ncols() > 3)
  {
    reporter->remark("Matrix contains unused columns, only first three are used.");
  }

  typename FIELD::mesh_type::Node::size_type nsize;
  mesh->size(nsize);
  if (((int)nsize) != matrix->nrows())
  {
    reporter->error("Matrix rows do not fit in this mesh.  May need transpose.");
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

#endif // SwapNodeLocationsWithMatrixEntries_h
