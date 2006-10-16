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


//    File   : ManageFieldData.h
//    Author : Michael Callahan
//    Date   : July 2001

#if !defined(ManageFieldData_h)
#define ManageFieldData_h

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Util/ProgressReporter.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Geometry/Tensor.h>


namespace SCIRun {

class ManageFieldDataAlgoField : public DynamicAlgoBase
{
public:
  virtual MatrixHandle execute(FieldHandle src) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *fsrc,
					    int svt_flag);
};


template <class Fld>
class ManageFieldDataAlgoFieldScalar : public ManageFieldDataAlgoField
{
public:
  //! virtual interface. 
  virtual MatrixHandle execute(FieldHandle src);
};

template <class Fld>
MatrixHandle
ManageFieldDataAlgoFieldScalar<Fld>::execute(FieldHandle ifield_h)
{
  Fld *ifield = dynamic_cast<Fld *>(ifield_h.get_rep());

  ColumnMatrix *omatrix = scinew ColumnMatrix(ifield->fdata().size());
  int index = 0;
  typename Fld::fdata_type::iterator iter, eiter;
  iter = ifield->fdata().begin();
  eiter = ifield->fdata().end();

  while (iter != eiter)
  {
    typename Fld::value_type val = *iter;
    omatrix->put(index++, (double)val);
    ++iter;
  }

  string units;
  if (ifield_h->get_property("units", units))
    omatrix->set_property("units", units, false);
  return MatrixHandle(omatrix);
}


template <class Fld>
class ManageFieldDataAlgoFieldVector : public ManageFieldDataAlgoField
{
public:
  //! virtual interface. 
  virtual MatrixHandle execute(FieldHandle src);
};

template <class Fld>
MatrixHandle
ManageFieldDataAlgoFieldVector<Fld>::execute(FieldHandle ifield_h)
{
  Fld *ifield = dynamic_cast<Fld *>(ifield_h.get_rep());

  DenseMatrix *omatrix = scinew DenseMatrix(ifield->fdata().size(), 3);
  int index = 0;
  typename Fld::fdata_type::iterator iter, eiter;
  iter = ifield->fdata().begin();
  eiter = ifield->fdata().end();

  while (iter != eiter)
  {
    const typename Fld::value_type &val = *iter;
    omatrix->put(index, 0, val.x());
    omatrix->put(index, 1, val.y());
    omatrix->put(index, 2, val.z());
    index++;
    ++iter;
  }

  string units;
  if (ifield_h->get_property("units", units))
    omatrix->set_property("units", units, false);
  return MatrixHandle(omatrix);
}



template <class Fld>
class ManageFieldDataAlgoFieldTensor : public ManageFieldDataAlgoField
{
public:
  //! virtual interface. 
  virtual MatrixHandle execute(FieldHandle src);
};

template <class Fld>
MatrixHandle
ManageFieldDataAlgoFieldTensor<Fld>::execute(FieldHandle ifield_h)
{
  Fld *ifield = dynamic_cast<Fld *>(ifield_h.get_rep());

  DenseMatrix *omatrix = scinew DenseMatrix(ifield->fdata().size(), 9);
  int index = 0;
  typename Fld::fdata_type::iterator iter, eiter;
  iter = ifield->fdata().begin();
  eiter = ifield->fdata().end();

  while (iter != eiter)
  {
    const typename Fld::value_type &val = *iter;
    omatrix->put(index, 0, val.mat_[0][0]);
    omatrix->put(index, 1, val.mat_[0][1]);
    omatrix->put(index, 2, val.mat_[0][2]);
    omatrix->put(index, 3, val.mat_[1][0]);
    omatrix->put(index, 4, val.mat_[1][1]);
    omatrix->put(index, 5, val.mat_[1][2]);
    omatrix->put(index, 6, val.mat_[2][0]);
    omatrix->put(index, 7, val.mat_[2][1]);
    omatrix->put(index, 8, val.mat_[2][2]);
    index++;
    ++iter;
  }

  string units;
  if (ifield_h->get_property("units", units))
    omatrix->set_property("units", units, false);
  return MatrixHandle(omatrix);
}



class ManageFieldDataAlgoMesh : public DynamicAlgoBase
{
public:
  virtual FieldHandle execute(ProgressReporter *reporter,
			      MeshHandle src, MatrixHandle mat) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *fsrc,
                                            const string &oftn);
};



template <class VALUE>
void
get_value_row(VALUE &val, Matrix *mat, unsigned int index)
{
  val = mat->get(index, 0);
}

template <>
void
get_value_row(Vector &val, Matrix *mat, unsigned int index)
{
  val.x(mat->get(index, 0));
  val.y(mat->get(index, 1));
  val.z(mat->get(index, 2));
}

template <>
void
get_value_row(Tensor &val, Matrix *mat, unsigned int index)
{
  if (mat->ncols() == 9)
  {
    val.mat_[0][0] = mat->get(index, 0);
    val.mat_[0][1] = mat->get(index, 1);
    val.mat_[0][2] = mat->get(index, 2);

    val.mat_[1][0] = mat->get(index, 3);
    val.mat_[1][1] = mat->get(index, 4);
    val.mat_[1][2] = mat->get(index, 5);

    val.mat_[2][0] = mat->get(index, 6);
    val.mat_[2][1] = mat->get(index, 7);
    val.mat_[2][2] = mat->get(index, 8);
  }
  else
  {
    val.mat_[0][0] = mat->get(index, 0);
    val.mat_[0][1] = mat->get(index, 1);
    val.mat_[0][2] = mat->get(index, 2);

    val.mat_[1][0] = mat->get(index, 1);
    val.mat_[1][1] = mat->get(index, 3);
    val.mat_[1][2] = mat->get(index, 4);

    val.mat_[2][0] = mat->get(index, 2);
    val.mat_[2][1] = mat->get(index, 4);
    val.mat_[2][2] = mat->get(index, 5);
  }
}

template <class VALUE>
void
get_value_col(VALUE &val, Matrix *mat, unsigned int index)
{
  val = mat->get(0, index);
}

template <>
void
get_value_col(Vector &val, Matrix *mat, unsigned int index)
{
  val.x(mat->get(0, index));
  val.y(mat->get(1, index));
  val.z(mat->get(2, index));
}

template <>
void
get_value_col(Tensor &val, Matrix *mat, unsigned int index)
{
  if (mat->nrows() == 9)
  {
    val.mat_[0][0] = mat->get(0, index);
    val.mat_[0][1] = mat->get(1, index);
    val.mat_[0][2] = mat->get(2, index);

    val.mat_[1][0] = mat->get(3, index);
    val.mat_[1][1] = mat->get(4, index);
    val.mat_[1][2] = mat->get(5, index);

    val.mat_[2][0] = mat->get(6, index);
    val.mat_[2][1] = mat->get(7, index);
    val.mat_[2][2] = mat->get(8, index);
  }
  else
  {
    val.mat_[0][0] = mat->get(0, index);
    val.mat_[0][1] = mat->get(1, index);
    val.mat_[0][2] = mat->get(2, index);

    val.mat_[1][0] = mat->get(1, index);
    val.mat_[1][1] = mat->get(3, index);
    val.mat_[1][2] = mat->get(4, index);

    val.mat_[2][0] = mat->get(2, index);
    val.mat_[2][1] = mat->get(4, index);
    val.mat_[2][2] = mat->get(5, index);
  }
}


template <class FOUT>
class ManageFieldDataAlgoMeshT : public ManageFieldDataAlgoMesh
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(ProgressReporter *reporter,
			      MeshHandle src, MatrixHandle mat);

};

template <class FOUT>
FieldHandle
ManageFieldDataAlgoMeshT<FOUT>::execute(ProgressReporter *reporter,
                                        MeshHandle mesh,
                                        MatrixHandle matrix)
{
  typename FOUT::mesh_type *imesh =
    dynamic_cast<typename FOUT::mesh_type *>(mesh.get_rep());
  const unsigned int rows = matrix->nrows();
  const unsigned int columns = matrix->ncols();
  FOUT *ofield;

  //imesh->synchronize(Mesh::NODES_E);
  typename FOUT::mesh_type::Node::size_type nsize;
  imesh->size(nsize);
  if (rows && rows == (unsigned int)nsize)
  {
    int index = 0;
    ofield = scinew FOUT(typename FOUT::mesh_handle_type(imesh)); //, 1);
    typename FOUT::mesh_type::Node::iterator iter; imesh->begin(iter);
    typename FOUT::mesh_type::Node::iterator eiter; imesh->end(eiter);
    while (iter != eiter)
    {
      typename FOUT::value_type v;
      get_value_row(v, matrix.get_rep(), index);
      ofield->set_value(v, *iter);
      index++;
      ++iter;
    }
    return FieldHandle(ofield);
  }
  if (columns && columns == (unsigned int)nsize)
  {
    int index = 0;
    ofield = scinew FOUT(typename FOUT::mesh_handle_type(imesh)); //, 1);
    typename FOUT::mesh_type::Node::iterator iter; imesh->begin(iter);
    typename FOUT::mesh_type::Node::iterator eiter; imesh->end(eiter);
    while (iter != eiter)
    {
      typename FOUT::value_type v;
      get_value_col(v, matrix.get_rep(), index);
      ofield->set_value(v, *iter);
      index++;
      ++iter;
    }
    return FieldHandle(ofield);
  }

  //  imesh->synchronize(Mesh::ELEMS_E);
  typename FOUT::mesh_type::Elem::size_type csize;
  imesh->size(csize);
  if (rows && rows == (unsigned int)csize)
  {
    int index = 0;
    ofield = scinew FOUT(typename FOUT::mesh_handle_type(imesh)); //, 0);
    typename FOUT::mesh_type::Elem::iterator iter; imesh->begin(iter);
    typename FOUT::mesh_type::Elem::iterator eiter; imesh->end(eiter);
    while (iter != eiter)
    {
      typename FOUT::value_type v;
      get_value_row(v, matrix.get_rep(), index);
      ofield->set_value(v, *iter);
      index++;
      ++iter;
    }
    return FieldHandle(ofield);
  }
  if (columns && columns == (unsigned int)csize)
  {
    int index = 0;
    ofield = scinew FOUT(typename FOUT::mesh_handle_type(imesh)); //, 0);
    typename FOUT::mesh_type::Elem::iterator iter; imesh->begin(iter);
    typename FOUT::mesh_type::Elem::iterator eiter; imesh->end(eiter);
    while (iter != eiter)
    {
      typename FOUT::value_type v;
      get_value_col(v, matrix.get_rep(), index);
      ofield->set_value(v, *iter);
      index++;
      ++iter;
    }
    return FieldHandle(ofield);
  }

  reporter->warning("Matrix datasize does not match field geometry.");
  reporter->remark("Matrix size : " + to_string(rows) +
                   " " + to_string(columns));
  reporter->remark("Field size : " + to_string(nsize) +
                   " " + to_string(csize));
  return 0;
}


} // end namespace SCIRun

#endif // ManageFieldData_h
