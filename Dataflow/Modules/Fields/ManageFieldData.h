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
#include <Core/Geometry/Tensor.h>


namespace SCIRun {

class ManageFieldDataAlgoField : public DynamicAlgoBase
{
public:
  virtual MatrixHandle execute(FieldHandle src, int &datasize) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *fsrc,
					    int svt_flag);
};


template <class Fld>
class ManageFieldDataAlgoFieldScalar : public ManageFieldDataAlgoField
{
public:
  //! virtual interface. 
  virtual MatrixHandle execute(FieldHandle src, int &datasize);
};

template <class Fld>
MatrixHandle
ManageFieldDataAlgoFieldScalar<Fld>::execute(FieldHandle ifield_h,
					     int &datasize)
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
  datasize = index;

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
  virtual MatrixHandle execute(FieldHandle src, int &datasize);
};

template <class Fld>
MatrixHandle
ManageFieldDataAlgoFieldVector<Fld>::execute(FieldHandle ifield_h,
					     int &datasize)
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
  datasize = index;

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
  virtual MatrixHandle execute(FieldHandle src, int &datasize);
};

template <class Fld>
MatrixHandle
ManageFieldDataAlgoFieldTensor<Fld>::execute(FieldHandle ifield_h,
					     int &datasize)
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
  datasize = index;

  string units;
  if (ifield_h->get_property("units", units))
    omatrix->set_property("units", units, false);
  return MatrixHandle(omatrix);
}



class ManageFieldDataAlgoMesh : public DynamicAlgoBase
{
public:
  virtual FieldHandle execute(ProgressReporter *m,
			      MeshHandle src, MatrixHandle mat) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *fsrc,
					    int svt_flag);
};


template <class FOUT>
class ManageFieldDataAlgoMeshScalar : public ManageFieldDataAlgoMesh
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(ProgressReporter *m,
			      MeshHandle src, MatrixHandle mat);
};


template <class FOUT>
FieldHandle
ManageFieldDataAlgoMeshScalar<FOUT>::execute(ProgressReporter *mod,
					     MeshHandle mesh,
					     MatrixHandle matrix)
{
  typename FOUT::mesh_type *imesh =
    dynamic_cast<typename FOUT::mesh_type *>(mesh.get_rep());
  const unsigned int rows = matrix->nrows();
  const unsigned int columns = matrix->ncols();
  FOUT *ofield;

  imesh->synchronize(Mesh::NODES_E);
  typename FOUT::mesh_type::Node::size_type nsize;
  imesh->size(nsize);
  if (rows && rows == (unsigned int)nsize)
  {
    int index = 0;
    ofield = scinew FOUT(typename FOUT::mesh_handle_type(imesh), Field::NODE);
    typename FOUT::mesh_type::Node::iterator iter; imesh->begin(iter);
    typename FOUT::mesh_type::Node::iterator eiter; imesh->end(eiter);
    while (iter != eiter)
    {
      ofield->set_value(matrix->get(index++, 0), *iter);
      ++iter;
    }
    return FieldHandle(ofield);
  }
  if (columns && columns == (unsigned int)nsize)
  {
    int index = 0;
    ofield = scinew FOUT(typename FOUT::mesh_handle_type(imesh), Field::NODE);
    typename FOUT::mesh_type::Node::iterator iter; imesh->begin(iter);
    typename FOUT::mesh_type::Node::iterator eiter; imesh->end(eiter);
    while (iter != eiter)
    {
      ofield->set_value(matrix->get(0, index++), *iter);
      ++iter;
    }
    return FieldHandle(ofield);
  }

  imesh->synchronize(Mesh::CELLS_E);
  typename FOUT::mesh_type::Cell::size_type csize;
  imesh->size(csize);
  if (rows && rows == (unsigned int)csize)
  {
    int index = 0;
    ofield = scinew FOUT(typename FOUT::mesh_handle_type(imesh), Field::CELL);
    typename FOUT::mesh_type::Cell::iterator iter; imesh->begin(iter);
    typename FOUT::mesh_type::Cell::iterator eiter; imesh->end(eiter);
    while (iter != eiter)
    {
      ofield->set_value(matrix->get(index++, 0), *iter);
      ++iter;
    }
    return FieldHandle(ofield);
  }
  if (columns && columns == (unsigned int)csize)
  {
    int index = 0;
    ofield = scinew FOUT(typename FOUT::mesh_handle_type(imesh), Field::CELL);
    typename FOUT::mesh_type::Cell::iterator iter; imesh->begin(iter);
    typename FOUT::mesh_type::Cell::iterator eiter; imesh->end(eiter);
    while (iter != eiter)
    {
      ofield->set_value(matrix->get(0, index++), *iter);
      ++iter;
    }
    return FieldHandle(ofield);
  }

  imesh->synchronize(Mesh::FACES_E);
  typename FOUT::mesh_type::Face::size_type fsize;
  imesh->size(fsize);
  if (rows && rows == (unsigned int)fsize)
  {
    int index = 0;
    ofield = scinew FOUT(typename FOUT::mesh_handle_type(imesh), Field::FACE);
    typename FOUT::mesh_type::Face::iterator iter; imesh->begin(iter);
    typename FOUT::mesh_type::Face::iterator eiter; imesh->end(eiter);
    while (iter != eiter)
    {
      ofield->set_value(matrix->get(index++, 0), *iter);
      ++iter;
    }
    return FieldHandle(ofield);
  }
  if (columns && columns == (unsigned int)fsize)
  {
    int index = 0;
    ofield = scinew FOUT(typename FOUT::mesh_handle_type(imesh), Field::FACE);
    typename FOUT::mesh_type::Face::iterator iter; imesh->begin(iter);
    typename FOUT::mesh_type::Face::iterator eiter; imesh->end(eiter);
    while (iter != eiter)
    {
      ofield->set_value(matrix->get(0, index++), *iter);
      ++iter;
    }
    return FieldHandle(ofield);
  }

  imesh->synchronize(Mesh::EDGES_E);
  typename FOUT::mesh_type::Edge::size_type esize;
  imesh->size(esize);
  if (rows && rows == (unsigned int)esize)
  {
    int index = 0;
    ofield = scinew FOUT(typename FOUT::mesh_handle_type(imesh), Field::EDGE);
    typename FOUT::mesh_type::Edge::iterator iter; imesh->begin(iter);
    typename FOUT::mesh_type::Edge::iterator eiter; imesh->end(eiter);
    while (iter != eiter)
    {
      ofield->set_value(matrix->get(index++, 0), *iter);
      ++iter;
    }
    return FieldHandle(ofield);
  }
  if (columns && columns == (unsigned int)esize)
  {
    int index = 0;
    ofield = scinew FOUT(typename FOUT::mesh_handle_type(imesh), Field::EDGE);
    typename FOUT::mesh_type::Edge::iterator iter; imesh->begin(iter);
    typename FOUT::mesh_type::Edge::iterator eiter; imesh->end(eiter);
    while (iter != eiter)
    {
      ofield->set_value(matrix->get(0, index++), *iter);
      ++iter;
    }
    return FieldHandle(ofield);
  }

  mod->warning("Matrix datasize does not match field geometry.");
  mod->msgStream() << "Matrix size : " << rows << " " << columns << '\n';
  mod->msgStream() << "Field size : " << nsize << " " <<  esize <<
    " " << fsize << " " << csize << '\n';
  return 0;
}


template <class FOUT>
class ManageFieldDataAlgoMeshVector : public ManageFieldDataAlgoMesh
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(ProgressReporter *m,
			      MeshHandle src, MatrixHandle mat);
};

template <class FOUT>
FieldHandle
ManageFieldDataAlgoMeshVector<FOUT>::execute(ProgressReporter *mod,
					     MeshHandle mesh,
					     MatrixHandle matrix)
{
  typename FOUT::mesh_type *imesh =
    dynamic_cast<typename FOUT::mesh_type *>(mesh.get_rep());
  const unsigned int rows = matrix->nrows();
  const unsigned int columns = matrix->ncols();
  FOUT *ofield;

  imesh->synchronize(Mesh::NODES_E);
  typename FOUT::mesh_type::Node::size_type nsize;
  imesh->size(nsize);
  if (rows && rows == (unsigned int)nsize)
  {
    int index = 0;
    ofield = scinew FOUT(typename FOUT::mesh_handle_type(imesh), Field::NODE);
    typename FOUT::mesh_type::Node::iterator iter; imesh->begin(iter);
    typename FOUT::mesh_type::Node::iterator eiter; imesh->end(eiter);
    while (iter != eiter)
    {
      Vector v(matrix->get(index, 0),
	       matrix->get(index, 1),
	       matrix->get(index, 2));
      ofield->set_value(v, *iter);
      index++;
      ++iter;
    }
    return FieldHandle(ofield);
  }
  if (columns && columns == (unsigned int)nsize)
  {
    int index = 0;
    ofield = scinew FOUT(typename FOUT::mesh_handle_type(imesh), Field::NODE);
    typename FOUT::mesh_type::Node::iterator iter; imesh->begin(iter);
    typename FOUT::mesh_type::Node::iterator eiter; imesh->end(eiter);
    while (iter != eiter)
    {
      Vector v(matrix->get(0, index),
	       matrix->get(1, index),
	       matrix->get(2, index));
      ofield->set_value(v, *iter);
      index++;
      ++iter;
    }
    return FieldHandle(ofield);
  }

  imesh->synchronize(Mesh::CELLS_E);
  typename FOUT::mesh_type::Cell::size_type csize;
  imesh->size(csize);
  if (rows && rows == (unsigned int)csize)
  {
    int index = 0;
    ofield = scinew FOUT(typename FOUT::mesh_handle_type(imesh), Field::CELL);
    typename FOUT::mesh_type::Cell::iterator iter; imesh->begin(iter);
    typename FOUT::mesh_type::Cell::iterator eiter; imesh->end(eiter);
    while (iter != eiter)
    {
      Vector v(matrix->get(index, 0),
	       matrix->get(index, 1),
	       matrix->get(index, 2));
      ofield->set_value(v, *iter);
      index++;
      ++iter;
    }
    return FieldHandle(ofield);
  }
  if (columns && columns == (unsigned int)csize)
  {
    int index = 0;
    ofield = scinew FOUT(typename FOUT::mesh_handle_type(imesh), Field::CELL);
    typename FOUT::mesh_type::Cell::iterator iter; imesh->begin(iter);
    typename FOUT::mesh_type::Cell::iterator eiter; imesh->end(eiter);
    while (iter != eiter)
    {
      Vector v(matrix->get(index, 0),
	       matrix->get(index, 1),
	       matrix->get(index, 2));
      ofield->set_value(v, *iter);
      index++;
      ++iter;
    }
    return FieldHandle(ofield);
  }

  imesh->synchronize(Mesh::FACES_E);
  typename FOUT::mesh_type::Face::size_type fsize;
  imesh->size(fsize);
  if (rows && rows == (unsigned int)fsize)
  {
    int index = 0;
    ofield = scinew FOUT(typename FOUT::mesh_handle_type(imesh), Field::FACE);
    typename FOUT::mesh_type::Face::iterator iter; imesh->begin(iter);
    typename FOUT::mesh_type::Face::iterator eiter; imesh->end(eiter);
    while (iter != eiter)
    {
      Vector v(matrix->get(index, 0),
	       matrix->get(index, 1),
	       matrix->get(index, 2));
      ofield->set_value(v, *iter);
      index++;
      ++iter;
    }
    return FieldHandle(ofield);
  }
  if (columns && columns == (unsigned int)fsize)
  {
    int index = 0;
    ofield = scinew FOUT(typename FOUT::mesh_handle_type(imesh), Field::FACE);
    typename FOUT::mesh_type::Face::iterator iter; imesh->begin(iter);
    typename FOUT::mesh_type::Face::iterator eiter; imesh->end(eiter);
    while (iter != eiter)
    {
      Vector v(matrix->get(index, 0),
	       matrix->get(index, 1),
	       matrix->get(index, 2));
      ofield->set_value(v, *iter);
      index++;
      ++iter;
    }
    return FieldHandle(ofield);
  }

  imesh->synchronize(Mesh::EDGES_E);
  typename FOUT::mesh_type::Edge::size_type esize;
  imesh->size(esize);
  if (rows && rows == (unsigned int)esize)
  {
    int index = 0;
    ofield = scinew FOUT(typename FOUT::mesh_handle_type(imesh), Field::EDGE);
    typename FOUT::mesh_type::Edge::iterator iter; imesh->begin(iter);
    typename FOUT::mesh_type::Edge::iterator eiter; imesh->end(eiter);
    while (iter != eiter)
    {
      Vector v(matrix->get(index, 0),
	       matrix->get(index, 1),
	       matrix->get(index, 2));
      ofield->set_value(v, *iter);
      index++;
      ++iter;
    }
    return FieldHandle(ofield);
  }
  if (columns && columns == (unsigned int)esize)
  {
    int index = 0;
    ofield = scinew FOUT(typename FOUT::mesh_handle_type(imesh), Field::EDGE);
    typename FOUT::mesh_type::Edge::iterator iter; imesh->begin(iter);
    typename FOUT::mesh_type::Edge::iterator eiter; imesh->end(eiter);
    while (iter != eiter)
    {
      Vector v(matrix->get(index, 0),
	       matrix->get(index, 1),
	       matrix->get(index, 2));
      ofield->set_value(v, *iter);
      index++;
      ++iter;
    }
    return FieldHandle(ofield);
  }
  mod->warning("Matrix datasize does not match field geometry.");
  mod->msgStream() << "Matrix size : " << rows << " " << columns << '\n';
  mod->msgStream() << "Field size : " << nsize << " " <<  esize <<
    " " << fsize << " " << csize << '\n';
  return 0;
}


template <class FOUT>
class ManageFieldDataAlgoMeshTensor : public ManageFieldDataAlgoMesh
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(ProgressReporter *m,
			      MeshHandle src, MatrixHandle mat);
};


template <class FOUT>
FieldHandle
ManageFieldDataAlgoMeshTensor<FOUT>::execute(ProgressReporter *mod,
					     MeshHandle mesh,
					     MatrixHandle matrix)
{
  typename FOUT::mesh_type *imesh = 
    dynamic_cast<typename FOUT::mesh_type *>(mesh.get_rep());
  const unsigned int rows = matrix->nrows();
  const unsigned int columns = matrix->ncols();
  FOUT *ofield;

  imesh->synchronize(Mesh::NODES_E);
  typename FOUT::mesh_type::Node::size_type nsize;
  imesh->size(nsize);
  if (rows && rows == (unsigned int)nsize)
  {
    int index = 0;
    ofield = scinew FOUT(typename FOUT::mesh_handle_type(imesh), Field::NODE);
    typename FOUT::mesh_type::Node::iterator iter; imesh->begin(iter);
    typename FOUT::mesh_type::Node::iterator eiter; imesh->end(eiter);
    while (iter != eiter)
    {
      Tensor v;
      v.mat_[0][0] = matrix->get(index, 0);
      v.mat_[0][1] = matrix->get(index, 1);
      v.mat_[0][2] = matrix->get(index, 2);

      v.mat_[1][0] = matrix->get(index, 3);
      v.mat_[1][1] = matrix->get(index, 4);
      v.mat_[1][2] = matrix->get(index, 5);

      v.mat_[2][0] = matrix->get(index, 6);
      v.mat_[2][1] = matrix->get(index, 7);
      v.mat_[2][2] = matrix->get(index, 8);
      ofield->set_value(v, *iter);
      index++;
      ++iter;
    }
    return FieldHandle(ofield);
  }
  if (columns && columns == (unsigned int)nsize)
  {
    int index = 0;
    ofield = scinew FOUT(typename FOUT::mesh_handle_type(imesh), Field::NODE);
    typename FOUT::mesh_type::Node::iterator iter; imesh->begin(iter);
    typename FOUT::mesh_type::Node::iterator eiter; imesh->end(eiter);
    while (iter != eiter)
    {
      Tensor v;
      v.mat_[0][0] = matrix->get(index, 0);
      v.mat_[0][1] = matrix->get(index, 1);
      v.mat_[0][2] = matrix->get(index, 2);

      v.mat_[1][0] = matrix->get(index, 3);
      v.mat_[1][1] = matrix->get(index, 4);
      v.mat_[1][2] = matrix->get(index, 5);

      v.mat_[2][0] = matrix->get(index, 6);
      v.mat_[2][1] = matrix->get(index, 7);
      v.mat_[2][2] = matrix->get(index, 8);
      ofield->set_value(v, *iter);
      index++;
      ++iter;
    }
    return FieldHandle(ofield);
  }

  imesh->synchronize(Mesh::CELLS_E);
  typename FOUT::mesh_type::Cell::size_type csize;
  imesh->size(csize);
  if (rows && rows == (unsigned int)csize)
  {
    int index = 0;
    ofield = scinew FOUT(typename FOUT::mesh_handle_type(imesh), Field::CELL);
    typename FOUT::mesh_type::Cell::iterator iter; imesh->begin(iter);
    typename FOUT::mesh_type::Cell::iterator eiter; imesh->end(eiter);
    while (iter != eiter)
    {
      Tensor v;
      v.mat_[0][0] = matrix->get(index, 0);
      v.mat_[0][1] = matrix->get(index, 1);
      v.mat_[0][2] = matrix->get(index, 2);

      v.mat_[1][0] = matrix->get(index, 3);
      v.mat_[1][1] = matrix->get(index, 4);
      v.mat_[1][2] = matrix->get(index, 5);

      v.mat_[2][0] = matrix->get(index, 6);
      v.mat_[2][1] = matrix->get(index, 7);
      v.mat_[2][2] = matrix->get(index, 8);
      ofield->set_value(v, *iter);
      index++;
      ++iter;
    }
    return FieldHandle(ofield);
  }
  if (columns && columns == (unsigned int)csize)
  {
    int index = 0;
    ofield = scinew FOUT(typename FOUT::mesh_handle_type(imesh), Field::CELL);
    typename FOUT::mesh_type::Cell::iterator iter; imesh->begin(iter);
    typename FOUT::mesh_type::Cell::iterator eiter; imesh->end(eiter);
    while (iter != eiter)
    {
      Tensor v;
      v.mat_[0][0] = matrix->get(index, 0);
      v.mat_[0][1] = matrix->get(index, 1);
      v.mat_[0][2] = matrix->get(index, 2);

      v.mat_[1][0] = matrix->get(index, 3);
      v.mat_[1][1] = matrix->get(index, 4);
      v.mat_[1][2] = matrix->get(index, 5);

      v.mat_[2][0] = matrix->get(index, 6);
      v.mat_[2][1] = matrix->get(index, 7);
      v.mat_[2][2] = matrix->get(index, 8);
      ofield->set_value(v, *iter);
      index++;
      ++iter;
    }
    return FieldHandle(ofield);
  }

  imesh->synchronize(Mesh::FACES_E);
  typename FOUT::mesh_type::Face::size_type fsize;
  imesh->size(fsize);
  if (rows && rows == (unsigned int)fsize)
  {
    int index = 0;
    ofield = scinew FOUT(typename FOUT::mesh_handle_type(imesh), Field::FACE);
    typename FOUT::mesh_type::Face::iterator iter; imesh->begin(iter);
    typename FOUT::mesh_type::Face::iterator eiter; imesh->end(eiter);
    while (iter != eiter)
    {
      Tensor v;
      v.mat_[0][0] = matrix->get(index, 0);
      v.mat_[0][1] = matrix->get(index, 1);
      v.mat_[0][2] = matrix->get(index, 2);

      v.mat_[1][0] = matrix->get(index, 3);
      v.mat_[1][1] = matrix->get(index, 4);
      v.mat_[1][2] = matrix->get(index, 5);

      v.mat_[2][0] = matrix->get(index, 6);
      v.mat_[2][1] = matrix->get(index, 7);
      v.mat_[2][2] = matrix->get(index, 8);
      ofield->set_value(v, *iter);
      index++;
      ++iter;
    }
    return FieldHandle(ofield);
  }
  if (columns && columns == (unsigned int)fsize)
  {
    int index = 0;
    ofield = scinew FOUT(typename FOUT::mesh_handle_type(imesh), Field::FACE);
    typename FOUT::mesh_type::Face::iterator iter; imesh->begin(iter);
    typename FOUT::mesh_type::Face::iterator eiter; imesh->end(eiter);
    while (iter != eiter)
    {
      Tensor v;
      v.mat_[0][0] = matrix->get(index, 0);
      v.mat_[0][1] = matrix->get(index, 1);
      v.mat_[0][2] = matrix->get(index, 2);

      v.mat_[1][0] = matrix->get(index, 3);
      v.mat_[1][1] = matrix->get(index, 4);
      v.mat_[1][2] = matrix->get(index, 5);

      v.mat_[2][0] = matrix->get(index, 6);
      v.mat_[2][1] = matrix->get(index, 7);
      v.mat_[2][2] = matrix->get(index, 8);
      ofield->set_value(v, *iter);
      index++;
      ++iter;
    }
    return FieldHandle(ofield);
  }

  imesh->synchronize(Mesh::EDGES_E);
  typename FOUT::mesh_type::Edge::size_type esize;
  imesh->size(esize);
  if (rows && rows == (unsigned int)esize)
  {
    int index = 0;
    ofield = scinew FOUT(typename FOUT::mesh_handle_type(imesh), Field::EDGE);
    typename FOUT::mesh_type::Edge::iterator iter; imesh->begin(iter);
    typename FOUT::mesh_type::Edge::iterator eiter; imesh->end(eiter);
    while (iter != eiter)
    {
      Tensor v;
      v.mat_[0][0] = matrix->get(index, 0);
      v.mat_[0][1] = matrix->get(index, 1);
      v.mat_[0][2] = matrix->get(index, 2);

      v.mat_[1][0] = matrix->get(index, 3);
      v.mat_[1][1] = matrix->get(index, 4);
      v.mat_[1][2] = matrix->get(index, 5);

      v.mat_[2][0] = matrix->get(index, 6);
      v.mat_[2][1] = matrix->get(index, 7);
      v.mat_[2][2] = matrix->get(index, 8);
      ofield->set_value(v, *iter);
      index++;
      ++iter;
    }
    return FieldHandle(ofield);
  }
  if (columns && columns == (unsigned int)esize)
  {
    int index = 0;
    ofield = scinew FOUT(typename FOUT::mesh_handle_type(imesh), Field::EDGE);
    typename FOUT::mesh_type::Edge::iterator iter; imesh->begin(iter);
    typename FOUT::mesh_type::Edge::iterator eiter; imesh->end(eiter);
    while (iter != eiter)
    {
      Tensor v;
      v.mat_[0][0] = matrix->get(index, 0);
      v.mat_[0][1] = matrix->get(index, 1);
      v.mat_[0][2] = matrix->get(index, 2);

      v.mat_[1][0] = matrix->get(index, 3);
      v.mat_[1][1] = matrix->get(index, 4);
      v.mat_[1][2] = matrix->get(index, 5);

      v.mat_[2][0] = matrix->get(index, 6);
      v.mat_[2][1] = matrix->get(index, 7);
      v.mat_[2][2] = matrix->get(index, 8);
      ofield->set_value(v, *iter);
      index++;
      ++iter;
    }
    return FieldHandle(ofield);
  }
  mod->warning("Matrix datasize does not match field geometry.");
  mod->msgStream() << "Matrix size : " << rows << " " << columns << '\n';
  mod->msgStream() << "Field size : " << nsize << " " <<  esize <<
    " " << fsize << " " << csize << '\n';
  return 0;
}


} // end namespace SCIRun

#endif // ManageFieldData_h
