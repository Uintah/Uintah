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

#include <Core/Disclosure/TypeDescription.h>
#include <Core/Disclosure/DynamicLoader.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Geometry/Tensor.h>


namespace SCIRun {

class ManageFieldDataAlgoField : public DynamicAlgoBase
{
public:
  virtual MatrixHandle execute(FieldHandle src) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfo *get_compile_info(const TypeDescription *fsrc,
				       const TypeDescription *lsrc,
				       int svt_flag);
};


template <class Fld, class Loc>
class ManageFieldDataAlgoFieldScalar : public ManageFieldDataAlgoField
{
public:
  //! virtual interface. 
  virtual MatrixHandle execute(FieldHandle src);
};

template <class Fld, class Loc>
MatrixHandle
ManageFieldDataAlgoFieldScalar<Fld, Loc>::execute(FieldHandle ifield_h)
{
  Fld *ifield = dynamic_cast<Fld *>(ifield_h.get_rep());
  typename Fld::mesh_handle_type mesh = ifield->get_typed_mesh();
  typename Loc::size_type ssize;  mesh->size(ssize);
  ColumnMatrix *omatrix = scinew ColumnMatrix(ssize);
  int index = 0;
  typename Loc::iterator iter; mesh->begin(iter);
  typename Loc::iterator eiter; mesh->end(eiter);
  while (iter != eiter)
  {
    typename Fld::value_type val = ifield->value(*iter);
    omatrix->put(index++, (double)val);
    ++iter;
  }

  return MatrixHandle(omatrix);
}


template <class Fld, class Loc>
class ManageFieldDataAlgoFieldVector : public ManageFieldDataAlgoField
{
public:
  //! virtual interface. 
  virtual MatrixHandle execute(FieldHandle src);
};

template <class Fld, class Loc>
MatrixHandle
ManageFieldDataAlgoFieldVector<Fld, Loc>::execute(FieldHandle ifield_h)
{
  Fld *ifield = dynamic_cast<Fld *>(ifield_h.get_rep());
  typename Fld::mesh_handle_type mesh = ifield->get_typed_mesh();
  typename Loc::size_type ssize;  mesh->size(ssize);
  DenseMatrix *omatrix =  scinew DenseMatrix(ssize, 3);
  int index = 0;
  typename Loc::iterator iter; mesh->begin(iter);
  typename Loc::iterator eiter; mesh->end(eiter);
  while (iter != eiter)
  {
    typename Fld::value_type val = ifield->value(*iter);
    (*omatrix)[index][0]=val.x();
    (*omatrix)[index][1]=val.y();
    (*omatrix)[index][2]=val.z();
    index++;
    ++iter;
  }

  return MatrixHandle(omatrix);
}



template <class Fld, class Loc>
class ManageFieldDataAlgoFieldTensor : public ManageFieldDataAlgoField
{
public:
  //! virtual interface. 
  virtual MatrixHandle execute(FieldHandle src);
};

template <class Fld, class Loc>
MatrixHandle
ManageFieldDataAlgoFieldTensor<Fld, Loc>::execute(FieldHandle ifield_h)
{
  Fld *ifield = dynamic_cast<Fld *>(ifield_h.get_rep());
  typename Fld::mesh_handle_type mesh = ifield->get_typed_mesh();
  typename Loc::size_type ssize;  mesh->size(ssize);
  DenseMatrix *omatrix = scinew DenseMatrix(ssize, 9);
  int index = 0;
  typename Loc::iterator iter; mesh->begin(iter);
  typename Loc::iterator eiter; mesh->end(eiter);
  while (iter != eiter)
  {
    typename Fld::value_type val = ifield->value(*iter);
    (*omatrix)[index][0]=val.mat_[0][0];
    (*omatrix)[index][1]=val.mat_[0][1];
    (*omatrix)[index][2]=val.mat_[0][2];

    (*omatrix)[index][3]=val.mat_[1][0];
    (*omatrix)[index][4]=val.mat_[1][1];
    (*omatrix)[index][5]=val.mat_[1][2];

    (*omatrix)[index][6]=val.mat_[2][0];
    (*omatrix)[index][7]=val.mat_[2][1];
    (*omatrix)[index][8]=val.mat_[2][2];
    index++;
    ++iter;
  }

  return MatrixHandle(omatrix);
}



class ManageFieldDataAlgoMesh : public DynamicAlgoBase
{
public:
  virtual FieldHandle execute(MeshHandle src, MatrixHandle mat) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfo *get_compile_info(const TypeDescription *msrc,
				       const TypeDescription *fsrc,
				       int svt_flag);
};


template <class MSRC, class FOUT>
class ManageFieldDataAlgoMeshScalar : public ManageFieldDataAlgoMesh
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(MeshHandle src, MatrixHandle mat);
};


template <class MSRC, class FOUT>
FieldHandle
ManageFieldDataAlgoMeshScalar<MSRC, FOUT>::execute(MeshHandle mesh,
						   MatrixHandle matrix)
{
  MSRC *imesh = dynamic_cast<MSRC *>(mesh.get_rep());
  const unsigned int rows = matrix->nrows();
  FOUT *ofield;
  typename MSRC::Node::size_type nsize;
  typename MSRC::Node::size_type esize;
  typename MSRC::Node::size_type fsize;
  typename MSRC::Node::size_type csize;
  imesh->size(nsize);
  imesh->size(esize);
  imesh->size(fsize);
  imesh->size(csize);
  if (rows && rows == (unsigned int)nsize)
  {
    int index = 0;
    ofield = scinew FOUT(imesh, Field::NODE);
    typename MSRC::Node::iterator iter; imesh->begin(iter);
    typename MSRC::Node::iterator eiter; imesh->end(eiter);
    while (iter != eiter)
    {
      ofield->set_value(matrix->get(index++, 0), *iter);
      ++iter;
    }
  }
  else if (rows && rows == (unsigned int)esize)
  {
    int index = 0;
    ofield = scinew FOUT(imesh, Field::EDGE);
    typename MSRC::Edge::iterator iter; imesh->begin(iter);
    typename MSRC::Edge::iterator eiter; imesh->end(eiter);
    while (iter != eiter)
    {
      ofield->set_value(matrix->get(index++, 0), *iter);
      ++iter;
    }
  }
  else if (rows && rows == (unsigned int)fsize)
  {
    int index = 0;
    ofield = scinew FOUT(imesh, Field::FACE);
    typename MSRC::Face::iterator iter; imesh->begin(iter);
    typename MSRC::Face::iterator eiter; imesh->end(eiter);
    while (iter != eiter)
    {
      ofield->set_value(matrix->get(index++, 0), *iter);
      ++iter;
    }
  }
  else if (rows && rows == (unsigned int)csize)
  {
    int index = 0;
    ofield = scinew FOUT(imesh, Field::CELL);
    typename MSRC::Cell::iterator iter; imesh->begin(iter);
    typename MSRC::Cell::iterator eiter; imesh->end(eiter);
    while (iter != eiter)
    {
      ofield->set_value(matrix->get(index++, 0), *iter);
      ++iter;
    }
  }
  else
  {
    cout << "Matrix datasize does not match field geometry.";
    return 0;
  }

  return FieldHandle(ofield);
}



template <class MSRC, class FOUT>
class ManageFieldDataAlgoMeshVector : public ManageFieldDataAlgoMesh
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(MeshHandle src, MatrixHandle mat);
};

template <class MSRC, class FOUT>
FieldHandle
ManageFieldDataAlgoMeshVector<MSRC, FOUT>::execute(MeshHandle mesh,
						   MatrixHandle matrix)
{
  MSRC *imesh = dynamic_cast<MSRC *>(mesh.get_rep());
  const unsigned int rows = matrix->nrows();
  FOUT *ofield;
  typename MSRC::Node::size_type nsize;
  typename MSRC::Node::size_type esize;
  typename MSRC::Node::size_type fsize;
  typename MSRC::Node::size_type csize;
  imesh->size(nsize);
  imesh->size(esize);
  imesh->size(fsize);
  imesh->size(csize);
  if (rows && rows == (unsigned int)nsize)
  {
    int index = 0;
    ofield = scinew FOUT(imesh, Field::NODE);
    typename MSRC::Node::iterator iter; imesh->begin(iter);
    typename MSRC::Node::iterator eiter; imesh->end(eiter);
    while (iter != eiter)
    {
      Vector v(matrix->get(index, 0),
	       matrix->get(index, 1),
	       matrix->get(index, 2));
      ofield->set_value(v, *iter);
      index++;
      ++iter;
    }
  }
  else if (rows && rows == (unsigned int)esize)
  {
    int index = 0;
    ofield = scinew FOUT(imesh, Field::EDGE);
    typename MSRC::Edge::iterator iter; imesh->begin(iter);
    typename MSRC::Edge::iterator eiter; imesh->end(eiter);
    while (iter != eiter)
    {
      Vector v(matrix->get(index, 0),
	       matrix->get(index, 1),
	       matrix->get(index, 2));
      ofield->set_value(v, *iter);
      index++;
      ++iter;
    }
  }
  else if (rows && rows == (unsigned int)fsize)
  {
    int index = 0;
    ofield = scinew FOUT(imesh, Field::FACE);
    typename MSRC::Face::iterator iter; imesh->begin(iter);
    typename MSRC::Face::iterator eiter; imesh->end(eiter);
    while (iter != eiter)
    {
      Vector v(matrix->get(index, 0),
	       matrix->get(index, 1),
	       matrix->get(index, 2));
      ofield->set_value(v, *iter);
      index++;
      ++iter;
    }
  }
  else if (rows && rows == (unsigned int)csize)
  {
    int index = 0;
    ofield = scinew FOUT(imesh, Field::CELL);
    typename MSRC::Cell::iterator iter; imesh->begin(iter);
    typename MSRC::Cell::iterator eiter; imesh->end(eiter);
    while (iter != eiter)
    {
      Vector v(matrix->get(index, 0),
	       matrix->get(index, 1),
	       matrix->get(index, 2));
      ofield->set_value(v, *iter);
      index++;
      ++iter;
    }
  }
  else
  {
    cout << "Matrix datasize does not match field geometry.";
    return 0;
  }

  return FieldHandle(ofield);
}



template <class MSRC, class FOUT>
class ManageFieldDataAlgoMeshTensor : public ManageFieldDataAlgoMesh
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(MeshHandle src, MatrixHandle mat);
};


template <class MSRC, class FOUT>
FieldHandle
ManageFieldDataAlgoMeshTensor<MSRC, FOUT>::execute(MeshHandle mesh,
						   MatrixHandle matrix)
{
  MSRC *imesh = dynamic_cast<MSRC *>(mesh.get_rep());
  const unsigned int rows = matrix->nrows();
  FOUT *ofield;
  typename MSRC::Node::size_type nsize;
  typename MSRC::Node::size_type esize;
  typename MSRC::Node::size_type fsize;
  typename MSRC::Node::size_type csize;
  imesh->size(nsize);
  imesh->size(esize);
  imesh->size(fsize);
  imesh->size(csize);
  if (rows && rows == (unsigned int)nsize)
  {
    int index = 0;
    ofield = scinew FOUT(imesh, Field::NODE);
    typename MSRC::Node::iterator iter; imesh->begin(iter);
    typename MSRC::Node::iterator eiter; imesh->end(eiter);
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
  }
  else if (rows && rows == (unsigned int)esize)
  {
    int index = 0;
    ofield = scinew FOUT(imesh, Field::EDGE);
    typename MSRC::Edge::iterator iter; imesh->begin(iter);
    typename MSRC::Edge::iterator eiter; imesh->end(eiter);
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
  }
  else if (rows && rows == (unsigned int)fsize)
  {
    int index = 0;
    ofield = scinew FOUT(imesh, Field::FACE);
    typename MSRC::Face::iterator iter; imesh->begin(iter);
    typename MSRC::Face::iterator eiter; imesh->end(eiter);
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
  }
  else if (rows && rows == (unsigned int)csize)
  {
    int index = 0;
    ofield = scinew FOUT(imesh, Field::CELL);
    typename MSRC::Cell::iterator iter; imesh->begin(iter);
    typename MSRC::Cell::iterator eiter; imesh->end(eiter);
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
  }
  else
  {
    cout << "Matrix datasize does not match field geometry.";
    return 0;
  }

  return FieldHandle(ofield);
}


} // end namespace SCIRun

#endif // ManageFieldData_h
