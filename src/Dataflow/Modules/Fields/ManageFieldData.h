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
  ColumnMatrix *omatrix =
    scinew ColumnMatrix(mesh->tsize((typename Loc::size_type *)0));
  int index = 0;
  typename Loc::iterator iter = mesh->tbegin((typename Loc::iterator *)0);
  typename Loc::iterator eiter = mesh->tend((typename Loc::iterator *)0);
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
  ColumnMatrix *omatrix =
    scinew ColumnMatrix(mesh->tsize((typename Loc::size_type *)0));
  int index = 0;
  typename Loc::iterator iter = mesh->tbegin((typename Loc::iterator *)0);
  typename Loc::iterator eiter = mesh->tend((typename Loc::iterator *)0);
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
  ColumnMatrix *omatrix =
    scinew ColumnMatrix(mesh->tsize((typename Loc::size_type *)0));
  int index = 0;
  typename Loc::iterator iter = mesh->tbegin((typename Loc::iterator *)0);
  typename Loc::iterator eiter = mesh->tend((typename Loc::iterator *)0);
  while (iter != eiter)
  {
    typename Fld::value_type val = ifield->value(*iter);
    (*omatrix)[index][0]=val.mat_[0][0];
    (*omatrix)[index][1]=val.mat_[0][1];
    (*omatrix)[index][2]=val.mat_[0][2];;

    (*omatrix)[index][3]=val.mat_[1][0];;
    (*omatrix)[index][4]=val.mat_[1][1];;
    (*omatrix)[index][5]=val.mat_[1][2];;

    (*omatrix)[index][6]=val.mat_[2][0];;
    (*omatrix)[index][7]=val.mat_[2][1];;
    (*omatrix)[index][8]=val.mat_[2][2];;
    index++;
    ++iter;
  }

  return MatrixHandle(omatrix);
}



} // end namespace SCIRun

#endif // ManageFieldData_h
