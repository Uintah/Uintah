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


#ifndef Datatypes_MaskedLatVolField_h
#define Datatypes_MaskedLatVolField_h

#include <Core/Datatypes/GenericField.h>
#include <Core/Datatypes/MaskedLatVolMesh.h>
#include <Core/Geometry/Tensor.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Containers/Array3.h>
#include <Core/Math/MiscMath.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/Assert.h>
#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {

using std::string;

template <class Data>
class MFData3d : public Array3<Data> {
public:
  typedef Data value_type;
  typedef Data * iterator;
  typedef const Data * const_iterator;

  iterator begin() { return &(*this)(0,0,0); } 
  iterator end() { return &((*this)(dim1()-1,dim2()-1,dim3()-1))+1; }
  const_iterator begin() const { return &(*this)(0,0,0); } 
  const_iterator end() const { return &((*this)(dim1()-1,dim2()-1,dim3()-1))+1; }

    
  MFData3d() : Array3<Data>() {}
  MFData3d(int) : Array3<Data>() {} //default arg sgi bug workaround.
  MFData3d(const MFData3d& data) {Array3<Data>::copy(data); }
  virtual ~MFData3d();
  
  const value_type &operator[](typename MaskedLatVolMesh::Cell::index_type idx) const
  { return operator()(idx.k_,idx.j_,idx.i_); } 
  const value_type &operator[](typename MaskedLatVolMesh::Face::index_type idx) const
  { return operator()(0, 0, unsigned(idx)); }
  const value_type &operator[](typename MaskedLatVolMesh::Edge::index_type idx) const
  { return operator()(0, 0, unsigned(idx)); }    
  const value_type &operator[](typename MaskedLatVolMesh::Node::index_type idx) const
  { return operator()(idx.k_,idx.j_,idx.i_); }    

  value_type &operator[](typename MaskedLatVolMesh::Cell::index_type idx)
  { return operator()(idx.k_,idx.j_,idx.i_); } 
  value_type &operator[](typename MaskedLatVolMesh::Face::index_type idx)
  { return operator()(0, 0, unsigned(idx)); }
  value_type &operator[](typename MaskedLatVolMesh::Edge::index_type idx)
  { return operator()(0, 0, unsigned(idx)); }    
  value_type &operator[](typename MaskedLatVolMesh::Node::index_type idx)
  { return operator()(idx.k_,idx.j_,idx.i_); }    

  void resize(const MaskedLatVolMesh::Node::size_type &size)
  { Array3<Data>::resize(size.k_, size.j_, size.i_); }
  void resize(const MaskedLatVolMesh::Edge::size_type &size)
  { Array3<Data>::resize(1, 1, unsigned(size)); }
  void resize(const MaskedLatVolMesh::Face::size_type &size)
  { Array3<Data>::resize(1, 1, size); }
  void resize(const MaskedLatVolMesh::Cell::size_type &size)
  { Array3<Data>::resize(size.k_, size.j_, size.i_); }

  static const string type_name(int n = -1);
};


template <class Data>
MFData3d<Data>::~MFData3d()
{
}

  
template <class Data>
const string
MFData3d<Data>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;
  }
  else if (n == 0)
  {
    return "MFData3d";
  }
  else
  {
    return find_type_name((Data *)0);
  }
}


template <class Data>
class MaskedLatVolField : public GenericField< MaskedLatVolMesh, MFData3d<Data> >
{
public:
  // Avoids a warning with g++ 3.1
  // ../src/Core/Datatypes/QuadraticTetVolField.h:95: warning: `typename 
  // SCIRun::QuadraticTetVolField<T>::mesh_handle_type' is implicitly a typename
  // ../src/Core/Datatypes/QuadraticTetVolField.h:95: warning: implicit typename is 
  // deprecated, please see the documentation for details
  typedef typename GenericField<MaskedLatVolMesh, MFData3d<Data> >::mesh_handle_type mesh_handle_type;
  MaskedLatVolField();
  MaskedLatVolField(Field::data_location data_at);
  MaskedLatVolField(MaskedLatVolMeshHandle mesh, Field::data_location data_at);
  virtual MaskedLatVolField<Data> *clone() const;
  virtual ~MaskedLatVolField();

  //! Persistent IO
  static PersistentTypeID type_id;
  virtual void io(Piostream &stream);

  static const string type_name(int n = -1);
  virtual const TypeDescription* get_type_description(int n = -1) const;

private:
  static Persistent* maker();
};



template <class Data>
MaskedLatVolField<Data>::MaskedLatVolField()
  : GenericField<MaskedLatVolMesh, MFData3d<Data> >()
{
}


template <class Data>
MaskedLatVolField<Data>::MaskedLatVolField(Field::data_location data_at)
  : GenericField<MaskedLatVolMesh, MFData3d<Data> >(data_at)
{
}


template <class Data>
MaskedLatVolField<Data>::MaskedLatVolField(MaskedLatVolMeshHandle mesh,
			     Field::data_location data_at)
  : GenericField<MaskedLatVolMesh, MFData3d<Data> >(mesh, data_at)
{
}


template <class Data>
MaskedLatVolField<Data> *
MaskedLatVolField<Data>::clone() const
{
  return new MaskedLatVolField<Data>(*this);
}
  

template <class Data>
MaskedLatVolField<Data>::~MaskedLatVolField()
{
}


template <class Data>
const string
MaskedLatVolField<Data>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;

  }
  else if (n == 0)
  {
    return "MaskedLatVolField";
  }
  else
  {
    return find_type_name((Data *)0);
  }
} 

template <class T> 
const TypeDescription*
MaskedLatVolField<T>::get_type_description(int n) const
{
  ASSERT((n >= -1) && n <= 1);

  TypeDescription* td = 0;
  static string name( type_name(0) );
  static string namesp("SCIRun");
  static string path(__FILE__);

  if(!td){
    if (n == -1) {
      const TypeDescription *sub = SCIRun::get_type_description((T*)0);
      TypeDescription::td_vec *subs = scinew TypeDescription::td_vec(1);
      (*subs)[0] = sub;
      td = scinew TypeDescription(name, subs, path, namesp);
    }
    else if(n == 0) {
      td = scinew TypeDescription(name, 0, path, namesp);
    }
    else {
      td = (TypeDescription *) SCIRun::get_type_description((T*)0);
    }
  }
  return td;
}

#define LAT_VOL_FIELD_VERSION 3

template <class Data>
Persistent* 
MaskedLatVolField<Data>::maker()
{
  return scinew MaskedLatVolField<Data>;
}

template <class Data>
PersistentTypeID
MaskedLatVolField<Data>::type_id(type_name(-1),
		GenericField<MaskedLatVolMesh, MFData3d<Data> >::type_name(-1),
                maker); 

template <class Data>
void
MaskedLatVolField<Data>::io(Piostream &stream)
{
  int version = stream.begin_class(type_name(-1), LAT_VOL_FIELD_VERSION);
  GenericField<MaskedLatVolMesh, MFData3d<Data> >::io(stream);
  stream.end_class();                                                    
  if (version < 2) {
    MFData3d<Data> temp;
    temp.copy(fdata());
    resize_fdata();
    int i, j, k;
    for (i=0; i<fdata().dim1(); i++)
      for (j=0; j<fdata().dim2(); j++)
	for (k=0; k<fdata().dim3(); k++)
	  fdata()(i,j,k)=temp(k,j,i);
  }
}


} // end namespace SCIRun

#endif // Datatypes_MaskedLatVolField_h
