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


#ifndef Datatypes_ImageField_h
#define Datatypes_ImageField_h

#include <Core/Datatypes/GenericField.h>
#include <Core/Datatypes/ImageMesh.h>
#include <Core/Geometry/Tensor.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Containers/Array2.h>
#include <Core/Math/MiscMath.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/Assert.h>
#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {

using std::string;


template <class Data>
class FData2d : public Array2<Data> {
public:
  typedef Data value_type;
  typedef Data * iterator;
  typedef Data const * const_iterator;

  iterator begin() { return &(*this)(0,0); }
  iterator end() { return &((*this)(dim1()-1,dim2()-1))+1; }
  const_iterator begin() const { return &(*this)(0,0); }
  const_iterator end() const { return &((*this)(dim1()-1,dim2()-1))+1; }

  FData2d() : Array2<Data>() {}
  FData2d(int) : Array2<Data>() {} //default var sgi bug workaround.
  FData2d(const FData2d& data) { Array2<Data>::copy(data); }
  virtual ~FData2d();
  
  const value_type &operator[](typename ImageMesh::Cell::index_type idx) const
  { return operator()(0, idx); } 
  const value_type &operator[](typename ImageMesh::Face::index_type idx) const
  { return operator()(idx.j_, idx.i_); }
  const value_type &operator[](typename ImageMesh::Edge::index_type idx) const
  { return operator()(0, idx); }
  const value_type &operator[](typename ImageMesh::Node::index_type idx) const
  { return operator()(idx.j_, idx.i_); }

  value_type &operator[](typename ImageMesh::Cell::index_type idx)
  { return operator()(0, idx); } 
  value_type &operator[](typename ImageMesh::Face::index_type idx)
  { return operator()(idx.j_, idx.i_); }
  value_type &operator[](typename ImageMesh::Edge::index_type idx)
  { return operator()(0, idx); }
  value_type &operator[](typename ImageMesh::Node::index_type idx)
  { return operator()(idx.j_, idx.i_); }

  void resize(const ImageMesh::Node::size_type &size)
  { Array2<Data>::resize(size.j_, size.i_); }
  void resize(const ImageMesh::Edge::size_type &size)
  { Array2<Data>::resize(1, size); }
  void resize(const ImageMesh::Face::size_type &size)
  { Array2<Data>::resize(size.j_, size.i_); }
  void resize(const ImageMesh::Cell::size_type &size)
  { Array2<Data>::resize(1, size); }

  unsigned int size() { return dim1() * dim2(); }

  static const string type_name(int n = -1);
};


template <class Data>
FData2d<Data>::~FData2d()
{
}


template <class Data>
const string
FData2d<Data>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;
  }
  else if (n == 0)
  {
    return "FData2d";
  }
  else
  {
    return find_type_name((Data *)0);
  }
}


template <class Data>
class ImageField : public GenericField< ImageMesh, FData2d<Data> >
{
public:
  ImageField();
  ImageField(Field::data_location data_at);
  ImageField(ImageMeshHandle mesh, Field::data_location data_at);
  virtual ImageField<Data> *clone() const;
  virtual ~ImageField();

  //! Persistent IO
  static PersistentTypeID type_id;
  virtual void io(Piostream &stream);

  static const string type_name(int n = -1);
  virtual const TypeDescription* get_type_description(int n = -1) const;

private:
  static Persistent* maker();
};



template <class Data>
ImageField<Data>::ImageField()
  : GenericField<ImageMesh, FData2d<Data> >()
{
}


template <class Data>
ImageField<Data>::ImageField(Field::data_location data_at)
  : GenericField<ImageMesh, FData2d<Data> >(data_at)
{
}


template <class Data>
ImageField<Data>::ImageField(ImageMeshHandle mesh,
			     Field::data_location data_at)
  : GenericField<ImageMesh, FData2d<Data> >(mesh, data_at)
{
}


template <class Data>
ImageField<Data> *
ImageField<Data>::clone() const
{
  return new ImageField(*this);
}
  

template <class Data>
ImageField<Data>::~ImageField()
{
}


#define IMAGE_FIELD_VERSION 2

template <class Data>
Persistent* 
ImageField<Data>::maker()
{
  return scinew ImageField<Data>;
}

template <class Data>
PersistentTypeID
ImageField<Data>::type_id(type_name(-1),
		GenericField<ImageMesh, FData2d<Data> >::type_name(-1),
                maker); 

template <class Data>
void
ImageField<Data>::io(Piostream &stream)
{
  int version = stream.begin_class(type_name(-1), IMAGE_FIELD_VERSION);
  GenericField<ImageMesh, FData2d<Data> >::io(stream);
  stream.end_class();                                                         
  if (version < 2) {
    FData2d<Data> temp;
    temp.copy(fdata());
    resize_fdata();
    int i, j;
    for (i=0; i<fdata().dim1(); i++)
      for (j=0; j<fdata().dim2(); j++)
	fdata()(i,j)=temp(j,i);
  }  
}

template <class Data>
const string
ImageField<Data>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;

  }
  else if (n == 0)
  {
    return "ImageField";
  }
  else
  {
    return find_type_name((Data *)0);
  }
} 

template <class T> 
const TypeDescription*
ImageField<T>::get_type_description(int n) const
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

} // end namespace SCIRun

#endif // Datatypes_ImageField_h
