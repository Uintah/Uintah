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
#include <string>

namespace SCIRun {

using std::string;


template <class Data>
class FData2d : public Array2<Data> {
public:
  typedef Data value_type;
  typedef Data * iterator;

  Data *begin() { return &(*this)(0,0); }
  Data *end() { return &((*this)(dim1()-1,dim2()-1))+1; }
    
  FData2d():Array2<Data>() {}
  FData2d(const FData2d& data) : Array2<Data>(data) {} 
  virtual ~FData2d(){}
  
  const value_type &operator[](typename ImageMesh::cell_index idx) const 
    { return operator()(idx.i_, 0); } 
  const value_type &operator[](typename ImageMesh::face_index idx) const
    { return operator()(idx.i_, idx.j_); }
  const value_type &operator[](typename ImageMesh::edge_index idx) const 
    { return operator()(idx.i_, 0); }
  const value_type &operator[](typename ImageMesh::node_index idx) const
    { return operator()(idx.i_, idx.j_); }

  value_type &operator[](typename ImageMesh::cell_index idx)
    { return operator()(idx.i_, 0); } 
  value_type &operator[](typename ImageMesh::face_index idx)
    { return operator()(idx.i_, idx.j_); }
  value_type &operator[](typename ImageMesh::edge_index idx)
    { return operator()(idx.i_, 0); }
  value_type &operator[](typename ImageMesh::node_index idx)
    { return operator()(idx.i_, idx.j_); }

  static const string type_name(int n = -1);
  virtual const string get_type_name(int n = -1) const { return type_name(n); }

  void resize(const ImageMesh::node_size_type &size)
    { newsize(size.i_, size.j_); }
  void resize(ImageMesh::edge_size_type) {}
  void resize(const ImageMesh::face_size_type &size)
    { newsize(size.i_, size.j_); }
  void resize(ImageMesh::cell_size_type) {}
};

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

  static const string type_name(int n = -1);
  virtual const string get_type_name(int n = -1) const;
  static PersistentTypeID type_id;
  virtual void io(Piostream &stream);
  bool get_gradient(Vector &, Point &);

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


template <class Data>
const string
ImageField<Data>::get_type_name(int n) const
{
  return type_name(n);
}


#define LATTICEVOL_VERSION 1

template <class Data>
Persistent* 
ImageField<Data>::maker()
{
  return scinew ImageField<Data>;
}

template <class Data>
PersistentTypeID
ImageField<Data>::type_id(type_name(),
		GenericField<ImageMesh, FData2d<Data> >::type_name(),
                maker); 

template <class Data>
void
ImageField<Data>::io(Piostream &stream)
{
  stream.begin_class(type_name(-1), LATTICEVOL_VERSION);
  GenericField<ImageMesh, FData2d<Data> >::io(stream);
  stream.end_class();                                                         
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




//! compute the gradient g, at point p
template <> bool ImageField<Tensor>::get_gradient(Vector &, Point &p);
template <> bool ImageField<Vector>::get_gradient(Vector &, Point &p);

template <class Data>
bool ImageField<Data>::get_gradient(Vector &, Point &)
{
  return false;
}


} // end namespace SCIRun

#endif // Datatypes_ImageField_h
