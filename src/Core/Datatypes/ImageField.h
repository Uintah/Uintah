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
  FData2d(int):Array2<Data>() {}
  FData2d(const FData2d& data) {copy(data);} 
  virtual ~FData2d(){}
  
  const value_type &operator[](typename ImageMesh::Cell::index_type idx) const 
    { return operator()(0, idx.i_); } 
  const value_type &operator[](typename ImageMesh::Face::index_type idx) const
    { return operator()(idx.j_, idx.i_); }
  const value_type &operator[](typename ImageMesh::Edge::index_type idx) const 
    { return operator()(0, idx.i_); }
  const value_type &operator[](typename ImageMesh::Node::index_type idx) const
    { return operator()(idx.j_, idx.i_); }

  value_type &operator[](typename ImageMesh::Cell::index_type idx)
    { return operator()(0, idx.i_); } 
  value_type &operator[](typename ImageMesh::Face::index_type idx)
    { return operator()(idx.j_, idx.i_); }
  value_type &operator[](typename ImageMesh::Edge::index_type idx)
    { return operator()(0, idx.i_); }
  value_type &operator[](typename ImageMesh::Node::index_type idx)
    { return operator()(idx.j_, idx.i_); }

  static const string type_name(int n = -1);
  virtual const string get_type_name(int n = -1) const { return type_name(n); }

  void resize(const ImageMesh::Node::size_type &size)
    { newsize(size.j_, size.i_); }
  void resize(ImageMesh::Edge::size_type) {}
  void resize(const ImageMesh::Face::size_type &size)
    { newsize(size.j_, size.i_); }
  void resize(ImageMesh::Cell::size_type) {}
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

  virtual ScalarFieldInterface* query_scalar_interface() const;
  virtual VectorFieldInterface* query_vector_interface() const;
  virtual TensorFieldInterface* query_tensor_interface() const;

  static const string type_name(int n = -1);
  virtual const string get_type_name(int n = -1) const;
  static PersistentTypeID type_id;
  virtual void io(Piostream &stream);
  virtual const TypeDescription* get_type_description() const;

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


template <> ScalarFieldInterface *
ImageField<double>::query_scalar_interface() const;

template <> ScalarFieldInterface *
ImageField<float>::query_scalar_interface() const;

template <> ScalarFieldInterface *
ImageField<int>::query_scalar_interface() const;

template <> ScalarFieldInterface*
ImageField<short>::query_scalar_interface() const;

template <> ScalarFieldInterface*
ImageField<char>::query_scalar_interface() const;

template <> ScalarFieldInterface *
ImageField<unsigned int>::query_scalar_interface() const;

template <> ScalarFieldInterface*
ImageField<unsigned short>::query_scalar_interface() const;

template <> ScalarFieldInterface*
ImageField<unsigned char>::query_scalar_interface() const;

template <class T>
ScalarFieldInterface*
ImageField<T>::query_scalar_interface() const 
{
  return 0;
}

template <>
VectorFieldInterface*
ImageField<Vector>::query_vector_interface() const;

template <class T>
VectorFieldInterface*
ImageField<T>::query_vector_interface() const
{
  return 0;
}

template <>
TensorFieldInterface*
ImageField<Tensor>::query_tensor_interface() const;

template <class T>
TensorFieldInterface*
ImageField<T>::query_tensor_interface() const
{
  return 0;
}


template <class Data>
const string
ImageField<Data>::get_type_name(int n) const
{
  return type_name(n);
}


#define IMAGEFIELD_VERSION 2

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
  int version = stream.begin_class(type_name(-1), IMAGEFIELD_VERSION);
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

template <class T>
const TypeDescription* 
get_type_description(ImageField<T>*)
{
  static TypeDescription* td = 0;
  static string name("ImageField");
  static string namesp("SCIRun");
  static string path(__FILE__);
  if(!td){
    const TypeDescription *sub = SCIRun::get_type_description((T*)0);
    TypeDescription::td_vec *subs = scinew TypeDescription::td_vec(1);
    (*subs)[0] = sub;
    td = scinew TypeDescription(name, subs, path, namesp);
  }
  return td;
}

template <class T>
const TypeDescription* 
ImageField<T>::get_type_description() const 
{
  return SCIRun::get_type_description((ImageField<T>*)0);
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

} // end namespace SCIRun

#endif // Datatypes_ImageField_h
