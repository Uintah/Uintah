/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
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



#ifndef Datatypes_ImageField_h
#define Datatypes_ImageField_h

#include <Core/Datatypes/ImageMesh.h>
#include <Core/Datatypes/GenericField.h>
#include <Core/Geometry/Tensor.h>
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
  iterator end() { return &((*this)(this->dim1()-1,this->dim2()-1))+1; }
  const_iterator begin() const { return &(*this)(0,0); }
  const_iterator end() const { return &((*this)(this->dim1()-1, this->dim2()-1))+1; }

  FData2d() : Array2<Data>() {}
  FData2d(int) : Array2<Data>() {} //default var sgi bug workaround.
  FData2d(const FData2d& data) { Array2<Data>::copy(data); }
  virtual ~FData2d();
  
  const value_type &operator[](const ImageMesh::Cell::index_type &idx) const
  { return this->operator()(0, idx); } 
  const value_type &operator[](const ImageMesh::Face::index_type &idx) const
  { return this->operator()(idx.j_, idx.i_); }
  const value_type &operator[](const ImageMesh::Edge::index_type &idx) const
  { return this->operator()(0, idx); }
  const value_type &operator[](const ImageMesh::Node::index_type &idx) const
  { return this->operator()(idx.j_, idx.i_); }

  value_type &operator[](const ImageMesh::Cell::index_type &idx)
  { return this->operator()(0, idx); } 
  value_type &operator[](const ImageMesh::Face::index_type &idx)
  { return this->operator()(idx.j_, idx.i_); }
  value_type &operator[](const ImageMesh::Edge::index_type &idx)
  { return this->operator()(0, idx); }
  value_type &operator[](const ImageMesh::Node::index_type &idx)
  { return this->operator()(idx.j_, idx.i_); }

  void resize(const ImageMesh::Node::size_type &size)
  { Array2<Data>::resize(size.j_, size.i_); }
  void resize(const ImageMesh::Edge::size_type &size)
  { Array2<Data>::resize(1, size); }
  void resize(const ImageMesh::Face::size_type &size)
  { Array2<Data>::resize(size.j_, size.i_); }
  void resize(const ImageMesh::Cell::size_type &size)
  { Array2<Data>::resize(1, size); }

  unsigned int size() const { return this->dim1() * this->dim2(); }

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
  ImageField(int order);
  ImageField(ImageMeshHandle mesh, int order);
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
ImageField<Data>::ImageField(int order)
  : GenericField<ImageMesh, FData2d<Data> >(order)
{
}


template <class Data>
ImageField<Data>::ImageField(ImageMeshHandle mesh, int order)
  : GenericField<ImageMesh, FData2d<Data> >(mesh, order)
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
    temp.copy(this->fdata());
    this->resize_fdata();
    int i, j;
    for (i=0; i<this->fdata().dim1(); i++)
      for (j=0; j<this->fdata().dim2(); j++)
	this->fdata()(i,j)=temp(j,i);
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

  if (n == -1) {
    static TypeDescription* tdn1 = 0;
    if (tdn1 == 0) {
      const TypeDescription *sub = SCIRun::get_type_description((T*)0);
      TypeDescription::td_vec *subs = scinew TypeDescription::td_vec(1);
      (*subs)[0] = sub;
      tdn1 = scinew TypeDescription(name, subs, path, namesp);
    } 
    td = tdn1;
  }
  else if(n == 0) {
    static TypeDescription* tdn0 = 0;
    if (tdn0 == 0) {
      tdn0 = scinew TypeDescription(name, 0, path, namesp);
    }
    td = tdn0;
  }
  else {
    static TypeDescription* tdnn = 0;
    if (tdnn == 0) {
      tdnn = (TypeDescription *) SCIRun::get_type_description((T*)0);
    }
    td = tdnn;
  }
  return td;
}

} // end namespace SCIRun

#endif // Datatypes_ImageField_h
