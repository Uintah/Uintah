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


/*************************************************************
  This field is similar to the SCIRun::LatVolField except for
  it points to an itk::Image< Data, 3 >. 
*************************************************************/

#ifndef Datatypes_ITKImageField_h
#define Datatypes_ITKImageField_h

#include <Core/Datatypes/GenericField.h>
#include <Core/Datatypes/ImageMesh.h>
#include <Core/Geometry/Tensor.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Containers/Array2.h>
#include <Core/Math/MiscMath.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/Assert.h>
#include <string>

#include <itkImage.h>
#include <itkImageRegionIterator.h>
#include <itkImageRegionConstIterator.h>
#include <itkRegion.h>

namespace Insight {

using std::string;
using namespace SCIRun;

template<class T> class ITKFData2d;
template<class T> void Pio(Piostream& stream, ITKFData2d<T>& array);


template <class Data>
class ITKConstIterator2d : public itk::ImageRegionConstIterator<itk::Image<Data,2> > {
public:
  typedef itk::Image<Data, 2> ImageType;
  typedef typename ImageType::RegionType region_type;

  ITKConstIterator2d() : itk::ImageRegionConstIterator< ImageType >() {}
  ITKConstIterator2d(const ImageType* ptr, const region_type &region) : itk::ImageRegionConstIterator< ImageType >(ptr, region) { }
  virtual ~ITKConstIterator2d() {}
  const Data &operator*() { return itk::ImageRegionConstIterator<ImageType>::Value();}
};

template <class Data>
class ITKIterator2d : public ITKConstIterator2d<Data> {
public:
  typedef itk::Image<Data, 2> ImageType;
  typedef typename ImageType::RegionType region_type;

  ITKIterator2d() : ITKConstIterator2d<Data>() { }
  ITKIterator2d(const ImageType* ptr, const region_type &region) : ITKConstIterator2d<Data>(ptr, region) { }
  virtual ~ITKIterator2d() {}
  Data &operator*() { return Value();}
};


template <class Data>
class ITKFData2d {
public:
  typedef Data value_type;
  typedef itk::Image<Data,2> image_type;
  typedef ITKIterator2d<Data> iterator;
  typedef ITKConstIterator2d<Data> const_iterator;

  iterator *begin_;
  iterator *end_;

  const_iterator *const_begin_;
  const_iterator *const_end_;

  const iterator &begin() { 
    if(image_set_)
      return *begin_; 
    else
      ASSERTFAIL("ITKFData2d image not set");
  }
  const iterator &end() { 
    if(image_set_)
      return *end_; 
    else
      ASSERTFAIL("ITKFData2d image not set");
  }

  const const_iterator &begin() const { 
    if(image_set_)
      return *const_begin_; 
    else
      ASSERTFAIL("ITKFData2d image not set");
  }
  const const_iterator &end() const { 
    if(image_set_)
      return *const_end_; 
    else
      ASSERTFAIL("ITKFData2d image not set");
  }

  ITKFData2d();
  ITKFData2d(int);  //default var sgi bug workaround.
  ITKFData2d(const ITKFData2d& data);
  virtual ~ITKFData2d();
  
  const value_type &operator[](typename ImageMesh::Cell::index_type idx) const
  { 
    ASSERTFAIL("No const operator[] for ITKImageField at Cells");
    // check if image is set
  }
  const value_type &operator[](typename ImageMesh::Face::index_type idx) const
  { 
    if(image_set_) {
      typename image_type::IndexType index;
      index[0] = idx.i_;
      index[1] = idx.j_;
      return image_->GetPixel( index );
    }
    else
      ASSERTFAIL("ITKFData2d image not set");
  }
  const value_type &operator[](typename ImageMesh::Edge::index_type idx) const
  { 
    ASSERTFAIL("No const operator[] for ITKImageField at Edges");
    // check if image is set
  }
  const value_type &operator[](typename ImageMesh::Node::index_type idx) const
  { 
    if(image_set_) {
      typename image_type::IndexType index;
      index[0] = idx.i_;
      index[1] = idx.j_;
      return image_->GetPixel( index );
    }
    else
      ASSERTFAIL("ITKFData2d image not set");
  }
  
  value_type &operator[](typename ImageMesh::Cell::index_type idx)
  { 
    ASSERTFAIL("No operator[] for ITKImageField at Cells");
    // check if image is set
  }
  value_type &operator[](typename ImageMesh::Face::index_type idx)
  {
    if(image_set_) {
      typename image_type::IndexType index;
      index[0] = idx.i_;
      index[1] = idx.j_;
      return image_->GetPixel( index );
    }
    else
      ASSERTFAIL("ITKFData2d image not set");      
  }
  value_type &operator[](typename ImageMesh::Edge::index_type idx)
  { 
    ASSERTFAIL("No operator[] for ITKImageField at Edges");
    // check if image is set
  }
  value_type &operator[](typename ImageMesh::Node::index_type idx)
  {
    if(image_set_) {
      typename image_type::IndexType index;
      index[0] = idx.i_;
      index[1] = idx.j_;
      return image_->GetPixel( index );
    }
    else
      ASSERTFAIL("ITKFData2d image not set");
  }    

  // These do not do anything because and itk::Image takes care of
  // allocation.  This field merely points to an itk::Image.
  void resize(const ImageMesh::Node::size_type &size)
  { 
  }
  void resize(const ImageMesh::Edge::size_type &size)
  { 
  }
  void resize(const ImageMesh::Face::size_type &size)
  { 
  }
  void resize(const ImageMesh::Cell::size_type &size)
  { 
  }

  void set_image(itk::Image<Data, 2>* img) {
    image_ = img;

    if ( begin_ ) delete begin_;
    begin_ = new iterator(image_, image_->GetRequestedRegion());
    begin_->GoToBegin();
    if ( end_ ) delete end_;
    end_ = new iterator(image_, image_->GetRequestedRegion());
    end_->GoToEnd();

    if ( const_begin_ ) delete const_begin_;
    const_begin_ = new const_iterator(image_, image_->GetRequestedRegion());
    const_begin_->GoToBegin();
    if ( const_end_ ) delete const_end_;
    const_end_ = new const_iterator(image_, image_->GetRequestedRegion());
    const_end_->GoToEnd();
    
    image_set_ = true;
  }

  unsigned int size() { return dim1() * dim2(); }
  unsigned int dim1();
  unsigned int dim2();

  static const string type_name(int n = -1);

  typename image_type::Pointer get_image() { return image_; }

private:
  bool image_set_;
  typename image_type::Pointer image_;
};


template <class Data>
ITKFData2d<Data>::~ITKFData2d()
{
  if ( begin_ ) delete begin_;
  if ( end_) delete end_;

  if ( const_begin_ ) delete const_begin_;
  if ( const_end_) delete const_end_;
}

template <class Data>
ITKFData2d<Data>::ITKFData2d()
{
  image_set_ = false;
  image_ = image_type::New(); 
  begin_ = 0;
  end_ = 0;
  const_begin_ = 0;
  const_end_ = 0;
}

template <class Data>
ITKFData2d<Data>::ITKFData2d(int a)
{
  image_set_ = false;
  image_ = image_type::New(); 
  begin_ = 0;
  end_ = 0;
  const_begin_ = 0;
  const_end_ = 0;
}

template <class Data>
ITKFData2d<Data>::ITKFData2d(const ITKFData2d& data) {
  image_set_ = false;
  image_ = image_type::New();

  if(dynamic_cast<itk::Image<Data, 2>* >(data.image_.GetPointer() )) {
    set_image(dynamic_cast<itk::Image<Data, 2>* >(data.image_.GetPointer() ));
  }
  else {
    ASSERTFAIL("Copy Constructor for ITKImageField has wrong image type");
  }
  
}

template <class Data>
const string
ITKFData2d<Data>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;
  }
  else if (n == 0)
  {
    return "ITKFData2d";
  }
  else
  {
    return find_type_name((Data *)0);
  }
}

template <class Data>
unsigned int ITKFData2d<Data>::dim1()
{
  if(image_set_)
    return image_->GetLargestPossibleRegion().GetSize()[0];
  else
    ASSERTFAIL("ITKFData2d Image not set");
  return 0;
}

template <class Data>
unsigned int ITKFData2d<Data>::dim2()
{
  if(image_set_)
    return image_->GetLargestPossibleRegion().GetSize()[1];
  else
    ASSERTFAIL("ITKFData2d Image not set");
  return 0;
}

///////////////////////////////////////////////////////
template <class Data>
class ITKImageField : public GenericField< ImageMesh, ITKFData2d<Data> >
{
public:
  ITKImageField();
  ITKImageField(Field::data_location data_at);
  ITKImageField(ImageMeshHandle mesh, Field::data_location data_at);
  ITKImageField(ImageMeshHandle mesh, Field::data_location data_at, itk::Object* img);
  virtual ITKImageField<Data> *clone() const;
  virtual ~ITKImageField();

  void SetImage(itk::Object*);

  //! Persistent IO
  static PersistentTypeID type_id;
  virtual void io(Piostream &stream);

  static const string type_name(int n = -1);
  virtual const TypeDescription* get_type_description(int n = -1) const;


  itk::Object* get_image() { return fdata().get_image(); }

private:
  bool image_set_;
  static Persistent* maker();
};



template <class Data>
ITKImageField<Data>::ITKImageField()
  : GenericField<ImageMesh, ITKFData2d<Data> >()
{
  // need to set image
  image_set_ = false;
}


template <class Data>
ITKImageField<Data>::ITKImageField(Field::data_location data_at)
  : GenericField<ImageMesh, ITKFData2d<Data> >(data_at)
{
  // need to set image
  image_set_ = false;
}


template <class Data>
ITKImageField<Data>::ITKImageField(ImageMeshHandle mesh,
			     Field::data_location data_at)
  : GenericField<ImageMesh, ITKFData2d<Data> >(mesh, data_at)
{
  // need to set image
  image_set_ = false;
}

template <class Data>
ITKImageField<Data>::ITKImageField(ImageMeshHandle mesh,
			     Field::data_location data_at, itk::Object* img)
  : GenericField<ImageMesh, ITKFData2d<Data> >(mesh, data_at)
{
  this->SetImage(img);
}

template <class Data>
void ITKImageField<Data>::SetImage(itk::Object* img)
{
  if(dynamic_cast<itk::Image<Data, 2>* >(img)) {
    fdata().set_image(dynamic_cast<itk::Image<Data, 2>* >(img));
    image_set_ = true;
  }
  else {
    ASSERTFAIL("ITKImageField's SetImage has wrong image type");
  }
}


template <class Data>
ITKImageField<Data> *
ITKImageField<Data>::clone() const
{
  return new ITKImageField(*this);
}
  

template <class Data>
ITKImageField<Data>::~ITKImageField()
{
}


#define IMAGE_FIELD_VERSION 2

template <class Data>
Persistent* 
ITKImageField<Data>::maker()
{
  return scinew ITKImageField<Data>;
}

template <class Data>
PersistentTypeID
ITKImageField<Data>::type_id(type_name(-1),
		GenericField<ImageMesh, ITKFData2d<Data> >::type_name(-1),
                maker); 

template <class Data>
void
ITKImageField<Data>::io(Piostream &stream)
{
  ASSERTFAIL("ITKImageField::io not implemented yet");
  if(image_set_) {  
    /*
      int version = stream.begin_class(type_name(-1), IMAGE_FIELD_VERSION);
      GenericField<ImageMesh, ITKFData2d<Data> >::io(stream); 
      stream.end_class();                                                         
      if (version < 2) { 
      ITKFData2d<Data> temp; 
      temp.copy(fdata()); 
      resize_fdata(); 
      int i, j; 
      for (i=0; i<fdata().dim1(); i++) 
      for (j=0; j<fdata().dim2(); j++) 
      fdata()(i,j)=temp(j,i); 
      }   
    */
  }
  else
    ASSERTFAIL("ITKImageField image not set");
}

template <class Data>
const string
ITKImageField<Data>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;

  }
  else if (n == 0)
  {
    return "ITKImageField";
  }
  else
  {
    return find_type_name((Data *)0);
  }
} 

template <class T> 
const TypeDescription*
ITKImageField<T>::get_type_description(int n) const
{
  ASSERT((n >= -1) && n <= 1);

  TypeDescription* td = 0;
  static string name( type_name(0) );
  static string namesp("Insight");
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

template<class T> 
void Pio(Piostream& stream, ITKFData2d<T>& array)
{
  ASSERTFAIL("Pio not written for ITKImageField"); 
}

} // end namespace Insight

#endif // Datatypes_ITKImageField_h
