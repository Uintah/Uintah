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



/*************************************************************
  This field is similar to the SCIRun::LatVolField except for
  it points to an itk::Image< Data, 3 >. 
*************************************************************/

#ifndef Datatypes_ITKImageField_h
#define Datatypes_ITKImageField_h

#include <Core/Util/TypeDescription.h>
#include <Core/Geometry/Tensor.h>
#include <Core/Geometry/Vector.h>


// To make SCIRun compile on OSX 10.4, certain classes need to be defined before
// including GenericField.h, if not SCIRun does not compile.
// Apparently a templated class needs to have its templated datatypes defined before
// it is defined itself. 
// I just put the missing ones here, to make the Insight Package compile properly.
// -- jeroen

#include <Core/Datatypes/share.h>
namespace SCIRun {

template <class Data> class ITKFData2d;
template <class Data> class ITKImageField;

}

namespace SCIRun {

  SCISHARE const TypeDescription* get_type_description(SCIRun::ITKFData2d<SCIRun::Tensor>*);
  SCISHARE const TypeDescription* get_type_description(SCIRun::ITKFData2d<SCIRun::Vector>*);
  SCISHARE const TypeDescription* get_type_description(SCIRun::ITKFData2d<double>*);
  SCISHARE const TypeDescription* get_type_description(SCIRun::ITKFData2d<float>*);
  SCISHARE const TypeDescription* get_type_description(SCIRun::ITKFData2d<int>*);
  SCISHARE const TypeDescription* get_type_description(SCIRun::ITKFData2d<long>*);
  SCISHARE const TypeDescription* get_type_description(SCIRun::ITKFData2d<short>*);
  SCISHARE const TypeDescription* get_type_description(SCIRun::ITKFData2d<char>*);
  SCISHARE const TypeDescription* get_type_description(SCIRun::ITKFData2d<unsigned int>*);
  SCISHARE const TypeDescription* get_type_description(SCIRun::ITKFData2d<unsigned short>*);
  SCISHARE const TypeDescription* get_type_description(SCIRun::ITKFData2d<unsigned char>*);
  SCISHARE const TypeDescription* get_type_description(SCIRun::ITKFData2d<unsigned long>*);

}

#include <Core/Containers/LockingHandle.h>
#include <Core/Basis/QuadBilinearLgn.h>
#include <Core/Datatypes/ImageMesh.h>
#include <Core/Containers/FData.h>
#include <Core/Datatypes/GenericField.h>

#include <Core/Math/MiscMath.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/Assert.h>
#include <string>

#include <itkImage.h>
#include <itkImageRegionIterator.h>
#include <itkImageRegionConstIterator.h>
#include <itkRegion.h>

#include <Core/Geometry/BBox.h>

namespace SCIRun {

using std::string;

typedef ImageMesh<QuadBilinearLgn<Point> > IMesh_;

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
  Data &operator*() { return this->Value();}
};


template <class Data>
class ITKFData2d {
public:
  typedef Data value_type;
  typedef itk::Image<Data,2> image_type;
  typedef ITKIterator2d<Data> iterator;
  typedef ITKConstIterator2d<Data> const_iterator;
  //  typedef ImageMesh<QuadBilinearLgn<Data> > IMesh_;

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
  
  const value_type &operator[](typename IMesh_::Cell::index_type idx) const
  { 
    ASSERTFAIL("No const operator[] for ITKImageField at Cells");
    // check if image is set
  }
  const value_type &operator[](typename IMesh_::Face::index_type idx) const
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
  const value_type &operator[](typename IMesh_::Edge::index_type idx) const
  { 
    ASSERTFAIL("No const operator[] for ITKImageField at Edges");
    // check if image is set
  }
  const value_type &operator[](typename IMesh_::Node::index_type idx) const
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
  
  value_type &operator[](typename IMesh_::Cell::index_type idx)
  { 
    ASSERTFAIL("No operator[] for ITKImageField at Cells");
    // check if image is set
  }
  value_type &operator[](typename IMesh_::Face::index_type idx)
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
  value_type &operator[](typename IMesh_::Edge::index_type idx)
  { 
    ASSERTFAIL("No operator[] for ITKImageField at Edges");
    // check if image is set
  }
  value_type &operator[](typename IMesh_::Node::index_type idx)
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
  void resize(const typename IMesh_::Node::size_type &size)
  { 
  }
  void resize(const typename IMesh_::Edge::size_type &size)
  { 
  }
  void resize(const typename IMesh_::Face::size_type &size)
  { 
  }
  void resize(const typename IMesh_::Cell::size_type &size)
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

  void set_image_origin(double * origin) {
    image_->SetOrigin( origin );
  }

  void set_image_spacing(double * spacing) {
    image_->SetSpacing( spacing );
  }

  unsigned int size() const { return dim1() * dim2(); }
  unsigned int dim1() const;
  unsigned int dim2() const;

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
unsigned int ITKFData2d<Data>::dim1() const
{
  if(image_set_)
    return image_->GetLargestPossibleRegion().GetSize()[0];
  else
    ASSERTFAIL("ITKFData2d Image not set");
  return 0;
}

template <class Data>
unsigned int ITKFData2d<Data>::dim2() const
{
  if(image_set_)
    return image_->GetLargestPossibleRegion().GetSize()[1];
  else
    ASSERTFAIL("ITKFData2d Image not set");
  return 0;
}

///////////////////////////////////////////////////////
template <class Data>
class ITKImageField : public GenericField<IMesh_, QuadBilinearLgn<Data>, ITKFData2d<Data> >
{
public:
  typedef QuadBilinearLgn<Data> ITKIBasis;
  typedef GenericField<IMesh_, ITKIBasis, ITKFData2d<Data> > ITKIF;

  ITKImageField() :
    GenericField<IMesh_, QuadBilinearLgn<Data>, ITKFData2d<Data> >() {
    image_set_ = false;
  } 
  ITKImageField(typename ITKIF::mesh_handle_type mesh) :
    GenericField<IMesh_, QuadBilinearLgn<Data>, ITKFData2d<Data> >(mesh) {
    image_set_ = false;
  } 

  ITKImageField(typename ITKIF::mesh_handle_type mesh, itk::Object* img)  :
    GenericField<IMesh_, QuadBilinearLgn<Data>, ITKFData2d<Data> >(mesh) {
    this->SetImage(img);
  } 
  virtual ITKImageField<Data> *clone() const;
  virtual ~ITKImageField();

  void SetImage(itk::Object*);

  //! Persistent IO
  static PersistentTypeID type_id;
  virtual void io(Piostream &stream);

  static const string type_name(int n = -1);
  virtual const TypeDescription* get_type_description(int n = -1) const;


  itk::Object* get_image() { return this->fdata().get_image(); }

private:
  bool image_set_;
  static Persistent* maker();
};



template <class Data>
void ITKImageField<Data>::SetImage(itk::Object* img)
{
  if(dynamic_cast<itk::Image<Data, 2>* >(img)) {
    this->fdata().set_image(dynamic_cast<itk::Image<Data, 2>* >(img));
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


#define ITK_IMAGE_FIELD_VERSION 1

template <class Data>
Persistent* 
ITKImageField<Data>::maker()
{
  return scinew ITKImageField<Data>;
}

template <class Data>
PersistentTypeID
ITKImageField<Data>::type_id(type_name(-1),
			     GenericField<IMesh_, QuadBilinearLgn<Data>, 
			     ITKFData2d<Data> >::type_name(-1),
                maker); 

template <class Data>
void
ITKImageField<Data>::io(Piostream &stream)
{

  int version = stream.begin_class(type_name(-1), ITK_IMAGE_FIELD_VERSION);
  if (version) {
    if(stream.reading()) {
      GenericField<IMesh_, QuadBilinearLgn<Data>, ITKFData2d<Data> >::io(stream);
      stream.end_class();
      
      // set spacing
      typedef typename itk::Image<Data, 2> ImageType;
      const BBox bbox = this->mesh()->get_bounding_box();
      Point mesh_center;
      Vector mesh_size;
      if(bbox.valid()) {
	mesh_center = bbox.center();
	mesh_size = bbox.diagonal();
      }
      else {
	std::cerr << "No bounding box to get center\n"; // fix
	return;
      }
      
      // image origin and spacing
      double origin[ ImageType::ImageDimension ];
      origin[0] = mesh_center.x();
      origin[1] = mesh_center.y();
      
      this->fdata().set_image_origin( origin );
      
      double spacing[ ImageType::ImageDimension ];
      spacing[0] = mesh_size.x()/this->fdata().dim1();
      spacing[1] = mesh_size.y()/this->fdata().dim2();
      
      this->fdata().set_image_spacing( spacing );
      
      return;
    } else {
      GenericField<IMesh_, QuadBilinearLgn<Data>, ITKFData2d<Data> >::io(stream);
      stream.end_class();
      return;
    }
  }
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

#define ITKFData2d_VERSION 1

template<class T> 
void Pio(Piostream& stream, ITKFData2d<T>& data)
{
  stream.begin_class("ITKFData2d", ITKFData2d_VERSION);
  if(stream.reading()) {
    typedef typename itk::Image<T, 2> ImageType;
    typename ImageType::Pointer img = ImageType::New();

    // image start index
    typename ImageType::IndexType start;
    start[0] = 0;
    start[1] = 0;
    
    // image size
    unsigned int size_x = 0;
    unsigned int size_y = 0;

    Pio(stream, size_x);
    Pio(stream, size_y);

    typename ImageType::SizeType size;
    size[0] = size_x;
    size[1] = size_y;
    
    // allocate image
    typename ImageType::RegionType region;
    region.SetSize( size );
    region.SetIndex( start );
    
    img->SetRegions( region );
    img->Allocate();

    // set the image
    data.set_image(img);
  } else {
    unsigned int x = data.dim1();
    unsigned int y = data.dim2();
    Pio(stream, x);
    Pio(stream, y);
  }
  // loop through data and call Pio
  typename ITKFData2d<T>::const_iterator b, e;
  b = data.begin();
  e = data.end();
  while(b != e) {
    Pio(stream, const_cast<T&>(*b));
    ++b;
  }
  stream.end_class();
}

} // end namespace SCIRun


#endif // Datatypes_ITKImageField_h
