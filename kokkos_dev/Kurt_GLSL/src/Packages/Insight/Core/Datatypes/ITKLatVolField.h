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

#ifndef Datatypes_ITKLatVolField_h
#define Datatypes_ITKLatVolField_h

#include <Core/Util/TypeDescription.h>
#include <Core/Geometry/Tensor.h>
#include <Core/Geometry/Vector.h>


// To make SCIRun compile on OSX 10.4, certain classes need to be defined before
// including GenericField.h, if not SCIRun does not compile.
// Apparently a templated class needs to have its templated datatypes defined before
// it is defined itself. 
// I just put the missing ones here, to make the Insight Package compile properly.
// -- jeroen


#include <Packages/Insight/Core/Datatypes/share.h>
namespace Insight {

template <class Data> class ITKFData3d;
template <class Data> class ITKLatVolField;

}

namespace SCIRun {
  
  SCISHARE const TypeDescription* get_type_description(Insight::ITKFData3d<SCIRun::Tensor>*);
  SCISHARE const TypeDescription* get_type_description(Insight::ITKFData3d<SCIRun::Vector>*);
  SCISHARE const TypeDescription* get_type_description(Insight::ITKFData3d<double>*);
  SCISHARE const TypeDescription* get_type_description(Insight::ITKFData3d<float>*);
  SCISHARE const TypeDescription* get_type_description(Insight::ITKFData3d<int>*);
  SCISHARE const TypeDescription* get_type_description(Insight::ITKFData3d<short>*);
  SCISHARE const TypeDescription* get_type_description(Insight::ITKFData3d<char>*);
  SCISHARE const TypeDescription* get_type_description(Insight::ITKFData3d<unsigned int>*);
  SCISHARE const TypeDescription* get_type_description(Insight::ITKFData3d<unsigned short>*);
  SCISHARE const TypeDescription* get_type_description(Insight::ITKFData3d<unsigned char>*);
  SCISHARE const TypeDescription* get_type_description(Insight::ITKFData3d<unsigned long>*);
}

#include <Core/Containers/LockingHandle.h>
#include <Core/Basis/HexTrilinearLgn.h>
#include <Core/Datatypes/LatVolMesh.h>
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

namespace Insight {

using std::string;
using namespace SCIRun;

typedef LatVolMesh<HexTrilinearLgn<Point> > LVMesh_;

template<class T> class ITKFData3d;
template<class T> void Pio(Piostream& stream, ITKFData3d<T>& array);


template <class Data>
class ITKConstIterator : public itk::ImageRegionConstIterator<itk::Image<Data,3> > {
public:
  typedef itk::Image<Data, 3> ImageType;
  typedef typename itk::Image<Data, 3>::RegionType region_type;

  ITKConstIterator() : 
    itk::ImageRegionConstIterator< ImageType >() {}
  ITKConstIterator(const ImageType* ptr, const region_type &region) : 
    itk::ImageRegionConstIterator< ImageType >(ptr, region) { }
  virtual ~ITKConstIterator() {}
  const Data &operator*() { return itk::ImageRegionConstIterator<ImageType>::Value();}
};


template <class Data>
class ITKIterator : public ITKConstIterator<Data> {
public:
  typedef itk::Image<Data, 3> ImageType;
  typedef typename itk::Image<Data, 3>::RegionType region_type;

  ITKIterator() : ITKConstIterator<Data>() { }
  ITKIterator(const ImageType* ptr, const region_type &region) : 
    ITKConstIterator<Data>(ptr, region) { }
  virtual ~ITKIterator() {}
  Data &operator*() { return this->Value();}
};

#define ITKFData3d_VERSION 1

template <class Data>
class ITKFData3d {
public:
  typedef Data value_type;
  typedef itk::Image<Data, 3> image_type;
  typedef ITKIterator<Data> iterator;
  typedef ITKConstIterator<Data> const_iterator;
  
  iterator *begin_;
  iterator *end_;
  const_iterator *const_begin_;
  const_iterator *const_end_;

  const iterator &begin() { 
    if(image_set_)
      return *begin_; 
    else
      ASSERTFAIL("ITKFData3d image not set");
  }
  const iterator &end() { 
    if(image_set_)
      return *end_; 
    else
      ASSERTFAIL("ITKFData3d image not set");
  }

  const const_iterator &begin() const { 
    if(image_set_)
      return *const_begin_; 
    else
      ASSERTFAIL("ITKFData3d image not set");
  }
  const const_iterator &end() const { 
    if(image_set_)
      return *const_end_; 
    else
      ASSERTFAIL("ITKFData3d image not set");
  }


  ITKFData3d();
  ITKFData3d(int); //default arg sgi bug workaround.
  ITKFData3d(const ITKFData3d& data);
  virtual ~ITKFData3d();
  
  const value_type &operator[](typename LVMesh_::Cell::index_type idx) const
  { 
    if(image_set_) {
      typename image_type::IndexType pixel;
      pixel[0] = idx.i_;
      pixel[1] = idx.j_;
      pixel[2] = idx.k_;
      return image_->GetPixel( pixel ); 
    }
    else {
      ASSERTFAIL("ITKFData3D image not set");
    }
  } 
  const value_type &operator[](typename LVMesh_::Face::index_type idx) const
  { 
    ASSERTFAIL("const operator[] not defined for ITKLatVolField for Faces");
    if(image_set_) {
      typename image_type::IndexType pixel;
      pixel[0] = 0;
      pixel[1] = 0;
      pixel[2] = 0;
      return image_->GetPixel( pixel );  
    }
    else {
      ASSERTFAIL("ITKFData3d image not set");
    }
  }
  const value_type &operator[](typename LVMesh_::Edge::index_type idx) const
  { 
    ASSERTFAIL("const operator[] not defined for ITKLatVolField for Edges");
    if(image_set_) {
      typename image_type::IndexType pixel;
      pixel[0] = 0;
      pixel[1] = 0;
      pixel[2] = 0;
      return image_->GetPixel( pixel );  
    }
    else {
      ASSERTFAIL("ITKFData3d image not set");
    }
  }
  const value_type &operator[](typename LVMesh_::Node::index_type idx) const
  { 
    if(image_set_) {
      typename image_type::IndexType pixel;
      pixel[0] = idx.i_;
      pixel[1] = idx.j_;
      pixel[2] = idx.k_;
      return image_->GetPixel( pixel ); 
    }
    else {
      ASSERTFAIL("ITKFData3D image not set");
    } 
  }

  value_type &operator[](typename LVMesh_::Cell::index_type idx)
  { 
    if(image_set_) {
      typename image_type::IndexType pixel;
      pixel[0] = idx.i_;
      pixel[1] = idx.j_;
      pixel[2] = idx.k_;
      return image_->GetPixel( pixel ); 
    }
    else {
      ASSERTFAIL("ITKFData3d image not set");
    }
  }
  value_type &operator[](typename LVMesh_::Face::index_type idx)
  {
    ASSERTFAIL("operator[] not defined for ITKLatVolField for Faces");
    if(image_set_) {
      typename image_type::IndexType pixel;
      pixel[0] = 0;
      pixel[1] = 0;
      pixel[2] = 0;
      return image_->GetPixel( pixel );  
    }
    else {
      ASSERTFAIL("ITKFData3d image not set");
    }
  }
  value_type &operator[](typename LVMesh_::Edge::index_type idx)
  {
    ASSERTFAIL("operator[] not defined for ITKLatVolField for Edges");
    if(image_set_) {
      typename image_type::IndexType pixel;
      pixel[0] = 0;
      pixel[1] = 0;
      pixel[2] = 0;
      return image_->GetPixel( pixel );  
    }
    else {
      ASSERTFAIL("ITKFData3d image not set");
    }
  }
  value_type &operator[](typename LVMesh_::Node::index_type idx)
  {
    if(image_set_) {
      typename image_type::IndexType pixel;
      pixel[0] = idx.i_;
      pixel[1] = idx.j_;
      pixel[2] = idx.k_;
      return image_->GetPixel( pixel ); 
    }
    else {
      ASSERTFAIL("ITKFData3d image not set");
    }
  }

  // These do not do anything because and itk::Image takes care of
  // allocation.  This field merely points to an itk::Image.
  void resize(const typename LVMesh_::Node::size_type &size)
  { 
  }
  void resize(const typename LVMesh_::Edge::size_type &size)
  { 
  }
  void resize(const typename LVMesh_::Face::size_type &size)
  {
  }
  void resize(const typename LVMesh_::Cell::size_type &size)
  { 
  }
  
  void set_image(itk::Image<Data, 3>* img) {
    // set the image
    image_ = img;

    // create iterators
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

  unsigned int size() const { return (dim1() * dim2() * dim3()); }

  unsigned int dim1() const;
  unsigned int dim2() const;
  unsigned int dim3() const;

  static const string type_name(int n = -1);

  typename image_type::Pointer get_image() { return image_; }

private:
  bool image_set_;
  typename image_type::Pointer image_;
};


template <class Data>
ITKFData3d<Data>::ITKFData3d()
{
  image_set_ = false;
  image_ = image_type::New(); 
  begin_ = 0;
  end_ = 0;
  const_begin_ = 0;
  const_end_ = 0;
}

template <class Data>
ITKFData3d<Data>::ITKFData3d(int a)
{
  image_set_ = false;
  image_ = image_type::New(); 
  begin_ = 0;
  end_ = 0;
  const_begin_ = 0;
  const_end_ = 0;
}

template <class Data>
ITKFData3d<Data>::ITKFData3d(const ITKFData3d& data) {
  image_set_ = false;
  image_ = image_type::New();

  if(dynamic_cast<itk::Image<Data, 3>* >(data.image_.GetPointer() )) {
    set_image(dynamic_cast<itk::Image<Data, 3>* >(data.image_.GetPointer() ));
  }
  else {
    ASSERTFAIL("Incorrect image type in ITKFData3d copy constructor");
  }
  
}

template <class Data>
ITKFData3d<Data>::~ITKFData3d()
{
  if ( begin_ ) delete begin_;
  if ( end_) delete end_;

  if ( const_begin_ ) delete const_begin_;
  if ( const_end_) delete const_end_;
}
  
template <class Data>
const string
ITKFData3d<Data>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;
  }
  else if (n == 0)
  {
    return "ITKFData3d";
  }
  else
  {
    return find_type_name((Data *)0);
  }
}

template <class Data>
unsigned int ITKFData3d<Data>::dim1() const
{
  if(image_set_)
    return image_->GetLargestPossibleRegion().GetSize()[0];
  else {
    ASSERTFAIL("ITKFData3d Image not set");
    return 0;
  }
}

template <class Data>
unsigned int ITKFData3d<Data>::dim2() const
{
  if(image_set_)
    return image_->GetLargestPossibleRegion().GetSize()[1];
  else {
    ASSERTFAIL("ITKFData3d Image not set");
    return 0;
  }
}

template <class Data>
unsigned int ITKFData3d<Data>::dim3() const
{
  if(image_set_)
    return image_->GetLargestPossibleRegion().GetSize()[2];
  else {
    ASSERTFAIL("ITKFData3d Image not set");
    return 0;
  }
}


///////////////////////////////////////////////////
template <class Data>
class ITKLatVolField : public GenericField<LVMesh_, HexTrilinearLgn<Data>, ITKFData3d<Data> >
{

public:
  // Avoids a warning with g++ 3.1
  // ../src/Core/Datatypes/QuadraticTetVolField.h:95: warning: `typename 
  // SCIRun::QuadraticTetVolField<T>::mesh_handle_type' is implicitly a typename
  // ../src/Core/Datatypes/QuadraticTetVolField.h:95: warning: implicit typename is 
  // deprecated, please see the documentation for details


  typedef HexTrilinearLgn<Data> ITKLVBasis;
  typedef GenericField<LVMesh_, ITKLVBasis, ITKFData3d<Data> > ITKLVF;

  ITKLatVolField() :
    GenericField<LVMesh_, HexTrilinearLgn<Data>, ITKFData3d<Data> >() {
    image_set_ = false;
  }

  ITKLatVolField(typename ITKLVF::mesh_handle_type mesh)  :
    GenericField<LVMesh_, HexTrilinearLgn<Data>, ITKFData3d<Data> >(mesh) {
    image_set_ = false;
  }

  ITKLatVolField(typename ITKLVF::mesh_handle_type mesh, itk::Object* img) :
        GenericField<LVMesh_, HexTrilinearLgn<Data>, ITKFData3d<Data> >(mesh) {
    this->SetImage(img);
  }
  
  virtual ITKLatVolField<Data> *clone() const;
  virtual ~ITKLatVolField();

  void SetImage(itk::Object*);

  //! Persistent IO
  static PersistentTypeID type_id;
  virtual void io(Piostream &stream);

  static const string type_name(int n = -1);
  virtual const TypeDescription* get_type_description(int n = -1) const;

  // LatVolField Specific methods.
  bool get_gradient(SCIRun::Vector &, const Point &);

  itk::Object* get_image() { return this->fdata().get_image(); }

private:
  bool image_set_;
  static Persistent* maker();
};



template <class Data>
void ITKLatVolField<Data>::SetImage(itk::Object* img)
{
  if(dynamic_cast<itk::Image<Data, 3>* >(img)) {
    this->fdata().set_image(dynamic_cast<itk::Image<Data, 3>* >(img));
    image_set_ = true;
  }
  else {
    ASSERTFAIL("Image data is incorrect type.");
  }
}


template <class Data>
ITKLatVolField<Data> *
ITKLatVolField<Data>::clone() const
{
  return new ITKLatVolField<Data>(*this);
}
  

template <class Data>
ITKLatVolField<Data>::~ITKLatVolField()
{
}



template <class Data>
const string
ITKLatVolField<Data>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;

  }
  else if (n == 0)
  {
    return "ITKLatVolField";
  }
  else
  {
    return find_type_name((Data *)0);
  }
} 

template <class T> 
const TypeDescription*
ITKLatVolField<T>::get_type_description(int n) const
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

#define ITK_LAT_VOL_FIELD_VERSION 1

template <class Data>
Persistent* 
ITKLatVolField<Data>::maker()
{
  return scinew ITKLatVolField<Data>;
}

template <class Data>
PersistentTypeID
ITKLatVolField<Data>::type_id(type_name(-1),
		GenericField<LVMesh_, HexTrilinearLgn<Data>, ITKFData3d<Data> >::type_name(-1),
                maker); 

template <class Data>
void
ITKLatVolField<Data>::io(Piostream &stream)
{
  int version = stream.begin_class(type_name(-1), ITK_LAT_VOL_FIELD_VERSION);
  if(version) {
    if(stream.reading()) {
      GenericField<LVMesh_, HexTrilinearLgn<Data>, ITKFData3d<Data> >::io(stream);
      stream.end_class();
      
      // set spacing
      typedef typename itk::Image<Data, 3> ImageType;
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
      origin[2] = mesh_center.z();
      
      this->fdata().set_image_origin( origin );
      
      double spacing[ ImageType::ImageDimension ];
      spacing[0] = mesh_size.x()/this->fdata().dim1();
      spacing[1] = mesh_size.y()/this->fdata().dim2();
      spacing[2] = mesh_size.z()/this->fdata().dim3();
      
      this->fdata().set_image_spacing( spacing );
      
      return;
    } else {
      GenericField<LVMesh_, HexTrilinearLgn<Data>, ITKFData3d<Data> >::io(stream);
      stream.end_class();
      return;
    }
  }

}


//! compute the gradient g, at point p
template <> bool ITKLatVolField<Tensor>::get_gradient(Vector &, const Point &p);
template <> bool ITKLatVolField<Vector>::get_gradient(Vector &, const Point &p);


template <class Data>
bool ITKLatVolField<Data>::get_gradient(Vector &g, const Point &p)
{
  if( image_set_ ) {
    // for now we only know how to do this for fields with scalars at the nodes
    if (this->query_scalar_interface().get_rep())
    {
      typename GenericField<LVMesh_, HexTrilinearLgn<Data>, ITKFData3d<Data> >::mesh_handle_type mesh = this->get_typed_mesh();
      const Point r = mesh->get_transform().unproject(p);
      double x = r.x();
      double y = r.y();
      double z = r.z();
      
      int ni=mesh->get_ni();
      int nj=mesh->get_nj();
      int nk=mesh->get_nk();
      int ix0 = (int)x;
      int iy0 = (int)y;
      int iz0 = (int)z;
      int ix1 = ix0+1;
      int iy1 = iy0+1;
      int iz1 = iz0+1;
      if(ix0<0 || ix1>=ni)return false;
      if(iy0<0 || iy1>=nj)return false;
      if(iz0<0 || iz1>=nk)return false;
      double fx = x-ix0;
      double fy = y-iy0;
      double fz = z-iz0;
      LVMesh_ *mp = mesh.get_rep();
      double d000 = (double)this->value(LVMesh_::Node::index_type(mp,ix0,iy0,iz0));
      double d100 = (double)this->value(LVMesh_::Node::index_type(mp,ix1,iy0,iz0));
      double d010 = (double)this->value(LVMesh_::Node::index_type(mp,ix0,iy1,iz0));
      double d110 = (double)this->value(LVMesh_::Node::index_type(mp,ix1,iy1,iz0));
      double d001 = (double)this->value(LVMesh_::Node::index_type(mp,ix0,iy0,iz1));
      double d101 = (double)this->value(LVMesh_::Node::index_type(mp,ix1,iy0,iz1));
      double d011 = (double)this->value(LVMesh_::Node::index_type(mp,ix0,iy1,iz1));
      double d111 = (double)this->value(LVMesh_::Node::index_type(mp,ix1,iy1,iz1));
      double z00 = Interpolate(d000, d001, fz);
      double z01 = Interpolate(d010, d011, fz);
      double z10 = Interpolate(d100, d101, fz);
      double z11 = Interpolate(d110, d111, fz);
      double yy0 = Interpolate(z00, z01, fy);
      double yy1 = Interpolate(z10, z11, fy);
      double dx = (yy1-yy0);
      double x00 = Interpolate(d000, d100, fx);
      double x01 = Interpolate(d001, d101, fx);
      double x10 = Interpolate(d010, d110, fx);
      double x11 = Interpolate(d011, d111, fx);
      double y0 = Interpolate(x00, x10, fy);
      double y1 = Interpolate(x01, x11, fy);
      double dz = (y1-y0);
      double z0 = Interpolate(x00, x01, fz);
      double z1 = Interpolate(x10, x11, fz);
      double dy = (z1-z0);
      g = mesh->get_transform().unproject(Vector(dx, dy, dz));
      return true;
    }
    return false;
  }
  ASSERTFAIL("ITKLatVolField Image not set");
  return false;
}

template<class T> 
void Pio(Piostream& stream, ITKFData3d<T>& data)
{
  stream.begin_class("ITKFData3d", ITKFData3d_VERSION);
  if(stream.reading()) {
    typedef typename itk::Image<T, 3> ImageType;
    typename ImageType::Pointer img = ImageType::New();

    // image start index
    typename ImageType::IndexType start;
    start[0] = 0;
    start[1] = 0;
    start[2] = 0;
    
    // image size
    unsigned int size_x = 0;
    unsigned int size_y = 0;
    unsigned int size_z = 0;

    Pio(stream, size_x);
    Pio(stream, size_y);
    Pio(stream, size_z);

    typename ImageType::SizeType size;
    size[0] = size_x;
    size[1] = size_y;
    size[2] = size_z;
    
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
    unsigned int z = data.dim3();
    Pio(stream, x);
    Pio(stream, y);
    Pio(stream, z);
  }
  // loop through data and call Pio
  typename ITKFData3d<T>::const_iterator b, e;
  b = data.begin();
  e = data.end();
  while(b != e) {
    Pio(stream, const_cast<T&>(*b));
    ++b;
  }
  stream.end_class();

}

} // end namespace Insight


#endif // Datatypes_ITKLatVolField_h
