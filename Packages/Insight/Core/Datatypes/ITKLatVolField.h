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

#ifndef Datatypes_ITKLatVolField_h
#define Datatypes_ITKLatVolField_h

#include <Core/Datatypes/GenericField.h>
#include <Core/Datatypes/LatVolMesh.h>
#include <Core/Geometry/Tensor.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Containers/Array3.h>
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
  Data &operator*() { return Value();}
};


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
  
  const value_type &operator[](typename LatVolMesh::Cell::index_type idx) const
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
  const value_type &operator[](typename LatVolMesh::Face::index_type idx) const
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
  const value_type &operator[](typename LatVolMesh::Edge::index_type idx) const
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
  const value_type &operator[](typename LatVolMesh::Node::index_type idx) const
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

  value_type &operator[](typename LatVolMesh::Cell::index_type idx)
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
  value_type &operator[](typename LatVolMesh::Face::index_type idx)
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
  value_type &operator[](typename LatVolMesh::Edge::index_type idx)
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
  value_type &operator[](typename LatVolMesh::Node::index_type idx)
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
  void resize(const LatVolMesh::Node::size_type &size)
  { 
  }
  void resize(const LatVolMesh::Edge::size_type &size)
  { 
  }
  void resize(const LatVolMesh::Face::size_type &size)
  {
  }
  void resize(const LatVolMesh::Cell::size_type &size)
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
  
  unsigned int size() { return (dim1() * dim2() * dim3()); }

  unsigned int dim1();
  unsigned int dim2();
  unsigned int dim3();

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
unsigned int ITKFData3d<Data>::dim1()
{
  if(image_set_)
    return image_->GetLargestPossibleRegion().GetSize()[0];
  else {
    ASSERTFAIL("ITKFData3d Image not set");
    return 0;
  }
}

template <class Data>
unsigned int ITKFData3d<Data>::dim2()
{
  if(image_set_)
    return image_->GetLargestPossibleRegion().GetSize()[1];
  else {
    ASSERTFAIL("ITKFData3d Image not set");
    return 0;
  }
}

template <class Data>
unsigned int ITKFData3d<Data>::dim3()
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
class ITKLatVolField : public GenericField< LatVolMesh, ITKFData3d<Data> >
{
public:
  // Avoids a warning with g++ 3.1
  // ../src/Core/Datatypes/QuadraticTetVolField.h:95: warning: `typename 
  // SCIRun::QuadraticTetVolField<T>::mesh_handle_type' is implicitly a typename
  // ../src/Core/Datatypes/QuadraticTetVolField.h:95: warning: implicit typename is 
  // deprecated, please see the documentation for details
  typedef typename GenericField<LatVolMesh, ITKFData3d<Data> >::mesh_handle_type mesh_handle_type;
  typedef LatVolMesh mesh_type;

  ITKLatVolField();
  ITKLatVolField(Field::data_location data_at);
  ITKLatVolField(LatVolMeshHandle mesh, Field::data_location data_at);
  ITKLatVolField(LatVolMeshHandle mesh, Field::data_location data_at, itk::Object* img);
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

  itk::Object* get_image() { return fdata().get_image(); }

private:
  bool image_set_;
  static Persistent* maker();
};



template <class Data>
ITKLatVolField<Data>::ITKLatVolField()
  : GenericField<LatVolMesh, ITKFData3d<Data> >()
{
  // need to set image
  image_set_ = false;
}


template <class Data>
ITKLatVolField<Data>::ITKLatVolField(Field::data_location data_at)
  : GenericField<LatVolMesh, ITKFData3d<Data> >(data_at)
{
  // need to set image
  image_set_ = false;
}


template <class Data>
ITKLatVolField<Data>::ITKLatVolField(LatVolMeshHandle mesh,
			     Field::data_location data_at)
  : GenericField<LatVolMesh, ITKFData3d<Data> >(mesh, data_at)
{
  // need to set image
  image_set_ = false;
}

template <class Data>
ITKLatVolField<Data>::ITKLatVolField(LatVolMeshHandle mesh,
			     Field::data_location data_at, itk::Object* img)
  : GenericField<LatVolMesh, ITKFData3d<Data> >(mesh, data_at)
{
  this->SetImage(img);
}

template <class Data>
void ITKLatVolField<Data>::SetImage(itk::Object* img)
{
  if(dynamic_cast<itk::Image<Data, 3>* >(img)) {
    fdata().set_image(dynamic_cast<itk::Image<Data, 3>* >(img));
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

#define ITK_LAT_VOL_FIELD_VERSION 3

template <class Data>
Persistent* 
ITKLatVolField<Data>::maker()
{
  return scinew ITKLatVolField<Data>;
}

template <class Data>
PersistentTypeID
ITKLatVolField<Data>::type_id(type_name(-1),
		GenericField<LatVolMesh, ITKFData3d<Data> >::type_name(-1),
                maker); 

template <class Data>
void
ITKLatVolField<Data>::io(Piostream &stream)
{
  ASSERTFAIL("ITKLatVolField::io not implemented yet");
  if(image_set_) {
    /*  
	int version = stream.begin_class(type_name(-1), ITK_LAT_VOL_FIELD_VERSION);
	GenericField<LatVolMesh, ITKFData3d<Data> >::io(stream); 
	stream.end_class();                                                         
	if (version < 2) { 
	ITKFData3d <Data> temp;
	temp.copy(fdata()); 
	resize_fdata(); 
	int i, j, k; 
	for (i=0; i<fdata().dim1(); i++) 
	for (j=0; j<fdata().dim2(); j++) 
	for (k=0; k<fdata().dim3(); k++) 
	fdata()(i,j,k)=temp(k,j,i); 
	}
    */  
  }
  else {
    ASSERTFAIL("ITKLatVolField Image not set");
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
    if (query_scalar_interface().get_rep())
    {
      if( data_at() == Field::NODE)
      {
	mesh_handle_type mesh = get_typed_mesh();
	const Point r = mesh->get_transform().unproject(p);
	double x = r.x();
	double y = r.y();
	double z = r.z();
	
#if 0
	Vector pn=p-mesh->get_min();
	Vector diagonal = mesh->diagonal();
	int ni=mesh->get_ni();
	int nj=mesh->get_nj();
	int nk=mesh->get_nk();
	double diagx=diagonal.x();
	double diagy=diagonal.y();
	double diagz=diagonal.z();
	double x=pn.x()*(ni-1)/diagx;
	double y=pn.y()*(nj-1)/diagy;
	double z=pn.z()*(nk-1)/diagz;
#endif
	
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
	LatVolMesh *mp = mesh.get_rep();
	double d000 = (double)value(LatVolMesh::Node::index_type(mp,ix0,iy0,iz0));
	double d100 = (double)value(LatVolMesh::Node::index_type(mp,ix1,iy0,iz0));
	double d010 = (double)value(LatVolMesh::Node::index_type(mp,ix0,iy1,iz0));
	double d110 = (double)value(LatVolMesh::Node::index_type(mp,ix1,iy1,iz0));
	double d001 = (double)value(LatVolMesh::Node::index_type(mp,ix0,iy0,iz1));
	double d101 = (double)value(LatVolMesh::Node::index_type(mp,ix1,iy0,iz1));
	double d011 = (double)value(LatVolMesh::Node::index_type(mp,ix0,iy1,iz1));
	double d111 = (double)value(LatVolMesh::Node::index_type(mp,ix1,iy1,iz1));
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
    }
  return false;
  }
  ASSERTFAIL("ITKLatVolField Image not set");
  return false;
}

template<class T> 
void Pio(Piostream& stream, ITKFData3d<T>& array)
{
  ASSERTFAIL("Pio not written for ITKLatVolField");
}

} // end namespace Insight

#endif // Datatypes_ITKLatVolField_h
