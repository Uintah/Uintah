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


#ifndef Datatypes_LatVolField_h
#define Datatypes_LatVolField_h

#include <Core/Datatypes/GenericField.h>
#include <Core/Datatypes/LatVolMesh.h>
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
class FData3d : public Array3<Data> {
public:
  typedef Data value_type;
  typedef Data * iterator;
  typedef const Data * const_iterator;

  iterator begin() { return &(*this)(0,0,0); } 
  iterator end() { return &((*this)(dim1()-1,dim2()-1,dim3()-1))+1; }
  const_iterator begin() const { return &(*this)(0,0,0); } 
  const_iterator end() const { return &((*this)(dim1()-1,dim2()-1,dim3()-1))+1; }

    
  FData3d() : Array3<Data>() {}
  FData3d(int) : Array3<Data>() {} //default arg sgi bug workaround.
  FData3d(const FData3d& data) {Array3<Data>::copy(data); }
  virtual ~FData3d();
  
  const value_type &operator[](typename LatVolMesh::Cell::index_type idx) const
  { return operator()(idx.k_,idx.j_,idx.i_); } 
  const value_type &operator[](typename LatVolMesh::Face::index_type idx) const
  { return operator()(0, 0, idx); }
  const value_type &operator[](typename LatVolMesh::Edge::index_type idx) const
  { return operator()(0, 0, idx); }    
  const value_type &operator[](typename LatVolMesh::Node::index_type idx) const
  { return operator()(idx.k_,idx.j_,idx.i_); }    

  value_type &operator[](typename LatVolMesh::Cell::index_type idx)
  { return operator()(idx.k_,idx.j_,idx.i_); } 
  value_type &operator[](typename LatVolMesh::Face::index_type idx)
  { return operator()(0, 0, idx); }
  value_type &operator[](typename LatVolMesh::Edge::index_type idx)
  { return operator()(0, 0, idx); }    
  value_type &operator[](typename LatVolMesh::Node::index_type idx)
  { return operator()(idx.k_,idx.j_,idx.i_); }    

  void resize(const LatVolMesh::Node::size_type &size)
  { Array3<Data>::resize(size.k_, size.j_, size.i_); }
  void resize(const LatVolMesh::Edge::size_type &size)
  { Array3<Data>::resize(1, 1, size); }
  void resize(const LatVolMesh::Face::size_type &size)
  { Array3<Data>::resize(1, 1, size); }
  void resize(const LatVolMesh::Cell::size_type &size)
  { Array3<Data>::resize(size.k_, size.j_, size.i_); }

  unsigned int size() { return dim1() * dim2() * dim3(); }

  static const string type_name(int n = -1);
};


template <class Data>
FData3d<Data>::~FData3d()
{
}

  
template <class Data>
const string
FData3d<Data>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;
  }
  else if (n == 0)
  {
    return "FData3d";
  }
  else
  {
    return find_type_name((Data *)0);
  }
}


template <class Data>
class LatVolField : public GenericField< LatVolMesh, FData3d<Data> >
{
public:
  // Avoids a warning with g++ 3.1
  // ../src/Core/Datatypes/QuadraticTetVolField.h:95: warning: `typename 
  // SCIRun::QuadraticTetVolField<T>::mesh_handle_type' is implicitly a typename
  // ../src/Core/Datatypes/QuadraticTetVolField.h:95: warning: implicit typename is 
  // deprecated, please see the documentation for details
  typedef typename GenericField<LatVolMesh, FData3d<Data> >::mesh_handle_type mesh_handle_type;
  LatVolField();
  LatVolField(Field::data_location data_at);
  LatVolField(LatVolMeshHandle mesh, Field::data_location data_at);
  virtual LatVolField<Data> *clone() const;
  virtual ~LatVolField();

  //! Persistent IO
  static PersistentTypeID type_id;
  virtual void io(Piostream &stream);

  static const string type_name(int n = -1);
  virtual const TypeDescription* get_type_description(int n = -1) const;

  // LatVolField Specific methods.
  bool get_gradient(Vector &, const Point &);

private:
  static Persistent* maker();
};



template <class Data>
LatVolField<Data>::LatVolField()
  : GenericField<LatVolMesh, FData3d<Data> >()
{
}


template <class Data>
LatVolField<Data>::LatVolField(Field::data_location data_at)
  : GenericField<LatVolMesh, FData3d<Data> >(data_at)
{
}


template <class Data>
LatVolField<Data>::LatVolField(LatVolMeshHandle mesh,
			     Field::data_location data_at)
  : GenericField<LatVolMesh, FData3d<Data> >(mesh, data_at)
{
}


template <class Data>
LatVolField<Data> *
LatVolField<Data>::clone() const
{
  return new LatVolField<Data>(*this);
}
  

template <class Data>
LatVolField<Data>::~LatVolField()
{
}


template <class Data>
const string
LatVolField<Data>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;

  }
  else if (n == 0)
  {
    return "LatVolField";
  }
  else
  {
    return find_type_name((Data *)0);
  }
} 

template <class T> 
const TypeDescription*
LatVolField<T>::get_type_description(int n) const
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
LatVolField<Data>::maker()
{
  return scinew LatVolField<Data>;
}

template <class Data>
PersistentTypeID
LatVolField<Data>::type_id(type_name(-1),
		GenericField<LatVolMesh, FData3d<Data> >::type_name(-1),
                maker); 

template <class Data>
void
LatVolField<Data>::io(Piostream &stream)
{
  int version = stream.begin_class(type_name(-1), LAT_VOL_FIELD_VERSION);
  GenericField<LatVolMesh, FData3d<Data> >::io(stream);
  stream.end_class();                                                         
  if (version < 2) {
    FData3d<Data> temp;
    temp.copy(fdata());
    resize_fdata();
    int i, j, k;
    for (i=0; i<fdata().dim1(); i++)
      for (j=0; j<fdata().dim2(); j++)
	for (k=0; k<fdata().dim3(); k++)
	  fdata()(i,j,k)=temp(k,j,i);
  }
}


//! compute the gradient g, at point p
template <> bool LatVolField<Tensor>::get_gradient(Vector &, const Point &p);
template <> bool LatVolField<Vector>::get_gradient(Vector &, const Point &p);


template <class Data>
bool LatVolField<Data>::get_gradient(Vector &g, const Point &p)
{
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


} // end namespace SCIRun

#endif // Datatypes_LatVolField_h
