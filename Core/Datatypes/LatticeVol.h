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


#ifndef Datatypes_LatticeVol_h
#define Datatypes_LatticeVol_h

#include <Core/Datatypes/GenericField.h>
#include <Core/Datatypes/LatVolMesh.h>
#include <Core/Geometry/Tensor.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Containers/Array3.h>
#include <Core/Math/MiscMath.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/Assert.h>
#include <string>

namespace SCIRun {

using std::string;


template <class Data>
class FData3d : public Array3<Data> {
public:
  typedef Data value_type;
  typedef Data * iterator;

  Data *begin() { return &(*this)(0,0,0); }
  Data *end() { return &((*this)(dim1()-1,dim2()-1,dim3()-1))+1; }
    
  FData3d():Array3<Data>() {}
  FData3d(int):Array3<Data>() {}
  FData3d(const FData3d& data) {copy(data);}
  virtual ~FData3d(){}
  
  const value_type &operator[](typename LatVolMesh::Cell::index_type idx) const 
    { return operator()(idx.k_,idx.j_,idx.i_); } 
  const value_type &operator[](typename LatVolMesh::Face::index_type idx) const
    { return operator()(0, 0, idx.i_); }
  const value_type &operator[](typename LatVolMesh::Edge::index_type idx) const 
    { return operator()(0, 0, idx.i_); }
  const value_type &operator[](typename LatVolMesh::Node::index_type idx) const
    { return operator()(idx.k_,idx.j_,idx.i_); }

  value_type &operator[](typename LatVolMesh::Cell::index_type idx)
    { return operator()(idx.k_,idx.j_,idx.i_); } 
  value_type &operator[](typename LatVolMesh::Face::index_type idx)
    { return operator()(0, 0, idx.i_); }
  value_type &operator[](typename LatVolMesh::Edge::index_type idx)
    { return operator()(0, 0, idx.i_); }
  value_type &operator[](typename LatVolMesh::Node::index_type idx)
    { return operator()(idx.k_,idx.j_,idx.i_); }

  static const string type_name(int n = -1);
  virtual const string get_type_name(int n = -1) const { return type_name(n); }

  void resize(const LatVolMesh::Node::size_type &size)
    { newsize(size.k_, size.j_, size.i_); }
  void resize(LatVolMesh::Edge::size_type) {}
  void resize(LatVolMesh::Face::size_type) {}
  void resize(const LatVolMesh::Cell::size_type &size)
    { newsize(size.k_, size.j_, size.i_); }
};

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
class LatticeVol : public GenericField< LatVolMesh, FData3d<Data> >
{
public:
  LatticeVol();
  LatticeVol(Field::data_location data_at);
  LatticeVol(LatVolMeshHandle mesh, Field::data_location data_at);
  virtual LatticeVol<Data> *clone() const;
  virtual ~LatticeVol();

  virtual ScalarFieldInterface* query_scalar_interface() const;
  virtual VectorFieldInterface* query_vector_interface() const;
  virtual TensorFieldInterface* query_tensor_interface() const;

  static const string type_name(int n = -1);
  virtual const string get_type_name(int n = -1) const;
  static PersistentTypeID type_id;
  virtual void io(Piostream &stream);
  virtual const TypeDescription* get_type_description() const;

  // LatticeVol Specific methods.
  bool get_gradient(Vector &, const Point &);

private:
  static Persistent* maker();
};



template <class Data>
LatticeVol<Data>::LatticeVol()
  : GenericField<LatVolMesh, FData3d<Data> >()
{
}


template <class Data>
LatticeVol<Data>::LatticeVol(Field::data_location data_at)
  : GenericField<LatVolMesh, FData3d<Data> >(data_at)
{
}


template <class Data>
LatticeVol<Data>::LatticeVol(LatVolMeshHandle mesh,
			     Field::data_location data_at)
  : GenericField<LatVolMesh, FData3d<Data> >(mesh, data_at)
{
}


template <class Data>
LatticeVol<Data> *
LatticeVol<Data>::clone() const
{
  return new LatticeVol(*this);
}
  

template <class Data>
LatticeVol<Data>::~LatticeVol()
{
}


template <> ScalarFieldInterface *
LatticeVol<double>::query_scalar_interface() const;

template <> ScalarFieldInterface *
LatticeVol<float>::query_scalar_interface() const;

template <> ScalarFieldInterface *
LatticeVol<int>::query_scalar_interface() const;

template <> ScalarFieldInterface*
LatticeVol<short>::query_scalar_interface() const;

template <> ScalarFieldInterface*
LatticeVol<unsigned char>::query_scalar_interface() const;

template <class T>
ScalarFieldInterface*
LatticeVol<T>::query_scalar_interface() const 
{
  return 0;
}

template <>
VectorFieldInterface*
LatticeVol<Vector>::query_vector_interface() const;

template <class T>
VectorFieldInterface*
LatticeVol<T>::query_vector_interface() const
{
  return 0;
}

template <>
TensorFieldInterface*
LatticeVol<Tensor>::query_tensor_interface() const;

template <class T>
TensorFieldInterface*
LatticeVol<T>::query_tensor_interface() const
{
  return 0;
}


template <class Data>
const string
LatticeVol<Data>::get_type_name(int n) const
{
  return type_name(n);
}

template <class T>
const TypeDescription* 
get_type_description(LatticeVol<T>*)
{
  static TypeDescription* td = 0;
  static string name("LatticeVol");
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
LatticeVol<T>::get_type_description() const 
{
  return SCIRun::get_type_description((LatticeVol<T>*)0);
}

#define LATTICEVOL_VERSION 2

template <class Data>
Persistent* 
LatticeVol<Data>::maker()
{
  return scinew LatticeVol<Data>;
}

template <class Data>
PersistentTypeID
LatticeVol<Data>::type_id(type_name(-1),
		GenericField<LatVolMesh, FData3d<Data> >::type_name(-1),
                maker); 

template <class Data>
void
LatticeVol<Data>::io(Piostream &stream)
{
  int version = stream.begin_class(type_name(-1), LATTICEVOL_VERSION);
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


template <class Data>
const string
LatticeVol<Data>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;

  }
  else if (n == 0)
  {
    return "LatticeVol";
  }
  else
  {
    return find_type_name((Data *)0);
  }
} 




//! compute the gradient g, at point p
template <> bool LatticeVol<Tensor>::get_gradient(Vector &, const Point &p);
template <> bool LatticeVol<Vector>::get_gradient(Vector &, const Point &p);


template <class Data>
bool LatticeVol<Data>::get_gradient(Vector &g, const Point &p)
{
  // for now we only know how to do this for fields with scalars at the nodes
  if (query_scalar_interface())
  {
    if( data_at() == Field::NODE)
    {
      mesh_handle_type mesh = get_typed_mesh();
      const Point r = mesh->get_transform()->unproject(p);
      double x = r.x();
      double y = r.y();
      double z = r.z();
      
#if 0
      Vector pn=p-mesh->get_min();
      Vector diagonal = mesh->diagonal();
      int nx=mesh->get_nx();
      int ny=mesh->get_ny();
      int nz=mesh->get_nz();
      double diagx=diagonal.x();
      double diagy=diagonal.y();
      double diagz=diagonal.z();
      double x=pn.x()*(nx-1)/diagx;
      double y=pn.y()*(ny-1)/diagy;
      double z=pn.z()*(nz-1)/diagz;
#endif

      int ix0 = (int)x;
      int iy0 = (int)y;
      int iz0 = (int)z;
      int ix1 = ix0+1;
      int iy1 = iy0+1;
      int iz1 = iz0+1;
      if(ix0<0 || ix1>=nx)return false;
      if(iy0<0 || iy1>=ny)return false;
      if(iz0<0 || iz1>=nz)return false;
      double fx = x-ix0;
      double fy = y-iy0;
      double fz = z-iz0;
      double d000 = (double)value(LatVolMesh::Node::index_type(ix0,iy0,iz0));
      double d100 = (double)value(LatVolMesh::Node::index_type(ix1,iy0,iz0));
      double d010 = (double)value(LatVolMesh::Node::index_type(ix0,iy1,iz0));
      double d110 = (double)value(LatVolMesh::Node::index_type(ix1,iy1,iz0));
      double d001 = (double)value(LatVolMesh::Node::index_type(ix0,iy0,iz1));
      double d101 = (double)value(LatVolMesh::Node::index_type(ix1,iy0,iz1));
      double d011 = (double)value(LatVolMesh::Node::index_type(ix0,iy1,iz1));
      double d111 = (double)value(LatVolMesh::Node::index_type(ix1,iy1,iz1));
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
      g = mesh->get_transform()->unproject(Vector(dx, dy, dz));
      return true;
    }
  }
  return false;
}


} // end namespace SCIRun

#endif // Datatypes_LatticeVol_h
