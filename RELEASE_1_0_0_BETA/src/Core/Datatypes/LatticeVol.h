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
    
  FData3d():Array3<Data>(){}
  FData3d(const FData3d& data) :
    Array3<Data>(data) {} 
  virtual ~FData3d(){}
  
  const value_type &operator[](typename LatVolMesh::cell_index idx) const 
    { return operator()(idx.i_,idx.j_,idx.k_); } 
  const value_type &operator[](typename LatVolMesh::face_index idx) const
    { return operator()(idx.i_, 0, 0); }
  const value_type &operator[](typename LatVolMesh::edge_index idx) const 
    { return operator()(idx.i_, 0, 0); }
  const value_type &operator[](typename LatVolMesh::node_index idx) const
    { return operator()(idx.i_,idx.j_,idx.k_); }

  value_type &operator[](typename LatVolMesh::cell_index idx)
    { return operator()(idx.i_,idx.j_,idx.k_); } 
  value_type &operator[](typename LatVolMesh::face_index idx)
    { return operator()(idx.i_, 0, 0); }
  value_type &operator[](typename LatVolMesh::edge_index idx)
    { return operator()(idx.i_, 0, 0); }
  value_type &operator[](typename LatVolMesh::node_index idx)
    { return operator()(idx.i_,idx.j_,idx.k_); }

  static const string type_name(int n = -1);
  virtual const string get_type_name(int n = -1) const { return type_name(n); }

  void resize(const LatVolMesh::node_size_type &size)
    { newsize(size.i_, size.j_, size.k_); }
  void resize(LatVolMesh::edge_size_type) {}
  void resize(LatVolMesh::face_size_type) {}
  void resize(const LatVolMesh::cell_size_type &size)
    { newsize(size.i_, size.j_, size.k_); }
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
class LatticeVol : public GenericField< LatVolMesh, FData3d<Data> > { 

public:

  LatticeVol() :
    GenericField<LatVolMesh, FData3d<Data> >() {}
  LatticeVol(Field::data_location data_at) :
    GenericField<LatVolMesh, FData3d<Data> >(data_at) {}
  LatticeVol(LatVolMeshHandle mesh, Field::data_location data_at) : 
    GenericField<LatVolMesh, FData3d<Data> >(mesh, data_at) {}
  
  virtual ~LatticeVol(){}

  virtual LatticeVol<Data> *clone() const 
    { return new LatticeVol<Data>(*this); }
 
  static const string type_name(int n = -1);
  virtual const string get_type_name(int n = -1) const { return type_name(n); }
  static PersistentTypeID type_id;
  virtual void io(Piostream &stream);
  bool get_gradient(Vector &, Point &);
private:
  static Persistent* maker();
};

#define LATTICEVOL_VERSION 1

template <class Data>
Persistent* 
LatticeVol<Data>::maker()
{
  return scinew LatticeVol<Data>;
}

template <class Data>
PersistentTypeID
LatticeVol<Data>::type_id(type_name(),
		GenericField<LatVolMesh, FData3d<Data> >::type_name(),
                maker); 

template <class Data>
void
LatticeVol<Data>::io(Piostream &stream)
{
  stream.begin_class(type_name().c_str(), LATTICEVOL_VERSION);
  GenericField<LatVolMesh, FData3d<Data> >::io(stream);
  stream.end_class();                                                         
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
template <> bool LatticeVol<Tensor>::get_gradient(Vector &, Point &p);
template <> bool LatticeVol<Vector>::get_gradient(Vector &, Point &p);

template <class Data>
bool LatticeVol<Data>::get_gradient(Vector &g, Point &p) {
  // for now we only know how to do this for fields with doubles at the nodes
  ASSERT(data_at() == Field::NODE)
  ASSERT(type_name(1) == "double")
  LatticeVol<double> *lvd = dynamic_cast<LatticeVol<double> *>(this);

  Vector pn=p-get_typed_mesh()->get_min();
  Vector diagonal = get_typed_mesh()->diagonal();
  int nx=fdata().dim1();
  int ny=fdata().dim2();
  int nz=fdata().dim3();
  double diagx=diagonal.x();
  double diagy=diagonal.y();
  double diagz=diagonal.z();
  double x=pn.x()*(nx-1)/diagx;
  double y=pn.y()*(ny-1)/diagy;
  double z=pn.z()*(nz-1)/diagz;
  int ix0=(int)x;
  int iy0=(int)y;
  int iz0=(int)z;
  int ix1=ix0+1;
  int iy1=iy0+1;
  int iz1=iz0+1;
  if(ix0<0 || ix1>=nx)return false;
  if(iy0<0 || iy1>=ny)return false;
  if(iz0<0 || iz1>=nz)return false;
  double fx=x-ix0;
  double fy=y-iy0;
  double fz=z-iz0;
  double d000=lvd->value(LatVolMesh::node_index(ix0,iy0,iz0));
  double d100=lvd->value(LatVolMesh::node_index(ix1,iy0,iz0));
  double d010=lvd->value(LatVolMesh::node_index(ix0,iy1,iz0));
  double d110=lvd->value(LatVolMesh::node_index(ix1,iy1,iz0));
  double d001=lvd->value(LatVolMesh::node_index(ix0,iy0,iz1));
  double d101=lvd->value(LatVolMesh::node_index(ix1,iy0,iz1));
  double d011=lvd->value(LatVolMesh::node_index(ix0,iy1,iz1));
  double d111=lvd->value(LatVolMesh::node_index(ix1,iy1,iz1));
  double z00=Interpolate(d000, d001, fz);
  double z01=Interpolate(d010, d011, fz);
  double z10=Interpolate(d100, d101, fz);
  double z11=Interpolate(d110, d111, fz);
  double yy0=Interpolate(z00, z01, fy);
  double yy1=Interpolate(z10, z11, fy);
  double dx=(yy1-yy0)*(nx-1)/diagx;
  double x00=Interpolate(d000, d100, fx);
  double x01=Interpolate(d001, d101, fx);
  double x10=Interpolate(d010, d110, fx);
  double x11=Interpolate(d011, d111, fx);
  double y0=Interpolate(x00, x10, fy);
  double y1=Interpolate(x01, x11, fy);
  double dz=(y1-y0)*(nz-1)/diagz;
  double z0=Interpolate(x00, x01, fz);
  double z1=Interpolate(x10, x11, fz);
  double dy=(z1-z0)*(ny-1)/diagy;
  g = Vector(dx, dy, dz);
  return true;
}

} // end namespace SCIRun

#endif // Datatypes_LatticeVol_h
