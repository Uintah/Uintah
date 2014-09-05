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


#ifndef Datatypes_QuadraticLatVolField_h
#define Datatypes_QuadraticLatVolField_h

#include <Core/Datatypes/GenericField.h>
#include <Core/Datatypes/QuadraticLatVolMesh.h>
#include <Core/Geometry/Tensor.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/Assert.h>
#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {

using std::string;


template <class Data>
class QuadraticLatVolField : public GenericField< QuadraticLatVolMesh, vector<Data> >
{
public:
  // Avoids a warning with g++ 3.1
  // ../src/Core/Datatypes/QuadraticTetVolField.h:95: warning: `typename 
  // SCIRun::QuadraticTetVolField<T>::mesh_handle_type' is implicitly a typename
  // ../src/Core/Datatypes/QuadraticTetVolField.h:95: warning: implicit typename is 
  // deprecated, please see the documentation for details
  typedef typename GenericField<QuadraticLatVolMesh, vector<Data> >::mesh_handle_type mesh_handle_type;

  QuadraticLatVolField();
  QuadraticLatVolField(Field::data_location data_at);
  QuadraticLatVolField(QuadraticLatVolMeshHandle mesh, Field::data_location data_at);
  virtual QuadraticLatVolField<Data> *clone() const;
  virtual ~QuadraticLatVolField();

  //! Persistent IO
  static PersistentTypeID type_id;
  virtual void io(Piostream &stream);

  static const string type_name(int n = -1);
  virtual const TypeDescription* get_type_description(int n = -1) const;

  // QuadraticLatVolField Specific methods.
  bool get_gradient(Vector &, const Point &);

private:
  static Persistent* maker();
};



template <class Data>
QuadraticLatVolField<Data>::QuadraticLatVolField()
  : GenericField<QuadraticLatVolMesh, vector<Data> >()
{
}


template <class Data>
QuadraticLatVolField<Data>::QuadraticLatVolField(Field::data_location data_at)
  : GenericField<QuadraticLatVolMesh, vector<Data> >(data_at)
{
}


template <class Data>
QuadraticLatVolField<Data>::QuadraticLatVolField(QuadraticLatVolMeshHandle mesh,
			     Field::data_location data_at)
  : GenericField<QuadraticLatVolMesh, vector<Data> >(mesh, data_at)
{
}


template <class Data>
QuadraticLatVolField<Data> *
QuadraticLatVolField<Data>::clone() const
{
  return new QuadraticLatVolField<Data>(*this);
}
  

template <class Data>
QuadraticLatVolField<Data>::~QuadraticLatVolField()
{
}


template <class Data>
const string
QuadraticLatVolField<Data>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;

  }
  else if (n == 0)
  {
    return "QuadraticLatVolField";
  }
  else
  {
    return find_type_name((Data *)0);
  }
} 

template <class T> 
const TypeDescription*
QuadraticLatVolField<T>::get_type_description(int n) const
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

#define QUADRATIC_LAT_VOL_FIELD_VERSION 1

template <class Data>
Persistent* 
QuadraticLatVolField<Data>::maker()
{
  return scinew QuadraticLatVolField<Data>;
}

template <class Data>
PersistentTypeID
QuadraticLatVolField<Data>::type_id(type_name(-1),
		GenericField<QuadraticLatVolMesh, vector<Data> >::type_name(-1),
                maker); 

template <class Data>
void
QuadraticLatVolField<Data>::io(Piostream &stream)
{
  stream.begin_class(type_name(-1), QUADRATIC_LAT_VOL_FIELD_VERSION);
  GenericField<QuadraticLatVolMesh, vector<Data> >::io(stream);
  stream.end_class();                                                         
}


//! compute the gradient g, at point p
template <> bool QuadraticLatVolField<Tensor>::get_gradient(Vector &, const Point &p);
template <> bool QuadraticLatVolField<Vector>::get_gradient(Vector &, const Point &p);


template <class Data>
bool QuadraticLatVolField<Data>::get_gradient(Vector &g, const Point &p)
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
      QuadraticLatVolMesh *mp = mesh.get_rep();
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

#endif // Datatypes_QuadraticLatVolField_h
