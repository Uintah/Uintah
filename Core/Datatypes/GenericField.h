/*
 *  GenericField.h
 *
 *  Written by:
 *   Martin Cole
 *   School of Computing
 *   University of Utah
 *
 *  Copyright (C) 2001 SCI Institute
 */

#ifndef Datatypes_GenericField_h
#define Datatypes_GenericField_h

#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/FieldAlgo.h>
#include <Core/Datatypes/TypeName.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Malloc/Allocator.h>
#include <vector>
#include <iostream>

using std::cerr;
using std::endl;

namespace SCIRun {

template <class Mesh, class FData>
class GenericField: public Field 
{
public:
  //! Typedefs to support the Field concept.
  typedef typename FData::value_type      value_type;
  typedef Mesh                            mesh_type;
  typedef LockingHandle<mesh_type>        mesh_handle_type;
  typedef FData                           fdata_type;
  typedef GenericInterpolate<value_type>  interp_type;

  // only Pio should use this constructor
  GenericField() : 
    Field(),
    mesh_(mesh_handle_type(new mesh_type())),
    fdata_(fdata_type())
  {}

  GenericField(data_location data_at) : 
    Field(data_at),
    mesh_(mesh_handle_type(new mesh_type())),
    fdata_(fdata_type())
  {}

  GenericField(mesh_handle_type mesh, data_location data_at) : 
    Field(data_at),
    mesh_(mesh),
    fdata_(fdata_type())
  { resize_fdata(); }

  virtual ~GenericField() {}

  //! Required virtual functions from field base.
  virtual MeshBaseHandle mesh() const
  { return MeshBaseHandle(mesh_.get_rep()); }

  //! Required interfaces from field base.
  virtual interp_type* query_interpolate() const;
//  virtual InterpolateToScalar* query_interpolate_to_scalar() const;

  //! Required interface to support Field Concept.
  bool value(value_type &val, typename mesh_type::node_index i) const
  { val = fdata_[i]; return true; }
  bool value(value_type &val, typename mesh_type::edge_index i) const 
  { val = fdata_[i]; return true; }
  bool value(value_type &val, typename mesh_type::face_index i) const 
  { val = fdata_[i]; return true; }
  bool value(value_type &val, typename mesh_type::cell_index i) const 
  { val = fdata_[i]; return true; }

  //! No safety check for the following calls, be sure you know where data is.
  value_type value(typename mesh_type::node_index i) const
  { return fdata_[i]; }
  value_type value(typename mesh_type::edge_index i) const
  { return fdata_[i]; }
  value_type value(typename mesh_type::face_index i) const 
  { return fdata_[i]; }
  value_type value(typename mesh_type::cell_index i) const 
  { return fdata_[i]; }

  virtual void resize_fdata();

  fdata_type& fdata() { return fdata_; }
  const fdata_type& fdata() const { return fdata_; }

  mesh_handle_type get_typed_mesh() const { return mesh_; };

  //! Persistent I/O.
  virtual void io(Piostream &stream);
  static  PersistentTypeID type_id;
  static  const string type_name(int n = -1);
  virtual const string get_type_name(int n = -1) const { return type_name(n); }

private:

  //! generic interpolate object, using linear interpolation.
  template <class Data>
  struct GInterp : public GenericInterpolate<Data> {
    GInterp(const GenericField<Mesh, FData> *f) :
      f_(f) {}
    bool interpolate(const Point& p, Data &value) const;
    const GenericField<Mesh, FData> *f_;
  };

  static Persistent *maker();

  //! A (generic) mesh.
  mesh_handle_type             mesh_;
  //! Data container.
  fdata_type                   fdata_;
  //! minmax
//  bool has_minmax;
//  double min_, max_;
}; 

//! Virtual interface.
//! internal interp object
template <class Mesh, class FData> template <class Data>
bool 
GenericField<Mesh, FData>::GInterp<Data>::interpolate(const Point& p, 
						      Data& value) const
{
  bool rval = false;
  switch (f_->data_at()) {
  case Field::NODE :
    {
      LinearInterp<GenericField<Mesh, FData>, typename Mesh::node_index > ftor;
      rval = SCIRun::interpolate(*f_, p, ftor);
      if (rval) { value = ftor.result_; }
    }
    break;
  case Field::EDGE:
    break;
  case Field::FACE:
    break;
  case Field::CELL:
    break;
  case Field::NONE:
    cerr << "Error: Field data at location NONE!!" << endl;
    return false;
  } 

  return rval;
}

template <class Mesh, class FData>
GenericField<Mesh, FData>::interp_type* 
GenericField<Mesh, FData>::query_interpolate() const
{
  return new GInterp<value_type>(this); 
}


template <class Mesh, class FData>
void
GenericField<Mesh, FData>::resize_fdata()
{
  if (data_at() == NODE)
  {
    fdata().resize(get_typed_mesh()->nodes_size());
  }
  else if (data_at() == EDGE)
  {
    fdata().resize(get_typed_mesh()->edges_size());
  }
  else if (data_at() == FACE)
  {
    ASSERTFAIL("tetvol can't have data at faces (yet)");
  }
  else if (data_at() == CELL)
  {
    fdata().resize(get_typed_mesh()->cells_size());
  }
  else
  {
    ASSERTFAIL("data at unrecognized location");
  }
}


#if 0
template <class Mesh, class FData>
InterpolateToScalar* 
GenericField<Mesh, FData>::query_interpolate_to_scalar() const
{
  return new GInterp<double>(this);
}
#endif

// PIO
const int GENERICFIELD_VERSION = 1;


template <class Mesh, class FData>
Persistent *
GenericField<Mesh, FData>::maker()
{
  return scinew GenericField<Mesh, FData>;
}

template <class Mesh, class FData>
PersistentTypeID 
GenericField<Mesh, FData>::type_id(type_name(), "Field", maker);


template <class Mesh, class FData>
const string GenericField<Mesh, FData>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 2);
  if (n == -1)
  {
    static const string name = type_name(0) + FTNS + type_name(1) + FTNM
      + type_name(2) + FTNE;
    return name;
  }
  else if (n == 0)
  {
    static const string nm("GenericField");
    return nm;
  }
  else if (n == 1)
  {
    return find_type_name((Mesh *)0);
  }
  else
  {
    return find_type_name((FData *)0);
  }
}


template <class Mesh, class FData>
void GenericField<Mesh, FData>::io(Piostream& stream)
{
  stream.begin_class(type_name().c_str(), GENERICFIELD_VERSION);
  Field::io(stream);
  mesh_->io(stream);
  Pio(stream, fdata_);
  stream.end_class();
}


} // end namespace SCIRun

#endif // Datatypes_GenericField_h
















