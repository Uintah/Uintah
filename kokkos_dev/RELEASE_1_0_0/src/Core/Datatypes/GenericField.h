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

#include <Core/Datatypes/builtin.h>
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
  GenericField();
  GenericField(data_location data_at);
  GenericField(mesh_handle_type mesh, data_location data_at);

  virtual ~GenericField();

  virtual GenericField<Mesh, FData> *clone() const;

  //! Required virtual functions from field base.
  virtual MeshBaseHandle mesh() const;
  virtual void mesh_detach();

  //! Required interfaces from field base.
  virtual interp_type* query_interpolate() const;

  virtual bool is_scalar() const;

  //! Required interface to support Field Concept.
  bool value(value_type &val, typename mesh_type::node_index i) const;
  bool value(value_type &val, typename mesh_type::edge_index i) const;
  bool value(value_type &val, typename mesh_type::face_index i) const;
  bool value(value_type &val, typename mesh_type::cell_index i) const;

  //! Required interface to support Field Concept.
  void set_value(const value_type &val, typename mesh_type::node_index i);
  void set_value(const value_type &val, typename mesh_type::edge_index i);
  void set_value(const value_type &val, typename mesh_type::face_index i);
  void set_value(const value_type &val, typename mesh_type::cell_index i);

  //! No safety check for the following calls, be sure you know where data is.
  value_type value(typename mesh_type::node_index i) const;
  value_type value(typename mesh_type::edge_index i) const;
  value_type value(typename mesh_type::face_index i) const;
  value_type value(typename mesh_type::cell_index i) const;

  virtual void resize_fdata();

  fdata_type& fdata();
  const fdata_type& fdata() const;

  mesh_handle_type get_typed_mesh() const;

  //! Persistent I/O.
  virtual void io(Piostream &stream);
  static  PersistentTypeID type_id;
  static  const string type_name(int n = -1);
  virtual const string get_type_name(int n = -1) const;

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
    ASSERTFAIL("fields can't have data at faces (yet)");
  }
  else if (data_at() == CELL)
  {
    fdata().resize(get_typed_mesh()->cells_size());
  }
  else if (data_at() == NONE)
  {
    // do nothing (really, we want to resize to zero)
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




template <class Mesh, class FData>
GenericField<Mesh, FData>::GenericField()
  : Field(),
    mesh_(mesh_handle_type(new mesh_type()))
{
}

template <class Mesh, class FData>
GenericField<Mesh, FData>::GenericField(data_location data_at)
  : Field(data_at),
    mesh_(mesh_handle_type(new mesh_type())) 
{
}

template <class Mesh, class FData>
GenericField<Mesh, FData>::GenericField(mesh_handle_type mesh, data_location data_at)
  : Field(data_at),
    mesh_(mesh)
{
  if (data_at != NONE)
    resize_fdata();
}

template <class Mesh, class FData>
GenericField<Mesh, FData>::~GenericField()
{
}

template <class Mesh, class FData>
GenericField<Mesh, FData> *
GenericField<Mesh, FData>::clone() const
{
  return new GenericField<Mesh, FData>(*this);
}

template <class Mesh, class FData>
MeshBaseHandle
GenericField<Mesh, FData>::mesh() const
{
  return MeshBaseHandle(mesh_.get_rep());
}

template <class Mesh, class FData>
void
GenericField<Mesh, FData>::mesh_detach()
{
  mesh_.detach();
}

template <class Mesh, class FData>
bool
GenericField<Mesh, FData>::is_scalar() const
{
  return ::SCIRun::is_scalar<value_type>();
}

template <class Mesh, class FData>
bool
GenericField<Mesh, FData>::value(value_type &val, typename mesh_type::node_index i) const
{
  if (data_at() != NODE) return false; val = fdata_[i]; return true;
}

template <class Mesh, class FData>
bool
GenericField<Mesh, FData>::value(value_type &val, typename mesh_type::edge_index i) const
{
  if (data_at() != EDGE) return false; val = fdata_[i]; return true;
}

template <class Mesh, class FData>
bool
GenericField<Mesh, FData>::value(value_type &val, typename mesh_type::face_index i) const
{
  if (data_at() != FACE) return false; val = fdata_[i]; return true;
}

template <class Mesh, class FData>
bool
GenericField<Mesh, FData>::value(value_type &val, typename mesh_type::cell_index i) const
{
  if (data_at() != CELL) return false; val = fdata_[i]; return true;
} 

//! Required interface to support Field Concept.
template <class Mesh, class FData>
void
GenericField<Mesh, FData>::set_value(const value_type &val, typename mesh_type::node_index i)
{
  fdata_[i] = val;
}
template <class Mesh, class FData>
void
GenericField<Mesh, FData>::set_value(const value_type &val, typename mesh_type::edge_index i)
{
  fdata_[i] = val;
}
template <class Mesh, class FData>
void
GenericField<Mesh, FData>::set_value(const value_type &val, typename mesh_type::face_index i)
{
  fdata_[i] = val;
}
template <class Mesh, class FData>
void
GenericField<Mesh, FData>::set_value(const value_type &val, typename mesh_type::cell_index i)
{
  fdata_[i] = val;
}

template <class Mesh, class FData>
GenericField<Mesh, FData>::value_type
GenericField<Mesh, FData>::value(typename mesh_type::node_index i) const
{
  return fdata_[i];
}
template <class Mesh, class FData>
GenericField<Mesh, FData>::value_type
GenericField<Mesh, FData>::value(typename mesh_type::edge_index i) const
{
  return fdata_[i];
}
template <class Mesh, class FData>
GenericField<Mesh, FData>::value_type
GenericField<Mesh, FData>::value(typename mesh_type::face_index i) const 
{
  return fdata_[i];
}
template <class Mesh, class FData>
GenericField<Mesh, FData>::value_type
GenericField<Mesh, FData>::value(typename mesh_type::cell_index i) const 
{
  return fdata_[i];
}

template <class Mesh, class FData>
GenericField<Mesh, FData>::fdata_type &
GenericField<Mesh, FData>::fdata()
{
  return fdata_;
}

template <class Mesh, class FData>
const GenericField<Mesh, FData>::fdata_type&
GenericField<Mesh, FData>::fdata() const
{
  return fdata_;
}

template <class Mesh, class FData>
GenericField<Mesh, FData>::mesh_handle_type
GenericField<Mesh, FData>::get_typed_mesh() const
{
  return mesh_;
}

template <class Mesh, class FData>
const string
GenericField<Mesh, FData>::get_type_name(int n) const
{
  return type_name(n);
}


} // end namespace SCIRun

#endif // Datatypes_GenericField_h
















