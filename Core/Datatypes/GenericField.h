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

  GenericField() : 
    Field(),
    mesh_(mesh_handle_type(new mesh_type())),
    fdata_(fdata_type()),
    has_minmax(false)
  {};

  GenericField(data_location data_at) : 
    Field(data_at),
    mesh_(mesh_handle_type(new mesh_type())),
    fdata_(fdata_type()),
    has_minmax(false)
  {};

  virtual ~GenericField() {};

  //! Required virtual functions from field base.
  virtual MeshBaseHandle mesh() const
  { return MeshBaseHandle(mesh_.get_rep()); }

  //! Required interfaces from field base.
  virtual InterpolateToScalar* query_interpolate_to_scalar() const;
  bool get_minmax( double &, double &);
  bool compute_minmax();

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

  fdata_type& fdata() { return fdata_; }
  const fdata_type& fdata() const { return fdata_; }

  mesh_handle_type get_typed_mesh() const { return mesh_; };

  //! Persistent I/O.
  virtual void io(Piostream &stream);
  static  PersistentTypeID type_id;
  static  const string type_name(int n = -1);
  virtual const string get_type_name(int n = -1) const { return type_name(n); }

private:
  //! generic interpolate object.
  struct GInterp : public InterpolateToScalar {
    GInterp(const GenericField<Mesh, FData> *f) :
      f_(f) {}
    bool interpolate(const Point& p, double& value) const;
    const GenericField<Mesh, FData> *f_;
  };


  //! A (generic) mesh.
  mesh_handle_type             mesh_;
  //! Data container.
  fdata_type                   fdata_;
  //! minmax
  bool has_minmax;
  double min_, max_;
}; 

// Virtual interface.
// internal interp object 
template <class Mesh, class FData>
bool 
GenericField<Mesh, FData>::GInterp::interpolate(const Point& p, 
						double& value) const
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
InterpolateToScalar* 
GenericField<Mesh, FData>::query_interpolate_to_scalar() const
{
  return new GInterp(this);
}
   
template <class Mesh, class FData>
bool
GenericField<Mesh, FData>::compute_minmax()
{
  typename Mesh::node_iterator i = mesh_->node_begin();
  if ( i == mesh_->node_end() )
    return false;
  min_ = max_ = fdata_[*i];
  for (++i; i != mesh_->node_end(); ++i) {
    value_type v = fdata_[*i];
    if ( v < min_ ) min_ = v;
    else if ( v > max_ ) max_ = v;
  }
  
  return true;
}

template<class Mesh, class FData>
bool
GenericField<Mesh, FData>::get_minmax( double &min, double &max) 
{
  if ( !has_minmax ) { 
    if ( !compute_minmax() )
      return false;
    has_minmax = true;
  }

  min = min_;
  max = max_;

  return true;
}

#if defined(__sgi)  
// Turns off REMARKS like this:
//cc-1424 CC: REMARK File = ./Core/Datatypes/TetVol.h, Line = 45
//The template parameter "T" is not used in declaring the argument types of
//          function template "SCIRun::make_TetVol".
 
#pragma set woff 1424
#endif

// PIO
const double GENERICFIELD_VERSION = 1.0;


template <class Mesh, class FData>
Persistent* make_GenericField()
{
  return scinew GenericField<Mesh, FData>;
}

template <class Mesh, class FData>
PersistentTypeID 
GenericField<Mesh, FData>::type_id(type_name(), 
				   "Field",
				   &make_GenericField<Mesh, FData>);


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
    return "GenericField";
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

#if defined(__sgi)  
#pragma reset woff 1424
#endif

} // end namespace SCIRun

#endif // Datatypes_GenericField_h
















