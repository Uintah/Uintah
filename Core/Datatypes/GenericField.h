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
#include <Core/Datatypes/TypeName.h>
#include <Core/Containers/LockingHandle.h>
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
    fdata_(fdata_type())
  {};
  GenericField(data_location data_at) : 
    Field(data_at),
    mesh_(mesh_handle_type(new mesh_type())),
    fdata_(fdata_type()) 
  {};
  virtual ~GenericField() {};

  //! Required virtual functions from field base.
  virtual MeshBaseHandle get_mesh() const
  { return MeshBaseHandle(mesh_.get_rep()); }

  //! Required interfaces from field base.
  virtual InterpolateToScalar* query_interpolate_to_scalar() const;

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

  mesh_handle_type get_typed_mesh() const { return mesh_; };

  //! Persistent I/O.
  void    io(Piostream &stream);
  static  PersistentTypeID type_id;
  static  const string type_name(int);
  virtual const string get_type_name(int n) const { return type_name(n); }

private:
  //! generic interpolate object.
  struct GInterp : public InterpolateToScalar {
    
    bool interpolate(const Point& p, double& value) const;
  };


  //! A Tetrahedral Mesh.
  mesh_handle_type             mesh_;
  //! Data container.
  fdata_type                   fdata_;
}; 

// Virtual interface.
// internal interp object 
template <class Mesh, class FData>
bool 
GenericField<Mesh, FData>::GInterp::interpolate(const Point& /*p*/, 
						double& /*value*/) const
{
  cerr << "Error: NO interp defined!" << endl;
  assert(0);
}

template <class Mesh, class FData>
InterpolateToScalar* 
GenericField<Mesh, FData>::query_interpolate_to_scalar() const
{
  return new GInterp();
}
   

// PIO
const double GENERICFIELD_VERSION = 1.0;

template <class Mesh, class FData>
const string GenericField<Mesh, FData>::type_name(int)
{
  static string name = "GenericField<" + find_type_name((Mesh *)0) + ", "
    + find_type_name((FData *)0) + ">";
  return name;
}


template <class Mesh, class FData>
void GenericField<Mesh, FData>::io(Piostream& stream)
{
  stream.begin_class(type_name(0).c_str(), GENERICFIELD_VERSION);
  Field::io(stream);
  mesh_->io(stream);
  Pio(stream, fdata_);
  stream.end_class();
}

} // end namespace SCIRun

#endif // Datatypes_GenericField_h
















