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
#include <Core/Datatypes/MeshTet.h>
#include <Core/Containers/LockingHandle.h>
#include <vector>

namespace SCIRun {

template <class Mesh, class Data>
class GenericField: public Field 
{
public:
  //! Typedefs to support the Field concept.
  typedef Data                       value_type;
  typedef Mesh                       mesh_type;
  typedef Mesh::container_type<Data> fdata_type;
  typedef LockingHandle<Mesh>        mesh_handle_type;

  GenericField();
  GenericField(data_location data_at);
  virtual ~GenericField();

  //! Required virtual functions from field base.
  virtual MeshBaseHandle get_mesh() const;

  //! Required interfaces from field base.
  virtual InterpolateToScalar* query_interpolate_to_scalar() const;

  //! Required interface to support Field Concept.
  value_type operator[] (Mesh::node_index);
  value_type operator[] (Mesh::edge_index);
  value_type operator[] (Mesh::face_index);
  value_type operator[] (Mesh::cell_index);
  
  mesh_handle_type get_typed_mesh(); 

  //! Persistent I/O.
  void    io(Piostream &stream);
  static  PersistentTypeID type_id;
  static  const string type_name(int);
  virtual const string get_type_name(int n) const;
private:
  //! A Tetrahedral Mesh.
  mesh_handle_type             mesh_;
  //! Data container.
  fdata_type                   fdata_;
};

const double TET_VOL_VERSION = 1.0;

template <class Data>
void GenericField<Data>::io(Piostream& stream){

  stream.begin_class(typeName().c_str(), TET_VOL_VERSION);
  Field::io(stream);
  Pio(stream, mesh_.get_rep());
  Pio(stream, fdata_);
  stream.end_class();
}

} // end namespace SCIRun

#endif // Datatypes_GenericField_h
















