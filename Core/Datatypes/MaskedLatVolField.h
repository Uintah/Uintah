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
 *  MaskedLatVolField.h
 *
 *  Written by:
 *   Martin Cole
 *   School of Computing
 *   University of Utah
 *
 *  Copyright (C) 2001 SCI Institute
 */

#ifndef Datatypes_MaskedLatVolField_h
#define Datatypes_MaskedLatVolField_h

#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/LatVolField.h>
#include <Core/Datatypes/GenericField.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Persistent/PersistentSTL.h>
#include <vector>

namespace SCIRun {

template <class T> 
class MaskedLatVolField : public LatVolField<T> {
private:
  FData3d<char> mask_;  // since Pio isn't implemented for bool's
public:
  FData3d<char>& mask() { return mask_; }
  
  bool get_valid_nodes_and_data(vector<LatVolMesh::Node::index_type> &nodes,
				vector<T> &data) {
    nodes.resize(0);
    data.resize(0);
    if (data_at() != NODE) return false;
    LatVolMesh::Node::iterator ni, nie;
    get_typed_mesh()->begin(ni);
    get_typed_mesh()->end(nie);
    for (; ni != nie; ++ni) { 
      if (mask_[*ni]) { nodes.push_back(*ni); data.push_back(fdata()[*ni]); }
    }
    return true;
  }

  MaskedLatVolField() : LatVolField<T>() {}
  MaskedLatVolField(LatVolMeshHandle mesh, Field::data_location data_at)
    : LatVolField<T>(mesh, data_at)
  {
    resize_fdata();
  }

  virtual ~MaskedLatVolField() {};

  bool value(T &val, typename LatVolMesh::Node::index_type idx) const
  { if (!mask_[idx]) return false; val = fdata()[idx]; return true; }
  bool value(T &val, typename LatVolMesh::Edge::index_type idx) const
  { if (!mask_[idx]) return false; val = fdata()[idx]; return true; }
  bool value(T &val, typename LatVolMesh::Face::index_type idx) const
  { if (!mask_[idx]) return false; val = fdata()[idx]; return true; }
  bool value(T &val, typename LatVolMesh::Cell::index_type idx) const
  { if (!mask_[idx]) return false; val = fdata()[idx]; return true; }

  void initialize_mask(char masked) {
    for (char *c = mask_.begin(); c != mask_.end(); ++c) *c=masked;
  }

  void resize_fdata() {
    if (data_at() == NODE)
    {
      typename LatVolField<T>::mesh_type::Node::size_type ssize;
      get_typed_mesh()->size(ssize);
      mask_.resize(ssize);
    }
    else if (data_at() == EDGE)
    {
      typename LatVolField<T>::mesh_type::Edge::size_type ssize;
      get_typed_mesh()->size(ssize);
      mask_.resize(ssize);
    }
    else if (data_at() == FACE)
    {
      typename LatVolField<T>::mesh_type::Face::size_type ssize;
      get_typed_mesh()->size(ssize);
      mask_.resize(ssize);
    }
    else if (data_at() == CELL)
    {
      typename LatVolField<T>::mesh_type::Cell::size_type ssize;
      get_typed_mesh()->size(ssize);
      mask_.resize(ssize);
    }
    else
    {
      ASSERTFAIL("data at unrecognized location");
    }
    LatVolField<T>::resize_fdata();
  }

  //! Persistent IO
  static PersistentTypeID type_id;
  virtual void io(Piostream &stream);

  static const string type_name(int n = -1);
  virtual const TypeDescription* get_type_description(int n = -1) const;

private:
  static Persistent *maker();
};

// Pio defs.
const int MASKED_LAT_VOL_FIELD_VERSION = 1;

template <class T>
Persistent*
MaskedLatVolField<T>::maker()
{
  return scinew MaskedLatVolField<T>;
}

template <class T>
PersistentTypeID 
MaskedLatVolField<T>::type_id(type_name(-1), 
			 LatVolField<T>::type_name(-1),
			 maker);


template <class T>
void 
MaskedLatVolField<T>::io(Piostream& stream)
{
  /*int version=*/stream.begin_class(type_name(-1), 
				     MASKED_LAT_VOL_FIELD_VERSION);
  LatVolField<T>::io(stream);
  Pio(stream, mask_);
  stream.end_class();
}

template <class T> 
const string 
MaskedLatVolField<T>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;

  }
  else if (n == 0)
  {
    return "MaskedLatVolField";
  }
  else
  {
    return find_type_name((T *)0);
  }
}

template <class T> 
const TypeDescription*
MaskedLatVolField<T>::get_type_description(int n) const
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

} // end namespace SCIRun

#endif // Datatypes_MaskedLatVolField_h








