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
 *  MaskedHexVolField.h
 *
 *  Written by:
 *   Martin Cole
 *   School of Computing
 *   University of Utah
 *
 *  Copyright (C) 2001 SCI Institute
 */

#ifndef Datatypes_MaskedHexVolField_h
#define Datatypes_MaskedHexVolField_h

#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/HexVolField.h>
#include <Core/Datatypes/GenericField.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Persistent/PersistentSTL.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {

template <class T> 
class MaskedHexVolField : public HexVolField<T> {
private:
  vector<char> mask_;  // since Pio isn't implemented for bool's
public:
  vector<char>& mask() { return mask_; }

  MaskedHexVolField() :
    HexVolField<T>() {};
  MaskedHexVolField(Field::data_location data_at) : 
    HexVolField<T>(data_at) {};
  MaskedHexVolField(HexVolMeshHandle mesh, Field::data_location data_at) : 
    HexVolField<T>(mesh, data_at) 
  {
    resize_fdata();
  };

  bool get_valid_nodes_and_data(vector<pair<HexVolMesh::Node::index_type, T> > &data) {
    data.erase(data.begin(), data.end());
    if (data_at() != NODE) return false;
    HexVolMesh::Node::iterator ni, nie;
    get_typed_mesh()->begin(ni);
    get_typed_mesh()->end(nie);
    for (; ni != nie; ++ni) { 
      if (mask_[*ni]) { 
	pair<HexVolMesh::Node::index_type, T> p;
	p.first=*ni; 
	p.second=fdata()[*ni];
	data.push_back(p);
      }
    }
    return true;
  }

  virtual ~MaskedHexVolField() {};

  bool value(T &val, typename HexVolMesh::Node::index_type i) const
  { if (!mask_[i]) return false; val = fdata()[i]; return true; }
  bool value(T &val, typename HexVolMesh::Edge::index_type i) const
  { if (!mask_[i]) return false; val = fdata()[i]; return true; }
  bool value(T &val, typename HexVolMesh::Face::index_type i) const
  { if (!mask_[i]) return false; val = fdata()[i]; return true; }
  bool value(T &val, typename HexVolMesh::Cell::index_type i) const
  { if (!mask_[i]) return false; val = fdata()[i]; return true; }

  void initialize_mask(char masked) {
    for (vector<char>::iterator c = mask_.begin(); c != mask_.end(); ++c) *c=masked;
  }

  // Have to be explicit about where mesh_type comes from for IBM xlC
  // compiler... is there a better way to do this?
  typedef GenericField<HexVolMesh,vector<T> > GF;

  void resize_fdata() {
    if (data_at() == NODE)
    {
      typename GF::mesh_type::Node::size_type ssize;
      get_typed_mesh()->size(ssize);
      mask_.resize(ssize);
    }
    else if (data_at() == EDGE)
    {
      typename GF::mesh_type::Edge::size_type ssize;
      get_typed_mesh()->size(ssize);
      mask_.resize(ssize);
    }
    else if (data_at() == FACE)
    {
      typename GF::mesh_type::Face::size_type ssize;
      get_typed_mesh()->size(ssize);
      mask_.resize(ssize);
    }
    else if (data_at() == CELL)
    {
      typename GF::mesh_type::Cell::size_type ssize;
      get_typed_mesh()->size(ssize);
      mask_.resize(ssize);
    }
    else
    {
      ASSERTFAIL("data at unrecognized location");
    }
    HexVolField<T>::resize_fdata();
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
const int MASKED_HEX_VOL_FIELD_VERSION = 1;

template <class T>
Persistent*
MaskedHexVolField<T>::maker()
{
  return scinew MaskedHexVolField<T>;
}

template <class T>
PersistentTypeID 
MaskedHexVolField<T>::type_id(type_name(-1), 
			 HexVolField<T>::type_name(-1),
			 maker);


template <class T>
void 
MaskedHexVolField<T>::io(Piostream& stream)
{
  /*int version=*/stream.begin_class(type_name(-1), 
				     MASKED_HEX_VOL_FIELD_VERSION);
  HexVolField<T>::io(stream);
  Pio(stream, mask_);
  stream.end_class();
}

template <class T> 
const string 
MaskedHexVolField<T>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;

  }
  else if (n == 0)
  {
    return "MaskedHexVolField";
  }
  else
  {
    return find_type_name((T *)0);
  }
}

template <class T> 
const TypeDescription*
MaskedHexVolField<T>::get_type_description(int n) const
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

#endif // Datatypes_MaskedHexVolField_h
