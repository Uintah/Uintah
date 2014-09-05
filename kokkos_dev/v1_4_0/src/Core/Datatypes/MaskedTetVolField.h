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
 *  MaskedTetVolField.h
 *
 *  Written by:
 *   Martin Cole
 *   School of Computing
 *   University of Utah
 *
 *  Copyright (C) 2001 SCI Institute
 */

#ifndef Datatypes_MaskedTetVolField_h
#define Datatypes_MaskedTetVolField_h

#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/TetVolField.h>
#include <Core/Datatypes/GenericField.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Persistent/PersistentSTL.h>
#include <vector>

namespace SCIRun {

template <class T> 
class MaskedTetVolField : public TetVolField<T> {
private:
  vector<char> mask_;  // since Pio isn't implemented for bool's
public:
  vector<char>& mask() { return mask_; }

  MaskedTetVolField() :
    TetVolField<T>() {};
  MaskedTetVolField(Field::data_location data_at) : 
    TetVolField<T>(data_at) {};
  MaskedTetVolField(TetVolMeshHandle mesh, Field::data_location data_at) : 
    TetVolField<T>(mesh, data_at) 
  {
    resize_fdata();
  };

  bool get_valid_nodes_and_data(vector<pair<TetVolMesh::Node::index_type, T> > &data) {
    data.erase(data.begin(), data.end());
    if (data_at() != NODE) return false;
    TetVolMesh::Node::iterator ni, nie;
    get_typed_mesh()->begin(ni);
    get_typed_mesh()->end(nie);
    for (; ni != nie; ++ni) { 
      if (mask_[*ni]) { 
	pair<TetVolMesh::Node::index_type, T> p;
	p.first=*ni; 
	p.second=fdata()[*ni];
	data.push_back(p);
      }
    }
    return true;
  }


  virtual ~MaskedTetVolField() {};

  bool value(T &val, typename TetVolMesh::Node::index_type i) const
  { if (!mask_[i]) return false; val = fdata()[i]; return true; }
  bool value(T &val, typename TetVolMesh::Edge::index_type i) const
  { if (!mask_[i]) return false; val = fdata()[i]; return true; }
  bool value(T &val, typename TetVolMesh::Face::index_type i) const
  { if (!mask_[i]) return false; val = fdata()[i]; return true; }
  bool value(T &val, typename TetVolMesh::Cell::index_type i) const
  { if (!mask_[i]) return false; val = fdata()[i]; return true; }

  void    io(Piostream &stream);

  void initialize_mask(char masked) {
    for (vector<char>::iterator c = mask_.begin(); c != mask_.end(); ++c) *c=masked;
  }

  void resize_fdata() {
    if (data_at() == NODE)
    {
      typename mesh_type::Node::size_type ssize;
      get_typed_mesh()->size(ssize);
      mask_.resize(ssize);
    }
    else if (data_at() == EDGE)
    {
      typename mesh_type::Edge::size_type ssize;
      get_typed_mesh()->size(ssize);
      mask_.resize(ssize);
    }
    else if (data_at() == FACE)
    {
      typename mesh_type::Face::size_type ssize;
      get_typed_mesh()->size(ssize);
      mask_.resize(ssize);
    }
    else if (data_at() == CELL)
    {
      typename mesh_type::Cell::size_type ssize;
      get_typed_mesh()->size(ssize);
      mask_.resize(ssize);
    }
    else
    {
      ASSERTFAIL("data at unrecognized location");
    }
    TetVolField<T>::resize_fdata();
  }

  static  PersistentTypeID type_id;
  static const string type_name(int n = -1);
  virtual const string get_type_name(int n = -1) const { return type_name(n); }
  virtual const TypeDescription* get_type_description() const;
private:
  static Persistent *maker();
};

// Pio defs.
const int MASKED_TET_VOL_FIELD_VERSION = 1;

template <class T>
Persistent*
MaskedTetVolField<T>::maker()
{
  return scinew MaskedTetVolField<T>;
}

template <class T>
PersistentTypeID 
MaskedTetVolField<T>::type_id(type_name(-1), 
			 TetVolField<T>::type_name(-1),
			 maker);


template <class T>
void 
MaskedTetVolField<T>::io(Piostream& stream)
{
  /*int version=*/stream.begin_class(type_name(-1), 
				     MASKED_TET_VOL_FIELD_VERSION);
  TetVolField<T>::io(stream);
  Pio(stream, mask_);
  stream.end_class();
}

template <class T> 
const string 
MaskedTetVolField<T>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;

  }
  else if (n == 0)
  {
    return "MaskedTetVolField";
  }
  else
  {
    return find_type_name((T *)0);
  }
}


template <class T>
const TypeDescription* 
get_type_description(MaskedTetVolField<T>*)
{
  static TypeDescription* mtv_td = 0;
  if(!mtv_td){
    const TypeDescription *sub = SCIRun::get_type_description((T*)0);
    TypeDescription::td_vec *subs = scinew TypeDescription::td_vec(1);
    (*subs)[0] = sub;
    mtv_td = scinew TypeDescription("MaskedTetVolField", subs, __FILE__, "SCIRun");
  }
  return mtv_td;
}

template <class T>
const TypeDescription* 
MaskedTetVolField<T>::get_type_description() const 
{
  return SCIRun::get_type_description((MaskedTetVolField<T>*)0);
}

} // end namespace SCIRun

#endif // Datatypes_MaskedTetVolField_h
