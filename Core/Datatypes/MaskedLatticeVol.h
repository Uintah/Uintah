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
 *  MaskedTetVol.h
 *
 *  Written by:
 *   Martin Cole
 *   School of Computing
 *   University of Utah
 *
 *  Copyright (C) 2001 SCI Institute
 */

#ifndef Datatypes_MaskedLatticeVol_h
#define Datatypes_MaskedLatticeVol_h

#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/LatticeVol.h>
#include <Core/Datatypes/GenericField.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Persistent/PersistentSTL.h>
#include <vector>

namespace SCIRun {

template <class T> 
class MaskedLatticeVol : public LatticeVol<T> {
private:
  FData3d<char> mask_;  // since Pio isn't implemented for bool's
public:
  FData3d<char>& mask() { return mask_; }

  MaskedLatticeVol() :
    LatticeVol<T>() {};
  MaskedLatticeVol(Field::data_location data_at) : 
    LatticeVol<T>(data_at) {};
  MaskedLatticeVol(LatVolMeshHandle mesh, Field::data_location data_at) : 
    LatticeVol<T>(mesh, data_at) 
  {
    resize_fdata();
  };

  virtual ~MaskedLatticeVol() {};

  bool value(T &val, typename LatVolMesh::Node::index_type idx) const
  { if (!mask_[idx]) return false; val = fdata()[idx]; return true; }
  bool value(T &val, typename LatVolMesh::Edge::index_type idx) const
  { if (!mask_[idx]) return false; val = fdata()[idx]; return true; }
  bool value(T &val, typename LatVolMesh::Face::index_type idx) const
  { if (!mask_[idx]) return false; val = fdata()[idx]; return true; }
  bool value(T &val, typename LatVolMesh::Cell::index_type idx) const
  { if (!mask_[idx]) return false; val = fdata()[idx]; return true; }

  void    io(Piostream &stream);

  void initialize_mask(char masked) {
    for (char *c = mask_.begin(); c != mask_.end(); ++c) *c=masked;
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
    LatticeVol<T>::resize_fdata();
  }

  static  PersistentTypeID type_id;
  static const string type_name(int n = -1);
  virtual const string get_type_name(int n = -1) const { return type_name(n); }
  virtual const TypeDescription* get_type_description() const;
private:
  static Persistent *maker();
};

// Pio defs.
const int MASKED_LATTICE_VOL_VERSION = 1;

template <class T>
Persistent*
MaskedLatticeVol<T>::maker()
{
  return scinew MaskedLatticeVol<T>;
}

template <class T>
PersistentTypeID 
MaskedLatticeVol<T>::type_id(type_name(-1), 
			 LatticeVol<T>::type_name(-1),
			 maker);


template <class T>
void 
MaskedLatticeVol<T>::io(Piostream& stream)
{
  stream.begin_class(type_name(-1), MASKED_LATTICE_VOL_VERSION);
  LatticeVol<T>::io(stream);
  Pio(stream, mask_);
  stream.end_class();
}

template <class T> 
const string 
MaskedLatticeVol<T>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;

  }
  else if (n == 0)
  {
    return "MaskedLatticeVol";
  }
  else
  {
    return find_type_name((T *)0);
  }
}

template <class T>
const TypeDescription* 
get_type_description(MaskedLatticeVol<T>*)
{
  static TypeDescription* td = 0;
  static string name("MaskedLatticeVol");
  static string namesp("SCIRun");
  static string path(__FILE__);
  if(!td){
    const TypeDescription *sub = SCIRun::get_type_description((T*)0);
    TypeDescription::td_vec *subs = scinew TypeDescription::td_vec(1);
    (*subs)[0] = sub;
    td = scinew TypeDescription(name, subs, path, namesp);
  }
  return td;
}

template <class T>
const TypeDescription* 
MaskedLatticeVol<T>::get_type_description() const 
{
  return SCIRun::get_type_description((MaskedLatticeVol<T>*)0);
}

} // end namespace SCIRun

#endif // Datatypes_MaskedLatticeVol_h
