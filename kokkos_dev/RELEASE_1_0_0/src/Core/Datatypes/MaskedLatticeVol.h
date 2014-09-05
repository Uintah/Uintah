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

  bool value(T &val, typename LatVolMesh::node_index idx) const
  { if (!mask_(idx)) return false; val = fdata_[idx]; return true; }
  bool value(T &val, typename LatVolMesh::edge_index idx) const
  { if (!mask_(idx)) return false; val = fdata_[idx]; return true; }
  bool value(T &val, typename LatVolMesh::face_index idx) const
  { if (!mask_(idx)) return false; val = fdata_[idx]; return true; }
  bool value(T &val, typename LatVolMesh::cell_index idx) const
  { if (!mask_(idx)) return false; val = fdata_[idx]; return true; }

  void    io(Piostream &stream);

  void initialize_mask(char masked) {
    for (char *c = mask_.begin(); c != mask_.end(); ++c) *c=masked;
  }

  void resize_fdata() {
    if (data_at() == NODE)
      mask_.resize(get_typed_mesh()->nodes_size());
    else if (data_at() == EDGE)
      mask_.resize(get_typed_mesh()->edges_size());
    else if (data_at() == FACE)
      mask_.resize(get_typed_mesh()->faces_size());
    else if (data_at() == CELL)
      mask_.resize(get_typed_mesh()->cells_size());
    else
      ASSERTFAIL("data at unrecognized location")
    LatticeVol<T>::resize_fdata();
  }

  static  PersistentTypeID type_id;
  static const string type_name(int n = -1);
  virtual const string get_type_name(int n = -1) const { return type_name(n); }

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
			 GenericField<LatVolMesh, vector<T> >::type_name(-1),
			 maker);


template <class T>
void 
MaskedLatticeVol<T>::io(Piostream& stream)
{
  stream.begin_class(type_name(-1).c_str(), MASKED_LATTICE_VOL_VERSION);
  GenericField<LatVolMesh, FData3d<T> >::io(stream);
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

} // end namespace SCIRun

#endif // Datatypes_MaskedLatticeVol_h
