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

#ifndef Datatypes_MeshBase_h
#define Datatypes_MeshBase_h

#include <Core/Datatypes/PropertyManager.h>
#include <Core/Geometry/BBox.h>
#include <Core/Containers/LockingHandle.h>

namespace SCIRun {

class MeshBase : public PropertyManager {
public:

  virtual MeshBase *clone() = 0;
  virtual ~MeshBase();
  
  //! Required virtual functions
  virtual BBox get_bounding_box() const = 0;
  
  // Required interfaces


  //! Persistent I/O.
  void    io(Piostream &stream);
  static  PersistentTypeID type_id;
  static  const string type_name(int n = -1);
  //! All instantiable classes need to define this.
  virtual const string get_type_name(int n = -1) const { return type_name(n); }
};

typedef LockingHandle<MeshBase> MeshBaseHandle;

} // end namespace SCIRun

#endif // Datatypes_MeshBase_h
