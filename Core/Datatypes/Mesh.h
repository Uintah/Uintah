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

#ifndef Datatypes_Mesh_h
#define Datatypes_Mesh_h

#include <Core/Datatypes/PropertyManager.h>
#include <Core/Geometry/BBox.h>
#include <Core/Geometry/Transform.h>
#include <Core/Disclosure/TypeDescription.h>
#include <Core/Containers/LockingHandle.h>

namespace SCIRun {

class Mesh;
typedef LockingHandle<Mesh> MeshHandle;


class Mesh : public PropertyManager {
public:

  virtual Mesh *clone() = 0;

  Mesh();
  virtual ~Mesh();
  
  //! Required virtual functions.
  virtual BBox get_bounding_box() const = 0;
  virtual void transform(Transform &t) = 0;

  //! Optional virtual functions.
  //! finish all computed data within the mesh.
  virtual void flush_changes() {}; //Not all meshes need to do this.
  virtual bool has_normals() const { return false; }
  virtual bool is_editable() const { return false; } // supports add_elem(...)
  // Required interfaces
  
  //! Persistent I/O.
  void    io(Piostream &stream);
  static  PersistentTypeID type_id;
  static  const string type_name(int n = -1);
  //! All instantiable classes need to define this.
  virtual const string get_type_name(int n = -1) const { return type_name(n); }

  virtual const TypeDescription *get_type_description() const = 0;
};




} // end namespace SCIRun

#endif // Datatypes_Mesh_h
