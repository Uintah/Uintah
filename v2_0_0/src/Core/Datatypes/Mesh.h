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
#include <Core/Containers/LockingHandle.h>

namespace SCIRun {
  class BBox;
  class Mesh;
  class Transform;
  class TypeDescription;
  typedef LockingHandle<Mesh> MeshHandle;


class Mesh : public PropertyManager {
public:

  virtual Mesh *clone() = 0;

  Mesh();
  virtual ~Mesh();
  
  //! Required virtual functions.
  virtual BBox get_bounding_box() const = 0;

  //! Destructively applies the given transform to the mesh.
  virtual void transform(const Transform &t) = 0;

  //! Return the transformation that takes a 0-1 space bounding box 
  //! to the current bounding box of this mesh.
  virtual void get_canonical_transform(Transform &t);

  enum
  { 
    NONE_E		= 0,
    NODES_E		= 1 << 0,
    EDGES_E		= 1 << 1,
    FACES_E		= 1 << 2,
    CELLS_E		= 1 << 3,
    ALL_ELEMENTS_E      = NODES_E | EDGES_E | FACES_E | CELLS_E,
    NORMALS_E		= 1 << 4,
    NODE_NEIGHBORS_E	= 1 << 5,
    EDGE_NEIGHBORS_E	= 1 << 6,
    FACE_NEIGHBORS_E	= 1 << 7,
    LOCATE_E		= 1 << 8
  };
  virtual bool		synchronize(unsigned int) { return false; };


  //! Optional virtual functions.
  //! finish all computed data within the mesh.
  virtual bool has_normals() const { return false; }
  virtual bool is_editable() const { return false; } // supports add_elem(...)
  virtual int  dimensionality() const = 0;
  // Required interfaces
  
  //! Persistent I/O.
  void    io(Piostream &stream);
  static  PersistentTypeID type_id;
  static  const string type_name(int n = -1);
  virtual const string get_type_name(int n = -1) const;
  virtual const TypeDescription *get_type_description() const = 0;
};




} // end namespace SCIRun

#endif // Datatypes_Mesh_h
