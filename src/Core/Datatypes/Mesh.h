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
#include <bitset>

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
  virtual void transform(Transform &t) = 0;

  enum
  { 
    NONE_E = 0,
    NODES_E,
    EDGES_E,
    FACES_E,
    CELLS_E,
    //    ELEMENTS_E,
    NORMALS_E,
    NODE_NEIGHBORS_E,
    EDGE_NEIGHBORS_E,
    FACE_NEIGHBORS_E,
    GRID_E,
    SYNCHRONIZE_COUNT
  };

  typedef bitset<SYNCHRONIZE_COUNT>     synchronized_t;
  virtual bool		synchronize(const synchronized_t &which) { return false; };


  //! Optional virtual functions.
  //! finish all computed data within the mesh.
  virtual bool has_normals() const { return false; }
  virtual bool is_editable() const { return false; } // supports add_elem(...)
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
