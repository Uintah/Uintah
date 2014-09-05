/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/


#ifndef Datatypes_Mesh_h
#define Datatypes_Mesh_h

#include <sci_defs/hashmap_defs.h>

#include <Core/Datatypes/PropertyManager.h>
#include <Core/Containers/LockingHandle.h>

namespace SCIRun {

class BBox;
class Mesh;
class Transform;
class TypeDescription;
typedef LockingHandle<Mesh> MeshHandle;

// Maximum number of weights get_weights will return.
// Currently at 15 for QuadraticLatVolMesh.
#define MESH_WEIGHT_MAXSIZE 15
#define MESH_NO_NEIGHBOR 0xffffffff // 32 bit under_type!

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
  virtual bool get_dim(vector<unsigned int>&) const { return false;  }
  // Required interfaces
  
  //! Persistent I/O.
  void    io(Piostream &stream);
  static  PersistentTypeID type_id;
  static  const string type_name(int n = -1);
  virtual const string get_type_name(int n = -1) const;
  virtual const TypeDescription *get_type_description() const = 0;

  // The minimum value for elemental checking
  double MIN_ELEMENT_VAL;
};

} // end namespace SCIRun

#endif // Datatypes_Mesh_h
