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
 *  StructHexVolMesh.h: Templated Mesh defined on an 3D Structured Grid
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   April 2002
 *
 *  Copyright (C) 2002 SCI Group
 *
 */

#ifndef SCI_project_StructHexVolMesh_h
#define SCI_project_StructHexVolMesh_h 1

#include <Core/Geometry/Point.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Datatypes/LatVolField.h>
#include <Core/share/share.h>
#include <string>
#include <iostream>

namespace SCIRun {

using std::string;

class SCICORESHARE StructHexVolMesh : public LatVolMesh
{
public:

  StructHexVolMesh() : grid_lock_("StructHexVolMesh grid lock") {}
  StructHexVolMesh(unsigned int x, unsigned int y, unsigned int z)
    : LatVolMesh(x, y, z, Point(0.0, 0.0, 0.0), Point(1.0, 1.0, 1.0)),
      points_(x, y, z),
      grid_lock_("StructHexVolMesh grid lock")
  {}
  StructHexVolMesh(const StructHexVolMesh &copy)
    : LatVolMesh(copy),
      //points_(copy.points_),
      grid_lock_("StructHexVolMesh grid lock")
  {}
  virtual StructHexVolMesh *clone() { return new StructHexVolMesh(*this); }
  virtual ~StructHexVolMesh() {}

  //! get the mesh statistics
  virtual BBox get_bounding_box() const;
  virtual void transform(Transform &t);

  //! get the center point (in object space) of an element
  void get_center(Point &, const Node::index_type &) const;
  void get_center(Point &, const Edge::index_type &) const {}
  void get_center(Point &, const Face::index_type &) const {}
  void get_center(Point &, const Cell::index_type &) const;

  bool locate(Node::index_type &, const Point &);
  bool locate(Edge::index_type &, const Point &) const { return false; }
  bool locate(Face::index_type &, const Point &) const { return false; }
  bool locate(Cell::index_type &, const Point &);

  void get_weights(const Point &, Node::array_type &, vector<double> &);
  void get_weights(const Point &, Edge::array_type &, vector<double> &)
  {ASSERTFAIL("StructHexVolMesh::get_weights for edges isn't supported");}
  void get_weights(const Point &, Face::array_type &, vector<double> &)
  {ASSERTFAIL("StructHexVolMesh::get_weights for faces isn't supported");}
  void get_weights(const Point &, Cell::array_type &, vector<double> &);

  void get_point(Point &point, const Node::index_type &index) const
  { get_center(point, index); }
  void set_point(const Node::index_type &index, const Point &point);

  void get_normal(Vector &vector, const Node::index_type &index) const
  { ASSERTFAIL("not implemented") }

  void get_random_point(Point &p, const Elem::index_type &ei, int seed=0) const
  { ASSERTFAIL("not implemented") }

  //! the double return val is the volume of the Hex.
  double get_gradient_basis(Cell::index_type ci, Vector& g0, Vector& g1,
			    Vector& g2, Vector& g3, Vector& g4,
			    Vector& g5, Vector& g6, Vector& g7)
  { ASSERTFAIL("not implemented") }

  //void compute_edges();
  //void compute_faces();
  //void compute_node_neighbors();
  void compute_grid();

  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  static  const string type_name(int n = -1);
  virtual const string get_type_name(int n = -1) const { return type_name(n); }

  virtual const TypeDescription *get_type_description() const;

private:
  double inside8_p(Cell::index_type i, const Point &p) const;

  Array3<Point> points_;

  typedef LockingHandle<LatVolField<vector<Cell::index_type> > > grid_handle;
  grid_handle                 grid_;
  Mutex                       grid_lock_; // Bad traffic!

  // returns a StructHexVolMesh
  static Persistent *maker() { return new StructHexVolMesh(); }
};

typedef LockingHandle<StructHexVolMesh> StructHexVolMeshHandle;

const TypeDescription* get_type_description(StructHexVolMesh *);
} // namespace SCIRun

#endif // SCI_project_StructHexVolMesh_h
