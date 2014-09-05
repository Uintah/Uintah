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
 *  PointCloudMesh.h: countour mesh
 *
 *  Written by:
 *   Chris Moulding
 *   Department of Computer Science
 *   University of Utah
 *   January 2001
 *
 *  Copyright (C) 2001 SCI Group
 *
 */

#ifndef SCI_project_PointCloudMesh_h
#define SCI_project_PointCloudMesh_h 1

#include <Core/Geometry/Point.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Datatypes/MeshBase.h>
#include <Core/Datatypes/FieldIterator.h>
#include <string>
#include <vector>

namespace SCIRun {

using std::string;
using std::vector;

class SCICORESHARE PointCloudMesh : public MeshBase
{
public:
  typedef unsigned int under_type;

  //! Index and Iterator types required for Mesh Concept.
  struct Node {
    typedef NodeIndex<under_type>       index_type;
    typedef NodeIterator<under_type>    iterator;
    typedef NodeIndex<under_type>       size_type;
    typedef vector<index_type>          array_type;
  };

  struct Edge {
    typedef EdgeIndex<under_type>       index_type;
    typedef EdgeIterator<under_type>    iterator;
    typedef EdgeIndex<under_type>       size_type;
    typedef vector<index_type>          array_type;
  };

  struct Face {
    typedef FaceIndex<under_type>       index_type;
    typedef FaceIterator<under_type>    iterator;
    typedef FaceIndex<under_type>       size_type;
    typedef vector<index_type>          array_type;
  };

  struct Cell {
    typedef CellIndex<under_type>       index_type;
    typedef CellIterator<under_type>    iterator;
    typedef CellIndex<under_type>       size_type;
    typedef vector<index_type>          array_type;
  };

  typedef Node Elem;

  // storage types for get_* functions
  typedef vector<double>      weight_array;

  PointCloudMesh() {}
  PointCloudMesh(const PointCloudMesh &copy)
    : points_(copy.points_) {}
  virtual PointCloudMesh *clone() { return new PointCloudMesh(*this); }
  virtual ~PointCloudMesh() {}

  template <class I> I tbegin(I*) const;
  template <class I> I tend(I*) const;
  template <class S> S tsize(S*) const;

  Node::iterator node_begin() const;
  Node::iterator node_end() const;
  Edge::iterator edge_begin() const;
  Edge::iterator edge_end() const;
  Face::iterator face_begin() const;
  Face::iterator face_end() const;
  Cell::iterator cell_begin() const;
  Cell::iterator cell_end() const;

  //! get the mesh statistics
  Node::size_type nodes_size() const;
  Edge::size_type edges_size() const;
  Face::size_type faces_size() const;
  Cell::size_type cells_size() const;
  virtual BBox get_bounding_box() const;
  virtual void transform(Transform &t);

  //! set the mesh statistics
  void resize_nodes(Node::size_type n) { points_.resize(n); }

  //! get the child elements of the given index
  void get_nodes(Node::array_type &, Edge::index_type) const {}
  void get_nodes(Node::array_type &, Face::index_type) const {}
  void get_nodes(Node::array_type &, Cell::index_type) const {}
  void get_edges(Edge::array_type &, Face::index_type) const {}
  void get_edges(Edge::array_type &, Cell::index_type) const {}
  void get_faces(Face::array_type &, Cell::index_type) const {}

  //! get the parent element(s) of the given index
  unsigned get_edges(Edge::array_type &, Node::index_type) const { return 0; }
  unsigned get_faces(Face::array_type &, Node::index_type) const { return 0; }
  unsigned get_faces(Face::array_type &, Edge::index_type) const { return 0; }
  unsigned get_cells(Cell::array_type &, Node::index_type) const { return 0; }
  unsigned get_cells(Cell::array_type &, Edge::index_type) const { return 0; }
  unsigned get_cells(Cell::array_type &, Face::index_type) const { return 0; }

  //! similar to get_edges() with Node::index_type argument, but
  //  returns the "other" edge if it exists, not all that exist
  void get_neighbor(Edge::index_type &, Node::index_type) const {}

  //! get the center point (in object space) of an element
  void get_center(Point &result, Node::index_type idx) const
  { result = points_[idx]; }
  void get_center(Point &, Edge::index_type) const {}
  void get_center(Point &, Face::index_type) const {}
  void get_center(Point &, Cell::index_type) const {}

  bool locate(Node::index_type &, const Point &) const;
  bool locate(Edge::index_type &, const Point &) const { return false; }
  bool locate(Face::index_type &, const Point &) const { return false; }
  bool locate(Cell::index_type &, const Point &) const { return false; }

  void get_weights(const Point &p, Node::array_type &l, vector<double> &w);
  void get_weights(const Point &, Edge::array_type &, vector<double> &) {ASSERTFAIL("PointCloud::get_weights for edges isn't supported");}
  void get_weights(const Point &, Face::array_type &, vector<double> &) {ASSERTFAIL("PointCloud::get_weights for faces isn't supported");}
  void get_weights(const Point &, Cell::array_type &, vector<double> &) {ASSERTFAIL("PointCloud::get_weights for cells isn't supported");}

  void get_point(Point &result, Node::index_type idx) const
  { get_center(result,idx); }
  void get_normal(Vector & /* result */, Node::index_type /* index */) const
  { ASSERTFAIL("not implemented") }
  void set_point(const Point &point, Node::index_type index)
  { points_[index] = point; }

  //! use these to build up a new PointCloud mesh
  Node::index_type add_node(Point p)
  { points_.push_back(p); return points_.size()-1; }

  Node::index_type add_point(const Point &p);

  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  static  const string type_name(int n = -1);
  virtual const string get_type_name(int n = -1) const { return type_name(n); }

  virtual const TypeDescription *get_type_description() const;

private:

  //! the nodes
  vector<Point> points_;

  // returns a PointCloudMesh
  static Persistent *maker() { return new PointCloudMesh(); }
};

typedef LockingHandle<PointCloudMesh> PointCloudMeshHandle;

template <> PointCloudMesh::Node::size_type PointCloudMesh::tsize(PointCloudMesh::Node::size_type *) const;
template <> PointCloudMesh::Edge::size_type PointCloudMesh::tsize(PointCloudMesh::Edge::size_type *) const;
template <> PointCloudMesh::Face::size_type PointCloudMesh::tsize(PointCloudMesh::Face::size_type *) const;
template <> PointCloudMesh::Cell::size_type PointCloudMesh::tsize(PointCloudMesh::Cell::size_type *) const;
				
template <> PointCloudMesh::Node::iterator PointCloudMesh::tbegin(PointCloudMesh::Node::iterator *) const;
template <> PointCloudMesh::Edge::iterator PointCloudMesh::tbegin(PointCloudMesh::Edge::iterator *) const;
template <> PointCloudMesh::Face::iterator PointCloudMesh::tbegin(PointCloudMesh::Face::iterator *) const;
template <> PointCloudMesh::Cell::iterator PointCloudMesh::tbegin(PointCloudMesh::Cell::iterator *) const;
				
template <> PointCloudMesh::Node::iterator PointCloudMesh::tend(PointCloudMesh::Node::iterator *) const;
template <> PointCloudMesh::Edge::iterator PointCloudMesh::tend(PointCloudMesh::Edge::iterator *) const;
template <> PointCloudMesh::Face::iterator PointCloudMesh::tend(PointCloudMesh::Face::iterator *) const;
template <> PointCloudMesh::Cell::iterator PointCloudMesh::tend(PointCloudMesh::Cell::iterator *) const;

const TypeDescription* get_type_description(PointCloudMesh *);
const TypeDescription* get_type_description(PointCloudMesh::Node *);
const TypeDescription* get_type_description(PointCloudMesh::Edge *);
const TypeDescription* get_type_description(PointCloudMesh::Face *);
const TypeDescription* get_type_description(PointCloudMesh::Cell *);

} // namespace SCIRun

#endif // SCI_project_PointCloudMesh_h





