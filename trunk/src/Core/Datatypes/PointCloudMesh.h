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
#include <Core/Datatypes/Mesh.h>
#include <Core/Datatypes/FieldIterator.h>
#include <Core/Containers/StackVector.h>
#include <sgi_stl_warnings_off.h>
#include <string>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {

using std::string;
using std::vector;

typedef unsigned int under_type;

//! Index and Iterator types required for Mesh Concept.
struct PointCloudMeshNode {
  typedef NodeIndex<under_type>       index_type;
  typedef NodeIterator<under_type>    iterator;
  typedef NodeIndex<under_type>       size_type;
  typedef StackVector<index_type, 1>  array_type;
};

struct PointCloudMeshEdge {
  typedef EdgeIndex<under_type>       index_type;
  typedef EdgeIterator<under_type>    iterator;
  typedef EdgeIndex<under_type>       size_type;
  typedef vector<index_type>          array_type;
};

struct PointCloudMeshFace {
  typedef FaceIndex<under_type>       index_type;
  typedef FaceIterator<under_type>    iterator;
  typedef FaceIndex<under_type>       size_type;
  typedef vector<index_type>          array_type;
};

struct PointCloudMeshCell {
  typedef CellIndex<under_type>       index_type;
  typedef CellIterator<under_type>    iterator;
  typedef CellIndex<under_type>       size_type;
  typedef vector<index_type>          array_type;
};


class PointCloudMesh : public Mesh
{
public:

  typedef PointCloudMeshNode Node;
  typedef PointCloudMeshEdge Edge;
  typedef PointCloudMeshFace Face;
  typedef PointCloudMeshCell Cell;
  typedef Node Elem;

  PointCloudMesh() {}
  PointCloudMesh(const PointCloudMesh &copy)
    : points_(copy.points_) {}
  virtual PointCloudMesh *clone() { return new PointCloudMesh(*this); }
  virtual ~PointCloudMesh() {}

  bool get_dim(vector<unsigned int>&) const { return false;  }

  void begin(Node::iterator &) const;
  void begin(Edge::iterator &) const;
  void begin(Face::iterator &) const;
  void begin(Cell::iterator &) const;

  void end(Node::iterator &) const;
  void end(Edge::iterator &) const;
  void end(Face::iterator &) const;
  void end(Cell::iterator &) const;

  void size(Node::size_type &) const;
  void size(Edge::size_type &) const;
  void size(Face::size_type &) const;
  void size(Cell::size_type &) const;

  void to_index(Node::index_type &index, unsigned int i) const { index = i; }
  void to_index(Edge::index_type &index, unsigned int i) const { index = i; }
  void to_index(Face::index_type &index, unsigned int i) const { index = i; }
  void to_index(Cell::index_type &index, unsigned int i) const { index = i; }

  virtual BBox get_bounding_box() const;
  virtual void transform(const Transform &t);

  //! set the mesh statistics
  void resize_nodes(Node::size_type n) { points_.resize(n); }

  // This is actually get_nodes(Node::array_type &, Elem::index_type)
  // for compilation purposes.  IE It is redundant unless we are
  // templated by Elem type and we don't know that Elem is Node.
  // This is needed in ClipField, for example.
  void get_nodes(Node::array_type &a, Node::index_type i) const
    { a.resize(1); a[0] = i; }

  //! Get the children elemsnts of the given index
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

  //! get the center point (in object space) of an element
  void get_center(Point &p, Node::index_type i) const { p = points_[i]; }
  void get_center(Point &, Edge::index_type) const {}
  void get_center(Point &, Face::index_type) const {}
  void get_center(Point &, Cell::index_type) const {}

  //! Get the size of an elemnt (length, area, volume)
  double get_size(Node::index_type /*idx*/) const { return 0.0; }
  double get_size(Edge::index_type /*idx*/) const { return 0.0; }
  double get_size(Face::index_type /*idx*/) const { return 0.0; }
  double get_size(Cell::index_type /*idx*/) const { return 0.0; }
  double get_length(Edge::index_type idx) const { return get_size(idx); }
  double get_area(Face::index_type idx) const { return get_size(idx); }
  double get_volume(Cell::index_type idx) const { return get_size(idx); }

  int get_valence(Node::index_type /*idx*/) const { return 0; }
  int get_valence(Edge::index_type /*idx*/) const { return 0; }
  int get_valence(Face::index_type /*idx*/) const { return 0; }
  int get_valence(Cell::index_type /*idx*/) const { return 0; }

  bool locate(Node::index_type &, const Point &) const;
  bool locate(Edge::index_type &, const Point &) const { return false; }
  bool locate(Face::index_type &, const Point &) const { return false; }
  bool locate(Cell::index_type &, const Point &) const { return false; }

  int get_weights(const Point &p, Node::array_type &l, double *w);
  int get_weights(const Point & , Edge::array_type & , double * )
  { ASSERTFAIL("PointCloudField::get_weights for edges isn't supported"); }
  int get_weights(const Point & , Face::array_type & , double * )
  { ASSERTFAIL("PointCloudField::get_weights for faces isn't supported"); }
  int get_weights(const Point & , Cell::array_type & , double * )
  { ASSERTFAIL("PointCloudField::get_weights for cells isn't supported"); }

  void get_point(Point &p, Node::index_type i) const { get_center(p,i); }
  void set_point(const Point &p, Node::index_type i) { points_[i] = p; }
  void get_normal(Vector &/*normal*/, Node::index_type /*index*/) const
  { ASSERTFAIL("not implemented") }

  //! use these to build up a new PointCloudField mesh
  Node::index_type add_node(const Point &p) { return add_point(p); }
  Node::index_type add_point(const Point &p);
  Elem::index_type add_elem(Node::array_type a) { return a[0]; }
  void node_reserve(size_t s) { points_.reserve(s); }
  void elem_reserve(size_t s) { points_.reserve(s); }

  virtual bool is_editable() const { return true; }
  virtual int dimensionality() const { return 0; }

  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  static  const string type_name(int n = -1);
  virtual const TypeDescription *get_type_description() const;

private:
  //! the nodes
  vector<Point> points_;

  // returns a PointCloudMesh
  static Persistent *maker() { return new PointCloudMesh(); }
};  // end class PointCloudMesh

typedef LockingHandle<PointCloudMesh> PointCloudMeshHandle;

const TypeDescription* get_type_description(PointCloudMesh *);
const TypeDescription* get_type_description(PointCloudMesh::Node *);
const TypeDescription* get_type_description(PointCloudMesh::Edge *);
const TypeDescription* get_type_description(PointCloudMesh::Face *);
const TypeDescription* get_type_description(PointCloudMesh::Cell *);

} // namespace SCIRun

#endif // SCI_project_PointCloudMesh_h





