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
#include <Core/Datatypes/Mesh.h>
#include <Core/Datatypes/FieldIterator.h>
#include <sgi_stl_warnings_off.h>
#include <string>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {

using std::string;
using std::vector;

class SCICORESHARE PointCloudMesh : public Mesh
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

  virtual BBox get_bounding_box() const;
  virtual void transform(const Transform &t);

  //! set the mesh statistics
  void resize_nodes(Node::size_type n) { points_.resize(n); }

  //! TODO: This doesnt make sense to me (McKay), can someone explain?
  //! get the child elements of the given index
  // this first one is so get_node(Node::array_type &, Cell::index_type)
  //   will compile.  For PointCloudMesh, Cell==Node.  This is needed
  //   in ClipField, for example.
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
  double get_size(Node::index_type idx) const { return 0.0; };
  double get_size(Edge::index_type idx) const { return 0.0; };
  double get_size(Face::index_type idx) const { return 0.0; };
  double get_size(Cell::index_type idx) const { return 0.0; };
  double get_length(Edge::index_type idx) const { return get_size(idx); };
  double get_area(Face::index_type idx) const { return get_size(idx); };
  double get_volume(Cell::index_type idx) const { return get_size(idx); };

  int get_valence(Node::index_type idx) const { return 0; };
  int get_valence(Edge::index_type idx) const { return 0; };
  int get_valence(Face::index_type idx) const { return 0; };
  int get_valence(Cell::index_type idx) const { return 0; };

  bool locate(Node::index_type &, const Point &) const;
  bool locate(Edge::index_type &, const Point &) const { return false; }
  bool locate(Face::index_type &, const Point &) const { return false; }
  bool locate(Cell::index_type &, const Point &) const { return false; }

  void get_weights(const Point &p, Node::array_type &l, vector<double> &w);
  void get_weights(const Point &, Edge::array_type &, vector<double> &)
    {ASSERTFAIL("PointCloudField::get_weights for edges isn't supported");}
  void get_weights(const Point &, Face::array_type &, vector<double> &) 
    {ASSERTFAIL("PointCloudField::get_weights for faces isn't supported");}
  void get_weights(const Point &, Cell::array_type &, vector<double> &) 
    {ASSERTFAIL("PointCloudField::get_weights for cells isn't supported");}

  void get_point(Point &p, Node::index_type i) const { get_center(p,i); }
  void set_point(const Point &p, Node::index_type i) { points_[i] = p; }
  void get_normal(Vector &/*normal*/, Node::index_type /*index*/) const
  { ASSERTFAIL("not implemented") }

  //! use these to build up a new PointCloudField mesh
  Node::index_type add_node(const Point &p) { return add_point(p); }
  Node::index_type add_point(const Point &p);
  Elem::index_type add_elem(Node::array_type a) { return a[0]; }

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
};

typedef LockingHandle<PointCloudMesh> PointCloudMeshHandle;

const TypeDescription* get_type_description(PointCloudMesh *);
const TypeDescription* get_type_description(PointCloudMesh::Node *);
const TypeDescription* get_type_description(PointCloudMesh::Edge *);
const TypeDescription* get_type_description(PointCloudMesh::Face *);
const TypeDescription* get_type_description(PointCloudMesh::Cell *);

} // namespace SCIRun

#endif // SCI_project_PointCloudMesh_h





