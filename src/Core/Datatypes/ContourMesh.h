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
 *  ContourMesh.h: countour mesh
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

#ifndef SCI_project_ContourMesh_h
#define SCI_project_ContourMesh_h 1

#include <Core/Geometry/Point.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Datatypes/Mesh.h>
#include <Core/Datatypes/FieldIterator.h>
#include <string>
#include <vector>

namespace SCIRun {

using std::string;
using std::vector;

template <class Data>
struct IndexPair {
public:
  IndexPair() {}
  IndexPair(Data first, Data second)
    : first(first), second(second) {}
  IndexPair(const IndexPair &copy)
    : first(copy.first), second(copy.second) {}
  ~IndexPair() {}

  friend void TEMPLATE_TAG Pio TEMPLATE_BOX ( Piostream &,
					      IndexPair<Data> & );

  Data first;
  Data second;
};

#define INDEXPAIR_VERSION 1

template <class Data>
void Pio(Piostream &stream, IndexPair<Data> &data)
{
  stream.begin_class("IndexPair", INDEXPAIR_VERSION);

  // IO data members, in order
  Pio(stream,data.first);
  Pio(stream,data.second);

  stream.end_class();
}

class SCICORESHARE ContourMesh : public Mesh
{
public:
  typedef unsigned int under_type;

  //! Index and Iterator types required for Mesh Concept.
  typedef IndexPair<under_type>       index_pair;

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

  typedef Edge Elem;

  typedef vector<double>      weight_array;

  ContourMesh() {}
  ContourMesh(const ContourMesh &copy)
    : nodes_(copy.nodes_), edges_(copy.edges_) {}
  virtual ContourMesh *clone() { return new ContourMesh(*this); }
  virtual ~ContourMesh() {}

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
  virtual void transform(Transform &t);

  //! set the mesh statistics
  void resize_nodes(Node::size_type n) { nodes_.resize(n); }
  void resize_edges(Edge::size_type n) { edges_.resize(n); }

  //! get the child elements of the given index
  void get_nodes(Node::array_type &a, Edge::index_type i) const
    { a.resize(2,0); a[0] = edges_[i].first; a[1] = edges_[i].second; }
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
    { result = nodes_[idx]; }
  void get_center(Point &, Edge::index_type) const {}
  void get_center(Point &, Face::index_type) const {}
  void get_center(Point &, Cell::index_type) const {}

  bool locate(Node::index_type &, const Point &) const;
  bool locate(Edge::index_type &, const Point &) const;
  bool locate(Face::index_type &, const Point &) const { return false; }
  bool locate(Cell::index_type &, const Point &) const { return false; }

  void get_weights(const Point &p, Node::array_type &l, vector<double> &w);
  void get_weights(const Point &p, Edge::array_type &l, vector<double> &w);
  void get_weights(const Point &, Face::array_type &, vector<double> &) {ASSERTFAIL("ContourMesh::get_weights for faces isn't supported");}
  void get_weights(const Point &, Cell::array_type &, vector<double> &) {ASSERTFAIL("ContourMesh::get_weights for cells isn't supported");}

  void get_point(Point &result, Node::index_type idx) const
    { get_center(result,idx); }
  void get_normal(Vector & /* result */, Node::index_type /* index */) const
  { ASSERTFAIL("not implemented") }
  void set_point(const Point &point, Node::index_type index)
    { nodes_[index] = point; }

  //! use these to build up a new contour mesh
  Node::index_type add_node(Point p)
    { nodes_.push_back(p); return nodes_.size()-1; }
  Edge::index_type add_edge(Node::index_type i1, Node::index_type i2)
    { edges_.push_back(index_pair(i1,i2)); return nodes_.size()-1; }

  virtual MeshHandle clip(ClipperHandle c);

  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  static  const string type_name(int n = -1);
  virtual const string get_type_name(int n = -1) const { return type_name(n); }

  virtual const TypeDescription *get_type_description() const;

private:

  //! the nodes
  vector<Point> nodes_;

  //! the edges
  vector<index_pair> edges_;

  // returns a ContourMesh
  static Persistent *maker() { return new ContourMesh(); }
};

typedef LockingHandle<ContourMesh> ContourMeshHandle;

const TypeDescription* get_type_description(ContourMesh *);
const TypeDescription* get_type_description(ContourMesh::Node *);
const TypeDescription* get_type_description(ContourMesh::Edge *);
const TypeDescription* get_type_description(ContourMesh::Face *);
const TypeDescription* get_type_description(ContourMesh::Cell *);


} // namespace SCIRun

#endif // SCI_project_ContourMesh_h





