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
 *  CurveMesh.h: countour mesh
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

#ifndef SCI_project_CurveMesh_h
#define SCI_project_CurveMesh_h 1

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
using std::pair;


class SCICORESHARE CurveMesh : public Mesh
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

  typedef Edge Elem;

  typedef pair<Node::index_type, Node::index_type> index_pair_type;

  CurveMesh() {}
  CurveMesh(const CurveMesh &copy)
    : nodes_(copy.nodes_), edges_(copy.edges_) {}
  virtual CurveMesh *clone() { return new CurveMesh(*this); }
  virtual ~CurveMesh() {}

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

  //! get the center point (in object space) of an element
  void get_center(Point &result, Node::index_type idx) const
    { result = nodes_[idx]; }
  void get_center(Point &, Edge::index_type) const;
  void get_center(Point &, Face::index_type) const {}
  void get_center(Point &, Cell::index_type) const {}

  //! Get the size of an elemnt (length, area, volume)
  double get_size(Node::index_type idx) const { return 0.0; }
  double get_size(Edge::index_type idx) const 
  {
    Node::array_type arr;
    get_nodes(arr, idx);
    Point p0, p1;
    get_center(p0, arr[0]);
    get_center(p1, arr[1]);
    return (p1.asVector() - p0.asVector()).length();
  }
  double get_size(Face::index_type idx) const { return 0.0; }
  double get_size(Cell::index_type idx) const { return 0.0; }
  double get_length(Edge::index_type idx) const { return get_size(idx); };
  double get_area(Face::index_type idx) const   { return get_size(idx); };
  double get_volume(Cell::index_type idx) const { return get_size(idx); };

  int get_valence(Node::index_type idx) const;
  int get_valence(Edge::index_type idx) const { return 0; }
  int get_valence(Face::index_type idx) const { return 0; }
  int get_valence(Cell::index_type idx) const { return 0; }

  bool locate(Node::index_type &, const Point &) const;
  bool locate(Edge::index_type &, const Point &) const;
  bool locate(Face::index_type &, const Point &) const { return false; }
  bool locate(Cell::index_type &, const Point &) const { return false; }

  void get_weights(const Point &p, Node::array_type &l, vector<double> &w);
  void get_weights(const Point &p, Edge::array_type &l, vector<double> &w);
  void get_weights(const Point &, Face::array_type &, vector<double> &) 
    {ASSERTFAIL("CurveMesh::get_weights for faces isn't supported");}
  void get_weights(const Point &, Cell::array_type &, vector<double> &) 
    {ASSERTFAIL("CurveMesh::get_weights for cells isn't supported");}

  void get_point(Point &result, Node::index_type idx) const
    { get_center(result,idx); }
  void set_point(const Point &point, Node::index_type index)
    { nodes_[index] = point; }

  void get_normal(Vector & /* result */, Node::index_type /* index */) const
    { ASSERTFAIL("not implemented") }

  //! use these to build up a new contour mesh
  Node::index_type add_node(const Point &p)
    { nodes_.push_back(p); return nodes_.size()-1; }
  Node::index_type add_point(const Point &point) 
    { return add_node(point); }
  Edge::index_type add_edge(Node::index_type i1, Node::index_type i2)
    {
      edges_.push_back(index_pair_type(i1,i2));
      return static_cast<Edge::index_type>(nodes_.size()-1);
    }
  Elem::index_type add_elem(Node::array_type a)
    {
      edges_.push_back(index_pair_type(a[0],a[1]));
      return static_cast<Elem::index_type>(nodes_.size()-1);
    }

  virtual bool is_editable() const { return true; }
  virtual int dimensionality() const { return 1; }
    
  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  static  const string type_name(int n = -1);

  virtual const TypeDescription *get_type_description() const;
private:

  //! the nodes
  vector<Point> nodes_;

  //! the edges
  vector<index_pair_type> edges_;

  // returns a CurveMesh
  static Persistent *maker() { return new CurveMesh(); }
};

typedef LockingHandle<CurveMesh> CurveMeshHandle;

const TypeDescription* get_type_description(CurveMesh *);
const TypeDescription* get_type_description(CurveMesh::Node *);
const TypeDescription* get_type_description(CurveMesh::Edge *);
const TypeDescription* get_type_description(CurveMesh::Face *);
const TypeDescription* get_type_description(CurveMesh::Cell *);


} // namespace SCIRun

#endif // SCI_project_CurveMesh_h





