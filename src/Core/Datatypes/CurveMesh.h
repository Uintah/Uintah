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
#include <Core/Containers/StackVector.h>
#include <sgi_stl_warnings_off.h>
#include <string>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {

using std::string;
using std::vector;
using std::pair;

typedef unsigned int under_type;

//! Index and Iterator types required for Mesh Concept.
struct CurveMeshNode {
  typedef NodeIndex<under_type>       index_type;
  typedef NodeIterator<under_type>    iterator;
  typedef NodeIndex<under_type>       size_type;
  typedef StackVector<index_type, 2>  array_type;
};					
  					
struct CurveMeshEdge {				
  typedef EdgeIndex<under_type>       index_type;
  typedef EdgeIterator<under_type>    iterator;
  typedef EdgeIndex<under_type>       size_type;
  typedef vector<index_type>          array_type;
};					
					
struct CurveMeshFace {				
  typedef FaceIndex<under_type>       index_type;
  typedef FaceIterator<under_type>    iterator;
  typedef FaceIndex<under_type>       size_type;
  typedef vector<index_type>          array_type;
};					
					
struct CurveMeshCell {				
  typedef CellIndex<under_type>       index_type;
  typedef CellIterator<under_type>    iterator;
  typedef CellIndex<under_type>       size_type;
  typedef vector<index_type>          array_type;
};


class CurveMesh : public Mesh
{
public:
  typedef CurveMeshNode Node;
  typedef CurveMeshEdge Edge;
  typedef CurveMeshFace Face;
  typedef CurveMeshCell Cell;
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

  void to_index(Node::index_type &index, unsigned int i) const { index = i; }
  void to_index(Edge::index_type &index, unsigned int i) const { index = i; }
  void to_index(Face::index_type &index, unsigned int i) const { index = i; }
  void to_index(Cell::index_type &index, unsigned int i) const { index = i; }

  virtual BBox get_bounding_box() const;
  virtual void transform(const Transform &t);

  //! get the child elements of the given index
  void get_nodes(Node::array_type &a, Edge::index_type i) const
    { a.resize(2); a[0] = edges_[i].first; a[1] = edges_[i].second; }
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

  bool get_neighbor(Edge::index_type &neighbor, Edge::index_type edge,
		    Node::index_type node) const
  {ASSERTFAIL("CurveMesh::get_neighbor for edges needs to be implemented"); }
  void get_neighbors(vector<Node::index_type> &array,
                     Node::index_type idx) const
  {ASSERTFAIL("CurveMesh::get_neighbor for nodes needs to be implemented"); }

  //! Get the size of an element (length, area, volume)
  double get_size(Node::index_type /*idx*/) const { return 0.0; }
  double get_size(Edge::index_type idx) const 
  {
    Node::array_type arr;
    get_nodes(arr, idx);
    Point p0, p1;
    get_center(p0, arr[0]);
    get_center(p1, arr[1]);
    return (p1.asVector() - p0.asVector()).length();
  }
  double get_size(Face::index_type /*idx*/) const { return 0.0; }
  double get_size(Cell::index_type /*idx*/) const { return 0.0; }
  double get_length(Edge::index_type idx) const { return get_size(idx); };
  double get_area(Face::index_type idx) const   { return get_size(idx); };
  double get_volume(Cell::index_type idx) const { return get_size(idx); };

  int get_valence(Node::index_type idx) const;
  int get_valence(Edge::index_type /*idx*/) const { return 0; }
  int get_valence(Face::index_type /*idx*/) const { return 0; }
  int get_valence(Cell::index_type /*idx*/) const { return 0; }

  bool locate(Node::index_type &, const Point &) const;
  bool locate(Edge::index_type &, const Point &) const;
  bool locate(Face::index_type &, const Point &) const { return false; }
  bool locate(Cell::index_type &, const Point &) const { return false; }

  int get_weights(const Point &p, Node::array_type &l, double *w);
  int get_weights(const Point &p, Edge::array_type &l, double *w);
  int get_weights(const Point & , Face::array_type & , double * )
  {ASSERTFAIL("CurveMesh::get_weights for faces isn't supported"); }
  int get_weights(const Point & , Cell::array_type & , double * )
  {ASSERTFAIL("CurveMesh::get_weights for cells isn't supported"); }

  void get_point(Point &result, Node::index_type idx) const
    { get_center(result,idx); }
  void set_point(const Point &point, Node::index_type index)
    { nodes_[index] = point; }

  void get_normal(Vector & /* result */, Node::index_type /* index */) const
    { ASSERTFAIL("not implemented") }

  //! use these to build up a new contour mesh
  Node::index_type add_node(const Point &p)
    { nodes_.push_back(p); return static_cast<under_type>(nodes_.size()-1); }
  Node::index_type add_point(const Point &point) 
    { return add_node(point); }
  Edge::index_type add_edge(Node::index_type i1, Node::index_type i2)
    {
      edges_.push_back(index_pair_type(i1,i2));
      return static_cast<under_type>(nodes_.size()-1);
    }
  Elem::index_type add_elem(Node::array_type a)
    {
      edges_.push_back(index_pair_type(a[0],a[1]));
      return static_cast<under_type>(nodes_.size()-1);
    }
  void node_reserve(size_t s) { nodes_.reserve(s); }
  void elem_reserve(size_t s) { edges_.reserve(s*2); }

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





