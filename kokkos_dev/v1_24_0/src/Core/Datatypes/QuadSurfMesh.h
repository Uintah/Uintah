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
 *  QuadSurfMesh.h: Templated Meshs defined on a 3D Regular Grid
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   January 2001
 *
 *  Copyright (C) 2001 SCI Group
 *
 */

#ifndef SCI_project_QuadSurfMesh_h
#define SCI_project_QuadSurfMesh_h 1

#include <Core/Containers/LockingHandle.h>
#include <Core/Datatypes/Mesh.h>
#include <Core/Datatypes/FieldIterator.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Math/MusilRNG.h>
#include <Core/Containers/StackVector.h>

#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {

using std::vector;

typedef unsigned int                  under_type;

//! Index and Iterator types required for Mesh Concept.
struct QuadSurfMeshNode {
  typedef NodeIndex<under_type>       index_type;
  typedef NodeIterator<under_type>    iterator;
  typedef NodeIndex<under_type>       size_type;
  typedef StackVector<index_type, 4>  array_type;
};

struct QuadSurfMeshEdge {
  typedef EdgeIndex<under_type>       index_type;
  typedef EdgeIterator<under_type>    iterator;
  typedef EdgeIndex<under_type>       size_type;
  typedef vector<index_type>          array_type;
};

struct QuadSurfMeshFace {
  typedef FaceIndex<under_type>       index_type;
  typedef FaceIterator<under_type>    iterator;
  typedef FaceIndex<under_type>       size_type;
  typedef vector<index_type>          array_type;
};

struct QuadSurfMeshCell {
  typedef CellIndex<under_type>       index_type;
  typedef CellIterator<under_type>    iterator;
  typedef CellIndex<under_type>       size_type;
  typedef vector<index_type>          array_type;
};


class SCICORESHARE QuadSurfMesh : public Mesh
{
public:
  
  typedef QuadSurfMeshNode Node;
  typedef QuadSurfMeshEdge Edge;
  typedef QuadSurfMeshFace Face;
  typedef QuadSurfMeshCell Cell;
  typedef Face Elem;

  QuadSurfMesh();
  QuadSurfMesh(const QuadSurfMesh &copy);
  virtual QuadSurfMesh *clone() { return new QuadSurfMesh(*this); }
  virtual ~QuadSurfMesh();

  virtual BBox get_bounding_box() const;
  virtual void transform(const Transform &t);

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

  void get_nodes(Node::array_type &array, Edge::index_type idx) const;
  void get_nodes(Node::array_type &array, Face::index_type idx) const;
  void get_nodes(Node::array_type &array, Cell::index_type idx) const;
  void get_edges(Edge::array_type &array, Face::index_type idx) const;
  void get_edges(Edge::array_type &array, Cell::index_type idx) const;
  void get_faces(Face::array_type &array, Cell::index_type idx) const;

  //! get the parent element(s) of the given index
  bool get_edges(Edge::array_type &, Node::index_type) const { return 0; }
  bool get_faces(Face::array_type &, Node::index_type) const { return 0; }
  bool get_faces(Face::array_type &, Edge::index_type) const { return 0; }
  bool get_cells(Cell::array_type &, Node::index_type) const { return 0; }
  bool get_cells(Cell::array_type &, Edge::index_type) const { return 0; }
  bool get_cells(Cell::array_type &, Face::index_type) const { return 0; }

  bool get_neighbor(Face::index_type &neighbor,
		    Face::index_type from,
		    Edge::index_type idx) const;

  void get_neighbors(Face::array_type &array, Face::index_type idx) const;

  //! Get the size of an elemnt (length, area, volume)
  double get_size(Node::index_type /*idx*/) const { return 0; }
  double get_size(Edge::index_type idx) const
  {
    Node::array_type arr;
    get_nodes(arr, idx);
    return (point(arr[0]).asVector() - point(arr[1]).asVector()).length();
  }
  double get_size(Face::index_type idx) const
  {
    Node::array_type ra;
    get_nodes(ra,idx);
    const Point &p0 = point(ra[0]);
    const Point &p1 = point(ra[1]);
    const Point &p2 = point(ra[2]);
    const Point &p3 = point(ra[3]);
    return ((Cross(p0-p1,p2-p1)).length()+(Cross(p2-p3,p0-p3)).length()+
	    (Cross(p3-p0,p1-p0)).length()+(Cross(p1-p2,p3-p2)).length())*0.25;
  }
  double get_size(Cell::index_type /*idx*/) const { return 0; };
  double get_length(Edge::index_type idx) const { return get_size(idx); };
  double get_area(Face::index_type idx) const { return get_size(idx); };
  double get_volume(Cell::index_type idx) const { return get_size(idx); };

  void get_center(Point &p, Node::index_type i) const { get_point(p, i); }
  void get_center(Point &p, Edge::index_type i) const;
  void get_center(Point &p, Face::index_type i) const;
  void get_center(Point &, Cell::index_type) const {}

  bool locate(Node::index_type &loc, const Point &p) const;
  bool locate(Edge::index_type &loc, const Point &p) const;
  bool locate(Face::index_type &loc, const Point &p) const;
  bool locate(Cell::index_type &loc, const Point &p) const;

  int get_weights(const Point &p, Node::array_type &l, double *w);
  int get_weights(const Point & , Edge::array_type & , double * )
  {ASSERTFAIL("QuadSurfMesh::get_weights(Edges) not supported."); }
  int get_weights(const Point &p, Face::array_type &l, double *w);
  int get_weights(const Point & , Cell::array_type & , double * )
  {ASSERTFAIL("QuadSurfMesh::get_weights(Cells) not supported."); }

  void get_point(Point &p, Node::index_type i) const { p = points_[i]; }
  void get_normal(Vector &n, Node::index_type i) const { n = normals_[i]; }
  void set_point(const Point &p, Node::index_type i) { points_[i] = p; }

  int get_valence(Node::index_type /*idx*/) const { return 0; }
  int get_valence(Edge::index_type /*idx*/) const { return 0; }
  int get_valence(Face::index_type /*idx*/) const { return 0; }
  int get_valence(Cell::index_type /*idx*/) const { return 0; }

  virtual bool has_normals() const { return true; }

  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  static  const string type_name(int n = -1);
  virtual const TypeDescription *get_type_description() const;

  // Extra functionality needed by this specific geometry.
  Node::index_type add_find_point(const Point &p, double err = 1.0e-3);
  Elem::index_type add_quad(Node::index_type a, Node::index_type b,
		Node::index_type c, Node::index_type d);
  Elem::index_type add_quad(const Point &p0, const Point &p1,
		const Point &p2, const Point &p3);
  Elem::index_type add_elem(Node::array_type a);
  void node_reserve(size_t s) { points_.reserve(s); }
  void elem_reserve(size_t s) { faces_.reserve(s*4); }
  virtual bool is_editable() const { return true; }
  virtual int dimensionality() const { return 2; }
  Node::index_type add_point(const Point &p);

  virtual bool		synchronize(unsigned int);

private:

  const Point &point(Node::index_type i) const { return points_[i]; }

  void                  compute_edges();
  void			compute_normals();
  void			compute_edge_neighbors();

  int next(int i) { return ((i%4)==3) ? (i-3) : (i+1); }
  int prev(int i) { return ((i%4)==0) ? (i+3) : (i-1); }

  vector<Point>			points_;
  vector<int>                   edges_;
  vector<Node::index_type>	faces_;
  vector<int>			edge_neighbors_;
  vector<Vector>		normals_; //! normalized per node
  Mutex				point_lock_;
  Mutex				edge_lock_;
  Mutex				face_lock_;
  Mutex				edge_neighbor_lock_;
  Mutex				normal_lock_;
  unsigned int			synchronized_;

  // returns a QuadSurfMesh
  static Persistent *maker() { return new QuadSurfMesh(); }
};

typedef LockingHandle<QuadSurfMesh> QuadSurfMeshHandle;

const TypeDescription* get_type_description(QuadSurfMesh *);
const TypeDescription* get_type_description(QuadSurfMesh::Node *);
const TypeDescription* get_type_description(QuadSurfMesh::Edge *);
const TypeDescription* get_type_description(QuadSurfMesh::Face *);
const TypeDescription* get_type_description(QuadSurfMesh::Cell *);

} // namespace SCIRun


#endif // SCI_project_QuadSurfMesh_h
