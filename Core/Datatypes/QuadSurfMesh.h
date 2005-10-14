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
 *  QuadSurfMesh.h: Templated Mesh defined on an Irregular Grid
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
#include <Core/Geometry/Transform.h>
#include <Core/Geometry/CompGeom.h>
#include <Core/Math/MusilRNG.h>
#include <Core/Containers/StackVector.h>
#include <Core/Geometry/BBox.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {

using std::vector;

template <class Basis>
class QuadSurfMesh : public Mesh
{
public:
  typedef LockingHandle<QuadSurfMesh<Basis> > handle_type;
  typedef Basis                         basis_type;
  typedef unsigned int                  under_type;

  //! Index and Iterator types required for Mesh Concept.
  struct Node {
    typedef NodeIndex<under_type>       index_type;
    typedef NodeIterator<under_type>    iterator;
    typedef NodeIndex<under_type>       size_type;
    typedef StackVector<index_type, 4>  array_type;
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

  typedef Face Elem;

 friend class ElemData;
  
  class ElemData 
  {
  public:
    ElemData(const QuadSurfMesh<Basis>& msh, 
	     const typename Cell::index_type ind) :
      mesh_(msh),
      index_(ind)
    {}
    
    // the following designed to coordinate with ::get_nodes
    inline 
    unsigned node0_index() const {
      return mesh_.cells_[index_ * 4];
    }
    inline 
    unsigned node1_index() const {
      return mesh_.cells_[index_ * 4 + 1];
    }
    inline 
    unsigned node2_index() const {
      return mesh_.cells_[index_ * 4 + 2];
    }
    inline 
    unsigned node3_index() const {
      return mesh_.cells_[index_ * 4 + 3];
    }

    // the following designed to coordinate with ::get_edges
    inline 
    unsigned edge0_index() const {
      return index_ * 6;
    }
    inline 
    unsigned edge1_index() const {
      return index_ * 6 + 1;
    }
    inline 
    unsigned edge2_index() const {
      return index_ * 6 + 2;
    }
    inline 
    unsigned edge3_index() const {
      return index_ * 6 + 3;
    }

    inline 
    const Point node0() const {
      return mesh_.points_[node0_index()];
    }
    inline 
    const Point node1() const {
      return mesh_.points_[node1_index()];
    }
    inline 
    const Point node2() const {
      return mesh_.points_[node2_index()];
    }
    inline 
    const Point node3() const {
      return mesh_.points_[node3_index()];
    }

  private:
    const QuadSurfMesh<Basis>          &mesh_;
    const typename Elem::index_type  index_;
   };

  QuadSurfMesh();
  QuadSurfMesh(const QuadSurfMesh &copy);
  virtual QuadSurfMesh *clone() { return new QuadSurfMesh(*this); }
  virtual ~QuadSurfMesh();

  virtual BBox get_bounding_box() const;
  virtual void transform(const Transform &t);

  bool get_dim(vector<unsigned int>&) const { return false;  }

  void begin(typename Node::iterator &) const;
  void begin(typename Edge::iterator &) const;
  void begin(typename Face::iterator &) const;
  void begin(typename Cell::iterator &) const;

  void end(typename Node::iterator &) const;
  void end(typename Edge::iterator &) const;
  void end(typename Face::iterator &) const;
  void end(typename Cell::iterator &) const;

  void size(typename Node::size_type &) const;
  void size(typename Edge::size_type &) const;
  void size(typename Face::size_type &) const;
  void size(typename Cell::size_type &) const;

  void to_index(typename Node::index_type &index, unsigned int i) const { index = i; }
  void to_index(typename Edge::index_type &index, unsigned int i) const { index = i; }
  void to_index(typename Face::index_type &index, unsigned int i) const { index = i; }
  void to_index(typename Cell::index_type &index, unsigned int i) const { index = i; }

  void get_nodes(typename Node::array_type &array, typename Edge::index_type idx) const;
  void get_nodes(typename Node::array_type &array, typename Face::index_type idx) const;
  void get_nodes(typename Node::array_type &array, typename Cell::index_type idx) const;
  void get_edges(typename Edge::array_type &array, typename Face::index_type idx) const;
  void get_edges(typename Edge::array_type &array, typename Cell::index_type idx) const;
  void get_faces(typename Face::array_type &array, typename Cell::index_type idx) const;

  //! get the parent element(s) of the given index
  bool get_edges(typename Edge::array_type &, typename Node::index_type) const { return 0; }
  bool get_faces(typename Face::array_type &, typename Node::index_type) const { return 0; }
  bool get_faces(typename Face::array_type &, typename Edge::index_type) const { return 0; }
  bool get_cells(typename Cell::array_type &, typename Node::index_type) const { return 0; }
  bool get_cells(typename Cell::array_type &, typename Edge::index_type) const { return 0; }
  bool get_cells(typename Cell::array_type &, typename Face::index_type) const { return 0; }

  bool get_neighbor(typename Face::index_type &neighbor,
		    typename Face::index_type from,
		    typename Edge::index_type idx) const;

  void get_neighbors(typename Face::array_type &array, typename Face::index_type idx) const;

  //! Get the size of an elemnt (length, area, volume)
  double get_size(typename Node::index_type /*idx*/) const { return 0; }
  double get_size(typename Edge::index_type idx) const
  {
    typename Node::array_type arr;
    get_nodes(arr, idx);
    return (point(arr[0]).asVector() - point(arr[1]).asVector()).length();
  }
  double get_size(typename Face::index_type idx) const
  {
    typename Node::array_type ra;
    get_nodes(ra,idx);
    const Point &p0 = point(ra[0]);
    const Point &p1 = point(ra[1]);
    const Point &p2 = point(ra[2]);
    const Point &p3 = point(ra[3]);
    return ((Cross(p0-p1,p2-p1)).length()+(Cross(p2-p3,p0-p3)).length()+
	    (Cross(p3-p0,p1-p0)).length()+(Cross(p1-p2,p3-p2)).length())*0.25;
  }
  double get_size(typename Cell::index_type /*idx*/) const { return 0; };
  double get_length(typename Edge::index_type idx) const { return get_size(idx); };
  double get_area(typename Face::index_type idx) const { return get_size(idx); };
  double get_volume(typename Cell::index_type idx) const { return get_size(idx); };

  void get_center(Point &p, typename Node::index_type i) const { get_point(p, i); }
  void get_center(Point &p, typename Edge::index_type i) const;
  void get_center(Point &p, typename Face::index_type i) const;
  void get_center(Point &, typename Cell::index_type) const {}

  bool locate(typename Node::index_type &loc, const Point &p) const;
  bool locate(typename Edge::index_type &loc, const Point &p) const;
  bool locate(typename Face::index_type &loc, const Point &p) const;
  bool locate(typename Cell::index_type &loc, const Point &p) const;

  int get_weights(const Point &p, typename Node::array_type &l, double *w);
  int get_weights(const Point & , typename Edge::array_type & , double * )
  {ASSERTFAIL("QuadSurfMesh::get_weights(Edges) not supported."); }
  int get_weights(const Point &p, typename Face::array_type &l, double *w);
  int get_weights(const Point & , typename Cell::array_type & , double * )
  {ASSERTFAIL("QuadSurfMesh::get_weights(Cells) not supported."); }
  
  bool inside3_p(typename Face::index_type i, const Point &p) const;

  void get_point(Point &p, typename Node::index_type i) const { p = points_[i]; }
  void get_normal(Vector &n, typename Node::index_type i) const { n = normals_[i]; }
  void set_point(const Point &p, typename Node::index_type i) { points_[i] = p; }

  int get_valence(typename Node::index_type /*idx*/) const { return 0; }
  int get_valence(typename Edge::index_type /*idx*/) const { return 0; }
  int get_valence(typename Face::index_type /*idx*/) const { return 0; }
  int get_valence(typename Cell::index_type /*idx*/) const { return 0; }

  virtual bool has_normals() const { return true; }

  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  static  const string type_name(int n = -1);
  virtual const TypeDescription *get_type_description() const;

  // Extra functionality needed by this specific geometry.
  typename Node::index_type add_find_point(const Point &p, 
					   double err = 1.0e-3);
  typename Elem::index_type add_quad(typename Node::index_type a, 
			    typename Node::index_type b,
			    typename Node::index_type c, 
			    typename Node::index_type d);
  typename Elem::index_type add_quad(const Point &p0, const Point &p1,
		const Point &p2, const Point &p3);
  typename Elem::index_type add_elem(typename Node::array_type a);
  void node_reserve(size_t s) { points_.reserve(s); }
  void elem_reserve(size_t s) { faces_.reserve(s*4); }
  virtual bool is_editable() const { return true; }
  virtual int dimensionality() const { return 2; }
  typename Node::index_type add_point(const Point &p);

  virtual bool		synchronize(unsigned int);

  Basis& get_basis() { return basis_; }

  //! Generate the list of points that make up a sufficiently accurate
  //! piecewise linear approximation of an edge.
  void pwl_approx_edge(vector<vector<double> > &coords, 
		       typename Elem::index_type ci, 
		       typename Edge::index_type ei, 
		       unsigned div_per_unit) const
  {    
    // Needs to match unit_edges in Basis/QuadBilinearLgn.cc 
    // compare get_nodes order to the basis order

    //FIX_ME MC delete this comment when this is verified.

    typename Edge::array_type edges;
    get_edges(edges, ci);
    unsigned count = 0;
    typename Edge::array_type::iterator iter = edges.begin();
    while (iter != edges.end()) {
      if (ei == *iter++) break;
      ++count;
    }
    basis_.approx_edge(count, div_per_unit, coords); 
  }

  //! Generate the list of points that make up a sufficiently accurate
  //! piecewise linear approximation of an face.
  void pwl_approx_face(vector<vector<vector<double> > > &coords, 
		       typename Elem::index_type ci, 
		       typename Face::index_type fi, 
		       unsigned div_per_unit) const
  {
    // Needs to match unit_faces in Basis/QuadBilinearLgn.cc 
    // compare get_nodes order to the basis order

    //FIX_ME MC delete this comment when this is verified.

    typename Face::array_type faces;
    get_faces(faces, ci);
    unsigned count = 0;
    typename Face::array_type::iterator iter = faces.begin();
    while (iter != faces.end()) {
      if (fi == *iter++) break;
      ++count;
    }
    basis_.approx_face(count, div_per_unit, coords);
  }
  
  bool get_coords(vector<double> &coords, 
		  const Point &p,
		  typename Elem::index_type idx) const
  {
    ElemData ed(*this, idx);
    return basis_.get_coords(coords, p, ed); 
  }
  
  void interpolate(Point &pt, const vector<double> &coords, 
		   typename Elem::index_type idx) const
  {
    ElemData ed(*this, idx);
    pt = basis_.interpolate(coords, ed);
  }

  static const TypeDescription* node_type_description();
  static const TypeDescription* edge_type_description();
  static const TypeDescription* face_type_description();
  static const TypeDescription* cell_type_description();

  // returns a QuadSurfMesh
  static Persistent *maker() { return new QuadSurfMesh<Basis>(); }

private:

  static double
  distance2(const Point &p0, const Point &p1)
  {
    const double dx = p0.x() - p1.x();
    const double dy = p0.y() - p1.y();
    const double dz = p0.z() - p1.z();
    return dx * dx + dy * dy + dz * dz;
  }

  const Point &point(typename Node::index_type i) const { return points_[i]; }

  void                  compute_edges();
  void			compute_normals();
  void			compute_edge_neighbors();

  int next(int i) { return ((i%4)==3) ? (i-3) : (i+1); }
  int prev(int i) { return ((i%4)==0) ? (i+3) : (i-1); }

  vector<Point>			points_;
  vector<int>                   edges_;
  vector<typename Node::index_type>	faces_;
  vector<int>			edge_neighbors_;
  vector<Vector>		normals_; //! normalized per node
  Mutex				point_lock_;
  Mutex				edge_lock_;
  Mutex				face_lock_;
  Mutex				edge_neighbor_lock_;
  Mutex				normal_lock_;
  unsigned int			synchronized_;
  Basis                         basis_;


#ifdef HAVE_HASH_MAP

struct edgehash
{
  size_t operator()(const pair<int, int> &a) const
  {
    hash<int> hasher;
    return hasher(hasher(a.first) + a.second);
  }
#ifdef __ECC

  static const size_t bucket_size = 4;
  static const size_t min_buckets = 8;

  bool operator()(const pair<int, int> &a, const pair<int, int> &b) const
  {
    return a.first < b.first || a.first == b.first && a.second < b.second;
  }
#endif
};

struct edgecompare
{
  bool operator()(const pair<int, int> &a, const pair<int, int> &b) const
  {
    return a.first == b.first && a.second == b.second;
  }
};

#else

struct edgecompare
{
  bool operator()(const pair<int, int> &a, const pair<int, int> &b) const
  {
    return a.first < b.first || a.first == b.first && a.second < b.second;
  }
};

#endif

};

template <class Basis>
PersistentTypeID 
QuadSurfMesh<Basis>::type_id(type_name(-1), "Mesh", 
			     QuadSurfMesh<Basis>::maker);



template <class Basis>
const string
QuadSurfMesh<Basis>::type_name(int n)
{
  ASSERT(n >= -1 && n <= 0);
  static const string name = "QuadSurfMesh";
  return name;
}


template <class Basis>
QuadSurfMesh<Basis>::QuadSurfMesh()
  : points_(0),
    faces_(0),
    edge_neighbors_(0),
    point_lock_("QuadSurfMesh point_lock_"),
    edge_lock_("QuadSurfMesh edge_lock_"),
    face_lock_("QuadSurfMesh face_lock_"),
    edge_neighbor_lock_("QuadSurfMesh edge_neighbor_lock_"),
    normal_lock_("QuadSurfMesh normal_lock_"),
    synchronized_(NODES_E | FACES_E | CELLS_E)
{
}

template <class Basis>
QuadSurfMesh<Basis>::QuadSurfMesh(const QuadSurfMesh &copy)
  : points_(copy.points_),
    edges_(copy.edges_),
    faces_(copy.faces_),
    edge_neighbors_(copy.edge_neighbors_),
    normals_( copy.normals_ ),
    point_lock_("QuadSurfMesh point_lock_"),
    edge_lock_("QuadSurfMesh edge_lock_"),
    face_lock_("QuadSurfMesh face_lock_"),
    edge_neighbor_lock_("QuadSurfMesh edge_neighbor_lock_"),
    normal_lock_("QuadSurfMesh normal_lock_"),
    synchronized_(copy.synchronized_)
{
}

template <class Basis>
QuadSurfMesh<Basis>::~QuadSurfMesh()
{
}

template <class Basis>
BBox
QuadSurfMesh<Basis>::get_bounding_box() const
{
  BBox result;

  for (vector<Point>::size_type i = 0; i < points_.size(); i++)
  {
    result.extend(points_[i]);
  }

  return result;
}

template <class Basis>
void
QuadSurfMesh<Basis>::transform(const Transform &t)
{
  point_lock_.lock();
  vector<Point>::iterator itr = points_.begin();
  vector<Point>::iterator eitr = points_.end();
  while (itr != eitr)
  {
    *itr = t.project(*itr);
    ++itr;
  }
  point_lock_.unlock();
}

template <class Basis>
void
QuadSurfMesh<Basis>::begin(typename QuadSurfMesh::Node::iterator &itr) const
{
  ASSERTMSG(synchronized_ & NODES_E,
	    "Must call synchronize NODES_E on QuadSurfMesh first");
  itr = 0;
}

template <class Basis>
void
QuadSurfMesh<Basis>::end(typename QuadSurfMesh::Node::iterator &itr) const
{
  ASSERTMSG(synchronized_ & NODES_E,
	    "Must call synchronize NODES_E on QuadSurfMesh first");
  itr = (int)points_.size();
}
template <class Basis>
void
QuadSurfMesh<Basis>::begin(typename QuadSurfMesh::Edge::iterator &itr) const
{
  ASSERTMSG(synchronized_ & EDGES_E,
	    "Must call synchronize EDGES_E on QuadSurfMesh first");
  itr = 0;
}
template <class Basis>
void
QuadSurfMesh<Basis>::end(typename QuadSurfMesh::Edge::iterator &itr) const
{
  ASSERTMSG(synchronized_ & EDGES_E,
	    "Must call synchronize EDGES_E on QuadSurfMesh first");
  itr = static_cast<typename Edge::iterator>((int)edges_.size());
}
template <class Basis>
void
QuadSurfMesh<Basis>::begin(typename QuadSurfMesh::Face::iterator &itr) const
{
  ASSERTMSG(synchronized_ & FACES_E,
	    "Must call synchronize FACES_E on QuadSurfMesh first");
  itr = 0;
}
template <class Basis>
void
QuadSurfMesh<Basis>::end(typename QuadSurfMesh::Face::iterator &itr) const
{
  ASSERTMSG(synchronized_ & FACES_E,
	    "Must call synchronize FACES_E on QuadSurfMesh first");
  itr = static_cast<typename Face::iterator>((int)faces_.size() / 4);
}
template <class Basis>
void
QuadSurfMesh<Basis>::begin(typename QuadSurfMesh::Cell::iterator &itr) const
{
  ASSERTMSG(synchronized_ & CELLS_E,
	    "Must call synchronize CELLS_E on QuadSurfMesh first");
  itr = 0;
}
template <class Basis>
void
QuadSurfMesh<Basis>::end(typename QuadSurfMesh::Cell::iterator &itr) const
{
  ASSERTMSG(synchronized_ & CELLS_E,
	    "Must call synchronize CELLS_E on QuadSurfMesh first");
  itr = 0;
}

template <class Basis>
void
QuadSurfMesh<Basis>::get_nodes(typename Node::array_type &array, 
			       typename Edge::index_type eidx) const
{
  ASSERTMSG(synchronized_ & EDGES_E,
	    "Must call synchronize EDGES_E on QuadSurfMesh first");
  static int table[8][2] =
  {
    {0, 1},
    {1, 2},
    {2, 3},
    {3, 0},
  };

  const int idx = edges_[eidx];
  const int off = idx % 4;
  const int node = idx - off;
  array.clear();
  array.push_back(faces_[node + table[off][0]]);
  array.push_back(faces_[node + table[off][1]]);
}

template <class Basis>
void
QuadSurfMesh<Basis>::get_nodes(typename Node::array_type &array, 
			       typename Face::index_type idx) const
{
  ASSERTMSG(synchronized_ & FACES_E,
	    "Must call synchronize FACES_E on QuadSurfMesh first");
  array.clear();
  array.push_back(faces_[idx * 4 + 0]);
  array.push_back(faces_[idx * 4 + 1]);
  array.push_back(faces_[idx * 4 + 2]);
  array.push_back(faces_[idx * 4 + 3]);
}

template <class Basis>
void
QuadSurfMesh<Basis>::get_edges(typename Edge::array_type &array, 
			       typename Face::index_type idx) const
{
  ASSERTMSG(synchronized_ & EDGES_E,
	    "Must call synchronize EDGES_E on TriSurfMesh first");

  array.clear();

  unsigned int i;
  for (i=0; i < 4; i++)
  {
    const int a = idx * 4 + i;
    const int b = a - a % 4 + (a+1) % 4;
    int j = (int)edges_.size()-1;
    for (; j >= 0; j--)
    {
      const int c = edges_[j];
      const int d = c - c % 4 + (c+1) % 4;
      if (faces_[a] == faces_[c] && faces_[b] == faces_[d] ||
	  faces_[a] == faces_[d] && faces_[b] == faces_[c])
      {
	array.push_back(j);
	break;
      }
    }
  }
}

template <class Basis>
bool
QuadSurfMesh<Basis>::get_neighbor(typename Face::index_type &neighbor,
			   typename Face::index_type from,
			   typename Edge::index_type edge) const
{
  ASSERTMSG(synchronized_ & EDGE_NEIGHBORS_E,
	    "Must call synchronize EDGE_NEIGHBORS_E on QuadSurfMesh first");
  unsigned int n = edge_neighbors_[edges_[edge]];
  if (n != MESH_NO_NEIGHBOR && (n / 4) == from)
  {
    n = edge_neighbors_[n];
  }
  if (n != MESH_NO_NEIGHBOR)
  {
    neighbor = n / 4;
    return true;
  }
  return false;
}
template <class Basis>
void
QuadSurfMesh<Basis>::get_neighbors(typename Face::array_type &neighbor,
			    typename Face::index_type idx) const
{
  ASSERTMSG(synchronized_ & EDGE_NEIGHBORS_E,
	    "Must call synchronize EDGE_NEIGHBORS_E on QuadSurfMesh first");
  typename Edge::array_type edges;
  get_edges(edges, idx);

  neighbor.clear();
  typename Edge::array_type::iterator iter = edges.begin();
  while (iter != edges.end()) {
    typename Face::index_type f;
    if (get_neighbor(f, idx, *iter)) {
      neighbor.push_back(f);
    }
    ++iter;
  }
}


template <class Basis>
bool
QuadSurfMesh<Basis>::locate(typename Node::index_type &loc, 
			    const Point &p) const
{ 
  typename Node::iterator bi, ei;
  begin(bi);
  end(ei);
  loc = 0;
  
  bool found = false;
  double mindist = 0.0;
  while (bi != ei) 
  {
    const Point &center = point(*bi);
    const double dist = (p - center).length2();
    if (!found || dist < mindist) 
    {      
      loc = *bi;
      mindist = dist;
      found = true;
    }
    ++bi;
  }
  return found;
}


template <class Basis>
bool
QuadSurfMesh<Basis>::locate(typename Edge::index_type &loc, 
			    const Point &p) const
{
  ASSERTMSG(synchronized_ & EDGES_E,
	    "Must call synchronize EDGES_E on QuadSurfMesh first");

  typename Edge::iterator bi, ei;
  typename Node::array_type nodes;
  begin(bi);
  end(ei);
  loc = 0;
  
  bool found = false;
  double mindist = 0.0;
  while (bi != ei)
  {
    get_nodes(nodes,*bi);
    const double dist = distance_to_line2(p, points_[nodes[0]], 
					  points_[nodes[1]]);
    if (!found || dist < mindist)
    {
      loc = *bi;
      mindist = dist;
      found = true;
    }
    ++bi;
  }
  return found;
}

template <class Basis>
bool 
QuadSurfMesh<Basis>::inside3_p(typename Face::index_type i, 
			       const Point &p) const
{
  typename Node::array_type nodes;
  get_nodes(nodes, i);

  unsigned int n = nodes.size();

  Point * pts = new Point[n];
  
  for (unsigned int i = 0; i < n; i++) {
    get_center(pts[i], nodes[i]);
  }

  for (unsigned int i = 0; i < n; i+=2) {
    Point p0 = pts[(i+0)%n];
    Point p1 = pts[(i+1)%n];
    Point p2 = pts[(i+2)%n];

    Vector v01(p0-p1);
    Vector v02(p0-p2);
    Vector v0(p0-p);
    Vector v1(p1-p);
    Vector v2(p2-p);
    const double a = Cross(v01, v02).length(); // area of the whole triangle (2x)
    const double a0 = Cross(v1, v2).length();  // area opposite p0
    const double a1 = Cross(v2, v0).length();  // area opposite p1
    const double a2 = Cross(v0, v1).length();  // area opposite p2
    const double s = a0+a1+a2;

    // If the area of any of the sub triangles is very small then the point
    // is on the edge of the subtriangle.
    // TODO : How small is small ???
//     if( a0 < MIN_ELEMENT_VAL ||
// 	a1 < MIN_ELEMENT_VAL ||
// 	a2 < MIN_ELEMENT_VAL )
//       return true;

    // For the point to be inside a CONVEX quad it must be inside one
    // of the four triangles that can be formed by using three of the
    // quad vertices and the point in question.
    if( fabs(s - a) < MIN_ELEMENT_VAL && a > MIN_ELEMENT_VAL ) {
      delete [] pts;
      return true;
    }
  }
  delete [] pts;
  return false;
}

template <class Basis>
bool
QuadSurfMesh<Basis>::locate(typename Face::index_type &face, 
			    const Point &p) const
{  
  typename Face::iterator bi, ei;
  begin(bi);
  end(ei);

  while (bi != ei) {
    if( inside3_p( *bi, p ) ) {
      face = *bi;
      return true;
    }

    ++bi;
  }
  return false;
}

template <class Basis>
bool
QuadSurfMesh<Basis>::locate(typename Cell::index_type &loc, 
			    const Point &) const
{
  loc = 0;
  return false;
}

template <class Basis>
int
QuadSurfMesh<Basis>::get_weights(const Point &p, typename Face::array_type &l, 
				 double *w)
{
  typename Face::index_type idx;
  if (locate(idx, p))
  {
    l.resize(1);
    l[0] = idx;
    w[0] = 1.0;
    return 1;
  }
  return 0;
}

template <class Basis>
int 
QuadSurfMesh<Basis>::get_weights(const Point &p, typename Node::array_type &l, 
				 double *w)
{
  typename Face::index_type idx;
  if (locate(idx, p))
  {
    get_nodes(l,idx);
    vector<double> coords(2);
    if (get_coords(coords, p, idx)) {
      return basis_.get_weights(coords, w);
    }
  }
  return 0;
}

template <class Basis>
void
QuadSurfMesh<Basis>::get_center(Point &result, 
				typename Edge::index_type idx) const
{
  typename Node::array_type arr;
  get_nodes(arr, idx);
  result = point(arr[0]);
  result.asVector() += point(arr[1]).asVector();

  result.asVector() *= 0.5;
}

template <class Basis>
void
QuadSurfMesh<Basis>::get_center(Point &p, typename Face::index_type idx) const
{
  typename Node::array_type nodes;
  get_nodes(nodes, idx);
  ASSERT(nodes.size() == 4);
  typename Node::array_type::iterator nai = nodes.begin();
  get_center(p, *nai);
  ++nai;
  while (nai != nodes.end())
  {
    const Point &pp = point(*nai);
    p.asVector() += pp.asVector();
    ++nai;
  }
  p.asVector() *= (1.0 / 4.0);
}

template <class Basis>
bool
QuadSurfMesh<Basis>::synchronize(unsigned int tosync)
{
  if (tosync & EDGES_E && !(synchronized_ & EDGES_E))
    compute_edges();
  if (tosync & NORMALS_E && !(synchronized_ & NORMALS_E))
    compute_normals();
  if (tosync & EDGE_NEIGHBORS_E && !(synchronized_ & EDGE_NEIGHBORS_E)) 
    compute_edge_neighbors();
  return true;
}

template <class Basis>
void
QuadSurfMesh<Basis>::compute_normals()
{
  normal_lock_.lock();
  if (synchronized_ & NORMALS_E) {
    normal_lock_.unlock();
    return;
  }
  normals_.resize(points_.size()); // 1 per node

  // build table of faces that touch each node
  vector<vector<typename Face::index_type> > node_in_faces(points_.size());
  //! face normals (not normalized) so that magnitude is also the area.
  vector<Vector> face_normals(faces_.size());
  // Computing normal per face.
  typename Node::array_type nodes(4);
  typename Face::iterator iter, iter_end;
  begin(iter);
  end(iter_end);
  while (iter != iter_end)
  {
    get_nodes(nodes, *iter);

    Point p0, p1, p2, p3;
    get_point(p0, nodes[0]);
    get_point(p1, nodes[1]);
    get_point(p2, nodes[2]);
    get_point(p3, nodes[3]);

    // build table of faces that touch each node
    node_in_faces[nodes[0]].push_back(*iter);
    node_in_faces[nodes[1]].push_back(*iter);
    node_in_faces[nodes[2]].push_back(*iter);
    node_in_faces[nodes[3]].push_back(*iter);

    Vector v0 = p1 - p0;
    Vector v1 = p2 - p1;
    Vector n = Cross(v0, v1);
    face_normals[*iter] = n;

    ++iter;
  }
  //Averaging the normals.
  typename vector<vector<typename Face::index_type> >::iterator nif_iter = 
    node_in_faces.begin();
  int i = 0;
  while (nif_iter != node_in_faces.end()) {
    const vector<typename Face::index_type> &v = *nif_iter;
    typename vector<typename Face::index_type>::const_iterator fiter = 
      v.begin();
    Vector ave(0.L,0.L,0.L);
    while(fiter != v.end()) {
      ave += face_normals[*fiter];
      ++fiter;
    }
    ave.safe_normalize();
    normals_[i] = ave; ++i;
    ++nif_iter;
  }
  synchronized_ |= NORMALS_E;
  normal_lock_.unlock();
}

template <class Basis>
typename QuadSurfMesh<Basis>::Node::index_type
QuadSurfMesh<Basis>::add_find_point(const Point &p, double err)
{
  typename Node::index_type i;
  if (locate(i, p) && distance2(points_[i], p) < err)
  {
    return i;
  }
  else
  {
    point_lock_.lock();
    normal_lock_.lock();
    points_.push_back(p);
    if (synchronized_ & NORMALS_E) normals_.push_back(Vector());
    point_lock_.unlock();
    normal_lock_.unlock();
    return static_cast<typename Node::index_type>((int)points_.size() - 1);
  }
}

template <class Basis>
typename QuadSurfMesh<Basis>::Elem::index_type
QuadSurfMesh<Basis>::add_quad(typename Node::index_type a, 
			      typename Node::index_type b,
			      typename Node::index_type c, 
			      typename Node::index_type d)
{
  face_lock_.lock();
  faces_.push_back(a);
  faces_.push_back(b);
  faces_.push_back(c);
  faces_.push_back(d);
  face_lock_.unlock();
  synchronized_ &= ~NORMALS_E;
  synchronized_ &= ~EDGE_NEIGHBORS_E;
  return static_cast<typename Elem::index_type>(((int)faces_.size() - 1) >> 2);
}


template <class Basis>
typename QuadSurfMesh<Basis>::Elem::index_type
QuadSurfMesh<Basis>::add_elem(typename Node::array_type a)
{
  face_lock_.lock();
  faces_.push_back(a[0]);
  faces_.push_back(a[1]);
  faces_.push_back(a[2]);
  faces_.push_back(a[3]);
  face_lock_.unlock();
  synchronized_ &= ~NORMALS_E;
  synchronized_ &= ~EDGE_NEIGHBORS_E;
  return static_cast<typename Elem::index_type>(((int)faces_.size() - 1) >> 2);
}
#ifdef HAVE_HASH_MAP

struct edgehash
{
  size_t operator()(const pair<int, int> &a) const
  {
    hash<int> hasher;
    return hasher((int)hasher(a.first) + a.second);
  }
#ifdef __ECC

  static const size_t bucket_size = 4;
  static const size_t min_buckets = 8;

  bool operator()(const pair<int, int> &a, const pair<int, int> &b) const
  {
    return a.first < b.first || a.first == b.first && a.second < b.second;
  }
#endif
};

struct edgecompare
{
  bool operator()(const pair<int, int> &a, const pair<int, int> &b) const
  {
    return a.first == b.first && a.second == b.second;
  }
};

#else

struct edgecompare
{
  bool operator()(const pair<int, int> &a, const pair<int, int> &b) const
  {
    return a.first < b.first || a.first == b.first && a.second < b.second;
  }
};

#endif

#ifdef HAVE_HASH_MAP

#ifdef __ECC
typedef hash_map<pair<int, int>, int, edgehash> EdgeMapType;
#else
typedef hash_map<pair<int, int>, int, edgehash, edgecompare> EdgeMapType;
#endif

#else

typedef map<pair<int, int>, int, edgecompare> EdgeMapType;

#endif

template <class Basis>
void
QuadSurfMesh<Basis>::compute_edges()
{
  edge_lock_.lock();
  if (synchronized_ & EDGES_E) {
    edge_lock_.unlock();
    return;
  }

  EdgeMapType edge_map;
  
  for( int i=(int)faces_.size()-1; i >= 0; i--)
  {
    const int a = i;
    const int b = a - a % 4 + (a+1) % 4;

    int n0 = faces_[a];
    int n1 = faces_[b];
    int tmp;
    if (n0 > n1) { tmp = n0; n0 = n1; n1 = tmp; }

    pair<int, int> nodes(n0, n1);
    edge_map[nodes] = i;
  }

  typename EdgeMapType::iterator itr;

  for (itr = edge_map.begin(); itr != edge_map.end(); ++itr)
  {
    edges_.push_back((*itr).second);
  }

  synchronized_ |= EDGES_E;
  edge_lock_.unlock();
}

template <class Basis>
void
QuadSurfMesh<Basis>::compute_edge_neighbors()
{
  // TODO: This is probably broken with the new indexed edges.
  ASSERTMSG(synchronized_ & EDGES_E,
	    "Must call synchronize EDGES_E on TriSurfMesh first");
  edge_neighbor_lock_.lock();
  if (synchronized_ & EDGE_NEIGHBORS_E) {
    edge_neighbor_lock_.unlock();
    return;
  }

  EdgeMapType edge_map;
  
  edge_neighbors_.resize(faces_.size());
  for (unsigned int j = 0; j < edge_neighbors_.size(); j++)
  {
    edge_neighbors_[j] = MESH_NO_NEIGHBOR;
  }

  for(int i = (int)faces_.size()-1; i >= 0; i--)
  {
    const int a = i;
    const int b = a - a % 4 + (a+1) % 4;

    int n0 = faces_[a];
    int n1 = faces_[b];
    int tmp;
    if (n0 > n1) { tmp = n0; n0 = n1; n1 = tmp; }

    pair<int, int> nodes(n0, n1);
   
    EdgeMapType::iterator maploc;

    maploc = edge_map.find(nodes);
    if (maploc != edge_map.end())
    {
      edge_neighbors_[(*maploc).second] = i;
      edge_neighbors_[i] = (*maploc).second;
    }
    edge_map[nodes] = i;
  }

  synchronized_ |= EDGE_NEIGHBORS_E;
  edge_neighbor_lock_.unlock();
}

template <class Basis>
typename QuadSurfMesh<Basis>::Node::index_type
QuadSurfMesh<Basis>::add_point(const Point &p)
{
  point_lock_.lock();
  normal_lock_.lock();
  points_.push_back(p);
  if (synchronized_ & NORMALS_E) normals_.push_back(Vector());
  point_lock_.unlock();
  normal_lock_.unlock();
  return static_cast<typename Node::index_type>((int)points_.size() - 1);
}


template <class Basis>
typename QuadSurfMesh<Basis>::Elem::index_type
QuadSurfMesh<Basis>::add_quad(const Point &p0, const Point &p1,
			      const Point &p2, const Point &p3)
{
  return add_quad(add_find_point(p0), add_find_point(p1),
		  add_find_point(p2), add_find_point(p3));
}


#define QUADSURFMESH_VERSION 2
template <class Basis>
void
QuadSurfMesh<Basis>::io(Piostream &stream)
{
  const int version = stream.begin_class(type_name(-1), QUADSURFMESH_VERSION);

  Mesh::io(stream);

  Pio(stream, points_);
  Pio(stream, faces_);
  if (version == 1)
  {
    Pio(stream, edge_neighbors_);
  }

  stream.end_class();

  if (stream.reading())
  {
    synchronized_ = NODES_E | FACES_E | CELLS_E;
  }
}

template <class Basis>
void
QuadSurfMesh<Basis>::size(typename QuadSurfMesh::Node::size_type &s) const
{
  typename Node::iterator itr; end(itr);
  s = *itr;
}
template <class Basis>
void
QuadSurfMesh<Basis>::size(typename QuadSurfMesh::Edge::size_type &s) const
{
  ASSERTMSG(synchronized_ & EDGES_E,
	    "Must call synchronize EDGES_E on QuadSurfMesh first");
  s = edges_.size();
}
template <class Basis>
void
QuadSurfMesh<Basis>::size(typename QuadSurfMesh::Face::size_type &s) const
{
  typename Face::iterator itr; end(itr);
  s = *itr;
}
template <class Basis>
void
QuadSurfMesh<Basis>::size(typename QuadSurfMesh::Cell::size_type &s) const
{
  typename Cell::iterator itr; end(itr);
  s = *itr;
}


template <class Basis>
const TypeDescription*
get_type_description(QuadSurfMesh<Basis> *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    const TypeDescription *sub = SCIRun::get_type_description((Basis*)0);
    TypeDescription::td_vec *subs = scinew TypeDescription::td_vec(1);
    (*subs)[0] = sub;
    td = scinew TypeDescription(QuadSurfMesh<Basis>::type_name(0), subs,
				string(__FILE__),
				"SCIRun", 
				TypeDescription::MESH_E);
  }
  return td;
}

template <class Basis>
const TypeDescription*
QuadSurfMesh<Basis>::get_type_description() const
{
  return SCIRun::get_type_description((QuadSurfMesh<Basis> *)0);
}

template <class Basis>
const TypeDescription*
QuadSurfMesh<Basis>::node_type_description()
{
  static TypeDescription *td = 0;
  if (!td)
  {
    const TypeDescription *me = 
      SCIRun::get_type_description((QuadSurfMesh<Basis> *)0);
    td = scinew TypeDescription(me->get_name() + "::Node",
				string(__FILE__),
				"SCIRun", 
				TypeDescription::MESH_E);
  }
  return td;
}

template <class Basis>
const TypeDescription*
QuadSurfMesh<Basis>::edge_type_description()
{
  static TypeDescription *td = 0;
  if (!td)
  {
    const TypeDescription *me = 
      SCIRun::get_type_description((QuadSurfMesh<Basis> *)0);
    td = scinew TypeDescription(me->get_name() + "::Edge",
				string(__FILE__),
				"SCIRun", 
				TypeDescription::MESH_E);
  }
  return td;
}

template <class Basis>
const TypeDescription*
QuadSurfMesh<Basis>::face_type_description()
{
  static TypeDescription *td = 0;
  if (!td)
  {
    const TypeDescription *me = 
      SCIRun::get_type_description((QuadSurfMesh<Basis> *)0);
    td = scinew TypeDescription(me->get_name() + "::Face",
				string(__FILE__),
				"SCIRun", 
				TypeDescription::MESH_E);
  }
  return td;
}

template <class Basis>
const TypeDescription*
QuadSurfMesh<Basis>::cell_type_description()
{
  static TypeDescription *td = 0;
  if (!td)
  {
   const TypeDescription *me = 
      SCIRun::get_type_description((QuadSurfMesh<Basis> *)0);
    td = scinew TypeDescription(me->get_name() + "::Cell",
				string(__FILE__),
				"SCIRun", 
				TypeDescription::MESH_E);
  }
  return td;
}


} // namespace SCIRun


#endif // SCI_project_QuadSurfMesh_h
