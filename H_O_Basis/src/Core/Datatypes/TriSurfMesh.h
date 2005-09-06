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
 *  TriSurfMesh.h: Templated Meshs defined on a 3D Regular Grid
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   January 2001
 *
 *  Copyright (C) 2001 SCI Group
 *
 *  Modified:
 *   Lorena Kreda, Northeastern University, October 2003
 *
 */

#ifndef SCI_project_TriSurfMesh_h
#define SCI_project_TriSurfMesh_h 1

#include <Core/Containers/LockingHandle.h>
#include <Core/Datatypes/Mesh.h>
#include <Core/Datatypes/FieldIterator.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/BBox.h>
#include <Core/Math/Trig.h>
#include <Core/Math/MusilRNG.h>
#include <Core/Containers/StackVector.h>

#include <sgi_stl_warnings_off.h>
#include <vector>
#include <set>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {

using std::vector;

template <class Basis>
class TriSurfMesh : public Mesh
{
public:
  typedef LockingHandle<TriSurfMesh<Basis> > handle_type;
  typedef Basis                         basis_type;
  typedef unsigned int                  under_type;

  //! Index and Iterator types required for Mesh Concept.
  struct Node {
    typedef NodeIndex<under_type>       index_type;
    typedef NodeIterator<under_type>    iterator;
    typedef NodeIndex<under_type>       size_type;
    typedef StackVector<index_type, 4>  array_type;  // Extra for IsoClip quad
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

  TriSurfMesh();
  TriSurfMesh(const TriSurfMesh &copy);
  virtual TriSurfMesh *clone() { return new TriSurfMesh(*this); }
  virtual ~TriSurfMesh();

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

  //! Get the child elements of the given index.
  void get_nodes(typename Node::array_type &, typename Edge::index_type) const;
  void get_nodes(typename Node::array_type &, typename Face::index_type) const;
  void get_nodes(typename Node::array_type &, typename Cell::index_type) const;
  void get_edges(typename Edge::array_type &, typename Face::index_type) const;
  void get_edges(typename Edge::array_type &, typename Cell::index_type) const;
  void get_faces(typename Face::array_type &, typename Cell::index_type) const;

  //! get the parent element(s) of the given index
  unsigned get_edges(typename Edge::array_type &, 
		     typename Node::index_type) const { return 0; }
  unsigned get_faces(typename Face::array_type &, 
		     typename Node::index_type) const { return 0; }
  unsigned get_faces(typename Face::array_type &, 
		     typename Edge::index_type) const { return 0; }
  unsigned get_cells(typename Cell::array_type &, 
		     typename Node::index_type) const { return 0; }
  unsigned get_cells(typename Cell::array_type &, 
		     typename Edge::index_type) const { return 0; }
  unsigned get_cells(typename Cell::array_type &, 
		     typename Face::index_type) const { return 0; }

  bool get_neighbor(typename Face::index_type &neighbor, 
		    typename Face::index_type face,
		    typename Edge::index_type edge) const;
  void get_neighbors(typename Node::array_type &array, 
		     typename Node::index_type idx) const;

  //! Get the size of an elemnt (length, area, volume)
  double get_size(typename Node::index_type /*idx*/) const { return 0.0; };
  double get_size(typename Edge::index_type idx) const 
  {
    typename Node::array_type arr;
    get_nodes(arr, idx);
    Point p0, p1;
    get_center(p0, arr[0]);
    get_center(p1, arr[1]);
    return (p1.asVector() - p0.asVector()).length();
  }
  double get_size(typename Face::index_type idx) const
  {
    typename Node::array_type ra;
    get_nodes(ra,idx);
    Point p0,p1,p2;
    get_point(p0,ra[0]);
    get_point(p1,ra[1]);
    get_point(p2,ra[2]);
    return (Cross(p0-p1,p2-p0)).length()*0.5;
  }
  double get_size(typename Cell::index_type /*idx*/) const { return 0.0; };
  double get_length(typename Edge::index_type idx) const { return get_size(idx); };
  double get_area(typename Face::index_type idx) const { return get_size(idx); };
  double get_volume(typename Cell::index_type idx) const { return get_size(idx); };

  int get_valence(typename Node::index_type idx) const
  {
    typename Node::array_type nodes;
    get_neighbors(nodes, idx);
    return (int)nodes.size();
  }

  int get_valence(typename Edge::index_type /*idx*/) const { return 0; }
  int get_valence(typename Face::index_type /*idx*/) const { return 0; }
  int get_valence(typename Cell::index_type /*idx*/) const { return 0; }


  void get_center(Point &p, typename Node::index_type i) const 
  { get_point(p, i); }
  void get_center(Point &p, typename Edge::index_type i) const;
  void get_center(Point &p, typename Face::index_type i) const;
  void get_center(Point &, typename Cell::index_type) const {}

  bool locate(typename Node::index_type &loc, const Point &p) const;
  bool locate(typename Edge::index_type &loc, const Point &p) const;
  bool locate(typename Face::index_type &loc, const Point &p) const;
  bool locate(typename Cell::index_type &loc, const Point &p) const;

  void get_point(Point &result, typename Node::index_type index) const
    { result = points_[index]; }
  void get_normal(Vector &result, typename Node::index_type index) const
    { result = normals_[index]; }
  void set_point(const Point &point, typename Node::index_type index)
    { points_[index] = point; }


  void get_random_point(Point &, typename Face::index_type, int seed=0) const;

  //! the double return val is the area of the triangle.
  double get_gradient_basis(typename Face::index_type fi, Vector& g0, Vector& g1,
			    Vector& g2);

  //! function to test if at least one of face's nodes are in supplied range
  inline bool test_nodes_range(typename Face::index_type fi,
			       unsigned int sn,
			       unsigned int en)
  {
    return (faces_[fi*3]>=sn && faces_[fi*3]<en
	    || faces_[fi*3+1]>=sn && faces_[fi*3+1]<en
	    || faces_[fi*3+2]>=sn && faces_[fi*3+2]<en);
  }

  virtual bool		synchronize(unsigned int);

  virtual bool has_normals() const { return true; }

  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  static  const string type_name(int n = -1);
  virtual const TypeDescription *get_type_description() const;

  // Extra functionality needed by this specific geometry.

  typename Node::index_type add_find_point(const Point &p, 
					   double err = 1.0e-3);
  void add_triangle(typename Node::index_type, typename Node::index_type, 
		    typename Node::index_type);

  //! swap the shared edge between 2 faces, if they share an edge.
  bool swap_shared_edge(typename Face::index_type, typename Face::index_type);
  bool remove_face(typename Face::index_type);
  //! walk all the faces, enforcing consistent face orientations.
  void orient_faces();  
  //! flip the orientaion of all the faces 
  //! orient could make all faces face inward...
  void flip_faces();
  void flip_face(typename Face::index_type face);
  void add_triangle(const Point &p0, const Point &p1, const Point &p2);
  typename Elem::index_type add_elem(typename Node::array_type a);
  void node_reserve(size_t s) { points_.reserve(s); }
  void elem_reserve(size_t s) { faces_.reserve(s*3); }
  virtual bool is_editable() const { return true; }
  virtual int dimensionality() const { return 2; }

  typename Node::index_type add_point(const Point &p);

  //! Subdivision Methods
  bool			insert_node(const Point &p);
  void			insert_node(typename Face::index_type face, const Point &p);
  void			bisect_element(const typename Face::index_type);


  const Point &point(typename Node::index_type i) { return points_[i]; }
  Basis& get_basis() { return basis_; }
  static const TypeDescription* node_type_description();
  static const TypeDescription* edge_type_description();
  static const TypeDescription* face_type_description();
  static const TypeDescription* cell_type_description();

  // returns a TriSurfMesh
  static Persistent *maker() { return new TriSurfMesh<Basis>(); }

private:
  void                  walk_face_orient(typename Face::index_type face, 
					 vector<bool> &tested);
  void			compute_normals();
  void			compute_node_neighbors();  
  void			compute_edges();
  void			compute_edge_neighbors(double err = 1.0e-8);

  bool inside3_p(int, const Point &p) const;

  int next(int i) { return ((i%3)==2) ? (i-2) : (i+1); }
  int prev(int i) { return ((i%3)==0) ? (i+2) : (i-1); }

  vector<Point>		points_;
  vector<under_type>    edges_;
  vector<under_type>	faces_;
  vector<under_type>	edge_neighbors_;
  vector<Vector>	normals_;   //! normalized per node normal.
  vector<set<under_type> > node_neighbors_;
  Mutex		        point_lock_;
  Mutex		        edge_lock_;
  Mutex		        face_lock_;
  Mutex			edge_neighbor_lock_;
  Mutex			normal_lock_;
  Mutex			node_neighbor_lock_;
  unsigned int		synchronized_;
  Basis                 basis_;

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

#ifdef HAVE_HASH_MAP

#ifdef __ECC
typedef hash_map<pair<int, int>, int, edgehash> EdgeMapType;
#else
typedef hash_map<pair<int, int>, int, edgehash, edgecompare> EdgeMapType;
#endif

#else

typedef map<pair<int, int>, int, edgecompare> EdgeMapType;

#endif

};

using std::set;

struct less_int
{
  bool operator()(const int s1, const int s2) const
  {
    return s1 < s2;
  }
};


template <class Basis>
PersistentTypeID 
TriSurfMesh<Basis>::type_id(TriSurfMesh<Basis>::type_name(-1), "Mesh", maker);


template <class Basis>
const string
TriSurfMesh<Basis>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;
  }
  else if (n == 0)
  {
    static const string nm("TriSurfMesh");
    return nm;
  }
  else 
  {
    return find_type_name((Basis *)0);
  }
}


template <class Basis>
TriSurfMesh<Basis>::TriSurfMesh()
  : points_(0),
    faces_(0),
    edge_neighbors_(0),
    node_neighbors_(0),
    point_lock_("TriSurfMesh point_lock_"),
    edge_lock_("TriSurfMesh edge_lock_"),
    face_lock_("TriSurfMesh face_lock_"),
    edge_neighbor_lock_("TriSurfMesh edge_neighbor_lock_"),
    normal_lock_("TriSurfMesh normal_lock_"),    
    node_neighbor_lock_("TriSurfMesh node_neighbor_lock_"),
    synchronized_(NODES_E | FACES_E | CELLS_E)
{
}

template <class Basis>
TriSurfMesh<Basis>::TriSurfMesh(const TriSurfMesh &copy)
  : points_(copy.points_),
    edges_(copy.edges_),
    faces_(copy.faces_),
    edge_neighbors_(copy.edge_neighbors_),
    normals_(copy.normals_),
    node_neighbors_(copy.node_neighbors_),
    point_lock_("TriSurfMesh point_lock_"),
    edge_lock_("TriSurfMesh edge_lock_"),
    face_lock_("TriSurfMesh face_lock_"),
    edge_neighbor_lock_("TriSurfMesh edge_neighbor_lock_"),
    normal_lock_("TriSurfMesh normal_lock_"),    
    node_neighbor_lock_("TriSurfMesh node_neighbor_lock_"),
    synchronized_(copy.synchronized_)
{
}

template <class Basis>
TriSurfMesh<Basis>::~TriSurfMesh()
{
}

/* To generate a random point inside of a triangle, we generate random
   barrycentric coordinates (independent random variables between 0 and
   1 that sum to 1) for the point. */
template <class Basis>
void
TriSurfMesh<Basis>::get_random_point(Point &p, 
				     typename Face::index_type ei, int seed) const
{
  static MusilRNG rng;
  
  // get the positions of the vertices
  typename Node::array_type ra;
  get_nodes(ra,ei);
  Point p0,p1,p2;
  get_point(p0,ra[0]);
  get_point(p1,ra[1]);
  get_point(p2,ra[2]);

  // generate the barrycentric coordinates
  double u,v;
  if (seed) {
    MusilRNG rng1(seed);
    u = rng1(); 
    v = rng1()*(1.-u);
  } else {
    u = rng(); 
    v = rng()*(1.-u);
  }

  // compute the position of the random point
  p = p0+((p1-p0)*u)+((p2-p0)*v);
}


template <class Basis>
BBox
TriSurfMesh<Basis>::get_bounding_box() const
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
TriSurfMesh<Basis>::transform(const Transform &t)
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
TriSurfMesh<Basis>::begin(typename TriSurfMesh::Node::iterator &itr) const
{  
  ASSERTMSG(synchronized_ & NODES_E, 
	    "Must call synchronize NODES_E on TriSurfMesh first");
  itr = 0;
}


template <class Basis>
void
TriSurfMesh<Basis>::end(typename TriSurfMesh::Node::iterator &itr) const
{
  ASSERTMSG(synchronized_ & NODES_E, 
	    "Must call synchronize NODES_E on TriSurfMesh first");
  itr = points_.size();
}

template <class Basis>
void
TriSurfMesh<Basis>::begin(typename TriSurfMesh::Edge::iterator &itr) const
{
  ASSERTMSG(synchronized_ & EDGES_E,
	    "Must call synchronize EDGES_E on TriSurfMesh first");
  itr = 0;
}

template <class Basis>
void
TriSurfMesh<Basis>::end(typename TriSurfMesh::Edge::iterator &itr) const
{
  ASSERTMSG(synchronized_ & EDGES_E,
	    "Must call synchronize EDGES_E on TriSurfMesh first");
  itr = static_cast<typename Edge::iterator>(edges_.size());
}

template <class Basis>
void
TriSurfMesh<Basis>::begin(typename TriSurfMesh::Face::iterator &itr) const
{
  ASSERTMSG(synchronized_ & FACES_E,
	    "Must call synchronize FACES_E on TriSurfMesh first");
  itr = 0;
}

template <class Basis>
void
TriSurfMesh<Basis>::end(typename TriSurfMesh::Face::iterator &itr) const
{
  ASSERTMSG(synchronized_ & FACES_E,
	    "Must call synchronize FACES_E on TriSurfMesh first");
  itr = static_cast<typename Face::iterator>(faces_.size() / 3);
}

template <class Basis>
void
TriSurfMesh<Basis>::begin(typename TriSurfMesh::Cell::iterator &itr) const
{
  ASSERTMSG(synchronized_ & CELLS_E,
	    "Must call synchronize CELLS_E on TriSurfMesh first");
  itr = 0;
}

template <class Basis>
void
TriSurfMesh<Basis>::end(typename TriSurfMesh::Cell::iterator &itr) const
{
  ASSERTMSG(synchronized_ & CELLS_E,
	    "Must call synchronize CELLS_E on TriSurfMesh first");
  itr = 0;
}


template <class Basis>
void
TriSurfMesh<Basis>::get_nodes(typename Node::array_type &array, typename Edge::index_type idx) const
{
  int a = edges_[idx];
  int b = a - a % 3 + (a+1) % 3;
  array.clear();
  array.push_back(faces_[a]);
  array.push_back(faces_[b]);
}


template <class Basis>
void
TriSurfMesh<Basis>::get_nodes(typename Node::array_type &array, typename Face::index_type idx) const
{
  array.clear();
  array.push_back(faces_[idx * 3 + 0]);
  array.push_back(faces_[idx * 3 + 1]);
  array.push_back(faces_[idx * 3 + 2]);
}

template <class Basis>
void
TriSurfMesh<Basis>::get_nodes(typename Node::array_type &array, typename Cell::index_type cidx) const
{
  array.clear();
  array.push_back(faces_[cidx * 3 + 0]);
  array.push_back(faces_[cidx * 3 + 1]);
  array.push_back(faces_[cidx * 3 + 2]);
}


template <class Basis>
void
TriSurfMesh<Basis>::get_edges(typename Edge::array_type &array, typename Face::index_type idx) const
{
  ASSERTMSG(synchronized_ & EDGES_E,
	    "Must call synchronize EDGES_E on TriSurfMesh first");
  
  array.clear();

  unsigned int i;
  for (i=0; i < 3; i++)
  {
    const int a = idx * 3 + i;
    const int b = a - a % 3 + (a+1) % 3;
    int j = edges_.size()-1;
    for (; j >= 0; j--)
    {
      const int c = edges_[j];
      const int d = c - c % 3 + (c+1) % 3;
      if (faces_[a] == faces_[c] && faces_[b] == faces_[d] ||
	  faces_[a] == faces_[d] && faces_[b] == faces_[c])
      {
	array.push_back(j);
	break;
      }
    }
  }
  ASSERT(array.size() == 3);
}



template <class Basis>
bool
TriSurfMesh<Basis>::get_neighbor(typename Face::index_type &neighbor,
				 typename Face::index_type face,
				 typename Edge::index_type edge) const
{
  ASSERTMSG(synchronized_ & EDGE_NEIGHBORS_E,
	    "Must call synchronize EDGE_NEIGHBORS_E on TriSurfMesh first");
  unsigned int n = edge_neighbors_[edges_[edge]];
  if (n != MESH_NO_NEIGHBOR && (n / 3) == face)
  {
    n = edge_neighbors_[n];
  }
  if (n != MESH_NO_NEIGHBOR)
  {
    neighbor = n / 3;
    return true;
  }
  return false;
}


template <class Basis>
void
TriSurfMesh<Basis>::compute_node_neighbors()
{
  node_neighbor_lock_.lock();
  if (synchronized_ & NODE_NEIGHBORS_E) {
    node_neighbor_lock_.unlock();
    return;
  }
  node_neighbors_.clear();
  node_neighbors_.resize(points_.size(),set<unsigned int>());
  unsigned int nfaces = faces_.size();
  for (unsigned int f = 0; f < nfaces; ++f)
  {
    node_neighbors_[faces_[f]].insert(faces_[next(f)]);
    node_neighbors_[faces_[f]].insert(faces_[prev(f)]);
  }
  synchronized_ |= NODE_NEIGHBORS_E;
  node_neighbor_lock_.unlock();
}
      

//! Returns all nodes that share an edge with this node 
template <class Basis>
void
TriSurfMesh<Basis>::get_neighbors(typename Node::array_type &array, typename Node::index_type idx) const
{
  ASSERTMSG(synchronized_ & NODE_NEIGHBORS_E, 
	    "Must call synchronize NODE_NEIGHBORS_E on TriSurfMesh first"); 
  array.resize(node_neighbors_[idx].size());
  copy(node_neighbors_[idx].begin(), node_neighbors_[idx].end(), array.begin());
}




template <class Basis>
bool
TriSurfMesh<Basis>::locate(typename Node::index_type &loc, const Point &p) const
{
  typename Node::iterator ni, nie;
  begin(ni);
  end(nie);


  if (ni == nie)
  {
    return false;
  }

  double min_dist = (p - points_[*ni]).length2();
  loc = *ni;
  ++ni;

  while (ni != nie)
  {
    const double dist = (p - points_[*ni]).length2();
    if (dist < min_dist)
    {
      min_dist = dist;
      loc = *ni;
    }
    ++ni;
  }
  return true;
}


static double
distance_to_line2(const Point &p, const Point &a, const Point &b)
{
  Vector m = b - a;
  Vector n = p - a;
  if (m.length2() < 1e-6)
  {
    return n.length2();
  }
  else
  {
    const double t0 = Dot(m, n) / Dot(m, m);
    if (t0 <= 0) return (n).length2();
    else if (t0 >= 1.0) return (p - b).length2();
    else return (n - m * t0).length2();
  }
}


template <class Basis>
bool
TriSurfMesh<Basis>::locate(typename Edge::index_type &loc, const Point &p) const
{
  ASSERTMSG(synchronized_ & EDGES_E,
	    "Must call synchronize EDGES_E on TriSurfMesh first");
  
  typename Edge::iterator bi, ei;
  begin(bi);
  end(ei);
  
  bool found = false;
  double mindist = 0.0;
  while (bi != ei)
  {
    int a = *bi;
    int b = a - a % 3 + (a+1) % 3;

    const Point &p0 = points_[faces_[a]];
    const Point &p1 = points_[faces_[b]];

    const double dist = distance_to_line2(p, p0, p1);
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
TriSurfMesh<Basis>::locate(typename Face::index_type &loc, const Point &p) const
{
  typename Face::iterator fi, fie;
  begin(fi);
  end(fie);

  loc = *fi;
  if (fi == fie)
  {
    return false;
  }

  while (fi != fie) {
    if (inside3_p((*fi)*3, p)) {
      loc = *fi;
      return true;
    }
    ++fi;
  }
  return false;
}


template <class Basis>
bool
TriSurfMesh<Basis>::locate(typename Cell::index_type &loc, const Point &) const
{
  loc = 0;
  return false;
}

//! return the area of the triangle.
template <class Basis>
double
TriSurfMesh<Basis>::get_gradient_basis(typename Face::index_type fi, Vector& g0, Vector& g1,
				       Vector& g2)
{
  Point& p1 = points_[faces_[fi * 3]];
  Point& p2 = points_[faces_[fi * 3+1]];
  Point& p3 = points_[faces_[fi * 3+2]];

  double x1=p1.x();
  double y1=p1.y();
  double z1=p1.z();

  // rotate these points into the xy (z=0) plane
  Transform trans1;
  Transform rot1;
  Transform rot2;
  Transform rot3;

  Point newp1;
  Point newp2;
  Point newp3;
  newp1 = p1;
  newp2 = p2;
  newp3 = p3;

  // 1. Translate so that p1 is at the origin
  trans1.post_translate(Vector(-x1,-y1,-z1));
  newp1 = trans1.project(newp1);
  newp2 = trans1.project(newp2);
  newp3 = trans1.project(newp3);

  // 2. Rotate about z axis - this will bring the p1-p2 edge into the xz 
  //    plane and make p2y=0;
  double phi = Atan(newp2.y()/newp2.x());
  rot1.post_rotate(-phi, Vector(0,0,1));
  newp2 = rot1.project(newp2);
  newp3 = rot1.project(newp3);

  // 3. Rotate about y-axis so that newp2.z() = 0 and p1-p2 edge is 
  //    coincident with the x-axis
  double theta = Atan(newp2.z()/newp2.x());
  rot2.post_rotate(theta, Vector(0,1,0));
  newp2 = rot2.project(newp2);
  newp3 = rot2.project(newp3);

  // 4. Rotate p3 about x axis so that pnew3.z() = 0.
  theta = Atan(newp3.z()/newp3.y());
  rot3.post_rotate(theta, Vector(1,0,0));
  newp3 = rot3.project(newp3);

  // All nodes are now in the x-y plane. Compute basis vectors.

  // first compute iA2, 1/(2xArea) (with z=0)
  double x1prime=newp1.x();
  double y1prime=newp1.y();
  double x2prime=newp2.x();
  double y2prime=newp2.y();
  double x3prime=newp3.x();
  double y3prime=newp3.y();
  
  double a1=+(x2prime*y3prime-x3prime*y2prime);
  double a2=-(x1prime*y3prime-x3prime*y1prime);
  double a3=+(x1prime*y2prime-x2prime*y1prime);

  double iA2=1./(a1+a2+a3);

  // compute the gradient basis vectors (with z=0)
  double b0=y2prime-y3prime;
  double c0=x3prime-x2prime;

  Vector g0prime=Vector(b0*iA2, c0*iA2, 0);
  
  double b1=y3prime-y1prime;
  double c1=x1prime-x3prime;

  Vector g1prime=Vector(b1*iA2, c1*iA2, 0);

  double b2=y1prime-y2prime;
  double c2=x2prime-x1prime;

  Vector g2prime=Vector(b2*iA2, c2*iA2, 0);

  // apply the inverse rotations to the basis vectors
  g0 = rot3.unproject(g0prime);
  g0 = rot2.unproject(g0);
  g0 = rot1.unproject(g0);

  g1 = rot3.unproject(g1prime);
  g1 = rot2.unproject(g1);
  g1 = rot1.unproject(g1);
  
  g2 = rot3.unproject(g2prime);
  g2 = rot2.unproject(g2);
  g2 = rot1.unproject(g2);

  // apply the inverse translation
  g0 = trans1.unproject(g0);
  g1 = trans1.unproject(g1);
  g2 = trans1.unproject(g2);
    
  // return the area of the element
  double area=(1./iA2)/2.0;
  return(area);
}


template <class Basis>
bool 
TriSurfMesh<Basis>::inside3_p(int i, const Point &p) const
{
  const Point &p0 = points_[faces_[i+0]];
  const Point &p1 = points_[faces_[i+1]];
  const Point &p2 = points_[faces_[i+2]];
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
  return fabs(s - a) < 1.0e-12 && a > 1.0e-12;
}


template <class Basis>
void
TriSurfMesh<Basis>::get_center(Point &result, typename Edge::index_type idx) const
{
  ASSERTMSG(synchronized_ & EDGES_E,
	    "Must call synchronize EDGES_E on TriSurfMesh first");
  
  typename Node::array_type arr;
  get_nodes(arr, idx);
  Point p1;
  get_center(result, arr[0]);
  get_center(p1, arr[1]);

  result.asVector() += p1.asVector();
  result.asVector() *= 0.5;
}


template <class Basis>
void
TriSurfMesh<Basis>::get_center(Point &p, typename Face::index_type i) const
{
  typename Node::array_type nodes;
  get_nodes(nodes, i);
  typename Node::array_type::iterator nai = nodes.begin();
  ASSERT(nodes.size() == 3);
  get_point(p, *nai);
  ++nai;
  Point pp;
  while (nai != nodes.end())
  {
    get_point(pp, *nai);
    p.asVector() += pp.asVector();
    ++nai;
  }
  p.asVector() *= (1.0 / 3.0);
}


template <class Basis>
bool
TriSurfMesh<Basis>::synchronize(unsigned int tosync)
{
  if (tosync & EDGES_E && !(synchronized_ & EDGES_E))
    compute_edges();
  if (tosync & NORMALS_E && !(synchronized_ & NORMALS_E)) 
    compute_normals();
  if (tosync & NODE_NEIGHBORS_E && !(synchronized_ & NODE_NEIGHBORS_E))
    compute_node_neighbors();
  if (tosync & EDGE_NEIGHBORS_E && !(synchronized_ & EDGE_NEIGHBORS_E))
    compute_edge_neighbors();
  return true;
}


template <class Basis>
void
TriSurfMesh<Basis>::compute_normals()
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
  typename Node::array_type nodes(3);
  typename Face::iterator iter, iter_end;
  begin(iter);
  end(iter_end);
  while (iter != iter_end)
  {
    get_nodes(nodes, *iter);

    Point p1, p2, p3;
    get_point(p1, nodes[0]);
    get_point(p2, nodes[1]);
    get_point(p3, nodes[2]);
    // build table of faces that touch each node
    node_in_faces[nodes[0]].push_back(*iter);
    node_in_faces[nodes[1]].push_back(*iter);
    node_in_faces[nodes[2]].push_back(*iter);

    Vector v0 = p2 - p1;
    Vector v1 = p3 - p2;
    Vector n = Cross(v0, v1);
    face_normals[*iter] = n;
//    cerr << "normal mag: " << n.length() << endl;
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
void
TriSurfMesh<Basis>::insert_node(typename Face::index_type face, const Point &p)
{
  const bool do_neighbors = synchronized_ & EDGE_NEIGHBORS_E;
  const bool do_normals = false; // synchronized_ & NORMALS_E;
  
  typename Node::index_type pi = add_point(p);
  const unsigned f0 = face*3;
  const unsigned f1 = faces_.size();
  const unsigned f2 = f1+3;

  face_lock_.lock();
  edge_neighbor_lock_.lock();
  normal_lock_.lock();

  faces_.push_back(faces_[f0+1]);
  faces_.push_back(faces_[f0+2]);
  faces_.push_back(pi);

  faces_.push_back(faces_[f0+2]);
  faces_.push_back(faces_[f0+0]);
  faces_.push_back(pi);

  // must do last
  faces_[f0+2] = pi;

  if (do_neighbors)
  {
    edge_neighbors_.push_back(edge_neighbors_[f0+1]);
    if (edge_neighbors_.back() != MESH_NO_NEIGHBOR) 
      edge_neighbors_[edge_neighbors_.back()] = edge_neighbors_.size()-1;
    edge_neighbors_.push_back(f2+2);
    edge_neighbors_.push_back(f0+1);
    
    edge_neighbors_.push_back(edge_neighbors_[f0+2]);
    if (edge_neighbors_.back() != MESH_NO_NEIGHBOR) 
      edge_neighbors_[edge_neighbors_.back()] = edge_neighbors_.size()-1;
    edge_neighbors_.push_back(f0+2);
    edge_neighbors_.push_back(f1+1);
    
    edge_neighbors_[f0+1] = f1+2;
    edge_neighbors_[f0+2] = f2+1;
  }

  if (do_normals)
  {
    Vector normal = Vector( (p.asVector() +
			     normals_[faces_[f0]] + 
			     normals_[faces_[f1]] + 
			     normals_[faces_[f2]]).safe_normalize() );
    normals_.push_back(normals_[faces_[f1]]);
    normals_.push_back(normals_[faces_[f2]]);
    normals_.push_back(normal);

    normals_.push_back(normals_[faces_[f2]]);
    normals_.push_back(normals_[faces_[f0]]);
    normals_.push_back(normal);

    normals_[faces_[f0+2]] = normal;

  }

  if (!do_neighbors) synchronized_ &= ~NODE_NEIGHBORS_E;
  synchronized_ &= ~EDGES_E;
  if (!do_normals) synchronized_ &= ~NORMALS_E;

  face_lock_.unlock();
  edge_neighbor_lock_.unlock();
  normal_lock_.unlock();     
}

  
  


template <class Basis>
bool
TriSurfMesh<Basis>::insert_node(const Point &p)
{
  typename Face::index_type face;
  if (!locate(face,p)) return false;
  insert_node(face,p);
  return true;
}

 
/*             2
//             ^
//            / \
//           /f3 \
//        5 /-----\ 4
//         / \fac/ \
//        /f1 \ /f2 \
//       /     V     \
//      <------------->
//     0       3       1
*/

#define DEBUGINFO(f) cerr << "Face #" << f/3 << " N1: " << faces_[f+0] << " N2: " << faces_[f+1] << " N3: " << faces_[f+2] << "  B1: " << edge_neighbors_[f] << " B2: " << edge_neighbors_[f+1] << "  B3: " << edge_neighbors_[f+2] << endl;
template <class Basis>
void
TriSurfMesh<Basis>::bisect_element(const typename Face::index_type face)
{
  const bool do_neighbors = synchronized_ & EDGE_NEIGHBORS_E;
  const bool do_normals = false; //synchronized_ & NORMALS_E;
  
  const unsigned f0 = face*3;
  typename Node::array_type nodes;
  get_nodes(nodes,face);
  vector<Vector> normals(3);
  for (int edge = 0; edge < 3; ++edge)
  {
    Point p = ((points_[faces_[f0+edge]] + 
		points_[faces_[next(f0+edge)]]) / 2.0).asPoint();
    nodes.push_back(add_point(p));

    if (do_normals)
      normals[edge] = Vector((normals_[faces_[f0+edge]] + 
			     normals_[faces_[next(f0+edge)]]).safe_normalize());

  }
  face_lock_.lock();
  edge_neighbor_lock_.lock();
  normal_lock_.lock();

  const unsigned f1 = faces_.size();
  faces_.push_back(nodes[0]);
  faces_.push_back(nodes[3]);
  faces_.push_back(nodes[5]);

  const unsigned f2 = faces_.size();
  faces_.push_back(nodes[1]);
  faces_.push_back(nodes[4]);
  faces_.push_back(nodes[3]);

  const unsigned f3 = faces_.size();
  faces_.push_back(nodes[2]);
  faces_.push_back(nodes[5]);
  faces_.push_back(nodes[4]);

  faces_[f0+0] = nodes[3];
  faces_[f0+1] = nodes[4];
  faces_[f0+2] = nodes[5];


  if (do_neighbors)
  {
    edge_neighbors_.push_back(edge_neighbors_[f0+0]);
    edge_neighbors_.push_back(f0+2);
    edge_neighbors_.push_back(MESH_NO_NEIGHBOR);
    
    edge_neighbors_.push_back(edge_neighbors_[f0+1]);
    edge_neighbors_.push_back(f0+0);
    edge_neighbors_.push_back(MESH_NO_NEIGHBOR);
    
    edge_neighbors_.push_back(edge_neighbors_[f0+2]);
    edge_neighbors_.push_back(f0+1);
    edge_neighbors_.push_back(MESH_NO_NEIGHBOR);

    // must do last
    edge_neighbors_[f0+0] = f2+1;
    edge_neighbors_[f0+1] = f3+1;
    edge_neighbors_[f0+2] = f1+1;
  }

  if (do_normals)
  {
    normals_.push_back(normals_[f0+0]);
    normals_.push_back(normals[0]);
    normals_.push_back(normals[2]);
    
    normals_.push_back(normals_[f0+1]);
    normals_.push_back(normals[1]);
    normals_.push_back(normals[0]);
    
    normals_.push_back(normals_[f0+2]);
    normals_.push_back(normals[2]);
    normals_.push_back(normals[1]);    

    normals_[f0+0] = normals[0];
    normals_[f0+1] = normals[1];
    normals_[f0+2] = normals[2];
  }
  

  if (do_neighbors && edge_neighbors_[f1] != MESH_NO_NEIGHBOR)
  {
    const unsigned nbr = edge_neighbors_[f1];
    const unsigned pnbr = prev(nbr);
    const unsigned f4 = faces_.size();
    faces_.push_back(nodes[1]);
    faces_.push_back(nodes[3]);
    faces_.push_back(faces_[pnbr]);
    edge_neighbors_[f2+2] = f4;
    edge_neighbors_.push_back(f2+2);
    edge_neighbors_.push_back(pnbr);
    edge_neighbors_.push_back(edge_neighbors_[pnbr]);
    edge_neighbors_[edge_neighbors_.back()] = f4+2;
    faces_[nbr] = nodes[3];
    edge_neighbors_[pnbr] = f4+1;    
    if (do_normals)
    {
      normals_[nbr] = normals[0];
      normals_.push_back(normals_[f0+1]);
      normals_.push_back(normals[0]);
      normals_.push_back(normals_[pnbr]);
    }
    
  }

  if (do_neighbors && edge_neighbors_[f2] != MESH_NO_NEIGHBOR)
  {
    const unsigned nbr = edge_neighbors_[f2];
    const unsigned pnbr = prev(nbr);
    const unsigned f5 = faces_.size();
    faces_.push_back(nodes[2]);
    faces_.push_back(nodes[4]);
    faces_.push_back(faces_[pnbr]);
    edge_neighbors_[f3+2] = f5;
    edge_neighbors_.push_back(f3+2);
    edge_neighbors_.push_back(pnbr);
    edge_neighbors_.push_back(edge_neighbors_[pnbr]);
    edge_neighbors_[edge_neighbors_.back()] = f5+2;
    faces_[nbr] = nodes[4];
    edge_neighbors_[pnbr] = f5+1;
    if (do_normals)
    {
      normals_[nbr] = normals[1];
      normals_.push_back(normals_[f0+2]);
      normals_.push_back(normals[1]);
      normals_.push_back(normals_[pnbr]);
    }
  }

  if (do_neighbors && edge_neighbors_[f3] != MESH_NO_NEIGHBOR)
  {
    const unsigned nbr = edge_neighbors_[f3];
    const unsigned pnbr = prev(nbr);
    const unsigned f6 = faces_.size();
    faces_.push_back(nodes[0]);
    faces_.push_back(nodes[5]);
    faces_.push_back(faces_[pnbr]);
    edge_neighbors_[f1+2] = f6;
    edge_neighbors_.push_back(f1+2);
    edge_neighbors_.push_back(pnbr);
    edge_neighbors_.push_back(edge_neighbors_[pnbr]);
    edge_neighbors_[edge_neighbors_.back()] = f6+2;
    faces_[nbr] = nodes[5];
    edge_neighbors_[pnbr] = f6+1;
    if (do_normals)
    {
      normals_[nbr] = normals[2];
      normals_.push_back(normals_[f0+0]);
      normals_.push_back(normals[2]);
      normals_.push_back(normals_[pnbr]);
    }
  }   

  if (!do_neighbors) synchronized_ &= ~NODE_NEIGHBORS_E;
  synchronized_ &= ~EDGES_E;
  if (!do_normals) synchronized_ &= ~NORMALS_E;

  face_lock_.unlock();
  edge_neighbor_lock_.unlock();
  normal_lock_.unlock();
}


template <class Basis>
void
TriSurfMesh<Basis>::compute_edges()
{
  edge_lock_.lock();
  if (synchronized_ & EDGES_E) {
    edge_lock_.unlock();
    return;
  }
    
  EdgeMapType edge_map;

  int i;
  for (i=faces_.size()-1; i >= 0; i--)
  {
    const int a = i;
    const int b = a - a % 3 + (a+1) % 3;

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
typename TriSurfMesh<Basis>::Node::index_type
TriSurfMesh<Basis>::add_find_point(const Point &p, double err)
{
  typename Node::index_type i;
  if (locate(i, p) && (p - points_[i]).length2() < err)
  {
    return i;
  }
  else
  {
    node_neighbor_lock_.lock();
    point_lock_.lock();
    points_.push_back(p);
    node_neighbors_.push_back(set<unsigned int>());
    node_neighbor_lock_.unlock();
    point_lock_.unlock();
    return static_cast<typename Node::index_type>(points_.size() - 1);
  }
}

// swap the shared edge between 2 faces. If faces don't share an edge, 
// do nothing.
template <class Basis>
bool
TriSurfMesh<Basis>::swap_shared_edge(typename Face::index_type f1, 
				     typename Face::index_type f2) 
{
  const int face1 = f1 * 3;
  set<int, less_int> shared;
  shared.insert(faces_[face1]);
  shared.insert(faces_[face1 + 1]);
  shared.insert(faces_[face1 + 2]);
  
  int not_shar[2];
  int *ns = not_shar;
  const int face2 = f2 * 3;
  pair<set<int, less_int>::iterator, bool> p = shared.insert(faces_[face2]);
  if (!p.second) { *ns = faces_[face2]; ++ns;}
  p = shared.insert(faces_[face2 + 1]);
  if (!p.second) { *ns = faces_[face2 + 1]; ++ns;}
  p = shared.insert(faces_[face2 + 2]);
  if (!p.second) { *ns = faces_[face2]; }

  // no shared nodes means no shared edge.
  if (shared.size() > 4) return false;  

  set<int>::iterator iter = shared.find(not_shar[0]);
  shared.erase(iter);

  iter = shared.find(not_shar[1]);
  shared.erase(iter);

  iter = shared.begin();
  int s1 = *iter++;
  int s2 = *iter;
  face_lock_.lock();
  faces_[face1] = s1;
  faces_[face1 + 1] = not_shar[0];
  faces_[face1 + 2] = s2;

  faces_[face2] = s2;
  faces_[face2 + 1] = not_shar[1];
  faces_[face2 + 2] = s1;

  synchronized_ &= ~EDGE_NEIGHBORS_E;
  synchronized_ &= ~NODE_NEIGHBORS_E;
  synchronized_ &= ~NORMALS_E;
  face_lock_.unlock();
  return true;
}

template <class Basis>
bool
TriSurfMesh<Basis>::remove_face(typename Face::index_type f)
{
  bool rval = true;
  face_lock_.lock();
  vector<under_type>::iterator fb = faces_.begin() + f*3;
  vector<under_type>::iterator fe = fb + 3;

  if (fe <= faces_.end())
    faces_.erase(fb, fe);
  else {
    rval = false;
  }
  synchronized_ &= ~EDGE_NEIGHBORS_E;
  synchronized_ &= ~NODE_NEIGHBORS_E;
  synchronized_ &= ~NORMALS_E;
  face_lock_.unlock();
  return rval;
}

template <class Basis>
void
TriSurfMesh<Basis>::add_triangle(typename Node::index_type a, typename Node::index_type b, typename Node::index_type c)
{
  face_lock_.lock();
  faces_.push_back(a);
  faces_.push_back(b);
  faces_.push_back(c);
  synchronized_ &= ~EDGE_NEIGHBORS_E;
  synchronized_ &= ~NODE_NEIGHBORS_E;
  synchronized_ &= ~NORMALS_E;
  face_lock_.unlock();
}

template <class Basis>
typename TriSurfMesh<Basis>::Elem::index_type
TriSurfMesh<Basis>::add_elem(typename Node::array_type a)
{
  face_lock_.lock();
  faces_.push_back(a[0]);
  faces_.push_back(a[1]);
  faces_.push_back(a[2]);
  synchronized_ &= ~EDGE_NEIGHBORS_E;
  synchronized_ &= ~NODE_NEIGHBORS_E;
  synchronized_ &= ~NORMALS_E;
  face_lock_.unlock();
  return static_cast<typename Elem::index_type>((faces_.size() - 1) / 3);
}

template <class Basis>
void
TriSurfMesh<Basis>::flip_faces() 
{
  face_lock_.lock();
  typename Face::iterator fiter, fend;
  begin(fiter);
  end(fend);
  while (fiter != fend) {
    flip_face(*fiter);
    ++fiter;
  }
  synchronized_ &= ~EDGES_E;
  synchronized_ &= ~EDGE_NEIGHBORS_E;
  synchronized_ &= ~NODE_NEIGHBORS_E;
  synchronized_ &= ~NORMALS_E;
  face_lock_.unlock();
}

template <class Basis>
void 
TriSurfMesh<Basis>::flip_face(typename Face::index_type face)
{
  unsigned int base = face * 3;
  int tmp = faces_[base + 1];
  faces_[base + 1] = faces_[base + 2];  
  faces_[base + 2] = tmp;

  synchronized_ &= ~EDGES_E;
  synchronized_ &= ~EDGE_NEIGHBORS_E;
  synchronized_ &= ~NODE_NEIGHBORS_E;
  synchronized_ &= ~NORMALS_E;
}

template <class Basis>
void 
TriSurfMesh<Basis>::walk_face_orient(typename Face::index_type face, 
				     vector<bool> &tested)
{
  typename Face::index_type nbor;
  typename Edge::array_type edges(3);
  get_edges(edges, face);

  typename Node::array_type nodes;
  get_nodes(nodes, face);


  typename Edge::array_type::iterator iter = edges.begin();
  while(iter != edges.end()) {

    unsigned a = edges_[*iter];
    unsigned b = faces_[a - a % 3 + (a+1) % 3];
    a = faces_[a];
    //set order according to orientation in face
    if (((nodes[0] == b) && (nodes[1] == a)) || 
	((nodes[1] == b) && (nodes[2] == a)) ||
	((nodes[0] == a) && (nodes[2] == b)))
    {
      //swap 
      unsigned tmp = b;
      b = a;
      a = tmp;
    } 

    if (get_neighbor(nbor, face, *iter) && !tested[nbor]) {
      tested[nbor] = true;

      typename Node::array_type nbor_nodes;
      get_nodes(nbor_nodes, nbor);

      //order should be opposite of a,b
      if (((nbor_nodes[0] == a) && (nbor_nodes[1] == b)) || 
	  ((nbor_nodes[1] == a) && (nbor_nodes[2] == b)) ||
	  ((nbor_nodes[0] == b) && (nbor_nodes[2] == a)))
      {
	flip_face(nbor);
	synchronized_ &= ~EDGES_E;
	synchronized_ &= ~EDGE_NEIGHBORS_E;
	edges_.clear();
	compute_edges();
	compute_edge_neighbors(0.0);
      } 

      // recurse...
      walk_face_orient(nbor, tested);
    }
    ++iter;
  }
}

template <class Basis>
void
TriSurfMesh<Basis>::orient_faces() 
{
  face_lock_.lock();
  synchronize(EDGES_E | EDGE_NEIGHBORS_E);
  int nfaces = (int)faces_.size() / 3;
  vector<bool> tested(nfaces, false);

  typename Face::iterator fiter, fend;
  begin(fiter);
  end(fend);
  while (fiter != fend) {
    if (! tested[*fiter]) {
      tested[*fiter] = true;
      walk_face_orient(*fiter, tested);
    }
    ++fiter;
  }

  synchronized_ &= ~EDGE_NEIGHBORS_E;
  synchronized_ &= ~NODE_NEIGHBORS_E;
  synchronized_ &= ~NORMALS_E;
  face_lock_.unlock();
}

template <class Basis>
void
TriSurfMesh<Basis>::compute_edge_neighbors(double /*err*/)
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

  int i;
  for (i=faces_.size()-1; i >= 0; i--)
  {
    const int a = i;
    const int b = a - a % 3 + (a+1) % 3;

    int n0 = faces_[a];
    int n1 = faces_[b];
    int tmp;
    if (n0 > n1) { tmp = n0; n0 = n1; n1 = tmp; }

    pair<int, int> nodes(n0, n1);
    
    typename EdgeMapType::iterator maploc;

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
typename TriSurfMesh<Basis>::Node::index_type
TriSurfMesh<Basis>::add_point(const Point &p)
{
  normal_lock_.lock();
  node_neighbor_lock_.lock();
  point_lock_.lock();
  points_.push_back(p);
  if (synchronized_ & NORMALS_E) normals_.push_back(Vector());
  if (synchronized_ & NODE_NEIGHBORS_E)
  {
    node_neighbors_.push_back(set<unsigned int>());
  }
  normal_lock_.unlock();
  node_neighbor_lock_.unlock();
  point_lock_.unlock();
  return static_cast<typename Node::index_type>(points_.size() - 1);
}



template <class Basis>
void
TriSurfMesh<Basis>::add_triangle(const Point &p0, const Point &p1, const Point &p2)
{
  add_triangle(add_find_point(p0), add_find_point(p1), add_find_point(p2));
}


#define TRISURFMESH_VERSION 1

template <class Basis>
void
TriSurfMesh<Basis>::io(Piostream &stream)
{
  stream.begin_class(type_name(-1), TRISURFMESH_VERSION);
  
  Mesh::io(stream);

  Pio(stream, points_);
  Pio(stream, faces_);
  Pio(stream, edge_neighbors_);

  stream.end_class();

  if (stream.reading() && edge_neighbors_.size())
  {
    synchronized_ |= EDGE_NEIGHBORS_E;
  }
}


template <class Basis>
void
TriSurfMesh<Basis>::size(typename TriSurfMesh::Node::size_type &s) const
{
  typename Node::iterator itr; end(itr);
  s = *itr;
}

template <class Basis>
void
TriSurfMesh<Basis>::size(typename TriSurfMesh::Edge::size_type &s) const
{
  ASSERTMSG(synchronized_ & EDGES_E,
	    "Must call synchronize EDGES_E on TriSurfMesh first");
  
  typename Edge::iterator itr; end(itr);
  s = *itr;
}

template <class Basis>
void
TriSurfMesh<Basis>::size(typename TriSurfMesh::Face::size_type &s) const
{
  typename Face::iterator itr; end(itr);
  s = *itr;
}

template <class Basis>
void
TriSurfMesh<Basis>::size(typename TriSurfMesh::Cell::size_type &s) const
{
  typename Cell::iterator itr; end(itr);
  s = *itr;
}



template <class Basis>
const TypeDescription*
TriSurfMesh<Basis>::get_type_description() const
{
  return SCIRun::get_type_description((TriSurfMesh<Basis> *)0);
}

template <class Basis>
const TypeDescription*
get_type_description(TriSurfMesh<Basis> *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    const TypeDescription *sub = SCIRun::get_type_description((Basis*)0);
    TypeDescription::td_vec *subs = scinew TypeDescription::td_vec(1);
    (*subs)[0] = sub;
    td = scinew TypeDescription(TriSurfMesh<Basis>::type_name(0), subs,
				string(__FILE__),
				"SCIRun");
  }
  return td;
}

template <class Basis>
const TypeDescription*
TriSurfMesh<Basis>::node_type_description()
{
  static TypeDescription *td = 0;
  if (!td)
  {
    const TypeDescription *me = 
      SCIRun::get_type_description((TriSurfMesh<Basis> *)0);
    td = scinew TypeDescription(me->get_name() + "::Node",
				string(__FILE__),
				"SCIRun");
  }
  return td;
}

template <class Basis>
const TypeDescription*
TriSurfMesh<Basis>::edge_type_description()
{
  static TypeDescription *td = 0;
  if (!td)
  {
    const TypeDescription *me = 
      SCIRun::get_type_description((TriSurfMesh<Basis> *)0);
    td = scinew TypeDescription(me->get_name() + "::Edge",
				string(__FILE__),
				"SCIRun");
  }
  return td;
}

template <class Basis>
const TypeDescription*
TriSurfMesh<Basis>::face_type_description()
{
  static TypeDescription *td = 0;
  if (!td)
  {
    const TypeDescription *me = 
      SCIRun::get_type_description((TriSurfMesh<Basis> *)0);
    td = scinew TypeDescription(me->get_name() + "::Face",
				string(__FILE__),
				"SCIRun");
  }
  return td;
}

template <class Basis>
const TypeDescription*
TriSurfMesh<Basis>::cell_type_description()
{
  static TypeDescription *td = 0;
  if (!td)
  {
    const TypeDescription *me = 
      SCIRun::get_type_description((TriSurfMesh<Basis> *)0);
    td = scinew TypeDescription(me->get_name() + "::Cell",
				string(__FILE__),
				"SCIRun");
  }
  return td;
}



} // namespace SCIRun


#endif // SCI_project_TriSurfMesh_h
