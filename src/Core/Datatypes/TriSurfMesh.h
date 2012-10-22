/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
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
 *
 *  Modified:
 *   Lorena Kreda, Northeastern University, October 2003
 *
 */


#ifndef SCI_project_TriSurfMesh_h
#define SCI_project_TriSurfMesh_h 1

#include <Core/Containers/LockingHandle.h>
#include <Core/Datatypes/Mesh.h>
#include <Core/Datatypes/SearchGrid.h>
#include <Core/Datatypes/FieldIterator.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/BBox.h>
#include <Core/Geometry/Transform.h>
#include <Core/Geometry/CompGeom.h>
#include <Core/Math/Trig.h>
#include <Core/Containers/StackVector.h>

#include <vector>
#include <list>
#include <set>
#include <algorithm>
#include <cfloat> // for DBL_MAX

namespace SCIRun {

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
    typedef std::vector<index_type>     array_type;
  };

  struct Face {
    typedef FaceIndex<under_type>       index_type;
    typedef FaceIterator<under_type>    iterator;
    typedef FaceIndex<under_type>       size_type;
    typedef std::vector<index_type>     array_type;
  };

  struct Cell {
    typedef CellIndex<under_type>       index_type;
    typedef CellIterator<under_type>    iterator;
    typedef CellIndex<under_type>       size_type;
    typedef std::vector<index_type>     array_type;
  };

  typedef Face Elem;
  typedef Edge DElem;

  friend class ElemData;

  class ElemData
  {
  public:
    ElemData(const TriSurfMesh<Basis>& msh,
             const typename Elem::index_type ind) :
      mesh_(msh),
      index_(ind)
    {
      if (basis_type::polynomial_order() > 1) {
	mesh_.get_edges(edges_, ind);
      }
    }

    // the following designed to coordinate with ::get_nodes
    inline
    unsigned node0_index() const {
      return mesh_.faces_[index_ * 3];
    }
    inline
    unsigned node1_index() const {
      return mesh_.faces_[index_ * 3 + 1];
    }
    inline
    unsigned node2_index() const {
      return mesh_.faces_[index_ * 3 + 2];
    }

    // the following designed to coordinate with ::get_edges
    inline
    unsigned edge0_index() const {
      return edges_[0];
    }
    inline
    unsigned edge1_index() const {
      return edges_[1];
    }
    inline
    unsigned edge2_index() const {
      return edges_[2];
    }

    inline
    unsigned elem_index() const {
      return index_;
    }

    inline
    const Point &node0() const {
      return mesh_.points_[node0_index()];
    }
    inline
    const Point &node1() const {
      return mesh_.points_[node1_index()];
    }
    inline
    const Point &node2() const {
      return mesh_.points_[node2_index()];
    }

  private:
    const TriSurfMesh<Basis>          &mesh_;
    const typename Elem::index_type    index_;
    typename Edge::array_type          edges_;
  };

  TriSurfMesh();
  TriSurfMesh(const TriSurfMesh &copy);
  virtual TriSurfMesh *clone() { return new TriSurfMesh(*this); }
  virtual ~TriSurfMesh();

  virtual BBox get_bounding_box() const;
  virtual void transform(const Transform &t);

  bool get_dim(std::vector<unsigned int>&) const { return false;  }
  virtual int topology_geometry() const { return (UNSTRUCTURED | IRREGULAR); }

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
  void get_faces(typename Face::array_type &a,
                 typename Face::index_type f) const
  { a.push_back(f); }

  //! get the parent element(s) of the given index
  void get_elems(typename Elem::array_type &result,
                 typename Node::index_type idx) const
  {
    ASSERTMSG(synchronized_ & NODE_NEIGHBORS_E,
	      "Must call synchronize NODE_NEIGHBORS_E on TriSurfMesh first");
    result.clear();
    for (unsigned int i = 0; i < node_neighbors_[idx].size(); ++i)
      result.push_back(node_neighbors_[idx][i]/3);
  }
  void get_elems(typename Elem::array_type &result,
                 typename Edge::index_type idx) const {}
  void get_elems(typename Elem::array_type &result,
                 typename Face::index_type idx) const {}


  //! Wrapper to get the derivative elements from this element.
  void get_delems(typename DElem::array_type &result,
                  typename Elem::index_type idx) const
  {
    get_edges(result, idx);
  }

  bool get_neighbor(typename Face::index_type &neighbor,
                    typename Face::index_type face,
                    typename Edge::index_type edge) const;
  bool get_neighbor(unsigned int &nbr_half_edge,
                    unsigned int half_edge) const;
  void get_neighbors(std::vector<typename Node::index_type> &array,
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
  double get_size(typename Cell::index_type /*idx*/) const { return 0.0; }
  double get_length(typename Edge::index_type idx) const
  { return get_size(idx); }
  double get_area(typename Face::index_type idx) const
  { return get_size(idx); }
  double get_volume(typename Cell::index_type idx) const
  { return get_size(idx); }

  int get_valence(typename Node::index_type idx) const
  {
    std::vector<typename Node::index_type> nodes;
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

  virtual bool get_search_grid_info(int &i, int &j, int &k, Transform &trans)
  {
    synchronize(LOCATE_E);
    i = grid_->get_ni();
    j = grid_->get_nj();
    k = grid_->get_nk();
    grid_->get_canonical_transform(trans);
    return true;
  }

  int get_weights(const Point &p, typename Node::array_type &l, double *w);
  int get_weights(const Point & , typename Edge::array_type & , double * /*w*/)
  {ASSERTFAIL("TriSurfMesh::get_weights(Edges) not supported."); }
  int get_weights(const Point &p, typename Face::array_type &l, double *w);
  int get_weights(const Point & , typename Cell::array_type & , double * /*w*/)
  {ASSERTFAIL("TriSurfMesh::get_weights(Cells) not supported."); }

  void get_point(Point &result, typename Node::index_type index) const
  { result = points_[index]; }
  void get_normal(Vector &result, typename Node::index_type index) const
  {
    ASSERTMSG(synchronized_ & NORMALS_E,
	      "Must call synchronize NORMALS_E on TriSurfMesh first"); 
    result = normals_[index]; 
  }
  void set_point(const Point &point, typename Node::index_type index)
  { points_[index] = point; }

  void get_normal(Vector &result, std::vector<double> &coords,
                  typename Elem::index_type eidx, unsigned int)
  {
    if (basis_.polynomial_order() < 2) {
      ASSERTMSG(synchronized_ & NORMALS_E,
		"Must call synchronize NORMALS_E on TriSurfMesh first");
      
      typename Node::array_type arr(3);
      get_nodes(arr, eidx);

      const double c0_0 = fabs(coords[0]);
      const double c1_0 = fabs(coords[1]);
      const double c0_1 = fabs(coords[0] - 1.0L);
      const double c1_1 = fabs(coords[1] - 1.0L);

      if (c0_0 < 1e-7 && c1_0 < 1e-7) {
        // arr[0]
        result = normals_[arr[0]];
        return;
      } else if (c0_1 < 1e-7 && c1_0 < 1e-7) {
        // arr[1]
        result = normals_[arr[1]];
        return;
      } else if (c0_0 < 1e-7 && c1_1 < 1e-7) {
        // arr[2]
        result = normals_[arr[2]];
        return;
      }
    }

    ElemData ed(*this, eidx);
    std::vector<Point> Jv;
    basis_.derivate(coords, ed, Jv);
    result = Cross(Jv[0].asVector(), Jv[1].asVector());
    result.normalize();
  }


  void get_random_point(Point &, typename Elem::index_type, MusilRNG &rng) const;

  virtual bool synchronize(unsigned int);

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
  bool remove_orphan_nodes();
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
  bool                  insert_node(const Point &p);
  void                  insert_node(typename Face::index_type face, const Point &p);
  void                  bisect_element(const typename Face::index_type);

  bool              insert_node_in_edge_aux(typename Face::array_type &tris,
                                            typename Node::index_type &ni,
                                            unsigned int halfedge,
                                            const Point &p);

  bool              insert_node_in_face_aux(typename Face::array_type &tris,
                                            typename Node::index_type &ni,
                                            typename Face::index_type face,
                                            const Point &p);

  bool                  insert_node_in_face(typename Face::array_type &tris,
                                            typename Node::index_type &ni,
                                            typename Face::index_type face,
                                            const Point &p);


  const Point &point(typename Node::index_type i) { return points_[i]; }
  Basis& get_basis() { return basis_; }

  //! Generate the list of points that make up a sufficiently accurate
  //! piecewise linear approximation of an edge.
  void pwl_approx_edge(std::vector<std::vector<double> > &coords,
                       typename Elem::index_type ci,
                       unsigned which_edge,
                       unsigned div_per_unit) const
  {
    // Needs to match unit_edges in Basis/TriLinearLgn.cc
    // compare get_nodes order to the basis order
    basis_.approx_edge(which_edge, div_per_unit, coords);
  }

  //! Generate the list of points that make up a sufficiently accurate
  //! piecewise linear approximation of an face.
  void pwl_approx_face(std::vector<std::vector<std::vector<double> > > &coords,
                       typename Elem::index_type ci,
                       unsigned,
                       unsigned div_per_unit) const
  {
    basis_.approx_face(0, div_per_unit, coords);
  }

  bool get_coords(std::vector<double> &coords,
                  const Point &p,
                  typename Elem::index_type idx) const
  {
    ElemData ed(*this, idx);
    return basis_.get_coords(coords, p, ed);
  }

  void interpolate(Point &pt, const std::vector<double> &coords,
                   typename Elem::index_type idx) const
  {
    ElemData ed(*this, idx);
    pt = basis_.interpolate(coords, ed);
  }

  // get the Jacobian matrix
  void derivate(const std::vector<double> &coords,
                typename Elem::index_type idx,
                std::vector<Point> &J) const
  {
    ElemData ed(*this, idx);
    basis_.derivate(coords, ed, J);
  }

  double find_closest_elem(Point &result, typename Elem::index_type &elem,
                           const Point &p) const;

  double find_closest_elems(Point &result,
                            std::vector<typename Elem::index_type> &elem,
                            const Point &p) const;

  static const TypeDescription* node_type_description();
  static const TypeDescription* edge_type_description();
  static const TypeDescription* face_type_description();
  static const TypeDescription* cell_type_description();
  static const TypeDescription* elem_type_description()
  { return face_type_description(); }

  // returns a TriSurfMesh
  static Persistent *maker() { return new TriSurfMesh<Basis>(); }

private:
  void                  walk_face_orient(typename Face::index_type face,
                                         std::vector<bool> &tested,
                                         std::vector<bool> &flip);

  // These require the synchronize_lock_ to be held before calling.
  void                  compute_normals();
  void                  compute_node_neighbors();
  void                  compute_edges();
  void                  compute_edge_neighbors(double err = 1.0e-8);
  void                  compute_grid();

  //! Used to recompute data for individual cells.  Don't use these, they
  // are not synchronous.  Use create_cell_syncinfo instead.
  void insert_elem_into_grid(typename Elem::index_type ci);
  void remove_elem_from_grid(typename Elem::index_type ci);

  void                  debug_test_edge_neighbors();

  bool inside3_p(int face_times_three, const Point &p) const;

  static int next(int i) { return ((i%3)==2) ? (i-2) : (i+1); }
  static int prev(int i) { return ((i%3)==0) ? (i+2) : (i-1); }

  std::vector<Point>         points_;
  std::vector<under_type>    edges_;  // edges->halfedge map
  std::vector<under_type>    halfedge_to_edge_;  // halfedge->edge map
  std::vector<under_type>    faces_;
  std::vector<under_type>    edge_neighbors_;
  std::vector<Vector>        normals_;   //! normalized per node normal.
  std::vector<std::vector<under_type> > node_neighbors_;
  LockingHandle<SearchGridConstructor> grid_;
  unsigned int          synchronized_;
  Mutex                 synchronize_lock_;
  Basis                 basis_;

  Vector elem_epsilon_;

#ifdef HAVE_HASH_MAP

  struct edgehash
  {
    size_t operator()(const pair<int, int> &a) const
    {
#if defined(__ECC) || defined(_MSC_VER)
      hash_compare<int> hasher;
#else
      hash<int> hasher;
#endif
      return hasher(hasher(a.first) + a.second);
    }
#if defined(__ECC) || defined(_MSC_VER)

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
      return (a.first < b.first) || (a.first == b.first && a.second < b.second);
    }
  };

#endif

#ifdef HAVE_HASH_MAP

#if defined(__ECC) || defined(_MSC_VER)
  typedef hash_map<pair<int, int>, int, edgehash> EdgeMapType;
#else
  typedef hash_map<pair<int, int>, int, edgehash, edgecompare> EdgeMapType;
#endif

#else

  typedef map<pair<int, int>, int, edgecompare> EdgeMapType;

#endif

#ifdef HAVE_HASH_MAP

#if defined(__ECC) || defined(_MSC_VER)
  typedef hash_map<pair<int, int>, list<int>, edgehash> EdgeMapType2;
#else
  typedef hash_map<pair<int, int>, list<int>, edgehash, edgecompare> EdgeMapType2;
#endif

#else

  typedef map<pair<int, int>, list<int>, edgecompare> EdgeMapType2;

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
    grid_(0),
    synchronized_(NODES_E | FACES_E | CELLS_E),
    synchronize_lock_("TriSurfMesh synchronize_lock_")
{
}


template <class Basis>
TriSurfMesh<Basis>::TriSurfMesh(const TriSurfMesh &copy)
  : points_(0),
    edges_(0),
    halfedge_to_edge_(0),
    faces_(0),
    edge_neighbors_(0),
    normals_(0),
    node_neighbors_(0),
    grid_(0),
    synchronized_(NODES_E | FACES_E | CELLS_E),
    synchronize_lock_("TriSurfMesh synchronize_lock_")
{
  TriSurfMesh &lcopy = (TriSurfMesh &)copy;

  lcopy.synchronize_lock_.lock();

  points_ = copy.points_;

  edges_ = copy.edges_;
  halfedge_to_edge_ = copy.halfedge_to_edge_;
  synchronized_ |= copy.synchronized_ & EDGES_E;

  faces_ = copy.faces_;

  edge_neighbors_ = copy.edge_neighbors_;
  synchronized_ |= copy.synchronized_ & EDGE_NEIGHBORS_E;

  normals_ = copy.normals_;
  synchronized_ |= copy.synchronized_ & NORMALS_E;

  node_neighbors_ = copy.node_neighbors_;
  synchronized_ |= copy.synchronized_ & NODE_NEIGHBORS_E;

  synchronized_ &= ~LOCATE_E;
  if (copy.grid_.get_rep())
  {
    grid_ = scinew SearchGridConstructor(*(copy.grid_.get_rep()));
    elem_epsilon_ = copy.elem_epsilon_;
  }
  synchronized_ |= copy.synchronized_ & LOCATE_E;

  lcopy.synchronize_lock_.unlock();
}


template <class Basis>
TriSurfMesh<Basis>::~TriSurfMesh()
{
}


template <class Basis>
void
TriSurfMesh<Basis>::get_random_point(Point &p,
                                     typename Elem::index_type ei,
                                     MusilRNG &rng) const
{
  uniform_sample_triangle(p,
                          points_[faces_[ei*3+0]],
                          points_[faces_[ei*3+1]], 
                          points_[faces_[ei*3+2]],
                          rng);
}


template <class Basis>
BBox
TriSurfMesh<Basis>::get_bounding_box() const
{
  BBox result;

  for (std::vector<Point>::size_type i = 0; i < points_.size(); i++)
  {
    result.extend(points_[i]);
  }

  return result;
}


template <class Basis>
void
TriSurfMesh<Basis>::transform(const Transform &t)
{
  synchronize_lock_.lock();
  std::vector<Point>::iterator itr = points_.begin();
  std::vector<Point>::iterator eitr = points_.end();
  while (itr != eitr)
  {
    *itr = t.project(*itr);
    ++itr;
  }
  if (grid_.get_rep()) { grid_->transform(t); }
  synchronize_lock_.unlock();
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
TriSurfMesh<Basis>::get_nodes(typename Node::array_type &array,
                              typename Edge::index_type idx) const
{
  int a = edges_[idx];
  int b = a - a % 3 + (a+1) % 3;
  array.clear();
  array.push_back(faces_[a]);
  array.push_back(faces_[b]);
}


template <class Basis>
void
TriSurfMesh<Basis>::get_nodes(typename Node::array_type &array,
                              typename Face::index_type idx) const
{
  array.clear();
  array.push_back(faces_[idx * 3 + 0]);
  array.push_back(faces_[idx * 3 + 1]);
  array.push_back(faces_[idx * 3 + 2]);
}


template <class Basis>
void
TriSurfMesh<Basis>::get_nodes(typename Node::array_type &array,
                              typename Cell::index_type cidx) const
{
  array.clear();
  array.push_back(faces_[cidx * 3 + 0]);
  array.push_back(faces_[cidx * 3 + 1]);
  array.push_back(faces_[cidx * 3 + 2]);
}


template <class Basis>
void
TriSurfMesh<Basis>::get_edges(typename Edge::array_type &array,
                              typename Face::index_type idx) const
{
  ASSERTMSG(synchronized_ & EDGES_E,
            "Must call synchronize EDGES_E on TriSurfMesh first");

  array.clear();

  array.push_back(halfedge_to_edge_[idx * 3 + 0]);
  array.push_back(halfedge_to_edge_[idx * 3 + 1]);
  array.push_back(halfedge_to_edge_[idx * 3 + 2]);
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
bool
TriSurfMesh<Basis>::get_neighbor(unsigned int &nbr_half_edge,
                                 unsigned int half_edge) const
{
  ASSERTMSG(synchronized_ & EDGE_NEIGHBORS_E,
            "Must call synchronize EDGE_NEIGHBORS_E on TriSurfMesh first");
  nbr_half_edge = edge_neighbors_[half_edge];
  return nbr_half_edge != MESH_NO_NEIGHBOR;
}


template <class Basis>
void
TriSurfMesh<Basis>::compute_node_neighbors()
{
  node_neighbors_.clear();
  node_neighbors_.resize(points_.size());
  unsigned int nfaces = faces_.size();
  for (unsigned int f = 0; f < nfaces; ++f)
  {
    node_neighbors_[faces_[f]].push_back(f);
  }
  synchronized_ |= NODE_NEIGHBORS_E;
}


//! Returns all nodes that share an edge with this node
template <class Basis>
void
TriSurfMesh<Basis>::get_neighbors(std::vector<typename Node::index_type> &array,
                                  typename Node::index_type idx) const
{
  ASSERTMSG(synchronized_ & NODE_NEIGHBORS_E,
            "Must call synchronize NODE_NEIGHBORS_E on TriSurfMesh first");

  set<under_type> unique;
  for (unsigned int i = 0; i < node_neighbors_[idx].size(); ++i)
  {
    const under_type f = node_neighbors_[idx][i];
    unique.insert(faces_[prev(f)]);
    unique.insert(faces_[next(f)]);
  }

  array.resize(unique.size());
  copy(unique.begin(), unique.end(), array.begin());
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


template <class Basis>
bool
TriSurfMesh<Basis>::locate(typename Edge::index_type &loc,
                           const Point &p) const
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
TriSurfMesh<Basis>::locate(typename Face::index_type &loc,
                           const Point &p) const
{
  if (basis_.polynomial_order() > 1) return elem_locate(loc, *this, p);

  ASSERTMSG(synchronized_ & LOCATE_E,
            "TriSurfMesh::locate requires synchronization.");

  const list<unsigned int> *candidates;
  if (grid_->lookup(candidates, p))
  {
    list<unsigned int>::const_iterator iter = candidates->begin();
    while (iter != candidates->end())
    {
      if (inside3_p(*iter * 3, p))
      {
        loc = typename Face::index_type(*iter);
        return true;
      }
      ++iter;
    }
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


template <class Basis>
int
TriSurfMesh<Basis>::get_weights(const Point &p, typename Face::array_type &l,
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
TriSurfMesh<Basis>::get_weights(const Point &p, typename Node::array_type &l,
                                double *w)
{
  typename Face::index_type idx;
  if (locate(idx, p))
  {
    get_nodes(l,idx);
    std::vector<double> coords(2);
    if (get_coords(coords, p, idx)) {
      basis_.get_weights(coords, w);
      return basis_.dofs();
    }
  }
  return 0;
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

  // For the point to be inside a triangle it must be inside one
  // of the four triangles that can be formed by using three of the
  // triangle vertices and the point in question.
  return fabs(s - a) < MIN_ELEMENT_VAL && a > MIN_ELEMENT_VAL;
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
  synchronize_lock_.lock();
  if (((tosync & EDGES_E) && !(synchronized_ & EDGES_E)) ||
      ((tosync & LOCATE_E) && !(synchronized_ & EDGES_E)))
    compute_edges();
  if ((tosync & NORMALS_E) && !(synchronized_ & NORMALS_E))
    compute_normals();
  if ((tosync & NODE_NEIGHBORS_E) && !(synchronized_ & NODE_NEIGHBORS_E))
    compute_node_neighbors();
  if ((tosync & EDGE_NEIGHBORS_E) && !(synchronized_ & EDGE_NEIGHBORS_E))
    compute_edge_neighbors();
  if ((tosync & LOCATE_E) && !(synchronized_ & LOCATE_E))
    compute_grid();
  synchronize_lock_.unlock();
  return true;
}


template <class Basis>
void
TriSurfMesh<Basis>::compute_normals()
{
  normals_.resize(points_.size()); // 1 per node

  // build table of faces that touch each node
  std::vector<std::vector<typename Face::index_type> > node_in_faces(points_.size());
  //! face normals (not normalized) so that magnitude is also the area.
  std::vector<Vector> face_normals(faces_.size());
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
  typename std::vector<std::vector<typename Face::index_type> >::iterator nif_iter =
    node_in_faces.begin();
  int i = 0;
  while (nif_iter != node_in_faces.end()) {
    const std::vector<typename Face::index_type> &v = *nif_iter;
    typename std::vector<typename Face::index_type>::const_iterator fiter =
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

  synchronize_lock_.lock();

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

  synchronize_lock_.unlock();
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


template <class Basis>
void
TriSurfMesh<Basis>::debug_test_edge_neighbors()
{
  for (unsigned int i = 0; i < edge_neighbors_.size(); i++)
  {
    if (edge_neighbors_[i] != MESH_NO_NEIGHBOR &&
        edge_neighbors_[edge_neighbors_[i]] != i)
    {
      cout << "bad nbr[" << i << "] = " << edge_neighbors_[i] << ", nbr[" << edge_neighbors_[i] << "] = " << edge_neighbors_[edge_neighbors_[i]] << "\n";
    }
  }
}

template <class Basis>
bool
TriSurfMesh<Basis>::insert_node_in_edge_aux(typename Face::array_type &tris,
                                            typename Node::index_type &ni,
                                            unsigned int halfedge,
                                            const Point &p)
{
  ni = add_point(p);

  synchronize_lock_.lock();

  remove_elem_from_grid(halfedge/3);

  tris.clear();

  const unsigned int nbr = edge_neighbors_[halfedge];

  // f1
  const unsigned int f1 = faces_.size();
  
  tris.push_back(f1 / 3);
  faces_.push_back(ni);
  faces_.push_back(faces_[next(halfedge)]);
  faces_.push_back(faces_[prev(halfedge)]);
  edge_neighbors_.push_back(nbr);
  edge_neighbors_.push_back(edge_neighbors_[next(halfedge)]);
  edge_neighbors_.push_back(next(halfedge));

  if (edge_neighbors_[next(halfedge)] != MESH_NO_NEIGHBOR)
  {
    edge_neighbors_[edge_neighbors_[next(halfedge)]] = next(f1);
  }

  const unsigned int f3 = faces_.size(); // Only created if there's a neighbor.

  // f0
  tris.push_back(halfedge / 3);
  faces_[next(halfedge)] = ni;
  edge_neighbors_[halfedge] = (nbr!=MESH_NO_NEIGHBOR)?f3:MESH_NO_NEIGHBOR;
  edge_neighbors_[next(halfedge)] = prev(f1);
  edge_neighbors_[prev(halfedge)] = edge_neighbors_[prev(halfedge)];

  if (nbr != MESH_NO_NEIGHBOR)
  {
    remove_elem_from_grid(nbr / 3);

    // f3
    tris.push_back(f3 / 3);
    faces_.push_back(ni);
    faces_.push_back(faces_[next(nbr)]);
    faces_.push_back(faces_[prev(nbr)]);
    edge_neighbors_.push_back(halfedge);
    edge_neighbors_.push_back(edge_neighbors_[next(nbr)]);
    edge_neighbors_.push_back(next(nbr));
    
    if (edge_neighbors_[next(nbr)] != MESH_NO_NEIGHBOR)
    {
      edge_neighbors_[edge_neighbors_[next(nbr)]] = next(f3);
    }

    // f2
    tris.push_back(nbr / 3);
    faces_[next(nbr)] = ni;
    edge_neighbors_[nbr] = f1;
    edge_neighbors_[next(nbr)] = f3+2;
  }

  debug_test_edge_neighbors();

  synchronized_ &= ~NODE_NEIGHBORS_E;
  synchronized_ &= ~EDGES_E;
  synchronized_ &= ~NORMALS_E;

  for (unsigned int i = 0; i < tris.size(); i++)
  {
    insert_elem_into_grid(tris[i]);
  }

  synchronize_lock_.unlock();

  return true;
}


template <class Basis>
bool
TriSurfMesh<Basis>::insert_node_in_face_aux(typename Face::array_type &tris,
                                            typename Node::index_type &ni,
                                            typename Face::index_type face,
                                            const Point &p)
{
  ni = add_point(p);

  synchronize_lock_.lock();

  remove_elem_from_grid(face);
  
  const unsigned f0 = face*3;
  const unsigned f1 = faces_.size();
  const unsigned f2 = f1+3;

  tris.clear();

  if (edge_neighbors_[f0+1] != MESH_NO_NEIGHBOR)
  {
    edge_neighbors_[edge_neighbors_[f0+1]] = f1+0;
  }
  if (edge_neighbors_[f0+2] != MESH_NO_NEIGHBOR)
  {
    edge_neighbors_[edge_neighbors_[f0+2]] = f2+0;
  }

  tris.push_back(faces_.size() / 3);
  faces_.push_back(faces_[f0+1]);
  faces_.push_back(faces_[f0+2]);
  faces_.push_back(ni);
  edge_neighbors_.push_back(edge_neighbors_[f0+1]);
  edge_neighbors_.push_back(f2+2);
  edge_neighbors_.push_back(f0+1);

  tris.push_back(faces_.size() / 3);
  faces_.push_back(faces_[f0+2]);
  faces_.push_back(faces_[f0+0]);
  faces_.push_back(ni);
  edge_neighbors_.push_back(edge_neighbors_[f0+2]);
  edge_neighbors_.push_back(f0+2);
  edge_neighbors_.push_back(f1+1);

  // Must do last
  tris.push_back(face);
  faces_[f0+2] = ni;
  edge_neighbors_[f0+1] = f1+2;
  edge_neighbors_[f0+2] = f2+1;

  debug_test_edge_neighbors();

  synchronized_ &= ~NODE_NEIGHBORS_E;
  synchronized_ &= ~EDGES_E;
  synchronized_ &= ~NORMALS_E;

  for (unsigned int i = 0; i < tris.size(); i++)
  {
    insert_elem_into_grid(tris[i]);
  }

  synchronize_lock_.unlock();

  return true;
}


template <class Basis>
bool
TriSurfMesh<Basis>::insert_node_in_face(typename Face::array_type &tris,
                                        typename Node::index_type &ni,
                                        typename Face::index_type face,
                                        const Point &p)
{
  const Point &p0 = point(faces_[face * 3 + 0]);
  const Point &p1 = point(faces_[face * 3 + 1]);
  const Point &p2 = point(faces_[face * 3 + 2]);

  const double a0 = Cross(p - p1, p - p2).length2();
  const double a1 = Cross(p - p2, p - p0).length2();
  const double a2 = Cross(p - p0, p - p1).length2();

  unsigned int mask = 0;
  if (a0 >= MIN_ELEMENT_VAL) { mask |= 1; }
  if (a1 >= MIN_ELEMENT_VAL) { mask |= 2; }
  if (a2 >= MIN_ELEMENT_VAL) { mask |= 4; }

  if (mask == 7)
  {
    // Point is inside the face, do a three split.
    return insert_node_in_face_aux(tris, ni, face, p);
  }
  else if (mask == 0)
  {
    // Tri is degenerate, just return first point.
    tris.clear();
    tris.push_back(face);
    ni = faces_[face * 3 + 0];
    return true;
  }
  // The point is on a corner, return that corner.
  else if (mask == 1)
  {
    tris.clear();
    tris.push_back(face);
    ni = faces_[face * 3 + 0];
    return true;
  }
  else if (mask == 2)
  {
    tris.clear();
    tris.push_back(face);
    ni = faces_[face * 3 + 1];
    return true;
  }
  else if (mask == 4)
  {
    tris.clear();
    tris.push_back(face);
    ni = faces_[face * 3 + 2];
    return true;
  }
  // The point is on an edge, split that edge and neighboring triangle.
  else if (mask == 3)
  {
    return insert_node_in_edge_aux(tris, ni, face*3+0, p);
  }
  else if (mask == 5)
  {
    return insert_node_in_edge_aux(tris, ni, face*3+2, p);
  }
  else if (mask == 6)
  {
    return insert_node_in_edge_aux(tris, ni, face*3+1, p);
  }
  return false;
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
  std::vector<Vector> normals(3);
  for (int edge = 0; edge < 3; ++edge)
  {
    Point p = ((points_[faces_[f0+edge]] +
                points_[faces_[next(f0+edge)]]) / 2.0).asPoint();
    nodes.push_back(add_point(p));

    if (do_normals)
    {
      normals[edge] = Vector(normals_[faces_[f0+edge]] +
                             normals_[faces_[next(f0+edge)]]);
      normals[edge].safe_normalize();
    }
  }

  synchronize_lock_.lock();

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

  synchronize_lock_.unlock();
}


template <class Basis>
void
TriSurfMesh<Basis>::compute_edges()
{
  EdgeMapType2 edge_map;

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
    edge_map[nodes].push_front(i);
  }

  edges_.clear();
  edges_.reserve(edge_map.size());
  halfedge_to_edge_.resize(faces_.size());
  
  for( typename EdgeMapType2::iterator itr( edge_map.begin() ); itr != edge_map.end(); ++itr )
  {
    edges_.push_back((*itr).second.front());

    list<int>::iterator litr = (*itr).second.begin();
    while (litr != (*itr).second.end())
    {
      halfedge_to_edge_[*litr] = edges_.size()-1;
      ++litr;
    }
  }

  synchronized_ |= EDGES_E;
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
    synchronize_lock_.lock();
    points_.push_back(p);
    node_neighbors_.push_back(std::vector<under_type>());
    synchronize_lock_.unlock();
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
  if (!p.second) { *ns = faces_[face2 + 2]; }

  // no shared nodes means no shared edge.
  if (shared.size() > 4) return false;

  set<int, less_int>::iterator iter = shared.find(not_shar[0]);
  shared.erase(iter);

  iter = shared.find(not_shar[1]);
  shared.erase(iter);

  iter = shared.begin();
  int s1 = *iter++;
  int s2 = *iter;

  synchronize_lock_.lock();
  faces_[face1] = s1;
  faces_[face1 + 1] = not_shar[0];
  faces_[face1 + 2] = s2;

  faces_[face2] = s2;
  faces_[face2 + 1] = not_shar[1];
  faces_[face2 + 2] = s1;

  synchronized_ &= ~EDGE_NEIGHBORS_E;
  synchronized_ &= ~NODE_NEIGHBORS_E;
  synchronized_ &= ~NORMALS_E;
  synchronize_lock_.unlock();

  return true;
}

template <class Basis>
bool
TriSurfMesh<Basis>::remove_orphan_nodes()
{
  bool rval = false;
  
  //! find the orphan nodes.
  std::vector<under_type> onodes;
  //! check each point against the face list.
  for (under_type i = 0; i < points_.size(); i++) {
    if (find(faces_.begin(), faces_.end(), i) == faces_.end()) {
      //! node does not belong to a face
      onodes.push_back(i);
    }
  }

  if (onodes.size()) rval = true;

  //! check each point against the face list.
  std::vector<under_type>::reverse_iterator orph_iter = onodes.rbegin();
  while (orph_iter != onodes.rend()) {
    unsigned int i = *orph_iter++;
    std::vector<under_type>::iterator iter = faces_.begin();
    while (iter != faces_.end()) {
      under_type &node = *iter++;
      if (node > i) {
	node--;
      }
    }
    std::vector<Point>::iterator niter = points_.begin();
    niter += i;
    points_.erase(niter);
  }

  synchronized_ &= ~EDGE_NEIGHBORS_E;
  synchronized_ &= ~NODE_NEIGHBORS_E;
  synchronized_ &= ~NORMALS_E;
  return rval;
}

template <class Basis>
bool
TriSurfMesh<Basis>::remove_face(typename Face::index_type f)
{
  bool rval = true;

  synchronize_lock_.lock();
  std::vector<under_type>::iterator fb = faces_.begin() + f*3;
  std::vector<under_type>::iterator fe = fb + 3;

  if (fe <= faces_.end())
    faces_.erase(fb, fe);
  else {
    rval = false;
  }
  synchronized_ &= ~EDGE_NEIGHBORS_E;
  synchronized_ &= ~NODE_NEIGHBORS_E;
  synchronized_ &= ~NORMALS_E;
  synchronize_lock_.unlock();

  return rval;
}


template <class Basis>
void
TriSurfMesh<Basis>::add_triangle(typename Node::index_type a,
                                 typename Node::index_type b,
                                 typename Node::index_type c)
{
  synchronize_lock_.lock();
  faces_.push_back(a);
  faces_.push_back(b);
  faces_.push_back(c);
  synchronized_ &= ~EDGE_NEIGHBORS_E;
  synchronized_ &= ~NODE_NEIGHBORS_E;
  synchronized_ &= ~NORMALS_E;
  synchronize_lock_.unlock();
}


template <class Basis>
typename TriSurfMesh<Basis>::Elem::index_type
TriSurfMesh<Basis>::add_elem(typename Node::array_type a)
{
  ASSERTMSG(a.size() == 3, "Tried to add non-tri element.");

  synchronize_lock_.lock();
  faces_.push_back(a[0]);
  faces_.push_back(a[1]);
  faces_.push_back(a[2]);
  synchronized_ &= ~EDGE_NEIGHBORS_E;
  synchronized_ &= ~NODE_NEIGHBORS_E;
  synchronized_ &= ~NORMALS_E;
  synchronize_lock_.unlock();
  return static_cast<typename Elem::index_type>((faces_.size() - 1) / 3);
}


template <class Basis>
void
TriSurfMesh<Basis>::flip_faces()
{
  synchronize_lock_.lock();
  typename Face::iterator fiter, fend;
  begin(fiter);
  end(fend);
  while (fiter != fend)
  {
    flip_face(*fiter);
    ++fiter;
  }
  synchronized_ &= ~EDGES_E;
  synchronized_ &= ~EDGE_NEIGHBORS_E;
  synchronized_ &= ~NODE_NEIGHBORS_E;
  synchronized_ &= ~NORMALS_E;
  synchronize_lock_.unlock();
}


template <class Basis>
void
TriSurfMesh<Basis>::flip_face(typename Face::index_type face)
{
  const unsigned int base = face * 3;
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
                                     std::vector<bool> &tested,
                                     std::vector<bool> &flip)
{
  tested[face] = true;
  for (unsigned int i = 0; i < 3; i++)
  {
    const unsigned int edge = face * 3 + i;
    const unsigned int nbr = edge_neighbors_[edge];
    if (nbr != MESH_NO_NEIGHBOR && !tested[nbr/3])
    {
      if (!flip[face] && faces_[edge] == faces_[nbr] ||
          flip[face] && faces_[next(edge)] == faces_[nbr])
      {
        flip[nbr/3] = true;
      }
      walk_face_orient(nbr/3, tested, flip);
    }
  }
}

template <class Basis>
void
TriSurfMesh<Basis>::orient_faces()
{
  synchronize(EDGES_E | EDGE_NEIGHBORS_E);
  synchronize_lock_.lock();

  int nfaces = (int)faces_.size() / 3;
  std::vector<bool> tested(nfaces, false);
  std::vector<bool> flip(nfaces, false);

  typename Face::iterator fiter, fend;
  begin(fiter);
  end(fend);
  while (fiter != fend)
  {
    if (! tested[*fiter])
    {
      walk_face_orient(*fiter, tested, flip);
    }
    ++fiter;
  }

  begin(fiter);
  while (fiter != fend)
  {
    if (flip[*fiter])
    {
      flip_face(*fiter);
    }
    ++fiter;
  }

  synchronized_ &= ~EDGES_E;
  synchronized_ &= ~EDGE_NEIGHBORS_E;
  synchronized_ &= ~NORMALS_E;
  synchronize_lock_.unlock();
}


template <class Basis>
void
TriSurfMesh<Basis>::compute_edge_neighbors(double /*err*/)
{
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

    typename EdgeMapType::iterator maploc( edge_map.find(nodes) );

    if( maploc != edge_map.end() )
    {
      edge_neighbors_[(*maploc).second] = i;
      edge_neighbors_[i] = (*maploc).second;
    }
    edge_map[nodes] = i;
  }

  debug_test_edge_neighbors();

  synchronized_ |= EDGE_NEIGHBORS_E;
}


template <class Basis>
void
TriSurfMesh<Basis>::insert_elem_into_grid(typename Elem::index_type ci)
{
  // TODO:  This can crash if you insert a new cell outside of the grid.
  // Need to recompute grid at that point.

  BBox box;
  box.extend(points_[faces_[ci*3+0]]);
  box.extend(points_[faces_[ci*3+1]]);
  box.extend(points_[faces_[ci*3+2]]);
  const Point padmin(box.min() - elem_epsilon_);
  const Point padmax(box.max() + elem_epsilon_);
  box.extend(padmin);
  box.extend(padmax);
  grid_->insert(ci, box);
}


template <class Basis>
void
TriSurfMesh<Basis>::remove_elem_from_grid(typename Elem::index_type ci)
{
  BBox box;
  box.extend(points_[faces_[ci*3+0]]);
  box.extend(points_[faces_[ci*3+1]]);
  box.extend(points_[faces_[ci*3+2]]);
  const Point padmin(box.min() - elem_epsilon_);
  const Point padmax(box.max() + elem_epsilon_);
  box.extend(padmin);
  box.extend(padmax);
  grid_->remove(ci, box);
}


template <class Basis>
void
TriSurfMesh<Basis>::compute_grid()
{
  BBox bb = get_bounding_box();
  if (bb.valid())
  {
    // Cubed root of number of cells to get a subdivision ballpark.
    typename Elem::size_type csize;  size(csize);
    const int s = (int)(ceil(pow((double)csize , (1.0/3.0)))) / 2 + 1;
    int sx, sy, sz; sx = sy = sz = s;
    elem_epsilon_ = bb.diagonal() * (1.0e-3 / s);
    if (elem_epsilon_.x() < MIN_ELEMENT_VAL)
    {
      elem_epsilon_.x(MIN_ELEMENT_VAL * 100);
      sx = 1;
    }
    if (elem_epsilon_.y() < MIN_ELEMENT_VAL)
    {
      elem_epsilon_.y(MIN_ELEMENT_VAL * 100);
      sy = 1;
    }
    if (elem_epsilon_.z() < MIN_ELEMENT_VAL)
    {
      elem_epsilon_.z(MIN_ELEMENT_VAL * 100);
      sz = 1;
    }
    bb.extend(bb.min() - elem_epsilon_ * 10);
    bb.extend(bb.max() + elem_epsilon_ * 10);

    grid_ = scinew SearchGridConstructor(sx, sy, sz, bb.min(), bb.max());

    typename Node::array_type nodes;
    typename Elem::iterator ci, cie;
    begin(ci); end(cie);
    while(ci != cie)
    {
      insert_elem_into_grid(*ci);
      ++ci;
    }
  }

  synchronized_ |= LOCATE_E;
}


template <class Basis>
typename TriSurfMesh<Basis>::Node::index_type
TriSurfMesh<Basis>::add_point(const Point &p)
{
  synchronize_lock_.lock();
  points_.push_back(p);
  if (synchronized_ & NORMALS_E) normals_.push_back(Vector());
  if (synchronized_ & NODE_NEIGHBORS_E)
  {
    node_neighbors_.push_back(std::vector<under_type>());
  }
  synchronize_lock_.unlock();
  return static_cast<typename Node::index_type>(points_.size() - 1);
}



template <class Basis>
void
TriSurfMesh<Basis>::add_triangle(const Point &p0,
                                 const Point &p1,
                                 const Point &p2)
{
  add_triangle(add_find_point(p0), add_find_point(p1), add_find_point(p2));
}


#define TRISURFMESH_VERSION 2

template <class Basis>
void
TriSurfMesh<Basis>::io(Piostream &stream)
{
  int version = stream.begin_class(type_name(-1), TRISURFMESH_VERSION);

  Mesh::io(stream);

  Pio(stream, points_);
  Pio(stream, faces_);
  if (stream.writing()) {
    synchronize(EDGE_NEIGHBORS_E);
  }
  Pio(stream, edge_neighbors_);

  if (version >= 2) {
    basis_.io(stream);
  }

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
double
TriSurfMesh<Basis>::find_closest_elem(Point &result,
                                      typename TriSurfMesh::Elem::index_type &face,
                                      const Point &p) const
{
  // Walking the grid like this works really well if we're near the
  // surface.  It's degenerately bad if for example the point is
  // placed in the center of a sphere (because then we still have to
  // test all the faces, but with the grid overhead and triangle
  // duplication as well).
  ASSERTMSG(synchronized_ & LOCATE_E,
            "TriSurfMesh::find_closest_elem requires synchronize(LOCATE_E).")

  // Convert to grid coordinates.
  int oi, oj, ok;
  grid_->unsafe_locate(oi, oj, ok, p);

  // Clamp to closest point on the grid.
  oi = Max(Min(oi, grid_->get_ni()-1), 0);
  oj = Max(Min(oj, grid_->get_nj()-1), 0);
  ok = Max(Min(ok, grid_->get_nk()-1), 0);

  int bi, ei, bj, ej, bk, ek;
  bi = ei = oi;
  bj = ej = oj;
  bk = ek = ok;
  
  double dmin = DBL_MAX;
  bool found;
  do {
    const int bii = Max(bi, 0);
    const int eii = Min(ei, grid_->get_ni()-1);
    const int bjj = Max(bj, 0);
    const int ejj = Min(ej, grid_->get_nj()-1);
    const int bkk = Max(bk, 0);
    const int ekk = Min(ek, grid_->get_nk()-1);
    found = false;
    for (int i = bii; i <= eii; i++)
    {
      for (int j = bjj; j <= ejj; j++)
      {
        for (int k = bkk; k <= ekk; k++)
        {
          if (i == bi || i == ei || j == bj || j == ej || k == bk || k == ek)
          {
            if (grid_->min_distance_squared(p, i, j, k) < dmin)
            {
              found = true;
              const list<unsigned int> *candidates;
              grid_->lookup_ijk(candidates, i, j, k);

              list<unsigned int>::const_iterator iter = candidates->begin();
              while (iter != candidates->end())
              {
                Point rtmp;
                unsigned int idx = *iter * 3;
                closest_point_on_tri(rtmp, p,
                                     points_[faces_[idx  ]],
                                     points_[faces_[idx+1]],
                                     points_[faces_[idx+2]]);
                const double dtmp = (p - rtmp).length2();
                if (dtmp < dmin)
                {
                  result = rtmp;
                  face = *iter;
                  dmin = dtmp;
                }
                ++iter;
              }
            }
          }
        }
      }
    }
    bi--;ei++;
    bj--;ej++;
    bk--;ek++;
  } while (found) ;

#if 0
  // The old code, useful for debugging purposes.  Note that the old
  // and new code don't necessarily return the same face because if
  // you hit an edge or corner any of the edge or corner faces are
  // valid.
  double dmin2 = DBL_MAX;
  Point result2;
  typename Face::index_type face2;
  for (unsigned int i = 0; i < faces_.size(); i+=3)
  {
    Point rtmp;
    closest_point_on_tri(rtmp, p,
                         points_[faces_[i  ]],
                         points_[faces_[i+1]],
                         points_[faces_[i+2]]);
    const double dtmp = (p - rtmp).length2();
    if (dtmp < dmin2)
    {
      result2 = rtmp;
      face2 = i/3;
      dmin2 = dtmp;
    }
  }
  if (face != face2)
  {
    cout << "face != face2 (" << face << " " << face2 << "\n";
    cout << "dmin = " << dmin << ", dmin2 = " << dmin2 << "\n";
  }
#endif

  return sqrt(dmin);
}


template <class Basis>
double
TriSurfMesh<Basis>::find_closest_elems(Point &result,
                                       std::vector<typename TriSurfMesh::Elem::index_type> &faces,
                                       const Point &p) const
{
  // Walking the grid like this works really well if we're near the
  // surface.  It's degenerately bad if for example the point is
  // placed in the center of a sphere (because then we still have to
  // test all the faces, but with the grid overhead and triangle
  // duplication as well).
  ASSERTMSG(synchronized_ & LOCATE_E,
            "TriSurfMesh::find_closest_elem requires synchronize(LOCATE_E).")

  // Convert to grid coordinates.
  int oi, oj, ok;
  grid_->unsafe_locate(oi, oj, ok, p);

  // Clamp to closest point on the grid.
  oi = Max(Min(oi, grid_->get_ni()-1), 0);
  oj = Max(Min(oj, grid_->get_nj()-1), 0);
  ok = Max(Min(ok, grid_->get_nk()-1), 0);

  int bi, ei, bj, ej, bk, ek;
  bi = ei = oi;
  bj = ej = oj;
  bk = ek = ok;
  
  double dmin = DBL_MAX;
  bool found;
  do {
    const int bii = Max(bi, 0);
    const int eii = Min(ei, grid_->get_ni()-1);
    const int bjj = Max(bj, 0);
    const int ejj = Min(ej, grid_->get_nj()-1);
    const int bkk = Max(bk, 0);
    const int ekk = Min(ek, grid_->get_nk()-1);
    found = false;
    for (int i = bii; i <= eii; i++)
    {
      for (int j = bjj; j <= ejj; j++)
      {
        for (int k = bkk; k <= ekk; k++)
        {
          if (i == bi || i == ei || j == bj || j == ej || k == bk || k == ek)
          {
            if (grid_->min_distance_squared(p, i, j, k) < dmin)
            {
              found = true;
              const list<unsigned int> *candidates;
              grid_->lookup_ijk(candidates, i, j, k);

              list<unsigned int>::const_iterator iter = candidates->begin();
              while (iter != candidates->end())
              {
                Point rtmp;
                unsigned int idx = *iter * 3;
                closest_point_on_tri(rtmp, p,
                                     points_[faces_[idx  ]],
                                     points_[faces_[idx+1]],
                                     points_[faces_[idx+2]]);
                const double dtmp = (p - rtmp).length2();
                if (dtmp < dmin - MIN_ELEMENT_VAL)
                {
                  faces.clear();
                  result = rtmp;
                  faces.push_back(*iter);
                  dmin = dtmp;
                }
                else if (dtmp < dmin + MIN_ELEMENT_VAL)
                {
                  faces.push_back(*iter);
                }
                ++iter;
              }
            }
          }
        }
      }
    }
    bi--;ei++;
    bj--;ej++;
    bk--;ek++;
  } while (found) ;

  return sqrt(dmin);
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
    td = scinew TypeDescription("TriSurfMesh", subs,
                                string(__FILE__),
                                "SCIRun",
                                TypeDescription::MESH_E);
  }
  return td;
}


template <class Basis>
const TypeDescription*
TriSurfMesh<Basis>::get_type_description() const
{
  return SCIRun::get_type_description((TriSurfMesh<Basis> *)0);
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
                                "SCIRun",
                                TypeDescription::MESH_E);
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
                                "SCIRun",
                                TypeDescription::MESH_E);
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
                                "SCIRun",
                                TypeDescription::MESH_E);
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
                                "SCIRun",
                                TypeDescription::MESH_E);
  }
  return td;
}


} // namespace SCIRun


#endif // SCI_project_TriSurfMesh_h
