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
#include <Core/Datatypes/Mesh.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Datatypes/FieldIterator.h>
#include <Core/Geometry/BBox.h>
#include <Core/Containers/StackVector.h>
#include <Core/Geometry/Transform.h>
#include <Core/Math/MusilRNG.h>
#include <string>
#include <vector>
#include <cfloat>

namespace SCIRun {

using std::string;
using std::vector;
using std::pair;

template <class Basis>
class CurveMesh : public Mesh
{
public:
  typedef LockingHandle<CurveMesh<Basis> > handle_type;
  typedef Basis                            basis_type;
  typedef unsigned int                     under_type;

  //! Index and Iterator types required for Mesh Concept.
  struct Node {
    typedef NodeIndex<under_type>       index_type;
    typedef NodeIterator<under_type>    iterator;
    typedef NodeIndex<under_type>       size_type;
    typedef StackVector<index_type, 2>  array_type;
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
  typedef Node DElem;

  typedef pair<typename Node::index_type,
               typename Node::index_type> index_pair_type;


  friend class ElemData;

  class ElemData
  {
  public:
    ElemData(const CurveMesh<Basis>& msh, unsigned idx) :
      mesh_(msh),
      index_(idx)
    {}

    inline
    unsigned node0_index() const {
      return mesh_.edges_[index_].first;
    }
    inline
    unsigned node1_index() const {
      return mesh_.edges_[index_].second;
    }

    inline
    const Point &node0() const {
      return mesh_.nodes_[mesh_.edges_[index_].first];
    }
    inline
    const Point &node1() const {
      return mesh_.nodes_[mesh_.edges_[index_].second];
    }

    // the following designed to coordinate with ::get_edges
    inline
    unsigned edge0_index() const {
      return index_;
    }

  private:
    const CurveMesh<Basis>   &mesh_;
    unsigned                  index_;
  };



  CurveMesh() :
    synchronized_(ALL_ELEMENTS_E),
    synchronize_lock_("CurveMesh sync lock")
  {}
  CurveMesh(const CurveMesh &copy) :
    nodes_(copy.nodes_),
    edges_(copy.edges_),
    basis_(copy.basis_),
    synchronized_(copy.synchronized_),
    synchronize_lock_("CurveMesh sync lock")
  {
    CurveMesh &lcopy = (CurveMesh &)copy;

    lcopy.synchronize_lock_.lock();
    node_neighbors_ = copy.node_neighbors_;
    synchronized_ |= copy.synchronized_ & NODE_NEIGHBORS_E;
    lcopy.synchronize_lock_.unlock();
  }
  virtual CurveMesh *clone() { return new CurveMesh(*this); }
  virtual ~CurveMesh() {}

  Basis& get_basis() { return basis_; }

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

  void to_index(typename Node::index_type &index,
                unsigned int i) const { index = i; }
  void to_index(typename Edge::index_type &index,
                unsigned int i) const { index = i; }
  void to_index(typename Face::index_type &index,
                unsigned int i) const { index = i; }
  void to_index(typename Cell::index_type &index,
                unsigned int i) const { index = i; }

  virtual BBox get_bounding_box() const;
  //? FIX_ME mjc validate this with various basis fns.
  virtual void transform(const Transform &t);

  //! get the child elements of the given index
  void get_nodes(typename Node::array_type &a,
                 typename Edge::index_type i) const
  { a.resize(2); a[0] = edges_[i].first; a[1] = edges_[i].second; }

  void get_edges(typename Edge::array_type &a,
                 typename Edge::index_type i) const
  { a.push_back(i); }

  void get_nodes(typename Node::array_type &,
                 typename Face::index_type) const {}
  void get_nodes(typename Node::array_type &,
                 typename Cell::index_type) const {}
  void get_edges(typename Edge::array_type &,
                 typename Face::index_type) const {}
  void get_edges(typename Edge::array_type &,
                 typename Cell::index_type) const {}

  // Stub, used by ShowField.
  void get_faces(typename Face::array_type &,
                 typename Elem::index_type) const {}

  //! Get the parent element(s) of the given node index.
  void get_elems(typename Elem::array_type &result,
                 typename Node::index_type idx) const
  {
    result.clear();
    for (unsigned int i = 0; i < node_neighbors_[idx].size(); ++i)
      result.push_back(node_neighbors_[idx][i]);
  }

  void get_elems(typename Elem::array_type &result,
                 typename Edge::index_type idx) const {}
  void get_elems(typename Elem::array_type &result,
                 typename Face::index_type idx) const {}
                 
  //! Wrapper to get the derivative elements from this element.
  void get_delems(typename DElem::array_type &result,
                  typename Elem::index_type idx) const
  {
    get_nodes(result, idx);
  }

  //! Defined get_nodes(Node::array, DElem::index_type)
  void get_nodes(typename Node::array_type &a,
                 typename Node::index_type i) const
  {
    a.clear();
    a.push_back(i);
  }

  //! Generate the list of points that make up a sufficiently accurate
  //! piecewise linear approximation of an edge.
  void pwl_approx_edge(vector<vector<double> > &coords,
                       typename Elem::index_type,
                       unsigned which_edge,
                       unsigned div_per_unit) const
  {
    // only one edge in the unit edge.
    basis_.approx_edge(which_edge, div_per_unit, coords);
  }

  //! Generate the list of points that make up a sufficiently accurate
  //! piecewise linear approximation of an face.
  void pwl_approx_face(vector<vector<vector<double> > > &coords,
                       typename Elem::index_type ci,
                       typename Face::index_type fi,
                       unsigned div_per_unit) const
  {
    ASSERTFAIL("CurveMesh cannot approximiate faces");
  }

  bool get_coords(vector<double> &coords,
                  const Point &p,
                  typename Edge::index_type idx) const;


  void interpolate(Point &pt, const vector<double> &coords,
                   typename Elem::index_type idx) const
  {
    ElemData ed(*this, idx);
    pt = basis_.interpolate(coords, ed);
  }

  // get the Jacobian matrix
  void derivate(const vector<double> &coords,
                typename Elem::index_type idx,
                vector<Point> &J) const
  {
    ElemData ed(*this, idx);
    basis_.derivate(coords, ed, J);
  }

  //! get the center point (in object space) of an element
  void get_center(Point &result, typename Node::index_type idx) const
  { result = nodes_[idx]; }
  void get_center(Point &, typename Edge::index_type) const;
  void get_center(Point &, typename Face::index_type) const {}
  void get_center(Point &, typename Cell::index_type) const {}

  //! Get the size of an elemnt (length, area, volume)
  double get_size(typename Node::index_type /*idx*/) const { return 0.0; }
  //! get_size for edge is chord length.
  double get_size(typename Edge::index_type idx) const;
  double get_size(typename Face::index_type /*idx*/) const
  { return 0.0; }
  double get_size(typename Cell::index_type /*idx*/) const
  { return 0.0; }
  double get_length(typename Edge::index_type idx) const
  { return get_size(idx); };
  double get_area(typename Face::index_type idx) const
  { return get_size(idx); };
  double get_volume(typename Cell::index_type idx) const
  { return get_size(idx); };

  bool get_neighbor(typename Edge::index_type &/*neighbor*/,
                    typename Edge::index_type /*edge*/,
                    typename Node::index_type /*node*/) const
  { ASSERTFAIL("CurveMesh::get_neighbor for edges needs to be implemented"); }
  void get_neighbors(vector<typename Node::index_type> &/*array*/,
                     typename Node::index_type /*idx*/) const
  { ASSERTFAIL("CurveMesh::get_neighbor for nodes needs to be implemented"); }

  int get_valence(typename Node::index_type idx) const;
  int get_valence(typename Edge::index_type /*idx*/) const { return 0; }
  int get_valence(typename Face::index_type /*idx*/) const { return 0; }
  int get_valence(typename Cell::index_type /*idx*/) const { return 0; }

  bool locate(typename Node::index_type &, const Point &) const;
  bool locate(typename Edge::index_type &, const Point &) const;
  bool locate(typename Face::index_type &, const Point &) const
  { return false; }
  bool locate(typename Cell::index_type &, const Point &) const
  { return false; }

  int get_weights(const Point &p, typename Node::array_type &l, double *w);
  int get_weights(const Point &p, typename Edge::array_type &l, double *w);
  int get_weights(const Point & , typename Face::array_type & , double * )
  {ASSERTFAIL("CurveMesh::get_weights for faces isn't supported"); }
  int get_weights(const Point & , typename Cell::array_type & , double * )
  {ASSERTFAIL("CurveMesh::get_weights for cells isn't supported"); }

  void get_point(Point &result, typename Node::index_type idx) const
  { get_center(result,idx); }
  void set_point(const Point &point, typename Node::index_type index)
  { nodes_[index] = point; }

  void get_random_point(Point &, typename Elem::index_type, MusilRNG &r) const;

  void get_normal(Vector & /* result */,
                  typename Node::index_type /* index */) const
  { ASSERTFAIL("This mesh type does not have node normals."); }
  void get_normal(Vector &, vector<double> &, typename Elem::index_type,
                  unsigned int)
  { ASSERTFAIL("This mesh type does not have element normals."); }

  //! use these to build up a new contour mesh
  typename Node::index_type add_node(const Point &p)
  {
    nodes_.push_back(p);
    return static_cast<under_type>(nodes_.size() - 1);
  }
  typename Node::index_type add_point(const Point &point)
  { return add_node(point); }
  typename Edge::index_type add_edge(typename Node::index_type i1,
                                     typename Node::index_type i2)
  {
    edges_.push_back(index_pair_type(i1,i2));
    return static_cast<under_type>(edges_.size()-1);
  }
  typename Elem::index_type add_elem(typename Node::array_type a)
  {
    ASSERTMSG(a.size() == 2, "Tried to add non-line element.");
    edges_.push_back(index_pair_type(a[0],a[1]));
    return static_cast<under_type>(edges_.size()-1);
  }
  void node_reserve(size_t s) { nodes_.reserve(s); }
  void elem_reserve(size_t s) { edges_.reserve(s*2); }

  virtual bool is_editable() const { return true; }
  virtual int  dimensionality() const { return 1; }
  virtual int  topology_geometry() const { return (UNSTRUCTURED | IRREGULAR); }

  virtual bool synchronize(unsigned int mask);

  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  static  const string type_name(int n = -1);

  virtual const TypeDescription *get_type_description() const;

  static const TypeDescription* node_type_description();
  static const TypeDescription* edge_type_description();
  static const TypeDescription* face_type_description();
  static const TypeDescription* cell_type_description();
  static const TypeDescription* elem_type_description()
  { return edge_type_description(); }

  // returns a CurveMesh
  static Persistent *maker() { return new CurveMesh<Basis>(); }

private:
  void                    compute_node_neighbors();

  vector<Point>           nodes_;
  vector<index_pair_type> edges_;
  Basis                   basis_;

  unsigned int            synchronized_;
  Mutex                   synchronize_lock_;

  typedef vector<vector<typename Edge::index_type> > NodeNeighborMap;
  NodeNeighborMap         node_neighbors_;
};



template <class Basis>
const TypeDescription* get_type_description(CurveMesh<Basis> *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    const TypeDescription *sub = get_type_description((Basis*)0);
    TypeDescription::td_vec *subs = scinew TypeDescription::td_vec(1);
    (*subs)[0] = sub;
    td = scinew TypeDescription("CurveMesh", subs,
                                string(__FILE__),
                                "SCIRun",
                                TypeDescription::MESH_E);
  }
  return td;
}


template <class Basis>
const TypeDescription*
CurveMesh<Basis>::get_type_description() const
{
  return SCIRun::get_type_description((CurveMesh<Basis> *)0);
}


template <class Basis>
const TypeDescription*
CurveMesh<Basis>::node_type_description()
{
  static TypeDescription *td = 0;
  if (!td)
  {
    const TypeDescription *me =
      SCIRun::get_type_description((CurveMesh<Basis> *)0);
    td = scinew TypeDescription(me->get_name() + "::Node",
                                string(__FILE__),
                                "SCIRun",
                                TypeDescription::MESH_E);
  }
  return td;
}


template <class Basis>
const TypeDescription*
CurveMesh<Basis>::edge_type_description()
{
  static TypeDescription *td = 0;
  if (!td)
  {
    const TypeDescription *me =
      SCIRun::get_type_description((CurveMesh<Basis> *)0);
    td = scinew TypeDescription(me->get_name() + "::Edge",
                                string(__FILE__),
                                "SCIRun",
                                TypeDescription::MESH_E);
  }
  return td;
}


template <class Basis>
const TypeDescription*
CurveMesh<Basis>::face_type_description()
{
  static TypeDescription *td = 0;
  if (!td)
  {
    const TypeDescription *me =
      SCIRun::get_type_description((CurveMesh<Basis> *)0);
    td = scinew TypeDescription(me->get_name() + "::Face",
                                string(__FILE__),
                                "SCIRun",
                                TypeDescription::MESH_E);
  }
  return td;
}


template <class Basis>
const TypeDescription*
CurveMesh<Basis>::cell_type_description()
{
  static TypeDescription *td = 0;
  if (!td)
  {
    const TypeDescription *me =
      SCIRun::get_type_description((CurveMesh<Basis> *)0);
    td = scinew TypeDescription(me->get_name() + "::Cell",
                                string(__FILE__),
                                "SCIRun",
                                TypeDescription::MESH_E);
  }
  return td;
}


template <class Basis>
PersistentTypeID
CurveMesh<Basis>::type_id(type_name(-1), "Mesh", CurveMesh<Basis>::maker);


template <class Basis>
BBox
CurveMesh<Basis>::get_bounding_box() const
{
  BBox result;

  typename Node::iterator i, ie;
  begin(i);
  end(ie);

  while (i != ie)
  {
    result.extend(nodes_[*i]);
    ++i;
  }

  return result;
}


template <class Basis>
void
CurveMesh<Basis>::transform(const Transform &t)
{
  vector<Point>::iterator itr = nodes_.begin();
  vector<Point>::iterator eitr = nodes_.end();
  while (itr != eitr)
  {
    *itr = t.project(*itr);
    ++itr;
  }
}


template <class Basis>
bool
CurveMesh<Basis>::get_coords(vector<double> &coords,
                             const Point &p,
                             typename Edge::index_type idx) const
{
  ElemData cmcd(*this, idx);
  return basis_.get_coords(coords, p, cmcd);
}


template <class Basis>
double
CurveMesh<Basis>::get_size(typename Edge::index_type idx) const
{
  ElemData ed(*this, idx);
  vector<Point> pledge;
  vector<vector<double> > coords;
  // Perhaps there is a better choice for the number of divisions.
  pwl_approx_edge(coords, idx, 0, 5);

  double total = 0.0L;
  vector<vector<double> >::iterator iter = coords.begin();
  vector<vector<double> >::iterator last = coords.begin() + 1;
  while (last != coords.end()) {
    vector<double> &c0 = *iter++;
    vector<double> &c1 = *last++;
    Point p0 = basis_.interpolate(c0, ed);
    Point p1 = basis_.interpolate(c1, ed);
    total += (p1.asVector() - p0.asVector()).length();
  }
  return total;
}


template <class Basis>
void
CurveMesh<Basis>::get_center(Point &result,
                             typename Edge::index_type idx) const
{
  ElemData cmcd(*this, idx);
  vector<double> coord(1,0.5L);
  result =  basis_.interpolate(coord, cmcd);
}


template <class Basis>
int
CurveMesh<Basis>::get_valence(typename Node::index_type idx) const
{
  int count = 0;
  for (unsigned int i = 0; i < edges_.size(); i++)
    if (edges_[i].first == idx || edges_[i].second == idx) count++;
  return count;
}


template <class Basis>
bool
CurveMesh<Basis>::locate(typename Node::index_type &idx, const Point &p) const
{
  typename Node::iterator ni, nie;
  begin(ni);
  end(nie);

  idx = *ni;

  if (ni == nie)
  {
    return false;
  }

  double closest = (p-nodes_[*ni]).length2();

  ++ni;
  for (; ni != nie; ++ni)
  {
    if ( (p-nodes_[*ni]).length2() < closest )
    {
      closest = (p-nodes_[*ni]).length2();
      idx = *ni;
    }
  }

  return true;
}


template <class Basis>
bool
CurveMesh<Basis>::locate(typename Edge::index_type &idx, const Point &p) const
{
  if (basis_.polynomial_order() > 1) return elem_locate(idx, *this, p);
  typename Edge::iterator ei;
  typename Edge::iterator eie;
  double cosa, closest=DBL_MAX;
  typename Node::array_type nra;
  double dist1, dist2, dist3, dist4;
  Point n1,n2,q;

  begin(ei);
  end(eie);

  if (ei==eie)
  {
    return false;
  }

  for (; ei != eie; ++ei)
  {
    get_nodes(nra,*ei);

    n1 = nodes_[nra[0]];
    n2 = nodes_[nra[1]];

    dist1 = (p-n1).length();
    dist2 = (p-n2).length();
    dist3 = (n1-n2).length();

    cosa = Dot(n1-p,n1-n2)/((n1-p).length()*dist3);

    q = n1 + (n1-n2) * (n1-n2)/dist3;

    dist4 = (p-q).length();

    if ( (cosa > 0) && (cosa < dist3) && (dist4 < closest) ) {
      closest = dist4;
      idx = *ei;
    } else if ( (cosa < 0) && (dist1 < closest) ) {
      closest = dist1;
      idx = *ei;
    } else if ( (cosa > dist3) && (dist2 < closest) ) {
      closest = dist2;
      idx = *ei;
    }
  }

  return true;
}


template <class Basis>
int
CurveMesh<Basis>::get_weights(const Point &p, typename Node::array_type &l,
                              double *w)
{
  typename Edge::index_type idx;
  if (locate(idx, p))
  {
    get_nodes(l,idx);
    vector<double> coords(1);
    if (get_coords(coords, p, idx))
    {
      basis_.get_weights(coords, w);
      return basis_.dofs();
    }
  }
  return 0;
}


template <class Basis>
int
CurveMesh<Basis>::get_weights(const Point &p, typename Edge::array_type &l,
                              double *w)
{
  typename Edge::index_type idx;
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
void
CurveMesh<Basis>::get_random_point(Point &p,
                                   typename Elem::index_type ei,
                                   MusilRNG &rng) const
{
  const Point &p0 = nodes_[edges_[ei].first];
  const Point &p1 = nodes_[edges_[ei].second];

  p = p0 + (p1 - p0) * rng();
}


template <class Basis>
bool
CurveMesh<Basis>::synchronize(unsigned int tosync)
{
  synchronize_lock_.lock();

  if (tosync & NODE_NEIGHBORS_E && !(synchronized_ & NODE_NEIGHBORS_E))
    compute_node_neighbors();

  synchronize_lock_.unlock();
  return true;
}


template <class Basis>
void
CurveMesh<Basis>::compute_node_neighbors()
{
  node_neighbors_.clear();
  node_neighbors_.resize(nodes_.size());
  unsigned int i, num_elems = edges_.size();
  for (i = 0; i < num_elems; i++)
  {
    node_neighbors_[edges_[i].first].push_back(i);
    node_neighbors_[edges_[i].second].push_back(i);
  }
  synchronized_ |= NODE_NEIGHBORS_E;
}


#define CURVE_MESH_VERSION 2

template <class Basis>
void
CurveMesh<Basis>::io(Piostream& stream)
{
  int version = stream.begin_class(type_name(-1), CURVE_MESH_VERSION);

  Mesh::io(stream);

  // IO data members, in order
  Pio(stream, nodes_);
  Pio(stream, edges_);
  if (version >= 2) {
    basis_.io(stream);
  }
  stream.end_class();
}


template <class Basis>
const string
CurveMesh<Basis>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;
  }
  else if (n == 0)
  {
    static const string nm("CurveMesh");
    return nm;
  }
  else
  {
    return find_type_name((Basis *)0);
  }
}


template <class Basis>
void
CurveMesh<Basis>::begin(typename CurveMesh<Basis>::Node::iterator &itr) const
{
  itr = 0;
}


template <class Basis>
void
CurveMesh<Basis>::end(typename CurveMesh<Basis>::Node::iterator &itr) const
{
  itr = static_cast<typename Node::iterator>(nodes_.size());
}


template <class Basis>
void
CurveMesh<Basis>::begin(typename CurveMesh<Basis>::Edge::iterator &itr) const
{
  itr = 0;
}


template <class Basis>
void
CurveMesh<Basis>::end(typename CurveMesh<Basis>::Edge::iterator &itr) const
{
  itr = (unsigned)edges_.size();
}


template <class Basis>
void
CurveMesh<Basis>::begin(typename CurveMesh<Basis>::Face::iterator &itr) const
{
  itr = 0;
}


template <class Basis>
void
CurveMesh<Basis>::end(typename CurveMesh<Basis>::Face::iterator &itr) const
{
  itr = 0;
}


template <class Basis>
void
CurveMesh<Basis>::begin(typename CurveMesh<Basis>::Cell::iterator &itr) const
{
  itr = 0;
}


template <class Basis>
void
CurveMesh<Basis>::end(typename CurveMesh<Basis>::Cell::iterator &itr) const
{
  itr = 0;
}


template <class Basis>
void
CurveMesh<Basis>::size(typename CurveMesh<Basis>::Node::size_type &s) const
{
  s = (unsigned)nodes_.size();
}


template <class Basis>
void
CurveMesh<Basis>::size(typename CurveMesh<Basis>::Edge::size_type &s) const
{
  s = (unsigned)edges_.size();
}


template <class Basis>
void
CurveMesh<Basis>::size(typename CurveMesh<Basis>::Face::size_type &s) const
{
  s = 0;
}


template <class Basis>
void
CurveMesh<Basis>::size(typename CurveMesh<Basis>::Cell::size_type &s) const
{
  s = 0;
}


} // namespace SCIRun

#endif // SCI_project_CurveMesh_h





