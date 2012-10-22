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
 *  ScanlineMesh.h: Templated Mesh defined on a 3D Regular Grid
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   January 2001
 *
 *
 */


#ifndef SCI_project_ScanlineMesh_h
#define SCI_project_ScanlineMesh_h 1

#include <Core/Datatypes/Mesh.h>
#include <Core/Datatypes/FieldIterator.h>
#include <Core/Geometry/Transform.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/BBox.h>
#include <Core/Math/MusilRNG.h>
#include <Core/Containers/StackVector.h>

namespace SCIRun {

using std::string;

template <class Basis>
class ScanlineMesh : public Mesh
{
public:
  typedef Basis           basis_type;
  typedef unsigned int    under_type;

  //! Index and Iterator types required for Mesh Concept.
  struct Node {
    typedef NodeIndex<under_type>       index_type;
    typedef NodeIterator<under_type>    iterator;
    typedef NodeIndex<under_type>       size_type;
    typedef StackVector<index_type, 8>  array_type;
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

  friend class ElemData;

  class ElemData
  {
  public:
    ElemData(const ScanlineMesh<Basis>& msh,
             const typename Elem::index_type ind) :
      mesh_(msh),
      index_(ind)
    {}

    // the following designed to coordinate with ::get_nodes
    inline
    unsigned node0_index() const {
      return (index_);
    }
    inline
    unsigned node1_index() const {
      return (index_ + 1);
    }


    // the following designed to coordinate with ::get_edges
    inline
    unsigned edge0_index() const {
      return index_;
    }

    inline
    const Point node0() const {
      Point p(index_, 0.0, 0.0);
      return mesh_.transform_.project(p);
    }
    inline
    const Point node1() const {
      Point p(index_ + 1, 0.0, 0.0);
      return mesh_.transform_.project(p);
    }

  private:
    const ScanlineMesh<Basis>          &mesh_;
    const typename Elem::index_type     index_;
  };

  ScanlineMesh() : min_i_(0), ni_(0) {}
  ScanlineMesh(unsigned int nx, const Point &min, const Point &max);
  ScanlineMesh(ScanlineMesh* mh, unsigned int offset, unsigned int nx)
    : min_i_(offset), ni_(nx), transform_(mh->transform_) {}
  ScanlineMesh(const ScanlineMesh &copy)
    : min_i_(copy.get_min_i()), ni_(copy.get_ni()),
      transform_(copy.transform_) {}
  virtual ScanlineMesh *clone() { return new ScanlineMesh(*this); }
  virtual ~ScanlineMesh() {}

  //! get the mesh statistics
  unsigned get_min_i() const { return min_i_; }
  bool get_min(vector<unsigned int>&) const;
  unsigned get_ni() const { return ni_; }
  bool get_dim(vector<unsigned int>&) const;
  Vector diagonal() const;
  virtual BBox get_bounding_box() const;
  virtual void transform(const Transform &t);
  virtual void get_canonical_transform(Transform &t);

  //! set the mesh statistics
  void set_min_i(unsigned i) {min_i_ = i; }
  void set_min(vector<unsigned int> mins);
  void set_ni(unsigned i) { ni_ = i; }
  void set_dim(vector<unsigned int> dims);

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

  void to_index(typename Node::index_type &index, unsigned int i) const
  { index = i; }
  void to_index(typename Edge::index_type &index, unsigned int i) const
  { index = i; }
  void to_index(typename Face::index_type &index, unsigned int i) const
  { index = i; }
  void to_index(typename Cell::index_type &index, unsigned int i) const
  { index = i; }

  //! get the child elements of the given index
  void get_nodes(typename Node::array_type &, typename Edge::index_type) const;
  void get_nodes(typename Node::array_type &, typename Face::index_type) const
  {}
  void get_nodes(typename Node::array_type &, typename Cell::index_type) const
  {}
  void get_edges(typename Edge::array_type &, typename Edge::index_type) const
  {}
  void get_edges(typename Edge::array_type &, typename Face::index_type) const
  {}
  void get_edges(typename Edge::array_type &, typename Cell::index_type) const
  {}

  //Stub, used by ShowField.
  void get_faces(typename Face::array_type &, typename Elem::index_type) const
  {}

  //! get the parent element(s) of the given index
  void get_elems(typename Elem::array_type &result,
                 typename Node::index_type idx) const;
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

  //! Get the size of an elemnt (length, area, volume)
  double get_size(typename Node::index_type) const { return 0.0; }
  double get_size(typename Edge::index_type idx) const
  {
    typename Node::array_type ra;
    get_nodes(ra,idx);
    Point p0,p1;
    get_point(p0,ra[0]);
    get_point(p1,ra[1]);
    return (p0-p1).length();
  }
  double get_size(typename Face::index_type) const { return 0.0; };
  double get_size(typename Cell::index_type) const { return 0.0; };
  double get_length(typename Edge::index_type idx) const
  { return get_size(idx); }
  double get_area(typename Face::index_type idx) const
  { return get_size(idx); }
  double get_volume(typename Cell::index_type idx) const
  { return get_size(idx); }

  int get_valence(typename Node::index_type idx) const
  { return (idx == 0 || idx == ni_ - 1) ? 1 : 2; }
  int get_valence(typename Edge::index_type) const { return 0; }
  int get_valence(typename Face::index_type) const { return 0; }
  int get_valence(typename Cell::index_type) const { return 0; }

  //! get the center point (in object space) of an element
  void get_center(Point &, typename Node::index_type) const;
  void get_center(Point &, typename Edge::index_type) const;
  void get_center(Point &, typename Face::index_type) const {}
  void get_center(Point &, typename Cell::index_type) const {}

  bool locate(typename Node::index_type &, const Point &);
  bool locate(typename Edge::index_type &, const Point &);
  bool locate(typename Face::index_type &, const Point &) const { return false; }
  bool locate(typename Cell::index_type &, const Point &) const { return false; }

  int get_weights(const Point &p, typename Node::array_type &l, double *w);
  int get_weights(const Point &p, typename Edge::array_type &l, double *w);
  int get_weights(const Point & , typename Face::array_type & , double * )
  { ASSERTFAIL("ScanlineMesh::get_weights for faces isn't supported"); }
  int get_weights(const Point & , typename Cell::array_type & , double * )
  { ASSERTFAIL("ScanlineMesh::get_weights for cells isn't supported"); }

  void get_random_point(Point &p, typename Elem::index_type i, MusilRNG &rng) const;

  void get_point(Point &p, typename Node::index_type i) const { get_center(p, i); }
  void get_normal(Vector &, typename Node::index_type) const
  { ASSERTFAIL("This mesh type does not have node normals."); }
  void get_normal(Vector &, vector<double> &, typename Elem::index_type,
                  unsigned int)
  { ASSERTFAIL("This mesh type does not have element normals."); }

  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  static  const string type_name(int n = -1);
  virtual const TypeDescription *get_type_description() const;

  // Unsafe due to non-constness of unproject.
  Transform &get_transform() { return transform_; }
  Transform &set_transform(const Transform &trans)
  { transform_ = trans; return transform_; }

  virtual int dimensionality() const { return 1; }
  virtual int topology_geometry() const { return (STRUCTURED | REGULAR); }
  Basis& get_basis() { return basis_; }

 //! Generate the list of points that make up a sufficiently accurate
  //! piecewise linear approximation of an edge.
  void pwl_approx_edge(vector<vector<double> > &coords,
                       typename Elem::index_type ci,
                       unsigned,
                       unsigned div_per_unit) const
  {
    // Needs to match unit_edges in Basis/QuadBilinearLgn.cc
    // compare get_nodes order to the basis order
    basis_.approx_edge(0, div_per_unit, coords);
  }

  //! Generate the list of points that make up a sufficiently accurate
  //! piecewise linear approximation of an face.
  void pwl_approx_face(vector<vector<vector<double> > > &coords,
                       typename Elem::index_type ci,
                       typename Face::index_type fi,
                       unsigned div_per_unit) const
  {
    ASSERTFAIL("ScanlineMesh has no faces");
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

  // get the Jacobian matrix
  void derivate(const vector<double> &coords,
                typename Elem::index_type idx,
                vector<Point> &J) const
  {
    ElemData ed(*this, idx);
    basis_.derivate(coords, ed, J);
  }

  static const TypeDescription* node_type_description();
  static const TypeDescription* edge_type_description();
  static const TypeDescription* face_type_description();
  static const TypeDescription* cell_type_description();
  static const TypeDescription* elem_type_description()
  { return edge_type_description(); }

  // returns a ScanlineMesh
  static Persistent *maker() { return new ScanlineMesh(); }

protected:

  //! the min typename Node::index_type ( incase this is a subLattice )
  unsigned int         min_i_;

  //! the typename Node::index_type space extents of a ScanlineMesh
  //! (min=min_typename Node::index_type, max=min+extents-1)
  unsigned int         ni_;

  //! the object space extents of a ScanlineMesh
  Transform            transform_;

  //! the basis fn
  Basis                basis_;
};


template <class Basis>
PersistentTypeID
ScanlineMesh<Basis>::type_id(type_name(-1), "Mesh",
                             ScanlineMesh<Basis>::maker);


template <class Basis>
ScanlineMesh<Basis>::ScanlineMesh(unsigned int ni,
                                  const Point &min, const Point &max)
  : min_i_(0), ni_(ni)
{
  transform_.pre_scale(Vector(1.0 / (ni_ - 1.0), 1.0, 1.0));
  transform_.pre_scale(max - min);
  transform_.pre_translate(Vector(min));
  transform_.compute_imat();
}


template <class Basis>
BBox
ScanlineMesh<Basis>::get_bounding_box() const
{
  Point p0(0.0, 0.0, 0.0);
  Point p1(ni_ - 1, 0.0, 0.0);

  BBox result;
  result.extend(transform_.project(p0));
  result.extend(transform_.project(p1));
  return result;
}


template <class Basis>
Vector
ScanlineMesh<Basis>::diagonal() const
{
  return get_bounding_box().diagonal();
}


template <class Basis>
void
ScanlineMesh<Basis>::transform(const Transform &t)
{
  transform_.pre_trans(t);
}


template <class Basis>
void
ScanlineMesh<Basis>::get_canonical_transform(Transform &t)
{
  t = transform_;
  t.post_scale(Vector(ni_ - 1.0, 1.0, 1.0));
}


template <class Basis>
bool
ScanlineMesh<Basis>::get_min(vector<unsigned int> &array ) const
{
  array.resize(1);
  array.clear();

  array.push_back(min_i_);

  return true;
}


template <class Basis>
bool
ScanlineMesh<Basis>::get_dim(vector<unsigned int> &array) const
{
  array.resize(1);
  array.clear();

  array.push_back(ni_);

  return true;
}


template <class Basis>
void
ScanlineMesh<Basis>::set_min(vector<unsigned int> min)
{
  min_i_ = min[0];
}


template <class Basis>
void
ScanlineMesh<Basis>::set_dim(vector<unsigned int> dim)
{
  ni_ = dim[0];
}


template <class Basis>
void
ScanlineMesh<Basis>::get_nodes(typename Node::array_type &array,
                               typename Edge::index_type idx) const
{
  array.resize(2);
  array[0] = typename Node::index_type(idx);
  array[1] = typename Node::index_type(idx + 1);
}


template <class Basis>
void
ScanlineMesh<Basis>::get_elems(typename Edge::array_type &result,
                               typename Node::index_type index) const
{
  result.reserve(2);
  result.clear();
  if (index > 0)
  {
    result.push_back(typename Edge::index_type(index-1));
  }
  if (index < ni_-1)
  {
    result.push_back(typename Edge::index_type(index));
  }
}


template <class Basis>
void
ScanlineMesh<Basis>::get_center(Point &result,
                                typename Node::index_type idx) const
{
  Point p(idx, 0.0, 0.0);
  result = transform_.project(p);
}


template <class Basis>
void
ScanlineMesh<Basis>::get_center(Point &result,
                                typename Edge::index_type idx) const
{
  Point p(idx + 0.5, 0.0, 0.0);
  result = transform_.project(p);
}


// TODO: verify
template <class Basis>
bool
ScanlineMesh<Basis>::locate(typename Edge::index_type &elem, const Point &p)
{
  if (basis_.polynomial_order() > 1) return elem_locate(elem, *this, p);
  const Point r = transform_.unproject(p);
  elem = (unsigned int)(r.x());

  if (elem >= (ni_ - 1))
  {
    return false;
  }
  else
  {
    return true;
  }
}


// TODO: verify
template <class Basis>
bool
ScanlineMesh<Basis>::locate(typename Node::index_type &node, const Point &p)
{
  const Point r = transform_.unproject(p);
  node = (unsigned int)(r.x() + 0.5);

  if (node >= ni_)
  {
    return false;
  }
  else
  {
    return true;
  }
}


template <class Basis>
int
ScanlineMesh<Basis>::get_weights(const Point &p, typename Node::array_type &l,
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
ScanlineMesh<Basis>::get_weights(const Point &p, typename Edge::array_type &l,
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
ScanlineMesh<Basis>::get_random_point(Point &p,
                                      typename Elem::index_type ei,
                                      MusilRNG &rng) const
{
  Point p0, p1;
  get_center(p0, typename Node::index_type(ei));
  get_center(p1, typename Node::index_type(under_type(ei)+1));

  p = p0 + (p1 - p0) * rng();
}


#define SCANLINEMESH_VERSION 3

template <class Basis>
void
ScanlineMesh<Basis>::io(Piostream& stream)
{
  int version = stream.begin_class(type_name(-1), SCANLINEMESH_VERSION);

  Mesh::io(stream);

  // IO data members, in order
  Pio(stream, ni_);
  if (version < 2 && stream.reading() )
  {
    Pio_old(stream, transform_);
  }
  else
  {
    Pio(stream, transform_);
  }
  if (version >= 3)
  {
    basis_.io(stream);
  }
  stream.end_class();
}


template <class Basis>
const string
ScanlineMesh<Basis>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;
  }
  else if (n == 0)
  {
    static const string nm("ScanLineMesh");
    return nm;
  }
  else
  {
    return find_type_name((Basis *)0);
  }
}


template <class Basis>
void
ScanlineMesh<Basis>::begin(typename ScanlineMesh::Node::iterator &itr) const
{
  itr = typename Node::iterator(min_i_);
}


template <class Basis>
void
ScanlineMesh<Basis>::end(typename ScanlineMesh::Node::iterator &itr) const
{
  itr = typename Node::iterator(min_i_ + ni_);
}


template <class Basis>
void
ScanlineMesh<Basis>::size(typename ScanlineMesh::Node::size_type &s) const
{
  s = typename Node::size_type(ni_);
}


template <class Basis>
void
ScanlineMesh<Basis>::begin(typename ScanlineMesh::Edge::iterator &itr) const
{
  itr = typename Edge::iterator(min_i_);
}


template <class Basis>
void
ScanlineMesh<Basis>::end(typename ScanlineMesh::Edge::iterator &itr) const
{
  itr = typename Edge::iterator(min_i_+ni_-1);
}


template <class Basis>
void
ScanlineMesh<Basis>::size(typename ScanlineMesh::Edge::size_type &s) const
{
  s = typename Edge::size_type(ni_ - 1);
}


template <class Basis>
void
ScanlineMesh<Basis>::begin(typename ScanlineMesh::Face::iterator &itr) const
{
  itr = typename Face::iterator(0);
}


template <class Basis>
void
ScanlineMesh<Basis>::end(typename ScanlineMesh::Face::iterator &itr) const
{
  itr = typename Face::iterator(0);
}


template <class Basis>
void
ScanlineMesh<Basis>::size(typename ScanlineMesh::Face::size_type &s) const
{
  s = typename Face::size_type(0);
}


template <class Basis>
void
ScanlineMesh<Basis>::begin(typename ScanlineMesh::Cell::iterator &itr) const
{
  itr = typename Cell::iterator(0);
}


template <class Basis>
void
ScanlineMesh<Basis>::end(typename ScanlineMesh::Cell::iterator &itr) const
{
  itr = typename Cell::iterator(0);
}


template <class Basis>
void
ScanlineMesh<Basis>::size(typename ScanlineMesh::Cell::size_type &s) const
{
  s = typename Cell::size_type(0);
}


template <class Basis>
const TypeDescription*
get_type_description(ScanlineMesh<Basis> *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    const TypeDescription *sub = SCIRun::get_type_description((Basis*)0);
    TypeDescription::td_vec *subs = scinew TypeDescription::td_vec(1);
    (*subs)[0] = sub;
    td = scinew TypeDescription("ScanlineMesh", subs,
                                string(__FILE__),
                                "SCIRun",
                                TypeDescription::MESH_E);
  }
  return td;
}


template <class Basis>
const TypeDescription*
ScanlineMesh<Basis>::get_type_description() const
{
  return SCIRun::get_type_description((ScanlineMesh *)0);
}


template <class Basis>
const TypeDescription*
ScanlineMesh<Basis>::node_type_description()
{
  static TypeDescription *td = 0;
  if (!td)
  {
    const TypeDescription *me =
      SCIRun::get_type_description((ScanlineMesh<Basis> *)0);
    td = scinew TypeDescription(me->get_name() + "::Node",
                                string(__FILE__),
                                "SCIRun",
                                TypeDescription::MESH_E);
  }
  return td;
}


template <class Basis>
const TypeDescription*
ScanlineMesh<Basis>::edge_type_description()
{
  static TypeDescription *td = 0;
  if (!td)
  {
    const TypeDescription *me =
      SCIRun::get_type_description((ScanlineMesh<Basis> *)0);
    td = scinew TypeDescription(me->get_name() + "::Edge",
                                string(__FILE__),
                                "SCIRun",
                                TypeDescription::MESH_E);
  }
  return td;
}


template <class Basis>
const TypeDescription*
ScanlineMesh<Basis>::face_type_description()
{
  static TypeDescription *td = 0;
  if (!td)
  {
    const TypeDescription *me =
      SCIRun::get_type_description((ScanlineMesh<Basis> *)0);
    td = scinew TypeDescription(me->get_name() + "::Face",
                                string(__FILE__),
                                "SCIRun",
                                TypeDescription::MESH_E);
  }
  return td;
}


template <class Basis>
const TypeDescription*
ScanlineMesh<Basis>::cell_type_description()
{
  static TypeDescription *td = 0;
  if (!td)
  {
    const TypeDescription *me =
      SCIRun::get_type_description((ScanlineMesh<Basis> *)0);
    td = scinew TypeDescription(me->get_name() + "::Cell",
                                string(__FILE__),
                                "SCIRun",
                                TypeDescription::MESH_E);
  }
  return td;
}

} // namespace SCIRun

#endif // SCI_project_ScanlineMesh_h
