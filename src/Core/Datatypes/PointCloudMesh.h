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
 *  PointCloudMesh.h: countour mesh
 *
 *  Written by:
 *   Chris Moulding
 *   Department of Computer Science
 *   University of Utah
 *   January 2001
 *
 *
 */

#ifndef SCI_project_PointCloudMesh_h
#define SCI_project_PointCloudMesh_h 1

#include <Core/Geometry/Point.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Datatypes/Mesh.h>
#include <Core/Datatypes/FieldIterator.h>
#include <Core/Containers/StackVector.h>
#include <Core/Geometry/Transform.h>
#include <Core/Geometry/BBox.h>
#include <Core/Math/MusilRNG.h>
#include <string>
#include <vector>

namespace SCIRun {

using std::string;
using std::vector;


template <class Basis>
class PointCloudMesh : public Mesh
{
public:
  typedef LockingHandle<PointCloudMesh<Basis> > handle_type;
  typedef Basis                                 basis_type;
  typedef unsigned int                          under_type;

  //! Index and Iterator types required for Mesh Concept.
  struct Node {
    typedef NodeIndex<under_type>       index_type;
    typedef NodeIterator<under_type>    iterator;
    typedef NodeIndex<under_type>       size_type;
    typedef StackVector<index_type, 1>  array_type;
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
  // No DElem for PointCloudMesh

  friend class ElemData;

  class ElemData
  {
  public:
    ElemData(const PointCloudMesh<Basis>& msh,
             const typename Elem::index_type ind) :
      mesh_(msh),
      index_(ind)
    {}

    // the following designed to coordinate with ::get_nodes
    inline
    unsigned node0_index() const {
      return index_;
    }

    inline
    const Point &node0() const {
      return mesh_.points_[index_];
    }

  private:
    const PointCloudMesh<Basis>          &mesh_;
    const typename Elem::index_type       index_;
  };

  PointCloudMesh() {}
  PointCloudMesh(const PointCloudMesh &copy)
    : points_(copy.points_) {}
  virtual PointCloudMesh *clone() { return new PointCloudMesh(*this); }
  virtual ~PointCloudMesh() {}

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

  virtual BBox get_bounding_box() const;
  virtual void transform(const Transform &t);

  //! set the mesh statistics
  void resize_nodes(typename Node::size_type n) { points_.resize(n); }

  // This is actually get_nodes(typename Node::array_type &, Elem::index_type)
  // for compilation purposes.  IE It is redundant unless we are
  // templated by Elem type and we don't know that Elem is Node.
  // This is needed in ClipField, for example.
  void get_nodes(typename Node::array_type &a, typename Node::index_type i) const
    { a.resize(1); a[0] = i; }

  //! Get the children elemsnts of the given index
  void get_nodes(typename Node::array_type &, typename Edge::index_type) const {}
  void get_nodes(typename Node::array_type &, typename Face::index_type) const {}
  void get_nodes(typename Node::array_type &, typename Cell::index_type) const {}

  // Stubs, used by ShowField.
  void get_edges(typename Edge::array_type &, typename Elem::index_type) const {}
  void get_faces(typename Face::array_type &, typename Elem::index_type) const {}

  //! get the parent element(s) of the given index
  void get_elems(typename Elem::array_type &result,
                 typename Node::index_type idx) const
  { result.clear(); result.push_back(idx); }
  void get_elems(typename Elem::array_type &result,
                 typename Edge::index_type idx) const {}
  void get_elems(typename Elem::array_type &result,
                 typename Face::index_type idx) const {}


  //! get the center point (in object space) of an element
  void get_center(Point &p, typename Node::index_type i) const { p = points_[i]; }
  void get_center(Point &, typename Edge::index_type) const {}
  void get_center(Point &, typename Face::index_type) const {}
  void get_center(Point &, typename Cell::index_type) const {}

  //! Get the size of an elemnt (length, area, volume)
  double get_size(typename Node::index_type) const { return 0.0; }
  double get_size(typename Edge::index_type) const { return 0.0; }
  double get_size(typename Face::index_type) const { return 0.0; }
  double get_size(typename Cell::index_type) const { return 0.0; }
  double get_length(typename Edge::index_type idx) const
  { return get_size(idx); }
  double get_area(typename Face::index_type idx) const
  { return get_size(idx); }
  double get_volume(typename Cell::index_type idx) const
  { return get_size(idx); }

  int get_valence(typename Node::index_type) const { return 0; }
  int get_valence(typename Edge::index_type) const { return 0; }
  int get_valence(typename Face::index_type) const { return 0; }
  int get_valence(typename Cell::index_type) const { return 0; }

  bool locate(typename Node::index_type &, const Point &) const;
  bool locate(typename Edge::index_type &, const Point &) const
  { return false; }
  bool locate(typename Face::index_type &, const Point &) const
  { return false; }
  bool locate(typename Cell::index_type &, const Point &) const
  { return false; }

  int get_weights(const Point &p, typename Node::array_type &l, double *w);
  int get_weights(const Point & , typename Edge::array_type & , double * )
  { ASSERTFAIL("PointCloudField::get_weights for edges isn't supported"); }
  int get_weights(const Point & , typename Face::array_type & , double * )
  { ASSERTFAIL("PointCloudField::get_weights for faces isn't supported"); }
  int get_weights(const Point & , typename Cell::array_type & , double * )
  { ASSERTFAIL("PointCloudField::get_weights for cells isn't supported"); }

  void get_random_point(Point &p,
                        const typename Elem::index_type &i,
                        MusilRNG &rng) const
  { get_center(p, i); }

  void get_point(Point &p, typename Node::index_type i) const
  { get_center(p,i); }
  void set_point(const Point &p, typename Node::index_type i)
  { points_[i] = p; }
  void get_normal(Vector &, typename Node::index_type) const
  { ASSERTFAIL("This mesh type does not have node normals."); }
  void get_normal(Vector &, vector<double> &, typename Elem::index_type,
                  unsigned int)
  { ASSERTFAIL("This mesh type does not have element normals."); }

  //! use these to build up a new PointCloudField mesh
  typename Node::index_type add_node(const Point &p) { return add_point(p); }
  typename Node::index_type add_point(const Point &p);
  typename Elem::index_type add_elem(typename Node::array_type a)
  {
    ASSERTMSG(a.size() == 1, "Tried to add non-point element.");
    return a[0];
  }
  void node_reserve(size_t s) { points_.reserve(s); }
  void elem_reserve(size_t s) { points_.reserve(s); }

  virtual bool is_editable() const { return true; }
  virtual int dimensionality() const { return 0; }
  virtual int topology_geometry() const { return (UNSTRUCTURED | IRREGULAR); }

  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  static  const string type_name(int n = -1);
  virtual const TypeDescription *get_type_description() const;

  Basis& get_basis() { return basis_; }

  void pwl_approx_edge(vector<vector<double> > &,
                       typename Elem::index_type,
                       unsigned,
                       unsigned) const
  {}

  void pwl_approx_face(vector<vector<vector<double> > > &,
                       typename Elem::index_type,
                       unsigned,
                       unsigned div_per_unit) const
  {}

  bool get_coords(vector<double> &coords,
                  const Point &p,
                  typename Elem::index_type idx) const
  {
    coords.resize(1);
    coords[0] = 0.0L;
    return true;
  }

  void interpolate(Point &pt, const vector<double> &coords,
                   typename Node::index_type idx) const
  {
    get_center(pt, idx);
  }

  // get the Jacobian matrix
  void derivate(const vector<double> &coords,
                typename Elem::index_type idx,
                vector<Point> &J) const
  {
    J.resize(1);
    J[0].x(0.0L);
    J[0].y(0.0L);
    J[0].z(0.0L);
  }


  static const TypeDescription* node_type_description();
  static const TypeDescription* edge_type_description();
  static const TypeDescription* face_type_description();
  static const TypeDescription* cell_type_description();
  static const TypeDescription* elem_type_description()
  { return node_type_description(); }

  // returns a PointCloudMesh
  static Persistent *maker() { return new PointCloudMesh(); }

private:
  //! the nodes
  vector<Point> points_;

  //! basis fns
  Basis         basis_;

};  // end class PointCloudMesh


template <class Basis>
PersistentTypeID
PointCloudMesh<Basis>::type_id(type_name(-1), "Mesh",
                               PointCloudMesh<Basis>::maker);


template <class Basis>
BBox
PointCloudMesh<Basis>::get_bounding_box() const
{
  BBox result;

  typename Node::iterator i, ie;
  begin(i);
  end(ie);
  for ( ; i != ie; ++i)
  {
    result.extend(points_[*i]);
  }

  return result;
}


template <class Basis>
void
PointCloudMesh<Basis>::transform(const Transform &t)
{
  vector<Point>::iterator itr = points_.begin();
  vector<Point>::iterator eitr = points_.end();
  while (itr != eitr)
  {
    *itr = t.project(*itr);
    ++itr;
  }
}


template <class Basis>
bool
PointCloudMesh<Basis>::locate(typename Node::index_type &idx, const Point &p) const
{
  typename Node::iterator ni, nie;
  begin(ni);
  end(nie);

  idx = *ni;

  if (ni == nie)
  {
    return false;
  }

  double closest = (p-points_[*ni]).length2();

  ++ni;
  for (; ni != nie; ++ni)
  {
    if ( (p-points_[*ni]).length2() < closest )
    {
      closest = (p-points_[*ni]).length2();
      idx = *ni;
    }
  }

  return true;
}


template <class Basis>
int
PointCloudMesh<Basis>::get_weights(const Point &p,
                                   typename Node::array_type &l,
                                   double *w)
{
  typename Node::index_type idx;
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
typename PointCloudMesh<Basis>::Node::index_type
PointCloudMesh<Basis>::add_point(const Point &p)
{
  points_.push_back(p);
  return points_.size() - 1;
}


#define PointCloudFieldMESH_VERSION 2

template <class Basis>
void
PointCloudMesh<Basis>::io(Piostream& stream)
{
  int version = stream.begin_class(type_name(-1), PointCloudFieldMESH_VERSION);

  Mesh::io(stream);

  // IO data members, in order
  Pio(stream,points_);

  if (version >= 2) {
    basis_.io(stream);
  }

  stream.end_class();
}


template <class Basis>
const string
PointCloudMesh<Basis>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;
  }
  else if (n == 0)
  {
    static const string nm("PointCloudMesh");
    return nm;
  }
  else
  {
    return find_type_name((Basis *)0);
  }
}


template <class Basis>
void
PointCloudMesh<Basis>::begin(typename PointCloudMesh::Node::iterator &itr) const
{
  itr = 0;
}


template <class Basis>
void
PointCloudMesh<Basis>::end(typename PointCloudMesh::Node::iterator &itr) const
{
  itr = (unsigned)points_.size();
}


template <class Basis>
void
PointCloudMesh<Basis>::size(typename PointCloudMesh::Node::size_type &s) const
{
  s = (unsigned)points_.size();
}


template <class Basis>
void
PointCloudMesh<Basis>::begin(typename PointCloudMesh::Edge::iterator &itr) const
{
  itr = 0;
}


template <class Basis>
void
PointCloudMesh<Basis>::end(typename PointCloudMesh::Edge::iterator &itr) const
{
  itr = 0;
}


template <class Basis>
void
PointCloudMesh<Basis>::size(typename PointCloudMesh::Edge::size_type &s) const
{
  s = 0;
}


template <class Basis>
void
PointCloudMesh<Basis>::begin(typename PointCloudMesh::Face::iterator &itr) const
{
  itr = 0;
}


template <class Basis>
void
PointCloudMesh<Basis>::end(typename PointCloudMesh::Face::iterator &itr) const
{
  itr = 0;
}


template <class Basis>
void
PointCloudMesh<Basis>::size(typename PointCloudMesh::Face::size_type &s) const
{
  s = 0;
}


template <class Basis>
void
PointCloudMesh<Basis>::begin(typename PointCloudMesh::Cell::iterator &itr) const
{
  itr = 0;
}


template <class Basis>
void
PointCloudMesh<Basis>::end(typename PointCloudMesh::Cell::iterator &itr) const
{
  itr = 0;
}


template <class Basis>
void
PointCloudMesh<Basis>::size(typename PointCloudMesh::Cell::size_type &s) const
{
  s = 0;
}


template <class Basis>
const TypeDescription*
get_type_description(PointCloudMesh<Basis> *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    const TypeDescription *sub = SCIRun::get_type_description((Basis*)0);
    TypeDescription::td_vec *subs = scinew TypeDescription::td_vec(1);
    (*subs)[0] = sub;
    td = scinew TypeDescription("PointCloudMesh", subs,
                                string(__FILE__),
                                "SCIRun",
                                TypeDescription::MESH_E);
  }
  return td;
}


template <class Basis>
const TypeDescription*
PointCloudMesh<Basis>::get_type_description() const
{
  return SCIRun::get_type_description((PointCloudMesh<Basis> *)0);
}


template <class Basis>
const TypeDescription*
PointCloudMesh<Basis>::node_type_description()
{
  static TypeDescription *td = 0;
  if (!td)
  {
    const TypeDescription *me =
      SCIRun::get_type_description((PointCloudMesh<Basis> *)0);
    td = scinew TypeDescription(me->get_name() + "::Node",
                                string(__FILE__),
                                "SCIRun",
                                TypeDescription::MESH_E);
  }
  return td;
}


template <class Basis>
const TypeDescription*
PointCloudMesh<Basis>::edge_type_description()
{
  static TypeDescription *td = 0;
  if (!td)
  {
    const TypeDescription *me =
      SCIRun::get_type_description((PointCloudMesh<Basis> *)0);
    td = scinew TypeDescription(me->get_name() + "::Edge",
                                string(__FILE__),
                                "SCIRun",
                                TypeDescription::MESH_E);
  }
  return td;
}


template <class Basis>
const TypeDescription*
PointCloudMesh<Basis>::face_type_description()
{
  static TypeDescription *td = 0;
  if (!td)
  {
    const TypeDescription *me =
      SCIRun::get_type_description((PointCloudMesh<Basis> *)0);
    td = scinew TypeDescription(me->get_name() + "::Face",
                                string(__FILE__),
                                "SCIRun",
                                TypeDescription::MESH_E);
  }
  return td;
}


template <class Basis>
const TypeDescription*
PointCloudMesh<Basis>::cell_type_description()
{
  static TypeDescription *td = 0;
  if (!td)
  {
    const TypeDescription *me =
      SCIRun::get_type_description((PointCloudMesh<Basis> *)0);
    td = scinew TypeDescription(me->get_name() + "::Cell",
                                string(__FILE__),
                                "SCIRun",
                                TypeDescription::MESH_E);
  }
  return td;
}


} // namespace SCIRun

#endif // SCI_project_PointCloudMesh_h
