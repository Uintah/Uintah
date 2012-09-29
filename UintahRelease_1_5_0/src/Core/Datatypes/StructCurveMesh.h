/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

/*
 *  StructCurveMesh.h: Templated Mesh defined on a 1D Structured Grid
 *
 *  Written by:
 *   Allen R. Sanderson
 *   Department of Computer Science
 *   University of Utah
 *   November 2002
 *
 *
 */

/*
  A structured curve is a dataset with regular topology but with
  irregular geometry.  The line defined may have any shape but can not
  be overlapping or self-intersecting.

  The topology of structured curve is represented using a 1D vector with
  the points being stored in an index based array. The ordering of the curve is
  implicity defined based based upon its indexing.

  For more information on datatypes see Schroeder, Martin, and Lorensen,
  "The Visualization Toolkit", Prentice Hall, 1998.
*/


#ifndef SCI_project_StructCurveMesh_h
#define SCI_project_StructCurveMesh_h 1

#include <Core/Datatypes/ScanlineMesh.h>
#include <Core/Containers/Array1.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/BBox.h>
#include <Core/Math/MusilRNG.h>
#include <vector>
#include <cfloat>

namespace SCIRun {

template <class Basis>
class StructCurveMesh : public ScanlineMesh<Basis>
{
public:
  StructCurveMesh() {}
  StructCurveMesh(unsigned int n);
  StructCurveMesh(const StructCurveMesh &copy);
  virtual StructCurveMesh *clone() { return new StructCurveMesh(*this); }
  virtual ~StructCurveMesh() {}

  //! get the mesh statistics
  double get_cord_length() const;
  virtual BBox get_bounding_box() const;
  virtual void transform(const Transform &t);

  void set_dim(vector<unsigned int> dims) {
    ScanlineMesh<Basis>::set_dim(dims);
    points_.resize(dims[0]);
  }

  bool get_dim(vector<unsigned int>&) const;

  virtual int topology_geometry() const { return (Mesh::STRUCTURED | Mesh::IRREGULAR); }

  //! get the child elements of the given index
  void get_nodes(typename ScanlineMesh<Basis>::Node::array_type &,
                 typename ScanlineMesh<Basis>::Edge::index_type) const;
  void get_nodes(typename ScanlineMesh<Basis>::Node::array_type &,
                 typename ScanlineMesh<Basis>::Face::index_type) const {}
  void get_nodes(typename ScanlineMesh<Basis>::Node::array_type &,
                 typename ScanlineMesh<Basis>::Cell::index_type) const {}
  void get_edges(typename ScanlineMesh<Basis>::Edge::array_type &,
                 typename ScanlineMesh<Basis>::Face::index_type) const {}
  void get_edges(typename ScanlineMesh<Basis>::Edge::array_type &,
                 typename ScanlineMesh<Basis>::Cell::index_type) const {}
  void get_edges(typename ScanlineMesh<Basis>::Edge::array_type &a,
                 typename ScanlineMesh<Basis>::Edge::index_type idx) const
  { a.push_back(idx);}

  //! Get the size of an elemnt (length, area, volume)
  double get_size(typename ScanlineMesh<Basis>::Node::index_type) const
  { return 0.0; }
  double get_size(typename ScanlineMesh<Basis>::Edge::index_type idx) const
  {
    typename ScanlineMesh<Basis>::Node::array_type arr;
    get_nodes(arr, idx);
    Point p0, p1;
    get_center(p0, arr[0]);
    get_center(p1, arr[1]);
    return (p1.asVector() - p0.asVector()).length();
  }

  double get_size(typename ScanlineMesh<Basis>::Face::index_type) const
  { return 0.0; }
  double get_size(typename ScanlineMesh<Basis>::Cell::index_type) const
  { return 0.0; }
  double get_length(typename ScanlineMesh<Basis>::Edge::index_type idx) const
  { return get_size(idx); }
  double get_area(typename ScanlineMesh<Basis>::Face::index_type idx) const
  { return get_size(idx); }
  double get_volume(typename ScanlineMesh<Basis>::Cell::index_type idx) const
  { return get_size(idx); }

  int get_valence(typename ScanlineMesh<Basis>::Node::index_type idx) const
  { return (idx == (unsigned int) 0 ||
            idx == (unsigned int) (points_.size()-1)) ? 1 : 2; }

  int get_valence(typename ScanlineMesh<Basis>::Edge::index_type) const
  { return 0; }
  int get_valence(typename ScanlineMesh<Basis>::Face::index_type) const
  { return 0; }
  int get_valence(typename ScanlineMesh<Basis>::Cell::index_type) const
  { return 0; }

  //! get the center point (in object space) of an element
  void get_center(Point &,
                  const typename ScanlineMesh<Basis>::Node::index_type&) const;
  void get_center(Point &,
                  const typename ScanlineMesh<Basis>::Edge::index_type&) const;
  void get_center(Point &,
                  const typename ScanlineMesh<Basis>::Face::index_type&) const
  {}
  void get_center(Point &,
                  const typename ScanlineMesh<Basis>::Cell::index_type&) const
  {}

  bool locate(typename ScanlineMesh<Basis>::Node::index_type &,
              const Point &) const;
  bool locate(typename ScanlineMesh<Basis>::Edge::index_type &,
              const Point &) const;
  bool locate(typename ScanlineMesh<Basis>::Face::index_type &,
              const Point &) const
  { return false; }
  bool locate(typename ScanlineMesh<Basis>::Cell::index_type &,
              const Point &) const
  { return false; }

  int get_weights(const Point &,
                  typename ScanlineMesh<Basis>::Node::array_type &, double *w);
  int get_weights(const Point &,
                  typename ScanlineMesh<Basis>::Edge::array_type &, double *w);
  int get_weights(const Point &,
                  typename ScanlineMesh<Basis>::Face::array_type &, double *)
  {ASSERTFAIL("StructCurveMesh::get_weights for faces isn't supported"); }
  int get_weights(const Point &,
                  typename ScanlineMesh<Basis>::Cell::array_type &, double *)
  {ASSERTFAIL("StructCurveMesh::get_weights for cells isn't supported"); }

  void get_point(Point &p, typename ScanlineMesh<Basis>::Node::index_type i) const
  { get_center(p,i); }
  void set_point(const Point &p, typename ScanlineMesh<Basis>::Node::index_type i)
  { points_[i] = p; }

  void get_random_point(Point &p,
                        typename ScanlineMesh<Basis>::Elem::index_type idx,
                        MusilRNG &rng) const;

  void get_normal(Vector &,
                  typename ScanlineMesh<Basis>::Node::index_type) const
  { ASSERTFAIL("This mesh type does not have node normals."); }

  void get_normal(Vector &, vector<double> &,
                  typename ScanlineMesh<Basis>::Elem::index_type,
                  unsigned int)
  { ASSERTFAIL("This mesh type does not have element normals."); }

  class ElemData
  {
  public:
    ElemData(const StructCurveMesh<Basis>& msh,
             const typename ScanlineMesh<Basis>::Elem::index_type ind) :
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
    const Point &node0() const {
      return mesh_.points_[(unsigned int)(index_)];
    }
    inline
    const Point &node1() const {
      return mesh_.points_[index_+1];
    }

  private:
    const StructCurveMesh<Basis>          &mesh_;
    const typename ScanlineMesh<Basis>::Elem::index_type     index_;
  };

  friend class ElemData;

 //! Generate the list of points that make up a sufficiently accurate
  //! piecewise linear approximation of an edge.
  void pwl_approx_edge(vector<vector<double> > &coords,
                       typename ScanlineMesh<Basis>::Elem::index_type ci,
                       unsigned,
                       unsigned div_per_unit) const
  {
    // Needs to match unit_edges in Basis/QuadBilinearLgn.cc
    // compare get_nodes order to the basis order
    this->basis_.approx_edge(0, div_per_unit, coords);
  }

  //! Generate the list of points that make up a sufficiently accurate
  //! piecewise linear approximation of an face.
  void pwl_approx_face(vector<vector<vector<double> > > &coords,
                       typename ScanlineMesh<Basis>::Elem::index_type ci,
                       typename ScanlineMesh<Basis>::Face::index_type fi,
                       unsigned div_per_unit) const
  {
    ASSERTFAIL("ScanlineMesh has no faces");
  }

  bool get_coords(vector<double> &coords,
                  const Point &p,
                  typename ScanlineMesh<Basis>::Elem::index_type idx) const
  {
    ElemData ed(*this, idx);
    return this->basis_.get_coords(coords, p, ed);
  }

  void interpolate(Point &pt, const vector<double> &coords,
                   typename ScanlineMesh<Basis>::Elem::index_type idx) const
  {
    ElemData ed(*this, idx);
    pt = this->basis_.interpolate(coords, ed);
  }

  // get the Jacobian matrix
  void derivate(const vector<double> &coords,
                typename ScanlineMesh<Basis>::Elem::index_type idx,
                vector<Point> &J) const
  {
    ElemData ed(*this, idx);
    this->basis_.derivate(coords, ed, J);
  }



  virtual bool is_editable() const { return false; }

  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  static  const std::string type_name(int n = -1);
  virtual const TypeDescription *get_type_description() const;
  static const TypeDescription* node_type_description();
  static const TypeDescription* edge_type_description();
  static const TypeDescription* face_type_description();
  static const TypeDescription* cell_type_description();
  static const TypeDescription* elem_type_description()
  { return edge_type_description(); }

  // returns a StructCurveMesh
  static Persistent *maker() { return new StructCurveMesh<Basis>(); }

private:

  //! the points
  Array1<Point> points_;
}; // end class StructCurveMesh


template <class Basis>
PersistentTypeID
StructCurveMesh<Basis>::type_id(StructCurveMesh<Basis>::type_name(-1),
                                "Mesh", maker);

template <class Basis>
StructCurveMesh<Basis>::StructCurveMesh(unsigned int n)
  : ScanlineMesh<Basis>(n, Point(0.0, 0.0, 0.0), Point(1.0, 1.0, 1.0)),
    points_(n)
{
}


template <class Basis>
StructCurveMesh<Basis>::StructCurveMesh(const StructCurveMesh &copy)
  : ScanlineMesh<Basis>(copy),
    points_(copy.points_)
{
}


template <class Basis>
bool
StructCurveMesh<Basis>::get_dim(vector<unsigned int> &array) const
{
  array.resize(1);
  array.clear();

  array.push_back(this->ni_);

  return true;
}


template <class Basis>
BBox
StructCurveMesh<Basis>::get_bounding_box() const
{
  BBox result;

  typename ScanlineMesh<Basis>::Node::iterator i, ie;
  begin(i);
  end(ie);

  while (i != ie) {
    result.extend(points_[*i]);
    ++i;
  }

  return result;
}


template <class Basis>
void
StructCurveMesh<Basis>::transform(const Transform &t)
{
  typename ScanlineMesh<Basis>::Node::iterator i, ie;
  begin(i);
  end(ie);

  while (i != ie) {
    points_[*i] = t.project(points_[*i]);

    ++i;
  }
}


template <class Basis>
double
StructCurveMesh<Basis>::get_cord_length() const
{
  double result = 0.0;

  typename ScanlineMesh<Basis>::Node::iterator i, i1, ie;
  begin(i);
  begin(i1);
  end(ie);

  while (i1 != ie)
  {
    ++i1;
    result += (points_[*i] - points_[*i1]).length();
    ++i;
  }

  return result;
}


template <class Basis>
void
StructCurveMesh<Basis>::get_nodes(typename ScanlineMesh<Basis>::Node::array_type &array, typename ScanlineMesh<Basis>::Edge::index_type idx) const
{
  array.resize(2);
  array[0] = typename ScanlineMesh<Basis>::Node::index_type(idx);
  array[1] = typename ScanlineMesh<Basis>::Node::index_type(idx + 1);
}


template <class Basis>
void
StructCurveMesh<Basis>::get_center(Point &result, const typename ScanlineMesh<Basis>::Node::index_type &idx) const
{
  result = points_[idx];
}


template <class Basis>
void
StructCurveMesh<Basis>::get_center(Point &result, const typename ScanlineMesh<Basis>::Edge::index_type &idx) const
{
  const Point &p0 = points_[typename ScanlineMesh<Basis>::Node::index_type(idx)];
  const Point &p1 = points_[typename ScanlineMesh<Basis>::Node::index_type(idx+1)];

  result = Point(p0+p1)/2.0;
}


template <class Basis>
bool
StructCurveMesh<Basis>::locate(typename ScanlineMesh<Basis>::Node::index_type &idx, const Point &p) const
{
  typename ScanlineMesh<Basis>::Node::iterator ni, nie;
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
bool
StructCurveMesh<Basis>::locate(typename ScanlineMesh<Basis>::Edge::index_type &idx, const Point &p) const
{
  if (this->basis_.polynomial_order() > 1) return elem_locate(idx, *this, p);
  typename ScanlineMesh<Basis>::Edge::iterator ei;
  typename ScanlineMesh<Basis>::Edge::iterator eie;
  double cosa, closest=DBL_MAX;
  typename ScanlineMesh<Basis>::Node::array_type nra;
  double dist1, dist2, dist3, dist4;
  Point n1,n2,q;

  begin(ei);
  end(eie);

  if (ei==eie)
    return false;

  for (; ei != eie; ++ei)
  {
    get_nodes(nra,*ei);

    n1 = points_[nra[0]];
    n2 = points_[nra[1]];

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
StructCurveMesh<Basis>::get_weights(const Point &p,
                            typename ScanlineMesh<Basis>::Node::array_type &l,
                                    double *w)
{
  typename ScanlineMesh<Basis>::Edge::index_type idx;
  if (locate(idx, p))
  {
    get_nodes(l,idx);
    vector<double> coords(1);
    if (get_coords(coords, p, idx))
    {
      this->basis_.get_weights(coords, w);
      return this->basis_.dofs();
    }
  }
  return 0;
}


template <class Basis>
int
StructCurveMesh<Basis>::get_weights(const Point &p,
                            typename ScanlineMesh<Basis>::Edge::array_type &l,
                                    double *w)
{
  typename ScanlineMesh<Basis>::Edge::index_type idx;
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
StructCurveMesh<Basis>::get_random_point(Point &p,
                     typename ScanlineMesh<Basis>::Elem::index_type i,
                                         MusilRNG &rng) const
{
  const Point &p0 =points_[typename ScanlineMesh<Basis>::Node::index_type(i)];
  const Point &p1=points_[typename ScanlineMesh<Basis>::Node::index_type(i+1)];

  p = p0 + (p1 - p0) * rng();
}


#define STRUCT_CURVE_MESH_VERSION 1

template <class Basis>
void
StructCurveMesh<Basis>::io(Piostream& stream)
{
  stream.begin_class(type_name(-1), STRUCT_CURVE_MESH_VERSION);
  ScanlineMesh<Basis>::io(stream);

  // IO data members, in order
  Pio(stream, points_);

  stream.end_class();
}


template <class Basis>
const std::string
StructCurveMesh<Basis>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const std::string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;
  }
  else if (n == 0)
  {
    static const std::string nm("StructCurveMesh");
    return nm;
  }
  else
  {
    return find_type_name((Basis *)0);
  }
}


template <class Basis>
const TypeDescription*
get_type_description(StructCurveMesh<Basis> *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    const TypeDescription *sub = SCIRun::get_type_description((Basis*)0);
    TypeDescription::td_vec *subs = scinew TypeDescription::td_vec(1);
    (*subs)[0] = sub;
    td = scinew TypeDescription("StructCurveMesh", subs,
                                std::string(__FILE__),
                                "SCIRun",
                                TypeDescription::MESH_E);
  }
  return td;
}


template <class Basis>
const TypeDescription*
StructCurveMesh<Basis>::get_type_description() const
{
  return SCIRun::get_type_description((StructCurveMesh<Basis> *)0);
}


template <class Basis>
const TypeDescription*
StructCurveMesh<Basis>::node_type_description()
{
  static TypeDescription *td = 0;
  if (!td)
  {
    const TypeDescription *me =
      SCIRun::get_type_description((StructCurveMesh<Basis> *)0);
    td = scinew TypeDescription(me->get_name() + "::Node",
                                std::string(__FILE__),
                                "SCIRun",
                                TypeDescription::MESH_E);
  }
  return td;
}


template <class Basis>
const TypeDescription*
StructCurveMesh<Basis>::edge_type_description()
{
  static TypeDescription *td = 0;
  if (!td)
  {
    const TypeDescription *me =
      SCIRun::get_type_description((StructCurveMesh<Basis> *)0);
    td = scinew TypeDescription(me->get_name() + "::Edge",
                                std::string(__FILE__),
                                "SCIRun",
                                TypeDescription::MESH_E);
  }
  return td;
}


template <class Basis>
const TypeDescription*
StructCurveMesh<Basis>::face_type_description()
{
  static TypeDescription *td = 0;
  if (!td)
  {
    const TypeDescription *me =
      SCIRun::get_type_description((StructCurveMesh<Basis> *)0);
    td = scinew TypeDescription(me->get_name() + "::Face",
                                std::string(__FILE__),
                                "SCIRun",
                                TypeDescription::MESH_E);
  }
  return td;
}


template <class Basis>
const TypeDescription*
StructCurveMesh<Basis>::cell_type_description()
{
  static TypeDescription *td = 0;
  if (!td)
  {
    const TypeDescription *me =
      SCIRun::get_type_description((StructCurveMesh<Basis> *)0);
    td = scinew TypeDescription(me->get_name() + "::Cell",
                                std::string(__FILE__),
                                "SCIRun",
                                TypeDescription::MESH_E);
  }
  return td;
}

} // namespace SCIRun

#endif // SCI_project_StructCurveMesh_h
