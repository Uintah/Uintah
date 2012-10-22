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
 *  StructHexVolMesh.h: Templated Mesh defined on an 3D Structured Grid
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   April 2002
 *
 *
 */

/*
  A structured grid is a dataset with regular topology but with irregular
  geometry.

  The grid may have any shape but can not be overlapping or self-intersecting.

  The topology of a structured grid is represented using 2D, or 3D vector with
  the points being stored in an index based array. The faces (quadrilaterals)
  and cells (Hexahedron) are implicitly define based based upon their indexing.

  Structured grids are typically used in finite difference analysis.

  For more information on datatypes see Schroeder, Martin, and Lorensen,
  "The Visualization Toolkit", Prentice Hall, 1998.
*/


#ifndef SCI_project_StructHexVolMesh_h
#define SCI_project_StructHexVolMesh_h 1

#include <Core/Datatypes/LatVolMesh.h>
#include <Core/Datatypes/SearchGrid.h>
#include <Core/Containers/Array3.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Plane.h>
#include <Core/Geometry/CompGeom.h>

#include <vector>

namespace SCIRun {

template <class Basis>
class StructHexVolMesh : public LatVolMesh<Basis>
{
public:
  typedef LockingHandle<StructHexVolMesh<Basis> > handle_type;

  StructHexVolMesh();
  StructHexVolMesh(unsigned int i, unsigned int j, unsigned int k);
  StructHexVolMesh(const StructHexVolMesh<Basis> &copy);
  virtual StructHexVolMesh *clone()
  { return new StructHexVolMesh<Basis>(*this); }
  virtual ~StructHexVolMesh() {}


  class ElemData
  {
  public:
    ElemData(const StructHexVolMesh<Basis>& msh,
             const typename LatVolMesh<Basis>::Cell::index_type ind) :
      mesh_(msh),
      index_(ind)
    {}

    // the following designed to coordinate with ::get_nodes
    inline
    unsigned node0_index() const {
      return (index_.i_ + mesh_.get_ni()*index_.j_ +
              mesh_.get_ni()*mesh_.get_nj()*index_.k_);
    }
    inline
    unsigned node1_index() const {
      return (index_.i_+ 1 + mesh_.get_ni()*index_.j_ +
              mesh_.get_ni()*mesh_.get_nj()*index_.k_);
    }
    inline
    unsigned node2_index() const {
      return (index_.i_ + 1 + mesh_.get_ni()*(index_.j_ + 1) +
              mesh_.get_ni()*mesh_.get_nj()*index_.k_);

    }
    inline
    unsigned node3_index() const {
      return (index_.i_ + mesh_.get_ni()*(index_.j_ + 1) +
              mesh_.get_ni()*mesh_.get_nj()*index_.k_);
    }
    inline
    unsigned node4_index() const {
      return (index_.i_ + mesh_.get_ni()*index_.j_ +
              mesh_.get_ni()*mesh_.get_nj()*(index_.k_ + 1));
    }
    inline
    unsigned node5_index() const {
      return (index_.i_ + 1 + mesh_.get_ni()*index_.j_ +
              mesh_.get_ni()*mesh_.get_nj()*(index_.k_ + 1));
    }

    inline
    unsigned node6_index() const {
      return (index_.i_ + 1 + mesh_.get_ni()*(index_.j_ + 1) +
              mesh_.get_ni()*mesh_.get_nj()*(index_.k_ + 1));
    }
    inline
    unsigned node7_index() const {
      return (index_.i_ + mesh_.get_ni()*(index_.j_ + 1) +
              mesh_.get_ni()*mesh_.get_nj()*(index_.k_ + 1));
    }

    inline
    const Point &node0() const {
      return mesh_.points_(index_.i_, index_.j_, index_.k_);
    }
    inline
    const Point &node1() const {
      return mesh_.points_(index_.i_+1, index_.j_, index_.k_);
    }
    inline
    const Point &node2() const {
      return mesh_.points_(index_.i_+1, index_.j_+1, index_.k_);
    }
    inline
    const Point &node3() const {
      return mesh_.points_(index_.i_, index_.j_+1, index_.k_);
    }
    inline
    const Point &node4() const {
      return mesh_.points_(index_.i_, index_.j_, index_.k_+1);
    }
    inline
    const Point &node5() const {
      return mesh_.points_(index_.i_+1, index_.j_, index_.k_+1);
    }
    inline
    const Point &node6() const {
      return mesh_.points_(index_.i_+1, index_.j_+1, index_.k_+1);
    }
    inline
    const Point &node7() const {
      return mesh_.points_(index_.i_, index_.j_+1, index_.k_+1);
    }

  private:
    const StructHexVolMesh<Basis>          &mesh_;
    const typename LatVolMesh<Basis>::Cell::index_type  index_;
  };

  friend class ElemData;

  //! get the mesh statistics
  virtual BBox get_bounding_box() const;
  virtual void transform(const Transform &t);

  bool get_dim(vector<unsigned int>&) const;

  void set_dim(vector<unsigned int> dims) {
    LatVolMesh<Basis>::set_dim(dims);
    points_.resize(dims[0], dims[1], dims[2]);
  }

  virtual int topology_geometry() const { return (Mesh::STRUCTURED | Mesh::IRREGULAR); }

  //! get the center point (in object space) of an element
  void get_center(Point &,
                  const typename LatVolMesh<Basis>::Node::index_type &) const;
  void get_center(Point &, typename LatVolMesh<Basis>::Edge::index_type) const;
  void get_center(Point &, typename LatVolMesh<Basis>::Face::index_type) const;
  void get_center(Point &,
                  const typename LatVolMesh<Basis>::Cell::index_type &) const;

  double get_size(const typename LatVolMesh<Basis>::Node::index_type &idx) const;
  double get_size(typename LatVolMesh<Basis>::Edge::index_type idx) const;
  double get_size(typename LatVolMesh<Basis>::Face::index_type idx) const;
  double get_size(
               const typename LatVolMesh<Basis>::Cell::index_type &idx) const;
  double get_length(typename LatVolMesh<Basis>::Edge::index_type idx) const
  { return get_size(idx); };
  double get_area(typename LatVolMesh<Basis>::Face::index_type idx) const
  { return get_size(idx); };
  double get_volume(const
                    typename LatVolMesh<Basis>::Cell::index_type &i) const
  { return get_size(i); };

  bool locate(typename LatVolMesh<Basis>::Node::index_type &, const Point &);
  bool locate(typename LatVolMesh<Basis>::Edge::index_type &,
              const Point &) const
  { return false; }
  bool locate(typename LatVolMesh<Basis>::Face::index_type &,
              const Point &) const
  { return false; }
  bool locate(typename LatVolMesh<Basis>::Cell::index_type &, const Point &);


  int get_weights(const Point &,
                  typename LatVolMesh<Basis>::Node::array_type &, double *);
  int get_weights(const Point &,
                  typename LatVolMesh<Basis>::Edge::array_type &, double *)
  { ASSERTFAIL("StructHexVolMesh::get_weights for edges isn't supported"); }
  int get_weights(const Point &,
                  typename LatVolMesh<Basis>::Face::array_type &, double *)
  { ASSERTFAIL("StructHexVolMesh::get_weights for faces isn't supported"); }
  int get_weights(const Point &,
                  typename LatVolMesh<Basis>::Cell::array_type &, double *);

  void get_point(Point &point,
              const typename LatVolMesh<Basis>::Node::index_type &index) const
  { get_center(point, index); }
  void set_point(const Point &point,
                 const typename LatVolMesh<Basis>::Node::index_type &index);


  void get_random_point(Point &p,
                        const typename LatVolMesh<Basis>::Elem::index_type & i,
                        MusilRNG &rng) const;

  bool get_coords(vector<double> &coords,
                  const Point &p,
                  typename LatVolMesh<Basis>::Elem::index_type idx) const
  {
    ElemData ed(*this, idx);
    return this->basis_.get_coords(coords, p, ed);
  }

  void interpolate(Point &pt, const vector<double> &coords,
                   typename LatVolMesh<Basis>::Elem::index_type idx) const
  {
    ElemData ed(*this, idx);
    pt = this->basis_.interpolate(coords, ed);
  }

  // get the Jacobian matrix
  void derivate(const vector<double> &coords,
                typename LatVolMesh<Basis>::Cell::index_type idx,
                vector<Point> &J) const
  {
    ElemData ed(*this, idx);
    this->basis_.derivate(coords, ed, J);
  }

  virtual bool synchronize(unsigned int);

  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  static  const std::string type_name(int n = -1);

  virtual const TypeDescription *get_type_description() const;
  static const TypeDescription* node_type_description();
  static const TypeDescription* edge_type_description();
  static const TypeDescription* face_type_description();
  static const TypeDescription* cell_type_description();
  static const TypeDescription* elem_type_description()
  { return elem_type_description(); }

  // returns a StructHexVolMesh
  static Persistent *maker() { return new StructHexVolMesh<Basis>(); }

private:
  void compute_grid();

  bool inside8_p(typename LatVolMesh<Basis>::Cell::index_type i,
                 const Point &p) const;
  double polygon_area(const typename LatVolMesh<Basis>::Node::array_type &,
                      const Vector) const;
  double pyramid_volume(const typename LatVolMesh<Basis>::Node::array_type &,
                        const Point &)const;
  void get_face_weights(double *w,
                   const typename LatVolMesh<Basis>::Node::array_type &nodes,
                        const Point &p, int i0, int i1, int i2, int i3);
  const Point &point(
                const typename LatVolMesh<Basis>::Node::index_type &i) const
  { return points_(i.i_, i.j_, i.k_); }


  Array3<Point> points_;

  LockingHandle<SearchGrid>           grid_;
  Mutex                               grid_lock_; // Bad traffic!
  typename LatVolMesh<Basis>::Cell::index_type           locate_cache_;

  unsigned int   synchronized_;
};


template <class Basis>
PersistentTypeID
StructHexVolMesh<Basis>::type_id(StructHexVolMesh<Basis>::type_name(-1),
                                 "Mesh", maker);


template <class Basis>
StructHexVolMesh<Basis>::StructHexVolMesh():
  grid_(0),
  grid_lock_("StructHexVolMesh grid lock"),
  locate_cache_(this, 0, 0, 0),
  synchronized_(Mesh::ALL_ELEMENTS_E)
{
}


template <class Basis>
StructHexVolMesh<Basis>::StructHexVolMesh(unsigned int i,
                                          unsigned int j,
                                          unsigned int k) :
  LatVolMesh<Basis>(i, j, k, Point(0.0, 0.0, 0.0), Point(1.0, 1.0, 1.0)),
  points_(i, j, k),
  grid_(0),
  grid_lock_("StructHexVolMesh grid lock"),
  locate_cache_(this, 0, 0, 0),
  synchronized_(Mesh::ALL_ELEMENTS_E)
{
}


template <class Basis>
StructHexVolMesh<Basis>::StructHexVolMesh(const StructHexVolMesh<Basis> &copy):
  LatVolMesh<Basis>(copy),
  grid_(0),
  grid_lock_("StructHexVolMesh grid lock"),
  locate_cache_(this, 0, 0, 0),
  synchronized_(Mesh::ALL_ELEMENTS_E)
{
  points_.copy( copy.points_ );

  StructHexVolMesh &lcopy = (StructHexVolMesh &)copy;
  lcopy.grid_lock_.lock();

  synchronized_ &= ~Mesh::LOCATE_E;
  if (copy.grid_.get_rep())
  {
    grid_ = scinew SearchGrid(*(copy.grid_.get_rep()));
  }
  synchronized_ |= copy.synchronized_ & Mesh::LOCATE_E;

  lcopy.grid_lock_.unlock();
}


template <class Basis>
bool
StructHexVolMesh<Basis>::get_dim(vector<unsigned int> &array) const
{
  array.resize(3);
  array.clear();

  array.push_back(this->ni_);
  array.push_back(this->nj_);
  array.push_back(this->nk_);

  return true;
}


template <class Basis>
BBox
StructHexVolMesh<Basis>::get_bounding_box() const
{
  BBox result;

  typename LatVolMesh<Basis>::Node::iterator ni, nie;
  begin(ni);
  end(nie);
  while (ni != nie)
  {
    Point p;
    get_center(p, *ni);
    result.extend(p);
    ++ni;
  }
  return result;
}


template <class Basis>
void
StructHexVolMesh<Basis>::transform(const Transform &t)
{
  typename LatVolMesh<Basis>::Node::iterator i, ie;
  begin(i);
  end(ie);

  while (i != ie)
  {
    points_((*i).i_,(*i).j_,(*i).k_) =
      t.project(points_((*i).i_,(*i).j_,(*i).k_));

    ++i;
  }

  grid_lock_.lock();
  if (grid_.get_rep()) { grid_->transform(t); }
  grid_lock_.unlock();

}


template <class Basis>
void
StructHexVolMesh<Basis>::get_center(Point &result, const typename LatVolMesh<Basis>::Node::index_type &idx) const
{
  result = points_(idx.i_, idx.j_, idx.k_);
}


template <class Basis>
void
StructHexVolMesh<Basis>::get_center(Point &result, typename LatVolMesh<Basis>::Edge::index_type idx) const
{
  typename LatVolMesh<Basis>::Node::array_type arr;
  get_nodes(arr, idx);
  Point p1;
  get_center(result, arr[0]);
  get_center(p1, arr[1]);

  result.asVector() += p1.asVector();
  result.asVector() *= 0.5;
}


template <class Basis>
void
StructHexVolMesh<Basis>::get_center(Point &result, typename LatVolMesh<Basis>::Face::index_type idx) const
{
  typename LatVolMesh<Basis>::Node::array_type nodes;
  get_nodes(nodes, idx);
  ASSERT(nodes.size() == 4);
  typename LatVolMesh<Basis>::Node::array_type::iterator nai = nodes.begin();
  get_point(result, *nai);
  ++nai;
  Point pp;
  while (nai != nodes.end())
  {
    get_point(pp, *nai);
    result.asVector() += pp.asVector();
    ++nai;
  }
  result.asVector() *= (1.0 / 4.0);
}


template <class Basis>
void
StructHexVolMesh<Basis>::get_center(Point &result, const typename LatVolMesh<Basis>::Cell::index_type &idx) const
{
  typename LatVolMesh<Basis>::Node::array_type nodes;
  get_nodes(nodes, idx);
  ASSERT(nodes.size() == 8);
  typename LatVolMesh<Basis>::Node::array_type::iterator nai = nodes.begin();
  get_point(result, *nai);
  ++nai;
  Point pp;
  while (nai != nodes.end())
  {
    get_point(pp, *nai);
    result.asVector() += pp.asVector();
    ++nai;
  }
  result.asVector() *= (1.0 / 8.0);
}


template <class Basis>
bool
StructHexVolMesh<Basis>::inside8_p(
                             typename LatVolMesh<Basis>::Cell::index_type idx,
                             const Point &p) const
{
  static const int table[6][3][3] =
  {{{0, 0, 0},
    {0, 1, 0},
    {0, 0, 1}},

   {{0, 0, 0},
    {1, 0, 0},
    {0, 1, 0}},

   {{0, 0, 0},
    {0, 0, 1},
    {1, 0, 0}},

   {{1, 1, 1},
    {1, 1, 0},
    {1, 0, 1}},

   {{1, 1, 1},
    {1, 0, 1},
    {0, 1, 1}},

   {{1, 1, 1},
    {0, 1, 1},
    {1, 1, 0}}};

  Point center;
  get_center(center, idx);

  for (int i = 0; i < 6; i++)
  {
    typename LatVolMesh<Basis>::Node::index_type n0(this,
                        idx.i_ + table[i][0][0],
                        idx.j_ + table[i][0][1],
                        idx.k_ + table[i][0][2]);
    typename LatVolMesh<Basis>::Node::index_type n1(this,
                        idx.i_ + table[i][1][0],
                        idx.j_ + table[i][1][1],
                        idx.k_ + table[i][1][2]);
    typename LatVolMesh<Basis>::Node::index_type n2(this,
                        idx.i_ + table[i][2][0],
                        idx.j_ + table[i][2][1],
                        idx.k_ + table[i][2][2]);

    Point p0, p1, p2;
    get_center(p0, n0);
    get_center(p1, n1);
    get_center(p2, n2);

    const Vector v0(p1 - p0), v1(p2 - p0);
    const Vector normal = Cross(v0, v1);
    const Vector off0(p - p0);
    const Vector off1(center - p0);

    double dotprod = Dot(off0, normal);

    // Account for round off - the point may be on the plane!!
    if( fabs( dotprod ) < this->MIN_ELEMENT_VAL )
      continue;

    // If orientated correctly the second dot product is not needed.
    // Only need to check to see if the sign is negative.
    if (dotprod * Dot(off1, normal) < 0.0)
      return false;
  }
  return true;
}


template <class Basis>
bool
StructHexVolMesh<Basis>::locate(
                            typename LatVolMesh<Basis>::Cell::index_type &cell,
                            const Point &p)
{
  if (this->basis_.polynomial_order() > 1) return elem_locate(cell, *this, p);
  // Check last cell found first.  Copy cache to cell first so that we
  // don't care about thread safeness, such that worst case on
  // context switch is that cache is not found.
  cell = locate_cache_;
  if (cell > typename LatVolMesh<Basis>::Cell::index_type(this, 0, 0, 0) &&
      cell < typename LatVolMesh<Basis>::Cell::index_type(this, this->ni_ -1,
                                                          this->nj_ - 1,
                                                          this->nk_ - 1) &&
      inside8_p(cell, p))
  {
      return true;
  }

  ASSERT(synchronized_ & Mesh::LOCATE_E);
  cell.mesh_ = this;

  unsigned int *iter, *end;
  if (grid_->lookup(&iter, &end, p)) {
    while (iter != end) {
      typename LatVolMesh<Basis>::Cell::index_type idx;
      to_index(idx, *iter);

      if( inside8_p(idx, p) ) {
        cell = idx;
        locate_cache_ = cell;
        return true;
      }
      ++iter;
    }
  }

  return false;
}


template <class Basis>
bool
StructHexVolMesh<Basis>::locate(typename LatVolMesh<Basis>::Node::index_type &node, const Point &p)
{
  node.mesh_ = this;
  typename LatVolMesh<Basis>::Cell::index_type ci;
  if (locate(ci, p)) { // first try the fast way.
    typename LatVolMesh<Basis>::Node::array_type nodes;
    get_nodes(nodes, ci);

    double dmin =
      (p - points_(nodes[0].i_, nodes[0].j_, nodes[0].k_)).length2();
    node = nodes[0];
    for (unsigned int i = 1; i < nodes.size(); i++)
    {
      const double d =
        (p - points_(nodes[i].i_, nodes[i].j_, nodes[i].k_)).length2();
      if (d < dmin)
      {
        dmin = d;
        node = nodes[i];
      }
    }
    return true;
  }
  else
  {  // do exhaustive search.
    typename LatVolMesh<Basis>::Node::iterator ni, nie;
    begin(ni);
    end(nie);
    if (ni == nie) { return false; }

    double min_dist = (p - points_((*ni).i_, (*ni).j_, (*ni).k_)).length2();
    node = *ni;
    ++ni;

    while (ni != nie)
    {
      const double dist =
        (p - points_((*ni).i_, (*ni).j_, (*ni).k_)).length2();
      if (dist < min_dist)
      {
        node = *ni;
        min_dist = dist;
      }
      ++ni;
    }
    return true;
  }
}


template <class Basis>
int
StructHexVolMesh<Basis>::get_weights(const Point &p,
                             typename LatVolMesh<Basis>::Node::array_type &l,
                                     double *w)
{
  typename LatVolMesh<Basis>::Cell::index_type idx;
  if (locate(idx, p))
  {
    get_nodes(l,idx);
    vector<double> coords(3);
    if (get_coords(coords, p, idx)) {
      this->basis_.get_weights(coords, w);
      return this->basis_.dofs();
    }
  }
  return 0;
}


template <class Basis>
int
StructHexVolMesh<Basis>::get_weights(const Point &p,
                             typename LatVolMesh<Basis>::Cell::array_type &l,
                                     double *w)
{
  typename LatVolMesh<Basis>::Cell::index_type idx;
  if (locate(idx, p))
  {
    l.resize(1);
    l[0] = idx;
    w[0] = 1.0;
    return 1;
  }
  return 0;
}


//===================================================================
// area3D_Polygon(): computes the area of a 3D planar polygon
//    Input:  int n = the number of vertices in the polygon
//            Point* V = an array of n+2 vertices in a plane
//                       with V[n]=V[0] and V[n+1]=V[1]
//            Point N = unit normal vector of the polygon's plane
//    Return: the (float) area of the polygon

// Copyright 2000, softSurfer (www.softsurfer.com)
// This code may be freely used and modified for any purpose
// providing that this copyright notice is included with it.
// SoftSurfer makes no warranty for this code, and cannot be held
// liable for any real or imagined damage resulting from its use.
// Users of this code must verify correctness for their application.

template <class Basis>
double
StructHexVolMesh<Basis>::polygon_area(const typename LatVolMesh<Basis>::Node::array_type &ni, const Vector N) const
{
  double area = 0;
  double an, ax, ay, az;  // abs value of normal and its coords
  int   coord;           // coord to ignore: 1=x, 2=y, 3=z
  unsigned int   i, j, k;         // loop indices
  const unsigned int n = ni.size();

  // select largest abs coordinate to ignore for projection
  ax = (N.x()>0 ? N.x() : -N.x());     // abs x-coord
  ay = (N.y()>0 ? N.y() : -N.y());     // abs y-coord
  az = (N.z()>0 ? N.z() : -N.z());     // abs z-coord

  coord = 3;                     // ignore z-coord
  if (ax > ay) {
    if (ax > az) coord = 1;    // ignore x-coord
  }
  else if (ay > az) coord = 2;   // ignore y-coord

  // compute area of the 2D projection
  for (i=1, j=2, k=0; i<=n; i++, j++, k++)
    switch (coord) {
    case 1:
      area += (point(ni[i%n]).y() *
               (point(ni[j%n]).z() - point(ni[k%n]).z()));
      continue;
    case 2:
      area += (point(ni[i%n]).x() *
               (point(ni[j%n]).z() - point(ni[k%n]).z()));
      continue;
    case 3:
      area += (point(ni[i%n]).x() *
               (point(ni[j%n]).y() - point(ni[k%n]).y()));
      continue;
    }

  // scale to get area before projection
  an = sqrt( ax*ax + ay*ay + az*az);  // length of normal vector
  switch (coord) {
  case 1:
    area *= (an / (2*ax));
    break;
  case 2:
    area *= (an / (2*ay));
    break;
  case 3:
    area *= (an / (2*az));
  }
  return area;
}


template <class Basis>
double
StructHexVolMesh<Basis>::pyramid_volume(const typename LatVolMesh<Basis>::Node::array_type &face, const Point &p) const
{
  Vector e1(point(face[1])-point(face[0]));
  Vector e2(point(face[1])-point(face[2]));
  if (Cross(e1,e2).length2()>0.0)
  {
    Plane plane(point(face[0]), point(face[1]), point(face[2]));
    //double dist = plane.eval_point(p);
    return fabs(plane.eval_point(p)*polygon_area(face,plane.normal())*0.25);
  }
  Vector e3(point(face[3])-point(face[2]));
  if (Cross(e2,e3).length2()>0.0) {
    Plane plane(point(face[1]), point(face[2]), point(face[3]));
    //double dist = plane.eval_point(p);
    return fabs(plane.eval_point(p)*polygon_area(face,plane.normal())*0.25);
  }
  return 0.0;
}


extern
double
tri_area(const Point &a, const Point &b, const Point &c);

template <class Basis>
void
StructHexVolMesh<Basis>::get_face_weights(double *w,
                    const typename LatVolMesh<Basis>::Node::array_type &nodes,
                                          const Point &p,
                                          int i0, int i1, int i2, int i3)
{
  for (unsigned int j = 0; j < 8; j++)
    w[j] = 0.0;

  const Point &p0 = point(nodes[i0]);
  const Point &p1 = point(nodes[i1]);
  const Point &p2 = point(nodes[i2]);
  const Point &p3 = point(nodes[i3]);

  const double a0 = tri_area(p, p0, p1);
  if (a0 < this->MIN_ELEMENT_VAL)
  {
    const Vector v0 = p0 - p1;
    const Vector v1 = p - p1;
    w[i0] = Dot(v0, v1) / Dot(v0, v0);
    if( !finite( w[i0] ) ) w[i0] = 0.5;
    w[i1] = 1.0 - w[i0];
    return;
  }
  const double a1 = tri_area(p, p1, p2);
  if (a1 < this->MIN_ELEMENT_VAL)
  {
    const Vector v0 = p1 - p2;
    const Vector v1 = p - p2;
    w[i1] = Dot(v0, v1) / Dot(v0, v0);
    if( !finite( w[i1] ) ) w[i1] = 0.5;
    w[i2] = 1.0 - w[i1];
    return;
  }
  const double a2 = tri_area(p, p2, p3);
  if (a2 < this->MIN_ELEMENT_VAL)
  {
    const Vector v0 = p2 - p3;
    const Vector v1 = p - p3;
    w[i2] = Dot(v0, v1) / Dot(v0, v0);
    if( !finite( w[i2] ) ) w[i2] = 0.5;
    w[i3] = 1.0 - w[i2];
    return;
  }
  const double a3 = tri_area(p, p3, p0);
  if (a3 < this->MIN_ELEMENT_VAL)
  {
    const Vector v0 = p3 - p0;
    const Vector v1 = p - p0;
    w[i3] = Dot(v0, v1) / Dot(v0, v0);
    if( !finite( w[i3] ) ) w[i3] = 0.5;
    w[i0] = 1.0 - w[i3];
    return;
  }

  w[i0] = tri_area(p0, p1, p2) / (a0 * a3);
  w[i1] = tri_area(p1, p2, p0) / (a1 * a0);
  w[i2] = tri_area(p2, p3, p1) / (a2 * a1);
  w[i3] = tri_area(p3, p0, p2) / (a3 * a2);

  const double suminv = 1.0 / (w[i0] + w[i1] + w[i2] + w[i3]);
  w[i0] *= suminv;
  w[i1] *= suminv;
  w[i2] *= suminv;
  w[i3] *= suminv;
}


template <class Basis>
void
StructHexVolMesh<Basis>::compute_grid()
{
  grid_lock_.lock();
  if (grid_.get_rep() != 0) {grid_lock_.unlock(); return;} // only create once.

  BBox bb = get_bounding_box();
  if (bb.valid())
  {
    // Cubed root of number of cells to get a subdivision ballpark.
    typename LatVolMesh<Basis>::Cell::size_type csize;  size(csize);
    const int s = (int)(ceil(pow((double)csize , (1.0/3.0)))) / 2 + 1;
    const Vector cell_epsilon = bb.diagonal() * (1.0e-4 / s);
    bb.extend(bb.min() - cell_epsilon*2);
    bb.extend(bb.max() + cell_epsilon*2);

    SearchGridConstructor sgc(s, s, s, bb.min(), bb.max());

    BBox box;
    typename LatVolMesh<Basis>::Node::array_type nodes;
    typename LatVolMesh<Basis>::Cell::iterator ci, cie;
    begin(ci); end(cie);
    while(ci != cie)
    {
      get_nodes(nodes, *ci);

      box.reset();
      for (unsigned int i = 0; i < nodes.size(); i++)
      {
        box.extend(points_(nodes[i].i_, nodes[i].j_, nodes[i].k_));
      }
      const Point padmin(box.min() - cell_epsilon);
      const Point padmax(box.max() + cell_epsilon);
      box.extend(padmin);
      box.extend(padmax);

      sgc.insert(*ci, box);

      ++ci;
    }

    grid_ = scinew SearchGrid(sgc);
  }

  synchronized_ |= Mesh::LOCATE_E;
  grid_lock_.unlock();
}


template <class Basis>
double
StructHexVolMesh<Basis>::get_size(const typename LatVolMesh<Basis>::Node::index_type &idx) const
{
  return 0.0;
}


template <class Basis>
double
StructHexVolMesh<Basis>::get_size(typename LatVolMesh<Basis>::Edge::index_type idx) const
{
  typename LatVolMesh<Basis>::Node::array_type arr;
  get_nodes(arr, idx);
  Point p0, p1;
  get_center(p0, arr[0]);
  get_center(p1, arr[1]);

  return (p1.asVector() - p0.asVector()).length();
}


template <class Basis>
double
StructHexVolMesh<Basis>::get_size(typename LatVolMesh<Basis>::Face::index_type idx) const
{
  typename LatVolMesh<Basis>::Node::array_type nodes;
  get_nodes(nodes, idx);
  Point p0, p1, p2;
  get_point(p0, nodes[0]);
  get_point(p1, nodes[1]);
  get_point(p2, nodes[2]);
  Vector v0 = p1 - p0;
  Vector v1 = p2 - p0;
  return (v0.length() * v1.length());
}


template <class Basis>
double
StructHexVolMesh<Basis>::get_size(const typename LatVolMesh<Basis>::Cell::index_type &idx) const
{
  typename LatVolMesh<Basis>::Node::array_type nodes;
  get_nodes(nodes, idx);
  Point p0, p1, p2, p3;
  get_point(p0, nodes[0]);
  get_point(p1, nodes[1]);
  get_point(p2, nodes[3]);
  get_point(p3, nodes[4]);
  Vector v0 = p1 - p0;
  Vector v1 = p2 - p0;
  Vector v2 = p3 - p0;
  return (v0.length() * v1.length() * v2.length());
}


template <class Basis>
void
StructHexVolMesh<Basis>::set_point(const Point &p,
                       const typename LatVolMesh<Basis>::Node::index_type &i)
{
  points_(i.i_, i.j_, i.k_) = p;
}


template<class Basis>
void
StructHexVolMesh<Basis>::get_random_point(Point &p,
                     const typename LatVolMesh<Basis>::Elem::index_type &ei,
                                          MusilRNG &rng) const
{
  const Point &p0 = points_(ei.i_+0, ei.j_+0, ei.k_+0);
  const Point &p1 = points_(ei.i_+1, ei.j_+0, ei.k_+0);
  const Point &p2 = points_(ei.i_+1, ei.j_+1, ei.k_+0);
  const Point &p3 = points_(ei.i_+0, ei.j_+1, ei.k_+0);
  const Point &p4 = points_(ei.i_+0, ei.j_+0, ei.k_+1);
  const Point &p5 = points_(ei.i_+1, ei.j_+0, ei.k_+1);
  const Point &p6 = points_(ei.i_+1, ei.j_+1, ei.k_+1);
  const Point &p7 = points_(ei.i_+0, ei.j_+1, ei.k_+1);

  const double a0 = tetrahedra_volume(p0, p1, p2, p5);
  const double a1 = tetrahedra_volume(p0, p2, p3, p7);
  const double a2 = tetrahedra_volume(p0, p5, p2, p7);
  const double a3 = tetrahedra_volume(p0, p5, p7, p4);
  const double a4 = tetrahedra_volume(p5, p2, p7, p6);

  const double w = rng() * (a0 + a1 + a2 + a3 + a4);
  if (w > (a0 + a1 + a2 + a3))
  {
    uniform_sample_tetrahedra(p, p5, p2, p7, p6, rng);
  }
  else if (w > (a0 + a1 + a2))
  {
    uniform_sample_tetrahedra(p, p0, p5, p7, p4, rng);
  }
  else if (w > (a0 + a1))
  {
    uniform_sample_tetrahedra(p, p0, p5, p2, p7, rng);
  }
  else if (w > a0)
  {
    uniform_sample_tetrahedra(p, p0, p2, p3, p7, rng);
  }
  else
  {
    uniform_sample_tetrahedra(p, p0, p1, p2, p5, rng);
  }
}


template <class Basis>
bool
StructHexVolMesh<Basis>::synchronize(unsigned int tosync)
{
  if (tosync & Mesh::LOCATE_E && !(synchronized_ & Mesh::LOCATE_E))
    compute_grid();
  return true;
}


#define STRUCT_HEX_VOL_MESH_VERSION 1

template <class Basis>
void
StructHexVolMesh<Basis>::io(Piostream& stream)
{
  stream.begin_class(type_name(-1), STRUCT_HEX_VOL_MESH_VERSION);
  LatVolMesh<Basis>::io(stream);
  // IO data members, in order
  Pio(stream, points_);

  stream.end_class();
}


template <class Basis>
const std::string
StructHexVolMesh<Basis>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const std::string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;
  }
  else if (n == 0)
  {
    static const std::string nm("StructHexVolMesh");
    return nm;
  }
  else
  {
    return find_type_name((Basis *)0);
  }
}


template <class Basis>
const TypeDescription*
get_type_description(StructHexVolMesh<Basis> *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    const TypeDescription *sub = SCIRun::get_type_description((Basis*)0);
    TypeDescription::td_vec *subs = scinew TypeDescription::td_vec(1);
    (*subs)[0] = sub;
    td = scinew TypeDescription("StructHexVolMesh", subs,
                                std::string(__FILE__),
                                "SCIRun",
                                TypeDescription::MESH_E);
  }
  return td;
}


template <class Basis>
const TypeDescription*
StructHexVolMesh<Basis>::get_type_description() const
{
  return SCIRun::get_type_description((StructHexVolMesh<Basis> *)0);
}


template <class Basis>
const TypeDescription*
StructHexVolMesh<Basis>::node_type_description()
{
  static TypeDescription *td = 0;
  if (!td)
  {
    const TypeDescription *me =
      SCIRun::get_type_description((StructHexVolMesh<Basis> *)0);
    td = scinew TypeDescription(me->get_name() + "::Node",
                                std::string(__FILE__),
                                "SCIRun",
                                TypeDescription::MESH_E);
  }
  return td;
}


template <class Basis>
const TypeDescription*
StructHexVolMesh<Basis>::edge_type_description()
{
  static TypeDescription *td = 0;
  if (!td)
  {
    const TypeDescription *me =
      SCIRun::get_type_description((StructHexVolMesh<Basis> *)0);
    td = scinew TypeDescription(me->get_name() + "::Edge",
                                std::string(__FILE__),
                                "SCIRun",
                                TypeDescription::MESH_E);
  }
  return td;
}


template <class Basis>
const TypeDescription*
StructHexVolMesh<Basis>::face_type_description()
{
  static TypeDescription *td = 0;
  if (!td)
  {
    const TypeDescription *me =
      SCIRun::get_type_description((StructHexVolMesh<Basis> *)0);
    td = scinew TypeDescription(me->get_name() + "::Face",
                                std::string(__FILE__),
                                "SCIRun",
                                TypeDescription::MESH_E);
  }
  return td;
}


template <class Basis>
const TypeDescription*
StructHexVolMesh<Basis>::cell_type_description()
{
  static TypeDescription *td = 0;
  if (!td)
  {
    const TypeDescription *me =
      SCIRun::get_type_description((StructHexVolMesh<Basis> *)0);
    td = scinew TypeDescription(me->get_name() + "::Cell",
                                std::string(__FILE__),
                                "SCIRun",
                                TypeDescription::MESH_E);
  }
  return td;
}

} // namespace SCIRun

#endif // SCI_project_StructHexVolMesh_h
