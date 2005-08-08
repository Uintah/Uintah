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
 *  StructQuadSurfMesh.h: Templated Mesh defined on a 3D Structured Grid
 *
 *  Written by:
 *   Allen R. Sanderson
 *   Department of Computer Science
 *   University of Utah
 *   November 2002
 *
 *  Copyright (C) 2002 SCI Group
 *
 */

/*
  A structured grid is a dataset with regular topology but with irregular 
  geometry.  The grid may have any shape but can not be overlapping or 
  self-intersecting. The topology of a structured grid is represented 
  using a 2D, or 3D vector with the points being stored in an index 
  based array. The faces (quadrilaterals) and  cells (Hexahedron) are 
  implicitly define based based upon their indexing.

  Structured grids are typically used in finite difference analysis.

  For more information on datatypes see Schroeder, Martin, and Lorensen,
  "The Visualization Toolkit", Prentice Hall, 1998.
*/

#ifndef SCI_project_StructQuadSurfMesh_h
#define SCI_project_StructQuadSurfMesh_h 1

#include <Core/Datatypes/ImageMesh.h>
#include <Core/Containers/Array2.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/BBox.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {

using std::string;

template <class Basis>
class StructQuadSurfMesh : public ImageMesh<Basis>
{
public:
  StructQuadSurfMesh();
  StructQuadSurfMesh(unsigned int x, unsigned int y);
  StructQuadSurfMesh(const StructQuadSurfMesh<Basis> &copy);
  virtual StructQuadSurfMesh *clone() 
  { return new StructQuadSurfMesh<Basis>(*this); }
  virtual ~StructQuadSurfMesh() {}

  //! get the mesh statistics
  virtual BBox get_bounding_box() const;
  virtual void transform(const Transform &t);

  bool get_dim(vector<unsigned int> &array) const;
  void set_dim(vector<unsigned int> dims) {
    ImageMesh<Basis>::set_dim(dims);
    points_.resize(dims[0], dims[1]);
    normals_.resize(dims[0], dims[1]);
  }

  //! get the child elements of the given index
  void get_nodes(typename Node::array_type &, typename Edge::index_type) const;
  void get_nodes(typename Node::array_type &, 
		 const typename Face::index_type &) const;
  void get_nodes(typename Node::array_type &, 
		 typename Cell::index_type) const {}
  void get_edges(typename Edge::array_type &, 
		 const typename Face::index_type &) const;
  void get_edges(typename Edge::array_type &, 
		 typename Cell::index_type) const {}
  void get_faces(typename Face::array_type &, 
		 typename Cell::index_type) const {}

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

  //! return all face_indecies that overlap the BBox in arr.
  void get_faces(typename Face::array_type &, const BBox &)
  { ASSERTFAIL("ScanlineMesh::get_faces for BBox is not implemented."); }

  //! Get the size of an elemnt (length, area, volume)
  double get_size(const typename Node::index_type &) const { return 0.0; }
  double get_size(typename Edge::index_type idx) const 
  {
    typename Node::array_type arr;
    get_nodes(arr, idx);
    Point p0, p1;
    get_center(p0, arr[0]);
    get_center(p1, arr[1]);
    return (p1.asVector() - p0.asVector()).length();
  }
  double get_size(const typename Face::index_type &idx) const
  {
    typename Node::array_type ra;
    get_nodes(ra,idx);
    Point p0,p1,p2;
    get_point(p0,ra[0]);
    get_point(p1,ra[1]);
    get_point(p2,ra[2]);
    return (Cross(p0-p1,p2-p0)).length()*0.5;
  }

  double get_size(typename Cell::index_type) const { return 0.0; }
  double get_length(typename Edge::index_type idx) const 
  { return get_size(idx); }
  double get_area(const typename Face::index_type &idx) const 
  { return get_size(idx); }
  double get_volume(typename Cell::index_type idx) const 
  { return get_size(idx); }

  void get_normal(Vector &, const typename Node::index_type &) const;

  //! get the center point (in object space) of an element
  void get_center(Point &, const typename Node::index_type &) const;
  void get_center(Point &, typename Edge::index_type) const;
  void get_center(Point &, const typename Face::index_type &) const;
  void get_center(Point &, typename Cell::index_type) const {}

  bool locate(typename Node::index_type &, const Point &) const;
  bool locate(typename Edge::index_type &, const Point &) const;
  bool locate(typename Face::index_type &, const Point &) const;
  bool locate(typename Cell::index_type &, const Point &) const;

  void get_point(Point &point, const typename Node::index_type &index) const
  { get_center(point, index); }
  void set_point(const Point &point, const typename Node::index_type &index);

  void get_random_point(Point &, const typename Elem::index_type &, int) const
  { ASSERTFAIL("not implemented") }

  virtual bool is_editable() const { return false; }

  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  static  const string type_name(int n = -1);
  virtual const TypeDescription *get_type_description() const;

  virtual bool synchronize(unsigned int);
  
  static const TypeDescription* node_type_description();
  static const TypeDescription* edge_type_description();
  static const TypeDescription* face_type_description();
  static const TypeDescription* cell_type_description();

protected:
  void compute_normals();
  void compute_edge_neighbors(double err = 1.0e-8);

  const Point &point(const typename Node::index_type &idx)
  { return points_(idx.i_, idx.j_); }

  int next(int i) { return ((i%4)==3) ? (i-3) : (i+1); }
  int prev(int i) { return ((i%4)==0) ? (i+3) : (i-1); }

  Array2<Point>  points_;
  Array2<Vector> normals_; //! normalized per node
  Mutex		 normal_lock_;
  unsigned int   synchronized_;

  // returns a StructQuadSurfMesh
  static Persistent *maker() { return new StructQuadSurfMesh<Basis>(); }
};

template <class Basis>
PersistentTypeID 
StructQuadSurfMesh<Basis>::type_id(StructQuadSurfMesh<Basis>::type_name(-1), 
				   "Mesh", maker);

template <class Basis>
StructQuadSurfMesh<Basis>::StructQuadSurfMesh()
  : normal_lock_("StructQuadSurfMesh Normals Lock"),
    synchronized_(ALL_ELEMENTS_E)
{
}

template <class Basis>  
StructQuadSurfMesh<Basis>::StructQuadSurfMesh(unsigned int x, unsigned int y)
  : ImageMesh<Basis>(x, y, Point(0.0, 0.0, 0.0), Point(1.0, 1.0, 1.0)),
    points_(x, y),
    normals_(x,y),
    normal_lock_("StructQuadSurfMesh Normals Lock"),
    synchronized_(ALL_ELEMENTS_E)
{
}

template <class Basis>
StructQuadSurfMesh<Basis>::StructQuadSurfMesh(const StructQuadSurfMesh &copy)
  : ImageMesh<Basis>(copy),
    normal_lock_("StructQuadSurfMesh Normals Lock"),
    synchronized_(copy.synchronized_)
{
  points_.copy( copy.points_ );
  normals_.copy( copy.normals_ );
}

template <class Basis>
bool
StructQuadSurfMesh<Basis>::get_dim(vector<unsigned int> &array) const
{
  array.resize(2);
  array.clear();

  array.push_back(ni_);
  array.push_back(nj_);

  return true;
}

template <class Basis>
BBox
StructQuadSurfMesh<Basis>::get_bounding_box() const
{
  BBox result;

  typename Node::iterator ni, nie;
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
StructQuadSurfMesh<Basis>::transform(const Transform &t)
{
  typename Node::iterator i, ie;
  begin(i);
  end(ie);

  while (i != ie) {
    points_((*i).i_,(*i).j_) = t.project(points_((*i).i_,(*i).j_));

    ++i;
  }
}

template <class Basis>
void
StructQuadSurfMesh<Basis>::get_nodes(typename Node::array_type &array, 
				     typename Edge::index_type idx) const
{
  array.resize(2);

  const int yidx = idx - (ni_-1) * nj_;
  if (yidx >= 0)
  {
    const int i = yidx / (nj_ - 1);
    const int j = yidx % (nj_ - 1);
    array[0] = typename Node::index_type(this, i, j);
    array[1] = typename Node::index_type(this, i, j+1);
  }
  else
  {
    const int i = idx % (ni_ - 1);
    const int j = idx / (ni_ - 1);
    array[0] = typename Node::index_type(this, i, j);
    array[1] = typename Node::index_type(this, i+1, j);
  }
}

template <class Basis>
void
StructQuadSurfMesh<Basis>::get_nodes(typename Node::array_type &array, 
				   const typename Face::index_type &idx) const
{
  const int arr_size = 4;
  array.resize(arr_size);

  for (int i = 0; i < arr_size; i++)
    array[i].mesh_ = idx.mesh_;

  array[0].i_ = idx.i_;   array[0].j_ = idx.j_;
  array[1].i_ = idx.i_+1; array[1].j_ = idx.j_;
  array[2].i_ = idx.i_+1; array[2].j_ = idx.j_+1;
  array[3].i_ = idx.i_;   array[3].j_ = idx.j_+1;
}

template <class Basis>
void
StructQuadSurfMesh<Basis>::get_edges(typename Edge::array_type &array,
				   const typename Face::index_type &idx) const
{
  array.clear();
  array.push_back(idx * 4 + 0);
  array.push_back(idx * 4 + 1);
  array.push_back(idx * 4 + 2);
  array.push_back(idx * 4 + 3);
}

template <class Basis>
void
StructQuadSurfMesh<Basis>::get_normal(Vector &result,
				  const typename Node::index_type &idx ) const
{
  result = normals_(idx.i_, idx.j_);
}

template <class Basis>
void
StructQuadSurfMesh<Basis>::get_center(Point &result,
				    const typename Node::index_type &idx) const
{
  result = points_(idx.i_, idx.j_);
}

template <class Basis>
void
StructQuadSurfMesh<Basis>::get_center(Point &result, 
				      typename Edge::index_type idx) const
{
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
StructQuadSurfMesh<Basis>::get_center(Point &p,
				    const typename Face::index_type &idx) const
{
  typename Node::array_type nodes;
  get_nodes(nodes, idx);
  ASSERT(nodes.size() == 4);
  typename Node::array_type::iterator nai = nodes.begin();
  get_point(p, *nai);
  ++nai;
  Point pp;
  while (nai != nodes.end())
  {
    get_point(pp, *nai);
    p.asVector() += pp.asVector();
    ++nai;
  }
  p.asVector() *= (1.0 / 4.0);
}


template <class Basis>
bool
StructQuadSurfMesh<Basis>::locate(typename Node::index_type &node, 
				  const Point &p) const
{
  node.mesh_ = this;
  typename Face::index_type fi;
  if (locate(fi, p)) { // first try the fast way.
    typename Node::array_type nodes;
    get_nodes(nodes, fi);

    double dmin = (p-points_(nodes[0].i_, nodes[0].j_)).length2();
    node = nodes[0];
    for (unsigned int i = 1; i < nodes.size(); i++)
    {
      const double d = (p-points_(nodes[i].i_, nodes[i].j_)).length2();
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
    typename Node::iterator ni, nie;
    begin(ni);
    end(nie);
    if (ni == nie) { return false; }

    double min_dist = (p-points_((*ni).i_, (*ni).j_)).length2();
    node = *ni;
    ++ni;

    while (ni != nie)
    {
      const double dist = (p-points_((*ni).i_, (*ni).j_)).length2();
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
bool
StructQuadSurfMesh<Basis>::locate(typename Edge::index_type &node, 
				  const Point &) const
{
  ASSERTFAIL("Locate Edge not implemented in StructQuadSurfMesh");
  return false;
}


template <class Basis>
bool
StructQuadSurfMesh<Basis>::locate(typename Face::index_type &loc, 
				  const Point &) const
{
  ASSERTFAIL("Locate Face not implemented in StructQuadSurfMesh");
  return false;
}


template <class Basis>
bool
StructQuadSurfMesh<Basis>::locate(typename Cell::index_type &loc, 
				  const Point &) const
{
  ASSERTFAIL("Locate Cell not implemented in StructQuadSurfMesh");
  return false;
}


template <class Basis>
void
StructQuadSurfMesh<Basis>::set_point(const Point &point, 
				     const typename Node::index_type &index)
{
  points_(index.i_, index.j_) = point;
}

template <class Basis>
bool
StructQuadSurfMesh<Basis>::synchronize(unsigned int tosync)
{
  if (tosync & NORMALS_E && !(synchronized_ & NORMALS_E))
    compute_normals();
  return true;
}

template <class Basis>
void
StructQuadSurfMesh<Basis>::compute_normals()
{
  normal_lock_.lock();
  if (synchronized_ & NORMALS_E) {
    normal_lock_.unlock();
    return;
  }
  normals_.resize(points_.dim1(), points_.dim2()); // 1 per node

  // build table of faces that touch each node
  Array2< vector<typename Face::index_type> >
    node_in_faces(points_.dim1(), points_.dim2());

  //! face normals (not normalized) so that magnitude is also the area.
  Array2<Vector> face_normals((points_.dim1()-1),(points_.dim2()-1));

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
    node_in_faces(nodes[0].i_,nodes[0].j_).push_back(*iter);
    node_in_faces(nodes[1].i_,nodes[1].j_).push_back(*iter);
    node_in_faces(nodes[2].i_,nodes[2].j_).push_back(*iter);
    node_in_faces(nodes[3].i_,nodes[3].j_).push_back(*iter);

    Vector v0 = p1 - p0;
    Vector v1 = p2 - p1;
    Vector n = Cross(v0, v1);
    face_normals((*iter).i_, (*iter).j_) = n;

    ++iter;
  }

  //Averaging the normals.
  typename Node::iterator nif_iter, nif_iter_end;
  begin( nif_iter );
  end( nif_iter_end );

  while (nif_iter != nif_iter_end) {
    vector<typename Face::index_type> v = node_in_faces((*nif_iter).i_, 
							(*nif_iter).j_);
    typename vector<typename Face::index_type>::const_iterator fiter = 
      v.begin();
    Vector ave(0.L,0.L,0.L);
    while(fiter != v.end()) {
      ave += face_normals((*fiter).i_,(*fiter).j_);
      ++fiter;
    }
    ave.safe_normalize();
    normals_((*nif_iter).i_, (*nif_iter).j_) = ave;
    ++nif_iter;
  }
  synchronized_ |= NORMALS_E;
  normal_lock_.unlock();
}

#define STRUCT_QUAD_SURF_MESH_VERSION 2

template <class Basis>
void
StructQuadSurfMesh<Basis>::io(Piostream& stream)
{
  stream.begin_class(type_name(-1), STRUCT_QUAD_SURF_MESH_VERSION);
  ImageMesh<Basis>::io(stream);
  
  Pio(stream, points_);

  stream.end_class();
}

template <class Basis>
const string
StructQuadSurfMesh<Basis>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;
  }
  else if (n == 0)
  {
    static const string nm("StructQuadSurfMesh");
    return nm;
  }
  else 
  {
    return find_type_name((Basis *)0);
  }
}

template <class Basis>
const TypeDescription*
StructQuadSurfMesh<Basis>::get_type_description() const
{
  return SCIRun::get_type_description((StructQuadSurfMesh<Basis> *)0);
}

template <class Basis>
const TypeDescription*
get_type_description(StructQuadSurfMesh<Basis> *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    const TypeDescription *sub = SCIRun::get_type_description((Basis*)0);
    TypeDescription::td_vec *subs = scinew TypeDescription::td_vec(1);
    (*subs)[0] = sub;
    td = scinew TypeDescription(StructQuadSurfMesh<Basis>::type_name(0), subs,
				string(__FILE__),
				"SCIRun");
  }
  return td;
}

template <class Basis>
const TypeDescription*
StructQuadSurfMesh<Basis>::node_type_description()
{
  static TypeDescription *td = 0;
  if (!td)
  {
    const TypeDescription *me = 
      SCIRun::get_type_description((StructQuadSurfMesh<Basis> *)0);
    td = scinew TypeDescription(me->get_name() + "::Node",
				string(__FILE__),
				"SCIRun");
  }
  return td;
}

template <class Basis>
const TypeDescription*
StructQuadSurfMesh<Basis>::edge_type_description()
{
  static TypeDescription *td = 0;
  if (!td)
  {
    const TypeDescription *me = 
      SCIRun::get_type_description((StructQuadSurfMesh<Basis> *)0);
    td = scinew TypeDescription(me->get_name() + "::Edge",
				string(__FILE__),
				"SCIRun");
  }
  return td;
}

template <class Basis>
const TypeDescription*
StructQuadSurfMesh<Basis>::face_type_description()
{
  static TypeDescription *td = 0;
  if (!td)
  {
    const TypeDescription *me = 
      SCIRun::get_type_description((StructQuadSurfMesh<Basis> *)0);
    td = scinew TypeDescription(me->get_name() + "::Face",
				string(__FILE__),
				"SCIRun");
  }
  return td;
}

template <class Basis>
const TypeDescription*
StructQuadSurfMesh<Basis>::cell_type_description()
{
  static TypeDescription *td = 0;
  if (!td)
  {
    const TypeDescription *me = 
      SCIRun::get_type_description((StructQuadSurfMesh<Basis> *)0);
    td = scinew TypeDescription(me->get_name() + "::Cell",
				string(__FILE__),
				"SCIRun");
  }
  return td;
}

} // namespace SCIRun

#endif // SCI_project_StructQuadSurfMesh_h
