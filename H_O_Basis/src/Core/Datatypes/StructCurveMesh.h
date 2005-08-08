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
 *  StructCurveMesh.h: Templated Mesh defined on a 1D Structured Grid
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
  A sturctured curve is a dataset with regular topology but with irregular geometry.
  The line defined may have any shape but can not be overlapping or self-intersecting.
  
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
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {

using std::string;

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

  //! get the child elements of the given index
  void get_nodes(typename Node::array_type &, 
		 typename Edge::index_type) const;
  void get_nodes(typename Node::array_type &, 
		 typename Face::index_type) const {}
  void get_nodes(typename Node::array_type &, 
		 typename Cell::index_type) const {}
  void get_edges(typename Edge::array_type &, 
		 typename Face::index_type) const {}
  void get_edges(typename Edge::array_type &, 
		 typename Cell::index_type) const {}
  void get_edges(typename Edge::array_type &a, 
		 typename Edge::index_type idx) const
  { a.push_back(idx);}
  //void get_faces(typename Face::array_type &, typename Cell::index_type) const {}

  //! get the parent element(s) of the given index
  void get_edges(typename Edge::array_type &a, 
		 typename Node::index_type idx) const
  { a.push_back(typename Edge::index_type(idx));}
  bool get_faces(typename Face::array_type &, 
		 typename Node::index_type) const { return 0; }
  bool get_faces(typename Face::array_type &, 
		 typename Edge::index_type) const { return 0; }
  bool get_cells(typename Cell::array_type &, 
		 typename Node::index_type) const { return 0; }
  bool get_cells(typename Cell::array_type &, 
		 typename Edge::index_type) const { return 0; }
  bool get_cells(typename Cell::array_type &, 
		 typename Face::index_type) const { return 0; }

  //! return all edge_indecies that overlap the BBox in arr.
  void get_edges(typename Edge::array_type &, const BBox &) const
  { ASSERTFAIL("ScanlineMesh::get_edges for BBox is not implemented."); }

  //! Get the size of an elemnt (length, area, volume)
  double get_size(typename Node::index_type) const { return 0.0; }
  double get_size(typename Edge::index_type idx) const 
  {
    typename Node::array_type arr;
    get_nodes(arr, idx);
    Point p0, p1;
    get_center(p0, arr[0]);
    get_center(p1, arr[1]);
    return (p1.asVector() - p0.asVector()).length();
  }  

  double get_size(typename Face::index_type) const { return 0.0; }
  double get_size(typename Cell::index_type) const { return 0.0; }
  double get_length(typename Edge::index_type idx) const 
  { return get_size(idx); }
  double get_area(typename Face::index_type idx) const 
  { return get_size(idx); }
  double get_volume(typename Cell::index_type idx) const 
  { return get_size(idx); }

  int get_valence(typename Node::index_type idx) const 
  { return (idx == (unsigned int) 0 ||
	    idx == (unsigned int) (points_.size()-1)) ? 1 : 2; }

  int get_valence(typename Edge::index_type) const { return 0; }
  int get_valence(typename Face::index_type) const { return 0; }
  int get_valence(typename Cell::index_type) const { return 0; }

  //! get the center point (in object space) of an element
  void get_center(Point &, const typename Node::index_type &) const;
  void get_center(Point &, const typename Edge::index_type &) const;
  void get_center(Point &, const typename Face::index_type &) const {}
  void get_center(Point &, const typename Cell::index_type &) const {}

  bool locate(typename Node::index_type &, const Point &) const;
  bool locate(typename Edge::index_type &, const Point &) const;
  bool locate(typename Face::index_type &, const Point &) const 
  { return false; }
  bool locate(typename Cell::index_type &, const Point &) const 
  { return false; }

  void get_point(Point &p, typename Node::index_type i) const 
  { get_center(p,i); }
  void set_point(const Point &p, typename Node::index_type i) 
  { points_[i] = p; }

  void get_random_point(Point &, const typename Elem::index_type &, 
			int) const
  { ASSERTFAIL("not implemented") }

  virtual bool is_editable() const { return false; }
    
  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  static  const string type_name(int n = -1);
  virtual const TypeDescription *get_type_description() const;
  static const TypeDescription* node_type_description();
  static const TypeDescription* edge_type_description();
  static const TypeDescription* face_type_description();
  static const TypeDescription* cell_type_description();

private:

  //! the points
  Array1<Point> points_;

  // returns a StructCurveMesh
  static Persistent *maker() { return new StructCurveMesh<Basis>(); }
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

  array.push_back(ni_);

  return true;
}

template <class Basis>
BBox
StructCurveMesh<Basis>::get_bounding_box() const
{
  BBox result;
  
  typename Node::iterator i, ie;
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
  typename Node::iterator i, ie;
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
  
  typename Node::iterator i, i1, ie;
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
StructCurveMesh<Basis>::get_nodes(typename Node::array_type &array, typename Edge::index_type idx) const
{
  array.resize(2);
  array[0] = typename Node::index_type(idx);
  array[1] = typename Node::index_type(idx + 1);
}

template <class Basis>
void
StructCurveMesh<Basis>::get_center(Point &result, const typename Node::index_type &idx) const
{
  result = points_[idx];
}

template <class Basis>
void
StructCurveMesh<Basis>::get_center(Point &result, const typename Edge::index_type &idx) const
{
  Point p0 = points_[typename Node::index_type(idx)];
  Point p1 = points_[typename Node::index_type(idx+1)];

  result = Point(p0+p1)/2.0;
}

template <class Basis>
bool
StructCurveMesh<Basis>::locate(typename Node::index_type &idx, const Point &p) const
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
bool
StructCurveMesh<Basis>::locate(typename Edge::index_type &idx, const Point &p) const
{
  typename Edge::iterator ei;
  typename Edge::iterator eie;
  double cosa, closest=DBL_MAX;
  typename Node::array_type nra;
  double dist1, dist2, dist3, dist4;
  Point n1,n2,q;

  begin(ei);
  end(eie);

  if (ei==eie)
    return false;
  
  for (; ei != eie; ++ei) {
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
const string
StructCurveMesh<Basis>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;
  }
  else if (n == 0)
  {
    static const string nm("StructCurveMesh");
    return nm;
  }
  else 
  {
    return find_type_name((Basis *)0);
  }
}

template <class Basis>
const TypeDescription*
StructCurveMesh<Basis>::get_type_description() const
{
  return SCIRun::get_type_description((StructCurveMesh<Basis> *)0);
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
    td = scinew TypeDescription(StructCurveMesh<Basis>::type_name(0), subs,
				string(__FILE__),
				"SCIRun");
  }
  return td;
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
				string(__FILE__),
				"SCIRun");
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
				string(__FILE__),
				"SCIRun");
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
				string(__FILE__),
				"SCIRun");
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
				string(__FILE__),
				"SCIRun");
  }
  return td;
}

} // namespace SCIRun

#endif // SCI_project_StructCurveMesh_h
