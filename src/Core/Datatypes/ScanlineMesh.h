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
 *  ScanlineMesh.h: Templated Mesh defined on a 3D Regular Grid
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

#ifndef SCI_project_ScanlineMesh_h
#define SCI_project_ScanlineMesh_h 1

#include <Core/Datatypes/Mesh.h>
#include <Core/Datatypes/FieldIterator.h>
#include <Core/Geometry/Transform.h>
#include <Core/Geometry/Point.h>
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
  void get_edges(typename Edge::array_type &, typename Face::index_type) const 
  {}
  void get_edges(typename Edge::array_type &, typename Cell::index_type) const 
  {}
  void get_faces(typename Face::array_type &, typename Cell::index_type) const 
  {}

  //! get the parent element(s) of the given index
  void get_edges(typename Edge::array_type &a, 
		 typename Node::index_type idx) const
  { a.push_back(typename Edge::index_type(idx));}
  //! needed to support Mesh concept
  void get_edges(typename Edge::array_type &a, 
		 typename Edge::index_type idx) const
  { a.push_back(idx);}
  bool get_faces(typename Face::array_type &, typename Node::index_type) const 
  { return 0; }
  bool get_faces(typename Face::array_type &, typename Edge::index_type) const 
  { return 0; }
  bool get_cells(typename Cell::array_type &, typename Node::index_type) const 
  { return 0; }
  bool get_cells(typename Cell::array_type &, typename Edge::index_type) const 
  { return 0; }
  bool get_cells(typename Cell::array_type &, typename Face::index_type) const 
  { return 0; }

  //! return all edge_indecies that overlap the BBox in arr.
  void get_edges(typename Edge::array_type &arr, const BBox &box) const;

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

  void get_point(Point &p, typename Node::index_type i) const { get_center(p, i); }
  void get_normal(Vector &/*normal*/, typename Node::index_type /*index*/) const
  { ASSERTFAIL("not implemented") }

  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  static  const string type_name(int n = -1);
  virtual const TypeDescription *get_type_description() const;

  // Unsafe due to non-constness of unproject.
  Transform &get_transform() { return transform_; }
  Transform &set_transform(const Transform &trans) 
  { transform_ = trans; return transform_; }

  virtual int dimensionality() const { return 1; }
  Basis& get_basis() { return basis_; }

  static const TypeDescription* node_type_description();
  static const TypeDescription* edge_type_description();
  static const TypeDescription* face_type_description();
  static const TypeDescription* cell_type_description();

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

//! return all cell_indecies that overlap the BBox in arr.
template <class Basis>
void
ScanlineMesh<Basis>::get_edges(typename Edge::array_type &/* arr */, 
			       const BBox &/*bbox*/) const
{
  // TODO: implement this
}


template <class Basis>
void
ScanlineMesh<Basis>::get_center(Point &result, typename Node::index_type idx) const
{
  Point p(idx, 0.0, 0.0);
  result = transform_.project(p);
}


template <class Basis>
void
ScanlineMesh<Basis>::get_center(Point &result, typename Edge::index_type idx) const
{
  Point p(idx + 0.5, 0.0, 0.0);
  result = transform_.project(p);
}

// TODO: verify
template <class Basis>
bool
ScanlineMesh<Basis>::locate(typename Edge::index_type &elem, const Point &p)
{
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


#define SCANLINEMESH_VERSION 2

template <class Basis>
void
ScanlineMesh<Basis>::io(Piostream& stream)
{
  int version = stream.begin_class(type_name(-1), SCANLINEMESH_VERSION);

  Mesh::io(stream);

  // IO data members, in order
  Pio(stream, ni_);
  if (version < 2 && stream.reading() ) {
    Pio_old(stream, transform_);
  } else {
    Pio(stream, transform_);
  }
  stream.end_class();
}

template <class Basis>
const string
ScanlineMesh<Basis>::type_name(int n)
{
  ASSERT(n >= -1 && n <= 0);
  static const string name = "ScanlineMesh";
  return name;
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
ScanlineMesh<Basis>::get_type_description() const
{
  return SCIRun::get_type_description((ScanlineMesh *)0);
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
    td = scinew TypeDescription(ScanlineMesh<Basis>::type_name(0), subs,
				string(__FILE__),
				"SCIRun");
  }
  return td;
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
				"SCIRun");
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
				"SCIRun");
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
				"SCIRun");
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
				"SCIRun");
  }
  return td;
}

} // namespace SCIRun

#endif // SCI_project_ScanlineMesh_h
