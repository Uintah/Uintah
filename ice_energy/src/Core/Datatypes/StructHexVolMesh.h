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
 *  StructHexVolMesh.h: Templated Mesh defined on an 3D Structured Grid
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   April 2002
 *
 *  Copyright (C) 2002 SCI Group
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
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {

using std::string;

class StructHexVolMesh : public LatVolMesh
{
public:

  StructHexVolMesh();
  StructHexVolMesh(unsigned int i, unsigned int j, unsigned int k);
  StructHexVolMesh(const StructHexVolMesh &copy);
  virtual StructHexVolMesh *clone() { return new StructHexVolMesh(*this); }
  virtual ~StructHexVolMesh() {}

  //! get the mesh statistics
  virtual BBox get_bounding_box() const;
  virtual void transform(const Transform &t);

  bool get_dim(vector<unsigned int>&) const;

  void set_dim(vector<unsigned int> dims) {
    LatVolMesh::set_dim(dims);
    points_.resize(dims[0], dims[1], dims[2]);
  }

  //! get the center point (in object space) of an element
  void get_center(Point &, const Node::index_type &) const;
  void get_center(Point &, Edge::index_type) const;
  void get_center(Point &, Face::index_type) const;
  void get_center(Point &, const Cell::index_type &) const;

  double get_size(const Node::index_type &idx) const;
  double get_size(Edge::index_type idx) const;
  double get_size(Face::index_type idx) const;
  double get_size(const Cell::index_type &idx) const;
  double get_length(Edge::index_type idx) const { return get_size(idx); };
  double get_area(Face::index_type idx) const { return get_size(idx); };
  double get_volume(const Cell::index_type &i) const { return get_size(i); };

  bool locate(Node::index_type &, const Point &);
  bool locate(Edge::index_type &, const Point &) const { return false; }
  bool locate(Face::index_type &, const Point &) const { return false; }
  bool locate(Cell::index_type &, const Point &);

  int get_weights(const Point &, Node::array_type &, double *);
  int get_weights(const Point &, Edge::array_type &, double *)
  { ASSERTFAIL("StructHexVolMesh::get_weights for edges isn't supported"); }
  int get_weights(const Point &, Face::array_type &, double *)
  { ASSERTFAIL("StructHexVolMesh::get_weights for faces isn't supported"); }
  int get_weights(const Point &, Cell::array_type &, double *);

  void get_point(Point &point, const Node::index_type &index) const
  { get_center(point, index); }
  void set_point(const Point &point, const Node::index_type &index);

  void get_random_point(Point &/*p*/, const Elem::index_type &/*ei*/, int /*seed=0*/) const
  { ASSERTFAIL("not implemented") }

  //! the double return val is the volume of the Hex.
  double get_gradient_basis(Cell::index_type /*ci*/, Vector& /*g0*/, Vector& /*g1*/,
			    Vector& /*g2*/, Vector& /*g3*/, Vector& /*g4*/,
			    Vector& /*g5*/, Vector& /*g6*/, Vector& /*g7*/)
  { ASSERTFAIL("not implemented") }

  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  static  const string type_name(int n = -1);
  virtual const string get_type_name(int n = -1) const { return type_name(n); }

  virtual const TypeDescription *get_type_description() const;

private:
  void compute_grid();
  double inside8_p(Cell::index_type i, const Point &p) const;
  double polygon_area(const Node::array_type &, const Vector) const;
  double pyramid_volume(const Node::array_type &, const Point &) const;
  void get_face_weights(double *w, const Node::array_type &nodes,
			const Point &p, int i0, int i1, int i2, int i3);
  const Point &point(const Node::index_type &i) const 
  { return points_(i.i_, i.j_, i.k_); }
  

  Array3<Point> points_;

  LockingHandle<SearchGrid>   grid_;
  Mutex                       grid_lock_; // Bad traffic!
  Cell::index_type           locate_cache_;

  // returns a StructHexVolMesh
  static Persistent *maker() { return new StructHexVolMesh(); }
};

typedef LockingHandle<StructHexVolMesh> StructHexVolMeshHandle;

const TypeDescription* get_type_description(StructHexVolMesh *);
} // namespace SCIRun

#endif // SCI_project_StructHexVolMesh_h
