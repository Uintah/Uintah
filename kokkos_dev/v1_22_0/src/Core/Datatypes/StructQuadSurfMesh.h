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
  A sturctured grid is a dataset with regular topology but with irregular geometry.
  The grid may have any shape but can not be overlapping or self-intersecting.
  
  The topology of a structured grid is represented using a 2D, or 3D vector with
  the points being stored in an index based array. The faces (quadrilaterals) and
  cells (Hexahedron) are implicitly define based based upon their indexing.

  Structured grids are typically used in finite difference analysis.

  For more information on datatypes see Schroeder, Martin, and Lorensen,
  "The Visualization Toolkit", Prentice Hall, 1998.
*/

#ifndef SCI_project_StructQuadSurfMesh_h
#define SCI_project_StructQuadSurfMesh_h 1

#include <Core/Datatypes/ImageMesh.h>
#include <Core/Containers/Array2.h>
#include <Core/Geometry/Point.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {

using std::string;

class SCICORESHARE StructQuadSurfMesh : public ImageMesh
{
public:
  StructQuadSurfMesh();
  StructQuadSurfMesh(unsigned int x, unsigned int y);
  StructQuadSurfMesh(const StructQuadSurfMesh &copy);
  virtual StructQuadSurfMesh *clone() { return new StructQuadSurfMesh(*this); }
  virtual ~StructQuadSurfMesh() {}

  //! get the mesh statistics
  virtual BBox get_bounding_box() const;
  virtual void transform(const Transform &t);

  bool get_dim(vector<unsigned int> &array) const;
  void set_dim(vector<unsigned int> dims) {
    ImageMesh::set_dim(dims);
    points_.resize(dims[0], dims[1]);
    normals_.resize(dims[0], dims[1]);
  }

  //! get the child elements of the given index
  void get_nodes(Node::array_type &, Edge::index_type) const;
  void get_nodes(Node::array_type &, const Face::index_type &) const;
  void get_nodes(Node::array_type &, Cell::index_type) const {}
  void get_edges(Edge::array_type &, const Face::index_type &) const;
  void get_edges(Edge::array_type &, Cell::index_type) const {}
  void get_faces(Face::array_type &, Cell::index_type) const {}

  //! get the parent element(s) of the given index
  unsigned get_edges(Edge::array_type &, Node::index_type) const { return 0; }
  unsigned get_faces(Face::array_type &, Node::index_type) const { return 0; }
  unsigned get_faces(Face::array_type &, Edge::index_type) const { return 0; }
  unsigned get_cells(Cell::array_type &, Node::index_type) const { return 0; }
  unsigned get_cells(Cell::array_type &, Edge::index_type) const { return 0; }
  unsigned get_cells(Cell::array_type &, Face::index_type) const { return 0; }

  //! return all face_indecies that overlap the BBox in arr.
  void get_faces(Face::array_type &arr, const BBox &box)
  { ASSERTFAIL("ScanlineMesh::get_faces for BBox is not implemented."); }

  //! Get the size of an elemnt (length, area, volume)
  double get_size(const Node::index_type &idx) const { return 0.0; }
  double get_size(Edge::index_type idx) const 
  {
    Node::array_type arr;
    get_nodes(arr, idx);
    Point p0, p1;
    get_center(p0, arr[0]);
    get_center(p1, arr[1]);
    return (p1.asVector() - p0.asVector()).length();
  }
  double get_size(const Face::index_type &idx) const
  {
    Node::array_type ra;
    get_nodes(ra,idx);
    Point p0,p1,p2;
    get_point(p0,ra[0]);
    get_point(p1,ra[1]);
    get_point(p2,ra[2]);
    return (Cross(p0-p1,p2-p0)).length()*0.5;
  }
  double get_size(Cell::index_type idx) const { return 0.0; }
  double get_length(Edge::index_type idx) const { return get_size(idx); };
  double get_area(const Face::index_type &idx) const { return get_size(idx); };
  double get_volume(Cell::index_type idx) const { return get_size(idx); };

  void get_normal(Vector &, const Node::index_type &) const;

  //! get the center point (in object space) of an element
  void get_center(Point &, const Node::index_type &) const;
  void get_center(Point &, Edge::index_type) const;
  void get_center(Point &, const Face::index_type &) const;
  void get_center(Point &, Cell::index_type) const {}

  bool locate(Node::index_type &, const Point &) const;
  bool locate(Edge::index_type &, const Point &) const;
  bool locate(Face::index_type &, const Point &) const;
  bool locate(Cell::index_type &, const Point &) const;

  int get_weights(const Point &p, Node::array_type &l, double *w);
  int get_weights(const Point & , Edge::array_type & , double * )
  { ASSERTFAIL("StructQuadSurfMesh::get_weights for edges isn't supported"); }
  int get_weights(const Point &p, Face::array_type &l, double *w);
  int get_weights(const Point & , Cell::array_type & , double * )
  { ASSERTFAIL("StructQuadSurfMesh::get_weights for cells isn't supported"); }

  void get_point(Point &point, const Node::index_type &index) const
  { get_center(point, index); }
  void set_point(const Point &point, const Node::index_type &index);

  void get_random_point(Point &p, const Elem::index_type &ei, int seed=0) const
  { ASSERTFAIL("not implemented") }

  virtual bool is_editable() const { return false; }

  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  static  const string type_name(int n = -1);
  virtual const TypeDescription *get_type_description() const;

  virtual bool synchronize(unsigned int);

protected:
  void compute_normals();
  void compute_edge_neighbors(double err = 1.0e-8);

  const Point &point(const Node::index_type &idx)
  { return points_(idx.i_, idx.j_); }

  int next(int i) { return ((i%4)==3) ? (i-3) : (i+1); }
  int prev(int i) { return ((i%4)==0) ? (i+3) : (i-1); }

  Array2<Point>  points_;
  Array2<Vector> normals_; //! normalized per node
  Mutex		 normal_lock_;
  unsigned int   synchronized_;

  // returns a StructQuadSurfMesh
  static Persistent *maker() { return new StructQuadSurfMesh(); }
};

typedef LockingHandle<StructQuadSurfMesh> StructQuadSurfMeshHandle;

const TypeDescription* get_type_description(StructQuadSurfMesh *);
const TypeDescription* get_type_description(StructQuadSurfMesh::Node *);
const TypeDescription* get_type_description(StructQuadSurfMesh::Edge *);
const TypeDescription* get_type_description(StructQuadSurfMesh::Face *);
const TypeDescription* get_type_description(StructQuadSurfMesh::Cell *);

} // namespace SCIRun

#endif // SCI_project_StructQuadSurfMesh_h
