/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.

  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.

  The Original Source Code is SCIRun, released March 12, 2001.

  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994
  University of Utah. All Rights Reserved.
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
  virtual void transform(Transform &t);

  //! get the child elements of the given index
  void get_nodes(Node::array_type &, Edge::index_type) const;
  void get_nodes(Node::array_type &, Face::index_type) const;
  void get_nodes(Node::array_type &, Cell::index_type) const {}
  void get_edges(Edge::array_type &, Face::index_type) const;
  void get_edges(Edge::array_type &, Cell::index_type) const {}
  void get_faces(Face::array_type &, Cell::index_type) const {}

  //! return all face_indecies that overlap the BBox in arr.
  void get_faces(Face::array_type &arr, const BBox &box)
  { ASSERTFAIL("ScanlineMesh::get_faces for BBox is not implemented."); }

  //! similar to get_faces() with Face::index_type argument, but
  //  returns the "other" face if it exists, not all that exist
  bool get_neighbor(Face::index_type & /*neighbor*/, Face::index_type /*from*/,
		    Edge::index_type /*idx*/) const {
    ASSERTFAIL("StructQuadSurfMesh::get_neighbor not implemented.");
  }

  //! get the center point (in object space) of an element
  void get_center(Point &, const Node::index_type) const;
  void get_center(Point &, const Edge::index_type) const;
  void get_center(Point &, const Face::index_type) const;
  void get_center(Point &, const Cell::index_type) const {}

  bool locate(Node::index_type &, const Point &) const;
  bool locate(Edge::index_type &, const Point &) const;
  bool locate(Face::index_type &, const Point &) const;
  bool locate(Cell::index_type &, const Point &) const;

  void get_weights(const Point &p, Node::array_type &l, vector<double> &w);
  void get_weights(const Point &p, Edge::array_type &l, vector<double> &w)
  {ASSERTFAIL("StructQuadSurfMesh::get_weights for edges isn't supported");}
  void get_weights(const Point &p, Face::array_type &l, vector<double> &w);
  void get_weights(const Point &p, Cell::array_type &l, vector<double> &w)
  {ASSERTFAIL("StructQuadSurfMesh::get_weights for cells isn't supported");}

  void get_point(Point &point, Node::index_type index) const
  { get_center(point, index); }
  void set_point(const Node::index_type &index, const Point &point);

  void get_normal(Vector &vector, Node::index_type index) const
  { ASSERTFAIL("not implemented") }

  void get_random_point(Point &p, const Elem::index_type &ei, int seed=0) const
  { ASSERTFAIL("not implemented") }

  double get_area(const Face::index_type &fi) {
    Node::array_type ra;
    get_nodes(ra,fi);
    Point p0,p1,p2;
    get_point(p0,ra[0]);
    get_point(p1,ra[1]);
    get_point(p2,ra[2]);
    return (Cross(p0-p1,p2-p0)).length()*0.5;
  }

  double get_element_size(const Elem::index_type &fi) { return get_area(fi); }

  virtual bool has_normals() const { return true; }
  virtual bool is_editable() const { return true; }

  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  static  const string type_name(int n = -1);
  virtual const TypeDescription *get_type_description() const;

  virtual bool synchronize(unsigned int);

private:
  void compute_normals();
  void compute_edge_neighbors(double err = 1.0e-8);

  int next(int i) { return ((i%4)==3) ? (i-3) : (i+1); }
  int prev(int i) { return ((i%4)==0) ? (i+3) : (i-1); }

  //bool inside4_p(int, const Point &p);

  Array2<Point>  points_;
  Array2<Vector> normals_; //! normalized per node
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
