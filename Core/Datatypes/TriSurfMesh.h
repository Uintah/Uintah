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
 *  TriSurfMesh.h: Templated Meshs defined on a 3D Regular Grid
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

#ifndef SCI_project_TriSurfMesh_h
#define SCI_project_TriSurfMesh_h 1

#include <Core/Containers/LockingHandle.h>
#include <Core/Datatypes/Mesh.h>
#include <Core/Datatypes/FieldIterator.h>
#include <Core/Containers/Array1.h>
#include <Core/Math/MusilRNG.h>

#include <vector>

namespace SCIRun {

using std::vector;

class SCICORESHARE TriSurfMesh : public Mesh
{
public:

  typedef int                           under_type;

  //! Index and Iterator types required for Mesh Concept.
  struct Node {
    typedef NodeIndex<under_type>       index_type;
    typedef NodeIterator<under_type>    iterator;
    typedef NodeIndex<under_type>       size_type;
    typedef vector<index_type>          array_type;
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

  typedef Face Elem;

  typedef vector<double>     weight_array;

  TriSurfMesh();
  TriSurfMesh(const TriSurfMesh &copy);
  virtual TriSurfMesh *clone() { return new TriSurfMesh(*this); }
  virtual ~TriSurfMesh();

  virtual BBox get_bounding_box() const;
  virtual void transform(Transform &t);

  template <class I> I tbegin(I*) const;
  template <class I> I tend(I*) const;
  template <class S> S tsize(S*) const;

  Node::iterator node_begin() const;
  Edge::iterator edge_begin() const;
  Face::iterator face_begin() const;
  Cell::iterator cell_begin() const;

  Node::iterator node_end() const;
  Edge::iterator edge_end() const;
  Face::iterator face_end() const;
  Cell::iterator cell_end() const;

  Node::size_type nodes_size() const;
  Edge::size_type edges_size() const;
  Face::size_type faces_size() const;
  Cell::size_type cells_size() const;

  void get_nodes(Node::array_type &array, Edge::index_type idx) const;
  void get_nodes(Node::array_type &array, Face::index_type idx) const;
  void get_nodes(Node::array_type &array, Cell::index_type idx) const;
  void get_edges(Edge::array_type &array, Face::index_type idx) const;

  void get_neighbor(Face::index_type &neighbor, Edge::index_type idx) const;

  void get_center(Point &p, Node::index_type i) const { get_point(p, i); }
  void get_center(Point &p, Edge::index_type i) const;
  void get_center(Point &p, Face::index_type i) const;
  void get_center(Point &, Cell::index_type) const {}

  bool locate(Node::index_type &loc, const Point &p) const;
  bool locate(Edge::index_type &loc, const Point &p) const;
  bool locate(Face::index_type &loc, const Point &p) const;
  bool locate(Cell::index_type &loc, const Point &p) const;

  void get_weights(const Point &p, Node::array_type &l, vector<double> &w);
  void get_weights(const Point &, Edge::array_type &, vector<double> &) {ASSERTFAIL("TriSurfMesh::get_weights for edges isn't supported");}
  void get_weights(const Point &p, Face::array_type &l, vector<double> &w);
  void get_weights(const Point &, Cell::array_type &, vector<double> &) {ASSERTFAIL("TriSurfMesh::get_weights for cells isn't supported");}

  void get_point(Point &result, Node::index_type index) const
  { result = points_[index]; }
  void get_normal(Vector &result, Node::index_type index) const
  { result = normals_[index]; }
  void set_point(const Point &point, Node::index_type index)
  { points_[index] = point; }

  double get_volume(const Cell::index_type &) { return 0; }
  double get_area(const Face::index_type &fi) {
    Node::array_type ra;
    get_nodes(ra,fi);
    Point p0,p1,p2;
    get_point(p0,ra[0]);
    get_point(p1,ra[1]);
    get_point(p2,ra[2]);
    return (Cross(p0-p1,p2-p0)).length()*0.5;
  }

  void get_random_point(Point &p, const Face::index_type &ei, 
			int seed=0) const;

  double get_element_size(const Elem::index_type &fi) { return get_area(fi); }

  virtual void finish_mesh(); // to get normals calculated.
  void compute_normals();
  virtual bool has_normals() const { return true; }

  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  static  const string type_name(int n = -1);
  virtual const string get_type_name(int n = -1) const { return type_name(n); }

  virtual const TypeDescription *get_type_description() const;

  // Extra functionality needed by this specific geometry.

  Node::index_type add_find_point(const Point &p, double err = 1.0e-3);
  void add_triangle(Node::index_type a, Node::index_type b, Node::index_type c);
  void add_triangle(const Point &p0, const Point &p1, const Point &p2);

  // Must call connect after adding triangles this way.
  Node::index_type add_point(const Point &p);
  void add_triangle_unconnected(const Point &p0, const Point &p1,
				const Point &p2);

  void connect(double err = 1.0e-8);


  //bool intersect(const Point &p, const Vector &dir, double &min, double &max,
  //		 Face::index_type &face, double &u, double &v);


  const Point &point(Node::index_type i) { return points_[i]; }

private:

  int next(int i) { return ((i%3)==2) ? (i-2) : (i+1); }
  int prev(int i) { return ((i%3)==0) ? (i+2) : (i-1); }



  bool inside4_p(int, const Point &p);


  vector<Point>  points_;
  vector<int>    faces_;
  vector<int>    neighbors_;
  //! normalized per node normal.
  vector<Vector> normals_;
};

typedef LockingHandle<TriSurfMesh> TriSurfMeshHandle;

template <> TriSurfMesh::Node::size_type TriSurfMesh::tsize(TriSurfMesh::Node::size_type *) const;
template <> TriSurfMesh::Edge::size_type TriSurfMesh::tsize(TriSurfMesh::Edge::size_type *) const;
template <> TriSurfMesh::Face::size_type TriSurfMesh::tsize(TriSurfMesh::Face::size_type *) const;
template <> TriSurfMesh::Cell::size_type TriSurfMesh::tsize(TriSurfMesh::Cell::size_type *) const;
				
template <> TriSurfMesh::Node::iterator TriSurfMesh::tbegin(TriSurfMesh::Node::iterator *) const;
template <> TriSurfMesh::Edge::iterator TriSurfMesh::tbegin(TriSurfMesh::Edge::iterator *) const;
template <> TriSurfMesh::Face::iterator TriSurfMesh::tbegin(TriSurfMesh::Face::iterator *) const;
template <> TriSurfMesh::Cell::iterator TriSurfMesh::tbegin(TriSurfMesh::Cell::iterator *) const;
				
template <> TriSurfMesh::Node::iterator TriSurfMesh::tend(TriSurfMesh::Node::iterator *) const;
template <> TriSurfMesh::Edge::iterator TriSurfMesh::tend(TriSurfMesh::Edge::iterator *) const;
template <> TriSurfMesh::Face::iterator TriSurfMesh::tend(TriSurfMesh::Face::iterator *) const;
template <> TriSurfMesh::Cell::iterator TriSurfMesh::tend(TriSurfMesh::Cell::iterator *) const;

const TypeDescription* get_type_description(TriSurfMesh *);
const TypeDescription* get_type_description(TriSurfMesh::Node *);
const TypeDescription* get_type_description(TriSurfMesh::Edge *);
const TypeDescription* get_type_description(TriSurfMesh::Face *);
const TypeDescription* get_type_description(TriSurfMesh::Cell *);

} // namespace SCIRun


#endif // SCI_project_TriSurfMesh_h
