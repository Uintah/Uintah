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
#include <Core/Datatypes/MeshBase.h>
#include <Core/Datatypes/FieldIterator.h>
#include <Core/Containers/Array1.h>
#include <Core/Math/MusilRNG.h>

#include <vector>

namespace SCIRun {

using std::vector;

class SCICORESHARE TriSurfMesh : public MeshBase
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

  typedef Face::index_type                  elem_index;
  typedef Face::iterator               elem_iterator;
  
  typedef vector<double>     weight_array;

  TriSurfMesh();
  TriSurfMesh(const TriSurfMesh &copy);
  virtual TriSurfMesh *clone() { return new TriSurfMesh(*this); }
  virtual ~TriSurfMesh();

  virtual BBox get_bounding_box() const;

  Node::iterator node_begin() const;
  Node::iterator node_end() const;
  Edge::iterator edge_begin() const;
  Edge::iterator edge_end() const;
  Face::iterator face_begin() const;
  Face::iterator face_end() const;
  Cell::iterator cell_begin() const;
  Cell::iterator cell_end() const;
  elem_iterator elem_begin() const;
  elem_iterator elem_end() const;

  Node::size_type nodes_size() { return *node_end(); }
  Edge::size_type edges_size() { return *edge_end(); }
  Face::size_type faces_size() { return *face_end(); }
  Cell::size_type cells_size() { return *cell_end(); }

  void get_nodes(Node::array_type &array, Edge::index_type idx) const;
  void get_nodes(Node::array_type &array, Face::index_type idx) const;
  void get_nodes(Node::array_type &array, Cell::index_type idx) const;
  void get_edges(Edge::array_type &array, Face::index_type idx) const;
  //void get_edges_from_cell(Edge::array_type &array, Cell::index_type idx) const;
  //void get_faces_from_cell(Face::array_type &array, Cell::index_type idx) const;

  void get_neighbor(Face::index_type &neighbor, Edge::index_type idx) const;

  bool locate(Node::index_type &loc, const Point &p) const;
  bool locate(Edge::index_type &loc, const Point &p) const;
  bool locate(Face::index_type &loc, const Point &p) const;
  bool locate(Cell::index_type &loc, const Point &p) const;

  void get_center(Point &p, Node::index_type i) const { get_point(p, i); }
  void get_center(Point &p, Edge::index_type i) const;
  void get_center(Point &p, Face::index_type i) const;
  void get_center(Point &, Cell::index_type) const {}

  void get_point(Point &result, Node::index_type index) const
  { result = points_[index]; }
  void get_normal(Vector &result, Node::index_type index) const
  { result = normals_[index]; }
  void set_point(const Point &point, Node::index_type index)
  { points_[index] = point; }

  double get_volume(Cell::index_type &) { return 0; }
  double get_area(Face::index_type &fi) {
    Node::array_type ra; 
    get_nodes(ra,fi);
    return (Cross(ra[1]-ra[0],ra[2]-ra[0])).length2()*0.5;
  }

  void get_random_point(Point &p, const elem_index &ei) const {
    static MusilRNG rng(1249);
    node_array ra;
    get_nodes(ra,ei);
    Point p0,p1,p2;
    get_point(p0,ra[0]);
    get_point(p1,ra[1]);
    get_point(p2,ra[2]);
    Vector v0 = ra[1]-ra[0];
    Vector v1 = ra[2]-ra[0];
    double t = rng()*v0.length2();
    double u = rng()*v1.length2();
    if ( (t+u)>1 ) {
      t = 1.-t;
      u = 1.-u;
    }
    p = p0+(v0*t)+(v1*u);
  }
  
  double get_element_size(Face::index_type &fi) { return get_area(fi); }

  virtual void finish_mesh(); // to get normals calculated.
  void compute_normals();
  virtual bool has_normals() const { return true; }

  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  static  const string type_name(int n = -1);
  virtual const string get_type_name(int n = -1) const { return type_name(n); }

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

} // namespace SCIRun


#endif // SCI_project_TriSurfMesh_h
