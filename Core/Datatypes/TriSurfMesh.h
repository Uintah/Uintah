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

#include <vector>

namespace SCIRun {

using std::vector;

class SCICORESHARE TriSurfMesh : public MeshBase
{
public:

  typedef int                         index_type;

  //! Index and Iterator types required for Mesh Concept.
  typedef NodeIndex<index_type>       node_index;
  typedef NodeIterator<index_type>    node_iterator;

  typedef EdgeIndex<index_type>       edge_index;
  typedef EdgeIterator<index_type>    edge_iterator;

  typedef FaceIndex<index_type>       face_index;
  typedef FaceIterator<index_type>    face_iterator;

  typedef CellIndex<index_type>       cell_index;
  typedef CellIterator<index_type>    cell_iterator;

  
  typedef vector<node_index> node_array;
  typedef vector<edge_index> edge_array;
  typedef vector<double>     weight_array;
  //typedef vector<face_index> face_array;

  TriSurfMesh();
  TriSurfMesh(const TriSurfMesh &copy);
  virtual TriSurfMesh *clone() { return new TriSurfMesh(*this); }
  virtual ~TriSurfMesh();

  virtual BBox get_bounding_box() const;

  node_iterator node_begin() const;
  node_iterator node_end() const;
  edge_iterator edge_begin() const;
  edge_iterator edge_end() const;
  face_iterator face_begin() const;
  face_iterator face_end() const;
  cell_iterator cell_begin() const;
  cell_iterator cell_end() const;

  node_index nodes_size() { return *node_end(); }
  edge_index edges_size() { return *edge_end(); }
  face_index faces_size() { return *face_end(); }
  cell_index cells_size() { return *cell_end(); }

  void get_nodes(node_array &array, edge_index idx) const;
  void get_nodes(node_array &array, face_index idx) const;
  void get_nodes(node_array &array, cell_index idx) const;
  void get_edges(edge_array &array, face_index idx) const;
  //void get_edges_from_cell(edge_array &array, cell_index idx) const;
  //void get_faces_from_cell(face_array &array, cell_index idx) const;

  void get_neighbor(face_index &neighbor, edge_index idx) const;

  bool locate(node_index &loc, const Point &p) const;
  bool locate(edge_index &loc, const Point &p) const;
  bool locate(face_index &loc, const Point &p) const;
  bool locate(cell_index &loc, const Point &p) const;

  void get_center(Point &p, node_index i) const { get_point(p, i); }
  void get_center(Point &p, edge_index i) const;
  void get_center(Point &p, face_index i) const;
  void get_center(Point &, cell_index) const {}

  void get_point(Point &result, node_index index) const
  { result = points_[index]; }
  void get_normal(Vector &result, node_index index) const
  { result = normals_[index]; }
  void set_point(const Point &point, node_index index)
  { points_[index] = point; }

  double get_volume(cell_index &) { return 0; }
  double get_area(face_index &fi) {
    node_array ra; 
    get_nodes(ra,fi);
    return (Cross(ra[1]-ra[0],ra[2]-ra[0])).length2()*0.5;
  }
  double get_element_size(face_index &fi) { return get_area(fi); }

  virtual void finish_mesh(); // to get normals calculated.
  void compute_normals();
  virtual bool has_normals() const { return true; }

  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  static  const string type_name(int n = -1);
  virtual const string get_type_name(int n = -1) const { return type_name(n); }

  // Extra functionality needed by this specific geometry.

  node_index add_find_point(const Point &p, double err = 1.0e-3);
  void add_triangle(node_index a, node_index b, node_index c);
  void add_triangle(const Point &p0, const Point &p1, const Point &p2);

  // Must call connect after adding triangles this way.
  node_index add_point(const Point &p);
  void add_triangle_unconnected(const Point &p0, const Point &p1,
				const Point &p2);

  void connect(double err = 1.0e-8);


  //bool intersect(const Point &p, const Vector &dir, double &min, double &max,
  //		 face_index &face, double &u, double &v);


  const Point &point(node_index i) { return points_[i]; }

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
