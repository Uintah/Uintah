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
 *  PointCloudMesh.h: countour mesh
 *
 *  Written by:
 *   Chris Moulding
 *   Department of Computer Science
 *   University of Utah
 *   January 2001
 *
 *  Copyright (C) 2001 SCI Group
 *
 */

#ifndef SCI_project_PointCloudMesh_h
#define SCI_project_PointCloudMesh_h 1

#include <Core/Geometry/Point.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Datatypes/MeshBase.h>
#include <Core/Datatypes/FieldIterator.h>
#include <string>
#include <vector>

namespace SCIRun {

using std::string;
using std::vector;
 
class SCICORESHARE PointCloudMesh : public MeshBase
{
public:
  typedef unsigned index_type;

  //! Index and Iterator types required for Mesh Concept.
  typedef NodeIndex<index_type>       node_index;
  typedef NodeIterator<index_type>    node_iterator;
  typedef NodeIndex<index_type>       node_size_type;
 
  typedef EdgeIndex<index_type>       edge_index;
  typedef EdgeIterator<index_type>    edge_iterator;
  typedef EdgeIndex<index_type>       edge_size_type;
 
  typedef FaceIndex<index_type>       face_index;
  typedef FaceIterator<index_type>    face_iterator;
  typedef FaceIndex<index_type>       face_size_type;
 
  typedef CellIndex<index_type>       cell_index;
  typedef CellIterator<index_type>    cell_iterator;
  typedef CellIndex<index_type>       cell_size_type;

  // storage types for get_* functions
  typedef vector<node_index>  node_array;
  typedef vector<edge_index>  edge_array;
  typedef vector<face_index>  face_array;
  typedef vector<cell_index>  cell_array;
  typedef vector<double>      weight_array;

  PointCloudMesh() {}
  PointCloudMesh(const PointCloudMesh &copy) 
    : points_(copy.points_) {}
  virtual MeshBase *clone() { return new PointCloudMesh(*this); }
  virtual ~PointCloudMesh() {}

  node_iterator node_begin() const { return 0; }
  node_iterator node_end() const { return (unsigned)points_.size(); }
  edge_iterator edge_begin() const { return 0; }
  edge_iterator edge_end() const { return 0; }
  face_iterator face_begin() const { return 0; }
  face_iterator face_end() const { return 0; }
  cell_iterator cell_begin() const { return 0; }
  cell_iterator cell_end() const { return 0; }

  //! get the mesh statistics
  node_size_type nodes_size() const { return (unsigned)points_.size(); }
  edge_size_type edges_size() const { return 0; }
  face_size_type faces_size() const { return 0; }
  cell_size_type cells_size() const { return 0; }
  virtual BBox get_bounding_box() const;

  //! set the mesh statistics
  void resize_nodes(node_size_type n) { points_.resize(n); }

  //! get the child elements of the given index
  void get_nodes(node_array &, edge_index) const {}
  void get_nodes(node_array &, face_index) const {}
  void get_nodes(node_array &, cell_index) const {}
  void get_edges(edge_array &, face_index) const {}
  void get_edges(edge_array &, cell_index) const {}
  void get_faces(face_array &, cell_index) const {}

  //! get the parent element(s) of the given index
  unsigned get_edges(edge_array &, node_index) const { return 0; }
  unsigned get_faces(face_array &, node_index) const { return 0; }
  unsigned get_faces(face_array &, edge_index) const { return 0; }
  unsigned get_cells(cell_array &, node_index) const { return 0; }
  unsigned get_cells(cell_array &, edge_index) const { return 0; }
  unsigned get_cells(cell_array &, face_index) const { return 0; }

  //! similar to get_edges() with node_index argument, but
  //  returns the "other" edge if it exists, not all that exist
  void get_neighbor(edge_index &, node_index) const {}

  //! get the center point (in object space) of an element
  void get_center(Point &result, node_index idx) const
  { result = points_[idx]; }
  void get_center(Point &, edge_index) const {}
  void get_center(Point &, face_index) const {}
  void get_center(Point &, cell_index) const {}

  bool locate(node_index &, const Point &) const;
  bool locate(edge_index &, const Point &) const { return false; }
  bool locate(face_index &, const Point &) const { return false; }
  bool locate(cell_index &, const Point &) const { return false; }

  void unlocate(Point &result, const Point &p) const { result =  p; };

  void get_point(Point &result, node_index idx) const
  { get_center(result,idx); }
  void set_point(const Point &point, node_index index)
  { points_[index] = point; }

  //! use these to build up a new PointCloud mesh
  node_index add_node(Point p) 
  { points_.push_back(p); return points_.size()-1; }

  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  static  const string type_name(int n = -1);
  virtual const string get_type_name(int n = -1) const { return type_name(n); }
  node_index add_point(const Point &p);

private:

  //! the nodes
  vector<Point> points_;

  // returns a PointCloudMesh
  static Persistent *maker() { return new PointCloudMesh(); }
};

typedef LockingHandle<PointCloudMesh> PointCloudMeshHandle;



} // namespace SCIRun

#endif // SCI_project_PointCloudMesh_h





