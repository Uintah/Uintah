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

#include <Core/Geometry/Point.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Datatypes/MeshBase.h>
#include <Core/Datatypes/FieldIterator.h>
#include <Core/share/share.h>
#include <string>
#include <iostream>

namespace SCIRun {

using std::string;

class SCICORESHARE ScanlineMesh : public MeshBase
{
public:

  typedef unsigned int    index_type;
  
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
  typedef vector<double>      weight_array;

  ScanlineMesh()
    : offset_(0), length_(0), min_(Point(0,0,0)), max_(Point(1,1,1)) {}
  ScanlineMesh(unsigned int length, const Point &min, const Point &max) 
    : offset_(0), length_(length), min_(min), max_(max) {}
  ScanlineMesh(ScanlineMesh* mh, unsigned int offset, unsigned int length)
    : offset_(offset), length_(length), min_(mh->min_), max_(mh->max_) {}
  ScanlineMesh(const ScanlineMesh &copy)
    : offset_(copy.offset_), length_(copy.get_length()),
      min_(copy.get_min()), max_(copy.get_max()) {}
  virtual ScanlineMesh *clone() { return new ScanlineMesh(*this); }
  virtual ~ScanlineMesh() {}

  node_index  node(unsigned int i) const { return node_index(i); }
  node_iterator  node_begin() const { return node_iterator(offset_); }
  node_iterator  node_end() const { return node_iterator(offset_ + length_); }
  node_size_type nodes_size() const { return node_size_type(length_); }

  edge_index edge(unsigned int i) const { return edge_index(i); }
  edge_iterator  edge_begin() const { return edge_iterator(offset_); }
  edge_iterator  edge_end() const { return edge_iterator(offset_+length_-1); }
  edge_size_type edges_size() const { return edge_size_type(length_ - 1); }

  face_iterator  face_begin() const { return face_iterator(0); }
  face_iterator  face_end() const { return face_iterator(0); }
  face_size_type faces_size() const { return face_size_type(0); }

  cell_iterator  cell_begin() const { return cell_iterator(0); }
  cell_iterator  cell_end() const { return cell_iterator(0); }
  cell_size_type cells_size() const { return cell_size_type(0); }

  //! get the mesh statistics
  unsigned get_length() const { return length_; }
  Point get_min() const { return min_; }
  Point get_max() const { return max_; }
  Vector diagonal() const { return max_ - min_; }
  virtual BBox get_bounding_box() const;

  //! set the mesh statistics
  void set_offset(unsigned int x) { offset_ = x; }
  void set_length(unsigned int x) { length_ = x; }
  void set_min(Point p) { min_ = p; }
  void set_max(Point p) { max_ = p; }


  //! get the child elements of the given index
  void get_nodes(node_array &, edge_index) const;
  void get_nodes(node_array &, face_index) const {}
  void get_nodes(node_array &, cell_index) const {}
  void get_edges(edge_array &, face_index) const {}
  void get_edges(edge_array &, cell_index) const {}
  //void get_faces(face_array &, cell_index) const {}

  //! get the parent element(s) of the given index
  void get_edges(edge_array &a, node_index idx) const
  { a.push_back(edge_index(idx));}
  //bool get_faces(face_array &, node_index) const { return 0; }
  //bool get_faces(face_array &, edge_index) const { return 0; }
  //bool get_cells(cell_array &, node_index) const { return 0; }
  //bool get_cells(cell_array &, edge_index) const { return 0; }
  //bool get_cells(cell_array &, face_index) const { return 0; }

  //! return all edge_indecies that overlap the BBox in arr.
  void get_edges(edge_array &arr, const BBox &box) const;

  //! similar to get_cells() with face_index argument, but
  //  returns the "other" cell if it exists, not all that exist
  bool get_neighbor(cell_index & /*neighbor*/, cell_index /*from*/, 
		    face_index /*idx*/) const {
    ASSERTFAIL("ScanlineMesh::get_neighbor not implemented.");
  }
  //! get the center point (in object space) of an element
  void get_center(Point &, node_index) const;
  void get_center(Point &, edge_index) const;
  void get_center(Point &, face_index) const {}
  void get_center(Point &, cell_index) const {}

  bool locate(node_index &, const Point &) const;
  bool locate(edge_index &, const Point &) const;
  bool locate(face_index &, const Point &) const { return false; }
  bool locate(cell_index &, const Point &) const { return false; }

  void get_point(Point &p, node_index i) const
  { get_center(p, i); }
  void get_normal(Vector &/* result */, node_index /* index */) const
  { ASSERTFAIL("not implemented") }

  
  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  static  const string type_name(int n = -1);
  virtual const string get_type_name(int n = -1) const { return type_name(n); }

private:

  //! the min_node_index ( incase this is a subLattice )
  unsigned int offset_;

  //! the node_index space extents of a ScanlineMesh 
  //! (min=min_node_index, max=min+extents-1)
  unsigned int length_;

  //! the object space extents of a ScanlineMesh
  Point min_, max_;

  // returns a ScanlineMesh
  static Persistent *maker() { return new ScanlineMesh(); }
};

typedef LockingHandle<ScanlineMesh> ScanlineMeshHandle;



} // namespace SCIRun

#endif // SCI_project_ScanlineMesh_h
