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

  typedef unsigned int    under_type;
  
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

  // storage types for get_* functions
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

  Node::index_type  node(unsigned int i) const { return Node::index_type(i); }
  Node::iterator  node_begin() const { return Node::iterator(offset_); }
  Node::iterator  node_end() const { return Node::iterator(offset_ + length_); }
  Node::size_type nodes_size() const { return Node::size_type(length_); }

  Edge::index_type edge(unsigned int i) const { return Edge::index_type(i); }
  Edge::iterator  edge_begin() const { return Edge::iterator(offset_); }
  Edge::iterator  edge_end() const { return Edge::iterator(offset_+length_-1); }
  Edge::size_type edges_size() const { return Edge::size_type(length_ - 1); }

  Face::iterator  face_begin() const { return Face::iterator(0); }
  Face::iterator  face_end() const { return Face::iterator(0); }
  Face::size_type faces_size() const { return Face::size_type(0); }

  Cell::iterator  cell_begin() const { return Cell::iterator(0); }
  Cell::iterator  cell_end() const { return Cell::iterator(0); }
  Cell::size_type cells_size() const { return Cell::size_type(0); }

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
  void get_nodes(Node::array_type &, Edge::index_type) const;
  void get_nodes(Node::array_type &, Face::index_type) const {}
  void get_nodes(Node::array_type &, Cell::index_type) const {}
  void get_edges(Edge::array_type &, Face::index_type) const {}
  void get_edges(Edge::array_type &, Cell::index_type) const {}
  //void get_faces(Face::array_type &, Cell::index_type) const {}

  //! get the parent element(s) of the given index
  void get_edges(Edge::array_type &a, Node::index_type idx) const
  { a.push_back(Edge::index_type(idx));}
  //bool get_faces(Face::array_type &, Node::index_type) const { return 0; }
  //bool get_faces(Face::array_type &, Edge::index_type) const { return 0; }
  //bool get_cells(Cell::array_type &, Node::index_type) const { return 0; }
  //bool get_cells(Cell::array_type &, Edge::index_type) const { return 0; }
  //bool get_cells(Cell::array_type &, Face::index_type) const { return 0; }

  //! return all edge_indecies that overlap the BBox in arr.
  void get_edges(Edge::array_type &arr, const BBox &box) const;

  //! similar to get_cells() with Face::index_type argument, but
  //  returns the "other" cell if it exists, not all that exist
  bool get_neighbor(Cell::index_type & /*neighbor*/, Cell::index_type /*from*/, 
		    Face::index_type /*idx*/) const {
    ASSERTFAIL("ScanlineMesh::get_neighbor not implemented.");
  }
  //! get the center point (in object space) of an element
  void get_center(Point &, Node::index_type) const;
  void get_center(Point &, Edge::index_type) const;
  void get_center(Point &, Face::index_type) const {}
  void get_center(Point &, Cell::index_type) const {}

  bool locate(Node::index_type &, const Point &) const;
  bool locate(Edge::index_type &, const Point &) const;
  bool locate(Face::index_type &, const Point &) const { return false; }
  bool locate(Cell::index_type &, const Point &) const { return false; }

  void get_point(Point &p, Node::index_type i) const
  { get_center(p, i); }
  void get_normal(Vector &/* result */, Node::index_type /* index */) const
  { ASSERTFAIL("not implemented") }

  
  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  static  const string type_name(int n = -1);
  virtual const string get_type_name(int n = -1) const { return type_name(n); }

private:

  //! the min_Node::index_type ( incase this is a subLattice )
  unsigned int offset_;

  //! the Node::index_type space extents of a ScanlineMesh 
  //! (min=min_Node::index_type, max=min+extents-1)
  unsigned int length_;

  //! the object space extents of a ScanlineMesh
  Point min_, max_;

  // returns a ScanlineMesh
  static Persistent *maker() { return new ScanlineMesh(); }
};

typedef LockingHandle<ScanlineMesh> ScanlineMeshHandle;



} // namespace SCIRun

#endif // SCI_project_ScanlineMesh_h
