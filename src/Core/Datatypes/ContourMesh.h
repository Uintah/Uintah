/*
 *  ContourMesh.h: countour mesh
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

#ifndef SCI_project_ContourMesh_h
#define SCI_project_ContourMesh_h 1

#include <Core/Geometry/Point.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Datatypes/MeshBase.h>
#include <Core/Datatypes/FieldIterator.h>
#include <string>
#include <vector>

namespace SCIRun {

using std::string;
using std::vector;

template <class Data>
struct IndexPair {
public:
  IndexPair() {}
  IndexPair(Data first, Data second)
    : first(first), second(second) {}
  IndexPair(const IndexPair &copy) 
    : first(copy.first), second(copy.second) {}
  ~IndexPair() {}
  
  friend void TEMPLATE_TAG Pio TEMPLATE_BOX ( Piostream &, 
					      IndexPair<Data> & ); 
  
  Data first;
  Data second;
};

#define INDEXPAIR_VERSION 1

template <class Data>
void Pio(Piostream &stream, IndexPair<Data> &data)
{
  stream.begin_class("IndexPair", INDEXPAIR_VERSION);

  // IO data members, in order
  Pio(stream,data.first);
  Pio(stream,data.second);

  stream.end_class();  
}
 
class SCICORESHARE ContourMesh : public MeshBase
{
public:
  typedef unsigned index_type;

  //! Index and Iterator types required for Mesh Concept.
  typedef IndexPair<index_type>       index_pair;

  typedef NodeIndex<index_type>       node_index;
  typedef NodeIterator<index_type>    node_iterator;
 
  typedef EdgeIndex<index_type>       edge_index;
  typedef EdgeIterator<index_type>    edge_iterator;
 
  typedef unsigned                    face_index;
  typedef unsigned                    face_iterator;
 
  typedef int                         cell_index;
  typedef int                         cell_iterator;

  // storage types for get_* functions
  typedef vector<node_index>  node_array;
  typedef vector<edge_index>  edge_array;
  typedef vector<face_index>  face_array;
  typedef vector<cell_index>  cell_array;
  typedef vector<double>      weight_array;

  ContourMesh() {}
  ContourMesh(const ContourMesh &copy) 
    : nodes_(copy.nodes_), edges_(copy.edges_) {}
  virtual ~ContourMesh() {}

  node_iterator node_begin() const { return 0; }
  node_iterator node_end() const { return (unsigned)nodes_.size(); }
  edge_iterator edge_begin() const { return 0; }
  edge_iterator edge_end() const { return (unsigned)edges_.size(); }
  face_iterator face_begin() const { return 0; }
  face_iterator face_end() const { return 0; }
  cell_iterator cell_begin() const { return 0; }
  cell_iterator cell_end() const { return 0; }

  //! get the mesh statistics
  unsigned nodes_size() const { return (unsigned)nodes_.size(); }
  unsigned edges_size() const { return (unsigned)edges_.size(); }
  unsigned faces_size() const { return 0; }
  unsigned cells_size() const { return 0; }
  virtual BBox get_bounding_box() const;

  //! set the mesh statistics
  void resize_nodes(unsigned n) { nodes_.resize(n); }
  void resize_edges(unsigned n) { edges_.resize(n); }

  //! get the child elements of the given index
  void get_nodes(node_array &a, edge_index i) const
    { a[0] = edges_[i].first; a[1] = edges_[i].second; } 
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
    { result = nodes_[idx]; }
  void get_center(Point &, edge_index) const {}
  void get_center(Point &, face_index) const {}
  void get_center(Point &, cell_index) const {}

  bool locate(node_index &, const Point &) const;
  bool locate(edge_index &, const Point &) const;
  bool locate(face_index &, const Point &) const { return false; }
  bool locate(cell_index &, const Point &) const { return false; }

  void unlocate(Point &result, const Point &p) const { result =  p; };

  void get_point(Point &result, node_index idx) const
    { get_center(result,idx); }

  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  static  const string type_name(int n = -1);
  virtual const string get_type_name(int n = -1) const { return type_name(n); }

private:

  //! the nodes
  vector<Point> nodes_;

  //! the edges
  vector<index_pair> edges_;

  // returns a ContourMesh
  static Persistent *maker() { return new ContourMesh(); }
};

typedef LockingHandle<ContourMesh> ContourMeshHandle;



} // namespace SCIRun

#endif // SCI_project_ContourMesh_h





