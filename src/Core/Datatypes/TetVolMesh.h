/*
 *  TetVolMesh.h: Templated Meshs defined on a 3D Regular Grid
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

#ifndef SCI_project_TetVolMesh_h
#define SCI_project_TetVolMesh_h 1

#include <Core/Datatypes/MeshBase.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Datatypes/FieldIterator.h>
#include <vector>
#include <Core/Persistent/PersistentSTL.h>
#include <hash_set>

namespace SCIRun {

using std::hash_set;

class SCICORESHARE TetVolMesh : public MeshBase
{
public:
  typedef int index_type;

  //! Index and Iterator types required for Mesh Concept.
  typedef NodeIndex<index_type>       node_index;
  typedef EdgeIndex<index_type>       edge_index;
  typedef FaceIndex<index_type>       face_index;
  typedef CellIndex<index_type>       cell_index;


  //! Face information.
  struct Face {
    node_index         nodes_[3];   //! 3 nodes makes a face.
    cell_index         cells_[2];   //! 2 cells may have this face is in.
    
    Face();
    // nodes_ must be sorted. See Hash Function below.
    Face(node_index n1, node_index n2, node_index n3);

    inline
    bool shared() const { return ((cells_[0] != -1) && (cells_[1] != -1)); }
    
    //! true if both have the same nodes (order does not matter)
    inline
    bool operator==(const Face &f) const {
      return ((nodes_[0] == f.nodes_[0]) && (nodes_[1] == f.nodes_[1]) && 
	      (nodes_[2] == f.nodes_[2]));
    }
  };

  /*! hash the face's node_indecies such that faces with the same nodes 
   *  hash to the same value. nodes are sorted on face construction. */
  static const int sz_int = sizeof(int) * 8; // in bits
  struct FaceHash {
    static const int sz_third_int = (int)(sz_int * .333333); // in bits
    static const int up3_mask = ((~((int)0)) << sz_third_int << sz_third_int);
    static const int mid3_mask =  up3_mask ^ (~((int)0) << sz_third_int);
    static const int low3_mask = ~(up3_mask | mid3_mask);

    size_t operator()(const Face &f) const {
      return ((f.nodes_[0] << sz_third_int << sz_third_int) | 
	      (mid3_mask & (f.nodes_[1] << sz_third_int)) | 
	      (low3_mask & f.nodes_[2]));
    }
  };


  typedef hash_set<Face, FaceHash> face_table;

  struct HashedFIter : public FaceIterator<index_type> {
    face_table::iterator hiter_;

    HashedFIter() :
      FaceIterator<index_type>() {}
    
    HashedFIter(face_table::iterator h) :
      FaceIterator<index_type>(),
      hiter_(h)
    {}

    bool operator==(const HashedFIter &i) const {
      return (i.hiter_ == hiter_);
    }

    HashedFIter operator ++() { 
      ++hiter_;
      FaceIterator<index_type>::operator++();
      return *this; 
    }
  private:
    //! Hide this in private to prevent it from being called.
    HashedFIter operator ++(int) { 
      HashedFIter tmp = *this; 
      hiter_++; 
      FaceIterator<index_type>::operator++();
      return tmp;
    }
  };

  typedef NodeIterator<index_type>    node_iterator;
  typedef EdgeIterator<index_type>    edge_iterator;
  typedef HashedFIter                 face_iterator;
  typedef CellIterator<index_type>    cell_iterator;

  typedef vector<node_index> node_array;
  typedef vector<edge_index> edge_array;
  typedef vector<face_index> face_array;
  //! type for weights used by locate.
  typedef vector<double>     weight_array;

  TetVolMesh();
  TetVolMesh(const TetVolMesh &copy);
  //TetVolMesh(const MeshRG &lattice);
  virtual ~TetVolMesh();

  virtual BBox get_bounding_box() const;

  node_iterator node_begin() const;
  node_iterator node_end() const;
  index_type    nodes_size() const { return points_.size(); }
  edge_iterator edge_begin() const;
  edge_iterator edge_end() const;
  index_type    edges_size() const { return edges_.size(); }
  face_iterator face_begin() const;
  face_iterator face_end() const;
  //index_type    faces_size() const { return faces_.size(); }
  cell_iterator cell_begin() const;
  cell_iterator cell_end() const;
  index_type    cells_size() const { return cells_.size() >> 2; }

  void get_nodes(node_array &array, edge_index idx) const;
  void get_nodes(node_array &array, face_index idx) const;
  void get_nodes(node_array &array, cell_index idx) const;
  void get_edges(edge_array &array, face_index idx) const;
  void get_edges(edge_array &array, cell_index idx) const;
  void get_faces(face_array &array, cell_index idx) const;
  bool get_neighbor(cell_index &neighbor, cell_index from, 
		   face_index idx) const;
  void get_center(Point &result, node_index idx) const;
  void get_center(Point &result, edge_index idx) const;
  void get_center(Point &result, face_index idx) const;
  void get_center(Point &result, cell_index idx) const;

  //! return false if point is out of range.
  bool locate(node_index &loc, const Point &p) const;
  bool locate(edge_index &loc, const Point &p) const;
  bool locate(face_index &loc, const Point &p) const;
  bool locate(cell_index &loc, const Point &p) const;

  void unlocate(Point &result, const Point &p);

  void get_point(Point &result, node_index index) const;
  
  template <class Iter, class Functor>
  void fill_points(Iter begin, Iter end, Functor fill_ftor);
  template <class Iter, class Functor>
  void fill_cells(Iter begin, Iter end, Functor fill_ftor);
  template <class Iter, class Functor>
  void fill_neighbors(Iter begin, Iter end, Functor fill_ftor);
  template <class Iter, class Functor>
  void fill_data(Iter begin, Iter end, Functor fill_ftor);
  
  //! (re)create the edge and faces data based on cells.
  void finish();
  void compute_edges();
  void compute_faces();
 
  //! Persistent IO
  virtual void io(Piostream&);
  static PersistentTypeID type_id; 

  //! Convenience function to query types. Returns "TetVolMesh" always.
  static  const string type_name(int n = -1);
  virtual const string get_type_name(int n = -1) const { return type_name(n); }

private:

  bool inside4_p(int, const Point &p) const;

  //! all the nodes.
  vector<Point>            points_;
  //! each 4 indecies make up a tet
  vector<index_type>       cells_;
  //! face neighbors index to tet opposite the corresponding node in cells_
  vector<index_type>       neighbors_;

  /*! container for face storage. Must be computed each time 
    nodes or cells change. */
  face_table                faces_;
  
  //! Edge information.
  struct Edge {
    node_index         nodes_[2];   //! 2 nodes makes an edge.
    vector<cell_index> cells_;      //! list of all the cells this edge is in.
    
    Edge() : cells_(6) {
      nodes_[0] = -1;
      nodes_[1] = -1;
    }
    // node_[0] must be smaller than node_[1]. See Hash Function below.
    Edge(node_index n1, node_index n2) : cells_(6) {
      if (n1 < n2) {
	nodes_[0] = n1;
	nodes_[1] = n2; 
      } else {
	nodes_[0] = n2;
	nodes_[1] = n1;
      } 
    }

    bool shared() const { return cells_.size() > 1; }
    
    //! true if both have the same nodes (order does not matter)
    bool operator==(const Edge &e) const {
      return ((nodes_[0] == e.nodes_[0]) && (nodes_[1] == e.nodes_[1]));
    }
  };

  /*! container for edge storage. Must be computed each time 
    nodes or cells change. */
  vector<Edge>         edges_; 
  /*! hash the egde's node_indecies such that edges with the same nodes 
   *  hash to the same value. nodes are sorted on edge construction. */
  struct EdgeHash {
    static const int sz_int = sizeof(int) * 8; // in bits
    static const int sz_half_int = sizeof(int) << 2; // in bits
    static const int up_mask = ((~((int)0)) << sz_half_int);
    static const int low_mask = (~((int)0) ^ up_mask);

    size_t operator()(const Edge &e) const {
      return (e.nodes_[0] << sz_half_int) | (low_mask & e.nodes_[1]);
    }
  };


  inline
  void hash_edge(node_index n1, node_index n2, 
		 cell_index ci, hash_set<Edge, EdgeHash> &table) const;

  inline
  void hash_face(node_index n1, node_index n2, node_index n3,
		 cell_index ci, hash_set<Face, FaceHash> &table) const;
};

// Handle type for TetVolMesh mesh.
typedef LockingHandle<TetVolMesh> TetVolMeshHandle;



template <class Iter, class Functor>
void
TetVolMesh::fill_points(Iter begin, Iter end, Functor fill_ftor) {
  Iter iter = begin;
  points_.resize(end - begin); // resize to the new size
  vector<Point>::iterator piter = points_.begin();
  while (iter != end) {
    *piter = fill_ftor(*iter);
    ++piter; ++iter;
  } 
}

template <class Iter, class Functor>
void
TetVolMesh::fill_cells(Iter begin, Iter end, Functor fill_ftor) {
  Iter iter = begin;
  cells_.resize((end - begin) * 4); // resize to the new size
  vector<index_type>::iterator citer = cells_.begin();
  while (iter != end) {
    int *nodes = fill_ftor(*iter); // returns an array of length 4
    *citer = nodes[0];
    ++citer;
    *citer = nodes[1];
    ++citer;
    *citer = nodes[2];
    ++citer;
    *citer = nodes[3];
    ++citer; ++iter;
  } 
}

template <class Iter, class Functor>
void
TetVolMesh::fill_neighbors(Iter begin, Iter end, Functor fill_ftor) {
  Iter iter = begin;
  neighbors_.resize((end - begin) * 4); // resize to the new size
  vector<index_type>::iterator citer = neighbors_.begin();
  while (iter != end) {
    int *face_nbors = fill_ftor(*iter); // returns an array of length 4
    *citer = face_nbors[0];
    ++citer;
    *citer = face_nbors[1];
    ++citer;
    *citer = face_nbors[2];
    ++citer;
    *citer = face_nbors[3];
    ++citer; ++iter;
  } 
}

} // namespace SCIRun


#endif // SCI_project_TetVolMesh_h
