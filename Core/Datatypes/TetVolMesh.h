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

#include <Core/Thread/Mutex.h>
#include <Core/Datatypes/Mesh.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Datatypes/FieldIterator.h>
#include <Core/Datatypes/LatticeVol.h>
#include <vector>
#include <Core/Persistent/PersistentSTL.h>
#include <sci_hash_map.h>

namespace SCIRun {

using std::hash_map;

class SCICORESHARE TetVolMesh : public Mesh
{
public:
  typedef int under_type;

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

  typedef Cell Elem;

  typedef vector<double>     weight_array;

  TetVolMesh();
  TetVolMesh(const TetVolMesh &copy);
  //TetVolMesh(const MeshRG &lattice);
  virtual TetVolMesh *clone() { return new TetVolMesh(*this); }
  virtual ~TetVolMesh();

  virtual BBox get_bounding_box() const;
  virtual void transform(Transform &t);

  template <class I> I tbegin(I*) const;
  template <class I> I tend(I*) const;
  template <class S> S tsize(S*) const;

  Node::iterator  node_begin() const;
  Node::iterator  node_end() const;
  Node::size_type nodes_size() const;
  Edge::iterator  edge_begin() const;
  Edge::iterator  edge_end() const;
  Edge::size_type edges_size() const;
  Face::iterator  face_begin() const;
  Face::iterator  face_end() const;
  Face::size_type faces_size() const;
  Cell::iterator  cell_begin() const;
  Cell::iterator  cell_end() const;
  Cell::size_type cells_size() const;

  void get_nodes(Node::array_type &array, Edge::index_type idx) const;
  void get_nodes(Node::array_type &array, Face::index_type idx) const;
  void get_nodes(Node::array_type &array, Cell::index_type idx) const;
  void get_edges(Edge::array_type &array, Face::index_type idx) const;
  void get_edges(Edge::array_type &array, Cell::index_type idx) const;
  void get_faces(Face::array_type &array, Cell::index_type idx) const;
  bool get_neighbor(Cell::index_type &neighbor, Cell::index_type from,
		   Face::index_type idx) const;
  void get_neighbors(Cell::array_type &array, Cell::index_type idx) const;
  //! must call compute_node_neighbors before calling get_neighbors.
  void get_neighbors(Node::array_type &array, Node::index_type idx) const;
  void get_center(Point &result, Node::index_type idx) const;
  void get_center(Point &result, Edge::index_type idx) const;
  void get_center(Point &result, Face::index_type idx) const;
  void get_center(Point &result, Cell::index_type idx) const;

  //! return false if point is out of range.
  bool locate(Node::index_type &loc, const Point &p);
  bool locate(Edge::index_type &loc, const Point &p);
  bool locate(Face::index_type &loc, const Point &p);
  bool locate(Cell::index_type &loc, const Point &p);

  void get_weights(const Point &p, Node::array_type &l, vector<double> &w);
  void get_weights(const Point &, Edge::array_type &, vector<double> &) {ASSERTFAIL("TetVolMesh::get_weights for edges isn't supported");}
  void get_weights(const Point &, Face::array_type &, vector<double> &) {ASSERTFAIL("TetVolMesh::get_weights for faces isn't supported");}
  void get_weights(const Point &p, Cell::array_type &l, vector<double> &w);

  void get_point(Point &result, Node::index_type index) const
  { result = points_[index]; }
  void get_normal(Vector &/* result */, Node::index_type /* index */) const
  { ASSERTFAIL("not implemented") }

  void set_point(const Point &point, Node::index_type index)
  { points_[index] = point; }

  double get_volume(const Cell::index_type &ci) {
    Node::array_type ra(4);
    get_nodes(ra,ci);
    Point p0,p1,p2,p3;
    get_point(p0,ra[0]);
    get_point(p1,ra[1]);
    get_point(p2,ra[2]);
    get_point(p3,ra[3]);
    return (Cross(Cross(p1-p0,p2-p0),p3-p0)).length()*0.1666666666666666;
  }
  double get_area(const Face::index_type &) { return 0; }
  double get_element_size(const Elem::index_type &ci)
  { return get_volume(ci); }

  void get_random_point(Point &p, const Cell::index_type &ei, 
			int seed=0) const;

  //! the double return val is the volume of the tet.
  double get_gradient_basis(Cell::index_type ci, Vector& g0, Vector& g1,
			    Vector& g2, Vector& g3);

  //! function to test if at least one of cell's nodes are in supplied range
  inline bool test_nodes_range(Cell::index_type ci, int sn, int en){
    if (cells_[ci*4]>=sn && cells_[ci*4]<en
	|| cells_[ci*4+1]>=sn && cells_[ci*4+1]<en
	|| cells_[ci*4+2]>=sn && cells_[ci*4+2]<en
	|| cells_[ci*4+3]>=sn && cells_[ci*4+3]<en)
      return true;
    else
      return false;
  }
  template <class Iter, class Functor>
  void fill_points(Iter begin, Iter end, Functor fill_ftor);
  template <class Iter, class Functor>
  void fill_cells(Iter begin, Iter end, Functor fill_ftor);
  template <class Iter, class Functor>
  void fill_neighbors(Iter begin, Iter end, Functor fill_ftor);
  template <class Iter, class Functor>
  void fill_data(Iter begin, Iter end, Functor fill_ftor);

  //! (re)create the edge and faces data based on cells.
  virtual void flush_changes();
  void compute_edges();
  void compute_faces();
  void compute_node_neighbors();
  void compute_grid();

  //! Persistent IO
  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  static  const string type_name(int n = -1);
  virtual const string get_type_name(int n = -1) const { return type_name(n); }

  virtual const TypeDescription *get_type_description() const;

  // Extra functionality needed by this specific geometry.

  Node::index_type add_find_point(const Point &p, double err = 1.0e-3);
  void add_tet(Node::index_type a, Node::index_type b, Node::index_type c, Node::index_type d);
  void add_tet(const Point &p0, const Point &p1, const Point &p2,
	       const Point &p3);

  // Must call connect after adding tets this way.
  Node::index_type add_point(const Point &p);
  void add_tet_unconnected(const Point &p0, const Point &p1,
			   const Point &p2, const Point &p3);

  void connect(double err = 1.0e-3);


  //bool intersect(const Point &p, const Vector &dir, double &min, double &max,
  //		 Face::index_type &face, double &u, double &v);


  const Point &point(Node::index_type i) { return points_[i]; }
private:

  bool inside4_p(int, const Point &p) const;

  //! all the nodes.
  vector<Point>        points_;
  Mutex                points_lock_;

  //! each 4 indecies make up a tet
  vector<under_type>   cells_;
  Mutex                cells_lock_;
  //! face neighbors index to tet opposite the corresponding node in cells_
  vector<under_type>   neighbors_;
  Mutex                nbors_lock_;
  //! Face information.
  struct PFace {
    Node::index_type         nodes_[3];   //! 3 nodes makes a face.
    Cell::index_type         cells_[2];   //! 2 cells may have this face is in.

    PFace() {
      nodes_[0] = -1;
      nodes_[1] = -1;
      nodes_[2] = -1;
      cells_[0] = -1;
      cells_[1] = -1;
    }
    // nodes_ must be sorted. See Hash Function below.
    PFace(Node::index_type n1, Node::index_type n2, Node::index_type n3) {
      cells_[0] = -1;
      cells_[1] = -1;
      if ((n1 < n2) && (n1 < n3)) {
	nodes_[0] = n1;
	if (n2 < n3) {
	  nodes_[1] = n2;
	  nodes_[2] = n3;
	} else {
	  nodes_[1] = n3;
	  nodes_[2] = n2;
	}
      } else if ((n2 < n1) && (n2 < n3)) {
	nodes_[0] = n2;
	if (n1 < n3) {
	  nodes_[1] = n1;
	  nodes_[2] = n3;
	} else {
	  nodes_[1] = n3;
	  nodes_[2] = n1;
	}
      } else {
	nodes_[0] = n3;
	if (n1 < n2) {
	  nodes_[1] = n1;
	  nodes_[2] = n2;
	} else {
	  nodes_[1] = n2;
	  nodes_[2] = n1;
	}
      }
    }

    bool shared() const { return ((cells_[0] != -1) && (cells_[1] != -1)); }

    //! true if both have the same nodes (order does not matter)
    bool operator==(const PFace &f) const {
      return ((nodes_[0] == f.nodes_[0]) && (nodes_[1] == f.nodes_[1]) &&
	      (nodes_[2] == f.nodes_[2]));
    }
  };


  //! Edge information.
  struct PEdge {
    Node::index_type         nodes_[2];   //! 2 nodes makes an edge.
    vector<Cell::index_type> cells_;      //! list of all the cells this edge is in.

    PEdge() : cells_(0) {
      nodes_[0] = -1;
      nodes_[1] = -1;
    }
    // node_[0] must be smaller than node_[1]. See Hash Function below.
    PEdge(Node::index_type n1, Node::index_type n2) : cells_(0) {
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
    bool operator==(const PEdge &e) const {
      return ((nodes_[0] == e.nodes_[0]) && (nodes_[1] == e.nodes_[1]));
    }
  };

  /*! hash the egde's node_indecies such that edges with the same nodes
   *  hash to the same value. nodes are sorted on edge construction. */
  static const int sz_int = sizeof(int) * 8; // in bits
  struct FaceHash {
    static const int sz_third_int = (int)(sz_int * .333333); // in bits
    static const int up3_mask = ((~((int)0)) << sz_third_int << sz_third_int);
    static const int mid3_mask =  up3_mask ^ (~((int)0) << sz_third_int);
    static const int low3_mask = ~(up3_mask | mid3_mask);

    size_t operator()(const PFace &f) const {
      return ((f.nodes_[0] << sz_third_int << sz_third_int) |
	      (mid3_mask & (f.nodes_[1] << sz_third_int)) |
	      (low3_mask & f.nodes_[2]));
    }
  };

  /*! hash the egde's node_indecies such that edges with the same nodes
   *  hash to the same value. nodes are sorted on edge construction. */
  struct EdgeHash {
    static const int sz_int = sizeof(int) * 8; // in bits
    static const int sz_half_int = sizeof(int) << 2; // in bits
    static const int up_mask = ((~((int)0)) << sz_half_int);
    static const int low_mask = (~((int)0) ^ up_mask);

    size_t operator()(const PEdge &e) const {
      return (e.nodes_[0] << sz_half_int) | (low_mask & e.nodes_[1]);
    }
  };

  typedef hash_map<PFace, Face::index_type, FaceHash> face_ht;
  typedef hash_map<PEdge, Edge::index_type, EdgeHash> edge_ht;

  /*! container for face storage. Must be computed each time
    nodes or cells change. */
  vector<PFace>             faces_;
  face_ht                  face_table_;
  Mutex                    face_table_lock_;
  /*! container for edge storage. Must be computed each time
    nodes or cells change. */
  vector<PEdge>             edges_;
  edge_ht                  edge_table_;
  Mutex                    edge_table_lock_;



  inline
  void hash_edge(Node::index_type n1, Node::index_type n2,
		 Cell::index_type ci, edge_ht &table) const;

  inline
  void hash_face(Node::index_type n1, Node::index_type n2, Node::index_type n3,
		 Cell::index_type ci, face_ht &table) const;

  //! useful functors
  struct FillNodeNeighbors {
    FillNodeNeighbors(vector<vector<Node::index_type> > &n, const TetVolMesh &m) :
      nbor_vec_(n),
      mesh_(m)
    {}

    void operator()(Edge::index_type e) {
      nodes_.clear();
      mesh_.get_nodes(nodes_, e);
      nbor_vec_[nodes_[0]].push_back(nodes_[1]);
      nbor_vec_[nodes_[1]].push_back(nodes_[0]);
    }

    vector<vector<Node::index_type> > &nbor_vec_;
    const TetVolMesh            &mesh_;
    Node::array_type                   nodes_;
  };

  vector<vector<Node::index_type> > node_neighbors_;
  Mutex                       node_nbor_lock_;

  typedef LockingHandle<LatticeVol<vector<Cell::index_type> > > grid_handle;
  grid_handle                 grid_;
  Mutex                       grid_lock_; // Bad traffic!
};

// Handle type for TetVolMesh mesh.
typedef LockingHandle<TetVolMesh> TetVolMeshHandle;



template <class Iter, class Functor>
void
TetVolMesh::fill_points(Iter begin, Iter end, Functor fill_ftor) {
  points_lock_.lock();
  Iter iter = begin;
  points_.resize(end - begin); // resize to the new size
  vector<Point>::iterator piter = points_.begin();
  while (iter != end) {
    *piter = fill_ftor(*iter);
    ++piter; ++iter;
  }
  points_lock_.unlock();
}

template <class Iter, class Functor>
void
TetVolMesh::fill_cells(Iter begin, Iter end, Functor fill_ftor) {
  cells_lock_.lock();
  Iter iter = begin;
  cells_.resize((end - begin) * 4); // resize to the new size
  vector<under_type>::iterator citer = cells_.begin();
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
  cells_lock_.unlock();
}

template <class Iter, class Functor>
void
TetVolMesh::fill_neighbors(Iter begin, Iter end, Functor fill_ftor) {
  nbors_lock_.lock();
  Iter iter = begin;
  neighbors_.resize((end - begin) * 4); // resize to the new size
  vector<under_type>::iterator citer = neighbors_.begin();
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
  nbors_lock_.unlock();
}

template <> TetVolMesh::Node::size_type TetVolMesh::tsize(TetVolMesh::Node::size_type *) const;
template <> TetVolMesh::Edge::size_type TetVolMesh::tsize(TetVolMesh::Edge::size_type *) const;
template <> TetVolMesh::Face::size_type TetVolMesh::tsize(TetVolMesh::Face::size_type *) const;
template <> TetVolMesh::Cell::size_type TetVolMesh::tsize(TetVolMesh::Cell::size_type *) const;
				
template <> TetVolMesh::Node::iterator TetVolMesh::tbegin(TetVolMesh::Node::iterator *) const;
template <> TetVolMesh::Edge::iterator TetVolMesh::tbegin(TetVolMesh::Edge::iterator *) const;
template <> TetVolMesh::Face::iterator TetVolMesh::tbegin(TetVolMesh::Face::iterator *) const;
template <> TetVolMesh::Cell::iterator TetVolMesh::tbegin(TetVolMesh::Cell::iterator *) const;
				
template <> TetVolMesh::Node::iterator TetVolMesh::tend(TetVolMesh::Node::iterator *) const;
template <> TetVolMesh::Edge::iterator TetVolMesh::tend(TetVolMesh::Edge::iterator *) const;
template <> TetVolMesh::Face::iterator TetVolMesh::tend(TetVolMesh::Face::iterator *) const;
template <> TetVolMesh::Cell::iterator TetVolMesh::tend(TetVolMesh::Cell::iterator *) const;

const TypeDescription* get_type_description(TetVolMesh *);
const TypeDescription* get_type_description(TetVolMesh::Node *);
const TypeDescription* get_type_description(TetVolMesh::Edge *);
const TypeDescription* get_type_description(TetVolMesh::Face *);
const TypeDescription* get_type_description(TetVolMesh::Cell *);

} // namespace SCIRun


#endif // SCI_project_TetVolMesh_h
