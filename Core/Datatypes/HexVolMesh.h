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
 *  HexVolMesh.h: Templated Meshs defined on a 3D Regular Grid
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

#ifndef SCI_project_HexVolMesh_h
#define SCI_project_HexVolMesh_h 1

#include <Core/Thread/Mutex.h>
#include <Core/Datatypes/Mesh.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Datatypes/FieldIterator.h>
#include <Core/Datatypes/LatVolField.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>
#include <Core/Persistent/PersistentSTL.h>
#include <sci_hash_map.h>

namespace SCIRun {

class SCICORESHARE HexVolMesh : public Mesh
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

  HexVolMesh();
  HexVolMesh(const HexVolMesh &copy);
  //HexVolMesh(const MeshRG &lattice);
  virtual HexVolMesh *clone() { return new HexVolMesh(*this); }
  virtual ~HexVolMesh();

  bool get_dim(vector<unsigned int>&) const { return false;  }

  virtual BBox get_bounding_box() const;
  virtual void transform(const Transform &t);

  void begin(Node::iterator &) const;
  void begin(Edge::iterator &) const;
  void begin(Face::iterator &) const;
  void begin(Cell::iterator &) const;

  void end(Node::iterator &) const;
  void end(Edge::iterator &) const;
  void end(Face::iterator &) const;
  void end(Cell::iterator &) const;

  void size(Node::size_type &) const;
  void size(Edge::size_type &) const;
  void size(Face::size_type &) const;
  void size(Cell::size_type &) const;

  //! Get the child elements of the given index
  void get_nodes(Node::array_type &array, Edge::index_type idx) const;
  void get_nodes(Node::array_type &array, Face::index_type idx) const;
  void get_nodes(Node::array_type &array, Cell::index_type idx) const;
  void get_edges(Edge::array_type &array, Face::index_type idx) const;
  void get_edges(Edge::array_type &array, Cell::index_type idx) const;
  void get_faces(Face::array_type &array, Cell::index_type idx) const;

  //! Get the parent element(s) of the given index.
  unsigned get_edges(Edge::array_type &, Node::index_type) const { return 0; }
  unsigned get_faces(Face::array_type &, Node::index_type) const { return 0; }
  unsigned get_faces(Face::array_type &, Edge::index_type) const { return 0; }
  unsigned get_cells(Cell::array_type &, Node::index_type) const { return 0; }
  unsigned get_cells(Cell::array_type &, Edge::index_type) const { return 0; }
  unsigned get_cells(Cell::array_type &, Face::index_type) const { return 0; }

  bool get_neighbor(Cell::index_type &neighbor, Cell::index_type from,
		    Face::index_type idx) const;
  void get_neighbors(Node::array_type &array, Node::index_type idx) const;
  void get_neighbors(Cell::array_type &array, Cell::index_type idx) const;

  //! Get the center point of an element.
  void get_center(Point &result, Node::index_type idx) const;
  void get_center(Point &result, Edge::index_type idx) const;
  void get_center(Point &result, Face::index_type idx) const;
  void get_center(Point &result, Cell::index_type idx) const;

  //! Get the size of an elemnt (length, area, volume)
  double get_size(Node::index_type /*idx*/) const { return 0.0; }
  double get_size(Edge::index_type idx) const 
  {
    Node::array_type arr;
    get_nodes(arr, idx);
    Point p0, p1;
    get_center(p0, arr[0]);
    get_center(p1, arr[1]);
    return (p1.asVector() - p0.asVector()).length();
  }
  double get_size(Face::index_type idx) const
  {
    Node::array_type ra;
    get_nodes(ra,idx);
    Point p0,p1,p2;
    get_point(p0,ra[0]);
    get_point(p1,ra[1]);
    get_point(p2,ra[2]);
    return (Cross(p0-p1,p2-p0)).length()*0.5;
  }
  double get_size(Cell::index_type idx) const 
  { 
    Face::array_type faces;
    get_faces(faces, idx);
    Point center; 
    get_center(center, idx);
    double volume = 0.0;
    for (unsigned int f = 0; f < faces.size(); f++)
      {
	Node::array_type nodes;
	get_nodes(nodes, faces[f]);
	volume += pyramid_volume(nodes, center);
      }
    return volume;
  };
  double get_length(Edge::index_type idx) const { return get_size(idx); };
  double get_area(Face::index_type idx) const   { return get_size(idx); };
  double get_volume(Cell::index_type idx) const { return get_size(idx); };

  unsigned int get_valence(Node::index_type idx) const
  {
    Node::array_type nodes;
    get_neighbors(nodes, idx);
    return nodes.size();
  }
  unsigned int get_valence(Edge::index_type idx) const { return 0; }
  unsigned int get_valence(Face::index_type idx) const { return 0; }
  unsigned int get_valence(Cell::index_type idx) const { return 0; }

  //! returns false if point is out of range.
  bool locate(Node::index_type &loc, const Point &p);
  bool locate(Edge::index_type &loc, const Point &p);
  bool locate(Face::index_type &loc, const Point &p);
  bool locate(Cell::index_type &loc, const Point &p);

  void get_weights(const Point &p, Node::array_type &l, vector<double> &w);
  void get_weights(const Point &, Edge::array_type &, vector<double> &)
    {ASSERTFAIL("HexVolMesh::get_weights for edges isn't supported");}
  void get_weights(const Point &, Face::array_type &, vector<double> &) 
    {ASSERTFAIL("HexVolMesh::get_weights for faces isn't supported");}
  void get_weights(const Point &p, Cell::array_type &l, vector<double> &w);

  void get_point(Point &result, Node::index_type index) const
    { result = points_[index]; }
  void set_point(const Point &point, Node::index_type index)
    { points_[index] = point; }

  void get_normal(Vector &/*normal*/, Node::index_type /*index*/) const
  { ASSERTFAIL("not implemented") }

  //! TODO: Remove this function?
  double get_element_size(const Elem::index_type &ci) {return get_volume(ci);}

  void get_random_point(Point &, const Cell::index_type &i, int seed=0) const;

  //! the double return val is the volume of the Hex.
  double get_gradient_basis(Cell::index_type ci, Vector& g0, Vector& g1,
			    Vector& g2, Vector& g3, Vector& g4,
			    Vector& g5, Vector& g6, Vector& g7);

  //! function to test if at least one of cell's nodes are in supplied range
  inline bool test_nodes_range(Cell::index_type ci, int sn, int en){
    if (cells_[ci*8]>=sn && cells_[ci*8]<en
	|| cells_[ci*8+1]>=sn && cells_[ci*8+1]<en
	|| cells_[ci*8+2]>=sn && cells_[ci*8+2]<en
	|| cells_[ci*8+3]>=sn && cells_[ci*8+3]<en
	|| cells_[ci*8+4]>=sn && cells_[ci*8+4]<en
	|| cells_[ci*8+5]>=sn && cells_[ci*8+5]<en
	|| cells_[ci*8+6]>=sn && cells_[ci*8+6]<en
	|| cells_[ci*8+7]>=sn && cells_[ci*8+7]<en)
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


  //! Persistent IO
  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  static  const string type_name(int n = -1);
  virtual const TypeDescription *get_type_description() const;

  // Extra functionality needed by this specific geometry.

  Node::index_type add_find_point(const Point &p, double err = 1.0e-3);
  void add_hex(Node::index_type a, Node::index_type b, Node::index_type c,
	       Node::index_type d, Node::index_type e, Node::index_type f,
	       Node::index_type g, Node::index_type h);
  void add_hex(const Point &p0, const Point &p1, const Point &p2,
	       const Point &p3, const Point &p4, const Point &p5, 
	       const Point &p6, const Point &p7);
  Elem::index_type add_elem(Node::array_type a);
  virtual bool is_editable() const { return true; }
  virtual int dimensionality() const { return 3; }

  Node::index_type add_point(const Point &p);

  //bool intersect(const Point &p, const Vector &dir, double &min, double &max,
  //		 Face::index_type &face, double &u, double &v);

  virtual bool		synchronize(unsigned int);

  double pyramid_volume(const Node::array_type &, const Point &) const;
  double polygon_area(const Node::array_type &, const Vector) const;

private:

  void compute_edges();
  void compute_faces();
  void compute_node_neighbors();
  void compute_grid();
  void get_face_weights(vector<double> &w, const Node::array_type &nodes,
			const Point &p, int i0, int i1, int i2, int i3);
  const Point &point(Node::index_type i) { return points_[i]; }


  bool inside8_p(Cell::index_type i, const Point &p) const;

  //! all the nodes.
  vector<Point>        points_;
  Mutex                points_lock_;

  //! each 8 indecies make up a Hex
  vector<under_type>   cells_;
  Mutex                cells_lock_;

  //! Face information.
  struct PFace {
    Node::index_type         nodes_[4];   //! 4 nodes makes a face.
    Cell::index_type         cells_[2];   //! 2 cells may have this face is in.
    Node::index_type         snodes_[4];   //! sorted nodes, used for hashing


    PFace() {
      nodes_[0] = -1;
      nodes_[1] = -1;
      nodes_[2] = -1;
      nodes_[3] = -1;
      snodes_[0] = -1;
      snodes_[1] = -1;
      snodes_[2] = -1;
      snodes_[3] = -1;
      cells_[0] = -1;
      cells_[1] = -1;
    }
    // snodes_ must be sorted. See Hash Function below.
    PFace(Node::index_type n1, Node::index_type n2, Node::index_type n3, Node::index_type n4) {
      cells_[0] = -1;
      cells_[1] = -1;
      nodes_[0] = n1;
      nodes_[1] = n2;
      nodes_[2] = n3;
      nodes_[3] = n4;
      snodes_[0] = n1;
      snodes_[1] = n2;
      snodes_[2] = n3;
      snodes_[3] = n4;
      Node::index_type tmp;
      // bubble sort the 4 node indices -- smallest one goes to nodes_[0]
      int i,j;
      for (i=0; i<3; i++) {
	for (j=i+1; j<4; j++) {
	  if (snodes_[i] > snodes_[j]) {
	    tmp = snodes_[i]; snodes_[i] = snodes_[j]; snodes_[j] = tmp;
	  }
	}
      }
    }

    bool shared() const { return ((cells_[0] != -1) && (cells_[1] != -1)); }

    //! true if both have the same nodes (order does not matter)
    bool operator==(const PFace &f) const {
      return ((snodes_[0] == f.snodes_[0]) && (snodes_[1] == f.snodes_[1]) &&
	      (snodes_[1] == f.snodes_[1]) && (snodes_[3] == f.snodes_[3]));
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
    static const size_t bucket_size;
    static const size_t min_buckets;

    static const int sz_quarter_int;
    static const int top4_mask;
    static const int up4_mask;
    static const int mid4_mask;
    static const int low4_mask;

    size_t operator()(const PFace &f) const {
      return ((f.snodes_[0] << sz_quarter_int << sz_quarter_int <<sz_quarter_int) |
	      (up4_mask & (f.snodes_[1] << sz_quarter_int << sz_quarter_int)) |
	      (mid4_mask & (f.snodes_[2] << sz_quarter_int)) |
	      (low4_mask & f.snodes_[3]));
    }
    bool operator()(const PFace &f1, const PFace& f2) const {
      return f1 == f2;
    }
  };

  friend struct FaceHash; // needed by the gcc-2.95.3 compiler
  
  /*! hash the egde's node_indecies such that edges with the same nodes
   *  hash to the same value. nodes are sorted on edge construction. */
  struct EdgeHash {
    static const size_t bucket_size = 4;
    static const size_t min_buckets = 8;
    static const int sz_int = sizeof(int) * 8; // in bits
    static const int sz_half_int = sizeof(int) << 2; // in bits
    static const int up_mask = ((~((int)0)) << sz_half_int);
    static const int low_mask = (~((int)0) ^ up_mask);

    size_t operator()(const PEdge &e) const {
      return (e.nodes_[0] << sz_half_int) | (low_mask & e.nodes_[1]);
    }
    bool operator()(const PEdge &e1, const PEdge& e2) const {
      return e1 == e2;
    }
  };

#ifdef HAVE_HASH_MAP
  typedef hash_map<PFace, Face::index_type, FaceHash> face_ht;
  typedef hash_map<PEdge, Edge::index_type, EdgeHash> edge_ht;
#else
  typedef map<PFace, Face::index_type, FaceHash> face_ht;
  typedef map<PEdge, Edge::index_type, EdgeHash> edge_ht;
#endif
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
		 Node::index_type n4, Cell::index_type ci, face_ht &table) const;

  //! useful functors
  struct FillNodeNeighbors {
    FillNodeNeighbors(vector<vector<Node::index_type> > &n, const HexVolMesh &m) :
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
    const HexVolMesh            &mesh_;
    Node::array_type                   nodes_;
  };

  vector<vector<Node::index_type> > node_neighbors_;
  Mutex                       node_nbor_lock_;

  typedef LockingHandle<LatVolField<vector<Cell::index_type> > > grid_handle;
  grid_handle                 grid_;
  Mutex                       grid_lock_; // Bad traffic!

  unsigned int			synchronized_;
};

// Handle type for HexVolMesh mesh.
typedef LockingHandle<HexVolMesh> HexVolMeshHandle;



template <class Iter, class Functor>
void
HexVolMesh::fill_points(Iter begin, Iter end, Functor fill_ftor) {
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
HexVolMesh::fill_cells(Iter begin, Iter end, Functor fill_ftor) {
  cells_lock_.lock();
  Iter iter = begin;
  cells_.resize((end - begin) * 8); // resize to the new size
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
    *citer = nodes[4];
    ++citer; ++iter;
    *citer = nodes[5];
    ++citer; ++iter;
    *citer = nodes[6];
    ++citer; ++iter;
    *citer = nodes[7];
    ++citer; ++iter;
  }
  cells_lock_.unlock();
}

template <class Iter, class Functor>
void
HexVolMesh::fill_neighbors(Iter begin, Iter end, Functor fill_ftor) {
  nbors_lock_.lock();
  Iter iter = begin;
  neighbors_.resize((end - begin) * 8); // resize to the new size
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
    *citer = face_nbors[4];
    ++citer; ++iter;
    *citer = face_nbors[5];
    ++citer; ++iter;
    *citer = face_nbors[6];
    ++citer; ++iter;
    *citer = face_nbors[7];
    ++citer; ++iter;
  }
  nbors_lock_.unlock();
}

const TypeDescription* get_type_description(HexVolMesh *);
const TypeDescription* get_type_description(HexVolMesh::Node *);
const TypeDescription* get_type_description(HexVolMesh::Edge *);
const TypeDescription* get_type_description(HexVolMesh::Face *);
const TypeDescription* get_type_description(HexVolMesh::Cell *);

} // namespace SCIRun


#endif // SCI_project_HexVolMesh_h
