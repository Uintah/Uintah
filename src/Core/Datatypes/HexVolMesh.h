/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
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
#include <Core/Containers/StackVector.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>
#include <Core/Persistent/PersistentSTL.h>
#include <Core/Datatypes/SearchGrid.h>
#include <sci_hash_map.h>

namespace SCIRun {

template <class Basis>
class HexVolMesh : public Mesh
{
public:
  typedef LockingHandle<HexVolMesh<Basis> > handle_type;
  typedef Basis                         basis_type;
  typedef unsigned int                  under_type;

  //! Index and Iterator types required for Mesh Concept.
  struct Node {
    typedef NodeIndex<under_type>       index_type;
    typedef NodeIterator<under_type>    iterator;
    typedef NodeIndex<under_type>       size_type;
    typedef StackVector<index_type, 8>  array_type;
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
  friend class ElemData;
  
  class ElemData 
  {
  public:
    ElemData(const HexVolMesh<Basis>& msh, 
	     const typename Cell::index_type ind) :
      mesh_(msh),
      index_(ind)
    {
      mesh_.get_edges(edges_, index_);

    }
    
    // the following designed to coordinate with ::get_nodes
    inline 
    unsigned node0_index() const {
      return mesh_.cells_[index_ * 8];
    }
    inline 
    unsigned node1_index() const {
      return mesh_.cells_[index_ * 8 + 1];
    }
    inline 
    unsigned node2_index() const {
      return mesh_.cells_[index_ * 8 + 2];
    }
    inline 
    unsigned node3_index() const {
      return mesh_.cells_[index_ * 8 + 3];
    }
    inline 
    unsigned node4_index() const {
      return mesh_.cells_[index_ * 8 + 4];
    }
    inline 
    unsigned node5_index() const {
      return mesh_.cells_[index_ * 8 + 5];
    }
    inline 
    unsigned node6_index() const {
      return mesh_.cells_[index_ * 8 + 6];
    }
    inline 
    unsigned node7_index() const {
      return mesh_.cells_[index_ * 8 + 7];
    }

    // the following designed to coordinate with ::get_edges
    inline 
    unsigned edge0_index() const {
      return edges_[0];
    }
    inline 
    unsigned edge1_index() const {
      return edges_[1];
    }
    inline 
    unsigned edge2_index() const {
      return edges_[2];
    }
    inline 
    unsigned edge3_index() const {
      return edges_[3];
    }
    inline 
    unsigned edge4_index() const {
      return edges_[4];
    }
    inline 
    unsigned edge5_index() const {
      return edges_[5];
    }
    inline 
    unsigned edge6_index() const {
      return edges_[6];
    }
    inline 
    unsigned edge7_index() const {
      return edges_[7];
    }
    inline 
    unsigned edge8_index() const {
      return edges_[8];
    }
    inline 
    unsigned edge9_index() const {
      return edges_[9];
    }
    inline 
    unsigned edge10_index() const {
      return edges_[10];
    }
    inline 
    unsigned edg11_index() const {
      return edges_[11];
    }

    inline 
    const Point node0() const {
      return mesh_.points_[node0_index()];
    }
    inline 
    const Point node1() const {
      return mesh_.points_[node1_index()];
    }
    inline 
    const Point node2() const {
      return mesh_.points_[node2_index()];
    }
    inline 
    const Point node3() const {
      return mesh_.points_[node3_index()];
    }
    inline 
    const Point node4() const {
      return mesh_.points_[node4_index()];
    }
    inline 
    const Point node5() const {
      return mesh_.points_[node5_index()];
    }
    inline 
    const Point node6() const {
      return mesh_.points_[node6_index()];
    }
    inline 
    const Point node7() const {
      return mesh_.points_[node7_index()];
    }

  private:
    const HexVolMesh<Basis>          &mesh_;
    const typename Cell::index_type  index_;
    typename Edge::array_type        edges_;
   };

  HexVolMesh();
  HexVolMesh(const HexVolMesh &copy);
  //HexVolMesh(const MeshRG &lattice);
  virtual HexVolMesh *clone() { return new HexVolMesh(*this); }
  virtual ~HexVolMesh();

  bool get_dim(vector<unsigned int>&) const { return false;  }

  virtual BBox get_bounding_box() const;
  virtual void transform(const Transform &t);

  void begin(typename Node::iterator &) const;
  void begin(typename Edge::iterator &) const;
  void begin(typename Face::iterator &) const;
  void begin(typename Cell::iterator &) const;

  void end(typename Node::iterator &) const;
  void end(typename Edge::iterator &) const;
  void end(typename Face::iterator &) const;
  void end(typename Cell::iterator &) const;

  void size(typename Node::size_type &) const;
  void size(typename Edge::size_type &) const;
  void size(typename Face::size_type &) const;
  void size(typename Cell::size_type &) const;

  void to_index(typename Node::index_type &index, unsigned int i) const 
  { index = i; }
  void to_index(typename Edge::index_type &index, unsigned int i) const 
  { index = i; }
  void to_index(typename Face::index_type &index, unsigned int i) const 
  { index = i; }
  void to_index(typename Cell::index_type &index, unsigned int i) const 
  { index = i; }

  //! Get the child elements of the given index
  void get_nodes(typename Node::array_type &array, 
		 typename Edge::index_type idx) const;
  void get_nodes(typename Node::array_type &array, 
		 typename Face::index_type idx) const;
  void get_nodes(typename Node::array_type &array, 
		 typename Cell::index_type idx) const;
  void get_edges(typename Edge::array_type &array, 
		 typename Face::index_type idx) const;
  void get_edges(typename Edge::array_type &array, 
		 typename Cell::index_type idx) const;
  void get_faces(typename Face::array_type &array, 
		 typename Cell::index_type idx) const;

  //! Get the parent element(s) of the given index.
  unsigned get_edges(typename Edge::array_type &, 
		     typename Node::index_type) const { return 0; }
  unsigned get_faces(typename Face::array_type &, 
		     typename Node::index_type) const { return 0; }
  unsigned get_faces(typename Face::array_type &, 
		     typename Edge::index_type) const { return 0; }
  unsigned get_cells(typename Cell::array_type &, 
		     typename Node::index_type) const { return 0; }
  unsigned get_cells(typename Cell::array_type &, 
		     typename Edge::index_type) const { return 0; }
  unsigned get_cells(typename Cell::array_type &, 
		     typename Face::index_type) const { return 0; }

  bool get_neighbor(typename Cell::index_type &neighbor, 
		    typename Cell::index_type from,
		    typename Face::index_type idx) const;
  void get_neighbors(vector<typename Node::index_type> &array,
		     typename Node::index_type idx) const;
  void get_neighbors(typename Cell::array_type &array, 
		     typename Cell::index_type idx) const;

  //! Get the center point of an element.
  void get_center(Point &result, typename Node::index_type idx) const;
  void get_center(Point &result, typename Edge::index_type idx) const;
  void get_center(Point &result, typename Face::index_type idx) const;
  void get_center(Point &result, typename Cell::index_type idx) const;

  //! Get the size of an elemnt (length, area, volume)
  double get_size(typename Node::index_type /*idx*/) const { return 0.0; }
  double get_size(typename Edge::index_type idx) const 
  {
    typename Node::array_type arr;
    get_nodes(arr, idx);
    Point p0, p1;
    get_center(p0, arr[0]);
    get_center(p1, arr[1]);
    return (p1.asVector() - p0.asVector()).length();
  }
  double get_size(typename Face::index_type idx) const
  {
    typename Node::array_type ra;
    get_nodes(ra,idx);
    Point p0,p1,p2;
    get_point(p0,ra[0]);
    get_point(p1,ra[1]);
    get_point(p2,ra[2]);
    return (Cross(p0-p1,p2-p0)).length()*0.5;
  }
  double get_size(typename Cell::index_type idx) const 
  { 
    typename Face::array_type faces;
    get_faces(faces, idx);
    Point center; 
    get_center(center, idx);
    double volume = 0.0;
    for (unsigned int f = 0; f < faces.size(); f++)
      {
	typename Node::array_type nodes;
	get_nodes(nodes, faces[f]);
	volume += pyramid_volume(nodes, center);
      }
    return volume;
  };
  double get_length(typename Edge::index_type idx) const 
  { return get_size(idx); };
  double get_area(typename Face::index_type idx) const   
  { return get_size(idx); };
  double get_volume(typename Cell::index_type idx) const 
  { return get_size(idx); };

  unsigned int get_valence(typename Node::index_type idx) const
  {
    vector<typename Node::index_type> nodes;
    get_neighbors(nodes, idx);
    return (int)nodes.size();
  }
  unsigned int get_valence(typename Edge::index_type /*idx*/) const 
  { return 0; }
  unsigned int get_valence(typename Face::index_type /*idx*/) const 
  { return 0; }
  unsigned int get_valence(typename Cell::index_type /*idx*/) const 
  { return 0; }

  //! returns false if point is out of range.
  bool locate(typename Node::index_type &loc, const Point &p);
  bool locate(typename Edge::index_type &loc, const Point &p);
  bool locate(typename Face::index_type &loc, const Point &p);
  bool locate(typename Cell::index_type &loc, const Point &p);

  void get_point(Point &result, typename Node::index_type index) const
    { result = points_[index]; }
  void set_point(const Point &point, typename Node::index_type index)
    { points_[index] = point; }

  void get_normal(Vector &, typename Node::index_type /*index*/) const
  { ASSERTFAIL("not implemented") }

  void get_random_point(Point &p, typename Cell::index_type i, 
			int seed = 0) const;

  //! the double return val is the volume of the Hex.
  double get_gradient_basis(typename Cell::index_type ci, 
			    Vector& g0, Vector& g1,
			    Vector& g2, Vector& g3, Vector& g4,
			    Vector& g5, Vector& g6, Vector& g7);
  
  //! function to test if at least one of cell's nodes are in supplied range
  inline bool test_nodes_range(typename Cell::index_type ci,
			       unsigned int sn,
			       unsigned int en)
  {
    if (cells_[ci*8]>=sn && cells_[ci*8]<en
	|| cells_[ci*8+1]>=sn && cells_[ci*8+1]<en
	|| cells_[ci*8+2]>=sn && cells_[ci*8+2]<en
	|| cells_[ci*8+3]>=sn && cells_[ci*8+3]<en
	|| cells_[ci*8+4]>=sn && cells_[ci*8+4]<en
	|| cells_[ci*8+5]>=sn && cells_[ci*8+5]<en
	|| cells_[ci*8+6]>=sn && cells_[ci*8+6]<en
	|| cells_[ci*8+7]>=sn && cells_[ci*8+7]<en)
    {
      return true;
    }
    else
    {
      return false;
    }
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

  typename Node::index_type add_find_point(const Point &p, 
					   double err = 1.0e-3);
  void add_hex(typename Node::index_type a, typename Node::index_type b, 
	       typename Node::index_type c,
	       typename Node::index_type d, typename Node::index_type e, 
	       typename Node::index_type f,
	       typename Node::index_type g, typename Node::index_type h);
  void add_hex(const Point &p0, const Point &p1, const Point &p2,
	       const Point &p3, const Point &p4, const Point &p5, 
	       const Point &p6, const Point &p7);
  typename Elem::index_type add_elem(typename Node::array_type a);
  void node_reserve(size_t s) { points_.reserve(s); }
  void elem_reserve(size_t s) { edges_.reserve(s*8); }
  virtual bool is_editable() const { return true; }
  virtual int dimensionality() const { return 3; }

  typename Node::index_type add_point(const Point &p);

  //bool intersect(const Point &p, const Vector &dir, double &min, double &max,
  //		 typename Face::index_type &face, double &u, double &v);

  virtual bool		synchronize(unsigned int);

  double pyramid_volume(const typename Node::array_type &, 
			const Point &) const;
  double polygon_area(const typename Node::array_type &, const Vector) const;
  Basis& get_basis() { return basis_; }
  
  //! Generate the list of points that make up a sufficiently accurate
  //! piecewise linear approximation of an edge.
  void pwl_approx_edge(vector<vector<double> > &coords, 
		       typename Cell::index_type ci, 
		       typename Edge::index_type ei, 
		       unsigned div_per_unit) const
  {
    typename Edge::array_type edges;
    get_edges(edges, ci);
    PEdge e = edges_[ei];
    unsigned count = 0;
    typename Edge::array_type::iterator iter = edges.begin();
    while (iter != edges.end()) {
      if (e == edges_[*iter++]) break;
      ++count;
    }
    basis_.approx_edge(count, div_per_unit, coords); 
  }

  //! Generate the list of points that make up a sufficiently accurate
  //! piecewise linear approximation of an face.
  void pwl_approx_face(vector<vector<vector<double> > > &coords, 
		       typename Cell::index_type ci, 
		       typename Face::index_type ei, 
		       unsigned div_per_unit) const
  {
    typename Face::array_type faces;
    get_faces(faces, ci);
    PFace e = faces_[ei];
    unsigned count = 0;
    typename Face::array_type::iterator iter = faces.begin();
    while (iter != faces.end()) {
      if (e == faces_[*iter++]) break;
      ++count;
    }
    // map the order we have faces to the way the basis expects it for hexes.
    // this needs to be unified for all hex types to avoid this switch.
    switch (count) {
    case 0:      
      break;
    case 1:
      count = 5;
      break;
    case 2:
      count = 1;
      break;
    case 3:
      //count = 3;
      break;
    case 4:
      //count = 2;
      break;
    case 5:
      count = 2;
      break;
    };
    
    basis_.approx_face(count, div_per_unit, coords); 
  }

  void get_coords(vector<double> &coords, 
		  const Point &p,
		  typename Cell::index_type idx) const
  {
    ElemData ed(*this, idx);
    basis_.get_coords(coords, p, ed); 
  }
  
  void interpolate(Point &pt, const vector<double> &coords, 
		   typename Cell::index_type idx) const
  {
    ElemData ed(*this, idx);
    pt = basis_.interpolate(coords, ed);
  }

  static const TypeDescription* node_type_description();
  static const TypeDescription* edge_type_description();
  static const TypeDescription* face_type_description();
  static const TypeDescription* cell_type_description();
  static Persistent* maker() { return scinew HexVolMesh<Basis>; }

  //! must detach, if altering points!
  vector<Point>& get_points() { return points_; }

private:

  void compute_edges();
  void compute_faces();
  void compute_node_neighbors();
  void compute_grid();
  void get_face_weights(double *w, const typename Node::array_type &nodes,
			const Point &p, int i0, int i1, int i2, int i3);
  const Point &point(typename Node::index_type i) { return points_[i]; }


  bool inside8_p(typename Cell::index_type i, const Point &p) const;

  //! all the nodes.
  vector<Point>        points_;
  Mutex                points_lock_;

  //! each 8 indecies make up a Hex
  vector<under_type>   cells_;
  Mutex                cells_lock_;

  //! each 8 indecies make up a Hex
  vector<under_type>   neighbors_;
  Mutex                nbors_lock_;

  //! Face information.
  struct PFace {
    typename Node::index_type         nodes_[4];   //! 4 nodes makes a face.
    typename Cell::index_type         cells_[2];   //! 2 cells may have this face is in.
    typename Node::index_type         snodes_[4];   //! sorted nodes, used for hashing


    PFace() {
      nodes_[0] = MESH_NO_NEIGHBOR;
      nodes_[1] = MESH_NO_NEIGHBOR;
      nodes_[2] = MESH_NO_NEIGHBOR;
      nodes_[3] = MESH_NO_NEIGHBOR;
      snodes_[0] = MESH_NO_NEIGHBOR;
      snodes_[1] = MESH_NO_NEIGHBOR;
      snodes_[2] = MESH_NO_NEIGHBOR;
      snodes_[3] = MESH_NO_NEIGHBOR;
      cells_[0] = MESH_NO_NEIGHBOR;
      cells_[1] = MESH_NO_NEIGHBOR;
    }
    // snodes_ must be sorted. See Hash Function below.
    PFace(typename Node::index_type n1, typename Node::index_type n2, 
	  typename Node::index_type n3, typename Node::index_type n4) {
      cells_[0] = MESH_NO_NEIGHBOR;
      cells_[1] = MESH_NO_NEIGHBOR;
      nodes_[0] = n1;
      nodes_[1] = n2;
      nodes_[2] = n3;
      nodes_[3] = n4;
      snodes_[0] = n1;
      snodes_[1] = n2;
      snodes_[2] = n3;
      snodes_[3] = n4;
      typename Node::index_type tmp;
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

    bool shared() const { return ((cells_[0] != MESH_NO_NEIGHBOR) && 
				  (cells_[1] != MESH_NO_NEIGHBOR)); }

    //! true if both have the same nodes (order does not matter)
    bool operator==(const PFace &f) const {
      return ((snodes_[0] == f.snodes_[0]) && (snodes_[1] == f.snodes_[1]) &&
	      (snodes_[2] == f.snodes_[2]) && (snodes_[3] == f.snodes_[3]));
    }

    //! Compares each node.  When a non equal node is found the <
    //! operator is applied.
    bool operator<(const PFace &f) const {
      if (snodes_[0] == f.snodes_[0])
        if (snodes_[1] == f.snodes_[1])
          if (snodes_[2] == f.snodes_[2])
            return (snodes_[3] < f.snodes_[3]);
          else
            return (snodes_[2] < f.snodes_[2]);
        else
          return (snodes_[1] < f.snodes_[1]);
      else
        return (snodes_[0] < f.snodes_[0]);
    }
};


  //! Edge information.
  struct PEdge {
    typename Node::index_type         nodes_[2];   //! 2 nodes makes an edge.
    vector<typename Cell::index_type> cells_;      //! list of all the cells this edge is in.

    PEdge() : cells_(0) {
      nodes_[0] = MESH_NO_NEIGHBOR;
      nodes_[1] = MESH_NO_NEIGHBOR;
    }
    // node_[0] must be smaller than node_[1]. See Hash Function below.
    PEdge(typename Node::index_type n1, 
	  typename Node::index_type n2) : cells_(0) {
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

    //! Compares each node.  When a non equal node is found the <
    //! operator is applied.
    bool operator<(const PEdge &e) const {
      if (nodes_[0] == e.nodes_[0])
        return (nodes_[1] < e.nodes_[1]);
      else
        return (nodes_[0] < e.nodes_[0]);
    }
  };

  /*! hash the egde's node_indecies such that edges with the same nodes
   *  hash to the same value. nodes are sorted on edge construction. */
  static const int sz_int = sizeof(int) * 8; // in bits
  struct FaceHash {
    //! These are needed by the hash_map particularly
    // ANSI C++ allows us to initialize these variables in the
    // declaration.  However there may be compilers which will complain
    // about it.
    static const size_t bucket_size = 4;
    static const size_t min_buckets = 8;

    //! These are for our own use (making the hash function.
    static const int sz_quarter_int = (int)(sz_int * .25);
    static const int top4_mask = ((~((int)0)) << sz_quarter_int << sz_quarter_int << sz_quarter_int);
    static const int up4_mask = top4_mask ^ (~((int)0) << sz_quarter_int << sz_quarter_int);
    static const int mid4_mask =  top4_mask ^ (~((int)0) << sz_quarter_int);
    static const int low4_mask = ~(top4_mask | mid4_mask);

    //! This is the hash function
    size_t operator()(const PFace &f) const {
      return ((f.snodes_[0] << sz_quarter_int << sz_quarter_int <<sz_quarter_int) |
	      (up4_mask & (f.snodes_[1] << sz_quarter_int << sz_quarter_int)) |
	      (mid4_mask & (f.snodes_[2] << sz_quarter_int)) |
	      (low4_mask & f.snodes_[3]));
    }
    //! This should return less than rather than equal to.
    bool operator()(const PFace &f1, const PFace& f2) const {
      return f1 < f2;
    }
  };

  friend struct FaceHash; // needed by the gcc-2.95.3 compiler
  
  /*! hash the egde's node_indecies such that edges with the same nodes
   *  hash to the same value. nodes are sorted on edge construction. */
  struct EdgeHash {
    //! These are needed by the hash_map particularly
    // ANSI C++ allows us to initialize these variables in the
    // declaration.  However there may be compilers which will complain
    // about it.
    static const size_t bucket_size = 4;
    static const size_t min_buckets = 8;

    //! These are for our own use (making the hash function.
    static const int sz_int = sizeof(int) * 8; // in bits
    static const int sz_half_int = sizeof(int) << 2; // in bits
    static const int up_mask = ((~((int)0)) << sz_half_int);
    static const int low_mask = (~((int)0) ^ up_mask);

    //! This is the hash function
    size_t operator()(const PEdge &e) const {
      return (e.nodes_[0] << sz_half_int) | (low_mask & e.nodes_[1]);
    }
    //!  This should return less than rather than equal to.
    bool operator()(const PEdge &e1, const PEdge& e2) const {
      return e1 < e2;
    }
  };

#ifdef HAVE_HASH_MAP
  typedef hash_map<PFace, typename Face::index_type, FaceHash> face_ht;
  typedef hash_map<PEdge, typename Edge::index_type, EdgeHash> edge_ht;
#else
  typedef map<PFace, typename Face::index_type, FaceHash> face_ht;
  typedef map<PEdge, typename Edge::index_type, EdgeHash> edge_ht;
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
  void hash_edge(typename Node::index_type n1, typename Node::index_type n2,
		 typename Cell::index_type ci, edge_ht &table) const;

  inline
  void hash_face(typename Node::index_type n1, typename Node::index_type n2, 
		 typename Node::index_type n3, typename Node::index_type n4, 
		 typename Cell::index_type ci, face_ht &table) const;

  //! useful functors
  struct FillNodeNeighbors {
    FillNodeNeighbors(vector<vector<typename Node::index_type> > &n, 
		      const HexVolMesh &m) :
      nbor_vec_(n),
      mesh_(m)
    {}

    void operator()(typename Edge::index_type e) {
      nodes_.clear();
      mesh_.get_nodes(nodes_, e);
      nbor_vec_[nodes_[0]].push_back(nodes_[1]);
      nbor_vec_[nodes_[1]].push_back(nodes_[0]);
    }

    vector<vector<typename Node::index_type> > &nbor_vec_;
    const HexVolMesh            &mesh_;
    typename Node::array_type                   nodes_;
  };

  vector<vector<typename Node::index_type> > node_neighbors_;
  Mutex                       node_nbor_lock_;

  LockingHandle<SearchGrid>   grid_;
  Mutex                       grid_lock_; // Bad traffic!
  typename Cell::index_type            locate_cache_;

  unsigned int			synchronized_;
  Basis                         basis_;
};

template <class Basis>
template <class Iter, class Functor>
void
HexVolMesh<Basis>::fill_points(Iter begin, Iter end, Functor fill_ftor) {
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

template <class Basis>
template <class Iter, class Functor>
void
HexVolMesh<Basis>::fill_cells(Iter begin, Iter end, Functor fill_ftor) {
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

template <class Basis>
template <class Iter, class Functor>
void
HexVolMesh<Basis>::fill_neighbors(Iter begin, Iter end, Functor fill_ftor) {
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

template <class Basis>
PersistentTypeID 
HexVolMesh<Basis>::type_id(HexVolMesh<Basis>::type_name(-1), "Mesh",  maker);


template <class Basis>
const string
HexVolMesh<Basis>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;
  }
  else if (n == 0)
  {
    static const string nm("HexVolMesh");
    return nm;
  }
  else 
  {
    return find_type_name((Basis *)0);
  }
}

template <class Basis>
HexVolMesh<Basis>::HexVolMesh() :
  points_(0),
  points_lock_("HexVolMesh points_ fill lock"),
  cells_(0),
  cells_lock_("HexVolMesh cells_ fill lock"),
  neighbors_(0),
  nbors_lock_("HexVolMesh neighbors_ fill lock"),
  faces_(0),
  face_table_(),
  face_table_lock_("HexVolMesh faces_ fill lock"),
  edges_(0),
  edge_table_(),
  edge_table_lock_("HexVolMesh edge_ fill lock"),
  node_nbor_lock_("HexVolMesh node_neighbors_ fill lock"),
  grid_(0),
  grid_lock_("HexVolMesh grid_ fill lock"),
  locate_cache_(0),
  synchronized_(NODES_E | CELLS_E)
{
}

template <class Basis>
HexVolMesh<Basis>::HexVolMesh(const HexVolMesh &copy):
  points_(copy.points_),
  points_lock_("HexVolMesh points_ fill lock"),
  cells_(copy.cells_),
  cells_lock_("HexVolMesh cells_ fill lock"),
  neighbors_(copy.neighbors_),
  nbors_lock_("HexVolMesh neighbors_ fill lock"),
  faces_(copy.faces_),
  face_table_(copy.face_table_),
  face_table_lock_("HexVolMesh faces_ fill lock"),
  edges_(copy.edges_),
  edge_table_(copy.edge_table_),
  edge_table_lock_("HexVolMesh edge_ fill lock"),
  node_nbor_lock_("HexVolMesh node_neighbors_ fill lock"),
  grid_(copy.grid_),
  grid_lock_("HexVolMesh grid_ fill lock"),
  locate_cache_(0),
  synchronized_(copy.synchronized_)
{
}

template <class Basis>
HexVolMesh<Basis>::~HexVolMesh()
{
}

/* To generate a random point inside of a hexrahedron, we generate random
   barrycentric coordinates (independent random variables between 0 and
   1 that sum to 1) for the point. */
template <class Basis>
void 
HexVolMesh<Basis>::get_random_point(Point &/*p*/, typename Cell::index_type /*ei*/,
				    int /*seed*/) const
{
  ASSERTFAIL("don't know how to pick a random point in a hex");
}

template <class Basis>
BBox
HexVolMesh<Basis>::get_bounding_box() const
{
  BBox result;
  
  typename Node::iterator ni, nie;
  begin(ni);
  end(nie);
  while (ni != nie)
  {
    Point p;
    get_point(p, *ni);
    result.extend(p);
    ++ni;
  }
  return result;
}


template <class Basis>
void
HexVolMesh<Basis>::transform(const Transform &t)
{
  vector<Point>::iterator itr = points_.begin();
  vector<Point>::iterator eitr = points_.end();
  while (itr != eitr)
  {
    *itr = t.project(*itr);
    ++itr;
  }

  grid_lock_.lock();
  if (grid_.get_rep()) { grid_->transform(t); }
  grid_lock_.unlock();
}


template <class Basis>
void
HexVolMesh<Basis>::hash_face(typename Node::index_type n1, 
			     typename Node::index_type n2,
			     typename Node::index_type n3, 
			     typename Node::index_type n4,
			     typename Cell::index_type ci, 
			     face_ht &table) const 
{
  PFace f(n1, n2, n3, n4);
  
  typename face_ht::iterator iter = table.find(f);
  if (iter == table.end()) {
    f.cells_[0] = ci;
    table[f] = 0; // insert for the first time
  } else {
    PFace f = (*iter).first;
    if (f.cells_[1] != MESH_NO_NEIGHBOR) {
      cerr << "This Mesh has problems: Cells #" 
	   << f.cells_[0] << ", #" << f.cells_[1] << ", and #" << ci 
	   << " are illegally adjacent." << std::endl; 
      //     SCI_THROW(InternalError("Corrupt HexVolMesh"));      
    } else if (f.cells_[0] == ci) {
      cerr << "This Mesh has problems: Cells #" 
	   << f.cells_[0] << " and #" << ci 
	   << " are the same." << std::endl; 
      //     SCI_THROW(InternalError("Corrupt HexVolMesh"));      
    } else {
      f.cells_[1] = ci; // add this cell
      table.erase(iter);
      table[f] = 0;
    }
  }
}

template <class Basis>
void
HexVolMesh<Basis>::compute_faces()
{
  face_table_lock_.lock();
  if (synchronized_ & FACES_E) {
    face_table_lock_.unlock();
    return;
  }
  face_table_.clear();

  typename Cell::iterator ci, cie;
  begin(ci); end(cie);
  typename Node::array_type arr(8);
  while (ci != cie)
  {
    get_nodes(arr, *ci);
    // 6 faces -- each is entered CCW from outside looking in
    hash_face(arr[0], arr[1], arr[2], arr[3], *ci, face_table_);
    hash_face(arr[7], arr[6], arr[5], arr[4], *ci, face_table_);
    hash_face(arr[0], arr[4], arr[5], arr[1], *ci, face_table_);
    hash_face(arr[2], arr[6], arr[7], arr[3], *ci, face_table_);
    hash_face(arr[3], arr[7], arr[4], arr[0], *ci, face_table_);
    hash_face(arr[1], arr[5], arr[6], arr[2], *ci, face_table_);
    ++ci;
  }
  // dump edges into the edges_ container.
  faces_.resize(face_table_.size());
  typename vector<PFace>::iterator f_iter = faces_.begin();
  typename face_ht::iterator ht_iter = face_table_.begin();
  int i = 0;
  while (ht_iter != face_table_.end()) {
    *f_iter = (*ht_iter).first;
    (*ht_iter).second = i;
    ++f_iter; ++ht_iter; i++;
  }

  synchronized_ |= FACES_E;
  face_table_lock_.unlock();
}

template <class Basis>
void
HexVolMesh<Basis>::hash_edge(typename Node::index_type n1, 
			     typename Node::index_type n2,
			     typename Cell::index_type ci, 
			     edge_ht &table) const 
{
  PEdge e(n1, n2);
  typename edge_ht::iterator iter = table.find(e);
  if (iter == table.end()) {
    table[e] = 0; // insert for the first time
  } else {
    PEdge e = (*iter).first;
    e.cells_.push_back(ci); // add this cell
    table.erase(iter);
    table[e] = 0;
  }
}

template <class Basis>
void
HexVolMesh<Basis>::compute_edges()
{
  edge_table_lock_.lock();
  if (synchronized_ & EDGES_E) {
    edge_table_lock_.unlock();
    return;
  }
  typename Cell::iterator ci, cie;
  begin(ci); end(cie);
  typename Node::array_type arr;
  while (ci != cie)
  {
    get_nodes(arr, *ci);
    hash_edge(arr[0], arr[1], *ci, edge_table_);
    hash_edge(arr[1], arr[2], *ci, edge_table_);
    hash_edge(arr[2], arr[3], *ci, edge_table_);
    hash_edge(arr[3], arr[0], *ci, edge_table_);

    hash_edge(arr[4], arr[5], *ci, edge_table_);
    hash_edge(arr[5], arr[6], *ci, edge_table_);
    hash_edge(arr[6], arr[7], *ci, edge_table_);
    hash_edge(arr[7], arr[4], *ci, edge_table_);

    hash_edge(arr[0], arr[4], *ci, edge_table_);
    hash_edge(arr[5], arr[1], *ci, edge_table_);

    hash_edge(arr[2], arr[6], *ci, edge_table_);
    hash_edge(arr[7], arr[3], *ci, edge_table_);
    ++ci;
  }
  // dump edges into the edges_ container.
  edges_.resize(edge_table_.size());
  typename vector<PEdge>::iterator e_iter = edges_.begin();
  typename edge_ht::iterator ht_iter = edge_table_.begin();
  while (ht_iter != edge_table_.end()) {
    *e_iter = (*ht_iter).first;
    (*ht_iter).second = static_cast<typename Edge::index_type>(
						      e_iter - edges_.begin());
    ++e_iter; ++ht_iter;
  }
  
  synchronized_ |= EDGES_E;
  edge_table_lock_.unlock();
}




template <class Basis>
bool
HexVolMesh<Basis>::synchronize(unsigned int tosync)
{
  if (tosync & EDGES_E && !(synchronized_ & EDGES_E)) compute_edges();
  if (tosync & FACES_E && !(synchronized_ & FACES_E)) compute_faces();
  if (tosync & LOCATE_E && !(synchronized_ & LOCATE_E)) {
    compute_grid();
    if (!(synchronized_ & FACES_E)) compute_faces();
  }
  if (tosync & NODE_NEIGHBORS_E && !(synchronized_ & NODE_NEIGHBORS_E)) 
    compute_node_neighbors();
  return true;
}


template <class Basis>
void
HexVolMesh<Basis>::begin(typename HexVolMesh::Node::iterator &itr) const
{
  ASSERTMSG(synchronized_ & NODES_E,
	    "Must call synchronize NODES_E on HexVolMesh first");
  itr = 0;
}

template <class Basis>
void
HexVolMesh<Basis>::end(typename HexVolMesh::Node::iterator &itr) const
{
  ASSERTMSG(synchronized_ & NODES_E,
	    "Must call synchronize NODES_E on HexVolMesh first");
  itr = static_cast<typename Node::iterator>(points_.size());
}

template <class Basis>
void
HexVolMesh<Basis>::size(typename HexVolMesh::Node::size_type &s) const
{
  ASSERTMSG(synchronized_ & NODES_E,
	    "Must call synchronize NODES_E on HexVolMesh first");
  s = static_cast<typename Node::size_type>(points_.size());
}

template <class Basis>
void
HexVolMesh<Basis>::begin(typename HexVolMesh::Edge::iterator &itr) const
{
  ASSERTMSG(synchronized_ & EDGES_E,
	    "Must call synchronize EDGES_E on HexVolMesh first");
  itr = 0;
}

template <class Basis>
void
HexVolMesh<Basis>::end(typename HexVolMesh::Edge::iterator &itr) const
{
  ASSERTMSG(synchronized_ & EDGES_E,
	    "Must call synchronize EDGES_E on HexVolMesh first");
  itr = static_cast<typename Edge::iterator>(edges_.size());
}

template <class Basis>
void
HexVolMesh<Basis>::size(typename HexVolMesh::Edge::size_type &s) const
{
  ASSERTMSG(synchronized_ & EDGES_E,
	    "Must call synchronize EDGES_E on HexVolMesh first");
  s = static_cast<typename Edge::size_type>(edges_.size());
}

template <class Basis>
void
HexVolMesh<Basis>::begin(typename HexVolMesh::Face::iterator &itr) const
{
  ASSERTMSG(synchronized_ & FACES_E,
	    "Must call synchronize FACES_E on HexVolMesh first");
  itr = 0;
}

template <class Basis>
void
HexVolMesh<Basis>::end(typename HexVolMesh::Face::iterator &itr) const
{
  ASSERTMSG(synchronized_ & FACES_E,
	    "Must call synchronize FACES_E on HexVolMesh first");
  itr = static_cast<typename Face::iterator>(faces_.size());
}

template <class Basis>
void
HexVolMesh<Basis>::size(typename HexVolMesh::Face::size_type &s) const
{
  ASSERTMSG(synchronized_ & FACES_E,
	    "Must call synchronize FACES_E on HexVolMesh first");
  s = static_cast<typename Face::size_type>(faces_.size());
}

template <class Basis>
void
HexVolMesh<Basis>::begin(typename HexVolMesh::Cell::iterator &itr) const
{
  ASSERTMSG(synchronized_ & CELLS_E,
	    "Must call synchronize CELLS_E on HexVolMesh first");
  itr = 0;
}

template <class Basis>
void
HexVolMesh<Basis>::end(typename HexVolMesh::Cell::iterator &itr) const
{
  ASSERTMSG(synchronized_ & CELLS_E,
	    "Must call synchronize CELLS_E on HexVolMesh first");
  itr = static_cast<typename Cell::iterator>(cells_.size() >> 3);
}

template <class Basis>
void
HexVolMesh<Basis>::size(typename HexVolMesh::Cell::size_type &s) const
{
  ASSERTMSG(synchronized_ & CELLS_E,
	    "Must call synchronize CELLS_E on HexVolMesh first");
  s = static_cast<typename Cell::size_type>(cells_.size() >> 3);
}

template <class Basis>
void
HexVolMesh<Basis>::get_nodes(typename Node::array_type &array, 
			     typename Edge::index_type idx) const
{
  ASSERTMSG(synchronized_ & EDGES_E,
	    "Must call synchronize EDGES_E on HexVolMesh first");
  array.clear();
  PEdge e = edges_[idx];
  array.push_back(e.nodes_[0]);
  array.push_back(e.nodes_[1]);
}


template <class Basis>
void
HexVolMesh<Basis>::get_nodes(typename Node::array_type &array, typename Face::index_type idx) const
{
  ASSERTMSG(synchronized_ & FACES_E,
	    "Must call synchronize FACES_E on HexVolMesh first");
  array.clear();
  const PFace &f = faces_[idx];
  array.push_back(f.nodes_[0]);
  array.push_back(f.nodes_[1]);
  array.push_back(f.nodes_[2]);
  array.push_back(f.nodes_[3]);
}


template <class Basis>
void
HexVolMesh<Basis>::get_nodes(typename Node::array_type &array, typename Cell::index_type idx) const
{
  array.clear();
  array.push_back(cells_[idx * 8 + 0]);
  array.push_back(cells_[idx * 8 + 1]);
  array.push_back(cells_[idx * 8 + 2]);
  array.push_back(cells_[idx * 8 + 3]);
  array.push_back(cells_[idx * 8 + 4]);
  array.push_back(cells_[idx * 8 + 5]);
  array.push_back(cells_[idx * 8 + 6]);
  array.push_back(cells_[idx * 8 + 7]);
}

template <class Basis>
void
HexVolMesh<Basis>::get_edges(typename Edge::array_type &array, typename Face::index_type idx) const
{
  ASSERTMSG(synchronized_ & FACES_E,
	    "Must call synchronize FACES_E on HexVolMesh first");
  array.clear();
  const PFace &f = faces_[idx];
  PEdge e0(f.nodes_[0], f.nodes_[1]);
  PEdge e1(f.nodes_[1], f.nodes_[2]);
  PEdge e2(f.nodes_[2], f.nodes_[3]);
  PEdge e3(f.nodes_[3], f.nodes_[0]);

  ASSERTMSG(synchronized_ & EDGES_E,
	    "Must call synchronize EDGES_E on HexVolMesh first");
  array.push_back((*(edge_table_.find(e0))).second);
  array.push_back((*(edge_table_.find(e1))).second);
  array.push_back((*(edge_table_.find(e2))).second);
  array.push_back((*(edge_table_.find(e3))).second);
}


template <class Basis>
void
HexVolMesh<Basis>::get_edges(typename Edge::array_type &array, typename Cell::index_type idx) const
{
  array.clear();
  const int off = idx * 8;
  PEdge e00(cells_[off + 0], cells_[off + 1]);
  PEdge e01(cells_[off + 1], cells_[off + 2]);
  PEdge e02(cells_[off + 2], cells_[off + 3]);
  PEdge e03(cells_[off + 3], cells_[off + 0]);
  PEdge e04(cells_[off + 4], cells_[off + 5]);
  PEdge e05(cells_[off + 5], cells_[off + 6]);
  PEdge e06(cells_[off + 6], cells_[off + 7]);
  PEdge e07(cells_[off + 7], cells_[off + 4]);
  PEdge e08(cells_[off + 0], cells_[off + 4]);
  PEdge e09(cells_[off + 5], cells_[off + 1]);
  PEdge e10(cells_[off + 2], cells_[off + 6]);
  PEdge e11(cells_[off + 7], cells_[off + 3]);

  ASSERTMSG(synchronized_ & EDGES_E,
	    "Must call synchronize EDGES_E on HexVolMesh first");
  array.push_back((*(edge_table_.find(e00))).second);
  array.push_back((*(edge_table_.find(e01))).second);
  array.push_back((*(edge_table_.find(e02))).second);
  array.push_back((*(edge_table_.find(e03))).second);
  array.push_back((*(edge_table_.find(e04))).second);
  array.push_back((*(edge_table_.find(e05))).second);
  array.push_back((*(edge_table_.find(e06))).second);
  array.push_back((*(edge_table_.find(e07))).second);
  array.push_back((*(edge_table_.find(e08))).second);
  array.push_back((*(edge_table_.find(e09))).second);
  array.push_back((*(edge_table_.find(e10))).second);
  array.push_back((*(edge_table_.find(e11))).second);
}


template <class Basis>
void
HexVolMesh<Basis>::get_faces(typename Face::array_type &array, typename Cell::index_type idx) const
{
  array.clear();
  
  const int off = idx * 8;
  PFace f0(cells_[off + 0], cells_[off + 1], cells_[off + 2], cells_[off + 3]);
  PFace f1(cells_[off + 4], cells_[off + 5], cells_[off + 6], cells_[off + 7]);
  PFace f2(cells_[off + 0], cells_[off + 4], cells_[off + 5], cells_[off + 1]);
  PFace f3(cells_[off + 2], cells_[off + 6], cells_[off + 7], cells_[off + 3]);
  PFace f4(cells_[off + 3], cells_[off + 7], cells_[off + 4], cells_[off + 0]);
  PFace f5(cells_[off + 1], cells_[off + 5], cells_[off + 6], cells_[off + 2]);

  // operator[] not const safe...
  ASSERTMSG(synchronized_ & FACES_E,
	    "Must call synchronize FACES_E on HexVolMesh first");
  array.push_back((*(face_table_.find(f0))).second);
  array.push_back((*(face_table_.find(f1))).second);
  array.push_back((*(face_table_.find(f2))).second);
  array.push_back((*(face_table_.find(f3))).second);
  array.push_back((*(face_table_.find(f4))).second);
  array.push_back((*(face_table_.find(f5))).second);
}

template <class Basis>
bool
HexVolMesh<Basis>::get_neighbor(typename Cell::index_type &neighbor, typename Cell::index_type from,
				typename Face::index_type idx) const
{
  ASSERTMSG(synchronized_ & FACES_E,
	    "Must call synchronize FACES_E on HexVolMesh first");
  const PFace &f = faces_[idx];

  if (from == f.cells_[0]) {
    neighbor = f.cells_[1];
  } else {
    neighbor = f.cells_[0];
  }
  if (neighbor == MESH_NO_NEIGHBOR) return false;
  return true;
}

template <class Basis>
void
HexVolMesh<Basis>::get_neighbors(typename Cell::array_type &array, typename Cell::index_type idx) const
{
  typename Face::array_type faces;
  get_faces(faces, idx);
  array.clear();
  typename Face::array_type::iterator iter = faces.begin();
  while(iter != faces.end()) {
    typename Cell::index_type nbor;
    if (get_neighbor(nbor, idx, *iter)) {
      array.push_back(nbor);
    }
    ++iter;
  }
}

template <class Basis>
void
HexVolMesh<Basis>::get_neighbors(vector<typename Node::index_type> &array,
				 typename Node::index_type idx) const
{
  ASSERTMSG(synchronized_ & NODE_NEIGHBORS_E,
	    "Must call synchronize NODE_NEIGHBORS_E on HexVolMesh first");
  array.clear();
  array.insert(array.end(), node_neighbors_[idx].begin(),
	       node_neighbors_[idx].end());
}

template <class Basis>
void
HexVolMesh<Basis>::compute_node_neighbors()
{
  if (!(synchronized_ & EDGES_E)) synchronize(EDGES_E);
  node_nbor_lock_.lock();
  if (synchronized_ & NODE_NEIGHBORS_E) {
    node_nbor_lock_.unlock();
    return;
  }
  node_neighbors_.clear();
  node_neighbors_.resize(points_.size());
  typename Edge::iterator ei, eie;
  begin(ei); end(eie);
  for_each(ei, eie, FillNodeNeighbors(node_neighbors_, *this));
  synchronized_ |= NODE_NEIGHBORS_E;
  node_nbor_lock_.unlock();
}

template <class Basis>
void
HexVolMesh<Basis>::get_center(Point &p, typename Node::index_type idx) const
{
  get_point(p, idx);
}


template <class Basis>
void
HexVolMesh<Basis>::get_center(Point &result, typename Edge::index_type idx) const
{
  typename Node::array_type arr;
  get_nodes(arr, idx);
  Point p1;
  get_center(result, arr[0]);
  get_center(p1, arr[1]);
  
  result.asVector() += p1.asVector();
  result.asVector() *= 0.5;
}


template <class Basis>
void
HexVolMesh<Basis>::get_center(Point &p, typename Face::index_type idx) const
{
  typename Node::array_type nodes;
  get_nodes(nodes, idx);
  ASSERT(nodes.size() == 4);
  typename Node::array_type::iterator nai = nodes.begin();
  get_point(p, *nai);
  ++nai;
  Point pp;
  while (nai != nodes.end())
  {
    get_point(pp, *nai);
    p.asVector() += pp.asVector();
    ++nai;
  }
  p.asVector() *= (1.0 / 4.0);
}

template <class Basis>
void
HexVolMesh<Basis>::get_center(Point &p, typename Cell::index_type idx) const
{
  typename Node::array_type nodes;
  get_nodes(nodes, idx);
  ASSERT(nodes.size() == 8);
  typename Node::array_type::iterator nai = nodes.begin();
  get_point(p, *nai);
  ++nai;
  Point pp;
  while (nai != nodes.end())
  {
    get_point(pp, *nai);
    p.asVector() += pp.asVector();
    ++nai;
  }
  p.asVector() *= (1.0 / 8.0);
}

static double
distance2(const Point &p0, const Point &p1)
{
  const double dx = p0.x() - p1.x();
  const double dy = p0.y() - p1.y();
  const double dz = p0.z() - p1.z();
  return dx * dx + dy * dy + dz * dz;
}


template <class Basis>
bool
HexVolMesh<Basis>::locate(typename Node::index_type &loc, const Point &p)
{
  typename Cell::index_type ci;
  if (locate(ci, p)) { // first try the fast way.
    typename Node::array_type nodes;
    get_nodes(nodes, ci);
    
    double dmin = distance2(p, points_[nodes[0]]);
    loc = nodes[0];
    for (unsigned int i = 1; i < nodes.size(); i++)
    {
      const double d = distance2(p, points_[nodes[i]]);
      if (d < dmin)
      {
	dmin = d;
	loc = nodes[i];
      }
    }
    return true;
  }
  else
  {  // do exhaustive search.
    typename Node::iterator ni, nie;
    begin(ni);
    end(nie);
    if (ni == nie) { return false; }

    double min_dist = distance2(p, points_[*ni]);
    loc = *ni;
    ++ni;

    while (ni != nie)
    {
      const double dist = distance2(p, points_[*ni]);
      if (dist < min_dist)
      {
	loc = *ni;
	min_dist = dist;
      }
      ++ni;
    }
    return true;
  }
}


template <class Basis>
bool
HexVolMesh<Basis>::locate(typename Edge::index_type &edge, const Point &p)
{
  typename Cell::index_type cell;
  if (locate(cell, p))
  {
    typename Edge::array_type edges;
    get_edges(edges, cell);

    if (edges.size() == 0) { return false; }

    edge = edges[0];
    Point loc;
    get_center(loc, edges[0]);
    double mindist = distance2(p, loc);
    for (unsigned int i = 0; i < edges.size(); i++)
    {
      get_center(loc, edges[i]);
      const double dist = distance2(p, loc);
      if (dist < mindist)
      {
	edge = edges[i];
	mindist = dist;
      }
    }
    return true;
  }
  return false;
}


template <class Basis>
bool
HexVolMesh<Basis>::locate(typename Face::index_type &face, const Point &p)
{
  typename Cell::index_type cell;
  if (locate(cell, p))
  {
    typename Face::array_type faces;
    get_faces(faces, cell);

    if (faces.size() == 0) { return false; }

    face = faces[0];
    Point loc;
    get_center(loc, faces[0]);
    double mindist = distance2(p, loc);
    for (unsigned int i = 0; i < faces.size(); i++)
    {
      get_center(loc, faces[i]);
      const double dist = distance2(p, loc);
      if (dist < mindist)
      {
	face = faces[i];
	mindist = dist;
      }
    }
    return true;
  }
  return false;
}


template <class Basis>
bool
HexVolMesh<Basis>::locate(typename Cell::index_type &cell, const Point &p)
{
  // Check last cell found first.  Copy cache to cell first so that we
  // don't care about thread safeness, such that worst case on
  // context switch is that cache is not found.
  cell = locate_cache_;
  if (cell > typename Cell::index_type(0) &&
      cell < typename Cell::index_type(cells_.size()/8) &&
      inside8_p(cell, p))
  {
      return true;
  }
  

  if (!(synchronized_ & LOCATE_E))
    synchronize(LOCATE_E);
  ASSERT(grid_.get_rep());

  unsigned int *iter, *end;
  if (grid_->lookup(&iter, &end, p))
  {
    while (iter != end)
    {
      if (inside8_p(typename Cell::index_type(*iter), p))
      {
	cell = typename Cell::index_type(*iter);
	locate_cache_ = cell;
	return true;
      }
      ++iter;
    }
  }

  return false;
}

//===================================================================

// area3D_Polygon(): computes the area of a 3D planar polygon
//    Input:  int n = the number of vertices in the polygon
//            Point* V = an array of n+2 vertices in a plane
//                       with V[n]=V[0] and V[n+1]=V[1]
//            Point N = unit normal vector of the polygon's plane
//    Return: the (float) area of the polygon

// Copyright 2000, softSurfer (www.softsurfer.com)
// This code may be freely used and modified for any purpose
// providing that this copyright notice is included with it.
// SoftSurfer makes no warranty for this code, and cannot be held
// liable for any real or imagined damage resulting from its use.
// Users of this code must verify correctness for their application.

template <class Basis>
double
HexVolMesh<Basis>::polygon_area(const typename Node::array_type &ni, const Vector N) const
{
  double area = 0;
  double an, ax, ay, az;  // abs value of normal and its coords
  int   coord;           // coord to ignore: 1=x, 2=y, 3=z
    unsigned int   i, j, k;         // loop indices
    const unsigned int n = ni.size();

    // select largest abs coordinate to ignore for projection
    ax = (N.x()>0 ? N.x() : -N.x());     // abs x-coord
    ay = (N.y()>0 ? N.y() : -N.y());     // abs y-coord
    az = (N.z()>0 ? N.z() : -N.z());     // abs z-coord

    coord = 3;                     // ignore z-coord
    if (ax > ay) {
        if (ax > az) coord = 1;    // ignore x-coord
    }
    else if (ay > az) coord = 2;   // ignore y-coord

    // compute area of the 2D projection
    for (i=1, j=2, k=0; i<=n; i++, j++, k++)
        switch (coord) {
        case 1:
            area += (points_[ni[i%n]].y() *
		     (points_[ni[j%n]].z() - points_[ni[k%n]].z()));
            continue;
        case 2:
            area += (points_[ni[i%n]].x() * 
		     (points_[ni[j%n]].z() - points_[ni[k%n]].z()));
            continue;
        case 3:
            area += (points_[ni[i%n]].x() * 
		     (points_[ni[j%n]].y() - points_[ni[k%n]].y()));
            continue;
        }

    // scale to get area before projection
    an = sqrt( ax*ax + ay*ay + az*az);  // length of normal vector
    switch (coord) {
    case 1:
        area *= (an / (2*ax));
        break;
    case 2:
        area *= (an / (2*ay));
        break;
    case 3:
        area *= (an / (2*az));
    }
    return area;
}

template <class Basis>
double
HexVolMesh<Basis>::pyramid_volume(const typename Node::array_type &face, const Point &p) const
{
  Vector e1(points_[face[1]]-points_[face[0]]);
  Vector e2(points_[face[1]]-points_[face[2]]);
  if (Cross(e1,e2).length2()>0.0) {
    Plane plane(points_[face[0]], points_[face[1]], points_[face[2]]);
    //double dist = plane.eval_point(p);
    return fabs(plane.eval_point(p)*polygon_area(face,plane.normal())*0.25);
  }
  Vector e3(points_[face[3]]-points_[face[2]]);
  if (Cross(e2,e3).length2()>0.0) {
    Plane plane(points_[face[1]], points_[face[2]], points_[face[3]]);
    //double dist = plane.eval_point(p);
    return fabs(plane.eval_point(p)*polygon_area(face,plane.normal())*0.25);
  }
  return 0.0;
}


static double
tri_area(const Point &a, const Point &b, const Point &c)
{
  return Cross(b-a, c-a).length();
}


template <class Basis>
void
HexVolMesh<Basis>::get_face_weights(double *w, const typename Node::array_type &nodes,
				    const Point &p, int i0, int i1, int i2, int i3)
{
  for (unsigned int j = 0; j < 8; j++)
  {
    w[j] = 0.0;
  }

  const Point &p0 = point(nodes[i0]);
  const Point &p1 = point(nodes[i1]);
  const Point &p2 = point(nodes[i2]);
  const Point &p3 = point(nodes[i3]);

  const double a0 = tri_area(p, p0, p1);
  if (a0 < 1.0e-6)
  {
    const Vector v0 = p0 - p1;
    const Vector v1 = p - p1;
    w[i0] = Dot(v0, v1) / Dot(v0, v0);
    w[i1] = 1.0 - w[i0];
    return;
  }
  const double a1 = tri_area(p, p1, p2);
  if (a1 < 1.0e-6)
  {
    const Vector v0 = p1 - p2;
    const Vector v1 = p - p2;
    w[i1] = Dot(v0, v1) / Dot(v0, v0);
    w[i2] = 1.0 - w[i1];
    return;
  }
  const double a2 = tri_area(p, p2, p3);
  if (a2 < 1.0e-6)
  {
    const Vector v0 = p2 - p3;
    const Vector v1 = p - p3;
    w[i2] = Dot(v0, v1) / Dot(v0, v0);
    w[i3] = 1.0 - w[i2];
    return;
  }
  const double a3 = tri_area(p, p3, p0);
  if (a3 < 1.0e-6)
  {
    const Vector v0 = p3 - p0;
    const Vector v1 = p - p0;
    w[i3] = Dot(v0, v1) / Dot(v0, v0);
    w[i0] = 1.0 - w[i3];
    return;
  }

  w[i0] = tri_area(p0, p1, p2) / (a0 * a3);
  w[i1] = tri_area(p1, p2, p0) / (a1 * a0);
  w[i2] = tri_area(p2, p3, p1) / (a2 * a1);
  w[i3] = tri_area(p3, p0, p2) / (a3 * a2);

  const double suminv = 1.0 / (w[i0] + w[i1] + w[i2] + w[i3]);
  w[i0] *= suminv;
  w[i1] *= suminv;
  w[i2] *= suminv;
  w[i3] *= suminv;
}
  


template <class Basis>
void
HexVolMesh<Basis>::compute_grid()
{
  grid_lock_.lock();
  if (synchronized_ & LOCATE_E) {
    grid_lock_.unlock();
    return;
  }
  if (grid_.get_rep() != 0) {grid_lock_.unlock(); return;} // only create once.

  BBox bb = get_bounding_box();
  if (!bb.valid()) { grid_lock_.unlock(); return; }

  // Cubed root of number of cells to get a subdivision ballpark.
  const double one_third = 1.L/3.L;
  typename Cell::size_type csize;  size(csize);
  const int s = ((int)ceil(pow((double)csize , one_third))) / 2 + 1;
  const Vector cell_epsilon = bb.diagonal() * (0.01 / s);
  bb.extend(bb.min() - cell_epsilon*2);
  bb.extend(bb.max() + cell_epsilon*2);

  SearchGridConstructor sgc(s, s, s, bb.min(), bb.max());

  BBox box;
  typename Node::array_type nodes;
  typename Cell::iterator ci, cie;
  begin(ci); end(cie);
  while(ci != cie)
  {
    get_nodes(nodes, *ci);

    box.reset();
    for (unsigned int i = 0; i < nodes.size(); i++)
    {
      box.extend(points_[nodes[i]]);
    }
    const Point padmin(box.min() - cell_epsilon);
    const Point padmax(box.max() + cell_epsilon);
    box.extend(padmin);
    box.extend(padmax);

    sgc.insert(*ci, box);

    ++ci;
  }

  grid_ = scinew SearchGrid(sgc);

  synchronized_ |= LOCATE_E;
  grid_lock_.unlock();
}



template <class Basis>
bool
HexVolMesh<Basis>::inside8_p(typename Cell::index_type i, const Point &p) const
{
  typename Face::array_type faces;
  get_faces(faces, i);
  
  Point center;
  get_center(center, i);

  for (unsigned int i = 0; i < faces.size(); i++)
  {
    typename Node::array_type nodes;
    get_nodes(nodes, faces[i]);
    Point p0, p1, p2;
    get_center(p0, nodes[0]);
    get_center(p1, nodes[1]);
    get_center(p2, nodes[2]);

    const Vector v0(p1 - p0), v1(p2 - p0);
    const Vector normal = Cross(v0, v1);
    const Vector off0(p - p0);
    const Vector off1(center - p0);
    if (Dot(off0, normal) * Dot(off1, normal) < 0.0)
    {
      return false;
    }
  }
  return true;
}
    


//! return the volume of the hex.
template <class Basis>
double
HexVolMesh<Basis>::get_gradient_basis(typename Cell::index_type /*ci*/, 
				      Vector& /*g0*/, Vector& /*g1*/,
				      Vector& /*g2*/, Vector& /*g3*/,
				      Vector& /*g4*/, Vector& /*g5*/,
				      Vector& /*g6*/, Vector& /*g7*/)
{
  ASSERTFAIL("get_gradient_basis not implemented for hexes");
}

template <class Basis>
typename HexVolMesh<Basis>::Node::index_type
HexVolMesh<Basis>::add_find_point(const Point &p, double err)
{
  typename Node::index_type i;
  if (locate(i, p) && distance2(points_[i], p) < err)
  {
    return i;
  }
  else
  {
    points_.push_back(p);
    return static_cast<typename Node::index_type>(points_.size() - 1);
  }
}


template <class Basis>
void
HexVolMesh<Basis>::add_hex(typename Node::index_type a, typename Node::index_type b, 
			   typename Node::index_type c, typename Node::index_type d,
			   typename Node::index_type e, typename Node::index_type f,
			   typename Node::index_type g, typename Node::index_type h)
{
  cells_.push_back(a);
  cells_.push_back(b);
  cells_.push_back(c);
  cells_.push_back(d);
  cells_.push_back(e);
  cells_.push_back(f);
  cells_.push_back(g);
  cells_.push_back(h);
}


template <class Basis>
typename HexVolMesh<Basis>::Node::index_type
HexVolMesh<Basis>::add_point(const Point &p)
{
  points_.push_back(p);
  return static_cast<typename Node::index_type>(points_.size() - 1);
}


template <class Basis>
void
HexVolMesh<Basis>::add_hex(const Point &p0, const Point &p1, 
			   const Point &p2, const Point &p3,
			   const Point &p4, const Point &p5,
			   const Point &p6, const Point &p7)
{
  add_hex(add_find_point(p0), add_find_point(p1),
	  add_find_point(p2), add_find_point(p3),
	  add_find_point(p4), add_find_point(p5),
	  add_find_point(p6), add_find_point(p7));
}

template <class Basis>
typename HexVolMesh<Basis>::Elem::index_type
HexVolMesh<Basis>::add_elem(typename Node::array_type a)
{
  cells_.push_back(a[0]);
  cells_.push_back(a[1]);
  cells_.push_back(a[2]);
  cells_.push_back(a[3]);
  cells_.push_back(a[4]);
  cells_.push_back(a[5]);
  cells_.push_back(a[6]);
  cells_.push_back(a[7]);
  return static_cast<typename Elem::index_type>((cells_.size() - 1) >> 3);
}


#define HEXVOLMESH_VERSION 3

template <class Basis>
void
HexVolMesh<Basis>::io(Piostream &stream)
{
  const int version = stream.begin_class(type_name(-1), HEXVOLMESH_VERSION);
  Mesh::io(stream);
  
  SCIRun::Pio(stream, points_);
  SCIRun::Pio(stream, cells_);
  if (version == 1)
  {
    vector<under_type>  face_neighbors;
    SCIRun::Pio(stream, face_neighbors);
  }

  if (version >= 3) { 
    basis_.io(stream);
  }

  stream.end_class();

  if (stream.reading())
  {
    synchronized_ = NODES_E | CELLS_E;
  }
}



template <class Basis>
const TypeDescription* get_type_description(HexVolMesh<Basis> *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    const TypeDescription *sub = SCIRun::get_type_description((Basis*)0);
    TypeDescription::td_vec *subs = scinew TypeDescription::td_vec(1);
    (*subs)[0] = sub;
    td = scinew TypeDescription(HexVolMesh<Basis>::type_name(0), subs,
				string(__FILE__),
				"SCIRun");
  }
  return td;
}

template <class Basis>
const TypeDescription*
HexVolMesh<Basis>::get_type_description() const
{
  return SCIRun::get_type_description((HexVolMesh<Basis> *)0);
}


template <class Basis>
const TypeDescription*
HexVolMesh<Basis>::node_type_description()
{
  static TypeDescription *td = 0;
  if (!td)
  {
    const TypeDescription *me = 
      SCIRun::get_type_description((HexVolMesh<Basis> *)0);
    td = scinew TypeDescription(me->get_name() + "::Node",
				string(__FILE__),
				"SCIRun");
  }
  return td;
}

template <class Basis>
const TypeDescription*
HexVolMesh<Basis>::edge_type_description()
{
  static TypeDescription *td = 0;
  if (!td)
  {
    const TypeDescription *me = 
      SCIRun::get_type_description((HexVolMesh<Basis> *)0);
    td = scinew TypeDescription(me->get_name() + "::Edge",
				string(__FILE__),
				"SCIRun");
  }
  return td;
}

template <class Basis>
const TypeDescription*
HexVolMesh<Basis>::face_type_description()
{
  static TypeDescription *td = 0;
  if (!td)
  {
    const TypeDescription *me = 
      SCIRun::get_type_description((HexVolMesh<Basis> *)0);
    td = scinew TypeDescription(me->get_name() + "::Face",
				string(__FILE__),
				"SCIRun");
  }
  return td;
}

template <class Basis>
const TypeDescription*
HexVolMesh<Basis>::cell_type_description()
{
  static TypeDescription *td = 0;
  if (!td)
  {
    const TypeDescription *me = 
      SCIRun::get_type_description((HexVolMesh<Basis> *)0);
    td = scinew TypeDescription(me->get_name() + "::Cell",
				string(__FILE__),
				"SCIRun");
  }
  return td;
}


} // namespace SCIRun


#endif // SCI_project_HexVolMesh_h
