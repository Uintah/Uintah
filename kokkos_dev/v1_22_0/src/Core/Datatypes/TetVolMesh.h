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
#include <Core/Datatypes/LatVolField.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>
#include <Core/Persistent/PersistentSTL.h>
#include <Core/Math/MinMax.h>
#include <sci_hash_set.h>
#include <sci_hash_map.h>
#include <sgi_stl_warnings_off.h>
#include <set>
#include <sgi_stl_warnings_on.h>


namespace SCIRun {

template <class T, int CAPACITY>
class StackVector {
public:
  int size_;
  T data_[CAPACITY];

  typedef T        value_type;
  typedef T*       pointer;
  typedef T&       reference;
  typedef const T& const_reference;
  typedef size_t   size_type;
  //typedef difference_type;
  typedef T*       iterator;
  typedef const T* const_iterator;
  //typedef reverse_iterator;
  //typedef const_reverse_iterator;

  iterator begin() { return data_; }
  iterator end() { return data_+size_; }
  const_iterator begin() const { return data_; }
  const_iterator end() const { return data_+size_; }
  size_type size() const { ASSERT(size_ <= CAPACITY); return size_; }
  size_type max_size() const { return CAPACITY; }
  size_type capacity() const { return CAPACITY; }
  bool empty() const { return size_; }
  reference operator[](size_type n) { return data_[n]; }
  const_reference operator[](size_type n) const { return data_[n]; }
  void resize(size_type s) { ASSERT(s <= CAPACITY); size_ = s; }
  void push_back(T t) { ASSERT(size_ < CAPACITY); data_[size_] = t; size_++; }
  void clear() { size_ = 0; }
  StackVector() { size_ = 0xDEADBABE; }
  StackVector(size_type s) { ASSERT(s <= CAPACITY); size_ = s; }
  ~StackVector() {}
};



class SCICORESHARE TetVolMesh : public Mesh
{
public:
  typedef unsigned int under_type;

  //! Index and Iterator types required for Mesh Concept.
  struct Node {
    typedef NodeIndex<under_type>       index_type;
    typedef NodeIterator<under_type>    iterator;
    typedef NodeIndex<under_type>       size_type;
    typedef StackVector<index_type, 10> array_type; // 10 = quadratic size
    //typedef vector<index_type>          array_type;
  };					

  struct Cell {				
    typedef CellIndex<under_type>       index_type;
    typedef CellIterator<under_type>    iterator;
    typedef CellIndex<under_type>       size_type;
    typedef vector<index_type>          array_type;
  };

  // Used for hashing operations below
  static const int sizeof_uint = sizeof(unsigned int) * 8; // in bits


  //! An edge is indexed via the cells structure.
  //! There are 6 unique edges in each cell of 4 nodes.
  //! Therefore, the edge index / 6 == cell index
  //! And, the edge index % 6 == which edge in that cell
  //! Edges indices are stored in a hash_set and a hash_multiset.
  //! The hash_set stores shared edges only once.
  //! The hash_multiset stores all shared edges together.   
  struct Edge {				
    typedef EdgeIndex<under_type>       index_type;

    //! edgei return the two nodes make the edge
    static pair<Node::index_type, Node::index_type> edgei(index_type idx)
    { 
      const int b = (idx / 6) * 4;
      switch (idx % 6)
      {
      case 0: return pair<Node::index_type,Node::index_type>(b+0,b+1);
      case 1: return pair<Node::index_type,Node::index_type>(b+0,b+2);
      case 2: return pair<Node::index_type,Node::index_type>(b+0,b+3);
      case 3: return pair<Node::index_type,Node::index_type>(b+1,b+2);
      case 4: return pair<Node::index_type,Node::index_type>(b+2,b+3);
      default:
      case 5: return pair<Node::index_type,Node::index_type>(b+1,b+3);
      }
    }
    
    static index_type opposite_edge(index_type idx) 
    {
      const int cell = (idx / 6);
      switch (idx % 6)
      {
      case 0: return cell * 6 + 4;
      case 1: return cell * 6 + 5;
      case 2: return cell * 6 + 3;
      case 3: return cell * 6 + 2;
      case 4: return cell * 6 + 0;
      default:	  
      case 5: return cell * 6 + 1;
      }      
    }

    //! A fucntor that returns a boolean indicating weather two
    //! edges indices share the same nodes, and thus the same edge in space
    //! Used as a template parameter to STL containers typedef'd below
    struct eqEdge : public binary_function<index_type, index_type, bool>
    {
    private:
      const vector<under_type> &cells_;
    public:
      eqEdge(const vector<under_type> &cells) : 
	cells_(cells) {};
      bool operator()(index_type ei1, index_type ei2) const
      {
	const pair<index_type, index_type> e1 = edgei(ei1), e2 = edgei(ei2);
	return (Max(cells_[e1.first], cells_[e1.second]) == 
		Max(cells_[e2.first], cells_[e2.second]) &&
		Min(cells_[e1.first], cells_[e1.second]) == 
		Min(cells_[e2.first], cells_[e2.second]));
      };
    };
      
#ifdef HAVE_HASH_SET
    //! A functor that hashes an edge index according to the node
    //! indices of that edge
    //! Used as a template parameter to STL hash_[set,map] containers
    struct CellEdgeHasher : public unary_function<size_t, index_type>
    {
    private:
      const vector<under_type> &cells_;
    public:
      CellEdgeHasher(const vector<under_type> &cells) : 
	cells_(cells) {};
      static const int size = sizeof_uint / 2; // in bits
      static const int mask = (~(unsigned int)0) >> (sizeof_uint - size);
      size_t operator()(index_type cell) const 
      { 
	pair<index_type,index_type> e = edgei(cell);
	const int n0 = cells_[e.first] & mask;
	const int n1 = cells_[e.second] & mask;
	return Min(n0, n1) << size | Max(n0, n1);
      }
    };   

    typedef hash_multiset<index_type, CellEdgeHasher,eqEdge> HalfEdgeSet;
    typedef hash_set<index_type, CellEdgeHasher, eqEdge> EdgeSet;
#else
    typedef multiset<index_type, eqEdge> HalfEdgeSet;
    typedef set<index_type, eqEdge> EdgeSet;
#endif
    //! This iterator will traverse each shared edge once in no 
    //! particular order.
    typedef EdgeSet::iterator		iterator;
    typedef EdgeIndex<under_type>       size_type;
    typedef vector<index_type>          array_type;
  };					
  

  //! A face is directly opposite the same indexed node in the cells_ structure
  struct Face 
  {				
    typedef FaceIndex<under_type>       index_type;
    
    struct eqFace : public binary_function<index_type, index_type, bool>
    {
    private:
      const vector<under_type> &cells_;
    public:
      eqFace(const vector<under_type> &cells) : 
	cells_(cells) {};
      bool operator()(index_type fi1, index_type fi2) const
      {
	const int f1_offset = fi1 % 4;
	const int f1_base = fi1 - f1_offset;
	const under_type f1_n0 = cells_[f1_base + (f1_offset < 1 ? 1 : 0)];
	const under_type f1_n1 = cells_[f1_base + (f1_offset < 2 ? 2 : 1)];
	const under_type f1_n2 = cells_[f1_base + (f1_offset < 3 ? 3 : 2)];
	const int f2_offset = fi2 % 4;
	const int f2_base = fi2 - f2_offset;
	const under_type f2_n0 = cells_[f2_base + (f2_offset < 1 ? 1 : 0)];
	const under_type f2_n1 = cells_[f2_base + (f2_offset < 2 ? 2 : 1)];
	const under_type f2_n2 = cells_[f2_base + (f2_offset < 3 ? 3 : 2)];

	return (Max(f1_n0, f1_n1, f1_n2) == Max(f2_n0, f2_n1, f2_n2) &&
		Mid(f1_n0, f1_n1, f1_n2) == Mid(f2_n0, f2_n1, f2_n2) &&
		Min(f1_n0, f1_n1, f1_n2) == Min(f2_n0, f2_n1, f2_n2));
      }
    };
    
#ifdef HAVE_HASH_SET
    struct CellFaceHasher: public unary_function<size_t, index_type>
    {
    private:
      const vector<under_type> &cells_;
    public:
      CellFaceHasher(const vector<under_type> &cells) : 
	cells_(cells) {};
      static const int size = sizeof_uint / 3; // in bits
      static const int mask = (~(unsigned int)0) >> (sizeof_uint - size);
      size_t operator()(index_type cell) const 
      {
	const int offset = cell % 4;
	const int base = cell - offset;
	const under_type n0 = cells_[base + (offset < 1 ? 1 : 0)] & mask;
	const under_type n1 = cells_[base + (offset < 2 ? 2 : 1)] & mask;
	const under_type n2 = cells_[base + (offset < 3 ? 3 : 2)] & mask;      
	return Min(n0,n1,n2)<<size*2 | Mid(n0,n1,n2)<<size | Max(n0,n1,n2);
      }
    };
    typedef hash_multiset<index_type, CellFaceHasher,eqFace> HalfFaceSet;
    typedef hash_set<index_type, CellFaceHasher, eqFace> FaceSet;
#else
    typedef multiset<index_type, eqFace> HalfFaceSet;
    typedef set<index_type, eqFace> FaceSet;
#endif
    typedef FaceSet::iterator		iterator;
    typedef FaceIndex<under_type>       size_type;
    typedef vector<index_type>          array_type;
  };					
  

  typedef Cell Elem;
  enum { ELEMENTS_E = CELLS_E };

  TetVolMesh();
  TetVolMesh(const TetVolMesh &copy);
  virtual TetVolMesh *clone() { return new TetVolMesh(*this); }
  virtual ~TetVolMesh();

  virtual BBox get_bounding_box() const;
  virtual void transform(const Transform &t);

  bool get_dim(vector<unsigned int>&) const { return false;  }

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

  void get_nodes(Node::array_type &array, Edge::index_type idx) const;
  void get_nodes(Node::array_type &array, Face::index_type idx) const;
  void get_nodes(Node::array_type &array, Cell::index_type idx) const;

  bool get_edge(Edge::index_type &ei, Cell::index_type ci, 
		Node::index_type n1, Node::index_type n2) const;

  void get_edges(Edge::array_type &array, Node::index_type idx) const;
  void get_edges(Edge::array_type &array, Face::index_type idx) const;
  void get_edges(Edge::array_type &array, Cell::index_type idx) const;

  void get_faces(Face::array_type &array, Node::index_type idx) const;
  void get_faces(Face::array_type &array, Edge::index_type idx) const;
  void get_faces(Face::array_type &array, Cell::index_type idx) const;

  //! not part of the mesh concept but rather specific to tetvol
  //! Return in fi the face that is opposite the node ni in the cell ci.
  //! Return false if bad input, else true indicating the face was found.
  bool get_face_opposite_node(Face::index_type &fi, Cell::index_type ci, 
			      Node::index_type ni) const;

  void get_cells(Cell::array_type &array, Node::index_type idx) const;
  void get_cells(Cell::array_type &array, Edge::index_type idx) const;
  void get_cells(Cell::array_type &array, Face::index_type idx) const;
  
  bool get_neighbor(Cell::index_type &neighbor, Cell::index_type from,
		   Face::index_type idx) const;
  // Use this one instead
  bool get_neighbor(Face::index_type &neighbor, Face::index_type idx) const;
  void get_neighbors(Cell::array_type &array, Cell::index_type idx) const;
  // This uses vector instead of array_type because we cannot make any
  // guarantees about the maximum valence size of any node in the
  // mesh.
  void get_neighbors(vector<Node::index_type> &array,
		     Node::index_type idx) const;

  void get_center(Point &result, Node::index_type idx) const { result = points_[idx]; }
  void get_center(Point &result, Edge::index_type idx) const;
  void get_center(Point &result, Face::index_type idx) const;
  void get_center(Point &result, Cell::index_type idx) const;


  //! Get the size of an elemnt (length, area, volume)
  double get_size(Node::index_type /*idx*/) const { return 0.0; }
  double get_size(Edge::index_type idx) const 
  {
    Node::array_type ra;
    get_nodes(ra, idx);
    const Point &p0 = point(ra[0]);
    const Point &p1 = point(ra[1]);
    return (p1.asVector() - p0.asVector()).length();
  }
  double get_size(Face::index_type idx) const
  {
    Node::array_type ra;
    get_nodes(ra,idx);
    const Point &p0 = point(ra[0]);
    const Point &p1 = point(ra[1]);
    const Point &p2 = point(ra[2]);
    return (Cross(p0-p1,p2-p0)).length()*0.5;
  }
  double get_size(Cell::index_type idx) const
  {
    Node::array_type ra;
    get_nodes(ra,idx);
    const Point &p0 = point(ra[0]);
    const Point &p1 = point(ra[1]);
    const Point &p2 = point(ra[2]);
    const Point &p3 = point(ra[3]);

    return fabs(Dot(Cross(p1-p0,p2-p0),p3-p0)) / 6.0;
  } 

  double get_length(Edge::index_type idx) const { return get_size(idx); };
  double get_area(Face::index_type idx) const   { return get_size(idx); };
  double get_volume(Cell::index_type idx) const { return get_size(idx); };


  unsigned int get_valence(Node::index_type idx) const
  {
    vector<Node::index_type> arr;
    get_neighbors(arr, idx);
    return arr.size();
  }
  unsigned int get_valence(Edge::index_type /*idx*/) const { return 0; }
  unsigned int get_valence(Face::index_type idx) const
  {
    Face::index_type tmp;
    return (get_neighbor(tmp, idx) ? 1 : 0);
  }
  unsigned int get_valence(Cell::index_type idx) const 
  {
    Cell::array_type arr;
    get_neighbors(arr, idx);
    return arr.size();
  }


  //! return false if point is out of range.
  bool locate(Node::index_type &loc, const Point &p);
  bool locate(Edge::index_type &loc, const Point &p);
  bool locate(Face::index_type &loc, const Point &p);
  bool locate(Cell::index_type &loc, const Point &p);

  int get_weights(const Point &p, Node::array_type &l, double *w);
  int get_weights(const Point & , Edge::array_type & , double * )
  {ASSERTFAIL("TetVolMesh::get_weights for edges isn't supported"); }
  int get_weights(const Point & , Face::array_type & , double * )
  {ASSERTFAIL("TetVolMesh::get_weights for faces isn't supported"); }
  int get_weights(const Point &p, Cell::array_type &l, double *w);

  void get_point(Point &result, Node::index_type index) const
  { result = points_[index]; }

  void get_normal(Vector &/* result */, Node::index_type /* index */) const
  { ASSERTFAIL("not implemented") }

  void set_point(const Point &point, Node::index_type index)
  { points_[index] = point; }

  void get_random_point(Point &p, Cell::index_type ei, int seed=0) const;

  //! the double return val is the volume of the tet.
  double get_gradient_basis(Cell::index_type ci, Vector& g0, Vector& g1,
			    Vector& g2, Vector& g3);

  //! function to test if at least one of cell's nodes are in supplied range
  inline bool test_nodes_range(Cell::index_type ci, unsigned int sn,
			       unsigned int en)
  {
    if (cells_[ci*4]>=sn && cells_[ci*4]<en
	|| cells_[ci*4+1]>=sn && cells_[ci*4+1]<en
	|| cells_[ci*4+2]>=sn && cells_[ci*4+2]<en
	|| cells_[ci*4+3]>=sn && cells_[ci*4+3]<en)
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


  void flip(Cell::index_type, bool recalculate = false);
  void rewind_mesh();


  virtual bool		synchronize(unsigned int);

  //! Persistent IO
  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  static  const string type_name(int n = -1);
  virtual const TypeDescription *get_type_description() const;


  // Extra functionality needed by this specific geometry.
  void			set_nodes(Node::array_type &, Cell::index_type);

  Node::index_type	add_point(const Point &p);
  Node::index_type	add_find_point(const Point &p, double err = 1.0e-3);

  Elem::index_type	add_tet(Node::index_type a, 
				Node::index_type b,
				Node::index_type c,
				Node::index_type d);
  Elem::index_type	add_tet(const Point &p0,
				const Point &p1,
				const Point &p2,
				const Point &p3);
  Elem::index_type	add_elem(Node::array_type a);

  void node_reserve(size_t s) { points_.reserve(s); }
  void elem_reserve(size_t s) { cells_.reserve(s*4); }


  //! Subdivision methods
  //! given 2 cells that share a face, split the 2 tets into 3 by connecting
  //! the 2 nodes not on the shared face.
  bool                  split_2_to_3(Cell::array_type &new_tets, 
				     Node::index_type &c1_node,
				     Node::index_type &c2_node,
				     Cell::index_type c1, 
				     Cell::index_type c2, 
				     Face::index_type between);
  //! given a cell, and the face index which is hte boundary face,
  //! split the cell into 3, by adding a point at the center of the boundary
  //! face.
  bool                  split_cell_at_boundary(Cell::array_type &new_tets, 
					       Node::index_type &new_node, 
					       Cell::index_type ci, 
					       Face::index_type bface);

  //! given an edge that has exactly 3 tets sharing the edge, create 2 tets in 
  //! thier place.  The 3 points not on the edge become a face shared between 
  //! the new 2 tet combo. 
  //! Warning: this invalidates iterators.  removed has the invalid cell index
  bool                  combine_3_to_2(Cell::index_type &removed,
				       Edge::index_type shared_edge);

  bool                  combine_4_to_1_cell(Cell::array_type &split_tets, 
					    set<unsigned int> &removed_tets,
					    set<unsigned int> &removed_nodes);

  void                  nbors_from_2_to_3_split(Cell::index_type ci, 
						Cell::array_type &split_tets);
  void                  nbors_from_center_split(Cell::index_type ci, 
						Cell::array_type &split_tets);
    
  bool                  insert_node_in_cell(Cell::array_type &tets, 
					    Cell::index_type ci, 
					    Node::index_type &ni,
					    const Point &p);
  bool			insert_node(const Point &p);
  Node::index_type	insert_node_watson(const Point &p, 
					   Cell::array_type *new_cells = 0, 
					   Cell::array_type *mod_cells = 0);
  void			bisect_element(const Cell::index_type c);


  void			delete_cells(set<unsigned int> &to_delete);
  void			delete_nodes(set<unsigned int> &to_delete);
  
  bool			is_edge(Node::index_type n0,
				Node::index_type n1,
				Edge::array_type *edges = 0);

  bool			is_face(Node::index_type n0,
				Node::index_type n1,
				Node::index_type n2,
				Face::array_type *faces = 0);


  virtual bool		is_editable() const { return true; }
  virtual int           dimensionality() const { return 3; }

protected:
  const Point &point(Node::index_type idx) const { return points_[idx]; }
  
  void			compute_node_neighbors();
  void			compute_edges();
  void			compute_faces();
  void			compute_grid();

  void			orient(Cell::index_type ci);
  bool			inside(Cell::index_type idx, const Point &p);
  pair<Point,double>	circumsphere(const Cell::index_type);

  //! Used to recompute data for individual cells
  void			create_cell_edges(Cell::index_type);
  void			delete_cell_edges(Cell::index_type);
  void			create_cell_faces(Cell::index_type);
  void			delete_cell_faces(Cell::index_type);
  void			create_cell_node_neighbors(Cell::index_type);
  void			delete_cell_node_neighbors(Cell::index_type);

 
  Elem::index_type	mod_tet(Cell::index_type cell, 
				Node::index_type a,
				Node::index_type b,
				Node::index_type c,
				Node::index_type d);



  //! all the vertices
  vector<Point>		points_;
  Mutex			points_lock_;

  //! each 4 indecies make up a tet
  vector<under_type>	cells_;
  Mutex			cells_lock_;

  typedef LockingHandle<Edge::HalfEdgeSet> HalfEdgeSetHandle;
  typedef LockingHandle<Edge::EdgeSet> EdgeSetHandle;
#ifdef HAVE_HASH_SET
  Edge::CellEdgeHasher	edge_hasher_;
#endif
  Edge::eqEdge		edge_eq_;
  Edge::HalfEdgeSet	all_edges_;
#if defined(__digital__) || defined(_AIX) || defined(__ECC)
  mutable
#endif
  Edge::EdgeSet		edges_;
  Mutex			edge_lock_;

  typedef LockingHandle<Face::HalfFaceSet> HalfFaceSetHandle;
  typedef LockingHandle<Face::FaceSet> FaceSetHandle;
#ifdef HAVE_HASH_SET
  Face::CellFaceHasher	face_hasher_;
#endif
  Face::eqFace		face_eq_;
  Face::HalfFaceSet	all_faces_;
#if defined(__digital__) || defined(_AIX) || defined(__ECC)
  mutable
#endif
  Face::FaceSet		faces_;
  Mutex			face_lock_;

  typedef vector<vector<Cell::index_type> > NodeNeighborMap;
  //  typedef LockingHandle<NodeMap> NodeMapHandle;
  NodeNeighborMap	node_neighbors_;
  Mutex			node_neighbor_lock_;

  //! This grid is used as an acceleration structure to expedite calls
  //!  to locate.  For each cell in the grid, we store a list of which
  //!  tets overlap that grid cell -- to find the tet which contains a
  //!  point, we simply find which grid cell contains that point, and
  //!  then search just those tets that overlap that grid cell.
  //!  The grid is only built if synchronize(Mesh::LOCATE_E) is called.
  typedef LockingHandle<LatVolField<vector<Cell::index_type> > > grid_handle;
  grid_handle           grid_;
  Mutex                 grid_lock_; // Bad traffic!

  unsigned int		synchronized_;
public:
  inline grid_handle get_grid() {return grid_;}

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
  dirty_ = true;
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
  dirty_ = true;
}


const TypeDescription* get_type_description(TetVolMesh *);
const TypeDescription* get_type_description(TetVolMesh::Node *);
const TypeDescription* get_type_description(TetVolMesh::Edge *);
const TypeDescription* get_type_description(TetVolMesh::Face *);
const TypeDescription* get_type_description(TetVolMesh::Cell *);

} // namespace SCIRun


#endif // SCI_project_TetVolMesh_h
