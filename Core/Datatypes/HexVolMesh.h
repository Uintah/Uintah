/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

/*
 *  HexVolMesh.h: Templated Mesh defined on a 3D Irregular Grid
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   January 2001
 *
 *
 */

#ifndef SCI_project_HexVolMesh_h
#define SCI_project_HexVolMesh_h 1

#include <Core/Datatypes/Mesh.h>
#include <Core/Datatypes/FieldIterator.h>
#include <Core/Containers/StackVector.h>
#include <Core/Persistent/PersistentSTL.h>
#include <Core/Datatypes/SearchGrid.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Geometry/Plane.h>
#include <Core/Geometry/CompGeom.h>
#include <vector>
#include <set>
#include <sci_hash_map.h>
#include <algorithm>

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
  typedef Face DElem;

  friend class ElemData;

  class ElemData
  {
  public:
    ElemData(const HexVolMesh<Basis>& msh,
             const typename Cell::index_type ind) :
      mesh_(msh),
      index_(ind)
    {
      //Linear and Constant Basis never use edges_
      if (basis_type::polynomial_order() > 1) {
        mesh_.get_edges(edges_, index_);
      }
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
    unsigned edge11_index() const {
      return edges_[11];
    }

    inline
    const Point &node0() const {
      return mesh_.points_[node0_index()];
    }
    inline
    const Point &node1() const {
      return mesh_.points_[node1_index()];
    }
    inline
    const Point &node2() const {
      return mesh_.points_[node2_index()];
    }
    inline
    const Point &node3() const {
      return mesh_.points_[node3_index()];
    }
    inline
    const Point &node4() const {
      return mesh_.points_[node4_index()];
    }
    inline
    const Point &node5() const {
      return mesh_.points_[node5_index()];
    }
    inline
    const Point &node6() const {
      return mesh_.points_[node6_index()];
    }
    inline
    const Point &node7() const {
      return mesh_.points_[node7_index()];
    }

  private:
    const HexVolMesh<Basis>          &mesh_;
    const typename Cell::index_type  index_;
    typename Edge::array_type        edges_;
   };

  HexVolMesh();
  HexVolMesh(const HexVolMesh &copy);
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

  bool get_face(typename Face::index_type &array,
                typename Node::index_type n1, 
                typename Node::index_type n2,
                typename Node::index_type n3, 
                typename Node::index_type n4) const;

  //! Get the parent element(s) of the given index.
  void get_elems(typename Elem::array_type &result,
                 typename Node::index_type idx) const;
  void get_elems(typename Elem::array_type &result,
                 typename Edge::index_type idx) const;
  void get_elems(typename Elem::array_type &result,
                 typename Face::index_type idx) const;


  //! Wrapper to get the derivative elements from this element.
  void get_delems(typename DElem::array_type &result,
                  typename Elem::index_type idx) const
  {
    get_faces(result, idx);
  }


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
    Point p0,p1,p2,p3;
    get_point(p0,ra[0]);
    get_point(p1,ra[1]);
    get_point(p2,ra[2]);
    get_point(p3,ra[3]);
    return ( (Cross(p0-p1,p2-p0)).length()*0.5 + (Cross(p2-p3,p0-p2)).length()*0.5) ;
  }
  double get_size(typename Cell::index_type idx) const;

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

  int get_weights(const Point &p, typename Node::array_type &l, double *w);
  int get_weights(const Point & , typename Edge::array_type & , double * )
  { ASSERTFAIL("HexVolMesh::get_weights for edges isn't supported"); }
  int get_weights(const Point & , typename Face::array_type & , double * )
  { ASSERTFAIL("HexVolMesh::get_weights for faces isn't supported"); }
  int get_weights(const Point &p, typename Cell::array_type &l, double *w);

  void get_point(Point &result, typename Node::index_type index) const
    { result = points_[index]; }
  void set_point(const Point &point, typename Node::index_type index)
    { points_[index] = point; }

  void get_normal(Vector &, typename Node::index_type /*index*/) const
  { ASSERTFAIL("This mesh type does not have node normals."); }

  void get_normal(Vector &result, vector<double> &coords,
                  typename Elem::index_type eidx, unsigned int f)
  {
    ElemData ed(*this, eidx);
    vector<Point> Jv;
    basis_.derivate(coords, ed, Jv);

    // load the matrix with the Jacobian
    DenseMatrix J(3, Jv.size());
    int i = 0;
    vector<Point>::iterator iter = Jv.begin();
    while(iter != Jv.end()) {
      Point &p = *iter++;
      J.put(i, 0, p.x());
      J.put(i, 1, p.y());
      J.put(i, 2, p.z());
      ++i;
    }
    J.invert();
    unsigned fmap[] = {0, 5, 1, 3, 4, 2};
    unsigned face = fmap[f];
    ColumnMatrix localV(3);

    localV.put(0, basis_.unit_face_normals[face][0]);
    localV.put(1, basis_.unit_face_normals[face][1]);
    localV.put(2, basis_.unit_face_normals[face][2]);

    ColumnMatrix m(3);
    Mult(m, J, localV);

    result.x(m.get(0));
    result.y(m.get(1));
    result.z(m.get(2));
    result.normalize();
  }

  void get_random_point(Point &p, typename Elem::index_type i, MusilRNG &r) const;

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
  virtual int topology_geometry() const { return (Mesh::UNSTRUCTURED | Mesh::IRREGULAR); }

  typename Node::index_type add_point(const Point &p);

  virtual bool          synchronize(unsigned int);
  //! only call this if you have modified the geometry, this will delete
  //! extra computed synch data and reset the flags to an unsynchronized state.
  void                  unsynchronize();


  virtual bool has_normals() const { return basis_.polynomial_order() > 1; }

  Basis& get_basis() { return basis_; }

  //! Generate the list of points that make up a sufficiently accurate
  //! piecewise linear approximation of an edge.
  void pwl_approx_edge(vector<vector<double> > &coords,
                       typename Cell::index_type ci,
                       unsigned which_edge,
                       unsigned div_per_unit) const
  {
    basis_.approx_edge(which_edge, div_per_unit, coords);
  }

  //! Generate the list of points that make up a sufficiently accurate
  //! piecewise linear approximation of an face.
  void pwl_approx_face(vector<vector<vector<double> > > &coords,
                       typename Cell::index_type ci,
                       unsigned which_face,
                       unsigned div_per_unit) const
  {
    basis_.approx_face(which_face, div_per_unit, coords);
  }

  bool get_coords(vector<double> &coords,
                  const Point &p,
                  typename Cell::index_type idx)
  {
    synchronize(Mesh::FACES_E | Mesh::EDGES_E);
    ElemData ed(*this, idx);
    return basis_.get_coords(coords, p, ed);
  }

  void interpolate(Point &pt, const vector<double> &coords,
                   typename Cell::index_type idx)
  {
    synchronize(Mesh::FACES_E | Mesh::EDGES_E);
    ElemData ed(*this, idx);
    pt = basis_.interpolate(coords, ed);
  }

  // get the Jacobian matrix
  void derivate(const vector<double> &coords,
                typename Cell::index_type idx,
                vector<Point> &J) const
  {
    ElemData ed(*this, idx);
    basis_.derivate(coords, ed, J);
  }

  static const TypeDescription* node_type_description();
  static const TypeDescription* edge_type_description();
  static const TypeDescription* face_type_description();
  static const TypeDescription* cell_type_description();
  static const TypeDescription* elem_type_description()
  { return cell_type_description(); }

  static Persistent* maker() { return scinew HexVolMesh<Basis>; }

  //! must detach, if altering points!
  vector<Point>& get_points() { return points_; }

private:

  void compute_edges();
  void compute_faces();
  void compute_node_neighbors();
  void compute_grid();
  const Point &point(typename Node::index_type i) { return points_[i]; }


  bool inside8_p(typename Cell::index_type i, const Point &p) const;

  //! all the nodes.
  vector<Point>        points_;
  //! each 8 indecies make up a Hex
  vector<under_type>   cells_;

  //! Face information.
  struct PFace {
    typename Node::index_type         nodes_[4];   //! 4 nodes makes a face.
    typename Cell::index_type         cells_[2];   //! 2 cells may have this face is in common.
 
    PFace() {
      nodes_[0] = MESH_NO_NEIGHBOR;
      nodes_[1] = MESH_NO_NEIGHBOR;
      nodes_[2] = MESH_NO_NEIGHBOR;
      nodes_[3] = MESH_NO_NEIGHBOR;
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
    }

    bool shared() const { return ((cells_[0] != MESH_NO_NEIGHBOR) &&
                                  (cells_[1] != MESH_NO_NEIGHBOR)); }

    //! true if both have the same nodes (order does not matter)
    bool operator==(const PFace &f) const {
      return (((nodes_[0] == f.nodes_[0])&&(nodes_[2] == f.nodes_[2])) &&
              (((nodes_[1]==f.nodes_[1])&&(nodes_[3] == f.nodes_[3]))||
                ((nodes_[1]==f.nodes_[3])&&(nodes_[3] == f.nodes_[1]))));
              }

    //! Compares each node.  When a non equal node is found the <
    //! operator is applied.
    bool operator<(const PFace &f) const {
      if ((nodes_[1] < nodes_[3]) && (f.nodes_[1] < f.nodes_[3]))
      {
        if (nodes_[0] == f.nodes_[0])
          if (nodes_[1] == f.nodes_[1])
            if (nodes_[2] == f.nodes_[2])
              return (nodes_[3] < f.nodes_[3]);
            else
              return (nodes_[2] < f.nodes_[2]);
          else
            return (nodes_[1] < f.nodes_[1]);
        else
          return (nodes_[0] < f.nodes_[0]);
      }
      else if ((nodes_[1] < nodes_[3]) && (f.nodes_[1] >= f.nodes_[3]))
      {
        if (nodes_[0] == f.nodes_[0])
          if (nodes_[1] == f.nodes_[3])
            if (nodes_[2] == f.nodes_[2])
              return (nodes_[3] < f.nodes_[1]);
            else
              return (nodes_[2] < f.nodes_[2]);
          else
            return (nodes_[1] < f.nodes_[3]);
        else
          return (nodes_[0] < f.nodes_[0]);      
      }
      else if ((nodes_[1] >= nodes_[3]) && (f.nodes_[1] < f.nodes_[3]))
      {
        if (nodes_[0] == f.nodes_[0])
          if (nodes_[3] == f.nodes_[1])
            if (nodes_[2] == f.nodes_[2])
              return (nodes_[1] < f.nodes_[3]);
            else
              return (nodes_[2] < f.nodes_[2]);
          else
            return (nodes_[3] < f.nodes_[1]);
        else
          return (nodes_[0] < f.nodes_[0]);      
      }
      else
      {
        if (nodes_[0] == f.nodes_[0])
          if (nodes_[3] == f.nodes_[3])
            if (nodes_[2] == f.nodes_[2])
              return (nodes_[1] < f.nodes_[1]);
            else
              return (nodes_[2] < f.nodes_[2]);
          else
            return (nodes_[3] < f.nodes_[3]);
        else
          return (nodes_[0] < f.nodes_[0]);      
      }
    }
  };


  //! Edge information.
  struct PEdge {
    typename Node::index_type         nodes_[2];   //! 2 nodes makes an edge.
    //! list of all the cells this edge is in.
    vector<typename Cell::index_type> cells_;

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
      if (f.nodes_[1] < f.nodes_[3] )
      {
        return ((f.nodes_[0] << sz_quarter_int << sz_quarter_int <<sz_quarter_int) |
              (up4_mask & (f.nodes_[1] << sz_quarter_int << sz_quarter_int)) |
              (mid4_mask & (f.nodes_[2] << sz_quarter_int)) |
              (low4_mask & f.nodes_[3]));
      }
      else
      {
        return ((f.nodes_[0] << sz_quarter_int << sz_quarter_int <<sz_quarter_int) |
              (up4_mask & (f.nodes_[3] << sz_quarter_int << sz_quarter_int)) |
              (mid4_mask & (f.nodes_[2] << sz_quarter_int)) |
              (low4_mask & f.nodes_[1]));
      }
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
  vector<PFace>            faces_;
  face_ht                  face_table_;
  /*! container for edge storage. Must be computed each time
    nodes or cells change. */
  vector<PEdge>            edges_;
  edge_ht                  edge_table_;



  inline
  void hash_edge(typename Node::index_type n1, typename Node::index_type n2,
                 typename Cell::index_type ci, edge_ht &table) const;

  inline
  void hash_face(typename Node::index_type n1, typename Node::index_type n2,
                 typename Node::index_type n3, typename Node::index_type n4,
                 typename Cell::index_type ci, unsigned int facenumber,
                 face_ht &table) const;

  inline bool order_face_nodes(typename Node::index_type& n1, typename Node::index_type& n2,
                 typename Node::index_type& n3, typename Node::index_type& n4) const;


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
    typename Node::array_type   nodes_;
  };

  vector<vector<typename Node::index_type> > node_neighbors_;

  LockingHandle<SearchGrid>     grid_;
  typename Cell::index_type     locate_cache_;

  Mutex                         synchronize_lock_;
  unsigned int                  synchronized_;
  Basis                         basis_;
};


template <class Basis>
template <class Iter, class Functor>
void
HexVolMesh<Basis>::fill_points(Iter begin, Iter end, Functor fill_ftor)
{
  synchronize_lock_.lock();
  Iter iter = begin;
  points_.resize(end - begin); // resize to the new size
  vector<Point>::iterator piter = points_.begin();
  while (iter != end)
  {
    *piter = fill_ftor(*iter);
    ++piter; ++iter;
  }
  synchronize_lock_.unlock();
}


template <class Basis>
template <class Iter, class Functor>
void
HexVolMesh<Basis>::fill_cells(Iter begin, Iter end, Functor fill_ftor)
{
  synchronize_lock_.lock();
  Iter iter = begin;
  cells_.resize((end - begin) * 8); // resize to the new size
  vector<under_type>::iterator citer = cells_.begin();
  while (iter != end)
  {
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
  synchronize_lock_.unlock();
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
  cells_(0),
  faces_(0),
  face_table_(),
  edges_(0),
  edge_table_(),
  grid_(0),
  locate_cache_(0),
  synchronize_lock_("HexVolMesh synchronize_lock_"),
  synchronized_(NODES_E | CELLS_E)
{
}


template <class Basis>
HexVolMesh<Basis>::HexVolMesh(const HexVolMesh &copy):
  points_(0),
  cells_(0),
  faces_(0),
  face_table_(),
  edges_(0),
  edge_table_(),
  grid_(0),
  locate_cache_(0),
  synchronize_lock_("HexVolMesh synchronize_lock_"),
  synchronized_(NODES_E | CELLS_E)
{
  HexVolMesh &lcopy = (HexVolMesh &)copy;

  lcopy.synchronize_lock_.lock();

  points_ = copy.points_;
  cells_ = copy.cells_;

  face_table_ = copy.face_table_;
  faces_ = copy.faces_;
  synchronized_ |= copy.synchronized_ & FACES_E;

  edge_table_ = copy.edge_table_;
  edges_ = copy.edges_;
  synchronized_ |= copy.synchronized_ & EDGES_E;

  synchronized_ &= ~LOCATE_E;
  if (copy.grid_.get_rep())
  {
    grid_ = scinew SearchGrid(*(copy.grid_.get_rep()));
  }
  synchronized_ |= copy.synchronized_ & LOCATE_E;

  lcopy.synchronize_lock_.unlock();
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
HexVolMesh<Basis>::get_random_point(Point &p,
                                    typename Cell::index_type ei,
                                    MusilRNG &rng) const
{
  const Point &p0 = points_[cells_[ei*8+0]];
  const Point &p1 = points_[cells_[ei*8+1]];
  const Point &p2 = points_[cells_[ei*8+2]];
  const Point &p3 = points_[cells_[ei*8+3]];
  const Point &p4 = points_[cells_[ei*8+4]];
  const Point &p5 = points_[cells_[ei*8+5]];
  const Point &p6 = points_[cells_[ei*8+6]];
  const Point &p7 = points_[cells_[ei*8+7]];

  const double a0 = tetrahedra_volume(p0, p1, p2, p5);
  const double a1 = tetrahedra_volume(p0, p2, p3, p7);
  const double a2 = tetrahedra_volume(p0, p5, p2, p7);
  const double a3 = tetrahedra_volume(p0, p5, p7, p4);
  const double a4 = tetrahedra_volume(p5, p2, p7, p6);

  const double w = rng() * (a0 + a1 + a2 + a3 + a4);
  if (w > (a0 + a1 + a2 + a3))
  {
    uniform_sample_tetrahedra(p, p5, p2, p7, p6, rng);
  }
  else if (w > (a0 + a1 + a2))
  {
    uniform_sample_tetrahedra(p, p0, p5, p7, p4, rng);
  }
  else if (w > (a0 + a1))
  {
    uniform_sample_tetrahedra(p, p0, p5, p2, p7, rng);
  }
  else if (w > a0)
  {
    uniform_sample_tetrahedra(p, p0, p2, p3, p7, rng);
  }
  else
  {
    uniform_sample_tetrahedra(p, p0, p1, p2, p5, rng);
  }
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

  synchronize_lock_.lock();
  if (grid_.get_rep()) { grid_->transform(t); }
  synchronize_lock_.unlock();
}

template <class Basis>
bool
HexVolMesh<Basis>::order_face_nodes(typename Node::index_type& n1,
                                    typename Node::index_type& n2,
                                    typename Node::index_type& n3,
                                    typename Node::index_type& n4) const
{
  // Check for degenerate or misformed face
  // Opposite faces cannot be equal
  if ((n1 == n3)||(n2==n4)) return (false);

  // Face must have three unique identifiers otherwise it was condition
  // n1==n3 || n2==n4 would be met.
  
  if (n1==n2)
  {
    if (n3==n4) return (false); // this is a line not a face
    typename Node::index_type t;
    // shift one position to left
    t = n1; n1 = n2; n2 = n3; n3 = n4; n4 = t; 
    return (true);
  }
  else if (n2 == n3)
  {
    if (n1==n4) return (false); // this is a line not a face
    typename Node::index_type t;
    // shift two positions to left
    t = n1; n1 = n3; n3 = t; t = n2; n2 = n4; n4 = t;
    return (true);
  }
  else if (n3 == n4)
  {
    typename Node::index_type t;
    // shift one position to right
    t = n4; n4 = n3; n3 = n2; n2 = n1; n1 = t;    
    return (true);
  }
  else if (n4 == n1)
  {
    // proper order
    return (true);
  }
  else
  {
    if ((n1 < n2)&&(n1 < n3)&&(n1 < n4))
    {
      // proper order
      return (true);
    }
    else if ((n2 < n3)&&(n2 < n4))
    {
      typename Node::index_type t;
      // shift one position to left
      t = n1; n1 = n2; n2 = n3; n3 = n4; n4 = t; 
      return (true);    
    }
    else if (n3 < n4)
    {
      typename Node::index_type t;
      // shift two positions to left
      t = n1; n1 = n3; n3 = t; t = n2; n2 = n4; n4 = t;
      return (true);    
    }
    else
    {
      typename Node::index_type t;
      // shift one positions to right
      t = n4; n4 = n3; n3 = n2; n2 = n1; n1 = t;    
      return (true);    
    }
  }
}


template <class Basis>
void
HexVolMesh<Basis>::hash_face(typename Node::index_type n1,
                             typename Node::index_type n2,
                             typename Node::index_type n3,
                             typename Node::index_type n4,
                             typename Cell::index_type ci,
                             unsigned int face_number, 
                             face_ht &table) const
{
  // Reorder nodes while maintaining CCW or CW orientation
  // Check for degenerate faces, if faces has degeneracy it
  // will be ignored (e.g. nodes on opposite corners are equal,
  // or more then two nodes are equal)
   
  if (!(order_face_nodes(n1,n2,n3,n4))) return;
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
    } else if (f.cells_[0] == ci) {
      cerr << "This Mesh has problems: Cells #"
           << f.cells_[0] << " and #" << ci
           << " are the same." << std::endl;
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
  face_table_.clear();

  typename Cell::iterator ci, cie;
  begin(ci); end(cie);
  typename Node::array_type arr(8);
  while (ci != cie)
  {
    get_nodes(arr, *ci);
    // 6 faces -- each is entered CCW from outside looking in

    hash_face(arr[0], arr[1], arr[2 ], arr[3], *ci, 0, face_table_);
    hash_face(arr[7], arr[6], arr[5], arr[4], *ci, 1, face_table_);
    hash_face(arr[0], arr[4], arr[5], arr[1], *ci, 2, face_table_);
    hash_face(arr[2], arr[6], arr[7], arr[3], *ci, 3, face_table_);
    hash_face(arr[3], arr[7], arr[4], arr[0], *ci, 4, face_table_);
    hash_face(arr[1], arr[5], arr[6], arr[2], *ci, 5, face_table_);

    ++ci;
  }
  // dump edges into the edges_ container.
  faces_.resize(face_table_.size());
  typename vector<PFace>::iterator f_iter = faces_.begin();
  typename face_ht::iterator ht_iter = face_table_.begin();
  int i = 0;
  while (ht_iter != face_table_.end())
  {
    *f_iter = (*ht_iter).first;
    (*ht_iter).second = i;  
    ++f_iter; ++ht_iter; i++;
  }

  synchronized_ |= FACES_E;
}


template <class Basis>
void
HexVolMesh<Basis>::hash_edge(typename Node::index_type n1,
                             typename Node::index_type n2,
                             typename Cell::index_type ci,
                             edge_ht &table) const
{
  if (n1 == n2) return;
  PEdge e(n1, n2);
  typename edge_ht::iterator iter = table.find(e);
  if (iter == table.end()) {
    e.cells_.push_back(ci); // add this cell
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
}


template <class Basis>
bool
HexVolMesh<Basis>::synchronize(unsigned int tosync)
{
  synchronize_lock_.lock();

  if (tosync & EDGES_E && !(synchronized_ & EDGES_E))
  {
    compute_edges();
  }
  if (tosync & FACES_E && !(synchronized_ & FACES_E))
  {
    compute_faces();
  }
  if (tosync & LOCATE_E && !(synchronized_ & LOCATE_E))
  {
    if (!(synchronized_ & FACES_E)) compute_faces();
    compute_grid();
  }
  if (tosync & NODE_NEIGHBORS_E && !(synchronized_ & NODE_NEIGHBORS_E))
  {
    compute_node_neighbors();
  }

  synchronize_lock_.unlock();
  return true;
}

template <class Basis>
void
HexVolMesh<Basis>::unsynchronize()
{
  synchronize_lock_.lock();

  if (synchronized_ & NODE_NEIGHBORS_E)
  {
    node_neighbors_.clear();
  }
  if (synchronized_&EDGES_E || synchronized_&EDGE_NEIGHBORS_E)
  {
    edge_table_.clear();
  }
  if (synchronized_&FACES_E || synchronized_&FACE_NEIGHBORS_E)
  {
    face_table_.clear();
  }
  if (synchronized_ & LOCATE_E)
  {
    grid_ = 0;
  }
  synchronized_ = NODES_E | CELLS_E;

  synchronize_lock_.unlock();
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
HexVolMesh<Basis>::get_nodes(typename Node::array_type &array,
                             typename Face::index_type idx) const
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
HexVolMesh<Basis>::get_nodes(typename Node::array_type &array,
                             typename Cell::index_type idx) const
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
HexVolMesh<Basis>::get_edges(typename Edge::array_type &array,
                             typename Face::index_type idx) const
{
  ASSERTMSG(synchronized_ & FACES_E,
            "Must call synchronize FACES_E on HexVolMesh first");
  ASSERTMSG(synchronized_ & EDGES_E,
            "Must call synchronize EDGES_E on HexVolMesh first");

  array.clear();
  array.reserve(4);
  const PFace &f = faces_[idx];
  
  if (f.nodes_[0] != f.nodes_[1])
  {
    PEdge e(f.nodes_[0], f.nodes_[1]);  
    array.push_back((*(edge_table_.find(e))).second);
  }
  if (f.nodes_[1] != f.nodes_[2])
  {
    PEdge e(f.nodes_[1], f.nodes_[2]);  
    array.push_back((*(edge_table_.find(e))).second);
  }
  if (f.nodes_[2] != f.nodes_[3])
  {
    PEdge e(f.nodes_[2], f.nodes_[3]);  
    array.push_back((*(edge_table_.find(e))).second);
  }
  if (f.nodes_[3] != f.nodes_[0])
  {
    PEdge e(f.nodes_[3], f.nodes_[0]);  
    array.push_back((*(edge_table_.find(e))).second);
  }
}


template <class Basis>
void
HexVolMesh<Basis>::get_edges(typename Edge::array_type &array,
                             typename Cell::index_type idx) const
{
  array.clear();
  array.reserve(12);
  const int off = idx * 8;
  typename Node::index_type n1,n2;

  ASSERTMSG(synchronized_ & EDGES_E,
            "Must call synchronize EDGES_E on HexVolMesh first");
  
  n1 = cells_[off   ]; n2 = cells_[off + 1];
  if (n1 != n2) { PEdge e(n1,n2); array.push_back((*(edge_table_.find(e))).second); }
  n1 = cells_[off + 1]; n2 = cells_[off + 2];
  if (n1 != n2) { PEdge e(n1,n2); array.push_back((*(edge_table_.find(e))).second); }
  n1 = cells_[off + 2]; n2 = cells_[off + 3];
  if (n1 != n2) { PEdge e(n1,n2); array.push_back((*(edge_table_.find(e))).second); }
  n1 = cells_[off + 3]; n2 = cells_[off   ];
  if (n1 != n2) { PEdge e(n1,n2); array.push_back((*(edge_table_.find(e))).second); }

  n1 = cells_[off + 4]; n2 = cells_[off + 5];
  if (n1 != n2) { PEdge e(n1,n2); array.push_back((*(edge_table_.find(e))).second); }
  n1 = cells_[off + 5]; n2 = cells_[off + 6];
  if (n1 != n2) { PEdge e(n1,n2); array.push_back((*(edge_table_.find(e))).second); }
  n1 = cells_[off + 6]; n2 = cells_[off + 7];
  if (n1 != n2) { PEdge e(n1,n2); array.push_back((*(edge_table_.find(e))).second); }
  n1 = cells_[off + 7]; n2 = cells_[off + 4];
  if (n1 != n2) { PEdge e(n1,n2); array.push_back((*(edge_table_.find(e))).second); }

  n1 = cells_[off    ]; n2 = cells_[off + 4];
  if (n1 != n2) { PEdge e(n1,n2); array.push_back((*(edge_table_.find(e))).second); }
  n1 = cells_[off + 5]; n2 = cells_[off + 1];
  if (n1 != n2) { PEdge e(n1,n2); array.push_back((*(edge_table_.find(e))).second); }
  n1 = cells_[off + 2]; n2 = cells_[off + 6];
  if (n1 != n2) { PEdge e(n1,n2); array.push_back((*(edge_table_.find(e))).second); }
  n1 = cells_[off + 7]; n2 = cells_[off + 3];
  if (n1 != n2) { PEdge e(n1,n2); array.push_back((*(edge_table_.find(e))).second); }
}

template <class Basis>
bool
HexVolMesh<Basis>::get_face(typename Face::index_type &face,
                            typename Node::index_type n1, 
                            typename Node::index_type n2,
                            typename Node::index_type n3, 
                            typename Node::index_type n4) const
{
  ASSERTMSG(synchronized_ & FACES_E,
            "Must call synchronize FACES_E on HexVolMesh first");
  if(!(order_face_nodes(n1,n2,n3,n4))) return (false);
  PFace f(n1, n2, n3, n4);
  typename face_ht::const_iterator fiter = face_table_.find(f);
  if (fiter == face_table_.end()) {
    return false;
  }
  face = (*fiter).second;
  return true;
}

template <class Basis>
void
HexVolMesh<Basis>::get_faces(typename Face::array_type &array,
                             typename Cell::index_type idx) const
{
  array.clear();
  array.reserve(8);

  ASSERTMSG(synchronized_ & FACES_E,
            "Must call synchronize FACES_E on HexVolMesh first");
  
  const int off = idx * 8;
  typename Node::index_type n1,n2,n3,n4;
  
  // Put faces in node ordering from smallest node and then following CW or CCW
  // ordering. Test for degenerate elements. Degenerate faces are only added if they
  // are valid (only two neighboring nodes are equal)
  n1 = cells_[off    ]; n2 = cells_[off + 1]; n3 = cells_[off + 2]; n4 = cells_[off + 3];
  if (order_face_nodes(n1,n2,n3,n4))
  {
    PFace f(n1,n2,n3,n4);
    array.push_back((*(face_table_.find(f))).second);
  }
  n1 = cells_[off + 7]; n2 = cells_[off + 6]; n3 = cells_[off + 5]; n4 = cells_[off + 4];
  if (order_face_nodes(n1,n2,n3,n4))
  {
    PFace f(n1,n2,n3,n4);
    array.push_back((*(face_table_.find(f))).second);
  }
  n1 = cells_[off    ]; n2 = cells_[off + 4]; n3 = cells_[off + 5]; n4 = cells_[off + 1];
  if (order_face_nodes(n1,n2,n3,n4))
  {
    PFace f(n1,n2,n3,n4);
    array.push_back((*(face_table_.find(f))).second);
  }
  n1 = cells_[off + 2]; n2 = cells_[off + 6]; n3 = cells_[off + 7]; n4 = cells_[off + 3];
  if (order_face_nodes(n1,n2,n3,n4))
  {
    PFace f(n1,n2,n3,n4);
    array.push_back((*(face_table_.find(f))).second);
  }
  n1 = cells_[off + 3]; n2 = cells_[off + 7]; n3 = cells_[off + 4]; n4 = cells_[off    ];
  if (order_face_nodes(n1,n2,n3,n4))
  {
    PFace f(n1,n2,n3,n4);
    array.push_back((*(face_table_.find(f))).second);
  }
  n1 = cells_[off + 1]; n2 = cells_[off + 5]; n3 = cells_[off + 6]; n4 = cells_[off + 2];
  if (order_face_nodes(n1,n2,n3,n4))
  { 
    PFace f(n1,n2,n3,n4);
    array.push_back((*(face_table_.find(f))).second);
  }
}


template <class Basis>
bool
HexVolMesh<Basis>::get_neighbor(typename Cell::index_type &neighbor,
                                typename Cell::index_type from,
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
HexVolMesh<Basis>::get_neighbors(typename Cell::array_type &array,
                                 typename Cell::index_type idx) const
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
  if (!(synchronized_ & EDGES_E)) { compute_edges(); }

  node_neighbors_.clear();
  node_neighbors_.resize(points_.size());
  typename Edge::iterator ei, eie;
  begin(ei); end(eie);
  for_each(ei, eie, FillNodeNeighbors(node_neighbors_, *this));

  synchronized_ |= NODE_NEIGHBORS_E;
}


template <class Basis>
void
HexVolMesh<Basis>::get_center(Point &p, typename Node::index_type idx) const
{
  get_point(p, idx);
}


template <class Basis>
void
HexVolMesh<Basis>::get_center(Point &result,
                              typename Edge::index_type idx) const
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


template <class Basis>
double
HexVolMesh<Basis>::get_size(typename Cell::index_type idx) const
{
  const Point &p0 = points_[cells_[idx*8+0]];
  const Point &p1 = points_[cells_[idx*8+1]];
  const Point &p2 = points_[cells_[idx*8+2]];
  const Point &p3 = points_[cells_[idx*8+3]];
  const Point &p4 = points_[cells_[idx*8+4]];
  const Point &p5 = points_[cells_[idx*8+5]];
  const Point &p6 = points_[cells_[idx*8+6]];
  const Point &p7 = points_[cells_[idx*8+7]];

  const double a0 = tetrahedra_volume(p0, p1, p2, p5);
  const double a1 = tetrahedra_volume(p0, p2, p3, p7);
  const double a2 = tetrahedra_volume(p0, p5, p2, p7);
  const double a3 = tetrahedra_volume(p0, p5, p7, p4);
  const double a4 = tetrahedra_volume(p5, p2, p7, p6);
  
  return a0 + a1 + a2 + a3 + a4;
}


template <class Basis>
bool
HexVolMesh<Basis>::locate(typename Node::index_type &loc, const Point &p)
{
  typename Cell::index_type ci;
  if (locate(ci, p)) { // first try the fast way.
    typename Node::array_type nodes;
    get_nodes(nodes, ci);

    double dmin = (p - points_[nodes[0]]).length2();
    loc = nodes[0];
    for (unsigned int i = 1; i < nodes.size(); i++)
    {
      const double d = (p - points_[nodes[i]]).length2();
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

    double min_dist = (p - points_[*ni]).length2();
    loc = *ni;
    ++ni;

    while (ni != nie)
    {
      const double dist = (p - points_[*ni]).length2();
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
    double mindist = (p -loc).length2();
    for (unsigned int i = 0; i < edges.size(); i++)
    {
      get_center(loc, edges[i]);
      const double dist = (p -loc).length2();
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
    double mindist = (p - loc).length2();
    for (unsigned int i = 0; i < faces.size(); i++)
    {
      get_center(loc, faces[i]);
      const double dist = (p - loc).length2();
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
  if (basis_.polynomial_order() > 1) return elem_locate(cell, *this, p);
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


template <class Basis>
int
HexVolMesh<Basis>::get_weights(const Point &p, typename Node::array_type &l,
                               double *w)
{
  typename Cell::index_type idx;
  if (locate(idx, p))
  {
    get_nodes(l,idx);
    vector<double> coords(3);
    if (get_coords(coords, p, idx))
    {
      basis_.get_weights(coords, w);
      return basis_.dofs();
    }
  }
  return 0;
}


template <class Basis>
int
HexVolMesh<Basis>::get_weights(const Point &p, typename Cell::array_type &l,
                              double *w)
{
  typename Cell::index_type idx;
  if (locate(idx, p))
  {
    l.resize(1);
    l[0] = idx;
    w[0] = 1.0;
    return 1;
  }
  return 0;
}


template <class Basis>
void
HexVolMesh<Basis>::compute_grid()
{
  if (grid_.get_rep() != 0) {return; } // only create once.

  BBox bb = get_bounding_box();
  if (!bb.valid()) {return; }

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
}


template <class Basis>
bool
HexVolMesh<Basis>::inside8_p(typename Cell::index_type i,
                             const Point &p) const
{
  typename Face::array_type faces;
  get_faces(faces, i);

  Point center;
  get_center(center, i);

  for (unsigned int i = 0; i < faces.size(); i++) {
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

    double dotprod = Dot(off0, normal);

    // Account for round off - the point may be on the plane!!
    if( fabs( dotprod ) < MIN_ELEMENT_VAL )
      continue;

    // If orientated correctly the second dot product is not needed.
    // Only need to check to see if the sign is negative.
    if (dotprod * Dot(off1, normal) < 0.0)
      return false;
  }

  return true;
}


template <class Basis>
typename HexVolMesh<Basis>::Node::index_type
HexVolMesh<Basis>::add_find_point(const Point &p, double err)
{
  typename Node::index_type i;
  if (locate(i, p) && ((points_[i] - p).length2() < err))
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
HexVolMesh<Basis>::add_hex(typename Node::index_type a,
                           typename Node::index_type b,
                           typename Node::index_type c,
                           typename Node::index_type d,
                           typename Node::index_type e,
                           typename Node::index_type f,
                           typename Node::index_type g,
                           typename Node::index_type h)
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
  ASSERTMSG(a.size() == 8, "Tried to add non-hex element.");
  cells_.push_back(a[0]);
  cells_.push_back(a[1]);
  cells_.push_back(a[2]);
  cells_.push_back(a[3]);
  cells_.push_back(a[4]);
  cells_.push_back(a[5]);
  cells_.push_back(a[6]);
  cells_.push_back(a[7]);
  return static_cast<typename Elem::index_type>((cells_.size() >> 3)-1);
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
    td = scinew TypeDescription("HexVolMesh", subs,
                                string(__FILE__),
                                "SCIRun",
                                TypeDescription::MESH_E);
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
                                "SCIRun",
                                TypeDescription::MESH_E);
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
                                "SCIRun",
                                TypeDescription::MESH_E);
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
                                "SCIRun",
                                TypeDescription::MESH_E);
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
                                "SCIRun",
                                TypeDescription::MESH_E);
  }
  return td;
}


template <class Basis>
void
HexVolMesh<Basis>::get_elems(typename Cell::array_type &array,
                             typename Edge::index_type idx) const
{
  ASSERTMSG(synchronized_ & EDGES_E,
            "Must call synchronize EDGES_E on HexVolMesh first");
  array = edges_[idx].cells_;
}

template <class Basis>
void
HexVolMesh<Basis>::get_elems(typename Cell::array_type &array,
                             typename Face::index_type idx) const
{
  ASSERTMSG(synchronized_ & FACES_E,
            "Must call synchronize FACES_E on HexVolMesh first");
  if (faces_[idx].cells_[1] == MESH_NO_NEIGHBOR)
  {
    array.resize(1);
    array[0] = faces_[idx].cells_[0];
  }
  else
  {
    array.resize(2);
    array[0] = faces_[idx].cells_[0];
    array[1] = faces_[idx].cells_[1];
  }
}


template <class Basis>
void
HexVolMesh<Basis>::get_elems(typename Elem::array_type &cells,
                             typename Node::index_type node) const
{
  ASSERTMSG(synchronized_ & NODE_NEIGHBORS_E,
            "Must call synchronize NODE_NEIGHBORS_E on HexVolMesh first");
  ASSERTMSG(synchronized_ & EDGES_E,
            "Must call synchronize EDGES_E on HexVolMesh first");

  vector<typename Node::index_type> neighbors;
  set<typename Cell::index_type> unique_cells;
  // Get all the nodes that share an edge with this node
  get_neighbors(neighbors, node);
  // Iterate through all those edges
  for (unsigned int n = 0; n < neighbors.size(); n++)
  {
    // Get the edge information for the current edge
    typename edge_ht::const_iterator iter =
                                edge_table_.find(PEdge(node,neighbors[n]));
    ASSERTMSG(iter != edge_table_.end(),
              "Edge not found in HexVolMesh::edge_table_");
    // Insert all cells that share this edge into
    // the unique set of cell indices
    for (unsigned int c = 0; c < (iter->first).cells_.size(); c++)
      unique_cells.insert((iter->first).cells_[c]);
  }

  // Copy the unique set of cells to our Cells array return argument
  cells.resize(unique_cells.size());
  copy(unique_cells.begin(), unique_cells.end(), cells.begin());
}


} // namespace SCIRun


#endif // SCI_project_HexVolMesh_h
