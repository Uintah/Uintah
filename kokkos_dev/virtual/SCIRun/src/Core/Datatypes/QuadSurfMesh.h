/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
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

#ifndef CORE_DATATYPES_QUADSURFMESH_H
#define CORE_DATATYPES_QUADSURFMESH_H 1

#include <Core/Containers/LockingHandle.h>
#include <Core/Datatypes/Mesh.h>
#include <Core/Datatypes/SearchGrid.h>
#include <Core/Datatypes/FieldIterator.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Transform.h>
#include <Core/Geometry/CompGeom.h>
#include <Core/Containers/StackVector.h>
#include <Core/Geometry/BBox.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {

using std::vector;

template <class Basis>
class QuadSurfMesh : public Mesh
{
public:
  typedef LockingHandle<QuadSurfMesh<Basis> > handle_type;
  typedef Basis                         basis_type;
  typedef unsigned int                  under_type;

  //! Index and Iterator types required for Mesh Concept.
  struct Node {
    typedef NodeIndex<under_type>       index_type;
    typedef NodeIterator<under_type>    iterator;
    typedef NodeIndex<under_type>       size_type;
    typedef StackVector<index_type, 4>  array_type;
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

  typedef Face Elem;
  typedef Edge DElem;

  friend class ElemData;

  class ElemData
  {
  public:
    ElemData(const QuadSurfMesh<Basis>& msh,
             const typename Elem::index_type ind) :
      mesh_(msh),
      index_(ind)
    {}

    // the following designed to coordinate with ::get_nodes
    inline
    unsigned node0_index() const {
      return mesh_.faces_[index_ * 4];
    }
    inline
    unsigned node1_index() const {
      return mesh_.faces_[index_ * 4 + 1];
    }
    inline
    unsigned node2_index() const {
      return mesh_.faces_[index_ * 4 + 2];
    }
    inline
    unsigned node3_index() const {
      return mesh_.faces_[index_ * 4 + 3];
    }

    // the following designed to coordinate with ::get_edges
    inline
    unsigned edge0_index() const {
      return index_ * 6;
    }
    inline
    unsigned edge1_index() const {
      return index_ * 6 + 1;
    }
    inline
    unsigned edge2_index() const {
      return index_ * 6 + 2;
    }
    inline
    unsigned edge3_index() const {
      return index_ * 6 + 3;
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

  private:
    const QuadSurfMesh<Basis>          &mesh_;
    const typename Elem::index_type  index_;
   };

  QuadSurfMesh();
  QuadSurfMesh(const QuadSurfMesh &copy);
  virtual QuadSurfMesh *clone() { return new QuadSurfMesh(*this); }
  virtual ~QuadSurfMesh();

  virtual int basis_order() { return (basis_.polynomial_order()); }

  virtual BBox get_bounding_box() const;
  virtual void transform(const Transform &t);

  bool get_dim(vector<unsigned int>&) const { return false;  }

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

  //! Get the child elements of the given index.
  void get_nodes(typename Node::array_type &array, 
                 typename Node::index_type idx) const
  { array.resize(1); array[0]= idx; }
  void get_nodes(typename Node::array_type &array, 
                 typename Edge::index_type idx) const
  { get_nodes_from_edge(array,idx); }
  void get_nodes(typename Node::array_type &array, 
                 typename Face::index_type idx) const
  { get_nodes_from_face(array,idx); }
  void get_nodes(typename Node::array_type &array, 
                 typename Cell::index_type idx) const
  { ASSERTFAIL("QuadSurfMesh: get_nodes has not been implemented for cells"); }

  void get_edges(typename Edge::array_type &array, 
                 typename Node::index_type idx) const
  { get_edges_from_node(array,idx); }
  void get_edges(typename Edge::array_type &array, 
                 typename Edge::index_type idx) const
  { array.resize(1); array[0]= idx; }
  void get_edges(typename Edge::array_type &array, 
                 typename Face::index_type idx) const
  { get_edges_from_face(array,idx); }
  void get_edges(typename Edge::array_type &array, 
                 typename Cell::index_type idx) const
  { ASSERTFAIL("QuadSurfMesh: get_edges has not been implemented for cells"); }

  void get_faces(typename Face::array_type &array, 
                 typename Node::index_type idx) const
  { get_faces_from_node(array,idx); }
  void get_faces(typename Face::array_type &array, 
                 typename Edge::index_type idx) const
  { get_faces_from_edge(array,idx); }
  void get_faces(typename Face::array_type &array, 
                 typename Face::index_type idx) const
  { array.resize(1); array[0]= idx; }
  void get_faces(typename Face::array_type &array, 
                 typename Cell::index_type idx) const
  { ASSERTFAIL("QuadSurfMesh: get_faces has not been implemented for cells"); }

  void get_cells(typename Cell::array_type &array, 
                 typename Node::index_type idx) const
  { ASSERTFAIL("QuadSurfMesh: get_cells has not been implemented"); }
  void get_cells(typename Cell::array_type &array, 
                 typename Edge::index_type idx) const
  { ASSERTFAIL("QuadSurfMesh: get_cells has not been implemented"); }
  void get_cells(typename Cell::array_type &array, 
                 typename Face::index_type idx) const
  { ASSERTFAIL("QuadSurfMesh: get_cells has not been implemented"); }
  void get_cells(typename Cell::array_type &array, 
                 typename Cell::index_type idx) const
  { ASSERTFAIL("QuadSurfMesh: get_cells has not been implemented"); }

  void get_elems(typename Elem::array_type &array, 
                 typename Node::index_type idx) const
  { get_faces_from_node(array,idx); }
  void get_elems(typename Elem::array_type &array, 
                 typename Edge::index_type idx) const
  { get_faces_from_edge(array,idx); }
  void get_elems(typename Elem::array_type &array, 
                 typename Face::index_type idx) const
  { array.resize(1); array[0]= idx; }
  void get_elems(typename Face::array_type &array, 
                 typename Cell::index_type idx) const
  { ASSERTFAIL("QuadSurfMesh: get_elems has not been implemented for cells"); }

  void get_delems(typename DElem::array_type &array, 
                  typename Node::index_type idx) const
  { get_edges_from_node(array,idx); }
  void get_delems(typename DElem::array_type &array, 
                  typename Edge::index_type idx) const
  { array.resize(1); array[0]= idx; }
  void get_delems(typename DElem::array_type &array, 
                  typename Face::index_type idx) const
  { get_edges_from_face(array,idx); }
  void get_delems(typename DElem::array_type &array, 
                  typename Cell::index_type idx) const
  { ASSERTFAIL("QuadSurfMesh: get_delems has not been implemented for cells"); }

  bool get_neighbor(typename Face::index_type &neighbor,
                    typename Face::index_type from,
                    typename Edge::index_type idx) const;

  void get_neighbors(typename Face::array_type &array, typename Face::index_type idx) const;
  void get_neighbors(vector<typename Node::index_type> &array, typename Node::index_type idx) const
  { ASSERTFAIL("Get neighbors has not been implemented for QuadSurfMesh"); }

  //! Get the size of an elemnt (length, area, volume)
  double get_size(typename Node::index_type /*idx*/) const { return 0; }
  double get_size(typename Edge::index_type idx) const
  {
    typename Node::array_type arr;
    get_nodes(arr, idx);
    return (point(arr[0]).asVector() - point(arr[1]).asVector()).length();
  }
  double get_size(typename Face::index_type idx) const
  {
    typename Node::array_type ra;
    get_nodes(ra,idx);
    const Point &p0 = point(ra[0]);
    const Point &p1 = point(ra[1]);
    const Point &p2 = point(ra[2]);
    const Point &p3 = point(ra[3]);
    return ((Cross(p0-p1,p2-p1)).length()+(Cross(p2-p3,p0-p3)).length()+
            (Cross(p3-p0,p1-p0)).length()+(Cross(p1-p2,p3-p2)).length())*0.25;
  }
  double get_size(typename Cell::index_type /*idx*/) const { return 0; };
  double get_length(typename Edge::index_type idx) const { return get_size(idx); };
  double get_area(typename Face::index_type idx) const { return get_size(idx); };
  double get_volume(typename Cell::index_type idx) const { return get_size(idx); };

  void get_center(Point &p, typename Node::index_type i) const { get_point(p, i); }
  void get_center(Point &p, typename Edge::index_type i) const;
  void get_center(Point &p, typename Face::index_type i) const;
  void get_center(Point &, typename Cell::index_type) const {}

  bool locate(typename Node::index_type &loc, const Point &p) const;
  bool locate(typename Edge::index_type &loc, const Point &p) const;
  bool locate(typename Face::index_type &loc, const Point &p) const;
  bool locate(typename Cell::index_type &loc, const Point &p) const;

  int get_weights(const Point &p, typename Node::array_type &l, double *w);
  int get_weights(const Point & , typename Edge::array_type & , double * )
  {ASSERTFAIL("QuadSurfMesh::get_weights(Edges) not supported."); }
  int get_weights(const Point &p, typename Face::array_type &l, double *w);
  int get_weights(const Point & , typename Cell::array_type & , double * )
  {ASSERTFAIL("QuadSurfMesh::get_weights(Cells) not supported."); }

  bool inside4_p(typename Face::index_type i, const Point &p) const;

  void get_point(Point &p, typename Node::index_type i) const { p = points_[i]; }
  void get_normal(Vector &n, typename Node::index_type i) const { n = normals_[i]; }

  void get_normal(Vector &result, vector<double> &coords,
                  typename Elem::index_type eidx, unsigned int)
  {

    if (basis_.polynomial_order() < 2) {
      typename Node::array_type arr(3);
      get_nodes(arr, eidx);

      const double c0_0 = fabs(coords[0]);
      const double c1_0 = fabs(coords[1]);
      const double c0_1 = fabs(coords[0] - 1.0L);
      const double c1_1 = fabs(coords[1] - 1.0L);

      if (c0_0 < 1e-7 && c1_0 < 1e-7) {
        // arr[0]
        result = normals_[arr[0]];
        return;
      } else if (c0_1 < 1e-7 && c1_0 < 1e-7) {
        // arr[1]
        result = normals_[arr[1]];
        return;
      } else if (c0_1 < 1e-7 && c1_1 < 1e-7) {
        // arr[2]
        result = normals_[arr[2]];
        return;
      } else if (c0_0 < 1e-7 && c1_1 < 1e-7) {
        // arr[3]
        result = normals_[arr[3]];
        return;
      }
    }

    ElemData ed(*this, eidx);
    vector<Point> Jv;
    basis_.derivate(coords, ed, Jv);
    result = Cross(Jv[0].asVector(), Jv[1].asVector());
    result.normalize();
  }

  void set_point(const Point &p, typename Node::index_type i)
  { points_[i] = p; }

  void get_random_point(Point &, typename Elem::index_type, MusilRNG &rng) const;

  int get_valence(typename Node::index_type /*idx*/) const { return 0; }
  int get_valence(typename Edge::index_type /*idx*/) const { return 0; }
  int get_valence(typename Face::index_type /*idx*/) const { return 0; }
  int get_valence(typename Cell::index_type /*idx*/) const { return 0; }

  virtual bool has_normals() const { return true; }

  virtual void io(Piostream&);
  static PersistentTypeID type_idqs;
  static MeshTypeID mesh_idqs;
  
  static  const string type_name(int n = -1);
  virtual const TypeDescription *get_type_description() const;

  // Extra functionality needed by this specific geometry.
  typename Node::index_type add_find_point(const Point &p,
                                           double err = 1.0e-3);
  typename Elem::index_type add_quad(typename Node::index_type a,
                            typename Node::index_type b,
                            typename Node::index_type c,
                            typename Node::index_type d);
  typename Elem::index_type add_quad(const Point &p0, const Point &p1,
                const Point &p2, const Point &p3);
  typename Elem::index_type add_elem(typename Node::array_type a);
  virtual void node_reserve(size_t s) { points_.reserve(s); }
  virtual void elem_reserve(size_t s) { faces_.reserve(s*4); }
  virtual bool is_editable() const { return true; }
  virtual int dimensionality() const { return 2; }
  virtual int topology_geometry() const { return (UNSTRUCTURED | IRREGULAR); }
  typename Node::index_type add_point(const Point &p);
  typename Node::index_type add_node(const Point &p);

  virtual bool          synchronize(unsigned int);

  Basis& get_basis() { return basis_; }

  //! Generate the list of points that make up a sufficiently accurate
  //! piecewise linear approximation of an edge.
  void pwl_approx_edge(vector<vector<double> > &coords,
                       typename Elem::index_type ci,
                       unsigned which_edge,
                       unsigned div_per_unit) const
  {
    // Needs to match unit_edges in Basis/QuadBilinearLgn.cc
    // compare get_nodes order to the basis order

    //FIX_ME MC delete this comment when this is verified.
    basis_.approx_edge(which_edge, div_per_unit, coords);
  }

  //! Generate the list of points that make up a sufficiently accurate
  //! piecewise linear approximation of an face.
  void pwl_approx_face(vector<vector<vector<double> > > &coords,
                       typename Elem::index_type ci,
                       unsigned,
                       unsigned div_per_unit) const
  {
    // Needs to match unit_faces in Basis/QuadBilinearLgn.cc
    // compare get_nodes order to the basis order
    basis_.approx_face(0, div_per_unit, coords);
  }

  bool get_coords(vector<double> &coords,
                  const Point &p,
                  typename Elem::index_type idx) const
  {
    ElemData ed(*this, idx);
    return basis_.get_coords(coords, p, ed);
  }

  void interpolate(Point &pt, const vector<double> &coords,
                   typename Elem::index_type idx) const
  {
    ElemData ed(*this, idx);
    pt = basis_.interpolate(coords, ed);
  }

  // get the Jacobian matrix
  template<class VECTOR>
  void derivate(const vector<double> &coords,
                typename Elem::index_type idx,
                VECTOR &J) const
  {
    ElemData ed(*this, idx);
    basis_.derivate(coords, ed, J);
  }

  static const TypeDescription* node_type_description();
  static const TypeDescription* edge_type_description();
  static const TypeDescription* face_type_description();
  static const TypeDescription* cell_type_description();
  static const TypeDescription* elem_type_description()
  { return face_type_description(); }

  // returns a QuadSurfMesh
  static Persistent *maker() { return scinew QuadSurfMesh<Basis>(); }
  static MeshHandle mesh_maker() { return scinew QuadSurfMesh<Basis>(); }
 
  double find_closest_elem(Point &result, typename Elem::index_type &elem,
                           const Point &p) const
  {
    ASSERTFAIL("Search grid has not yet implemented for this mesh");
  }

  double find_closest_elems(Point &result,
                            vector<typename Elem::index_type> &elem,
                            const Point &p) const
  {
    ASSERTFAIL("Search grid has not yet implemented for this mesh");
  }	
	  

public:
  //! VIRTUAL INTERFACE FUNCTIONS
  virtual bool has_virtual_interface() const;
  
  virtual void size(Mesh::VNode::size_type& size) const;
  virtual void size(Mesh::VEdge::size_type& size) const;
  virtual void size(Mesh::VFace::size_type& size) const;
  virtual void size(Mesh::VElem::size_type& size) const;
  virtual void size(Mesh::VDElem::size_type& size) const;
  
  virtual void get_nodes(VNode::array_type& nodes, VEdge::index_type i) const;
  virtual void get_nodes(VNode::array_type& nodes, VFace::index_type i) const;
  virtual void get_nodes(VNode::array_type& nodes, VElem::index_type i) const;
  virtual void get_nodes(VNode::array_type& nodes, VDElem::index_type i) const;
  
  virtual void get_edges(VEdge::array_type& edges, VFace::index_type i) const;
  virtual void get_edges(VEdge::array_type& edges, VElem::index_type i) const;
  virtual void get_edges(VEdge::array_type& edges, VDElem::index_type i) const;

  virtual void get_faces(VFace::array_type& faces, VNode::index_type i) const;
  virtual void get_faces(VFace::array_type& faces, VEdge::index_type i) const;
  virtual void get_faces(VFace::array_type& faces, VElem::index_type i) const;
  virtual void get_faces(VFace::array_type& faces, VDElem::index_type i) const;
  
  virtual void get_elems(VElem::array_type& elems, VNode::index_type i) const;
  virtual void get_elems(VElem::array_type& elems, VEdge::index_type i) const;
  virtual void get_elems(VElem::array_type& elems, VFace::index_type i) const;
  virtual void get_elems(VElem::array_type& elems, VDElem::index_type i) const;

  virtual void get_delems(VDElem::array_type& delems, VFace::index_type i) const;
  virtual void get_delems(VDElem::array_type& delems, VElem::index_type i) const;

  virtual void get_center(Point &point, VNode::index_type i) const;
  virtual void get_center(Point &point, VEdge::index_type i) const;
  virtual void get_center(Point &point, VFace::index_type i) const;
  virtual void get_center(Point &point, VElem::index_type i) const;
  virtual void get_center(Point &point, VDElem::index_type i) const;

  virtual double get_size(VNode::index_type i) const;
  virtual double get_size(VEdge::index_type i) const;
  virtual double get_size(VFace::index_type i) const;
  virtual double get_size(VElem::index_type i) const;
  virtual double get_size(VDElem::index_type i) const;
  
  virtual void get_weights(const Point& p,VNode::array_type& nodes,
                                                vector<double>& weights) const;
  virtual void get_weights(const Point& p,VElem::array_type& elems,
                                                vector<double>& weights) const;
                                                  
  virtual bool locate(VNode::index_type &i, const Point &point) const;
  virtual bool locate(VElem::index_type &i, const Point &point) const;

  virtual bool get_coords(vector<double> &coords, const Point &point, 
                                                    VElem::index_type i) const;  
  virtual void interpolate(Point &p, const vector<double> &coords, 
                                                    VElem::index_type i) const;
  virtual void derivate(vector<Point> &p, const vector<double> &coords, 
                                                    VElem::index_type i) const;

  virtual void get_random_point(Point &p, VElem::index_type i,MusilRNG &rng) const;
  virtual void set_point(const Point &point, VNode::index_type i);
  
  virtual void get_points(vector<Point>& points) const;

  virtual void add_node(const Point &point,VNode::index_type &i);
  virtual void add_elem(const VNode::array_type &nodes,VElem::index_type &i);

  virtual bool get_neighbor(VElem::index_type &neighbor, 
                       VElem::index_type from, VDElem::index_type delem) const;
  virtual void get_neighbors(VElem::array_type &elems, 
                                                    VElem::index_type i) const;
  virtual void get_neighbors(VNode::array_type &nodes, 
                                                    VNode::index_type i) const;

  virtual void pwl_approx_edge(vector<vector<double> > &coords, 
                               VElem::index_type ci, unsigned int which_edge, 
                               unsigned int div_per_unit) const;
  virtual void pwl_approx_face(vector<vector<vector<double> > > &coords, 
                               VElem::index_type ci, unsigned int which_face, 
                              unsigned int div_per_unit) const;
                              
  virtual void get_normal(Vector& norm,VNode::index_type i) const;  
  


private:

  //////////////////////////////////////////////////////////////
  // These functions are templates and are used to define the
  // dynamic compilation interface and the virtual interface
  // as they both use different datatypes as indices and arrays
  // the following functions have been templated and are inlined
  // at the places where they are needed.
  //
  // Secondly these templates allow for the use of the stack vector
  // as well as the STL vector. When an algorithm supports non linear
  // functions an STL vector is a better choice, in the other cases
  // often a StackVector is enough (The latter improves performance).
   
  template<class ARRAY, class INDEX>
  inline void get_nodes_from_edge(ARRAY& array, INDEX i) const
  {
    ASSERTMSG(synchronized_ & EDGES_E,
              "QuadSurfMesh: Must call synchronize EDGES_E on QuadSurfMesh first");
    static int table[8][2] = { {0, 1}, {1, 2}, {2, 3}, {3, 0} };

    const int idx = edges_[i];
    const int off = idx % 4;
    const int node = idx - off;
    array.resize(2);
    array[0] = static_cast<typename ARRAY::value_type>(faces_[node + table[off][0]]);
    array[1] = static_cast<typename ARRAY::value_type>(faces_[node + table[off][1]]);
  }
  
  
  template<class ARRAY, class INDEX>
  inline void get_nodes_from_face(ARRAY& array, INDEX idx) const
  {  
    ASSERTMSG(synchronized_ & FACES_E,
            "QuadSurfMesh: Must call synchronize FACES_E on QuadSurfMesh first");
    array.resize(4);
    array[0] = static_cast<typename ARRAY::value_type>(faces_[idx * 4 + 0]);
    array[1] = static_cast<typename ARRAY::value_type>(faces_[idx * 4 + 1]);
    array[2] = static_cast<typename ARRAY::value_type>(faces_[idx * 4 + 2]);
    array[3] = static_cast<typename ARRAY::value_type>(faces_[idx * 4 + 3]);
    order_face_nodes(array[0],array[1],array[2],array[3]);
  }
  
  
  template<class ARRAY, class INDEX>
  inline void get_edges_from_face(ARRAY& array, INDEX idx) const
  {
    ASSERTMSG(synchronized_ & EDGES_E,
            "QuadSurfMesh: Must call synchronize EDGES_E on QuadSurfMesh first");

    array.clear();

    unsigned int i;
    for (i=0; i < 4; i++)
    {
      const int a = idx * 4 + i;
      const int b = a - a % 4 + (a+1) % 4;
      int j = (int)edges_.size()-1;
      for (; j >= 0; j--)
      {
        const int c = edges_[j];
        const int d = c - c % 4 + (c+1) % 4;
        if (faces_[a] == faces_[c] && faces_[b] == faces_[d] ||
            faces_[a] == faces_[d] && faces_[b] == faces_[c])
        {
          array.push_back(static_cast<typename ARRAY::value_type>(j));
          break;
        }
      }
    }
  }
  
  
  template<class ARRAY, class INDEX>
  inline void get_faces_from_node(ARRAY& array, INDEX idx) const
  {
    ASSERTMSG(synchronized_ & NODE_NEIGHBORS_E,
	      "QuadSurfMesh: Must call synchronize NODE_NEIGHBORS_E on QuadSurfMesh first");
    array.resize(node_neighbors_[idx].size());    
    for (unsigned int i = 0; i < node_neighbors_[idx].size(); ++i)
      array[i] = static_cast<typename ARRAY::value_type>(node_neighbors_[idx][i]/4);
  }
  
 
  template<class ARRAY, class INDEX>
  inline void get_faces_from_edge(ARRAY& array, INDEX idx) const
  {
    ASSERTFAIL("QuadSurfMesh: get_faces(faces,edge) has not been implemented");
  }


  template<class ARRAY, class INDEX>
  inline void get_edges_from_node(ARRAY& array, INDEX idx) const
  {
    ASSERTFAIL("QuadSurfMesh: get_edges(edges,node) has not been implemented");
  }





  const Point &point(typename Node::index_type i) const { return points_[i]; }

  // These require the synchronize_lock_ to be held before calling.
  void                  compute_edges();
  void                  compute_normals();
  void                  compute_node_neighbors();
  void                  compute_edge_neighbors();
  void                  compute_grid();

  template <class NODE>
  bool order_face_nodes(NODE& n1,NODE& n2, NODE& n3, NODE& n4) const
  {
    // Check for degenerate or misformed face
    // Opposite faces cannot be equal
    if ((n1 == n3)||(n2==n4)) return (false);

    // Face must have three unique identifiers otherwise it was condition
    // n1==n3 || n2==n4 would be met.
    
    if (n1==n2)
    {
      if (n3==n4) return (false); // this is a line not a face
      NODE t;
      // shift one position to left
      t = n1; n1 = n2; n2 = n3; n3 = n4; n4 = t; 
      return (true);
    }
    else if (n2 == n3)
    {
      if (n1==n4) return (false); // this is a line not a face
      NODE t;
      // shift two positions to left
      t = n1; n1 = n3; n3 = t; t = n2; n2 = n4; n4 = t;
      return (true);
    }
    else if (n3 == n4)
    {
      NODE t;
      // shift one positions to right
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
        NODE t;
        // shift one position to left
        t = n1; n1 = n2; n2 = n3; n3 = n4; n4 = t; 
        return (true);    
      }
      else if (n3 < n4)
      {
        NODE t;
        // shift two positions to left
        t = n1; n1 = n3; n3 = t; t = n2; n2 = n4; n4 = t;
        return (true);    
      }
      else
      {
        NODE t;
        // shift one positions to right
        t = n4; n4 = n3; n3 = n2; n2 = n1; n1 = t;    
        return (true);    
      }
    }
  }


  int next(int i) { return ((i%4)==3) ? (i-3) : (i+1); }
  int prev(int i) { return ((i%4)==0) ? (i+3) : (i-1); }

  vector<Point>                         points_;
  vector<typename Node::index_type>     faces_;
  vector<under_type>                    edges_;
  vector<under_type>                 halfedge_to_edge_;  // halfedge->edge map
  typedef vector<vector<typename Elem::index_type> > NodeNeighborMap;
  NodeNeighborMap                       node_neighbors_;
  vector<under_type>                    edge_neighbors_;
  vector<Vector>                        normals_; //! normalized per node
  LockingHandle<SearchGrid>             grid_;

  Mutex                                 synchronize_lock_;
  unsigned int                          synchronized_;
  Basis                                 basis_;


#ifdef HAVE_HASH_MAP
  struct edgehash
  {
    size_t operator()(const pair<int, int> &a) const
    {
#if defined(__ECC) || defined(_MSC_VER)
      hash_compare<int> hasher;
#else
      hash<int> hasher;
#endif
      return hasher(hasher(a.first) + a.second);
    }
#if defined(__ECC) || defined(_MSC_VER)

    static const size_t bucket_size = 4;
    static const size_t min_buckets = 8;

    bool operator()(const pair<int, int> &a, const pair<int, int> &b) const
    {
      return a.first < b.first || a.first == b.first && a.second < b.second;
    }
#endif
  };

  struct edgecompare
  {
    bool operator()(const pair<int, int> &a, const pair<int, int> &b) const
    {
      return a.first == b.first && a.second == b.second;
    }
  };

#else // HAVE_HASH_MAP

  struct edgecompare
  {
    bool operator()(const pair<int, int> &a, const pair<int, int> &b) const
    {
      return a.first < b.first || a.first == b.first && a.second < b.second;
    }
  };
#endif // HAVE_HASH_MAP
};


template <class Basis>
PersistentTypeID
QuadSurfMesh<Basis>::type_idqs(type_name(-1), "Mesh",
                               QuadSurfMesh<Basis>::maker);

template <class Basis>
MeshTypeID
QuadSurfMesh<Basis>::mesh_idqs(type_name(-1), QuadSurfMesh<Basis>::mesh_maker);



template <class Basis>
const string
QuadSurfMesh<Basis>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;
  }
  else if (n == 0)
  {
    static const string nm("QuadSurfMesh");
    return nm;
  }
  else
  {
    return find_type_name((Basis *)0);
  }
}


template <class Basis>
QuadSurfMesh<Basis>::QuadSurfMesh()
  : points_(0),
    faces_(0),
    edges_(0),
    edge_neighbors_(0),
    normals_(0),
    grid_(0),
    synchronize_lock_("QuadSurfMesh synchronize_lock_"),
    synchronized_(NODES_E | FACES_E | CELLS_E)
{
}


template <class Basis>
QuadSurfMesh<Basis>::QuadSurfMesh(const QuadSurfMesh &copy)
  : points_(0),
    faces_(0),
    edges_(0),
    edge_neighbors_(0),
    normals_(0),
    grid_(0),
    synchronize_lock_("QuadSurfMesh synchronize_lock_"),
    synchronized_(NODES_E | FACES_E | CELLS_E)
{
  QuadSurfMesh &lcopy = (QuadSurfMesh &)copy;
  
  lcopy.synchronize_lock_.lock();

  points_ = copy.points_;

  edges_ = copy.edges_;
  halfedge_to_edge_ = copy.halfedge_to_edge_;
  synchronized_ |= copy.synchronized_ & EDGES_E;

  faces_ = copy.faces_;

  node_neighbors_ = copy.node_neighbors_;
  synchronized_ |= copy.synchronized_ & NODE_NEIGHBORS_E;

  edge_neighbors_ = copy.edge_neighbors_;
  synchronized_ |= copy.synchronized_ & EDGE_NEIGHBORS_E;

  normals_ = copy.normals_;
  synchronized_ |= copy.synchronized_ & NORMALS_E;

  synchronized_ &= ~LOCATE_E;
  if (copy.grid_.get_rep())
  {
    grid_ = scinew SearchGrid(*(copy.grid_.get_rep()));
  }
  synchronized_ |= copy.synchronized_ & LOCATE_E;

  lcopy.synchronize_lock_.unlock();
}


template <class Basis>
QuadSurfMesh<Basis>::~QuadSurfMesh()
{
}


template <class Basis>
BBox
QuadSurfMesh<Basis>::get_bounding_box() const
{
  BBox result;

  for (vector<Point>::size_type i = 0; i < points_.size(); i++)
  {
    result.extend(points_[i]);
  }

  return result;
}


template <class Basis>
void
QuadSurfMesh<Basis>::transform(const Transform &t)
{
  synchronize_lock_.lock();
  vector<Point>::iterator itr = points_.begin();
  vector<Point>::iterator eitr = points_.end();
  while (itr != eitr)
  {
    *itr = t.project(*itr);
    ++itr;
  }
  if (grid_.get_rep()) { grid_->transform(t); }
  synchronize_lock_.unlock();
}


template <class Basis>
void
QuadSurfMesh<Basis>::begin(typename QuadSurfMesh::Node::iterator &itr) const
{
  ASSERTMSG(synchronized_ & NODES_E,
            "Must call synchronize NODES_E on QuadSurfMesh first");
  itr = 0;
}


template <class Basis>
void
QuadSurfMesh<Basis>::end(typename QuadSurfMesh::Node::iterator &itr) const
{
  ASSERTMSG(synchronized_ & NODES_E,
            "Must call synchronize NODES_E on QuadSurfMesh first");
  itr = (int)points_.size();
}


template <class Basis>
void
QuadSurfMesh<Basis>::begin(typename QuadSurfMesh::Edge::iterator &itr) const
{
  ASSERTMSG(synchronized_ & EDGES_E,
            "Must call synchronize EDGES_E on QuadSurfMesh first");
  itr = 0;
}


template <class Basis>
void
QuadSurfMesh<Basis>::end(typename QuadSurfMesh::Edge::iterator &itr) const
{
  ASSERTMSG(synchronized_ & EDGES_E,
            "Must call synchronize EDGES_E on QuadSurfMesh first");
  itr = static_cast<typename Edge::iterator>((int)edges_.size());
}


template <class Basis>
void
QuadSurfMesh<Basis>::begin(typename QuadSurfMesh::Face::iterator &itr) const
{
  ASSERTMSG(synchronized_ & FACES_E,
            "Must call synchronize FACES_E on QuadSurfMesh first");
  itr = 0;
}


template <class Basis>
void
QuadSurfMesh<Basis>::end(typename QuadSurfMesh::Face::iterator &itr) const
{
  ASSERTMSG(synchronized_ & FACES_E,
            "Must call synchronize FACES_E on QuadSurfMesh first");
  itr = static_cast<typename Face::iterator>((int)faces_.size() / 4);
}


template <class Basis>
void
QuadSurfMesh<Basis>::begin(typename QuadSurfMesh::Cell::iterator &itr) const
{
  ASSERTMSG(synchronized_ & CELLS_E,
            "Must call synchronize CELLS_E on QuadSurfMesh first");
  itr = 0;
}


template <class Basis>
void
QuadSurfMesh<Basis>::end(typename QuadSurfMesh::Cell::iterator &itr) const
{
  ASSERTMSG(synchronized_ & CELLS_E,
            "Must call synchronize CELLS_E on QuadSurfMesh first");
  itr = 0;
}

template <class Basis>

bool
QuadSurfMesh<Basis>::get_neighbor(typename Face::index_type &neighbor,
                                  typename Face::index_type from,
                                  typename Edge::index_type edge) const
{
  ASSERTMSG(synchronized_ & EDGE_NEIGHBORS_E,
            "Must call synchronize EDGE_NEIGHBORS_E on QuadSurfMesh first");
  unsigned int n = edge_neighbors_[edges_[edge]];
  if (n != MESH_NO_NEIGHBOR && (n / 4) == from)
  {
    n = edge_neighbors_[n];
  }
  if (n != MESH_NO_NEIGHBOR)
  {
    neighbor = n / 4;
    return true;
  }
  return false;
}


template <class Basis>
void
QuadSurfMesh<Basis>::get_neighbors(typename Face::array_type &neighbor,
                                   typename Face::index_type idx) const
{
  ASSERTMSG(synchronized_ & EDGE_NEIGHBORS_E,
            "Must call synchronize EDGE_NEIGHBORS_E on QuadSurfMesh first");
  typename Edge::array_type edges;
  get_edges(edges, idx);

  neighbor.clear();
  typename Edge::array_type::iterator iter = edges.begin();
  while (iter != edges.end())
  {
    typename Face::index_type f;
    if (get_neighbor(f, idx, *iter))
    {
      neighbor.push_back(f);
    }
    ++iter;
  }
}


template <class Basis>
bool
QuadSurfMesh<Basis>::locate(typename Node::index_type &loc,
                            const Point &p) const
{
  typename Node::iterator bi, ei;
  begin(bi);
  end(ei);
  loc = 0;

  bool found = false;
  double mindist = 0.0;
  while (bi != ei)
  {
    const Point &center = point(*bi);
    const double dist = (p - center).length2();
    if (!found || dist < mindist)
    {
      loc = *bi;
      mindist = dist;
      found = true;
    }
    ++bi;
  }
  return found;
}


template <class Basis>
bool
QuadSurfMesh<Basis>::locate(typename Edge::index_type &loc,
                            const Point &p) const
{
  ASSERTMSG(synchronized_ & EDGES_E,
            "Must call synchronize EDGES_E on QuadSurfMesh first");

  typename Edge::iterator bi, ei;
  typename Node::array_type nodes;
  begin(bi);
  end(ei);
  loc = 0;

  bool found = false;
  double mindist = 0.0;
  while (bi != ei)
  {
    get_nodes(nodes,*bi);
    const double dist = distance_to_line2(p, points_[nodes[0]],
                                          points_[nodes[1]]);
    if (!found || dist < mindist)
    {
      loc = *bi;
      mindist = dist;
      found = true;
    }
    ++bi;
  }
  return found;
}


template <class Basis>
bool
QuadSurfMesh<Basis>::inside4_p(typename Face::index_type idx,
                               const Point &p) const
{
  for (unsigned int i = 0; i < 4; i+=2)
  {
    const Point &p0 = points_[faces_[idx*4 + ((i+0)%4)]];
    const Point &p1 = points_[faces_[idx*4 + ((i+1)%4)]];
    const Point &p2 = points_[faces_[idx*4 + ((i+2)%4)]];

    Vector v01(p0-p1);
    Vector v02(p0-p2);
    Vector v0(p0-p);
    Vector v1(p1-p);
    Vector v2(p2-p);
    const double a = Cross(v01, v02).length(); // area of the whole triangle (2x)
    const double a0 = Cross(v1, v2).length();  // area opposite p0
    const double a1 = Cross(v2, v0).length();  // area opposite p1
    const double a2 = Cross(v0, v1).length();  // area opposite p2
    const double s = a0+a1+a2;

    // For the point to be inside a CONVEX quad it must be inside one
    // of the four triangles that can be formed by using three of the
    // quad vertices and the point in question.
    if( fabs(s - a) < MIN_ELEMENT_VAL && a > MIN_ELEMENT_VAL ) {
      return true;
    }
  }

  return false;
}


template <class Basis>
bool
QuadSurfMesh<Basis>::locate(typename Face::index_type &face,
                            const Point &p) const
{
  if (basis_.polynomial_order() > 1) return elem_locate(face, *this, p);

  ASSERTMSG(synchronized_ & LOCATE_E,
            "QuadSurfMesh:: requires synchronization.");

  unsigned int *iter, *end;
  if (grid_.get_rep() == 0) return false;
  
  if (grid_->lookup(&iter, &end, p))
  {
    while (iter != end)
    {
      if (inside4_p(typename Face::index_type(*iter), p))
      {
        face = typename Face::index_type(*iter);
        return true;
      }
      ++iter;
    }
  }
  return false;
}


template <class Basis>
bool
QuadSurfMesh<Basis>::locate(typename Cell::index_type &loc,
                            const Point &) const
{
  loc = 0;
  return false;
}


template <class Basis>
int
QuadSurfMesh<Basis>::get_weights(const Point &p, typename Face::array_type &l,
                                 double *w)
{
  typename Face::index_type idx;
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
int
QuadSurfMesh<Basis>::get_weights(const Point &p, typename Node::array_type &l,
                                 double *w)
{
  typename Face::index_type idx;
  if (locate(idx, p))
  {
    get_nodes(l,idx);
    vector<double> coords(2);
    if (get_coords(coords, p, idx))
    {
      basis_.get_weights(coords, w);
      return basis_.dofs();
    }
  }
  return 0;
}


template <class Basis>
void
QuadSurfMesh<Basis>::get_random_point(Point &p,
                                      typename Elem::index_type ei,
                                      MusilRNG &rng) const
{
  const Point &a0 = points_[faces_[ei*4+0]];
  const Point &a1 = points_[faces_[ei*4+1]];
  const Point &a2 = points_[faces_[ei*4+2]];

  const Point &b0 = points_[faces_[ei*4+2]];
  const Point &b1 = points_[faces_[ei*4+3]];
  const Point &b2 = points_[faces_[ei*4+0]];

  const double aarea = Cross(a1 - a0, a2 - a0).length();
  const double barea = Cross(b1 - b0, b2 - b0).length();

  if (rng() * (aarea + barea) < aarea)
  {
    uniform_sample_triangle(p, a0, a1, a2, rng);
  }
  else
  {
    uniform_sample_triangle(p, b0, b1, b2, rng);
  }
}


template <class Basis>
void
QuadSurfMesh<Basis>::get_center(Point &result,
                                typename Edge::index_type idx) const
{
  typename Node::array_type arr;
  get_nodes(arr, idx);
  result = point(arr[0]);
  result.asVector() += point(arr[1]).asVector();

  result.asVector() *= 0.5;
}


template <class Basis>
void
QuadSurfMesh<Basis>::get_center(Point &p, typename Face::index_type idx) const
{
  typename Node::array_type nodes;
  get_nodes(nodes, idx);
  ASSERT(nodes.size() == 4);
  typename Node::array_type::iterator nai = nodes.begin();
  get_center(p, *nai);
  ++nai;
  while (nai != nodes.end())
  {
    const Point &pp = point(*nai);
    p.asVector() += pp.asVector();
    ++nai;
  }
  p.asVector() *= (1.0 / 4.0);
}


template <class Basis>
bool
QuadSurfMesh<Basis>::synchronize(unsigned int tosync)
{
  synchronize_lock_.lock();
  if (tosync & EDGES_E && !(synchronized_ & EDGES_E))
    compute_edges();
  if (tosync & NORMALS_E && !(synchronized_ & NORMALS_E))
    compute_normals();
  if (tosync & NODE_NEIGHBORS_E && !(synchronized_ & NODE_NEIGHBORS_E))
    compute_node_neighbors();
  if (tosync & EDGE_NEIGHBORS_E && !(synchronized_ & EDGE_NEIGHBORS_E))
    compute_edge_neighbors();
  if (tosync & LOCATE_E && !(synchronized_ & LOCATE_E))
    compute_grid();
  synchronize_lock_.unlock();
  return true;
}


template <class Basis>
void
QuadSurfMesh<Basis>::compute_normals()
{
  normals_.resize(points_.size()); // 1 per node

  // build table of faces that touch each node
  vector<vector<typename Face::index_type> > node_in_faces(points_.size());
  //! face normals (not normalized) so that magnitude is also the area.
  vector<Vector> face_normals(faces_.size());
  // Computing normal per face.
  typename Node::array_type nodes(4);
  typename Face::iterator iter, iter_end;
  begin(iter);
  end(iter_end);
  while (iter != iter_end)
  {
    get_nodes(nodes, *iter);

    Point p0, p1, p2, p3;
    get_point(p0, nodes[0]);
    get_point(p1, nodes[1]);
    get_point(p2, nodes[2]);
    get_point(p3, nodes[3]);

    // build table of faces that touch each node
    node_in_faces[nodes[0]].push_back(*iter);
    node_in_faces[nodes[1]].push_back(*iter);
    node_in_faces[nodes[2]].push_back(*iter);
    node_in_faces[nodes[3]].push_back(*iter);

    Vector v0 = p1 - p0;
    Vector v1 = p2 - p1;
    Vector n = Cross(v0, v1);
    face_normals[*iter] = n;

    ++iter;
  }
  //Averaging the normals.
  typename vector<vector<typename Face::index_type> >::iterator nif_iter =
    node_in_faces.begin();
  int i = 0;
  while (nif_iter != node_in_faces.end()) {
    const vector<typename Face::index_type> &v = *nif_iter;
    typename vector<typename Face::index_type>::const_iterator fiter =
      v.begin();
    Vector ave(0.L,0.L,0.L);
    while(fiter != v.end()) {
      ave += face_normals[*fiter];
      ++fiter;
    }
    ave.safe_normalize();
    normals_[i] = ave; ++i;
    ++nif_iter;
  }
  synchronized_ |= NORMALS_E;
}


template <class Basis>
typename QuadSurfMesh<Basis>::Node::index_type
QuadSurfMesh<Basis>::add_find_point(const Point &p, double err)
{
  typename Node::index_type i;
  if (locate(i, p) && (points_[i] - p).length2() < err)
  {
    return i;
  }
  else
  {
    points_.push_back(p);
    return static_cast<typename Node::index_type>((int)points_.size() - 1);
  }
}


template <class Basis>
typename QuadSurfMesh<Basis>::Elem::index_type
QuadSurfMesh<Basis>::add_quad(typename Node::index_type a,
                              typename Node::index_type b,
                              typename Node::index_type c,
                              typename Node::index_type d)
{
  ASSERTMSG(order_face_nodes(a,b,c,d), "add_quad: element that is being added is invalid");
  faces_.push_back(a);
  faces_.push_back(b);
  faces_.push_back(c);
  faces_.push_back(d);
  return static_cast<typename Elem::index_type>(((int)faces_.size() - 1) >> 2);
}


template <class Basis>
typename QuadSurfMesh<Basis>::Elem::index_type
QuadSurfMesh<Basis>::add_elem(typename Node::array_type a)
{
  ASSERTMSG(a.size() == 4, "Tried to add non-quad element.");
  ASSERTMSG(order_face_nodes(a[0],a[1],a[2],a[3]), "add_elem: element that is being added is invalid");

  faces_.push_back(a[0]);
  faces_.push_back(a[1]);
  faces_.push_back(a[2]);
  faces_.push_back(a[3]);
  return static_cast<typename Elem::index_type>(((int)faces_.size() - 1) >> 2);
}


#ifdef HAVE_HASH_MAP

struct edgehash
{
  size_t operator()(const pair<int, int> &a) const
  {
#if defined(__ECC) || defined(_MSC_VER)
    hash_compare<int> hasher;
#else
    hash<int> hasher;
#endif
    return hasher((int)hasher(a.first) + a.second);
  }
#if defined(__ECC) || defined(_MSC_VER)

  static const size_t bucket_size = 4;
  static const size_t min_buckets = 8;

  bool operator()(const pair<int, int> &a, const pair<int, int> &b) const
  {
    return a.first < b.first || a.first == b.first && a.second < b.second;
  }
#endif
};

struct edgecompare
{
  bool operator()(const pair<int, int> &a, const pair<int, int> &b) const
  {
    return a.first == b.first && a.second == b.second;
  }
};

#else

struct edgecompare
{
  bool operator()(const pair<int, int> &a, const pair<int, int> &b) const
  {
    return a.first < b.first || a.first == b.first && a.second < b.second;
  }
};

#endif

#ifdef HAVE_HASH_MAP

#if defined(__ECC) || defined(_MSC_VER)
typedef hash_map<pair<int, int>, int, edgehash> EdgeMapType;
#else
typedef hash_map<pair<int, int>, int, edgehash, edgecompare> EdgeMapType;
#endif

#else

typedef map<pair<int, int>, int, edgecompare> EdgeMapType;

#endif

#ifdef HAVE_HASH_MAP

#if defined(__ECC) || defined(_MSC_VER)
typedef hash_map<pair<int, int>, list<int>, edgehash> EdgeMapType2;
#else
typedef hash_map<pair<int, int>, list<int>, edgehash, edgecompare> EdgeMapType2;
#endif

#else

typedef map<pair<int, int>, list<int>, edgecompare> EdgeMapType2;

#endif

template <class Basis>
void
QuadSurfMesh<Basis>::compute_edges()
{
  EdgeMapType2 edge_map;

  for( int i=(int)faces_.size()-1; i >= 0; i--)
  {
    const int a = i;
    const int b = a - a % 4 + (a+1) % 4;

    int n0 = faces_[a];
    int n1 = faces_[b];
    int tmp;
    if (n0 > n1) { tmp = n0; n0 = n1; n1 = tmp; }

    if (n0 != n1)
    {
      pair<int, int> nodes(n0, n1);
      edge_map[nodes].push_front(i);
    }
  }

  typename EdgeMapType2::iterator itr;
  edges_.clear();
  edges_.reserve(edge_map.size());
  halfedge_to_edge_.resize(faces_.size());
  for (itr = edge_map.begin(); itr != edge_map.end(); ++itr)
  {
    edges_.push_back((*itr).second.front());

    list<int>::iterator litr = (*itr).second.begin();
    while (litr != (*itr).second.end())
    {
      halfedge_to_edge_[*litr] = edges_.size()-1;
      ++litr;
    }
  }

  synchronized_ |= EDGES_E;
}


template <class Basis>
void
QuadSurfMesh<Basis>::compute_node_neighbors()
{
  node_neighbors_.clear();
  node_neighbors_.resize(points_.size());
  unsigned int i, num_elems = faces_.size();
  for (i = 0; i < num_elems; i++)
  {
    node_neighbors_[faces_[i]].push_back(i);
  }
  synchronized_ |= NODE_NEIGHBORS_E;
}


template <class Basis>
void
QuadSurfMesh<Basis>::compute_edge_neighbors()
{
  EdgeMapType edge_map;

  edge_neighbors_.resize(faces_.size());
  for (unsigned int j = 0; j < edge_neighbors_.size(); j++)
  {
    edge_neighbors_[j] = MESH_NO_NEIGHBOR;
  }

  for(int i = (int)faces_.size()-1; i >= 0; i--)
  {
    const int a = i;
    const int b = a - a % 4 + (a+1) % 4;

    int n0 = faces_[a];
    int n1 = faces_[b];
    int tmp;
    if (n0 > n1) { tmp = n0; n0 = n1; n1 = tmp; }

    pair<int, int> nodes(n0, n1);

    EdgeMapType::iterator maploc;

    maploc = edge_map.find(nodes);
    if (maploc != edge_map.end())
    {
      edge_neighbors_[(*maploc).second] = i;
      edge_neighbors_[i] = (*maploc).second;
    }
    edge_map[nodes] = i;
  }

  synchronized_ |= EDGE_NEIGHBORS_E;
}


template <class Basis>
void
QuadSurfMesh<Basis>::compute_grid()
{
  BBox bb = get_bounding_box();
  if (bb.valid())
  {
    // Cubed root of number of elems to get a subdivision ballpark.
    const double one_third = 1.L/3.L;
    typename Elem::size_type csize;  size(csize);
    const int s = ((int)ceil(pow((double)csize , one_third))) / 2 + 1;
    int sx, sy, sz; sx = sy = sz = s;
    Vector elem_epsilon = bb.diagonal() * (1.0e-3 / s);
    if (elem_epsilon.x() < MIN_ELEMENT_VAL)
    {
      elem_epsilon.x(MIN_ELEMENT_VAL * 100);
      sx = 1;
    }
    if (elem_epsilon.y() < MIN_ELEMENT_VAL)
    {
      elem_epsilon.y(MIN_ELEMENT_VAL * 100);
      sy = 1;
    }
    if (elem_epsilon.z() < MIN_ELEMENT_VAL)
    {
      elem_epsilon.z(MIN_ELEMENT_VAL * 100);
      sz = 1;
    }
    bb.extend(bb.min() - elem_epsilon * 10);
    bb.extend(bb.max() + elem_epsilon * 10);

    SearchGridConstructor sgc(sx, sy, sz, bb.min(), bb.max());

    BBox box;
    typename Node::array_type nodes;
    typename Elem::iterator ci, cie;
    begin(ci); end(cie);
    while(ci != cie)
    {
      get_nodes(nodes, *ci);

      box.reset();
      for (unsigned int i = 0; i < nodes.size(); i++)
      {
        box.extend(points_[nodes[i]]);
      }
      const Point padmin(box.min() - elem_epsilon);
      const Point padmax(box.max() + elem_epsilon);
      box.extend(padmin);
      box.extend(padmax);

      sgc.insert(*ci, box);

      ++ci;
    }

    grid_ = scinew SearchGrid(sgc);
  }

  synchronized_ |= LOCATE_E;
}


template <class Basis>
typename QuadSurfMesh<Basis>::Node::index_type
QuadSurfMesh<Basis>::add_point(const Point &p)
{
  points_.push_back(p);
  return static_cast<typename Node::index_type>((int)points_.size() - 1);
}

template <class Basis>
typename QuadSurfMesh<Basis>::Node::index_type
QuadSurfMesh<Basis>::add_node(const Point &p)
{
  points_.push_back(p);
  return static_cast<typename Node::index_type>((int)points_.size() - 1);
}


template <class Basis>
typename QuadSurfMesh<Basis>::Elem::index_type
QuadSurfMesh<Basis>::add_quad(const Point &p0, const Point &p1,
                              const Point &p2, const Point &p3)
{
  return add_quad(add_find_point(p0), add_find_point(p1),
                  add_find_point(p2), add_find_point(p3));
}


#define QUADSURFMESH_VERSION 3
template <class Basis>
void
QuadSurfMesh<Basis>::io(Piostream &stream)
{
  const int version = stream.begin_class(type_name(-1), QUADSURFMESH_VERSION);

  Mesh::io(stream);

  Pio(stream, points_);
  Pio(stream, faces_);
  
  // In case the face is degenerate
  // move the degerenaracy to the end
  // this way the visualization works fine 
  if (version != 1)
  {
    if (stream.reading())
    {
      for (unsigned int i=0; i < faces_.size(); i += 4)
       if(!( order_face_nodes(faces_[i],faces_[i+1],faces_[i+2],faces_[i+3])))
         std::cerr << "Detected an invalid quadrilateral face\n";
    }
  }
  
  if (version == 1)
  {
    Pio(stream, edge_neighbors_);
  }

  if (version >= 3) {
    basis_.io(stream);
  }

  stream.end_class();

  if (stream.reading())
  {
    synchronized_ = NODES_E | FACES_E | CELLS_E;
  }
}


template <class Basis>
void
QuadSurfMesh<Basis>::size(typename QuadSurfMesh::Node::size_type &s) const
{
  typename Node::iterator itr; end(itr);
  s = *itr;
}


template <class Basis>
void
QuadSurfMesh<Basis>::size(typename QuadSurfMesh::Edge::size_type &s) const
{
  ASSERTMSG(synchronized_ & EDGES_E,
            "Must call synchronize EDGES_E on QuadSurfMesh first");
  s = edges_.size();
}


template <class Basis>
void
QuadSurfMesh<Basis>::size(typename QuadSurfMesh::Face::size_type &s) const
{
  typename Face::iterator itr; end(itr);
  s = *itr;
}


template <class Basis>
void
QuadSurfMesh<Basis>::size(typename QuadSurfMesh::Cell::size_type &s) const
{
  typename Cell::iterator itr; end(itr);
  s = *itr;
}


template <class Basis>
const TypeDescription*
get_type_description(QuadSurfMesh<Basis> *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    const TypeDescription *sub = SCIRun::get_type_description((Basis*)0);
    TypeDescription::td_vec *subs = scinew TypeDescription::td_vec(1);
    (*subs)[0] = sub;
    td = scinew TypeDescription("QuadSurfMesh", subs,
                                string(__FILE__),
                                "SCIRun",
                                TypeDescription::MESH_E);
  }
  return td;
}


template <class Basis>
const TypeDescription*
QuadSurfMesh<Basis>::get_type_description() const
{
  return SCIRun::get_type_description((QuadSurfMesh<Basis> *)0);
}


template <class Basis>
const TypeDescription*
QuadSurfMesh<Basis>::node_type_description()
{
  static TypeDescription *td = 0;
  if (!td)
  {
    const TypeDescription *me =
      SCIRun::get_type_description((QuadSurfMesh<Basis> *)0);
    td = scinew TypeDescription(me->get_name() + "::Node",
                                string(__FILE__),
                                "SCIRun",
                                TypeDescription::MESH_E);
  }
  return td;
}


template <class Basis>
const TypeDescription*
QuadSurfMesh<Basis>::edge_type_description()
{
  static TypeDescription *td = 0;
  if (!td)
  {
    const TypeDescription *me =
      SCIRun::get_type_description((QuadSurfMesh<Basis> *)0);
    td = scinew TypeDescription(me->get_name() + "::Edge",
                                string(__FILE__),
                                "SCIRun",
                                TypeDescription::MESH_E);
  }
  return td;
}


template <class Basis>
const TypeDescription*
QuadSurfMesh<Basis>::face_type_description()
{
  static TypeDescription *td = 0;
  if (!td)
  {
    const TypeDescription *me =
      SCIRun::get_type_description((QuadSurfMesh<Basis> *)0);
    td = scinew TypeDescription(me->get_name() + "::Face",
                                string(__FILE__),
                                "SCIRun",
                                TypeDescription::MESH_E);
  }
  return td;
}


template <class Basis>
const TypeDescription*
QuadSurfMesh<Basis>::cell_type_description()
{
  static TypeDescription *td = 0;
  if (!td)
  {
   const TypeDescription *me =
      SCIRun::get_type_description((QuadSurfMesh<Basis> *)0);
    td = scinew TypeDescription(me->get_name() + "::Cell",
                                string(__FILE__),
                                "SCIRun",
                                TypeDescription::MESH_E);
  }
  return td;
}



// Virtual interface

template <class Basis>
bool
QuadSurfMesh<Basis>::has_virtual_interface() const
{
  return (true);
}
  

template <class Basis>
void
QuadSurfMesh<Basis>::size(VNode::size_type& sz) const
{
  typename Node::index_type s; size(s); sz = VNode::index_type(s);
}

template <class Basis>
void
QuadSurfMesh<Basis>::size(VEdge::size_type& sz) const
{
  typename Edge::index_type s; size(s); sz = VEdge::index_type(s);
}

template <class Basis>
void
QuadSurfMesh<Basis>::size(VFace::size_type& sz) const
{
  typename Face::index_type s; size(s); sz = VFace::index_type(s);
}

template <class Basis>
void
QuadSurfMesh<Basis>::size(VDElem::size_type& sz) const
{
  typename DElem::index_type s; size(s); sz = VDElem::index_type(s);
}

template <class Basis>
void
QuadSurfMesh<Basis>::size(VElem::size_type& sz) const
{
  typename Elem::index_type s; size(s); sz = VElem::index_type(s);
}

template <class Basis>
void 
QuadSurfMesh<Basis>::get_nodes(VNode::array_type& nodes, 
                               VEdge::index_type i) const
{
  get_nodes_from_edge(nodes,i);
}

template <class Basis>
void 
QuadSurfMesh<Basis>::get_nodes(VNode::array_type& nodes, 
                               VFace::index_type i) const
{
  get_nodes_from_face(nodes,i);
}

template <class Basis>
void 
QuadSurfMesh<Basis>::get_nodes(VNode::array_type& nodes, 
                               VElem::index_type i) const
{
  get_nodes_from_face(nodes,i);
}

template <class Basis>
void 
QuadSurfMesh<Basis>::get_nodes(VNode::array_type& nodes, 
                               VDElem::index_type i) const
{
  get_nodes_from_edge(nodes,i);
}

template <class Basis>
void 
QuadSurfMesh<Basis>::get_edges(VEdge::array_type& edges, 
                               VFace::index_type i) const
{
  get_edges_from_face(edges,i);
}

template <class Basis>
void 
QuadSurfMesh<Basis>::get_edges(VEdge::array_type& edges, 
                               VElem::index_type i) const
{
  get_edges_from_face(edges,i);
}

template <class Basis>
void 
QuadSurfMesh<Basis>::get_edges(VEdge::array_type& edges, 
                               VDElem::index_type i) const
{
  edges.resize(1); edges[0] = static_cast<VEdge::index_type>(i);
}

template <class Basis>
void 
QuadSurfMesh<Basis>::get_faces(VFace::array_type& faces, 
                               VNode::index_type i) const
{
  get_faces_from_node(faces,i);
}

template <class Basis>
void 
QuadSurfMesh<Basis>::get_faces(VFace::array_type& faces, 
                               VEdge::index_type i) const
{
  get_faces_from_edge(faces,i);
}

template <class Basis>
void 
QuadSurfMesh<Basis>::get_faces(VFace::array_type& faces, 
                               VElem::index_type i) const
{
  faces.resize(1); faces[0] = static_cast<VFace::index_type>(i);
}

template <class Basis>
void 
QuadSurfMesh<Basis>::get_faces(VFace::array_type& faces, 
                               VDElem::index_type i) const
{
  get_faces_from_edge(faces,i);
}

template <class Basis>
void 
QuadSurfMesh<Basis>::get_elems(VElem::array_type& elems, 
                               VNode::index_type i) const
{
  get_faces_from_node(elems,i);
}

template <class Basis>
void 
QuadSurfMesh<Basis>::get_elems(VElem::array_type& elems, 
                               VEdge::index_type i) const
{
  get_faces_from_edge(elems,i);
}

template <class Basis>
void 
QuadSurfMesh<Basis>::get_elems(VElem::array_type& elems, 
                               VFace::index_type i) const
{
  elems.resize(1); elems[0] = static_cast<VElem::index_type>(i);
}

template <class Basis>
void 
QuadSurfMesh<Basis>::get_elems(VElem::array_type& elems, 
                               VDElem::index_type i) const
{
  get_faces_from_edge(elems,i);
}

template <class Basis>
void 
QuadSurfMesh<Basis>::get_delems(VDElem::array_type& delems, 
                                VFace::index_type i) const
{
  get_edges_from_face(delems,i);
}

template <class Basis>
void 
QuadSurfMesh<Basis>::get_delems(VDElem::array_type& delems, 
                                VElem::index_type i) const
{
  get_edges_from_face(delems,i);
}


template <class Basis>
void
QuadSurfMesh<Basis>::get_center(Point &p, Mesh::VNode::index_type idx) const
{
  p = points_[idx]; 
}

template <class Basis>
void
QuadSurfMesh<Basis>::get_center(Point &p,Mesh::VEdge::index_type idx) const
{
  get_center(p, typename Edge::index_type(idx));
}

template <class Basis>
void
QuadSurfMesh<Basis>::get_center(Point &p, Mesh::VFace::index_type idx) const
{
  get_center(p, typename Face::index_type(idx));
}

template <class Basis>
void
QuadSurfMesh<Basis>::get_center(Point &p, Mesh::VElem::index_type idx) const
{
  get_center(p, typename Elem::index_type(idx));
}

template <class Basis>
void
QuadSurfMesh<Basis>::get_center(Point &p, Mesh::VDElem::index_type idx) const
{
  get_center(p, typename DElem::index_type(idx));
}


template <class Basis>
void
QuadSurfMesh<Basis>::get_weights(const Point& p,VNode::array_type& nodes,
                                              vector<double>& weights) const
{
  typename Face::index_type idx;
  
  if (locate(idx, p))
  {
    get_nodes_from_face(nodes,idx);
    vector<double> coords(3);
    if (get_coords(coords, p, idx))
    {
      weights.resize(basis_.dofs());
      basis_.get_weights(coords, &(weights[0]));
    }
  }
}

template <class Basis>
void
QuadSurfMesh<Basis>::get_weights(const Point& p,VElem::array_type& elems,
                                              vector<double>& weights) const
{
  typename Face::index_type idx;
  if (locate(idx, p))
  {
    elems.resize(1);
    weights.resize(1);
    elems[0] = static_cast<VElem::index_type>(idx);
    weights[0] = 1.0;
  }
  else
  {
    elems.resize(0);
    weights.resize(0);
  }
}

template <class Basis>
bool 
QuadSurfMesh<Basis>::locate(VNode::index_type &vi, const Point &point) const
{
  typename Node::index_type i;
  bool ret = locate(i,point);
  vi = static_cast<VNode::index_type>(i);
  return (ret);
}

template <class Basis>
bool 
QuadSurfMesh<Basis>::locate(VElem::index_type &vi, const Point &point) const
{
  typename Elem::index_type i;
  bool ret = locate(i,point);
  vi = static_cast<VElem::index_type>(i);
  return (ret);
}

template <class Basis>
bool 
QuadSurfMesh<Basis>::get_coords(vector<double> &coords, const Point &point,
                                                    VElem::index_type i) const
{
  return(get_coords(coords,point,typename Elem::index_type(i)));
}  
  
template <class Basis>
void 
QuadSurfMesh<Basis>::interpolate(Point &p, const vector<double> &coords, 
                                                    VElem::index_type i) const
{
  interpolate(p,coords,typename Elem::index_type(i));
}

template <class Basis>
void 
QuadSurfMesh<Basis>::derivate(vector<Point> &p, const vector<double> &coords, 
                                                    VElem::index_type i) const
{
  derivate(coords,typename Elem::index_type(i),p);
}

template <class Basis>
void 
QuadSurfMesh<Basis>::get_points(vector<Point>& points) const
{
  points = points_; 
}

template <class Basis>
void 
QuadSurfMesh<Basis>::set_point(const Point &point, VNode::index_type i)
{
  points_[i] = point;
}

template <class Basis>
void 
QuadSurfMesh<Basis>::add_node(const Point &point,VNode::index_type &vi)
{
  vi = static_cast<VNode::index_type>(add_point(point));
}  
  
template <class Basis>
void 
QuadSurfMesh<Basis>::add_elem(const VNode::array_type &nodes,VElem::index_type &vi)
{
  typename Node::array_type nnodes;
  convert_vector(nnodes,nodes);
  vi = static_cast<VElem::index_type>(add_elem(nnodes));
}  


template <class Basis>
bool 
QuadSurfMesh<Basis>::get_neighbor(VElem::index_type &neighbor, 
                        VElem::index_type from, VDElem::index_type delem) const
{
  typename Elem::index_type n;
  bool ret = get_neighbor(n,typename Elem::index_type(from),
                            typename DElem::index_type(delem));
  neighbor = static_cast<VElem::index_type>(n);
  return (ret);
}

template <class Basis>
void 
QuadSurfMesh<Basis>::get_neighbors(VElem::array_type &varray, 
                                 VElem::index_type i) const
{
  typename Elem::array_type array;
  get_neighbors(array,typename Elem::index_type(i));
  convert_vector(varray,array);
}

template <class Basis>
void 
QuadSurfMesh<Basis>::get_neighbors(VNode::array_type &varray, 
                                 VNode::index_type i) const
{
  vector<typename Node::index_type> array;
  get_neighbors(array,typename Node::index_type(i));
  convert_vector(varray,array);
}

template <class Basis>
double
QuadSurfMesh<Basis>::get_size(VNode::index_type i) const
{
  return (0.0);
}

template <class Basis>
double
QuadSurfMesh<Basis>::get_size(VEdge::index_type i) const
{
  return (get_size(typename Edge::index_type(i)));
}

template <class Basis>
double
QuadSurfMesh<Basis>::get_size(VFace::index_type i) const
{
  return (get_size(typename Face::index_type(i)));
}

template <class Basis>
double
QuadSurfMesh<Basis>::get_size(VElem::index_type i) const
{
  return (get_size(typename Elem::index_type(i)));
}

template <class Basis>
double
QuadSurfMesh<Basis>::get_size(VDElem::index_type i) const
{
  return (get_size(typename DElem::index_type(i)));
}

template <class Basis>
void 
QuadSurfMesh<Basis>::pwl_approx_edge(vector<vector<double> > &coords, 
                                  VElem::index_type ci, unsigned int which_edge,
                                  unsigned int div_per_unit) const
{
  basis_.approx_edge(which_edge, div_per_unit, coords);
}

template <class Basis>
void 
QuadSurfMesh<Basis>::pwl_approx_face(vector<vector<vector<double> > > &coords, 
                                  VElem::index_type ci, unsigned int which_face,
                                  unsigned int div_per_unit) const
{
  basis_.approx_face(which_face, div_per_unit, coords);
}

template <class Basis>
void 
QuadSurfMesh<Basis>::get_random_point(Point &p, VElem::index_type i,MusilRNG &rng) const
{
  get_random_point(p,typename Elem::index_type(i),rng);
}

template <class Basis>
void 
QuadSurfMesh<Basis>::get_normal(Vector& norm,VNode::index_type i) const
{
  get_normal(norm,typename Node::index_type(i));
}  


} // namespace SCIRun


#endif // SCI_project_QuadSurfMesh_h
