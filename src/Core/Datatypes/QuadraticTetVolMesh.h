//  The contents of this file are subject to the University of Utah Public
//  License (the "License"); you may not use this file except in compliance
//  with the License.
//  
//  Software distributed under the License is distributed on an "AS IS"
//  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
//  License for the specific language governing rights and limitations under
//  the License.
//  
//  The Original Source Code is SCIRun, released March 12, 2001.
//  
//  The Original Source Code was developed by the University of Utah.
//  Portions created by UNIVERSITY are Copyright (C) 2001, 1994
//  University of Utah. All Rights Reserved.
//  
//    File   : QuadraticTetVolMesh.h
//    Author : Martin Cole
//    Date   : Sun Feb 24 14:25:39 2002

#ifndef Datatypes_QuadraticTetVolMesh_h
#define Datatypes_QuadraticTetVolMesh_h

#include <Core/Datatypes/TetVolMesh.h>

namespace SCIRun {

class SCICORESHARE QuadraticTetVolMesh : public TetVolMesh
{
public:
  QuadraticTetVolMesh();
  QuadraticTetVolMesh(const TetVolMesh &tv);
  QuadraticTetVolMesh(const QuadraticTetVolMesh &copy);

  virtual QuadraticTetVolMesh *clone() 
  { return new QuadraticTetVolMesh(*this); }
  virtual ~QuadraticTetVolMesh();

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
  void get_edges(Edge::array_type &array, Face::index_type idx) const;
  void get_edges(Edge::array_type &array, Cell::index_type idx) const;
  void get_faces(Face::array_type &array, Cell::index_type idx) const;

  //! get the parent element(s) of the given index
  bool get_edges(Edge::array_type &, Node::index_type) const { return 0; }
  bool get_faces(Face::array_type &, Node::index_type) const { return 0; }
  bool get_faces(Face::array_type &, Edge::index_type) const { return 0; }
  bool get_cells(Cell::array_type &, Node::index_type) const { return 0; }
  bool get_cells(Cell::array_type &, Edge::index_type) const { return 0; }
  bool get_cells(Cell::array_type &, Face::index_type) const { return 0; }

  //! must call compute_node_neighbors before calling get_neighbors.
  bool get_neighbor(Cell::index_type &neighbor, Cell::index_type from,
		   Face::index_type idx) const;
  void get_neighbors(Cell::array_type &array, Cell::index_type idx) const;
  void get_neighbors(Node::array_type &array, Node::index_type idx) const;

  //! Get the size of an elemnt (length, area, volume)
  double get_size(Node::index_type idx) const { return 0.0; }
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
    ASSERTFAIL("dont know how to compute the volume for Quad Tets yet");
    return 0.0;
  };
  double get_length(Edge::index_type idx) const { return get_size(idx); };
  double get_area(Face::index_type idx) const { return get_size(idx); };
  double get_volume(Cell::index_type idx) const { return get_size(idx); };

  int get_valence(Node::index_type idx) const
  {
    Node::array_type arr;
    get_neighbors(arr, idx);
    return arr.size();
  }
  int get_valence(Edge::index_type idx) const { return 0; }
  int get_valence(Face::index_type idx) const { return 0; }
  int get_valence(Cell::index_type idx) const
  {
    Cell::array_type arr;
    get_neighbors(arr, idx);
    return arr.size();
  }


  //! get the center point (in object space) of an element
  void get_center(Point &result, Node::index_type idx) const;
  void get_center(Point &result, Edge::index_type idx) const;
  void get_center(Point &result, Face::index_type idx) const;
  void get_center(Point &result, Cell::index_type idx) const;

  //! return false if point is out of range.
  bool locate(Node::index_type &loc, const Point &p);
  bool locate(Edge::index_type &loc, const Point &p);
  bool locate(Face::index_type &loc, const Point &p);
  bool locate(Cell::index_type &loc, const Point &p);

  void get_point(Point &result, Node::index_type index) const;
  void get_normal(Vector &/*normal*/, Node::index_type /*index*/) const
  { ASSERTFAIL("not implemented") }

  void get_weights(const Point& p, Node::array_type &l, vector<double> &w);
  void get_weights(const Point &, Edge::array_type &, vector<double> &) 
  { ASSERTFAIL("QuadraticTetVolMesh::get_weights for edges isn't supported"); }
  void get_weights(const Point &, Face::array_type &, vector<double> &) 
  { ASSERTFAIL("QuadraticTetVolMesh::get_weights for faces isn't supported"); }
  void get_weights(const Point &p, Cell::array_type &l, vector<double> &w)
  { TetVolMesh::get_weights(p, l, w); }
  //! get gradient relative to point p
  void get_gradient_basis(Cell::index_type ci, const Point& p,
			    Vector& g0, Vector& g1, Vector& g2, Vector& g3, 
			    Vector& g4, Vector& g5, Vector& g6, Vector& g7, 
			    Vector& g8, Vector& g9) const;

  //! gradient for gauss pts 
  double get_gradient_basis(Cell::index_type ci, int gaussPt, const Point&, 
			    Vector& g0, Vector& g1, Vector& g2, Vector& g3, 
			    Vector& g4, Vector& g5, Vector& g6, Vector& g7, 
			    Vector& g8, Vector& g9) const;

  void add_node_neighbors(Node::array_type &array, Node::index_type node, 
			  const vector<bool> &bc, bool apBC=true);


  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  static  const string type_name(int n = -1);
  virtual const TypeDescription *get_type_description() const;

  bool test_nodes_range(Cell::index_type ci, int sn, int en);

  virtual void compute_nodes();

protected:


private:
  vector<int> node_2_edge_;
#ifdef HAVE_HASH_MAP
  typedef hash_multimap<int,int,Edge::CellEdgeHasher, Edge::eqEdge> E2N;
#else
  typedef multimap<int,int,Edge::eqEdge> E2N;
#endif
  E2N edge_2_node_;
  bool phantom_nodes_computed_p_;

  double calc_jac_derivs(Vector &dxi, Vector &dnu, Vector &dgam, 
			 double xi, double nu, double gam, 
			 Cell::index_type ci) const;
  double calc_dphi_dgam(int ptNum, double xi, double nu, 
			double gam) const;
  double calc_dphi_dnu(int ptNum, double xi, double nu, 
		       double gam) const;
  double calc_dphi_dxi(int ptNum, double xi, double nu, 
		       double gam) const;
  void heapify(QuadraticTetVolMesh::Node::array_type &data,
	       int n, int i);
};

// Handle type for TetVolMesh mesh.
typedef LockingHandle<QuadraticTetVolMesh> QuadraticTetVolMeshHandle;
const TypeDescription* get_type_description(QuadraticTetVolMesh *);

} // namespace SCIRun


#endif // Datatypes_QuadraticTetVolMesh_h
