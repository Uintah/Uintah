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

//    File   : QuadraticTetVolMesh.h
//    Author : Martin Cole
//    Date   : Sun Feb 24 14:25:39 2002

#ifndef Datatypes_QuadraticTetVolMesh_h
#define Datatypes_QuadraticTetVolMesh_h

#include <Core/Datatypes/TetVolMesh.h>

namespace SCIRun {

class QuadraticTetVolMesh : public TetVolMesh
{
public:
  QuadraticTetVolMesh();
  QuadraticTetVolMesh(const TetVolMesh &tv);
  QuadraticTetVolMesh(const QuadraticTetVolMesh &copy);

  virtual QuadraticTetVolMesh *clone() 
  { return new QuadraticTetVolMesh(*this); }
  virtual ~QuadraticTetVolMesh();

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


  void get_point(Point &result, Node::index_type index) const;

  int get_weights(const Point& p, Node::array_type &l, double *w);
  int get_weights(const Point & , Edge::array_type & , double * )
  { ASSERTFAIL("QuadraticTetVolMesh::get_weights for edges isn't supported"); }
  int get_weights(const Point & , Face::array_type & , double * )
  { ASSERTFAIL("QuadraticTetVolMesh::get_weights for faces isn't supported"); }
  int get_weights(const Point &p, Cell::array_type &l, double *w)
  { return TetVolMesh::get_weights(p, l, w); }

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

  void add_node_neighbors(vector<Node::index_type> &array,
			  Node::index_type node, 
			  const vector<bool> &bc, bool apBC=true);


  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  static  const string type_name(int n = -1);
  virtual const TypeDescription *get_type_description() const;

  bool test_nodes_range(Cell::index_type ci,
			unsigned int sn,
			unsigned int en);

  virtual void compute_nodes();

private:

  vector<unsigned int> node_2_edge_;
#ifdef HAVE_HASH_MAP
#ifdef __ECC
  typedef hash_multimap<unsigned int, unsigned int, Edge::CellEdgeHasher> E2N;
#else
  typedef hash_multimap<unsigned int, unsigned int, Edge::CellEdgeHasher, Edge::eqEdge> E2N;
#endif // ifdef __ECC
#else
  typedef multimap<unsigned int, unsigned int, Edge::lessEdge> E2N;
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
