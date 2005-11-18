//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2004 Scientific Computing and Imaging Institute,
//  University of Utah.
//  
//  License for the specific language governing rights and limitations under
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//  
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
//  
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//  
//    File   : QuadBiquadraticLgn.h
//    Author : Marty Cole, Frank B. Sachse
//    Date   : 30 Nov 2004

#if !defined(QuadBiquadraticLgn_h)
#define QuadBiquadraticLgn_h

#include <Core/Basis/QuadBilinearLgn.h>

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
// Turn off 'implicit conversion... loss of accuracy' messages.
#  pragma set woff 1506
#endif


namespace SCIRun {
 
//! Class for describing unit geometry of QuadBiquadraticLgn 
class QuadBiquadraticLgnUnitElement {
public: 
  static double unit_vertices[8][2]; //!< Parametric coordinates of vertices of unit edge
  static int unit_edges[4][2];  //!< References to vertices of unit edge 
  static int unit_faces[1][4]; //!< References to vertices of unit face
  
  QuadBiquadraticLgnUnitElement() {}
  virtual ~QuadBiquadraticLgnUnitElement() {}
  
  static int domain_dimension() { return 2; } //!< return dimension of domain 
  
  static int number_of_vertices() { return 8; } //!< return number of vertices
  static int number_of_mesh_vertices() { return 4; } //!< return number of vertices in mesh
  static int number_of_edges() { return 4; } //!< return number of edges
  
  static int vertices_of_face() { return 4; } //!< return number of vertices per face 

  static int faces_of_cell() { return 4; } //!< return number of faces per cell 
};


//! Class for handling of element of type quad with 
//! biquadratic lagrangian interpolation
template <class T>
class QuadBiquadraticLgn : public BasisSimple<T>, 
                           public QuadApprox, 
			   public QuadGaussian3<double>, 
			   public QuadBiquadraticLgnUnitElement
{
public:
  typedef T value_type;

  QuadBiquadraticLgn() {}
  virtual ~QuadBiquadraticLgn() {}
  
  int polynomial_order() const { return 2; }
  
  inline
  int get_weights(const vector<double> &coords, double *w) const
  { 
    const double x=coords[0], y=coords[1];  
    
    w[0] = -((-1 + x)*(-1 + y)*(-1 + 2*x + 2*y));
    w[1] = -(x*(-1 + 2*x - 2*y)*(-1 +y));
    w[2] = +x*y*(-3 + 2*x + 2*y);
    w[3] = +(-1 + x)*(1 + 2*x - 2*y)*y;
    w[4] = +4*(-1 + x)*x*(-1 + y);
    w[5] = -4*x*(-1 + y)*y;
    w[6] = -4*(-1 + x)*x*y;
    w[7] = +4*(-1 + x)*(-1 + y)*y;
    
    return 8;
  }
  
  //! get first derivative at parametric coordinate 
  template <class ElemData>
  T interpolate(const vector<double> &coords, const ElemData &cd) const
  {
    double w[8];
    get_weights(coords, w); 

    return (T)(w[0] * cd.node0() +
	       w[1] * cd.node1() +
	       w[2] * cd.node2() +
	       w[3] * cd.node3() +
	       w[4] * nodes_[cd.edge0_index()] +
	       w[5] * nodes_[cd.edge1_index()] +
	       w[6] * nodes_[cd.edge2_index()] +
	       w[7] * nodes_[cd.edge3_index()]);
  }
  
  //! get first derivative at parametric coordinate
  template <class ElemData>
  void derivate(const vector<double> &coords, const ElemData &cd, 
		vector<T> &derivs) const
  {
    const double x=coords[0], y=coords[1];  

    derivs.resize(2);
    
    derivs[0]=
      T(-((-1 + y)*(-3 + 4*x + 2*y))*cd.node0()
	-((-1 + 4*x - 2*y)*(-1 + y))*cd.node1()
	+y*(-3+ 4*x + 2*y)*cd.node2()
	+(-1 + 4*x - 2*y)*y*cd.node3()
	+4*(-1 + 2*x)*(-1 + y)*nodes_[cd.edge0_index()]
	-4*(-1 + y)*y*nodes_[cd.edge1_index()]
	+(4 - 8*x)*y*nodes_[cd.edge2_index()]
	+4*(-1 + y)*y*nodes_[cd.edge3_index()]);
    
    derivs[1]=
      T(-((-1 + x)*(-3 + 2*x +4*y))*cd.node0()
	+x*(-1 - 2*x + 4*y)*cd.node1()
	+x*(-3 + 2*x + 4*y)*cd.node2()
	+(-1 + x)*(1 + 2*x -4*y)*cd.node3()
	+4*(-1 + x)*x*nodes_[cd.edge0_index()]
	+x*(4 -8*y)*nodes_[cd.edge1_index()]
	-4*(-1 + x)*x*nodes_[cd.edge2_index()]
	+4*(-1 + x)*(-1 +2*y)*nodes_[cd.edge3_index()]);
  }  
  
  //! get parametric coordinate for value within the element
  template <class ElemData>
  bool get_coords(vector<double> &coords, const T& value, 
		  const ElemData &cd) const
  {
    QuadLocate< QuadBiquadraticLgn<T> > CL;
    return CL.get_coords(this, coords, value, cd);
  }  

  //! add a node value corresponding to edge
  void add_node_value(const T &p) { nodes_.push_back(p); }

  static  const string type_name(int n = -1);
  virtual void io (Piostream& str);
  
protected:
  //! Additional support values.

  //! Quadratic Lagrangian only needs additional nodes stored for each edge
  //! in the topology.
  vector<T>          nodes_; 
};


template <class T>
const TypeDescription* get_type_description(QuadBiquadraticLgn<T> *)
{
  static TypeDescription* td = 0;
  if(!td){
    const TypeDescription *sub = get_type_description((T*)0);
    TypeDescription::td_vec *subs = scinew TypeDescription::td_vec(1);
    (*subs)[0] = sub;
    td = scinew TypeDescription("QuadBiquadraticLgn", subs, 
				string(__FILE__),
				"SCIRun", 
				TypeDescription::BASIS_E);
  }
  return td;
}

template <class T>
const string
QuadBiquadraticLgn<T>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;
  }
  else if (n == 0)
  {
    static const string nm("QuadBiquadraticLgn");
    return nm;
  } else {
    return find_type_name((T *)0);
  }
}


const int QUADBIQUADRATICLGN_VERSION = 1;
template <class T>
void
QuadBiquadraticLgn<T>::io(Piostream &stream)
{
  stream.begin_class(get_type_description(this)->get_name(),
                     QUADBIQUADRATICLGN_VERSION );
  Pio(stream, nodes_);
  stream.end_class();
}


} //namespace SCIRun

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
// Turn back on 'implicit conversion... loss of accuracy' messages.
#  pragma reset woff 1506
#endif


#endif // QuadBiquadraticLgn_h
