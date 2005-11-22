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
//    File   : CrvQuadraticLgn.h
//    Author : Marty Cole, Frank B. Sachse
//    Date   : Nov 30 2004

#if !defined(CrvQuadraticLgn_h)
#define CrvQuadraticLgn_h

#include <Core/Basis/CrvLinearLgn.h>

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
// Turn off 'implicit conversion... loss of accuracy' messages.
#  pragma set woff 1506
#endif


namespace SCIRun {

//! Class for describing unit geometry of CrvLinearLgn 
class CrvQuadraticLgnUnitElement {
public: 
  static double unit_vertices[3][1]; //!< Parametric coordinates of vertices of unit edge
  static int unit_edges[1][2];    //!< References to vertices of unit edge 

  CrvQuadraticLgnUnitElement() {}
  virtual ~CrvQuadraticLgnUnitElement() {}
  
  static int domain_dimension() { return 1; } //!< return dimension of domain 
  
  static int number_of_vertices() { return 3; } //!< return number of vertices
  static int number_of_mesh_vertices() { return 2; } //!< return number of vertices in mesh
  static int number_of_edges() { return 2; } //!< return number of edges
  
  static int vertices_of_face() { return 0; } //!< return number of vertices per face 

  static int faces_of_cell() { return 0; } //!< return number of faces per cell 
};


//! Class for handling of element of type curve with 
//! quadratic lagrangian interpolation
template <class T>
  class CrvQuadraticLgn : public BasisAddNodes<T>, 
                          public CrvApprox, 
			  public CrvGaussian2<double>, 
			  public CrvQuadraticLgnUnitElement
{
public:
  typedef T value_type;
     
  CrvQuadraticLgn() {}
  virtual ~CrvQuadraticLgn() {}
  
  int polynomial_order() const { return 2; }

  //! get weight factors at parametric coordinate 
  inline
  int get_weights(const vector<double> &coords, double *w) const
  {
    const double x = coords[0];
    w[0] = 1 - 3*x + 2*x*x;
    w[1] = x*(-1 + 2*x);
    w[2] = -4*(-1 + x)*x;

    return 3;
  }
  
  //! get value at parametric coordinate
  template <class ElemData>
  T interpolate(const vector<double> &coords, const ElemData &cd) const
  {
    double w[3];
    get_weights(coords, w); 
 
    return T(w[0] * cd.node0() +
	     w[1] * cd.node1() +
	     w[2] * this->nodes_[cd.edge0_index()]);
  }
    
  //! get first derivative at parametric coordinate
  template <class ElemData>
  void derivate(const vector<double> &coords, const ElemData &cd, 
		vector<T> &derivs) const
  {
    const double x=coords[0];  
    
    derivs.resize(1);

    derivs[0] = T((-3 + 4*x) * cd.node0() 
		  +(-1 + 4*x)* cd.node1()
		  +(4 - 8*x)* this->nodes_[cd.edge0_index()]);
  }
  
  static  const string type_name(int n = -1);

  //! get parametric coordinate for value within the element
  template <class ElemData>
  bool get_coords(vector<double> &coords, const T& value, 
		  const ElemData &cd) const  
  {
    CrvLocate< CrvQuadraticLgn<T> > CL;
    return CL.get_coords(this, coords, value, cd);
  }
     
  virtual void io (Piostream& str);
};


template <class T>
const TypeDescription* get_type_description(CrvQuadraticLgn<T> *)
{
  static TypeDescription* td = 0;
  if(!td){
    const TypeDescription *sub = get_type_description((T*)0);
    TypeDescription::td_vec *subs = scinew TypeDescription::td_vec(1);
    (*subs)[0] = sub;
    td = scinew TypeDescription("CrvQuadraticLgn", subs, 
				string(__FILE__),
				"SCIRun", 
				TypeDescription::BASIS_E);
  }
  return td;
}

template <class T>
const string
CrvQuadraticLgn<T>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;
  }
  else if (n == 0)
  {
    static const string nm("CrvQuadraticLgn");
    return nm;
  } else {
    return find_type_name((T *)0);
  }
}

const int CRVQUADRATICLGN_VERSION = 1;
template <class T>
void
CrvQuadraticLgn<T>::io(Piostream &stream)
{
  stream.begin_class(get_type_description(this)->get_name(),
                     CRVQUADRATICLGN_VERSION);
  Pio(stream, this->nodes_);
  stream.end_class();
}
  
} //namespace SCIRun

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
// Turn back on 'implicit conversion... loss of accuracy' messages.
#  pragma reset woff 1506
#endif

#endif // CrvQuadraticLgn_h
