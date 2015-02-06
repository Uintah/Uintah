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

namespace SCIRun {
 
//! Class for describing unit geometry of QuadBiquadraticLgn 
  class QuadBiquadraticLgnUnitElement : public QuadBilinearLgnUnitElement {
public: 
  static SCISHARE double unit_vertices[8][2]; //!< Parametric coordinates of vertices of unit edge

  QuadBiquadraticLgnUnitElement() {}
  virtual ~QuadBiquadraticLgnUnitElement() {}
  
  static int number_of_vertices() { return 8; } //!< return number of vertices
  static int dofs() { return 8; } //!< return degrees of freedom
};


//! Class for handling of element of type quad with 
//! biquadratic lagrangian interpolation
template <class T>
class QuadBiquadraticLgn : public BasisAddNodes<T>, 
                           public QuadApprox, 
			   public QuadGaussian3<double>, 
			   public QuadBiquadraticLgnUnitElement
{
public:
  typedef T value_type;

  QuadBiquadraticLgn() {}
  virtual ~QuadBiquadraticLgn() {}
  
  static int polynomial_order() { return 2; }
  
  inline
  static void get_weights(const std::vector<double> &coords, double *w) 
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
  }
  
  //! get first derivative at parametric coordinate 
  template <class ElemData>
  T interpolate(const std::vector<double> &coords, const ElemData &cd) const
  {
    double w[8];
    get_weights(coords, w); 

    return (T)(w[0] * cd.node0() +
	       w[1] * cd.node1() +
	       w[2] * cd.node2() +
	       w[3] * cd.node3() +
	       w[4] * this->nodes_[cd.edge0_index()] +
	       w[5] * this->nodes_[cd.edge1_index()] +
	       w[6] * this->nodes_[cd.edge2_index()] +
	       w[7] * this->nodes_[cd.edge3_index()]);
  }

  //! get derivative weight factors at parametric coordinate 
  inline
  static void get_derivate_weights(const std::vector<double> &coords, double *w) 
  {
    const double x=coords[0], y=coords[1];
    w[0]= -((-1 + y)*(-3 + 4*x + 2*y));
    w[1]= -((-1 + 4*x - 2*y)*(-1 + y));
    w[2]= +y*(-3+ 4*x + 2*y);
    w[3]= +(-1 + 4*x - 2*y)*y;
    w[4]= +4*(-1 + 2*x)*(-1 + y);
    w[5]= -4*(-1 + y)*y;
    w[6]= +(4 - 8*x)*y;
    w[7]= +4*(-1 + y)*y;
    w[8]= -((-1 + x)*(-3 + 2*x +4*y));
    w[9]= +x*(-1 - 2*x + 4*y);
    w[10]= +x*(-3 + 2*x + 4*y);
    w[11]= +(-1 + x)*(1 + 2*x -4*y);
    w[12]= +4*(-1 + x)*x;
    w[13]= +x*(4 -8*y);
    w[14]= -4*(-1 + x)*x;
    w[15]= +4*(-1 + x)*(-1 +2*y);
  }
  
  //! get first derivative at parametric coordinate
  template <class ElemData>
  void derivate(const std::vector<double> &coords, const ElemData &cd, 
		std::vector<T> &derivs) const
  {
    const double x=coords[0], y=coords[1];  

    derivs.resize(2);
    
    derivs[0]=
      T(-((-1 + y)*(-3 + 4*x + 2*y))*cd.node0()
	-((-1 + 4*x - 2*y)*(-1 + y))*cd.node1()
	+y*(-3+ 4*x + 2*y)*cd.node2()
	+(-1 + 4*x - 2*y)*y*cd.node3()
	+4*(-1 + 2*x)*(-1 + y)*this->nodes_[cd.edge0_index()]
	-4*(-1 + y)*y*this->nodes_[cd.edge1_index()]
	+(4 - 8*x)*y*this->nodes_[cd.edge2_index()]
	+4*(-1 + y)*y*this->nodes_[cd.edge3_index()]);
    
    derivs[1]=
      T(-((-1 + x)*(-3 + 2*x +4*y))*cd.node0()
	+x*(-1 - 2*x + 4*y)*cd.node1()
	+x*(-3 + 2*x + 4*y)*cd.node2()
	+(-1 + x)*(1 + 2*x -4*y)*cd.node3()
	+4*(-1 + x)*x*this->nodes_[cd.edge0_index()]
	+x*(4 -8*y)*this->nodes_[cd.edge1_index()]
	-4*(-1 + x)*x*this->nodes_[cd.edge2_index()]
	+4*(-1 + x)*(-1 +2*y)*this->nodes_[cd.edge3_index()]);
  }  
  
  //! get parametric coordinate for value within the element
  template <class ElemData>
  bool get_coords(std::vector<double> &coords, const T& value, 
		  const ElemData &cd) const
  {
    QuadLocate< QuadBiquadraticLgn<T> > CL;
    return CL.get_coords(this, coords, value, cd);
  }  

  //! get arc length for edge
  template <class ElemData>
  double get_arc_length(const unsigned edge, const ElemData &cd) const  
  {
    return get_arc2d_length<CrvGaussian2<double> >(this, edge, cd);
  }
 
 //! get area
  template <class ElemData>
    double get_area(const unsigned face, const ElemData &cd) const  
  {
    return get_area2<QuadGaussian2<double> >(this, face, cd);
  }
 
  //! get volume
  template <class ElemData>
    double get_volume(const ElemData & /* cd */) const  
  {
    return 0.;
  }
  
  static  const std::string type_name(int n = -1);

  virtual void io (Piostream& str);
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
				std::string(__FILE__),
				"SCIRun", 
				TypeDescription::BASIS_E);
  }
  return td;
}

template <class T>
const std::string
QuadBiquadraticLgn<T>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const std::string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;
  }
  else if (n == 0)
  {
    static const std::string nm("QuadBiquadraticLgn");
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
  Pio(stream, this->nodes_);
  stream.end_class();
}


} //namespace SCIRun


#endif // QuadBiquadraticLgn_h
