/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */
//  
//    File   : QuadBicubicHmt.h
//    Author : Martin Cole, Frank B. Sachse
//    Date   : Dec 04 2004

#if !defined(QuadBicubicHmt_h)
#define QuadBicubicHmt_h

#include <Core/Basis/QuadBilinearLgn.h>

namespace SCIRun {
  
//! Class for describing unit geometry of QuadBicubicHmt
class QuadBicubicHmtUnitElement : public QuadBilinearLgnUnitElement {
public:
  QuadBicubicHmtUnitElement() {}
  virtual ~QuadBicubicHmtUnitElement() {}

  static int dofs() { return 12; } //!< return degrees of freedom
};


//! Class for handling of element of type quad with 
//! bicubic hermitian interpolation
template <class T>
class QuadBicubicHmt : public BasisAddDerivatives<T>, 
                       public QuadApprox, 
		       public QuadGaussian2<double>, 
		       public QuadBicubicHmtUnitElement
{
public:
  typedef T value_type;

  QuadBicubicHmt() {}
  virtual ~QuadBicubicHmt() {}
  
  static int polynomial_order() { return 3; }

  inline
  static void get_weights(const std::vector<double> &coords, double *w) 
  { 
    const double x=coords[0], y=coords[1];  
    w[0]  = -((-1 + x)*(-1 + y)*(-1 - x + 2*x*x - y + 2*y*y));
    w[1]  = -((x-1)*(x-1)*x*(-1 + y));
    w[2]  = -((-1 + x)*(y-1)*(y-1)*y);
    w[3]  = +x*(-1 + y)*(-3*x + 2*x*x + y*(-1 + 2*y));
    w[4]  = +x*x*(-1 + x + y - x*y);
    w[5]  = +x*(y-1)*(y-1)*y;
    w[6]  = +x*y*(-1 + 3*x - 2*x*x + 3*y - 2*y*y);
    w[7]  = +(-1 + x)*x*x*y;
    w[8]  = +x*(-1 + y)*y*y;
    w[9]  = +(-1 + x)*y*(-x + 2*x*x + y*(-3 + 2*y));
    w[10] = +(x-1)*(x-1)*x*y;
    w[11] = +y*y*(-1 + x + y - x*y);
  }
  //! get value at parametric coordinate 
  template <class ElemData>
  T interpolate(const std::vector<double> &coords, const ElemData &cd) const
  {
    double w[12];
    get_weights(coords, w); 
    return (T)(w[0]  * cd.node0()                   +
	       w[1]  * this->derivs_[cd.node0_index()][0] +
	       w[2]  * this->derivs_[cd.node0_index()][1] +
	       w[3]  * cd.node1()		    +
	       w[4]  * this->derivs_[cd.node1_index()][0] +
	       w[5]  * this->derivs_[cd.node1_index()][1] +
	       w[6]  * cd.node2()		    +
	       w[7]  * this->derivs_[cd.node2_index()][0] +
	       w[8]  * this->derivs_[cd.node2_index()][1] +
	       w[9]  * cd.node3()		    +
	       w[10] * this->derivs_[cd.node3_index()][0] +
	       w[11] * this->derivs_[cd.node3_index()][1]);
  }
  
  //! get derivative weight factors at parametric coordinate 
  inline
  static void get_derivate_weights(const std::vector<double> &coords, double *w) 
  {
    const double x=coords[0], y=coords[1];
    w[0]= -((-1 + y)*(-6*x + 6*x*x + y*(-1 + 2*y)));
    w[1]= -((1 - 4*x + 3*x*x)*(-1 + y));
    w[2]= -((y-1)*(y-1)*y);
    w[3]= +(-1 + y)*(-6*x + 6*x*x + y*(-1 + 2*y));
    w[4]= -(x*(-2 + 3*x)*(-1 + y));
    w[5]= +(y-1)*(y-1)*y;
    w[6]= +y*(-1 + 6*x - 6*x*x + 3*y - 2*y*y);
    w[7]= +x*(-2 + 3*x)*y;
    w[8]= +(-1 + y)*y*y;
    w[9]= +y*(1 - 6*x + 6*x*x - 3*y + 2*y*y);
    w[10]= +(1 - 4*x + 3*x*x)*y;
    w[11]= -((-1 + y)*y*y);
    w[12]= -((-1 + x)*(-x + 2*x*x + 6*(-1 + y)*y));
    w[13]= -((x-1)*(x-1)*x);
    w[14]= -((-1 + x)*(1 - 4*y + 3*y*y));
    w[15]= +x*(1 - 3*x + 2*x*x - 6*y + 6*y*y);
    w[16]= -((-1 + x)*x*x);
    w[17]= +x*(1 - 4*y + 3*y*y);
    w[18]= +x*(-1 + 3*x - 2*x*x + 6*y - 6*y*y);
    w[19]= +(-1 + x)*x*x;
    w[20]= +x*y*(-2 + 3*y);
    w[21]= +(-1 + x)*(-x + 2*x*x + 6*(-1 + y)*y);
    w[22]= +(x-1)*(x-1)*x;
    w[23]= -((-1 + x)*y*(-2 + 3*y));
  }

  //! get first derivative at parametric coordinate
  template <class ElemData>
  void derivate(const std::vector<double> &coords, const ElemData &cd, 
		std::vector<T> &derivs) const
  {
    const double x=coords[0], y=coords[1];  

    derivs.resize(2);

    derivs[0]=
      T(-((-1 + y)*(-6*x + 6*x*x + y*(-1 + 2*y)))*cd.node0()
	-((1 - 4*x + 3*x*x)*(-1 + y))*this->derivs_[cd.node0_index()][0]
	-((y-1)*(y-1)*y)*this->derivs_[cd.node0_index()][1]
	+(-1 + y)*(-6*x + 6*x*x + y*(-1 + 2*y))*cd.node1()
	-(x*(-2 + 3*x)*(-1 + y))*this->derivs_[cd.node1_index()][0]
	+(y-1)*(y-1)*y*this->derivs_[cd.node1_index()][1]
	+y*(-1 + 6*x - 6*x*x + 3*y - 2*y*y)*cd.node2()
	+x*(-2 + 3*x)*y*this->derivs_[cd.node2_index()][0]
	+(-1 + y)*y*y*this->derivs_[cd.node2_index()][1]
	+y*(1 - 6*x + 6*x*x - 3*y + 2*y*y)*cd.node3()
	+(1 - 4*x + 3*x*x)*y*this->derivs_[cd.node3_index()][0]
	-((-1 + y)*y*y)*this->derivs_[cd.node3_index()][1]);
	
    derivs[1]= 
      T(-((-1 + x)*(-x + 2*x*x + 6*(-1 + y)*y))*cd.node0()
	-((x-1)*(x-1)*x)*this->derivs_[cd.node0_index()][0]
	-((-1 + x)*(1 - 4*y + 3*y*y))*this->derivs_[cd.node0_index()][1]
	+x*(1 - 3*x + 2*x*x - 6*y + 6*y*y)*cd.node1()
	-((-1 + x)*x*x)*this->derivs_[cd.node1_index()][0]
	+x*(1 - 4*y + 3*y*y)*this->derivs_[cd.node1_index()][1]
	+x*(-1 + 3*x - 2*x*x + 6*y - 6*y*y)*cd.node2()
	+(-1 + x)*x*x*this->derivs_[cd.node2_index()][0]
	+x*y*(-2 + 3*y)*this->derivs_[cd.node2_index()][1]
	+(-1 + x)*(-x + 2*x*x + 6*(-1 + y)*y)*cd.node3()
	+(x-1)*(x-1)*x*this->derivs_[cd.node3_index()][0]
	-((-1 + x)*y*(-2 + 3*y))*this->derivs_[cd.node3_index()][1]);
  }
  
  //! get parametric coordinate for value within the element
  template <class ElemData>
  bool get_coords(std::vector<double> &coords, const T& value, 
		  const ElemData &cd) const
  {
    QuadLocate< QuadBicubicHmt<T> > CL;
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
    return get_area2<QuadGaussian3<double> >(this, face, cd);
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
const TypeDescription* get_type_description(QuadBicubicHmt<T> *)
{
  static TypeDescription* td = 0;
  if(!td){
    const TypeDescription *sub = get_type_description((T*)0);
    TypeDescription::td_vec *subs = scinew TypeDescription::td_vec(1);
    (*subs)[0] = sub;
    td = scinew TypeDescription("QuadBicubicHmt", subs, 
				std::string(__FILE__),
				"SCIRun", 
				TypeDescription::BASIS_E);
  }
  return td;
}

template <class T>
const std::string
QuadBicubicHmt<T>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const std::string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;
  }
  else if (n == 0)
  {
    static const std::string nm("QuadBicubicHmt");
    return nm;
  } else {
    return find_type_name((T *)0);
  }
}


const int QUADBICUBICHMT_VERSION = 1;
template <class T>
void
QuadBicubicHmt<T>::io(Piostream &stream)
{
  stream.begin_class(get_type_description(this)->get_name(),
                     QUADBICUBICHMT_VERSION);
  Pio(stream, this->derivs_);
  stream.end_class();
}


} //namespace SCIRun

#endif // QuadBicubicHmt_h
