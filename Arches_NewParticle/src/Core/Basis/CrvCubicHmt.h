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
//    File   : CrvCubicHmt.h
//    Author : Martin Cole, Frank B. Sachse
//    Date   : Dec 04 2004

#if !defined(CrvCubicHmt_h)
#define CrvCubicHmt_h

#include <Core/Basis/CrvLinearLgn.h>

namespace SCIRun {

//! Class for describing unit geometry of CrvCubicHmt 
class CrvCubicHmtUnitElement : public CrvLinearLgnUnitElement {
public:
  CrvCubicHmtUnitElement() {}
  virtual ~CrvCubicHmtUnitElement() {}

  static int dofs() { return 4; } //!< return degrees of freedom
};


//! Class for handling of element of type curve with 
//! cubic hermitian interpolation
template <class T>
class CrvCubicHmt : public BasisAddDerivatives<T>, 
                    public CrvApprox, 
		    public CrvGaussian3<double>, 
		    public CrvCubicHmtUnitElement
{
public:
  typedef T value_type;

  static int    GaussianNum;
  static double GaussianPoints[3][1];
  static double GaussianWeights[3];
  
  CrvCubicHmt() {}
  virtual ~CrvCubicHmt() {}
  
  static int polynomial_order() { return 3; }

  //! get weight factors at parametric coordinate 
  inline
  static void get_weights(const std::vector<double> &coords, double *w) 
  {
    const double x = coords[0];
    w[0] = (x-1)*(x-1)*(1 + 2*x);
    w[1] = (x-1)*(x-1)*x;
    w[2] = (3 - 2*x)*x*x;
    w[3] = (-1+x)*x*x;
  }
  
  //! get value at parametric coordinate
  template <class ElemData>
  T interpolate(const std::vector<double> &coords, const ElemData &cd) const
  {
    double w[4];
    get_weights(coords, w); 
    return T(w[0] * cd.node0() +
	     w[1] * this->derivs_[cd.node0_index()][0] +
	     w[2] * cd.node1() +
	     w[3] * this->derivs_[cd.node1_index()][0]);
  }
  
  //! get derivative weight factors at parametric coordinate 
  inline
  static void get_derivate_weights(const std::vector<double> &coords, double *w) 
  {
    const double x = coords[0];
    w[0] = 6*(-1 + x)*x;
    w[1] = (1 - 4*x + 3*x*x);
    w[2] = -6*(-1 + x)*x;
    w[3] = x*(-2 + 3*x);
  }

  //! get first derivative at parametric coordinate
  template <class ElemData>
  void derivate(const std::vector<double> &coords, const ElemData &cd, 
		std::vector<T> &derivs) const
  {
    const double x=coords[0]; 
 
    derivs.resize(1);

    derivs[0] = T(6*(-1 + x)*x * cd.node0() 
		  +(1 - 4*x + 3*x*x) * this->derivs_[cd.node0_index()][0] 
		  -6*(-1 + x)*x * cd.node1() 
		  +x*(-2 + 3*x) * this->derivs_[cd.node1_index()][0]);
  }

  //! get parametric coordinate for value within the element
  template <class ElemData>
  bool get_coords(std::vector<double> &coords, const T& value, 
		  const ElemData &cd) const  
  {
    CrvLocate< CrvCubicHmt<T> > CL;
    return CL.get_coords(this, coords, value, cd);
  }
     
  //! get arc length for edge
  template <class ElemData>
  double get_arc_length(const unsigned edge, const ElemData &cd) const  
  {
    return get_arc1d_length<CrvGaussian3<double> >(this, edge, cd);
  }
 
  //! get area
  template <class ElemData>
    double get_area(const unsigned /* face */, const ElemData & /* cd */) const  
  {
    return 0.;
  }
 
  //! get volume
  template <class ElemData>
    double get_volume(const ElemData & /* cd */) const  
  {
    return 0.;
  }
  
  static  const string type_name(int n = -1);

  virtual void io (Piostream& str);
};


template <class T>
const TypeDescription* get_type_description(CrvCubicHmt<T> *)
{
  static TypeDescription* td = 0;
  if(!td){
    const TypeDescription *sub = get_type_description((T*)0);
    TypeDescription::td_vec *subs = scinew TypeDescription::td_vec(1);
    (*subs)[0] = sub;
    td = scinew TypeDescription("CrvCubicHmt", subs, 
				std::string(__FILE__),
				"SCIRun", 
				TypeDescription::BASIS_E);
  }
  return td;
}


template <class T>
const std::string
CrvCubicHmt<T>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const std::string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;
  }
  else if (n == 0)
  {
    static const std::string nm("CrvCubicHmt");
    return nm;
  } else {
    return find_type_name((T *)0);
  }
}

const int CRVCUBICHMT_BASIS_VERSION = 1;
template <class T>
void
CrvCubicHmt<T>::io(Piostream &stream)
{
  stream.begin_class(get_type_description(this)->get_name(),
                     CRVCUBICHMT_BASIS_VERSION);
  Pio(stream, this->derivs_);
  stream.end_class();
}
 

} //namespace SCIRun


#endif // CrvCubicHmt_h
