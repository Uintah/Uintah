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
//    File   : TriCubicHmtScaleFactors.h
//    Author : Martin Cole, Frank B. Sachse
//    Date   : Mar 01 2005

#if !defined(TriCubicHmtScaleFactors_h)
#define TriCubicHmtScaleFactors_h

#include <Core/Persistent/PersistentSTL.h>
#include <Core/Basis/TriLinearLgn.h>

namespace SCIRun {

//! Class for describing unit geometry of TetCubicHmt
class TriCubicScaleFactorsHmtUnitElement : public TriLinearLgnUnitElement {
public:
  TriCubicScaleFactorsHmtUnitElement() {}
  virtual ~TriCubicScaleFactorsHmtUnitElement() {}

  static int dofs() { return 12; } //!< return degrees of freedom
};


//! Class for handling of element of type triangle with 
//! cubic hermitian interpolation with scale factors
template <class T>
class TriCubicHmtScaleFactors : public BasisAddDerivativesScaleFactors<T>, 
                                public TriApprox, 
				public TriGaussian3<double>,
				public TriCubicScaleFactorsHmtUnitElement
{
public:
  typedef T value_type;

  TriCubicHmtScaleFactors() {}
  virtual ~TriCubicHmtScaleFactors() {}

  inline
  static void get_weights(const std::vector<double> &coords, double *w) 
  {
    const double x=coords[0], y=coords[1];  
    const double x2=x*x, x3=x2*x, y2=y*y, y3=y2*y;

    w[0]  = (-1 + x + y)*(-1 - x + 2*x2 - y - 2*x*y + 2*y2);
    w[1]  = +x*(1 - 2*x + x2 - 3*y2 + 2*y3);
    w[2]  = +y*(1 - 3*x2 + 2*x3 - 2*y + y2);
    w[3]  = +x*y*(1 - 2*x + x2 - 2*y + y2);
    w[4]  = -(x2*(-3 + 2*x));
    w[5]  = +(-1 + x)*x2;
    w[6]  = -(x2*(-3 + 2*x)*y);
    w[7]  = +(-1 + x)*x2*y;
    w[8]  = -y2*(-3 + 2*y);
    w[9]  = -(x*y2*(-3 + 2*y));
    w[10] = +(-1 + y)*y2;
    w[11] = +x*(-1 + y)*y2;
  }

  //! get value at parametric coordinate 
  template <class ElemData>
  T interpolate(const std::vector<double> &coords, const ElemData &cd) const
  {
    double w[12];
    get_weights(coords, w); 

    unsigned elem=cd.elem_index();
    const T sdx0=this->derivs_[cd.node0_index()][0]*this->scalefactors_[elem][0];
    const T sdx1=this->derivs_[cd.node1_index()][0]*this->scalefactors_[elem][0];
    const T sdx2=this->derivs_[cd.node2_index()][0]*this->scalefactors_[elem][0];

    const T sdy0=this->derivs_[cd.node0_index()][1]*this->scalefactors_[elem][1];
    const T sdy1=this->derivs_[cd.node1_index()][1]*this->scalefactors_[elem][1];
    const T sdy2=this->derivs_[cd.node2_index()][1]*this->scalefactors_[elem][1];

    const T sdxy0=this->derivs_[cd.node0_index()][2]*this->scalefactors_[elem][0]*this->scalefactors_[elem][1];
    const T sdxy1=this->derivs_[cd.node1_index()][2]*this->scalefactors_[elem][0]*this->scalefactors_[elem][1];
    const T sdxy2=this->derivs_[cd.node2_index()][2]*this->scalefactors_[elem][0]*this->scalefactors_[elem][1];

    return (T)(w[0]  * cd.node0()
	       +w[1]  * sdx0
	       +w[2]  * sdy0
	       +w[3]  * sdxy0
	       +w[4]  * cd.node1()
	       +w[5]  * sdx1
	       +w[6]  * sdy1
	       +w[7]  * sdxy1
	       +w[8]  * cd.node2()
	       +w[9]  * sdx2
	       +w[10] * sdy2
	       +w[11] * sdxy2);
  }
  
  //! get derivative weight factors at parametric coordinate 
  inline
  static void get_derivate_weights(const std::vector<double> &coords, double *w) 
  {
    const double x=coords[0], y=coords[1];  
    const double x2=x*x, x3=x2*x, y2=y*y;
    const double y12=(y-1)*(y-1);
    w[0] = 6*(-1 + x)*x;
    w[1] = (-4*x + 3*x2 + y12*(1 + 2*y));
    w[2] = 6*(-1 + x)*x*y;
    w[3] = (-4*x + 3*x2 + y12)*y;
    w[4] = -6*(-1 + x)*x;
    w[5] = x*(-2 + 3*x);
    w[6] = -6*(-1 + x)*x*y;
    w[7] = x*(-2 + 3*x)*y;
    w[8] = (3 - 2*y)*y2;
    w[9] = (-1 + y)*y2;
    w[10] = 6*(-1 + y)*y;
    w[11] = 6*x*(-1 + y)*y;
    w[12] = (1 - 3*x2 + 2*x3 - 4*y + 3*y2);
    w[13] = x*(1 - 2*x + x2 - 4*y + 3*y2);
    w[14] = (3 - 2*x)*x2;
    w[15] = (-1 + x)*x2;
    w[16] = -6*(-1 + y)*y;
    w[17] = -6*x*(-1 + y);
    w[18] = y*(-2 + 3*y);
    w[19] = x*y*(-2 + 3*y);
  }

  //! get first derivative at parametric coordinate
  template <class ElemData>
  void derivate(const std::vector<double> &coords, const ElemData &cd, 
		std::vector<T> &derivs) const
  {
    const double x=coords[0], y=coords[1];  
    const double x2=x*x, x3=x2*x, y2=y*y;
    const double y12=(y-1)*(y-1);

    unsigned elem=cd.elem_index();

    const T sdx0=this->derivs_[cd.node0_index()][0]*this->scalefactors_[elem][0];
    const T sdx1=this->derivs_[cd.node1_index()][0]*this->scalefactors_[elem][0];
    const T sdx2=this->derivs_[cd.node2_index()][0]*this->scalefactors_[elem][0];

    const T sdy0=this->derivs_[cd.node0_index()][1]*this->scalefactors_[elem][1];
    const T sdy1=this->derivs_[cd.node1_index()][1]*this->scalefactors_[elem][1];
    const T sdy2=this->derivs_[cd.node2_index()][1]*this->scalefactors_[elem][1];

    const T sdxy0=this->derivs_[cd.node0_index()][2]*this->scalefactors_[elem][0]*this->scalefactors_[elem][1];
    const T sdxy1=this->derivs_[cd.node1_index()][2]*this->scalefactors_[elem][0]*this->scalefactors_[elem][1];
    const T sdxy2=this->derivs_[cd.node2_index()][2]*this->scalefactors_[elem][0]*this->scalefactors_[elem][1];

    derivs.resize(2);

    derivs[0]=T(6*(-1 + x)*x*cd.node0()
		+(-4*x + 3*x2 + y12*(1 + 2*y))*sdx0
		+6*(-1 + x)*x*y*sdy0
		+(-4*x + 3*x2 + y12)*y*sdxy0
		-6*(-1 + x)*x*cd.node1()
		+x*(-2 + 3*x)*sdx1
		-6*(-1 + x)*x*y*sdy1
		+x*(-2 + 3*x)*y*sdxy1
		+(3 - 2*y)*y2*sdx2
		+(-1 + y)*y2*sdxy2);

    derivs[1]=T(6*(-1 + y)*y*cd.node0()
		+6*x*(-1 + y)*y*sdx0
		+(1 - 3*x2 + 2*x3 - 4*y + 3*y2)*sdy0
		+x*(1 - 2*x + x2 - 4*y + 3*y2)*sdxy0
		+(3 - 2*x)*x2*sdy1
		+(-1 + x)*x2*sdxy1
		-6*(-1 + y)*y*cd.node2()
		-6*x*(-1 + y)*y*sdx2
		+y*(-2 + 3*y)*sdy2 
		+x*y*(-2 + 3*y)*sdxy2);
  }
  
  //! get the parametric coordinate for value within the element.
  template <class ElemData>
  bool get_coords(std::vector<double> &coords, const T& value, 
		  const ElemData &cd) const  
  {
    TriLocate< TriCubicHmtScaleFactors<T> > CL;
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
    return get_area2<TriGaussian3<double> >(this, face, cd);
  }
 
  //! get volume
  template <class ElemData>
    double get_volume(const ElemData & /* cd */) const  
  {
    return 0.;
  }

  static const std::string type_name(int n = -1);

  virtual void io (Piostream& str); 
};


template <class T>
const TypeDescription* get_type_description(TriCubicHmtScaleFactors<T> *)
{
  static TypeDescription* td = 0;
  if(!td){
    const TypeDescription *sub = get_type_description((T*)0);
    TypeDescription::td_vec *subs = scinew TypeDescription::td_vec(1);
    (*subs)[0] = sub;
    td = scinew TypeDescription("TriCubicHmtScaleFactors", subs, 
				std::string(__FILE__),
				"SCIRun", 
				TypeDescription::BASIS_E);
  }
  return td;
}

template <class T>
const std::string
TriCubicHmtScaleFactors<T>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const std::string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;
  }
  else if (n == 0)
  {
    static const std::string nm("TriCubicHmtScaleFactors");
    return nm;
  } else {
    return find_type_name((T *)0);
  }
}


const int TRICUBICHMTSCALEFACTORS_VERSION = 1;
template <class T>
void
  TriCubicHmtScaleFactors<T>::io(Piostream &stream)
{
  stream.begin_class(get_type_description(this)->get_name(),
                     TRICUBICHMTSCALEFACTORS_VERSION);
  Pio(stream, this->derivs_);
  Pio(stream, this->scalefactors_);
  stream.end_class();
}

} //namespace SCIRun


#endif // TriCubicHmtScaleFactors_h
