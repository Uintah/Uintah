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
//    File   : PrismCubicHmt.h
//    Author : Martin Cole, Frank B. Sachse
//    Date   : Dec 1 2004

#if !defined(PrismCubicHmt_h)
#define PrismCubicHmt_h

#include <Core/Basis/PrismLinearLgn.h>

namespace SCIRun {

//! Class for describing unit geometry of PrismCubicHmt
class PrismCubicHmtUnitElement : public PrismLinearLgnUnitElement {
public:
  PrismCubicHmtUnitElement() {};
  virtual ~PrismCubicHmtUnitElement() {};
};


//! Class for handling of element of type prism with 
//! cubic hermitian interpolation
template <class T>
class PrismCubicHmt : public PrismApprox, 
		      public PrismGaussian2<T>, 
		      public PrismCubicHmtUnitElement 
{
public:
  typedef T value_type;

  PrismCubicHmt() {}
  virtual ~PrismCubicHmt() {}

  int polynomial_order() const { return 3; }

  //! get weight factors at parametric coordinate 
  inline
  int get_weights(const vector<double> &coords, double *w) const
  {
    const double x=coords[0], y=coords[1], z=coords[2];  
    const double x2=x*x;
    const double y2=y*y;
    const double z2=z*z;
    const double x12=(x-1)*(x-1);
    const double y12=(y-1)*(y-1);
    const double z12=(z-1)*(z-1);
      
    w[0]  = -((-1 + x + y)*(-1 + z)*(-1 + 2*x2 - y + 2*y2 - x*(1 + 2*y) - z + 2*z2));
    w[1]  = -(x*(1 - 2*x + x2 - y2)*(-1 + z));
    w[2]  = +(x2 - y12)*y*(-1 + z);
    w[3]  = -((-1 + x + y)*z12*z);
    w[4]  = +x*(-1 + z)*(-3*x + 2*x2 + z*(-1 + 2*z));
    w[5]  = +x2*(-1 + x + z - x*z);
    w[6]  = -(x2*y*(-1 + z));
    w[7]  = +x*z12*z;
    w[8]  = +y*(-1 + z)*(-3*y + 2*y2 + z*(-1 + 2*z));
    w[9]  = -(x*y2*(-1 + z));
    w[10] = +y2*(-1 + y + z - y*z);
    w[11] = +y*z12*z;
    w[12] = +(-1 + x + y)*z*(2*x2 - y + 2*y2 - x*(1 + 2*y) + z*(-3 + 2*z));
    w[13] = +x*(1 - 2*x + x2 - y2)*z;
    w[14] = +(-x2 + y12)*y*z;
    w[15] = -((-1 + x + y)*(-1 + z)*z2);
    w[16] = +x*z*(-1 + 3*x - 2*x2 + 3*z - 2*z2);
    w[17] = +(-1 + x)*x2*z;
    w[18] = +x2*y*z;
    w[19] = x*(-1 + z)*z2;
    w[20] = +y*z*(-1 + 3*y - 2*y2 + 3*z - 2*z2);
    w[21] = +x*y2*z;
    w[22] = +(-1 + y)*y2*z;
    w[23] = +y*(-1 + z)*z2;

    return 24;
  }
  //! get value at parametric coordinate 
  template <class ElemData>
  T interpolate(const vector<double> &coords, const ElemData &cd) const
  {
    double w[24];
    get_weights(coords, w); 
    
    return (T)(w[0]  * cd.node0()                   +
	       w[1]  * derivs_[cd.node0_index()][0] +
	       w[2]  * derivs_[cd.node0_index()][1] +
	       w[3]  * derivs_[cd.node0_index()][2] +
	       w[4]  * cd.node1()		    +
	       w[5]  * derivs_[cd.node1_index()][0] +
	       w[6]  * derivs_[cd.node1_index()][1] +
	       w[7]  * derivs_[cd.node1_index()][2] +
	       w[8]  * cd.node2()		    +
	       w[9]  * derivs_[cd.node2_index()][0] +
	       w[10] * derivs_[cd.node2_index()][1] +
	       w[11] * derivs_[cd.node2_index()][2] +
	       w[12] * cd.node3()		    +
	       w[13] * derivs_[cd.node3_index()][0] +
	       w[14] * derivs_[cd.node3_index()][1] +
	       w[15] * derivs_[cd.node3_index()][2] +
	       w[16] * cd.node4()		    +
	       w[17] * derivs_[cd.node4_index()][0] +
	       w[18] * derivs_[cd.node4_index()][1] +
	       w[19] * derivs_[cd.node4_index()][2] +
	       w[20] * cd.node5()		    +
	       w[21] * derivs_[cd.node5_index()][0] +
	       w[22] * derivs_[cd.node5_index()][1] +
	       w[23] * derivs_[cd.node5_index()][2]);
  };
  
  //! get first derivative at parametric coordinate
  template <class ElemData>
  void derivate(const vector<double> &coords, const ElemData &cd, 
		vector<T> &derivs) const
  {
    const double x=coords[0], y=coords[1], z=coords[2];  
    const double x2=x*x;
    const double y2=y*y;
    const double z2=z*z;
    const double x12=(x-1)*(x-1);
    const double y12=(y-1)*(y-1);
    const double z12=(z-1)*(z-1);

    derivs.resize(3);

    derivs[0]=
      T(-((-1 + z)*(-6*x + 6*x2 + z*(-1 + 2*z)))*cd.node0()
	-((1 - 4*x + 3*x2 - y2)*(-1 + z))*derivs_[cd.node0_index()][0]
	+2*x*y*(-1 + z)*derivs_[cd.node0_index()][1]
	-(z12*z)*derivs_[cd.node0_index()][2]
	+(-1 + z)*(-6*x + 6*x2 + z*(-1 + 2*z))*cd.node1()
	-(x*(-2 + 3*x)*(-1 + z))*derivs_[cd.node1_index()][0]
	-2*x*y*(-1 + z)*derivs_[cd.node1_index()][1]
	+z12*z*derivs_[cd.node1_index()][2]
	-(y2*(-1 + z))*derivs_[cd.node2_index()][0]
	+z*(1- 6*x + 6*x2 - 3*z + 2*z2)*cd.node3()
	+(1 - 4*x + 3*x2 - y2)*z*derivs_[cd.node3_index()][0]
	-2*x*y*z*derivs_[cd.node3_index()][1]
	-((-1 + z)*z2)*derivs_[cd.node3_index()][2]
	+z*(-1 + 6*x - 6*x2 + 3*z - 2*z2)*cd.node4()
	+x*(-2 + 3*x)*z*derivs_[cd.node4_index()][0]
	2*x*y*z*derivs_[cd.node4_index()][1]
	+(-1 + z)*z2*derivs_[cd.node4_index()][2]
	+y2*z*derivs_[cd.node5_index()][0]);

    derivs[1]=
      T(-((-1 + z)*(-6*y + 6*y2 + z*(-1 + 2*z)))*cd.node0()
	+2*x*y*(-1 + z)*derivs_[cd.node0_index()][0]
	+(-1 + x2 + 4*y - 3*y2)*(-1 + z)*derivs_[cd.node0_index()][1]
	-(z12*z)*derivs_[cd.node0_index()][2]
	-(x2*(-1 + z))*derivs_[cd.node1_index()][1]
	+(-1 + z)*(-6*y + 6*y2 + z*(-1 + 2*z))*cd.node2()
	-2*x*y*(-1 + z)*derivs_[cd.node2_index()][0]
	-(y*(-2 + 3*y)*(-1 + z))*derivs_[cd.node2_index()][1]
	+z12*z*derivs_[cd.node2_index()][2]
	+z*(1 - 6*y + 6*y2 - 3*z + 2*z2)*cd.node3()
	-2*x*y*z*derivs_[cd.node3_index()][0]
	-((-1 + x2 + 4*y - 3*y2)*z)*derivs_[cd.node3_index()][1]
	-((-1 + z)*z2)*derivs_[cd.node3_index()][2]
	+x2*z*derivs_[cd.node4_index()][1]
	+z*(-1 + 6*y - 6*y2 + 3*z  2*z2)*cd.node5()
	+2*x*y*z*derivs_[cd.node5_index()][0]
	+y*(-2 + 3*y)*z*derivs_[cd.node5_index()][1]
	+(-1 + z)*z2*derivs_[cd.node5_index()][2]);

    derivs[2]=
      T(-((-1 + x + y)*(2*x2 - y + 2*y2 - x*(1 + 2*y) + 6*(-1 + z)*z))*cd.node0()
	+x*(-1 + 2*x - x2 + y2)*derivs_[cd.node0_index()][0]
	+(x2 - y12)*y*derivs_[cd.node0_index()][1]
	-((-1 + x + y)*(1 - 4*z + 3*z2))*derivs_[cd.node0_index()][2]
	+x*(1 - 3*x + 2*x2 - 6*z + 6*z2)*cd.node1()
	-((-1 + x)*x2)*derivs_[cd.node1_index()][0]
	-(x2*y)*derivs_[cd.node1_index()][0]
	+x*(1 - 4*z + 3*z2)*derivs_[cd.node1_index()][0]
	+y*(1 - 3*y + 2*y2 - 6*z + 6*z2)*cd.node2()
	-(x*y2)*derivs_[cd.node2_index()][0]
	-((-1 + y)*y2)*derivs_[cd.node2_index()][1]
	+y*(1 - 4*z + 3*z2)*derivs_[cd.node2_index()][2]
	+(-1 + x + y)*(2*x2 - y + 2*y2 - x*(1 + 2*y) + 6*(-1 + z)*z)*cd.node3()
	+x*(1 - 2*x + x2 - y2)*derivs_[cd.node3_index()][0]
	+(-x2 + y12)*y*derivs_[cd.node3_index()][1]
	-((-1 + x + y)*z*(-2 + 3*z))*derivs_[cd.node3_index()][2]
	+x*(-1 + 3*x - 2*x2 + 6*z - 6*z2)*cd.node4()
	(-1 + x)*x2*derivs_[cd.node4_index()][0]
	+x2*y*derivs_[cd.node4_index()][1]
	+x*z*(-2 + 3*z)*derivs_[cd.node4_index()][2]
	+y*(-1 + 3*y - 2*y2 + 6*z - 6*z2)*cd.node5()
	+x*y2*derivs_[cd.node5_index()][0]
	+(-1 + y)*y2*derivs_[cd.node5_index()][1]
	+y*z*(-2 + 3*z)*derivs_[cd.node5_index()][2]);
  };  

  //! get parametric coordinate for value within the element
  //! iterative solution...
  template <class ElemData>
  bool get_coords(vector<double> &coords, const T& value, 
		  const ElemData &cd) const;  
  {
    PrismLocate< PrismCubicHmt<T> > CL;
    return CL.get_coords(this, coords, value, cd);
  }  
 
  //! add derivative values (dx, dy, dz) for nodes.
  void add_derivative(const T &p[3]) { derivs_.push_back(p); }

  static  const string type_name(int n = -1);

  virtual void io (Piostream& str);

protected:
  //! support data (node data is elsewhere)
  vector<T[3]>          derivs_; 
};


template <class T>
const TypeDescription* get_type_description(PrismCubicHmt<T> *)
{
  static TypeDescription* td = 0;
  if(!td){
    const TypeDescription *sub = get_type_description((T*)0);
    TypeDescription::td_vec *subs = scinew TypeDescription::td_vec(1);
    (*subs)[0] = sub;
    td = scinew TypeDescription(PrismCubicHmt<T>::type_name(0), subs, 
				string(__FILE__),
				"SCIRun", 
				TypeDescription::BASIS_E);
  }
  return td;
}

template <class T>
const string
PrismCubicHmt<T>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;
  }
  else if (n == 0)
  {
    static const string nm("PrismCubicHmt");
    return nm;
  } else {
    return find_type_name((T *)0);
  }
}


const int PRISMCUBICHMT_VERSION = 1;
template <class T>
void
PrismCubicHmt<T>::io(Piostream &stream)
{
  stream.begin_class(type_name(-1), PRISMCUBICHMT_VERSION);
  Pio(stream, derivs_);
  stream.end_class();
}

} //namespace SCIRun

#endif // PrismCubicHmt_h
