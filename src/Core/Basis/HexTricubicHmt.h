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
//    File   : HexTricubicHmt.h
//    Author : Martin Cole, Frank B. Sachse
//    Date   : Dec 3 2004

#if !defined(HexTricubicHmt_h)
#define HexTricubicHmt_h

#include <Core/Basis/HexTrilinearLgn.h>

namespace SCIRun {

//! Class for describing unit geometry of HexTricubicHmt
class HexTricubicHmtUnitElement : public HexTrilinearLgnUnitElement {
public:
  HexTricubicHmtUnitElement() {};
  virtual ~HexTricubicHmtUnitElement() {};

  static int dofs() { return 64; } //!< return degrees of freedom
};


//! Class for handling of element of type hexahedron with 
//! tricubic hermitian interpolation
template <class T>
class HexTricubicHmt : public BasisAddDerivatives<T>, 
                       public HexApprox, 
		       public HexGaussian3<double>, 
		       public HexTricubicHmtUnitElement
{
public:
  typedef T value_type;

  HexTricubicHmt() {}
  virtual ~HexTricubicHmt() {}
  
  static int polynomial_order() { return 3; }

  //! get weight factors at parametric coordinate 
  inline
  static void get_weights(const std::vector<double> &coords, double *w) 
  {    
    const double x=coords[0], y=coords[1], z=coords[2];  
    const double x2=x*x;
    const double y2=y*y;
    const double z2=z*z;
    const double x12=(x-1)*(x-1);
    const double y12=(y-1)*(y-1);
    const double z12=(z-1)*(z-1);

    w[0]  = x12*(1 + 2*x)*y12*(1 + 2*y)*z12*(1 + 2*z);
    w[1]  = +x12*x*y12*(1 + 2*y)*z12*(1 + 2*z);
    w[2]  = +x12*(1 + 2*x)*y12*y*z12*(1 + 2*z);
    w[3]  = +x12*(1 + 2*x)*y12*(1 + 2*y)*z12*z;
    w[4]  = +x12*x*y12*y*z12*(1 + 2*z);
    w[5]  = +x12*(1 + 2*x)*y12*y*z12*z;
    w[6]  = +x12*x*y12*(1 + 2*y)*z12*z;
    w[7]  = +x12*x*y12*y*z12*z;
    w[8]  = -(x2*(-3 + 2*x)*y12*(1 + 2*y)*z12*(1 + 2*z));
    w[9]  = +(-1 + x)*x2*y12*(1 + 2*y)*z12*(1 + 2*z);
    w[10] = -(x2*(-3 + 2*x)*y12*y*z12*(1 + 2*z));
    w[11] = -(x2*(-3 + 2*x)*y12*(1 + 2*y)*z12*z);
    w[12] = +(-1 + x)*x2*y12*y*z12*(1 + 2*z);
    w[13] = -(x2*(-3 + 2*x)*y12*y*z12*z) ;
    w[14] = +(-1 + x)*x2*y12*(1 + 2*y)*z12*z;
    w[15] = +(-1 + x)*x2*y12*y*z12*z;
    w[16] = +x2*(-3 + 2*x)*y2*(-3 + 2*y)*z12*(1 + 2*z);
    w[17] = -((-1 + x)*x2*y2*(-3 + 2*y)*z12*(1 + 2*z));
    w[18] = -(x2*(-3 + 2*x)*(-1 + y)*y2*z12*(1 + 2*z));
    w[19] = +x2*(-3 + 2*x)*y2*(-3 + 2*y)*z12*z;
    w[20] = +(-1 + x)*x2*(-1 + y)*y2*z12*(1 + 2*z);
    w[21] = -(x2*(-3 + 2*x)*(-1 + y)*y2*z12*z);
    w[22] = -((-1 + x)*x2*y2*(-3 + 2*y)*z12*z);
    w[23] = +(-1 + x)*x2*(-1 + y)*y2*z12*z;
    w[24] = -(x12*(1 + 2*x)*y2*(-3 + 2*y)*z12*(1 + 2*z));
    w[25] = -(x12*x*y2*(-3 + 2*y)*z12*(1 + 2*z));
    w[26] = +x12*(1 + 2*x)*(-1 + y)*y2*z12*(1 + 2*z);
    w[27] = -(x12*(1 + 2*x)*y2*(-3 + 2*y)*z12*z);
    w[28] = +x12*x*(-1 + y)*y2*z12*(1 + 2*z);
    w[29] = +x12*(1 + 2*x)*(-1 + y)*y2*z12*z;
    w[30] = -(x12*x*y2*(-3 + 2*y)*z12*z);
    w[31] = +x12*x*(-1 + y)*y2*z12*z;
    w[32] = -(x12*(1 + 2*x)*y12*(1 + 2*y)*z2*(-3 + 2*z));
    w[33] = -(x12*x*y12*(1 + 2*y)*z2*(-3 + 2*z));
    w[34] = -(x12*(1 + 2*x)*y12*y*z2*(-3 + 2*z));
    w[35] = +x12*(1 + 2*x)*y12*(1 + 2*y)*(-1 + z)*z2;
    w[36] = -(x12*x*y12*y*z2*(-3 + 2*z));
    w[37] = +x12*(1 + 2*x)*y12*y*(-1 + z)*z2;
    w[38] = +x12*x*y12*(1 + 2*y)*(-1 + z)*z2;
    w[39] = +x12*x*y12*y*(-1 + z)*z2;
    w[40] = +x2*(-3 + 2*x)*y12*(1 + 2*y)*z2*(-3 + 2*z);
    w[41] = -((-1 + x)*x2*y12*(1 + 2*y)*z2*(-3 + 2*z));
    w[42] = +x2*(-3 + 2*x)*y12*y*z2*(-3 + 2*z);
    w[43] = -(x2*(-3 + 2*x)*y12*(1 + 2*y)*(-1 + z)*z2);
    w[44] = -((-1 + x)*x2*y12*y*z2*(-3 + 2*z));
    w[45] = -(x2*(-3 + 2*x)*y12*y*(-1 + z)*z2);
    w[46] = +(-1 + x)*x2*y12*(1 + 2*y)*(-1 + z)*z2;
    w[47] = +(-1 + x)*x2*y12*y*(-1 + z)*z2;
    w[48] = -(x2*(-3 + 2*x)*y2*(-3 + 2*y)*z2*(-3 + 2*z));
    w[49] = +(-1 + x)*x2*y2*(-3 + 2*y)*z2*(-3 + 2*z);
    w[50] = +x2*(-3 + 2*x)*(-1 + y)*y2*z2*(-3 + 2*z);
    w[51] = +x2*(-3 + 2*x)*y2*(-3 + 2*y)*(-1 + z)*z2;
    w[52] = -((-1 + x)*x2*(-1 + y)*y2*z2*(-3 + 2*z));
    w[53] = -(x2*(-3 + 2*x)*(-1 + y)*y2*(-1 + z)*z2);
    w[54] = -((-1 + x)*x2*y2*(-3 + 2*y)*(-1 + z)*z2);
    w[55] = +(-1 + x)*x2*(-1 + y)*y2*(-1 + z)*z2;
    w[56] = +x12*(1 + 2*x)*y2*(-3 + 2*y)*z2*(-3 + 2*z);
    w[57] = +x12*x*y2*(-3 + 2*y)*z2*(-3 + 2*z);
    w[58] = -(x12*(1 + 2*x)*(-1 + y)*y2*z2*(-3 + 2*z));
    w[59] = -(x12*(1 + 2*x)*y2*(-3 + 2*y)*(-1 + z)*z2);
    w[60] = -(x12*x*(-1 + y)*y2*z2*(-3 + 2*z));
    w[61] = +x12*(1 + 2*x)*(-1 + y)*y2*(-1 + z)*z2;
    w[62] = -(x12*x*y2*(-3 + 2*y)*(-1 + z)*z2);
    w[63] = +x12*x*(-1 + y)*y2*(-1 + z)*z2;
  }

  //! get value at parametric coordinate 
  template <class ElemData>
  T interpolate(const std::vector<double> &coords, const ElemData &cd) const
  {
    double w[64];
    get_weights(coords, w); 
    return (T)(w[0]  * cd.node0() +
	       w[1]  * this->derivs_[cd.node0_index()][0] +
	       w[2]  * this->derivs_[cd.node0_index()][1] +
	       w[3]  * this->derivs_[cd.node0_index()][2] +
	       w[4]  * this->derivs_[cd.node0_index()][3] +
	       w[5]  * this->derivs_[cd.node0_index()][4] +
	       w[6]  * this->derivs_[cd.node0_index()][5] +
	       w[7]  * this->derivs_[cd.node0_index()][6] +
	       w[8]  * cd.node1()		    +
	       w[9]  * this->derivs_[cd.node1_index()][0] +
	       w[10] * this->derivs_[cd.node1_index()][1] +
	       w[11] * this->derivs_[cd.node1_index()][2] +
	       w[12] * this->derivs_[cd.node1_index()][3] +
	       w[13] * this->derivs_[cd.node1_index()][4] +  
	       w[14] * this->derivs_[cd.node1_index()][5] +
	       w[15] * this->derivs_[cd.node1_index()][6] +
	       w[16] * cd.node2()		    +
	       w[17] * this->derivs_[cd.node2_index()][0] +
	       w[18] * this->derivs_[cd.node2_index()][1] +
	       w[19] * this->derivs_[cd.node2_index()][2] +
	       w[20] * this->derivs_[cd.node2_index()][3] +
	       w[21] * this->derivs_[cd.node2_index()][4] +
	       w[22] * this->derivs_[cd.node2_index()][5] +
	       w[23] * this->derivs_[cd.node2_index()][6] +
	       w[24] * cd.node3()		    +
	       w[25] * this->derivs_[cd.node3_index()][0] +
	       w[26] * this->derivs_[cd.node3_index()][1] +
	       w[27] * this->derivs_[cd.node3_index()][2] +
	       w[28] * this->derivs_[cd.node3_index()][3] +
	       w[29] * this->derivs_[cd.node3_index()][4] +
	       w[30] * this->derivs_[cd.node3_index()][5] +
	       w[31] * this->derivs_[cd.node3_index()][6] +
	       w[32] * cd.node4()		    +
	       w[33] * this->derivs_[cd.node4_index()][0] +
	       w[34] * this->derivs_[cd.node4_index()][1] +
	       w[35] * this->derivs_[cd.node4_index()][2] +
	       w[36] * this->derivs_[cd.node4_index()][3] +
	       w[37] * this->derivs_[cd.node4_index()][4] +
	       w[38] * this->derivs_[cd.node4_index()][5] +
	       w[39] * this->derivs_[cd.node4_index()][6] +
	       w[40] * cd.node5()		    +
	       w[41] * this->derivs_[cd.node5_index()][0] +
	       w[42] * this->derivs_[cd.node5_index()][1] +
	       w[43] * this->derivs_[cd.node5_index()][2] +
	       w[44] * this->derivs_[cd.node5_index()][3] +
	       w[45] * this->derivs_[cd.node5_index()][4] +
	       w[46] * this->derivs_[cd.node5_index()][5] +
	       w[47] * this->derivs_[cd.node5_index()][6] +
	       w[48] * cd.node6()		    +
	       w[49] * this->derivs_[cd.node6_index()][0] +
	       w[50] * this->derivs_[cd.node6_index()][1] +
	       w[51] * this->derivs_[cd.node6_index()][2] +
	       w[52] * this->derivs_[cd.node6_index()][3] +
	       w[53] * this->derivs_[cd.node6_index()][4] +
	       w[54] * this->derivs_[cd.node6_index()][5] +
	       w[55] * this->derivs_[cd.node6_index()][6] +
	       w[56] * cd.node7()		    +
	       w[57] * this->derivs_[cd.node7_index()][0] +
	       w[58] * this->derivs_[cd.node7_index()][1] +
	       w[59] * this->derivs_[cd.node7_index()][2] +
	       w[60] * this->derivs_[cd.node7_index()][3] +
	       w[61] * this->derivs_[cd.node7_index()][4] +
	       w[62] * this->derivs_[cd.node7_index()][5] +
	       w[63] * this->derivs_[cd.node7_index()][6]);
  }
  
  //! get derivative weight factors at parametric coordinate 
  inline
  static void get_derivate_weights(const std::vector<double> &coords, double *w) 
  {
    const double x=coords[0], y=coords[1], z=coords[2];  
    const double x2=x*x;
    const double y2=y*y;
    const double z2=z*z;
    const double x12=(x-1)*(x-1);
    const double y12=(y-1)*(y-1);
    const double z12=(z-1)*(z-1);

    w[0]=6*(-1 + x)*x*y12*(1 + 2*y)*z12*(1 + 2*z);
    w[1]=+(1 - 4*x + 3*x2)*y12*(1 + 2*y)*z12*(1 + 2*z);
    w[2]=+6*(-1 + x)*x*y12*y*z12*(1 + 2*z);
    w[3]=+6*(-1 + x)*x*y12*(1 + 2*y)*z12*z;
    w[4]=+(1 - 4*x + 3*x2)*y12*y*z12*(1 + 2*z);
    w[5]=+6*(-1 + x)*x*y12*y*z12*z;
    w[6]=+(1 - 4*x + 3*x2)*y12*(1 + 2*y)*z12*z;
    w[7]=+(1 - 4*x + 3*x2)*y12*y*z12*z;
    w[8]=-6*(-1 + x)*x*y12*(1 + 2*y)*z12*(1 + 2*z);
    w[9]=+x*(-2 + 3*x)*y12*(1 + 2*y)*z12*(1 + 2*z);
    w[10]=-6*(-1 + x)*x*y12*y*z12*(1 + 2*z);
    w[11]=-6*(-1 + x)*x*y12*(1 + 2*y)*z12*z;
    w[12]=+x*(-2 + 3*x)*y12*y*z12*(1 + 2*z);
    w[13]=-6*(-1 + x)*x*y12*y*z12*z;
    w[14]=+x*(-2 + 3*x)*y12*(1 + 2*y)*z12*z;
    w[15]=+x*(-2 + 3*x)*y12*y*z12*z;
    w[16]=+6*(-1 + x)*x*y2*(-3 + 2*y)*z12*(1 + 2*z);
    w[17]=-(x*(-2 + 3*x)*y2*(-3 + 2*y)*z12*(1 + 2*z));
    w[18]=-6*(-1 + x)*x*(-1 + y)*y2*z12*(1 + 2*z);
    w[19]=+6*(-1 + x)*x*y2*(-3 + 2*y)*z12*z;
    w[20]=+x*(-2 + 3*x)*(-1 + y)*y2*z12*(1 + 2*z);
    w[21]=-6*(-1 + x)*x*(-1 + y)*y2*z12*z;
    w[22]=-(x*(-2 + 3*x)*y2*(-3 + 2*y)*z12*z);
    w[23]=+x*(-2 + 3*x)*(-1 + y)*y2*z12*z;
    w[24]=-6*(-1 + x)*x*y2*(-3 + 2*y)*z12*(1 + 2*z);
    w[25]=-((1 - 4*x + 3*x2)*y2*(-3 + 2*y)*z12*(1 + 2*z));
    w[26]=+6*(-1 + x)*x*(-1 + y)*y2*z12*(1 + 2*z);
    w[27]=-6*(-1 + x)*x*y2*(-3 + 2*y)*z12*z;
    w[28]=+(1 - 4*x + 3*x2)*(-1 + y)*y2*z12*(1 + 2*z);
    w[29]=+6*(-1 + x)*x*(-1 + y)*y2*z12*z;
    w[30]=-((1 - 4*x + 3*x2)*y2*(-3 + 2*y)*z12*z);
    w[31]=+(1 - 4*x + 3*x2)*(-1 + y)*y2*z12*z;
    w[32]=-6*(-1 + x)*x*y12*(1 + 2*y)*z2*(-3 + 2*z);
    w[33]=-((1 - 4*x + 3*x2)*y12*(1 + 2*y)*z2*(-3 + 2*z));
    w[34]=-6*(-1 + x)*x*y12*y*z2*(-3 + 2*z);
    w[35]=+6*(-1 + x)*x*y12*(1 + 2*y)*(-1 + z)*z2;
    w[36]=-((1 - 4*x + 3*x2)*y12*y*z2*(-3 + 2*z));
    w[37]=+6*(-1 + x)*x*y12*y*(-1 + z)*z2;
    w[38]=+(1 - 4*x + 3*x2)*y12*(1 + 2*y)*(-1 + z)*z2;
    w[39]=+(1 - 4*x + 3*x2)*y12*y*(-1 + z)*z2;
    w[40]=+6*(-1 + x)*x*y12*(1 + 2*y)*z2*(-3 + 2*z);
    w[41]=-(x*(-2 + 3*x)*y12*(1 + 2*y)*z2*(-3 + 2*z));
    w[42]=+6*(-1 + x)*x*y12*y*z2*(-3 + 2*z);
    w[43]=-6*(-1 + x)*x*y12*(1 + 2*y)*(-1 + z)*z2;
    w[44]=-(x*(-2 + 3*x)*y12*y*z2*(-3 + 2*z));
    w[45]=-6*(-1 + x)*x*y12*y*(-1 + z)*z2;
    w[46]=+x*(-2 + 3*x)*y12*(1 + 2*y)*(-1 + z)*z2;
    w[47]=+x*(-2 + 3*x)*y12*y*(-1 + z)*z2;
    w[48]=-6*(-1 + x)*x*y2*(-3 + 2*y)*z2*(-3 + 2*z);
    w[49]=+x*(-2 + 3*x)*y2*(-3 + 2*y)*z2*(-3 + 2*z);
    w[50]=+6*(-1 + x)*x*(-1 + y)*y2*z2*(-3 + 2*z);
    w[51]=+6*(-1 + x)*x*y2*(-3 + 2*y)*(-1 + z)*z2;
    w[52]=-(x*(-2 + 3*x)*(-1 + y)*y2*z2*(-3 + 2*z));
    w[53]=-6*(-1 + x)*x*(-1 + y)*y2*(-1 + z)*z2;
    w[54]=-(x*(-2 + 3*x)*y2*(-3 + 2*y)*(-1 + z)*z2);
    w[55]=+x*(-2 + 3*x)*(-1 + y)*y2*(-1 + z)*z2;
    w[56]=+6*(-1 + x)*x*y2*(-3 +  2*y)*z2*(-3 + 2*z);
    w[57]=+(1 - 4*x + 3*x2)*y2*(-3 + 2*y)*z2*(-3 + 2*z);
    w[58]=-6*(-1 + x)*x*(-1 + y)*y2*z2*(-3 + 2*z);
    w[59]=-6*(-1 + x)*x*y2*(-3 + 2*y)*(-1 + z)*z2;
    w[60]=-((1 - 4*x + 3*x2)*(-1 + y)*y2*z2*(-3 + 2*z));
    w[61]=+6*(-1 + x)*x*(-1 + y)*y2*(-1 + z)*z2;
    w[62]=-((1 - 4*x + 3*x2)*y2*(-3 + 2*y)*(-1 + z)*z2);
    w[63]=+(1 - 4*x + 3*x2)*(-1 + y)*y2*(-1 + z)*z2;
      
    w[64]=6*x12*(1 + 2*x)*(-1 + y)*y*z12*(1 + 2*z);
    w[65]=+6*x12*x*(-1 + y)*y*z12*(1 + 2*z);
    w[66]=+x12*(1 + 2*x)*(1 - 4*y + 3*y2)*z12*(1 + 2*z);
    w[67]=+6*x12*(1 + 2*x)*(-1 + y)*y*z12*z;
    w[68]=+x12*x*(1 - 4*y + 3*y2)*z12*(1 + 2*z);
    w[69]=+x12*(1 + 2*x)*(1 - 4*y + 3*y2)*z12*z;
    w[70]=+6*x12*x*(-1 + y)*y*z12*z;
    w[71]=+x12*x*(1 - 4*y + 3*y2)*z12*z;
    w[72]=-6*x2*(-3 + 2*x)*(-1 + y)*y*z12*(1 + 2*z);
    w[73]=+6*(-1 + x)*x2*(-1 + y)*y*z12*(1 + 2*z);
    w[74]=-(x2*(-3 + 2*x)*(1 - 4*y + 3*y2)*z12*(1 + 2*z));
    w[75]=-6*x2*(-3 + 2*x)*(-1 + y)*y*z12*z;
    w[76]=+(-1 + x)*x2*(1 - 4*y + 3*y2)*z12*(1 + 2*z);
    w[77]=-(x2*(-3 + 2*x)*(1 - 4*y + 3*y2)*z12*z);
    w[78]=+6*(-1 + x)*x2*(-1 + y)*y*z12*z;
    w[79]=+(-1 + x)*x2*(1 - 4*y + 3*y2)*z12*z;
    w[80]=+6*x2*(-3 + 2*x)*(-1 + y)*y*z12*(1 + 2*z);
    w[81]=-6*(-1 + x)*x2*(-1 + y)*y*z12*(1 + 2*z);
    w[82]=-(x2*(-3 + 2*x)*y*(-2 + 3*y)*z12*(1 + 2*z));
    w[83]=+6*x2*(-3 + 2*x)*(-1 + y)*y*z12*z;
    w[84]=+(-1 + x)*x2*y*(-2 + 3*y)*z12*(1 + 2*z);
    w[85]=-(x2*(-3 + 2*x)*y*(-2 + 3*y)*z12*z);
    w[86]=-6*(-1 + x)*x2*(-1 + y)*y*z12*z;
    w[87]=+(-1 + x)*x2*y*(-2 + 3*y)*z12*z;
    w[88]=-6*x12*(1 + 2*x)*(-1 + y)*y*z12*(1 + 2*z);
    w[89]=-6*x12*x*(-1 + y)*y*z12*(1 + 2*z);
    w[90]=+x12*(1 + 2*x)*y*(-2 + 3*y)*z12*(1 + 2*z);
    w[91]=-6*x12*(1 + 2*x)*(-1 + y)*y*z12*z;
    w[92]=+x12*x*y*(-2 + 3*y)*z12*(1 + 2*z);
    w[93]=+x12*(1 + 2*x)*y*(-2 + 3*y)*z12*z;
    w[94]=-6*x12*x*(-1 + y)*y*z12*z;
    w[95]=+x12*x*y*(-2 + 3*y)*z12*z;
    w[96]=-6*x12*(1 + 2*x)*(-1 + y)*y*z2*(-3 + 2*z);
    w[97]=-6*x12*x*(-1 + y)*y*z2*(-3 + 2*z);
    w[98]=-(x12*(1 + 2*x)*(1 - 4*y + 3*y2)*z2*(-3 + 2*z));
    w[99]=+6*x12*(1 + 2*x)*(-1 + y)*y*(-1 + z)*z2;
    w[100]=-(x12*x*(1 - 4*y + 3*y2)*z2*(-3 + 2*z));
    w[101]=+x12*(1 + 2*x)*(1 - 4*y + 3*y2)*(-1 + z)*z2;
    w[102]=+6*x12*x*(-1 + y)*y*(-1 + z)*z2;
    w[103]=+x12*x*(1 - 4*y + 3*y2)*(-1 + z)*z2;
    w[104]=+6*x2*(-3 + 2*x)*(-1 + y)*y*z2*(-3 + 2*z);
    w[105]=-6*(-1 + x)*x2*(-1 + y)*y*z2*(-3 + 2*z);
    w[106]=+x2*(-3 + 2*x)*(1 - 4*y + 3*y2)*z2*(-3 + 2*z);
    w[107]=-6*x2*(-3 + 2*x)*(-1 + y)*y*(-1 + z)*z2;
    w[108]=-((-1 + x)*x2*(1 - 4*y + 3*y2)*z2*(-3 + 2*z));
    w[109]=-(x2*(-3 + 2*x)*(1 - 4*y + 3*y2)*(-1 + z)*z2);
    w[110]=+6*(-1 + x)*x2*(-1 + y)*y*(-1 + z)*z2;
    w[111]=+(-1 + x)*x2*(1 - 4*y + 3*y2)*(-1 + z)*z2;
    w[112]=-6*x2*(-3 + 2*x)*(-1 + y)*y*z2*(-3 + 2*z);
    w[113]=+6*(-1 + x)*x2*(-1 + y)*y*z2*(-3 + 2*z);
    w[114]=+x2*(-3 + 2*x)*y*(-2 + 3*y)*z2*(-3 + 2*z);
    w[115]=+6*x2*(-3 + 2*x)*(-1 + y)*y*(-1 + z)*z2;
    w[116]=-((-1 + x)*x2*y*(-2 + 3*y)*z2*(-3 + 2*z));
    w[117]=-(x2*(-3 + 2*x)*y*(-2 + 3*y)*(-1 + z)*z2);
    w[118]=-6*(-1 + x)*x2*(-1 + y)*y*(-1 + z)*z2;
    w[119]=+(-1 + x)*x2*y*(-2 + 3*y)*(-1 + z)*z2;
    w[120]=+6*x12*(1 + 2*x)*(-1 + y)*y*z2*(-3 + 2*z);
    w[121]=+6*x12*x*(-1 + y)*y*z2*(-3 + 2*z);
    w[122]=-(x12*(1 + 2*x)*y*(-2 + 3*y)*z2*(-3 + 2*z));
    w[123]=-6*x12*(1 + 2*x)*(-1 + y)*y*(-1 + z)*z2;
    w[124]=-(x12*x*y*(-2 + 3*y)*z2*(-3 + 2*z));
    w[125]=+x12*(1 + 2*x)*y*(-2 + 3*y)*(-1 + z)*z2;
    w[126]=-6*x12*x*(-1 + y)*y*(-1 + z)*z2;
    w[127]=+x12*x*y*(-2 + 3*y)*(-1 + z)*z2;
      
    w[128]=6*x12*(1 + 2*x)*y12*(1 + 2*y)*(-1 + z);
    w[129]=+6*x12*x*y12*(1 + 2*y)*(-1 + z)*z;
    w[130]=+6*x12*(1 + 2*x)*y12*y*(-1 + z)*z;
    w[131]=+x12*(1 + 2*x)*y12*(1 + 2*y)*(1 - 4*z + 3*z2);
    w[132]=+6*x12*x*y12*y*(-1 + z)*z;
    w[133]=+x12*(1 + 2*x)*y12*y*(1 - 4*z + 3*z2);
    w[134]=+x12*x*y12*(1 + 2*y)*(1 - 4*z + 3*z2);
    w[135]=+x12*x*y12*y*(1 - 4*z + 3*z2);
    w[136]=-6*x2*(-3 + 2*x)*y12*(1 + 2*y)*(-1 + z)*z;
    w[137]=+6*(-1 + x)*x2*y12*(1 + 2*y)*(-1 + z)*z;
    w[138]=-6*x2*(-3 + 2*x)*y12*y*(-1 + z)*z;
    w[139]=-(x2*(-3 + 2*x)*y12*(1 + 2*y)*(1 - 4*z + 3*z2));
    w[140]=+6*(-1 + x)*x2*y12*y*(-1 + z)*z;
    w[141]=-(x2*(-3 + 2*x)*y12*y*(1 - 4*z + 3*z2));
    w[142]=+(-1 + x)*x2*y12*(1 + 2*y)*(1 - 4*z + 3*z2);
    w[143]=+(-1 + x)*x2*y12*y*(1 - 4*z + 3*z2);
    w[144]=+6*x2*(-3 + 2*x)*y2*(-3 + 2*y)*(-1 + z)*z;
    w[145]=-6*(-1 + x)*x2*y2*(-3 + 2*y)*(-1 + z)*z;
    w[146]=-6*x2*(-3 + 2*x)*(-1 + y)*y2*(-1 + z)*z;
    w[147]=+x2*(-3 + 2*x)*y2*(-3 + 2*y)*(1 - 4*z + 3*z2);
    w[148]=+6*(-1 + x)*x2*(-1 + y)*y2*(-1 + z)*z;
    w[149]=-(x2*(-3 + 2*x)*(-1 + y)*y2*(1 - 4*z + 3*z2));
    w[150]=-((-1 + x)*x2*y2*(-3 + 2*y)*(1 - 4*z + 3*z2));
    w[151]=+(-1 + x)*x2*(-1 + y)*y2*(1 - 4*z + 3*z2);
    w[152]=-6*x12*(1 + 2*x)*y2*(-3 + 2*y)*(-1 + z)*z;
    w[153]=-6*x12*x*y2*(-3 + 2*y)*(-1 + z)*z;
    w[154]=+6*x12*(1 + 2*x)*(-1 + y)*y2*(-1 + z)*z;
    w[155]=-(x12*(1 + 2*x)*y2*(-3 + 2*y)*(1 - 4*z + 3*z2));
    w[156]=+6*x12*x*(-1 + y)*y2*(-1 + z)*z;
    w[157]=+x12*(1 + 2*x)*(-1 + y)*y2*(1 - 4*z + 3*z2);
    w[158]=-(x12*x*y2*(-3 + 2*y)*(1 - 4*z + 3*z2));
    w[159]=+x12*x*(-1 + y)*y2*(1 - 4*z + 3*z2);
    w[160]=-6*x12*(1 + 2*x)*y12*(1 + 2*y)*(-1 + z)*z;
    w[161]=-6*x12*x*y12*(1 + 2*y)*(-1 + z)*z;
    w[162]=-6*x12*(1 + 2*x)*y12*y*(-1 + z)*z;
    w[163]=+x12*(1 + 2*x)*y12*(1 + 2*y)*z*(-2 + 3*z);
    w[164]=-6*x12*x*y12*y*(-1 + z)*z;
    w[165]=+x12*(1 + 2*x)*y12*y*z*(-2 + 3*z);
    w[166]=+x12*x*y12*(1 + 2*y)*z*(-2 + 3*z);
    w[167]=+x12*x*y12*y*z*(-2 + 3*z);
    w[168]=+6*x2*(-3 + 2*x)*y12*(1 + 2*y)*(-1 + z)*z;
    w[169]=-6*(-1 + x)*x2*y12*(1 + 2*y)*(-1 + z)*z;
    w[170]=+6*x2*(-3 + 2*x)*y12*y*(-1 + z)*z;
    w[171]=-(x2*(-3 + 2*x)*y12*(1 + 2*y)*z*(-2 + 3*z));
    w[172]=-6*(-1 + x)*x2*y12*y*(-1 + z)*z;
    w[173]=-(x2*(-3 + 2*x)*y12*y*z*(-2 + 3*z));
    w[174]=+(-1 + x)*x2*y12*(1 + 2*y)*z*(-2 + 3*z);
    w[175]=+(-1 + x)*x2*y12*y*z*(-2 + 3*z);
    w[176]=-6*x2*(-3 + 2*x)*y2*(-3 + 2*y)*(-1 + z)*z;
    w[177]=+6*(-1 + x)*x2*y2*(-3 + 2*y)*(-1 + z)*z;
    w[178]=+6*x2*(-3 + 2*x)*(-1 + y)*y2*(-1 + z)*z;
    w[179]=+x2*(-3 + 2*x)*y2*(-3 + 2*y)*z*(-2 + 3*z);
    w[180]=-6*(-1 + x)*x2*(-1 + y)*y2*(-1 + z)*z;
    w[181]=-(x2*(-3 + 2*x)*(-1 + y)*y2*z*(-2 + 3*z));
    w[182]=-((-1 + x)*x2*y2*(-3 + 2*y)*z*(-2 + 3*z));
    w[183]=+(-1 + x)*x2*(-1 + y)*y2*z*(-2 + 3*z);
    w[184]=+6*x12*(1 + 2*x)*y2*(-3 + 2*y)*(-1 + z)*z;
    w[185]=+6*x12*x*y2*(-3 + 2*y)*(-1 + z)*z;
    w[186]=-6*x12*(1 + 2*x)*(-1 + y)*y2*(-1 + z)*z;
    w[187]=-(x12*(1 + 2*x)*y2*(-3 + 2*y)*z*(-2 + 3*z));
    w[188]=-6*x12*x*(-1 + y)*y2*(-1 + z)*z;
    w[189]=+x12*(1 + 2*x)*(-1 + y)*y2*z*(-2 + 3*z);
    w[190]=-(x12*x*y2*(-3 + 2*y)*z*(-2 + 3*z));
    w[191]=+x12*x*(-1 + y)*y2*z*(-2 + 3*z);
  }

  //! get first derivative at parametric coordinate
  template <class ElemData>
  void derivate(const std::vector<double> &coords, const ElemData &cd,
		std::vector<T> &derivs) const
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
      T(6*(-1 + x)*x*y12*(1 + 2*y)*z12*(1 + 2*z)*cd.node0()
	+(1 - 4*x + 3*x2)*y12*(1 + 2*y)*z12*(1 + 2*z)*this->derivs_[cd.node0_index()][0]
	+6*(-1 + x)*x*y12*y*z12*(1 + 2*z)*this->derivs_[cd.node0_index()][1]
	+6*(-1 + x)*x*y12*(1 + 2*y)*z12*z*this->derivs_[cd.node0_index()][2]
	+(1 - 4*x + 3*x2)*y12*y*z12*(1 + 2*z)*this->derivs_[cd.node0_index()][3]
	+6*(-1 + x)*x*y12*y*z12*z*this->derivs_[cd.node0_index()][4]
	+(1 - 4*x + 3*x2)*y12*(1 + 2*y)*z12*z*this->derivs_[cd.node0_index()][5]
	+(1 - 4*x + 3*x2)*y12*y*z12*z*this->derivs_[cd.node0_index()][6]
	-6*(-1 + x)*x*y12*(1 + 2*y)*z12*(1 + 2*z)*cd.node1()
	+x*(-2 + 3*x)*y12*(1 + 2*y)*z12*(1 + 2*z)*this->derivs_[cd.node1_index()][0]
	-6*(-1 + x)*x*y12*y*z12*(1 + 2*z)*this->derivs_[cd.node1_index()][1]
	-6*(-1 + x)*x*y12*(1 + 2*y)*z12*z*this->derivs_[cd.node1_index()][2]
	+x*(-2 + 3*x)*y12*y*z12*(1 + 2*z)*this->derivs_[cd.node1_index()][3]
	-6*(-1 + x)*x*y12*y*z12*z*this->derivs_[cd.node1_index()][4]
	+x*(-2 + 3*x)*y12*(1 + 2*y)*z12*z*this->derivs_[cd.node1_index()][5]
	+x*(-2 + 3*x)*y12*y*z12*z*this->derivs_[cd.node1_index()][6]
	+6*(-1 + x)*x*y2*(-3 + 2*y)*z12*(1 + 2*z)*cd.node2()
	-(x*(-2 + 3*x)*y2*(-3 + 2*y)*z12*(1 + 2*z))*this->derivs_[cd.node2_index()][0]
	-6*(-1 + x)*x*(-1 + y)*y2*z12*(1 + 2*z)*this->derivs_[cd.node2_index()][1]
	+6*(-1 + x)*x*y2*(-3 + 2*y)*z12*z*this->derivs_[cd.node2_index()][2]
	+x*(-2 + 3*x)*(-1 + y)*y2*z12*(1 + 2*z)*this->derivs_[cd.node2_index()][3]
	-6*(-1 + x)*x*(-1 + y)*y2*z12*z*this->derivs_[cd.node2_index()][4]
	-(x*(-2 + 3*x)*y2*(-3 + 2*y)*z12*z)*this->derivs_[cd.node2_index()][5]
	+x*(-2 + 3*x)*(-1 + y)*y2*z12*z*this->derivs_[cd.node2_index()][6]
	-6*(-1 + x)*x*y2*(-3 + 2*y)*z12*(1 + 2*z)*cd.node3()
	-((1 - 4*x + 3*x2)*y2*(-3 + 2*y)*z12*(1 + 2*z))*this->derivs_[cd.node3_index()][0]
	+6*(-1 + x)*x*(-1 + y)*y2*z12*(1 + 2*z)*this->derivs_[cd.node3_index()][2]
	-6*(-1 + x)*x*y2*(-3 + 2*y)*z12*z*this->derivs_[cd.node3_index()][3]
	+(1 - 4*x + 3*x2)*(-1 + y)*y2*z12*(1 + 2*z)*this->derivs_[cd.node3_index()][4]
	+6*(-1 + x)*x*(-1 + y)*y2*z12*z*this->derivs_[cd.node3_index()][5]
	-((1 - 4*x + 3*x2)*y2*(-3 + 2*y)*z12*z)*this->derivs_[cd.node3_index()][6]
	+(1 - 4*x + 3*x2)*(-1 + y)*y2*z12*z*this->derivs_[cd.node3_index()][7]
	-6*(-1 + x)*x*y12*(1 + 2*y)*z2*(-3 + 2*z)*cd.node4()
	-((1 - 4*x + 3*x2)*y12*(1 + 2*y)*z2*(-3 + 2*z))*this->derivs_[cd.node4_index()][0]
	-6*(-1 + x)*x*y12*y*z2*(-3 + 2*z)*this->derivs_[cd.node4_index()][1]
	+6*(-1 + x)*x*y12*(1 + 2*y)*(-1 + z)*z2*this->derivs_[cd.node4_index()][2]
	-((1 - 4*x + 3*x2)*y12*y*z2*(-3 + 2*z))*this->derivs_[cd.node4_index()][3]
	+6*(-1 + x)*x*y12*y*(-1 + z)*z2*this->derivs_[cd.node4_index()][4]
	+(1 - 4*x + 3*x2)*y12*(1 + 2*y)*(-1 + z)*z2*this->derivs_[cd.node4_index()][5]
	+(1 - 4*x + 3*x2)*y12*y*(-1 + z)*z2*this->derivs_[cd.node4_index()][6]
	+6*(-1 + x)*x*y12*(1 + 2*y)*z2*(-3 + 2*z)*cd.node5()
	-(x*(-2 + 3*x)*y12*(1 + 2*y)*z2*(-3 + 2*z))*this->derivs_[cd.node5_index()][0]
	+6*(-1 + x)*x*y12*y*z2*(-3 + 2*z)*this->derivs_[cd.node5_index()][1]
	-6*(-1 + x)*x*y12*(1 + 2*y)*(-1 + z)*z2*this->derivs_[cd.node5_index()][2]
	-(x*(-2 + 3*x)*y12*y*z2*(-3 + 2*z))*this->derivs_[cd.node5_index()][3]
	-6*(-1 + x)*x*y12*y*(-1 + z)*z2*this->derivs_[cd.node5_index()][4]
	+x*(-2 + 3*x)*y12*(1 + 2*y)*(-1 + z)*z2*this->derivs_[cd.node5_index()][5]
	+x*(-2 + 3*x)*y12*y*(-1 + z)*z2*this->derivs_[cd.node5_index()][6]
	-6*(-1 + x)*x*y2*(-3 + 2*y)*z2*(-3 + 2*z)*cd.node6()
	+x*(-2 + 3*x)*y2*(-3 + 2*y)*z2*(-3 + 2*z)*this->derivs_[cd.node6_index()][0]
	+6*(-1 + x)*x*(-1 + y)*y2*z2*(-3 + 2*z)*this->derivs_[cd.node6_index()][1]
	+6*(-1 + x)*x*y2*(-3 + 2*y)*(-1 + z)*z2*this->derivs_[cd.node6_index()][2]
	-(x*(-2 + 3*x)*(-1 + y)*y2*z2*(-3 + 2*z))*this->derivs_[cd.node6_index()][3]
	-6*(-1 + x)*x*(-1 + y)*y2*(-1 + z)*z2*this->derivs_[cd.node6_index()][4]
	-(x*(-2 + 3*x)*y2*(-3 + 2*y)*(-1 + z)*z2)*this->derivs_[cd.node6_index()][5]
	+x*(-2 + 3*x)*(-1 + y)*y2*(-1 + z)*z2*this->derivs_[cd.node6_index()][6]
	+6*(-1 + x)*x*y2*(-3 +  2*y)*z2*(-3 + 2*z)*cd.node7()
	+(1 - 4*x + 3*x2)*y2*(-3 + 2*y)*z2*(-3 + 2*z)*this->derivs_[cd.node7_index()][0]
	-6*(-1 + x)*x*(-1 + y)*y2*z2*(-3 + 2*z)*this->derivs_[cd.node7_index()][1]
	-6*(-1 + x)*x*y2*(-3 + 2*y)*(-1 + z)*z2*this->derivs_[cd.node7_index()][2]
	-((1 - 4*x + 3*x2)*(-1 + y)*y2*z2*(-3 + 2*z))*this->derivs_[cd.node7_index()][3]
	+6*(-1 + x)*x*(-1 + y)*y2*(-1 + z)*z2*this->derivs_[cd.node7_index()][4]
	-((1 - 4*x + 3*x2)*y2*(-3 + 2*y)*(-1 + z)*z2)*this->derivs_[cd.node7_index()][5]
	+(1 - 4*x + 3*x2)*(-1 + y)*y2*(-1 + z)*z2*this->derivs_[cd.node7_index()][6]);
      
    derivs[1]=
      T(6*x12*(1 + 2*x)*(-1 + y)*y*z12*(1 + 2*z)*cd.node0()
	+6*x12*x*(-1 + y)*y*z12*(1 + 2*z)*this->derivs_[cd.node0_index()][0]
	+x12*(1 + 2*x)*(1 - 4*y + 3*y2)*z12*(1 + 2*z)*this->derivs_[cd.node0_index()][1]
	+6*x12*(1 + 2*x)*(-1 + y)*y*z12*z*this->derivs_[cd.node0_index()][2]
	+x12*x*(1 - 4*y + 3*y2)*z12*(1 + 2*z)*this->derivs_[cd.node0_index()][3]
	+x12*(1 + 2*x)*(1 - 4*y + 3*y2)*z12*z*this->derivs_[cd.node0_index()][4]
	+6*x12*x*(-1 + y)*y*z12*z*this->derivs_[cd.node0_index()][5]
	+x12*x*(1 - 4*y + 3*y2)*z12*z*this->derivs_[cd.node0_index()][6]
	-6*x2*(-3 + 2*x)*(-1 + y)*y*z12*(1 + 2*z)*cd.node1()
	+6*(-1 + x)*x2*(-1 + y)*y*z12*(1 + 2*z)*this->derivs_[cd.node1_index()][0]
	-(x2*(-3 + 2*x)*(1 - 4*y + 3*y2)*z12*(1 + 2*z))*this->derivs_[cd.node1_index()][1]
	-6*x2*(-3 + 2*x)*(-1 + y)*y*z12*z*this->derivs_[cd.node1_index()][2]
	+(-1 + x)*x2*(1 - 4*y + 3*y2)*z12*(1 + 2*z)*this->derivs_[cd.node1_index()][3]
	-(x2*(-3 + 2*x)*(1 - 4*y + 3*y2)*z12*z)*this->derivs_[cd.node1_index()][4]
	+6*(-1 + x)*x2*(-1 + y)*y*z12*z*this->derivs_[cd.node1_index()][5]
	+(-1 + x)*x2*(1 - 4*y + 3*y2)*z12*z*this->derivs_[cd.node1_index()][6]
	+6*x2*(-3 + 2*x)*(-1 + y)*y*z12*(1 + 2*z)*cd.node2()
	-6*(-1 + x)*x2*(-1 + y)*y*z12*(1 + 2*z)*this->derivs_[cd.node2_index()][0]
	-(x2*(-3 + 2*x)*y*(-2 + 3*y)*z12*(1 + 2*z))*this->derivs_[cd.node2_index()][1]
	+6*x2*(-3 + 2*x)*(-1 + y)*y*z12*z*this->derivs_[cd.node2_index()][2]
	+(-1 + x)*x2*y*(-2 + 3*y)*z12*(1 + 2*z)*this->derivs_[cd.node2_index()][3]
	-(x2*(-3 + 2*x)*y*(-2 + 3*y)*z12*z)*this->derivs_[cd.node2_index()][4]
	-6*(-1 + x)*x2*(-1 + y)*y*z12*z*this->derivs_[cd.node2_index()][5]
	+(-1 + x)*x2*y*(-2 + 3*y)*z12*z*this->derivs_[cd.node2_index()][6]
	-6*x12*(1 + 2*x)*(-1 + y)*y*z12*(1 + 2*z)*cd.node3()
	-6*x12*x*(-1 + y)*y*z12*(1 + 2*z)*this->derivs_[cd.node3_index()][0]
	+x12*(1 + 2*x)*y*(-2 + 3*y)*z12*(1 + 2*z)*this->derivs_[cd.node3_index()][1]
	-6*x12*(1 + 2*x)*(-1 + y)*y*z12*z*this->derivs_[cd.node3_index()][2]
	+x12*x*y*(-2 + 3*y)*z12*(1 + 2*z)*this->derivs_[cd.node3_index()][3]
	+x12*(1 + 2*x)*y*(-2 + 3*y)*z12*z*this->derivs_[cd.node3_index()][4]
	-6*x12*x*(-1 + y)*y*z12*z*this->derivs_[cd.node3_index()][5]
	+x12*x*y*(-2 + 3*y)*z12*z*this->derivs_[cd.node3_index()][6]
	-6*x12*(1 + 2*x)*(-1 + y)*y*z2*(-3 + 2*z)*cd.node4()
	-6*x12*x*(-1 + y)*y*z2*(-3 + 2*z)*this->derivs_[cd.node4_index()][0]
	-(x12*(1 + 2*x)*(1 - 4*y + 3*y2)*z2*(-3 + 2*z))*this->derivs_[cd.node4_index()][1]
	+6*x12*(1 + 2*x)*(-1 + y)*y*(-1 + z)*z2*this->derivs_[cd.node4_index()][2]
	-(x12*x*(1 - 4*y + 3*y2)*z2*(-3 + 2*z))*this->derivs_[cd.node4_index()][3]
	+x12*(1 + 2*x)*(1 - 4*y + 3*y2)*(-1 + z)*z2*this->derivs_[cd.node4_index()][4]
	+6*x12*x*(-1 + y)*y*(-1 + z)*z2*this->derivs_[cd.node4_index()][5]
	+x12*x*(1 - 4*y + 3*y2)*(-1 + z)*z2*this->derivs_[cd.node4_index()][6]
	+6*x2*(-3 + 2*x)*(-1 + y)*y*z2*(-3 + 2*z)*cd.node5()
	-6*(-1 + x)*x2*(-1 + y)*y*z2*(-3 + 2*z)*this->derivs_[cd.node5_index()][0]
	+x2*(-3 + 2*x)*(1 - 4*y + 3*y2)*z2*(-3 + 2*z)*this->derivs_[cd.node5_index()][1]
	-6*x2*(-3 + 2*x)*(-1 + y)*y*(-1 + z)*z2*this->derivs_[cd.node5_index()][2]
	-((-1 + x)*x2*(1 - 4*y + 3*y2)*z2*(-3 + 2*z))*this->derivs_[cd.node5_index()][3]
	-(x2*(-3 + 2*x)*(1 - 4*y + 3*y2)*(-1 + z)*z2)*this->derivs_[cd.node5_index()][4]
	+6*(-1 + x)*x2*(-1 + y)*y*(-1 + z)*z2*this->derivs_[cd.node5_index()][5]
	+(-1 + x)*x2*(1 - 4*y + 3*y2)*(-1 + z)*z2*this->derivs_[cd.node5_index()][6]
	-6*x2*(-3 + 2*x)*(-1 + y)*y*z2*(-3 + 2*z)*cd.node6()
	+6*(-1 + x)*x2*(-1 + y)*y*z2*(-3 + 2*z)*this->derivs_[cd.node6_index()][0]
	+x2*(-3 + 2*x)*y*(-2 + 3*y)*z2*(-3 + 2*z)*this->derivs_[cd.node6_index()][1]
	+6*x2*(-3 + 2*x)*(-1 + y)*y*(-1 + z)*z2*this->derivs_[cd.node6_index()][2]
	-((-1 + x)*x2*y*(-2 + 3*y)*z2*(-3 + 2*z))*this->derivs_[cd.node6_index()][3]
	-(x2*(-3 + 2*x)*y*(-2 + 3*y)*(-1 + z)*z2)*this->derivs_[cd.node6_index()][4]
	-6*(-1 + x)*x2*(-1 + y)*y*(-1 + z)*z2*this->derivs_[cd.node6_index()][5]
	+(-1 + x)*x2*y*(-2 + 3*y)*(-1 + z)*z2*this->derivs_[cd.node6_index()][6]
	+6*x12*(1 + 2*x)*(-1 + y)*y*z2*(-3 + 2*z)*cd.node7()
	+6*x12*x*(-1 + y)*y*z2*(-3 + 2*z)*this->derivs_[cd.node7_index()][0]
	-(x12*(1 + 2*x)*y*(-2 + 3*y)*z2*(-3 + 2*z))*this->derivs_[cd.node7_index()][1]
	-6*x12*(1 + 2*x)*(-1 + y)*y*(-1 + z)*z2*this->derivs_[cd.node7_index()][2]
	-(x12*x*y*(-2 + 3*y)*z2*(-3 + 2*z))*this->derivs_[cd.node7_index()][3]
	+x12*(1 + 2*x)*y*(-2 + 3*y)*(-1 + z)*z2*this->derivs_[cd.node7_index()][4]
	-6*x12*x*(-1 + y)*y*(-1 + z)*z2*this->derivs_[cd.node7_index()][5]
	+x12*x*y*(-2 + 3*y)*(-1 + z)*z2*this->derivs_[cd.node7_index()][6]);
      
    derivs[2]=
      T(6*x12*(1 + 2*x)*y12*(1 + 2*y)*(-1 + z)*z*cd.node0()
	+6*x12*x*y12*(1 + 2*y)*(-1 + z)*z*this->derivs_[cd.node0_index()][0]
	+6*x12*(1 + 2*x)*y12*y*(-1 + z)*z*this->derivs_[cd.node0_index()][1]
	+x12*(1 + 2*x)*y12*(1 + 2*y)*(1 - 4*z + 3*z2)*this->derivs_[cd.node0_index()][2]
	+6*x12*x*y12*y*(-1 + z)*z*this->derivs_[cd.node0_index()][3]
	+x12*(1 + 2*x)*y12*y*(1 - 4*z + 3*z2)*this->derivs_[cd.node0_index()][4]
	+x12*x*y12*(1 + 2*y)*(1 - 4*z + 3*z2)*this->derivs_[cd.node0_index()][5]
	+x12*x*y12*y*(1 - 4*z + 3*z2)*this->derivs_[cd.node0_index()][6]
	-6*x2*(-3 + 2*x)*y12*(1 + 2*y)*(-1 + z)*z*cd.node1()
	+6*(-1 + x)*x2*y12*(1 + 2*y)*(-1 + z)*z*this->derivs_[cd.node1_index()][0]
	-6*x2*(-3 + 2*x)*y12*y*(-1 + z)*z*this->derivs_[cd.node1_index()][1]
	-(x2*(-3 + 2*x)*y12*(1 + 2*y)*(1 - 4*z + 3*z2))*this->derivs_[cd.node1_index()][2]
	+6*(-1 + x)*x2*y12*y*(-1 + z)*z*this->derivs_[cd.node1_index()][3]
	-(x2*(-3 + 2*x)*y12*y*(1 - 4*z + 3*z2))*this->derivs_[cd.node1_index()][4]
	+(-1 + x)*x2*y12*(1 + 2*y)*(1 - 4*z + 3*z2)*this->derivs_[cd.node1_index()][5]
	+(-1 + x)*x2*y12*y*(1 - 4*z + 3*z2)*this->derivs_[cd.node1_index()][6]
	+6*x2*(-3 + 2*x)*y2*(-3 + 2*y)*(-1 + z)*z*cd.node2()
	-6*(-1 + x)*x2*y2*(-3 + 2*y)*(-1 + z)*z*this->derivs_[cd.node2_index()][0]
	-6*x2*(-3 + 2*x)*(-1 + y)*y2*(-1 + z)*z*this->derivs_[cd.node2_index()][1]
	+x2*(-3 + 2*x)*y2*(-3 + 2*y)*(1 - 4*z + 3*z2)*this->derivs_[cd.node2_index()][2]
	+6*(-1 + x)*x2*(-1 + y)*y2*(-1 + z)*z*this->derivs_[cd.node2_index()][3]
	-(x2*(-3 + 2*x)*(-1 + y)*y2*(1 - 4*z + 3*z2))*this->derivs_[cd.node2_index()][4]
	-((-1 + x)*x2*y2*(-3 + 2*y)*(1 - 4*z + 3*z2))*this->derivs_[cd.node2_index()][5]
	+(-1 + x)*x2*(-1 + y)*y2*(1 - 4*z + 3*z2)*this->derivs_[cd.node2_index()][6]
	-6*x12*(1 + 2*x)*y2*(-3 + 2*y)*(-1 + z)*z*cd.node3()
	-6*x12*x*y2*(-3 + 2*y)*(-1 + z)*z*this->derivs_[cd.node3_index()][0]
	+6*x12*(1 + 2*x)*(-1 + y)*y2*(-1 + z)*z*this->derivs_[cd.node3_index()][1]
	-(x12*(1 + 2*x)*y2*(-3 + 2*y)*(1 - 4*z + 3*z2))*this->derivs_[cd.node3_index()][2]
	+6*x12*x*(-1 + y)*y2*(-1 + z)*z*this->derivs_[cd.node3_index()][3]
	+x12*(1 + 2*x)*(-1 + y)*y2*(1 - 4*z + 3*z2)*this->derivs_[cd.node3_index()][4]
	-(x12*x*y2*(-3 + 2*y)*(1 - 4*z + 3*z2))*this->derivs_[cd.node3_index()][5]
	+x12*x*(-1 + y)*y2*(1 - 4*z + 3*z2)*this->derivs_[cd.node3_index()][6]
	-6*x12*(1 + 2*x)*y12*(1 + 2*y)*(-1 + z)*z*cd.node4()
	-6*x12*x*y12*(1 + 2*y)*(-1 + z)*z*this->derivs_[cd.node4_index()][0]
	-6*x12*(1 + 2*x)*y12*y*(-1 + z)*z*this->derivs_[cd.node4_index()][1]
	+x12*(1 + 2*x)*y12*(1 + 2*y)*z*(-2 + 3*z)*this->derivs_[cd.node4_index()][2]
	-6*x12*x*y12*y*(-1 + z)*z*this->derivs_[cd.node4_index()][3]
	+x12*(1 + 2*x)*y12*y*z*(-2 + 3*z)*this->derivs_[cd.node4_index()][4]
	+x12*x*y12*(1 + 2*y)*z*(-2 + 3*z)*this->derivs_[cd.node4_index()][5]
	+x12*x*y12*y*z*(-2 + 3*z)*this->derivs_[cd.node4_index()][6]
	+6*x2*(-3 + 2*x)*y12*(1 + 2*y)*(-1 + z)*z*cd.node5()
	-6*(-1 + x)*x2*y12*(1 + 2*y)*(-1 + z)*z*this->derivs_[cd.node5_index()][0]
	+6*x2*(-3 + 2*x)*y12*y*(-1 + z)*z*this->derivs_[cd.node5_index()][1]
	-(x2*(-3 + 2*x)*y12*(1 + 2*y)*z*(-2 + 3*z))*this->derivs_[cd.node5_index()][2]
	-6*(-1 + x)*x2*y12*y*(-1 + z)*z*this->derivs_[cd.node5_index()][3]
	-(x2*(-3 + 2*x)*y12*y*z*(-2 + 3*z))*this->derivs_[cd.node5_index()][4]
	+(-1 + x)*x2*y12*(1 + 2*y)*z*(-2 + 3*z)*this->derivs_[cd.node5_index()][5]
	+(-1 + x)*x2*y12*y*z*(-2 + 3*z)*this->derivs_[cd.node5_index()][6]
	-6*x2*(-3 + 2*x)*y2*(-3 + 2*y)*(-1 + z)*z*cd.node6()
	+6*(-1 + x)*x2*y2*(-3 + 2*y)*(-1 + z)*z*this->derivs_[cd.node6_index()][0]
	+6*x2*(-3 + 2*x)*(-1 + y)*y2*(-1 + z)*z*this->derivs_[cd.node6_index()][1]
	+x2*(-3 + 2*x)*y2*(-3 + 2*y)*z*(-2 + 3*z)*this->derivs_[cd.node6_index()][2]
	-6*(-1 + x)*x2*(-1 + y)*y2*(-1 + z)*z*this->derivs_[cd.node6_index()][3]
	-(x2*(-3 + 2*x)*(-1 + y)*y2*z*(-2 + 3*z))*this->derivs_[cd.node6_index()][4]
	-((-1 + x)*x2*y2*(-3 + 2*y)*z*(-2 + 3*z))*this->derivs_[cd.node6_index()][5]
	+(-1 + x)*x2*(-1 + y)*y2*z*(-2 + 3*z)*this->derivs_[cd.node6_index()][6]
	+6*x12*(1 + 2*x)*y2*(-3 + 2*y)*(-1 + z)*z*cd.node7()
	+6*x12*x*y2*(-3 + 2*y)*(-1 + z)*z*this->derivs_[cd.node7_index()][0]
	-6*x12*(1 + 2*x)*(-1 + y)*y2*(-1 + z)*z*this->derivs_[cd.node7_index()][1]
	-(x12*(1 + 2*x)*y2*(-3 + 2*y)*z*(-2 + 3*z))*this->derivs_[cd.node7_index()][2]
	-6*x12*x*(-1 + y)*y2*(-1 + z)*z*this->derivs_[cd.node7_index()][3]
	+x12*(1 + 2*x)*(-1 + y)*y2*z*(-2 + 3*z)*this->derivs_[cd.node7_index()][4]
	-(x12*x*y2*(-3 + 2*y)*z*(-2 + 3*z))*this->derivs_[cd.node7_index()][5]
	+x12*x*(-1 + y)*y2*z*(-2 + 3*z)*this->derivs_[cd.node7_index()][6]);
  }
  

  //! get parametric coordinate for value within the element
  template <class ElemData>
  bool get_coords(std::vector<double> &coords, const T& value, 
		  const ElemData &cd) const  
  {
    HexLocate< HexTricubicHmt<T> > CL;
    return CL.get_coords(this, coords, value, cd);
  }
    
  //! get arc length for edge
  template <class ElemData>
  double get_arc_length(const unsigned edge, const ElemData &cd) const  
  {
    return get_arc3d_length<CrvGaussian2<double> >(this, edge, cd);
  }

  //! get area
  template <class ElemData>
    double get_area(const unsigned face, const ElemData &cd) const  
  {
    return get_area3<QuadGaussian3<double> >(this, face, cd);
  }
  
  //! get volume
  template <class ElemData>
    double get_volume(const ElemData & cd) const  
  {
    return get_volume3(this, cd);
  }
   
  static  const std::string type_name(int n = -1);
  virtual void io (Piostream& str);
};


template <class T>
const string
HexTricubicHmt<T>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const std::string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;
  }
  else if (n == 0)
  {
    static const std::string nm("HexTricubicHmt");
    return nm;
  } else {
    return find_type_name((T *)0);
  }
}

template <class T>
const TypeDescription*
get_type_description(HexTricubicHmt<T> *)
{
  static TypeDescription* td = 0;
  if(!td){
    const TypeDescription *sub = get_type_description((T*)0);
    TypeDescription::td_vec *subs = scinew TypeDescription::td_vec(1);
    (*subs)[0] = sub;
    td = scinew TypeDescription("HexTricubicHmt", subs, 
				std::string(__FILE__),
				"SCIRun", 
				TypeDescription::BASIS_E);
  }
  return td;
}

const int HEXTRICUBICHMT_VERSION = 1;
template <class T>
void
HexTricubicHmt<T>::io(Piostream &stream)
{
  stream.begin_class(get_type_description(this)->get_name(),
                     HEXTRICUBICHMT_VERSION);
  Pio(stream, this->derivs_);
  stream.end_class();
}

} //namespace SCIRun


#endif // HexTricubicHmt_h
