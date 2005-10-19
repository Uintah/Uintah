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
};


//! Class for handling of element of type hexahedron with 
//! tricubic hermitian interpolation
template <class T>
class HexTricubicHmt : public HexApprox, 
		       public HexGaussian3<double>, 
		       public HexTricubicHmtUnitElement
{
public:
  typedef T value_type;

  HexTricubicHmt() {}
  virtual ~HexTricubicHmt() {}
  
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
    
    return 64;
  }

  //! get value at parametric coordinate 
  template <class CellData>
  T interpolate(const vector<double> &coords, const CellData &cd) const
  {
    double w[64];
    get_weights(coords, w); 
    return (T)(w[0]  * cd.node0() +
	       w[1]  * derivs_[cd.node0_index()][0] +
	       w[2]  * derivs_[cd.node0_index()][1] +
	       w[3]  * derivs_[cd.node0_index()][2] +
	       w[4]  * derivs_[cd.node0_index()][3] +
	       w[5]  * derivs_[cd.node0_index()][4] +
	       w[6]  * derivs_[cd.node0_index()][5] +
	       w[7]  * derivs_[cd.node0_index()][6] +
	       w[8]  * cd.node1()		    +
	       w[9]  * derivs_[cd.node1_index()][0] +
	       w[10] * derivs_[cd.node1_index()][1] +
	       w[11] * derivs_[cd.node1_index()][2] +
	       w[12] * derivs_[cd.node1_index()][3] +
	       w[13] * derivs_[cd.node1_index()][4] +  
	       w[14] * derivs_[cd.node1_index()][5] +
	       w[15] * derivs_[cd.node1_index()][6] +
	       w[16] * cd.node2()		    +
	       w[17] * derivs_[cd.node2_index()][0] +
	       w[18] * derivs_[cd.node2_index()][1] +
	       w[19] * derivs_[cd.node2_index()][2] +
	       w[20] * derivs_[cd.node2_index()][3] +
	       w[21] * derivs_[cd.node2_index()][4] +
	       w[22] * derivs_[cd.node2_index()][5] +
	       w[23] * derivs_[cd.node2_index()][6] +
	       w[24] * cd.node3()		    +
	       w[25] * derivs_[cd.node3_index()][0] +
	       w[26] * derivs_[cd.node3_index()][1] +
	       w[27] * derivs_[cd.node3_index()][2] +
	       w[28] * derivs_[cd.node3_index()][3] +
	       w[29] * derivs_[cd.node3_index()][4] +
	       w[30] * derivs_[cd.node3_index()][5] +
	       w[31] * derivs_[cd.node3_index()][6] +
	       w[32] * cd.node4()		    +
	       w[33] * derivs_[cd.node4_index()][0] +
	       w[34] * derivs_[cd.node4_index()][1] +
	       w[35] * derivs_[cd.node4_index()][2] +
	       w[36] * derivs_[cd.node4_index()][3] +
	       w[37] * derivs_[cd.node4_index()][4] +
	       w[38] * derivs_[cd.node4_index()][5] +
	       w[39] * derivs_[cd.node4_index()][6] +
	       w[40] * cd.node5()		    +
	       w[41] * derivs_[cd.node5_index()][0] +
	       w[42] * derivs_[cd.node5_index()][1] +
	       w[43] * derivs_[cd.node5_index()][2] +
	       w[44] * derivs_[cd.node5_index()][3] +
	       w[45] * derivs_[cd.node5_index()][4] +
	       w[46] * derivs_[cd.node5_index()][5] +
	       w[47] * derivs_[cd.node5_index()][6] +
	       w[48] * cd.node6()		    +
	       w[49] * derivs_[cd.node6_index()][0] +
	       w[50] * derivs_[cd.node6_index()][1] +
	       w[51] * derivs_[cd.node6_index()][2] +
	       w[52] * derivs_[cd.node6_index()][3] +
	       w[53] * derivs_[cd.node6_index()][4] +
	       w[54] * derivs_[cd.node6_index()][5] +
	       w[55] * derivs_[cd.node6_index()][6] +
	       w[56] * cd.node7()		    +
	       w[57] * derivs_[cd.node7_index()][0] +
	       w[58] * derivs_[cd.node7_index()][1] +
	       w[59] * derivs_[cd.node7_index()][2] +
	       w[60] * derivs_[cd.node7_index()][3] +
	       w[61] * derivs_[cd.node7_index()][4] +
	       w[62] * derivs_[cd.node7_index()][5] +
	       w[63] * derivs_[cd.node7_index()][6]);
  }
  
  //! get first derivative at parametric coordinate
  template <class CellData>
  void derivate(const vector<double> &coords, const CellData &cd,
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
    derivs.clear();

    derivs[0]=
      T(6*(-1 + x)*x*y12*(1 + 2*y)*z12*(1 + 2*z)*cd.node0()
	+(1 - 4*x + 3*x2)*y12*(1 + 2*y)*z12*(1 + 2*z)*derivs_[cd.node0_index()][0]
	+6*(-1 + x)*x*y12*y*z12*(1 + 2*z)*derivs_[cd.node0_index()][1]
	+6*(-1 + x)*x*y12*(1 + 2*y)*z12*z*derivs_[cd.node0_index()][2]
	+(1 - 4*x + 3*x2)*y12*y*z12*(1 + 2*z)*derivs_[cd.node0_index()][3]
	+6*(-1 + x)*x*y12*y*z12*z*derivs_[cd.node0_index()][4]
	+(1 - 4*x + 3*x2)*y12*(1 + 2*y)*z12*z*derivs_[cd.node0_index()][5]
	+(1 - 4*x + 3*x2)*y12*y*z12*z*derivs_[cd.node0_index()][6]
	-6*(-1 + x)*x*y12*(1 + 2*y)*z12*(1 + 2*z)*cd.node1()
	+x*(-2 + 3*x)*y12*(1 + 2*y)*z12*(1 + 2*z)*derivs_[cd.node1_index()][0]
	-6*(-1 + x)*x*y12*y*z12*(1 + 2*z)*derivs_[cd.node1_index()][1]
	-6*(-1 + x)*x*y12*(1 + 2*y)*z12*z*derivs_[cd.node1_index()][2]
	+x*(-2 + 3*x)*y12*y*z12*(1 + 2*z)*derivs_[cd.node1_index()][3]
	-6*(-1 + x)*x*y12*y*z12*z*derivs_[cd.node1_index()][4]
	+x*(-2 + 3*x)*y12*(1 + 2*y)*z12*z*derivs_[cd.node1_index()][5]
	+x*(-2 + 3*x)*y12*y*z12*z*derivs_[cd.node1_index()][6]
	+6*(-1 + x)*x*y2*(-3 + 2*y)*z12*(1 + 2*z)*cd.node2()
	-(x*(-2 + 3*x)*y2*(-3 + 2*y)*z12*(1 + 2*z))*derivs_[cd.node2_index()][0]
	-6*(-1 + x)*x*(-1 + y)*y2*z12*(1 + 2*z)*derivs_[cd.node2_index()][1]
	+6*(-1 + x)*x*y2*(-3 + 2*y)*z12*z*derivs_[cd.node2_index()][2]
	+x*(-2 + 3*x)*(-1 + y)*y2*z12*(1 + 2*z)*derivs_[cd.node2_index()][3]
	-6*(-1 + x)*x*(-1 + y)*y2*z12*z*derivs_[cd.node2_index()][4]
	-(x*(-2 + 3*x)*y2*(-3 + 2*y)*z12*z)*derivs_[cd.node2_index()][5]
	+x*(-2 + 3*x)*(-1 + y)*y2*z12*z*derivs_[cd.node2_index()][6]
	-6*(-1 + x)*x*y2*(-3 + 2*y)*z12*(1 + 2*z)*cd.node3()
	-((1 - 4*x + 3*x2)*y2*(-3 + 2*y)*z12*(1 + 2*z))*derivs_[cd.node3_index()][0]
	+6*(-1 + x)*x*(-1 + y)*y2*z12*(1 + 2*z)*derivs_[cd.node3_index()][2]
	-6*(-1 + x)*x*y2*(-3 + 2*y)*z12*z*derivs_[cd.node3_index()][3]
	+(1 - 4*x + 3*x2)*(-1 + y)*y2*z12*(1 + 2*z)*derivs_[cd.node3_index()][4]
	+6*(-1 + x)*x*(-1 + y)*y2*z12*z*derivs_[cd.node3_index()][5]
	-((1 - 4*x + 3*x2)*y2*(-3 + 2*y)*z12*z)*derivs_[cd.node3_index()][6]
	+(1 - 4*x + 3*x2)*(-1 + y)*y2*z12*z*derivs_[cd.node3_index()][7]
	-6*(-1 + x)*x*y12*(1 + 2*y)*z2*(-3 + 2*z)*cd.node4()
	-((1 - 4*x + 3*x2)*y12*(1 + 2*y)*z2*(-3 + 2*z))*derivs_[cd.node4_index()][0]
	-6*(-1 + x)*x*y12*y*z2*(-3 + 2*z)*derivs_[cd.node4_index()][1]
	+6*(-1 + x)*x*y12*(1 + 2*y)*(-1 + z)*z2*derivs_[cd.node4_index()][2]
	-((1 - 4*x + 3*x2)*y12*y*z2*(-3 + 2*z))*derivs_[cd.node4_index()][3]
	+6*(-1 + x)*x*y12*y*(-1 + z)*z2*derivs_[cd.node4_index()][4]
	+(1 - 4*x + 3*x2)*y12*(1 + 2*y)*(-1 + z)*z2*derivs_[cd.node4_index()][5]
	+(1 - 4*x + 3*x2)*y12*y*(-1 + z)*z2*derivs_[cd.node4_index()][6]
	+6*(-1 + x)*x*y12*(1 + 2*y)*z2*(-3 + 2*z)*cd.node5()
	-(x*(-2 + 3*x)*y12*(1 + 2*y)*z2*(-3 + 2*z))*derivs_[cd.node5_index()][0]
	+6*(-1 + x)*x*y12*y*z2*(-3 + 2*z)*derivs_[cd.node5_index()][1]
	-6*(-1 + x)*x*y12*(1 + 2*y)*(-1 + z)*z2*derivs_[cd.node5_index()][2]
	-(x*(-2 + 3*x)*y12*y*z2*(-3 + 2*z))*derivs_[cd.node5_index()][3]
	-6*(-1 + x)*x*y12*y*(-1 + z)*z2*derivs_[cd.node5_index()][4]
	+x*(-2 + 3*x)*y12*(1 + 2*y)*(-1 + z)*z2*derivs_[cd.node5_index()][5]
	+x*(-2 + 3*x)*y12*y*(-1 + z)*z2*derivs_[cd.node5_index()][6]
	-6*(-1 + x)*x*y2*(-3 + 2*y)*z2*(-3 + 2*z)*cd.node6()
	+x*(-2 + 3*x)*y2*(-3 + 2*y)*z2*(-3 + 2*z)*derivs_[cd.node6_index()][0]
	+6*(-1 + x)*x*(-1 + y)*y2*z2*(-3 + 2*z)*derivs_[cd.node6_index()][1]
	+6*(-1 + x)*x*y2*(-3 + 2*y)*(-1 + z)*z2*derivs_[cd.node6_index()][2]
	-(x*(-2 + 3*x)*(-1 + y)*y2*z2*(-3 + 2*z))*derivs_[cd.node6_index()][3]
	-6*(-1 + x)*x*(-1 + y)*y2*(-1 + z)*z2*derivs_[cd.node6_index()][4]
	-(x*(-2 + 3*x)*y2*(-3 + 2*y)*(-1 + z)*z2)*derivs_[cd.node6_index()][5]
	+x*(-2 + 3*x)*(-1 + y)*y2*(-1 + z)*z2*derivs_[cd.node6_index()][6]
	+6*(-1 + x)*x*y2*(-3 +  2*y)*z2*(-3 + 2*z)*cd.node7()
	+(1 - 4*x + 3*x2)*y2*(-3 + 2*y)*z2*(-3 + 2*z)*derivs_[cd.node7_index()][0]
	-6*(-1 + x)*x*(-1 + y)*y2*z2*(-3 + 2*z)*derivs_[cd.node7_index()][1]
	-6*(-1 + x)*x*y2*(-3 + 2*y)*(-1 + z)*z2*derivs_[cd.node7_index()][2]
	-((1 - 4*x + 3*x2)*(-1 + y)*y2*z2*(-3 + 2*z))*derivs_[cd.node7_index()][3]
	+6*(-1 + x)*x*(-1 + y)*y2*(-1 + z)*z2*derivs_[cd.node7_index()][4]
	-((1 - 4*x + 3*x2)*y2*(-3 + 2*y)*(-1 + z)*z2)*derivs_[cd.node7_index()][5]
	+(1 - 4*x + 3*x2)*(-1 + y)*y2*(-1 + z)*z2*derivs_[cd.node7_index()][6]);
      
    derivs[1]=
      T(6*x12*(1 + 2*x)*(-1 + y)*y*z12*(1 + 2*z)*cd.node0()
	+6*x12*x*(-1 + y)*y*z12*(1 + 2*z)*derivs_[cd.node0_index()][0]
	+x12*(1 + 2*x)*(1 - 4*y + 3*y2)*z12*(1 + 2*z)*derivs_[cd.node0_index()][1]
	+6*x12*(1 + 2*x)*(-1 + y)*y*z12*z*derivs_[cd.node0_index()][2]
	+x12*x*(1 - 4*y + 3*y2)*z12*(1 + 2*z)*derivs_[cd.node0_index()][3]
	+x12*(1 + 2*x)*(1 - 4*y + 3*y2)*z12*z*derivs_[cd.node0_index()][4]
	+6*x12*x*(-1 + y)*y*z12*z*derivs_[cd.node0_index()][5]
	+x12*x*(1 - 4*y + 3*y2)*z12*z*derivs_[cd.node0_index()][6]
	-6*x2*(-3 + 2*x)*(-1 + y)*y*z12*(1 + 2*z)*cd.node1()
	+6*(-1 + x)*x2*(-1 + y)*y*z12*(1 + 2*z)*derivs_[cd.node1_index()][0]
	-(x2*(-3 + 2*x)*(1 - 4*y + 3*y2)*z12*(1 + 2*z))*derivs_[cd.node1_index()][1]
	-6*x2*(-3 + 2*x)*(-1 + y)*y*z12*z*derivs_[cd.node1_index()][2]
	+(-1 + x)*x2*(1 - 4*y + 3*y2)*z12*(1 + 2*z)*derivs_[cd.node1_index()][3]
	-(x2*(-3 + 2*x)*(1 - 4*y + 3*y2)*z12*z)*derivs_[cd.node1_index()][4]
	+6*(-1 + x)*x2*(-1 + y)*y*z12*z*derivs_[cd.node1_index()][5]
	+(-1 + x)*x2*(1 - 4*y + 3*y2)*z12*z*derivs_[cd.node1_index()][6]
	+6*x2*(-3 + 2*x)*(-1 + y)*y*z12*(1 + 2*z)*cd.node2()
	-6*(-1 + x)*x2*(-1 + y)*y*z12*(1 + 2*z)*derivs_[cd.node2_index()][0]
	-(x2*(-3 + 2*x)*y*(-2 + 3*y)*z12*(1 + 2*z))*derivs_[cd.node2_index()][1]
	+6*x2*(-3 + 2*x)*(-1 + y)*y*z12*z*derivs_[cd.node2_index()][2]
	+(-1 + x)*x2*y*(-2 + 3*y)*z12*(1 + 2*z)*derivs_[cd.node2_index()][3]
	-(x2*(-3 + 2*x)*y*(-2 + 3*y)*z12*z)*derivs_[cd.node2_index()][4]
	-6*(-1 + x)*x2*(-1 + y)*y*z12*z*derivs_[cd.node2_index()][5]
	+(-1 + x)*x2*y*(-2 + 3*y)*z12*z*derivs_[cd.node2_index()][6]
	-6*x12*(1 + 2*x)*(-1 + y)*y*z12*(1 + 2*z)*cd.node3()
	-6*x12*x*(-1 + y)*y*z12*(1 + 2*z)*derivs_[cd.node3_index()][0]
	+x12*(1 + 2*x)*y*(-2 + 3*y)*z12*(1 + 2*z)*derivs_[cd.node3_index()][1]
	-6*x12*(1 + 2*x)*(-1 + y)*y*z12*z*derivs_[cd.node3_index()][2]
	+x12*x*y*(-2 + 3*y)*z12*(1 + 2*z)*derivs_[cd.node3_index()][3]
	+x12*(1 + 2*x)*y*(-2 + 3*y)*z12*z*derivs_[cd.node3_index()][4]
	-6*x12*x*(-1 + y)*y*z12*z*derivs_[cd.node3_index()][5]
	+x12*x*y*(-2 + 3*y)*z12*z*derivs_[cd.node3_index()][6]
	-6*x12*(1 + 2*x)*(-1 + y)*y*z2*(-3 + 2*z)*cd.node4()
	-6*x12*x*(-1 + y)*y*z2*(-3 + 2*z)*derivs_[cd.node4_index()][0]
	-(x12*(1 + 2*x)*(1 - 4*y + 3*y2)*z2*(-3 + 2*z))*derivs_[cd.node4_index()][1]
	+6*x12*(1 + 2*x)*(-1 + y)*y*(-1 + z)*z2*derivs_[cd.node4_index()][2]
	-(x12*x*(1 - 4*y + 3*y2)*z2*(-3 + 2*z))*derivs_[cd.node4_index()][3]
	+x12*(1 + 2*x)*(1 - 4*y + 3*y2)*(-1 + z)*z2*derivs_[cd.node4_index()][4]
	+6*x12*x*(-1 + y)*y*(-1 + z)*z2*derivs_[cd.node4_index()][5]
	+x12*x*(1 - 4*y + 3*y2)*(-1 + z)*z2*derivs_[cd.node4_index()][6]
	+6*x2*(-3 + 2*x)*(-1 + y)*y*z2*(-3 + 2*z)*cd.node5()
	-6*(-1 + x)*x2*(-1 + y)*y*z2*(-3 + 2*z)*derivs_[cd.node5_index()][0]
	+x2*(-3 + 2*x)*(1 - 4*y + 3*y2)*z2*(-3 + 2*z)*derivs_[cd.node5_index()][1]
	-6*x2*(-3 + 2*x)*(-1 + y)*y*(-1 + z)*z2*derivs_[cd.node5_index()][2]
	-((-1 + x)*x2*(1 - 4*y + 3*y2)*z2*(-3 + 2*z))*derivs_[cd.node5_index()][3]
	-(x2*(-3 + 2*x)*(1 - 4*y + 3*y2)*(-1 + z)*z2)*derivs_[cd.node5_index()][4]
	+6*(-1 + x)*x2*(-1 + y)*y*(-1 + z)*z2*derivs_[cd.node5_index()][5]
	+(-1 + x)*x2*(1 - 4*y + 3*y2)*(-1 + z)*z2*derivs_[cd.node5_index()][6]
	-6*x2*(-3 + 2*x)*(-1 + y)*y*z2*(-3 + 2*z)*cd.node6()
	+6*(-1 + x)*x2*(-1 + y)*y*z2*(-3 + 2*z)*derivs_[cd.node6_index()][0]
	+x2*(-3 + 2*x)*y*(-2 + 3*y)*z2*(-3 + 2*z)*derivs_[cd.node6_index()][1]
	+6*x2*(-3 + 2*x)*(-1 + y)*y*(-1 + z)*z2*derivs_[cd.node6_index()][2]
	-((-1 + x)*x2*y*(-2 + 3*y)*z2*(-3 + 2*z))*derivs_[cd.node6_index()][3]
	-(x2*(-3 + 2*x)*y*(-2 + 3*y)*(-1 + z)*z2)*derivs_[cd.node6_index()][4]
	-6*(-1 + x)*x2*(-1 + y)*y*(-1 + z)*z2*derivs_[cd.node6_index()][5]
	+(-1 + x)*x2*y*(-2 + 3*y)*(-1 + z)*z2*derivs_[cd.node6_index()][6]
	+6*x12*(1 + 2*x)*(-1 + y)*y*z2*(-3 + 2*z)*cd.node7()
	+6*x12*x*(-1 + y)*y*z2*(-3 + 2*z)*derivs_[cd.node7_index()][0]
	-(x12*(1 + 2*x)*y*(-2 + 3*y)*z2*(-3 + 2*z))*derivs_[cd.node7_index()][1]
	-6*x12*(1 + 2*x)*(-1 + y)*y*(-1 + z)*z2*derivs_[cd.node7_index()][2]
	-(x12*x*y*(-2 + 3*y)*z2*(-3 + 2*z))*derivs_[cd.node7_index()][3]
	+x12*(1 + 2*x)*y*(-2 + 3*y)*(-1 + z)*z2*derivs_[cd.node7_index()][4]
	-6*x12*x*(-1 + y)*y*(-1 + z)*z2*derivs_[cd.node7_index()][5]
	+x12*x*y*(-2 + 3*y)*(-1 + z)*z2*derivs_[cd.node7_index()][6]);
      
    derivs[2]=
      T(6*x12*(1 + 2*x)*y12*(1 + 2*y)*(-1 + z)*z*cd.node0()
	+6*x12*x*y12*(1 + 2*y)*(-1 + z)*z*derivs_[cd.node0_index()][0]
	+6*x12*(1 + 2*x)*y12*y*(-1 + z)*z*derivs_[cd.node0_index()][1]
	+x12*(1 + 2*x)*y12*(1 + 2*y)*(1 - 4*z + 3*z2)*derivs_[cd.node0_index()][2]
	+6*x12*x*y12*y*(-1 + z)*z*derivs_[cd.node0_index()][3]
	+x12*(1 + 2*x)*y12*y*(1 - 4*z + 3*z2)*derivs_[cd.node0_index()][4]
	+x12*x*y12*(1 + 2*y)*(1 - 4*z + 3*z2)*derivs_[cd.node0_index()][5]
	+x12*x*y12*y*(1 - 4*z + 3*z2)*derivs_[cd.node0_index()][6]
	-6*x2*(-3 + 2*x)*y12*(1 + 2*y)*(-1 + z)*z*cd.node1()
	+6*(-1 + x)*x2*y12*(1 + 2*y)*(-1 + z)*z*derivs_[cd.node1_index()][0]
	-6*x2*(-3 + 2*x)*y12*y*(-1 + z)*z*derivs_[cd.node1_index()][1]
	-(x2*(-3 + 2*x)*y12*(1 + 2*y)*(1 - 4*z + 3*z2))*derivs_[cd.node1_index()][2]
	+6*(-1 + x)*x2*y12*y*(-1 + z)*z*derivs_[cd.node1_index()][3]
	-(x2*(-3 + 2*x)*y12*y*(1 - 4*z + 3*z2))*derivs_[cd.node1_index()][4]
	+(-1 + x)*x2*y12*(1 + 2*y)*(1 - 4*z + 3*z2)*derivs_[cd.node1_index()][5]
	+(-1 + x)*x2*y12*y*(1 - 4*z + 3*z2)*derivs_[cd.node1_index()][6]
	+6*x2*(-3 + 2*x)*y2*(-3 + 2*y)*(-1 + z)*z*cd.node2()
	-6*(-1 + x)*x2*y2*(-3 + 2*y)*(-1 + z)*z*derivs_[cd.node2_index()][0]
	-6*x2*(-3 + 2*x)*(-1 + y)*y2*(-1 + z)*z*derivs_[cd.node2_index()][1]
	+x2*(-3 + 2*x)*y2*(-3 + 2*y)*(1 - 4*z + 3*z2)*derivs_[cd.node2_index()][2]
	+6*(-1 + x)*x2*(-1 + y)*y2*(-1 + z)*z*derivs_[cd.node2_index()][3]
	-(x2*(-3 + 2*x)*(-1 + y)*y2*(1 - 4*z + 3*z2))*derivs_[cd.node2_index()][4]
	-((-1 + x)*x2*y2*(-3 + 2*y)*(1 - 4*z + 3*z2))*derivs_[cd.node2_index()][5]
	+(-1 + x)*x2*(-1 + y)*y2*(1 - 4*z + 3*z2)*derivs_[cd.node2_index()][6]
	-6*x12*(1 + 2*x)*y2*(-3 + 2*y)*(-1 + z)*z*cd.node3()
	-6*x12*x*y2*(-3 + 2*y)*(-1 + z)*z*derivs_[cd.node3_index()][0]
	+6*x12*(1 + 2*x)*(-1 + y)*y2*(-1 + z)*z*derivs_[cd.node3_index()][1]
	-(x12*(1 + 2*x)*y2*(-3 + 2*y)*(1 - 4*z + 3*z2))*derivs_[cd.node3_index()][2]
	+6*x12*x*(-1 + y)*y2*(-1 + z)*z*derivs_[cd.node3_index()][3]
	+x12*(1 + 2*x)*(-1 + y)*y2*(1 - 4*z + 3*z2)*derivs_[cd.node3_index()][4]
	-(x12*x*y2*(-3 + 2*y)*(1 - 4*z + 3*z2))*derivs_[cd.node3_index()][5]
	+x12*x*(-1 + y)*y2*(1 - 4*z + 3*z2)*derivs_[cd.node3_index()][6]
	-6*x12*(1 + 2*x)*y12*(1 + 2*y)*(-1 + z)*z*cd.node4()
	-6*x12*x*y12*(1 + 2*y)*(-1 + z)*z*derivs_[cd.node4_index()][0]
	-6*x12*(1 + 2*x)*y12*y*(-1 + z)*z*derivs_[cd.node4_index()][1]
	+x12*(1 + 2*x)*y12*(1 + 2*y)*z*(-2 + 3*z)*derivs_[cd.node4_index()][2]
	-6*x12*x*y12*y*(-1 + z)*z*derivs_[cd.node4_index()][3]
	+x12*(1 + 2*x)*y12*y*z*(-2 + 3*z)*derivs_[cd.node4_index()][4]
	+x12*x*y12*(1 + 2*y)*z*(-2 + 3*z)*derivs_[cd.node4_index()][5]
	+x12*x*y12*y*z*(-2 + 3*z)*derivs_[cd.node4_index()][6]
	+6*x2*(-3 + 2*x)*y12*(1 + 2*y)*(-1 + z)*z*cd.node5()
	-6*(-1 + x)*x2*y12*(1 + 2*y)*(-1 + z)*z*derivs_[cd.node5_index()][0]
	+6*x2*(-3 + 2*x)*y12*y*(-1 + z)*z*derivs_[cd.node5_index()][1]
	-(x2*(-3 + 2*x)*y12*(1 + 2*y)*z*(-2 + 3*z))*derivs_[cd.node5_index()][2]
	-6*(-1 + x)*x2*y12*y*(-1 + z)*z*derivs_[cd.node5_index()][3]
	-(x2*(-3 + 2*x)*y12*y*z*(-2 + 3*z))*derivs_[cd.node5_index()][4]
	+(-1 + x)*x2*y12*(1 + 2*y)*z*(-2 + 3*z)*derivs_[cd.node5_index()][5]
	+(-1 + x)*x2*y12*y*z*(-2 + 3*z)*derivs_[cd.node5_index()][6]
	-6*x2*(-3 + 2*x)*y2*(-3 + 2*y)*(-1 + z)*z*cd.node6()
	+6*(-1 + x)*x2*y2*(-3 + 2*y)*(-1 + z)*z*derivs_[cd.node6_index()][0]
	+6*x2*(-3 + 2*x)*(-1 + y)*y2*(-1 + z)*z*derivs_[cd.node6_index()][1]
	+x2*(-3 + 2*x)*y2*(-3 + 2*y)*z*(-2 + 3*z)*derivs_[cd.node6_index()][2]
	-6*(-1 + x)*x2*(-1 + y)*y2*(-1 + z)*z*derivs_[cd.node6_index()][3]
	-(x2*(-3 + 2*x)*(-1 + y)*y2*z*(-2 + 3*z))*derivs_[cd.node6_index()][4]
	-((-1 + x)*x2*y2*(-3 + 2*y)*z*(-2 + 3*z))*derivs_[cd.node6_index()][5]
	+(-1 + x)*x2*(-1 + y)*y2*z*(-2 + 3*z)*derivs_[cd.node6_index()][6]
	+6*x12*(1 + 2*x)*y2*(-3 + 2*y)*(-1 + z)*z*cd.node7()
	+6*x12*x*y2*(-3 + 2*y)*(-1 + z)*z*derivs_[cd.node7_index()][0]
	-6*x12*(1 + 2*x)*(-1 + y)*y2*(-1 + z)*z*derivs_[cd.node7_index()][1]
	-(x12*(1 + 2*x)*y2*(-3 + 2*y)*z*(-2 + 3*z))*derivs_[cd.node7_index()][2]
	-6*x12*x*(-1 + y)*y2*(-1 + z)*z*derivs_[cd.node7_index()][3]
	+x12*(1 + 2*x)*(-1 + y)*y2*z*(-2 + 3*z)*derivs_[cd.node7_index()][4]
	-(x12*x*y2*(-3 + 2*y)*z*(-2 + 3*z))*derivs_[cd.node7_index()][5]
	+x12*x*(-1 + y)*y2*z*(-2 + 3*z)*derivs_[cd.node7_index()][6]);
  }
  

  //! get parametric coordinate for value within the element
  template <class CellData>
  bool get_coords(vector<double> &coords, const T& value, 
		  const CellData &cd) const  
  {
    HexLocate< HexTricubicHmt<T> > CL;
    return CL.get_coords(this, coords, value, cd);
  };
    
  //! add derivative values (dx, dy, dz, dxy, dyz, dzx, dxyz) for nodes.
  void add_derivatives(const vector<T> &p) { derivs_.push_back(p); }

  static  const string type_name(int n = -1);
  virtual void io (Piostream& str);

protected:
  //! support data (node data is elsewhere)
  vector<vector<T> >          derivs_; 
};

template <class T>
const string
HexTricubicHmt<T>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;
  }
  else if (n == 0)
  {
    static const string nm("HexTricubicHmt");
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
				string(__FILE__),
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
  Pio(stream, derivs_);
  stream.end_class();
}

} //namespace SCIRun

#endif // HexTricubicHmt_h
