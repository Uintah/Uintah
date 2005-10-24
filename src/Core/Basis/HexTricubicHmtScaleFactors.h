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
//    File   : HexTricubicHmtScaleFactors.h
//    Author : Martin Cole, Frank B. Sachse
//    Date   : Mar 1 2005

#if !defined(HexTricubicHmtScaleFactors_h)
#define HexTricubicHmtScaleFactors_h

#include <Core/Persistent/PersistentSTL.h>
#include <Core/Basis/HexTrilinearLgn.h>

namespace SCIRun {

//! Class for describing unit geometry of HexTricubicHmtScaleFactors
class HexTricubicHmtScaleFactorsUnitElement : 
    public HexTrilinearLgnUnitElement {
public:
  HexTricubicHmtScaleFactorsUnitElement() {};
  virtual ~HexTricubicHmtScaleFactorsUnitElement() {};
};


//! Class for handling of element of type hexahedron with 
//! tricubic hermitian interpolation with scale factors
template <class T>
class HexTricubicHmtScaleFactors : public BasisSimple<T>, 
                                   public HexApprox, 
				   public HexGaussian3<double>, 
				   public HexTricubicHmtScaleFactorsUnitElement
{
public:
  typedef T value_type;

  HexTricubicHmtScaleFactors() {}
  virtual ~HexTricubicHmtScaleFactors() {}
  
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
    const T sdx0=derivs_[cd.node0_index()][0]*scalefactors_[cd.node0_index()][0];
    const T sdx1=derivs_[cd.node1_index()][0]*scalefactors_[cd.node1_index()][0];
    const T sdx2=derivs_[cd.node2_index()][0]*scalefactors_[cd.node2_index()][0];
    const T sdx3=derivs_[cd.node3_index()][0]*scalefactors_[cd.node3_index()][0];
    const T sdx4=derivs_[cd.node4_index()][0]*scalefactors_[cd.node4_index()][0];
    const T sdx5=derivs_[cd.node5_index()][0]*scalefactors_[cd.node5_index()][0];
    const T sdx6=derivs_[cd.node6_index()][0]*scalefactors_[cd.node6_index()][0];
    const T sdx7=derivs_[cd.node7_index()][0]*scalefactors_[cd.node7_index()][0];
    const T sdy0=derivs_[cd.node0_index()][1]*scalefactors_[cd.node0_index()][1];
    const T sdy1=derivs_[cd.node1_index()][1]*scalefactors_[cd.node1_index()][1];
    const T sdy2=derivs_[cd.node2_index()][1]*scalefactors_[cd.node2_index()][1];
    const T sdy3=derivs_[cd.node3_index()][1]*scalefactors_[cd.node3_index()][1];
    const T sdy4=derivs_[cd.node4_index()][1]*scalefactors_[cd.node4_index()][1];
    const T sdy5=derivs_[cd.node5_index()][1]*scalefactors_[cd.node5_index()][1];
    const T sdy6=derivs_[cd.node6_index()][1]*scalefactors_[cd.node6_index()][1];
    const T sdy7=derivs_[cd.node7_index()][1]*scalefactors_[cd.node7_index()][1];
    const T sdz0=derivs_[cd.node0_index()][2]*scalefactors_[cd.node0_index()][2];
    const T sdz1=derivs_[cd.node1_index()][2]*scalefactors_[cd.node1_index()][2];
    const T sdz2=derivs_[cd.node2_index()][2]*scalefactors_[cd.node2_index()][2];
    const T sdz3=derivs_[cd.node3_index()][2]*scalefactors_[cd.node3_index()][2];
    const T sdz4=derivs_[cd.node4_index()][2]*scalefactors_[cd.node4_index()][2];
    const T sdz5=derivs_[cd.node5_index()][2]*scalefactors_[cd.node5_index()][2];
    const T sdz6=derivs_[cd.node6_index()][2]*scalefactors_[cd.node6_index()][2];
    const T sdz7=derivs_[cd.node7_index()][2]*scalefactors_[cd.node7_index()][2];

    const T sdxy0=derivs_[cd.node0_index()][3]*scalefactors_[cd.node0_index()][0]*scalefactors_[cd.node0_index()][1];
    const T sdxy1=derivs_[cd.node1_index()][3]*scalefactors_[cd.node1_index()][0]*scalefactors_[cd.node1_index()][1];
    const T sdxy2=derivs_[cd.node2_index()][3]*scalefactors_[cd.node2_index()][0]*scalefactors_[cd.node2_index()][1];
    const T sdxy3=derivs_[cd.node3_index()][3]*scalefactors_[cd.node3_index()][0]*scalefactors_[cd.node3_index()][1];
    const T sdxy4=derivs_[cd.node4_index()][3]*scalefactors_[cd.node4_index()][0]*scalefactors_[cd.node4_index()][1];
    const T sdxy5=derivs_[cd.node5_index()][3]*scalefactors_[cd.node5_index()][0]*scalefactors_[cd.node5_index()][1];
    const T sdxy6=derivs_[cd.node6_index()][3]*scalefactors_[cd.node6_index()][0]*scalefactors_[cd.node6_index()][1];
    const T sdxy7=derivs_[cd.node7_index()][3]*scalefactors_[cd.node7_index()][0]*scalefactors_[cd.node7_index()][1];

    const T sdyz0=derivs_[cd.node0_index()][4]*scalefactors_[cd.node0_index()][2]*scalefactors_[cd.node0_index()][1];
    const T sdyz1=derivs_[cd.node1_index()][4]*scalefactors_[cd.node1_index()][2]*scalefactors_[cd.node1_index()][1];
    const T sdyz2=derivs_[cd.node2_index()][4]*scalefactors_[cd.node2_index()][2]*scalefactors_[cd.node2_index()][1];
    const T sdyz3=derivs_[cd.node3_index()][4]*scalefactors_[cd.node3_index()][2]*scalefactors_[cd.node3_index()][1];
    const T sdyz4=derivs_[cd.node4_index()][4]*scalefactors_[cd.node4_index()][2]*scalefactors_[cd.node4_index()][1];
    const T sdyz5=derivs_[cd.node5_index()][4]*scalefactors_[cd.node5_index()][2]*scalefactors_[cd.node5_index()][1];
    const T sdyz6=derivs_[cd.node6_index()][4]*scalefactors_[cd.node6_index()][2]*scalefactors_[cd.node6_index()][1];
    const T sdyz7=derivs_[cd.node7_index()][4]*scalefactors_[cd.node7_index()][2]*scalefactors_[cd.node7_index()][1];

    const T sdxz0=derivs_[cd.node0_index()][5]*scalefactors_[cd.node0_index()][2]*scalefactors_[cd.node0_index()][0];
    const T sdxz1=derivs_[cd.node1_index()][5]*scalefactors_[cd.node1_index()][2]*scalefactors_[cd.node1_index()][0];
    const T sdxz2=derivs_[cd.node2_index()][5]*scalefactors_[cd.node2_index()][2]*scalefactors_[cd.node2_index()][0];
    const T sdxz3=derivs_[cd.node3_index()][5]*scalefactors_[cd.node3_index()][2]*scalefactors_[cd.node3_index()][0];
    const T sdxz4=derivs_[cd.node4_index()][5]*scalefactors_[cd.node4_index()][2]*scalefactors_[cd.node4_index()][0];
    const T sdxz5=derivs_[cd.node5_index()][5]*scalefactors_[cd.node5_index()][2]*scalefactors_[cd.node5_index()][0];
    const T sdxz6=derivs_[cd.node6_index()][5]*scalefactors_[cd.node6_index()][2]*scalefactors_[cd.node6_index()][0];
    const T sdxz7=derivs_[cd.node7_index()][5]*scalefactors_[cd.node7_index()][2]*scalefactors_[cd.node7_index()][0];

    const T sdxyz0=derivs_[cd.node0_index()][6]*scalefactors_[cd.node0_index()][2]*scalefactors_[cd.node0_index()][1]*scalefactors_[cd.node0_index()][0];
    const T sdxyz1=derivs_[cd.node1_index()][6]*scalefactors_[cd.node1_index()][2]*scalefactors_[cd.node1_index()][1]*scalefactors_[cd.node1_index()][0];
    const T sdxyz2=derivs_[cd.node2_index()][6]*scalefactors_[cd.node2_index()][2]*scalefactors_[cd.node2_index()][1]*scalefactors_[cd.node2_index()][0];
    const T sdxyz3=derivs_[cd.node3_index()][6]*scalefactors_[cd.node3_index()][2]*scalefactors_[cd.node3_index()][1]*scalefactors_[cd.node3_index()][0];
    const T sdxyz4=derivs_[cd.node4_index()][6]*scalefactors_[cd.node4_index()][2]*scalefactors_[cd.node4_index()][1]*scalefactors_[cd.node4_index()][0];
    const T sdxyz5=derivs_[cd.node5_index()][6]*scalefactors_[cd.node5_index()][2]*scalefactors_[cd.node5_index()][1]*scalefactors_[cd.node5_index()][0];
    const T sdxyz6=derivs_[cd.node6_index()][6]*scalefactors_[cd.node6_index()][2]*scalefactors_[cd.node6_index()][1]*scalefactors_[cd.node6_index()][0];
    const T sdxyz7=derivs_[cd.node7_index()][6]*scalefactors_[cd.node7_index()][2]*scalefactors_[cd.node7_index()][1]*scalefactors_[cd.node7_index()][0];

    return (T)(w[0]  * cd.node0()+
	       w[1]  * sdx0	 +
	       w[2]  * sdy0	 +
	       w[3]  * sdz0	 +
	       w[4]  * sdxy0	 +
	       w[5]  * sdyz0	 +
	       w[6]  * sdxz0	 +
	       w[7]  * sdxyz0	 +
	       w[8]  * cd.node1()+
	       w[9]  * sdx1	 +
	       w[10] * sdy1	 +
	       w[11] * sdz1	 +
	       w[12] * sdxy1	 +
	       w[13] * sdyz1  	 +
	       w[14] * sdxz1	 +
	       w[15] * sdxyz1	 +
	       w[16] * cd.node2()+
	       w[17] * sdx2	 +
	       w[18] * sdy2	 +
	       w[19] * sdz2	 +
	       w[20] * sdxy2	 +
	       w[21] * sdyz2	 +
	       w[22] * sdxz2	 +
	       w[23] * sdxyz2 	 +
	       w[24] * cd.node3()+
	       w[25] * sdx3	 +
	       w[26] * sdy3	 +
	       w[27] * sdz3	 +
	       w[28] * sdxy3	 +
	       w[29] * sdyz3	 +
	       w[30] * sdxz3	 +
	       w[31] * sdxyz3	 +
	       w[32] * cd.node4()+
	       w[33] * sdx4	 +
	       w[34] * sdy4	 +
	       w[35] * sdz4	 +
	       w[36] * sdxy4	 +
	       w[37] * sdyz4	 +
	       w[38] * sdxz4	 +
	       w[39] * sdxyz4	 +
	       w[40] * cd.node5()+
	       w[41] * sdx5	 +
	       w[42] * sdy5	 +
	       w[43] * sdz5	 +
	       w[44] * sdxy5	 +
	       w[45] * sdyz5	 +
	       w[46] * sdxz5	 +
	       w[47] * sdxyz5	 +
	       w[48] * cd.node6()+
	       w[49] * sdx6	 +
	       w[50] * sdy6	 +
	       w[51] * sdz6	 +
	       w[52] * sdxy6	 +
	       w[53] * sdyz6	 +
	       w[54] * sdxz6	 +
	       w[55] * sdxyz6	 +
	       w[56] * cd.node7()+
	       w[57] * sdx7	 +
	       w[58] * sdy7	 +
	       w[59] * sdz7	 +
	       w[60] * sdxy7	 +
	       w[61] * sdyz7	 +
	       w[62] * sdxz7	 +
	       w[63] * sdxyz7);
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

    const T sdx0=derivs_[cd.node0_index()][0]*scalefactors_[cd.node0_index()][0];
    const T sdx1=derivs_[cd.node1_index()][0]*scalefactors_[cd.node1_index()][0];
    const T sdx2=derivs_[cd.node2_index()][0]*scalefactors_[cd.node2_index()][0];
    const T sdx3=derivs_[cd.node3_index()][0]*scalefactors_[cd.node3_index()][0];
    const T sdx4=derivs_[cd.node4_index()][0]*scalefactors_[cd.node4_index()][0];
    const T sdx5=derivs_[cd.node5_index()][0]*scalefactors_[cd.node5_index()][0];
    const T sdx6=derivs_[cd.node6_index()][0]*scalefactors_[cd.node6_index()][0];
    const T sdx7=derivs_[cd.node7_index()][0]*scalefactors_[cd.node7_index()][0];

    const T sdy0=derivs_[cd.node0_index()][1]*scalefactors_[cd.node0_index()][1];
    const T sdy1=derivs_[cd.node1_index()][1]*scalefactors_[cd.node1_index()][1];
    const T sdy2=derivs_[cd.node2_index()][1]*scalefactors_[cd.node2_index()][1];
    const T sdy3=derivs_[cd.node3_index()][1]*scalefactors_[cd.node3_index()][1];
    const T sdy4=derivs_[cd.node4_index()][1]*scalefactors_[cd.node4_index()][1];
    const T sdy5=derivs_[cd.node5_index()][1]*scalefactors_[cd.node5_index()][1];
    const T sdy6=derivs_[cd.node6_index()][1]*scalefactors_[cd.node6_index()][1];
    const T sdy7=derivs_[cd.node7_index()][1]*scalefactors_[cd.node7_index()][1];

    const T sdz0=derivs_[cd.node0_index()][2]*scalefactors_[cd.node0_index()][2];
    const T sdz1=derivs_[cd.node1_index()][2]*scalefactors_[cd.node1_index()][2];
    const T sdz2=derivs_[cd.node2_index()][2]*scalefactors_[cd.node2_index()][2];
    const T sdz3=derivs_[cd.node3_index()][2]*scalefactors_[cd.node3_index()][2];
    const T sdz4=derivs_[cd.node4_index()][2]*scalefactors_[cd.node4_index()][2];
    const T sdz5=derivs_[cd.node5_index()][2]*scalefactors_[cd.node5_index()][2];
    const T sdz6=derivs_[cd.node6_index()][2]*scalefactors_[cd.node6_index()][2];
    const T sdz7=derivs_[cd.node7_index()][2]*scalefactors_[cd.node7_index()][2];

    const T sdxy0=derivs_[cd.node0_index()][3]*scalefactors_[cd.node0_index()][0]*scalefactors_[cd.node0_index()][1];
    const T sdxy1=derivs_[cd.node1_index()][3]*scalefactors_[cd.node1_index()][0]*scalefactors_[cd.node1_index()][1];
    const T sdxy2=derivs_[cd.node2_index()][3]*scalefactors_[cd.node2_index()][0]*scalefactors_[cd.node2_index()][1];
    const T sdxy3=derivs_[cd.node3_index()][3]*scalefactors_[cd.node3_index()][0]*scalefactors_[cd.node3_index()][1];
    const T sdxy4=derivs_[cd.node4_index()][3]*scalefactors_[cd.node4_index()][0]*scalefactors_[cd.node4_index()][1];
    const T sdxy5=derivs_[cd.node5_index()][3]*scalefactors_[cd.node5_index()][0]*scalefactors_[cd.node5_index()][1];
    const T sdxy6=derivs_[cd.node6_index()][3]*scalefactors_[cd.node6_index()][0]*scalefactors_[cd.node6_index()][1];
    const T sdxy7=derivs_[cd.node7_index()][3]*scalefactors_[cd.node7_index()][0]*scalefactors_[cd.node7_index()][1];

    const T sdyz0=derivs_[cd.node0_index()][4]*scalefactors_[cd.node0_index()][2]*scalefactors_[cd.node0_index()][1];
    const T sdyz1=derivs_[cd.node1_index()][4]*scalefactors_[cd.node1_index()][2]*scalefactors_[cd.node1_index()][1];
    const T sdyz2=derivs_[cd.node2_index()][4]*scalefactors_[cd.node2_index()][2]*scalefactors_[cd.node2_index()][1];
    const T sdyz3=derivs_[cd.node3_index()][4]*scalefactors_[cd.node3_index()][2]*scalefactors_[cd.node3_index()][1];
    const T sdyz4=derivs_[cd.node4_index()][4]*scalefactors_[cd.node4_index()][2]*scalefactors_[cd.node4_index()][1];
    const T sdyz5=derivs_[cd.node5_index()][4]*scalefactors_[cd.node5_index()][2]*scalefactors_[cd.node5_index()][1];
    const T sdyz6=derivs_[cd.node6_index()][4]*scalefactors_[cd.node6_index()][2]*scalefactors_[cd.node6_index()][1];
    const T sdyz7=derivs_[cd.node7_index()][4]*scalefactors_[cd.node7_index()][2]*scalefactors_[cd.node7_index()][1];

    const T sdxz0=derivs_[cd.node0_index()][5]*scalefactors_[cd.node0_index()][2]*scalefactors_[cd.node0_index()][0];
    const T sdxz1=derivs_[cd.node1_index()][5]*scalefactors_[cd.node1_index()][2]*scalefactors_[cd.node1_index()][0];
    const T sdxz2=derivs_[cd.node2_index()][5]*scalefactors_[cd.node2_index()][2]*scalefactors_[cd.node2_index()][0];
    const T sdxz3=derivs_[cd.node3_index()][5]*scalefactors_[cd.node3_index()][2]*scalefactors_[cd.node3_index()][0];
    const T sdxz4=derivs_[cd.node4_index()][5]*scalefactors_[cd.node4_index()][2]*scalefactors_[cd.node4_index()][0];
    const T sdxz5=derivs_[cd.node5_index()][5]*scalefactors_[cd.node5_index()][2]*scalefactors_[cd.node5_index()][0];
    const T sdxz6=derivs_[cd.node6_index()][5]*scalefactors_[cd.node6_index()][2]*scalefactors_[cd.node6_index()][0];
    const T sdxz7=derivs_[cd.node7_index()][5]*scalefactors_[cd.node7_index()][2]*scalefactors_[cd.node7_index()][0];

    const T sdxyz0=derivs_[cd.node0_index()][6]*scalefactors_[cd.node0_index()][2]*scalefactors_[cd.node0_index()][1]*scalefactors_[cd.node0_index()][0];
    const T sdxyz1=derivs_[cd.node1_index()][6]*scalefactors_[cd.node1_index()][2]*scalefactors_[cd.node1_index()][1]*scalefactors_[cd.node1_index()][0];
    const T sdxyz2=derivs_[cd.node2_index()][6]*scalefactors_[cd.node2_index()][2]*scalefactors_[cd.node2_index()][1]*scalefactors_[cd.node2_index()][0];
    const T sdxyz3=derivs_[cd.node3_index()][6]*scalefactors_[cd.node3_index()][2]*scalefactors_[cd.node3_index()][1]*scalefactors_[cd.node3_index()][0];
    const T sdxyz4=derivs_[cd.node4_index()][6]*scalefactors_[cd.node4_index()][2]*scalefactors_[cd.node4_index()][1]*scalefactors_[cd.node4_index()][0];
    const T sdxyz5=derivs_[cd.node5_index()][6]*scalefactors_[cd.node5_index()][2]*scalefactors_[cd.node5_index()][1]*scalefactors_[cd.node5_index()][0];
    const T sdxyz6=derivs_[cd.node6_index()][6]*scalefactors_[cd.node6_index()][2]*scalefactors_[cd.node6_index()][1]*scalefactors_[cd.node6_index()][0];
    const T sdxyz7=derivs_[cd.node7_index()][6]*scalefactors_[cd.node7_index()][2]*scalefactors_[cd.node7_index()][1]*scalefactors_[cd.node7_index()][0];

    derivs[0]=
      T(6*(-1 + x)*x*y12*(1 + 2*y)*z12*(1 + 2*z)*cd.node0()
	+(1 - 4*x + 3*x2)*y12*(1 + 2*y)*z12*(1 + 2*z)*sdx0
	+6*(-1 + x)*x*y12*y*z12*(1 + 2*z)*sdy0
	+6*(-1 + x)*x*y12*(1 + 2*y)*z12*z*sdz0
	+(1 - 4*x + 3*x2)*y12*y*z12*(1 + 2*z)*sdxy0
	+6*(-1 + x)*x*y12*y*z12*z*sdyz0
	+(1 - 4*x + 3*x2)*y12*(1 + 2*y)*z12*z*sdxz0
	+(1 - 4*x + 3*x2)*y12*y*z12*z*sdxyz0
	-6*(-1 + x)*x*y12*(1 + 2*y)*z12*(1 + 2*z)*cd.node1()
	+x*(-2 + 3*x)*y12*(1 + 2*y)*z12*(1 + 2*z)*sdx1
	-6*(-1 + x)*x*y12*y*z12*(1 + 2*z)*sdy1
	-6*(-1 + x)*x*y12*(1 + 2*y)*z12*z*sdz1
	+x*(-2 + 3*x)*y12*y*z12*(1 + 2*z)*sdxy1
	-6*(-1 + x)*x*y12*y*z12*z*sdyz1
	+x*(-2 + 3*x)*y12*(1 + 2*y)*z12*z*sdxz1
	+x*(-2 + 3*x)*y12*y*z12*z*sdxyz1
	+6*(-1 + x)*x*y2*(-3 + 2*y)*z12*(1 + 2*z)*cd.node2()
	-(x*(-2 + 3*x)*y2*(-3 + 2*y)*z12*(1 + 2*z))*sdx2
	-6*(-1 + x)*x*(-1 + y)*y2*z12*(1 + 2*z)*sdy2
	+6*(-1 + x)*x*y2*(-3 + 2*y)*z12*z*sdz2
	+x*(-2 + 3*x)*(-1 + y)*y2*z12*(1 + 2*z)*sdxy2
	-6*(-1 + x)*x*(-1 + y)*y2*z12*z*sdyz2
	-(x*(-2 + 3*x)*y2*(-3 + 2*y)*z12*z)*sdxz2
	+x*(-2 + 3*x)*(-1 + y)*y2*z12*z*sdxyz2
	-6*(-1 + x)*x*y2*(-3 + 2*y)*z12*(1 + 2*z)*cd.node3()
	-((1 - 4*x + 3*x2)*y2*(-3 + 2*y)*z12*(1 + 2*z))*sdx3
	+6*(-1 + x)*x*(-1 + y)*y2*z12*(1 + 2*z)*sdy3
	-6*(-1 + x)*x*y2*(-3 + 2*y)*z12*z*sdz3
	+(1 - 4*x + 3*x2)*(-1 + y)*y2*z12*(1 + 2*z)*sdxy3
	+6*(-1 + x)*x*(-1 + y)*y2*z12*z*sdyz3
	-((1 - 4*x + 3*x2)*y2*(-3 + 2*y)*z12*z)*sdxz3
	+(1 - 4*x + 3*x2)*(-1 + y)*y2*z12*z*sdxyz3
	-6*(-1 + x)*x*y12*(1 + 2*y)*z2*(-3 + 2*z)*cd.node4()
	-((1 - 4*x + 3*x2)*y12*(1 + 2*y)*z2*(-3 + 2*z))*sdx4
	-6*(-1 + x)*x*y12*y*z2*(-3 + 2*z)*sdy4
	+6*(-1 + x)*x*y12*(1 + 2*y)*(-1 + z)*z2*sdz4
	-((1 - 4*x + 3*x2)*y12*y*z2*(-3 + 2*z))*sdxy4
	+6*(-1 + x)*x*y12*y*(-1 + z)*z2*sdyz4
	+(1 - 4*x + 3*x2)*y12*(1 + 2*y)*(-1 + z)*z2*sdxz4
	+(1 - 4*x + 3*x2)*y12*y*(-1 + z)*z2*sdxyz4
	+6*(-1 + x)*x*y12*(1 + 2*y)*z2*(-3 + 2*z)*cd.node5()
	-(x*(-2 + 3*x)*y12*(1 + 2*y)*z2*(-3 + 2*z))*sdx5
	+6*(-1 + x)*x*y12*y*z2*(-3 + 2*z)*sdy5
	-6*(-1 + x)*x*y12*(1 + 2*y)*(-1 + z)*z2*sdz5
	-(x*(-2 + 3*x)*y12*y*z2*(-3 + 2*z))*sdxy5
	-6*(-1 + x)*x*y12*y*(-1 + z)*z2*sdyz5
	+x*(-2 + 3*x)*y12*(1 + 2*y)*(-1 + z)*z2*sdxz5
	+x*(-2 + 3*x)*y12*y*(-1 + z)*z2*sdxyz5
	-6*(-1 + x)*x*y2*(-3 + 2*y)*z2*(-3 + 2*z)*cd.node6()
	+x*(-2 + 3*x)*y2*(-3 + 2*y)*z2*(-3 + 2*z)*sdx6
	+6*(-1 + x)*x*(-1 + y)*y2*z2*(-3 + 2*z)*sdy6
	+6*(-1 + x)*x*y2*(-3 + 2*y)*(-1 + z)*z2*sdz6
	-(x*(-2 + 3*x)*(-1 + y)*y2*z2*(-3 + 2*z))*sdxy6
	-6*(-1 + x)*x*(-1 + y)*y2*(-1 + z)*z2*sdyz6
	-(x*(-2 + 3*x)*y2*(-3 + 2*y)*(-1 + z)*z2)*sdxz6
	+x*(-2 + 3*x)*(-1 + y)*y2*(-1 + z)*z2*sdxyz6
	+6*(-1 + x)*x*y2*(-3 +  2*y)*z2*(-3 + 2*z)*cd.node7()
	+(1 - 4*x + 3*x2)*y2*(-3 + 2*y)*z2*(-3 + 2*z)*sdx7
	-6*(-1 + x)*x*(-1 + y)*y2*z2*(-3 + 2*z)*sdy7
	-6*(-1 + x)*x*y2*(-3 + 2*y)*(-1 + z)*z2*sdz7
	-((1 - 4*x + 3*x2)*(-1 + y)*y2*z2*(-3 + 2*z))*sdxy7
	+6*(-1 + x)*x*(-1 + y)*y2*(-1 + z)*z2*sdyz7
	-((1 - 4*x + 3*x2)*y2*(-3 + 2*y)*(-1 + z)*z2)*sdxz7
	+(1 - 4*x + 3*x2)*(-1 + y)*y2*(-1 + z)*z2*sdxyz7);
      
    derivs[1]=
      T(6*x12*(1 + 2*x)*(-1 + y)*y*z12*(1 + 2*z)*cd.node0()
	+6*x12*x*(-1 + y)*y*z12*(1 + 2*z)*sdx0
	+x12*(1 + 2*x)*(1 - 4*y + 3*y2)*z12*(1 + 2*z)*sdy0
	+6*x12*(1 + 2*x)*(-1 + y)*y*z12*z*sdz0
	+x12*x*(1 - 4*y + 3*y2)*z12*(1 + 2*z)*sdxy0
	+x12*(1 + 2*x)*(1 - 4*y + 3*y2)*z12*z*sdyz0
	+6*x12*x*(-1 + y)*y*z12*z*sdxz0
	+x12*x*(1 - 4*y + 3*y2)*z12*z*sdxyz0
	-6*x2*(-3 + 2*x)*(-1 + y)*y*z12*(1 + 2*z)*cd.node1()
	+6*(-1 + x)*x2*(-1 + y)*y*z12*(1 + 2*z)*sdx1
	-(x2*(-3 + 2*x)*(1 - 4*y + 3*y2)*z12*(1 + 2*z))*sdy1
	-6*x2*(-3 + 2*x)*(-1 + y)*y*z12*z*sdz1
	+(-1 + x)*x2*(1 - 4*y + 3*y2)*z12*(1 + 2*z)*sdxy1
	-(x2*(-3 + 2*x)*(1 - 4*y + 3*y2)*z12*z)*sdyz1
	+6*(-1 + x)*x2*(-1 + y)*y*z12*z*sdxz1
	+(-1 + x)*x2*(1 - 4*y + 3*y2)*z12*z*sdxyz1
	+6*x2*(-3 + 2*x)*(-1 + y)*y*z12*(1 + 2*z)*cd.node2()
	-6*(-1 + x)*x2*(-1 + y)*y*z12*(1 + 2*z)*sdx2
	-(x2*(-3 + 2*x)*y*(-2 + 3*y)*z12*(1 + 2*z))*sdy2
	+6*x2*(-3 + 2*x)*(-1 + y)*y*z12*z*sdz2
	+(-1 + x)*x2*y*(-2 + 3*y)*z12*(1 + 2*z)*sdxy2
	-(x2*(-3 + 2*x)*y*(-2 + 3*y)*z12*z)*sdyz2
	-6*(-1 + x)*x2*(-1 + y)*y*z12*z*sdxz2
	+(-1 + x)*x2*y*(-2 + 3*y)*z12*z*sdxyz2
	-6*x12*(1 + 2*x)*(-1 + y)*y*z12*(1 + 2*z)*cd.node3()
	-6*x12*x*(-1 + y)*y*z12*(1 + 2*z)*sdx3
	+x12*(1 + 2*x)*y*(-2 + 3*y)*z12*(1 + 2*z)*sdy3
	-6*x12*(1 + 2*x)*(-1 + y)*y*z12*z*sdz3
	+x12*x*y*(-2 + 3*y)*z12*(1 + 2*z)*sdxy3
	+x12*(1 + 2*x)*y*(-2 + 3*y)*z12*z*sdyz3
	-6*x12*x*(-1 + y)*y*z12*z*sdxz3
	+x12*x*y*(-2 + 3*y)*z12*z*sdxyz3
	-6*x12*(1 + 2*x)*(-1 + y)*y*z2*(-3 + 2*z)*cd.node4()
	-6*x12*x*(-1 + y)*y*z2*(-3 + 2*z)*sdx4
	-(x12*(1 + 2*x)*(1 - 4*y + 3*y2)*z2*(-3 + 2*z))*sdy4
	+6*x12*(1 + 2*x)*(-1 + y)*y*(-1 + z)*z2*sdz4
	-(x12*x*(1 - 4*y + 3*y2)*z2*(-3 + 2*z))*sdxy4
	+x12*(1 + 2*x)*(1 - 4*y + 3*y2)*(-1 + z)*z2*sdyz4
	+6*x12*x*(-1 + y)*y*(-1 + z)*z2*sdxz4
	+x12*x*(1 - 4*y + 3*y2)*(-1 + z)*z2*sdxyz4
	+6*x2*(-3 + 2*x)*(-1 + y)*y*z2*(-3 + 2*z)*cd.node5()
	-6*(-1 + x)*x2*(-1 + y)*y*z2*(-3 + 2*z)*sdx5
	+x2*(-3 + 2*x)*(1 - 4*y + 3*y2)*z2*(-3 + 2*z)*sdy5
	-6*x2*(-3 + 2*x)*(-1 + y)*y*(-1 + z)*z2*sdz5
	-((-1 + x)*x2*(1 - 4*y + 3*y2)*z2*(-3 + 2*z))*sdxy5
	-(x2*(-3 + 2*x)*(1 - 4*y + 3*y2)*(-1 + z)*z2)*sdyz5
	+6*(-1 + x)*x2*(-1 + y)*y*(-1 + z)*z2*sdxz5
	+(-1 + x)*x2*(1 - 4*y + 3*y2)*(-1 + z)*z2*sdxyz5
	-6*x2*(-3 + 2*x)*(-1 + y)*y*z2*(-3 + 2*z)*cd.node6()
	+6*(-1 + x)*x2*(-1 + y)*y*z2*(-3 + 2*z)*sdx6
	+x2*(-3 + 2*x)*y*(-2 + 3*y)*z2*(-3 + 2*z)*sdy6
	+6*x2*(-3 + 2*x)*(-1 + y)*y*(-1 + z)*z2*sdz6
	-((-1 + x)*x2*y*(-2 + 3*y)*z2*(-3 + 2*z))*sdxy6
	-(x2*(-3 + 2*x)*y*(-2 + 3*y)*(-1 + z)*z2)*sdyz6
	-6*(-1 + x)*x2*(-1 + y)*y*(-1 + z)*z2*sdxz6
	+(-1 + x)*x2*y*(-2 + 3*y)*(-1 + z)*z2*sdxyz6
	+6*x12*(1 + 2*x)*(-1 + y)*y*z2*(-3 + 2*z)*cd.node7()
	+6*x12*x*(-1 + y)*y*z2*(-3 + 2*z)*sdx7
	-(x12*(1 + 2*x)*y*(-2 + 3*y)*z2*(-3 + 2*z))*sdy7
	-6*x12*(1 + 2*x)*(-1 + y)*y*(-1 + z)*z2*sdz7
	-(x12*x*y*(-2 + 3*y)*z2*(-3 + 2*z))*sdxy7
	+x12*(1 + 2*x)*y*(-2 + 3*y)*(-1 + z)*z2*sdyz7
	-6*x12*x*(-1 + y)*y*(-1 + z)*z2*sdxz7
	+x12*x*y*(-2 + 3*y)*(-1 + z)*z2*sdxyz7);
      
    derivs[2]=
      T(6*x12*(1 + 2*x)*y12*(1 + 2*y)*(-1 + z)*z*cd.node0()
	+6*x12*x*y12*(1 + 2*y)*(-1 + z)*z*sdx0
	+6*x12*(1 + 2*x)*y12*y*(-1 + z)*z*sdy0
	+x12*(1 + 2*x)*y12*(1 + 2*y)*(1 - 4*z + 3*z2)*sdz0
	+6*x12*x*y12*y*(-1 + z)*z*sdxy0
	+x12*(1 + 2*x)*y12*y*(1 - 4*z + 3*z2)*sdyz0
	+x12*x*y12*(1 + 2*y)*(1 - 4*z + 3*z2)*sdxz0
	+x12*x*y12*y*(1 - 4*z + 3*z2)*sdxyz0
	-6*x2*(-3 + 2*x)*y12*(1 + 2*y)*(-1 + z)*z*cd.node1()
	+6*(-1 + x)*x2*y12*(1 + 2*y)*(-1 + z)*z*sdx1
	-6*x2*(-3 + 2*x)*y12*y*(-1 + z)*z*sdy1
	-(x2*(-3 + 2*x)*y12*(1 + 2*y)*(1 - 4*z + 3*z2))*sdz1
	+6*(-1 + x)*x2*y12*y*(-1 + z)*z*sdxy1
	-(x2*(-3 + 2*x)*y12*y*(1 - 4*z + 3*z2))*sdyz1
	+(-1 + x)*x2*y12*(1 + 2*y)*(1 - 4*z + 3*z2)*sdxz1
	+(-1 + x)*x2*y12*y*(1 - 4*z + 3*z2)*sdxyz1
	+6*x2*(-3 + 2*x)*y2*(-3 + 2*y)*(-1 + z)*z*cd.node2()
	-6*(-1 + x)*x2*y2*(-3 + 2*y)*(-1 + z)*z*sdx2
	-6*x2*(-3 + 2*x)*(-1 + y)*y2*(-1 + z)*z*sdy2
	+x2*(-3 + 2*x)*y2*(-3 + 2*y)*(1 - 4*z + 3*z2)*sdz2
	+6*(-1 + x)*x2*(-1 + y)*y2*(-1 + z)*z*sdxy2
	-(x2*(-3 + 2*x)*(-1 + y)*y2*(1 - 4*z + 3*z2))*sdyz2
	-((-1 + x)*x2*y2*(-3 + 2*y)*(1 - 4*z + 3*z2))*sdxz2
	+(-1 + x)*x2*(-1 + y)*y2*(1 - 4*z + 3*z2)*sdxyz2
	-6*x12*(1 + 2*x)*y2*(-3 + 2*y)*(-1 + z)*z*cd.node3()
	-6*x12*x*y2*(-3 + 2*y)*(-1 + z)*z*sdx3
	+6*x12*(1 + 2*x)*(-1 + y)*y2*(-1 + z)*z*sdy3
	-(x12*(1 + 2*x)*y2*(-3 + 2*y)*(1 - 4*z + 3*z2))*sdz3
	+6*x12*x*(-1 + y)*y2*(-1 + z)*z*sdxy3
	+x12*(1 + 2*x)*(-1 + y)*y2*(1 - 4*z + 3*z2)*sdyz3
	-(x12*x*y2*(-3 + 2*y)*(1 - 4*z + 3*z2))*sdxz3
	+x12*x*(-1 + y)*y2*(1 - 4*z + 3*z2)*sdxyz3
	-6*x12*(1 + 2*x)*y12*(1 + 2*y)*(-1 + z)*z*cd.node4()
	-6*x12*x*y12*(1 + 2*y)*(-1 + z)*z*sdx4
	-6*x12*(1 + 2*x)*y12*y*(-1 + z)*z*sdy4
	+x12*(1 + 2*x)*y12*(1 + 2*y)*z*(-2 + 3*z)*sdz4
	-6*x12*x*y12*y*(-1 + z)*z*sdxy4
	+x12*(1 + 2*x)*y12*y*z*(-2 + 3*z)*sdyz4
	+x12*x*y12*(1 + 2*y)*z*(-2 + 3*z)*sdxz4
	+x12*x*y12*y*z*(-2 + 3*z)*sdxyz4
	+6*x2*(-3 + 2*x)*y12*(1 + 2*y)*(-1 + z)*z*cd.node5()
	-6*(-1 + x)*x2*y12*(1 + 2*y)*(-1 + z)*z*sdx5
	+6*x2*(-3 + 2*x)*y12*y*(-1 + z)*z*sdy5
	-(x2*(-3 + 2*x)*y12*(1 + 2*y)*z*(-2 + 3*z))*sdz5
	-6*(-1 + x)*x2*y12*y*(-1 + z)*z*sdxy5
	-(x2*(-3 + 2*x)*y12*y*z*(-2 + 3*z))*sdyz5
	+(-1 + x)*x2*y12*(1 + 2*y)*z*(-2 + 3*z)*sdxz5
	+(-1 + x)*x2*y12*y*z*(-2 + 3*z)*sdxyz5
	-6*x2*(-3 + 2*x)*y2*(-3 + 2*y)*(-1 + z)*z*cd.node6()
	+6*(-1 + x)*x2*y2*(-3 + 2*y)*(-1 + z)*z*sdx6
	+6*x2*(-3 + 2*x)*(-1 + y)*y2*(-1 + z)*z*sdy6
	+x2*(-3 + 2*x)*y2*(-3 + 2*y)*z*(-2 + 3*z)*sdz6
	-6*(-1 + x)*x2*(-1 + y)*y2*(-1 + z)*z*sdxy6
	-(x2*(-3 + 2*x)*(-1 + y)*y2*z*(-2 + 3*z))*sdyz6
	-((-1 + x)*x2*y2*(-3 + 2*y)*z*(-2 + 3*z))*sdxz6
	+(-1 + x)*x2*(-1 + y)*y2*z*(-2 + 3*z)*sdxyz6
	+6*x12*(1 + 2*x)*y2*(-3 + 2*y)*(-1 + z)*z*cd.node7()
	+6*x12*x*y2*(-3 + 2*y)*(-1 + z)*z*sdx7
	-6*x12*(1 + 2*x)*(-1 + y)*y2*(-1 + z)*z*sdy7
	-(x12*(1 + 2*x)*y2*(-3 + 2*y)*z*(-2 + 3*z))*sdz7
	-6*x12*x*(-1 + y)*y2*(-1 + z)*z*sdxy7
	+x12*(1 + 2*x)*(-1 + y)*y2*z*(-2 + 3*z)*sdyz7
	-(x12*x*y2*(-3 + 2*y)*z*(-2 + 3*z))*sdxz7
	+x12*x*(-1 + y)*y2*z*(-2 + 3*z)*sdxyz7);
  }  

  //! get parametric coordinate for value within the element
  template <class CellData>
  bool get_coords(vector<double> &coords, const T& value, 
		  const CellData &cd) const  
  {
    HexLocate< HexTricubicHmtScaleFactors<T> > CL;
    return CL.get_coords(this, coords, value, cd);
  };

  //! add derivative values (dx, dy, dz, dxy, dyz, dzx, dxyz) for nodes.
  void add_derivative(const vector<T> &p) { derivs_.push_back(p); }

  //! add scale factors (sdx, sdy, sdz) for nodes.
  void add_scalefactors(const vector<double> &p) { scalefactors_.push_back(p); }

  static  const string type_name(int n = -1);
  virtual void io (Piostream& str);

protected:
  //! support data
  vector<vector<T> >          derivs_; 
  vector<vector<double> >          scalefactors_; 
};

template <class T>
const string
HexTricubicHmtScaleFactors<T>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;
  }
  else if (n == 0)
  {
    static const string nm("HexTricubicHmtScaleFactors");
    return nm;
  } else {
    return find_type_name((T *)0);
  }
}

template <class T>
const TypeDescription*
get_type_description(HexTricubicHmtScaleFactors<T> *)
{
  static TypeDescription* td = 0;
  if(!td){
    const TypeDescription *sub = get_type_description((T*)0);
    TypeDescription::td_vec *subs = scinew TypeDescription::td_vec(1);
    (*subs)[0] = sub;
    td = scinew TypeDescription("HexTricubicHmtScaleFactors", 
				subs, 
				string(__FILE__),
				"SCIRun", 
				TypeDescription::BASIS_E);
  }
  return td;
}


const int HEXTRICUBICHMTSCALEFACTORS_VERSION = 1;
template <class T>
void
HexTricubicHmtScaleFactors<T>::io(Piostream &stream)
{
  stream.begin_class(get_type_description(this)->get_name(),
                     HEXTRICUBICHMTSCALEFACTORS_VERSION);
  Pio(stream, derivs_);
  Pio(stream, scalefactors_);
  stream.end_class();
}

} //namespace SCIRun

#endif // HexTricubicHmtScaleFactors_h
