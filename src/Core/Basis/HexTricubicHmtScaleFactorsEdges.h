/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
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
//    File   : HexTricubicHmtScaleFactorsEdges.h
//    Author : Martin Cole, Frank B. Sachse
//    Date   : Nov 5 2005

#if !defined(HexTricubicHmtScaleFactorsEdges_h)
#define HexTricubicHmtScaleFactorsEdges_h

#include <Core/Persistent/PersistentSTL.h>
#include <Core/Basis/HexTrilinearLgn.h>

namespace SCIRun {

//! Class for describing unit geometry of HexTricubicHmtScaleFactorsEdges
class HexTricubicHmtScaleFactorsEdgesUnitElement : 
    public HexTrilinearLgnUnitElement {
public:
  HexTricubicHmtScaleFactorsEdgesUnitElement() {}
  virtual ~HexTricubicHmtScaleFactorsEdgesUnitElement() {}

  static int dofs() { return 64; } //!< return degrees of freedom
};


//! Class for handling of element of type hexahedron with 
//! tricubic hermitian interpolation with scale factors
template <class T>
class HexTricubicHmtScaleFactorsEdges : public BasisAddDerivativesScaleFactorsEdges<T>, 
                                   public HexApprox, 
				   public HexGaussian3<double>, 
				   public HexTricubicHmtScaleFactorsEdgesUnitElement
{
public:
  typedef T value_type;

  HexTricubicHmtScaleFactorsEdges() {}
  virtual ~HexTricubicHmtScaleFactorsEdges() {}
  
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

    const double sx0=this->scalefactors_[cd.edge0_index()];
    const double sx1=this->scalefactors_[cd.edge0_index()];
    const double sx2=this->scalefactors_[cd.edge2_index()];
    const double sx3=this->scalefactors_[cd.edge2_index()];
    const double sx4=this->scalefactors_[cd.edge4_index()];
    const double sx5=this->scalefactors_[cd.edge4_index()];
    const double sx6=this->scalefactors_[cd.edge6_index()];
    const double sx7=this->scalefactors_[cd.edge6_index()];

    const double sy0=this->scalefactors_[cd.edge3_index()];
    const double sy1=this->scalefactors_[cd.edge1_index()];
    const double sy2=this->scalefactors_[cd.edge1_index()];
    const double sy3=this->scalefactors_[cd.edge3_index()];
    const double sy4=this->scalefactors_[cd.edge7_index()];
    const double sy5=this->scalefactors_[cd.edge5_index()];
    const double sy6=this->scalefactors_[cd.edge5_index()];
    const double sy7=this->scalefactors_[cd.edge7_index()];

    const double sz0=this->scalefactors_[cd.edge8_index()];
    const double sz1=this->scalefactors_[cd.edge9_index()];
    const double sz2=this->scalefactors_[cd.edge10_index()];
    const double sz3=this->scalefactors_[cd.edge11_index()];
    const double sz4=this->scalefactors_[cd.edge8_index()];
    const double sz5=this->scalefactors_[cd.edge9_index()];
    const double sz6=this->scalefactors_[cd.edge10_index()];
    const double sz7=this->scalefactors_[cd.edge11_index()];

    const T sdx0=this->derivs_[cd.node0_index()][0]*sx0;
    const T sdx1=this->derivs_[cd.node1_index()][0]*sx1;
    const T sdx2=this->derivs_[cd.node2_index()][0]*sx2;
    const T sdx3=this->derivs_[cd.node3_index()][0]*sx3;
    const T sdx4=this->derivs_[cd.node4_index()][0]*sx4;
    const T sdx5=this->derivs_[cd.node5_index()][0]*sx5;
    const T sdx6=this->derivs_[cd.node6_index()][0]*sx6;
    const T sdx7=this->derivs_[cd.node7_index()][0]*sx7;

    const T sdy0=this->derivs_[cd.node0_index()][1]*sy0;
    const T sdy1=this->derivs_[cd.node1_index()][1]*sy1;
    const T sdy2=this->derivs_[cd.node2_index()][1]*sy2;
    const T sdy3=this->derivs_[cd.node3_index()][1]*sy3;
    const T sdy4=this->derivs_[cd.node4_index()][1]*sy4;
    const T sdy5=this->derivs_[cd.node5_index()][1]*sy5;
    const T sdy6=this->derivs_[cd.node6_index()][1]*sy6;
    const T sdy7=this->derivs_[cd.node7_index()][1]*sy7;

    const T sdz0=this->derivs_[cd.node0_index()][2]*sz0;
    const T sdz1=this->derivs_[cd.node1_index()][2]*sz1;
    const T sdz2=this->derivs_[cd.node2_index()][2]*sz2;
    const T sdz3=this->derivs_[cd.node3_index()][2]*sz3;
    const T sdz4=this->derivs_[cd.node4_index()][2]*sz4;
    const T sdz5=this->derivs_[cd.node5_index()][2]*sz5;
    const T sdz6=this->derivs_[cd.node6_index()][2]*sz6;
    const T sdz7=this->derivs_[cd.node7_index()][2]*sz7;

    const T sdxy0=this->derivs_[cd.node0_index()][3]*sx0*sy0;
    const T sdxy1=this->derivs_[cd.node1_index()][3]*sx1*sy1;
    const T sdxy2=this->derivs_[cd.node2_index()][3]*sx2*sy2;
    const T sdxy3=this->derivs_[cd.node3_index()][3]*sx3*sy3;
    const T sdxy4=this->derivs_[cd.node4_index()][3]*sx4*sy4;
    const T sdxy5=this->derivs_[cd.node5_index()][3]*sx5*sy5;
    const T sdxy6=this->derivs_[cd.node6_index()][3]*sx6*sy6;
    const T sdxy7=this->derivs_[cd.node7_index()][3]*sx7*sy7;

    const T sdyz0=this->derivs_[cd.node0_index()][4]*sy0*sz0;
    const T sdyz1=this->derivs_[cd.node1_index()][4]*sy1*sz1;
    const T sdyz2=this->derivs_[cd.node2_index()][4]*sy2*sz2;
    const T sdyz3=this->derivs_[cd.node3_index()][4]*sy3*sz3;
    const T sdyz4=this->derivs_[cd.node4_index()][4]*sy4*sz4;
    const T sdyz5=this->derivs_[cd.node5_index()][4]*sy5*sz5;
    const T sdyz6=this->derivs_[cd.node6_index()][4]*sy6*sz6;
    const T sdyz7=this->derivs_[cd.node7_index()][4]*sy7*sz7;

    const T sdxz0=this->derivs_[cd.node0_index()][5]*sx0*sz0;
    const T sdxz1=this->derivs_[cd.node1_index()][5]*sx1*sz1;
    const T sdxz2=this->derivs_[cd.node2_index()][5]*sx2*sz2;
    const T sdxz3=this->derivs_[cd.node3_index()][5]*sx3*sz3;
    const T sdxz4=this->derivs_[cd.node4_index()][5]*sx4*sz4;
    const T sdxz5=this->derivs_[cd.node5_index()][5]*sx5*sz5;
    const T sdxz6=this->derivs_[cd.node6_index()][5]*sx6*sz6;
    const T sdxz7=this->derivs_[cd.node7_index()][5]*sx7*sz7;

    const T sdxyz0=this->derivs_[cd.node0_index()][6]*sx0*sy0*sz0;
    const T sdxyz1=this->derivs_[cd.node1_index()][6]*sx1*sy1*sz1;
    const T sdxyz2=this->derivs_[cd.node2_index()][6]*sx2*sy2*sz2;
    const T sdxyz3=this->derivs_[cd.node3_index()][6]*sx3*sy3*sz3;
    const T sdxyz4=this->derivs_[cd.node4_index()][6]*sx4*sy4*sz4;
    const T sdxyz5=this->derivs_[cd.node5_index()][6]*sx5*sy5*sz5;
    const T sdxyz6=this->derivs_[cd.node6_index()][6]*sx6*sy6*sz6;
    const T sdxyz7=this->derivs_[cd.node7_index()][6]*sx7*sy7*sz7;

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

    const double sx0=this->scalefactors_[cd.edge0_index()];
    const double sx1=this->scalefactors_[cd.edge0_index()];
    const double sx2=this->scalefactors_[cd.edge2_index()];
    const double sx3=this->scalefactors_[cd.edge2_index()];
    const double sx4=this->scalefactors_[cd.edge4_index()];
    const double sx5=this->scalefactors_[cd.edge4_index()];
    const double sx6=this->scalefactors_[cd.edge6_index()];
    const double sx7=this->scalefactors_[cd.edge6_index()];

    const double sy0=this->scalefactors_[cd.edge3_index()];
    const double sy1=this->scalefactors_[cd.edge1_index()];
    const double sy2=this->scalefactors_[cd.edge1_index()];
    const double sy3=this->scalefactors_[cd.edge3_index()];
    const double sy4=this->scalefactors_[cd.edge7_index()];
    const double sy5=this->scalefactors_[cd.edge5_index()];
    const double sy6=this->scalefactors_[cd.edge5_index()];
    const double sy7=this->scalefactors_[cd.edge7_index()];

    const double sz0=this->scalefactors_[cd.edge8_index()];
    const double sz1=this->scalefactors_[cd.edge9_index()];
    const double sz2=this->scalefactors_[cd.edge10_index()];
    const double sz3=this->scalefactors_[cd.edge11_index()];
    const double sz4=this->scalefactors_[cd.edge8_index()];
    const double sz5=this->scalefactors_[cd.edge9_index()];
    const double sz6=this->scalefactors_[cd.edge10_index()];
    const double sz7=this->scalefactors_[cd.edge11_index()];

    const T sdx0=this->derivs_[cd.node0_index()][0]*sx0;
    const T sdx1=this->derivs_[cd.node1_index()][0]*sx1;
    const T sdx2=this->derivs_[cd.node2_index()][0]*sx2;
    const T sdx3=this->derivs_[cd.node3_index()][0]*sx3;
    const T sdx4=this->derivs_[cd.node4_index()][0]*sx4;
    const T sdx5=this->derivs_[cd.node5_index()][0]*sx5;
    const T sdx6=this->derivs_[cd.node6_index()][0]*sx6;
    const T sdx7=this->derivs_[cd.node7_index()][0]*sx7;

    const T sdy0=this->derivs_[cd.node0_index()][1]*sy0;
    const T sdy1=this->derivs_[cd.node1_index()][1]*sy1;
    const T sdy2=this->derivs_[cd.node2_index()][1]*sy2;
    const T sdy3=this->derivs_[cd.node3_index()][1]*sy3;
    const T sdy4=this->derivs_[cd.node4_index()][1]*sy4;
    const T sdy5=this->derivs_[cd.node5_index()][1]*sy5;
    const T sdy6=this->derivs_[cd.node6_index()][1]*sy6;
    const T sdy7=this->derivs_[cd.node7_index()][1]*sy7;

    const T sdz0=this->derivs_[cd.node0_index()][2]*sz0;
    const T sdz1=this->derivs_[cd.node1_index()][2]*sz1;
    const T sdz2=this->derivs_[cd.node2_index()][2]*sz2;
    const T sdz3=this->derivs_[cd.node3_index()][2]*sz3;
    const T sdz4=this->derivs_[cd.node4_index()][2]*sz4;
    const T sdz5=this->derivs_[cd.node5_index()][2]*sz5;
    const T sdz6=this->derivs_[cd.node6_index()][2]*sz6;
    const T sdz7=this->derivs_[cd.node7_index()][2]*sz7;

    const T sdxy0=this->derivs_[cd.node0_index()][3]*sx0*sy0;
    const T sdxy1=this->derivs_[cd.node1_index()][3]*sx1*sy1;
    const T sdxy2=this->derivs_[cd.node2_index()][3]*sx2*sy2;
    const T sdxy3=this->derivs_[cd.node3_index()][3]*sx3*sy3;
    const T sdxy4=this->derivs_[cd.node4_index()][3]*sx4*sy4;
    const T sdxy5=this->derivs_[cd.node5_index()][3]*sx5*sy5;
    const T sdxy6=this->derivs_[cd.node6_index()][3]*sx6*sy6;
    const T sdxy7=this->derivs_[cd.node7_index()][3]*sx7*sy7;

    const T sdyz0=this->derivs_[cd.node0_index()][4]*sy0*sz0;
    const T sdyz1=this->derivs_[cd.node1_index()][4]*sy1*sz1;
    const T sdyz2=this->derivs_[cd.node2_index()][4]*sy2*sz2;
    const T sdyz3=this->derivs_[cd.node3_index()][4]*sy3*sz3;
    const T sdyz4=this->derivs_[cd.node4_index()][4]*sy4*sz4;
    const T sdyz5=this->derivs_[cd.node5_index()][4]*sy5*sz5;
    const T sdyz6=this->derivs_[cd.node6_index()][4]*sy6*sz6;
    const T sdyz7=this->derivs_[cd.node7_index()][4]*sy7*sz7;

    const T sdxz0=this->derivs_[cd.node0_index()][5]*sx0*sz0;
    const T sdxz1=this->derivs_[cd.node1_index()][5]*sx1*sz1;
    const T sdxz2=this->derivs_[cd.node2_index()][5]*sx2*sz2;
    const T sdxz3=this->derivs_[cd.node3_index()][5]*sx3*sz3;
    const T sdxz4=this->derivs_[cd.node4_index()][5]*sx4*sz4;
    const T sdxz5=this->derivs_[cd.node5_index()][5]*sx5*sz5;
    const T sdxz6=this->derivs_[cd.node6_index()][5]*sx6*sz6;
    const T sdxz7=this->derivs_[cd.node7_index()][5]*sx7*sz7;

    const T sdxyz0=this->derivs_[cd.node0_index()][6]*sx0*sy0*sz0;
    const T sdxyz1=this->derivs_[cd.node1_index()][6]*sx1*sy1*sz1;
    const T sdxyz2=this->derivs_[cd.node2_index()][6]*sx2*sy2*sz2;
    const T sdxyz3=this->derivs_[cd.node3_index()][6]*sx3*sy3*sz3;
    const T sdxyz4=this->derivs_[cd.node4_index()][6]*sx4*sy4*sz4;
    const T sdxyz5=this->derivs_[cd.node5_index()][6]*sx5*sy5*sz5;
    const T sdxyz6=this->derivs_[cd.node6_index()][6]*sx6*sy6*sz6;
    const T sdxyz7=this->derivs_[cd.node7_index()][6]*sx7*sy7*sz7;

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
  template <class ElemData>
  bool get_coords(std::vector<double> &coords, const T& value, 
		  const ElemData &cd) const  
  {
    HexLocate< HexTricubicHmtScaleFactorsEdges<T> > CL;
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
    return get_volume(this, cd);
  }
  
  static  const std::string type_name(int n = -1);
  virtual void io (Piostream& str);
};


template <class T>
const std::string
HexTricubicHmtScaleFactorsEdges<T>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const std::string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;
  }
  else if (n == 0)
  {
    static const std::string nm("HexTricubicHmtScaleFactorsEdges");
    return nm;
  } else {
    return find_type_name((T *)0);
  }
}

template <class T>
const TypeDescription*
get_type_description(HexTricubicHmtScaleFactorsEdges<T> *)
{
  static TypeDescription* td = 0;
  if(!td){
    const TypeDescription *sub = get_type_description((T*)0);
    TypeDescription::td_vec *subs = scinew TypeDescription::td_vec(1);
    (*subs)[0] = sub;
    td = scinew TypeDescription("HexTricubicHmtScaleFactorsEdges", 
				subs, 
				std::string(__FILE__),
				"SCIRun", 
				TypeDescription::BASIS_E);
  }
  return td;
}


const int HEXTRICUBICHMTSCALEFACTORSEDGES_VERSION = 1;
template <class T>
void
HexTricubicHmtScaleFactorsEdges<T>::io(Piostream &stream)
{
  stream.begin_class(get_type_description(this)->get_name(),
                     HEXTRICUBICHMTSCALEFACTORSEDGES_VERSION);
  Pio(stream, this->derivs_);
  Pio(stream, this->scalefactors_);
  stream.end_class();
}

} //namespace SCIRun


#endif // HexTricubicHmtScaleFactorsEdges_h
