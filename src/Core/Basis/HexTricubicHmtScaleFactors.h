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
//    Date   : Dec 3 2004

#if !defined(HexTricubicHmtScaleFactors_h)
#define HexTricubicHmtScaleFactors_h

#include <Core/Basis/HexTrilinearLgn.h>

namespace SCIRun {

template <class T>
class HexTricubicHmtScaleFactors : public HexApprox
{
public:
  typedef T value_type;

  HexTricubicHmtScaleFactors() {}
  virtual ~HexTricubicHmtScaleFactors() {}
  
  int polynomial_order() const { return 3; }

  // Value at coord
  template <class CellData>
  T interpolate(const vector<double> &coords, const CellData &cd) const
  {
    const double x=coords[0], y=coords[1], z=coords[2];  
    const double x2=x*x;
    const double y2=y*y;
    const double z2=z*z;
    const double x12=(x-1)*(x-1);
    const double y12=(y-1)*(y-1);
    const double z12=(z-1)*(z-1);

    const double sdx0=derivs_[cd.node0_index()][0]*scalefactors_[cd.elem][0];
    const double sdx1=derivs_[cd.node1_index()][0]*scalefactors_[cd.elem][0];
    const double sdx2=derivs_[cd.node2_index()][0]*scalefactors_[cd.elem][0];
    const double sdx3=derivs_[cd.node3_index()][0]*scalefactors_[cd.elem][0];
    const double sdx4=derivs_[cd.node4_index()][0]*scalefactors_[cd.elem][0];
    const double sdx5=derivs_[cd.node5_index()][0]*scalefactors_[cd.elem][0];
    const double sdx6=derivs_[cd.node6_index()][0]*scalefactors_[cd.elem][0];
    const double sdx7=derivs_[cd.node7_index()][0]*scalefactors_[cd.elem][0];

    const double sdy0=derivs_[cd.node0_index()][1]*scalefactors_[cd.elem][1];
    const double sdy1=derivs_[cd.node1_index()][1]*scalefactors_[cd.elem][1];
    const double sdy2=derivs_[cd.node2_index()][1]*scalefactors_[cd.elem][1];
    const double sdy3=derivs_[cd.node3_index()][1]*scalefactors_[cd.elem][1];
    const double sdy4=derivs_[cd.node4_index()][1]*scalefactors_[cd.elem][1];
    const double sdy5=derivs_[cd.node5_index()][1]*scalefactors_[cd.elem][1];
    const double sdy6=derivs_[cd.node6_index()][1]*scalefactors_[cd.elem][1];
    const double sdy7=derivs_[cd.node7_index()][1]*scalefactors_[cd.elem][1];

    const double sdz0=derivs_[cd.node0_index()][2]*scalefactors_[cd.elem][2];
    const double sdz1=derivs_[cd.node1_index()][2]*scalefactors_[cd.elem][2];
    const double sdz2=derivs_[cd.node2_index()][2]*scalefactors_[cd.elem][2];
    const double sdz3=derivs_[cd.node3_index()][2]*scalefactors_[cd.elem][2];
    const double sdz4=derivs_[cd.node4_index()][2]*scalefactors_[cd.elem][2];
    const double sdz5=derivs_[cd.node5_index()][2]*scalefactors_[cd.elem][2];
    const double sdz6=derivs_[cd.node6_index()][2]*scalefactors_[cd.elem][2];
    const double sdz7=derivs_[cd.node7_index()][2]*scalefactors_[cd.elem][2];

    const double sdxy0=derivs_[cd.node0_index()][3]*scalefactors_[cd.elem][0]*scalefactors_[cd.elem][1];
    const double sdxy1=derivs_[cd.node1_index()][3]*scalefactors_[cd.elem][0]*scalefactors_[cd.elem][1];
    const double sdxy2=derivs_[cd.node2_index()][3]*scalefactors_[cd.elem][0]*scalefactors_[cd.elem][1];
    const double sdxy3=derivs_[cd.node3_index()][3]*scalefactors_[cd.elem][0]*scalefactors_[cd.elem][1];
    const double sdxy4=derivs_[cd.node4_index()][3]*scalefactors_[cd.elem][0]*scalefactors_[cd.elem][1];
    const double sdxy5=derivs_[cd.node5_index()][3]*scalefactors_[cd.elem][0]*scalefactors_[cd.elem][1];
    const double sdxy6=derivs_[cd.node6_index()][3]*scalefactors_[cd.elem][0]*scalefactors_[cd.elem][1];
    const double sdxy7=derivs_[cd.node7_index()][3]*scalefactors_[cd.elem][0]*scalefactors_[cd.elem][1];

    const double sdyz0=derivs_[cd.node0_index()][4]*scalefactors_[cd.elem][2]*scalefactors_[cd.elem][1];
    const double sdyz1=derivs_[cd.node1_index()][4]*scalefactors_[cd.elem][2]*scalefactors_[cd.elem][1];
    const double sdyz2=derivs_[cd.node2_index()][4]*scalefactors_[cd.elem][2]*scalefactors_[cd.elem][1];
    const double sdyz3=derivs_[cd.node3_index()][4]*scalefactors_[cd.elem][2]*scalefactors_[cd.elem][1];
    const double sdyz4=derivs_[cd.node4_index()][4]*scalefactors_[cd.elem][2]*scalefactors_[cd.elem][1];
    const double sdyz5=derivs_[cd.node5_index()][4]*scalefactors_[cd.elem][2]*scalefactors_[cd.elem][1];
    const double sdyz6=derivs_[cd.node6_index()][4]*scalefactors_[cd.elem][2]*scalefactors_[cd.elem][1];
    const double sdyz7=derivs_[cd.node7_index()][4]*scalefactors_[cd.elem][2]*scalefactors_[cd.elem][1];

    const double sdxz0=derivs_[cd.node0_index()][5]*scalefactors_[cd.elem][2]*scalefactors_[cd.elem][0];
    const double sdxz1=derivs_[cd.node1_index()][5]*scalefactors_[cd.elem][2]*scalefactors_[cd.elem][0];
    const double sdxz2=derivs_[cd.node2_index()][5]*scalefactors_[cd.elem][2]*scalefactors_[cd.elem][0];
    const double sdxz3=derivs_[cd.node3_index()][5]*scalefactors_[cd.elem][2]*scalefactors_[cd.elem][0];
    const double sdxz4=derivs_[cd.node4_index()][5]*scalefactors_[cd.elem][2]*scalefactors_[cd.elem][0];
    const double sdxz5=derivs_[cd.node5_index()][5]*scalefactors_[cd.elem][2]*scalefactors_[cd.elem][0];
    const double sdxz6=derivs_[cd.node6_index()][5]*scalefactors_[cd.elem][2]*scalefactors_[cd.elem][0];
    const double sdxz7=derivs_[cd.node7_index()][5]*scalefactors_[cd.elem][2]*scalefactors_[cd.elem][0];

    const double sdxyz0=derivs_[cd.node0_index()][6]*scalefactors_[cd.elem][2]*scalefactors_[cd.elem][1]*scalefactors_[cd.elem][0];
    const double sdxyz1=derivs_[cd.node1_index()][6]*scalefactors_[cd.elem][2]*scalefactors_[cd.elem][1]*scalefactors_[cd.elem][0];
    const double sdxyz2=derivs_[cd.node2_index()][6]*scalefactors_[cd.elem][2]*scalefactors_[cd.elem][1]*scalefactors_[cd.elem][0];
    const double sdxyz3=derivs_[cd.node3_index()][6]*scalefactors_[cd.elem][2]*scalefactors_[cd.elem][1]*scalefactors_[cd.elem][0];
    const double sdxyz4=derivs_[cd.node4_index()][6]*scalefactors_[cd.elem][2]*scalefactors_[cd.elem][1]*scalefactors_[cd.elem][0];
    const double sdxyz5=derivs_[cd.node5_index()][6]*scalefactors_[cd.elem][2]*scalefactors_[cd.elem][1]*scalefactors_[cd.elem][0];
    const double sdxyz6=derivs_[cd.node6_index()][6]*scalefactors_[cd.elem][2]*scalefactors_[cd.elem][1]*scalefactors_[cd.elem][0];
    const double sdxyz7=derivs_[cd.node7_index()][6]*scalefactors_[cd.elem][2]*scalefactors_[cd.elem][1]*scalefactors_[cd.elem][0];

    return (T)(x12*(1 + 2*x)*y12*(1 + 2*y)*z12*(1 + 2*z)*cd.node0()
      +x12*x*y12*(1 + 2*y)*z12*(1 + 2*z)*sdx0
      +x12*(1 + 2*x)*y12*y*z12*(1 + 2*z)*sdy0
      +x12*(1 + 2*x)*y12*(1 + 2*y)*z12*z*sdz0
      +x12*x*y12*y*z12*(1 + 2*z)*sdxy0
      +x12*(1 + 2*x)*y12*y*z12*z*sdyz0
      +x12*x*y12*(1 + 2*y)*z12*z*sdxz0
      +x12*x*y12*y*z12*z*sdxyz0
      -(x2*(-3 + 2*x)*y12*(1 + 2*y)*z12*(1 + 2*z))*cd.node1()
      +(-1 + x)*x2*y12*(1 + 2*y)*z12*(1 + 2*z)*sdx1
      -(x2*(-3 + 2*x)*y12*y*z12*(1 + 2*z))*sdy1
      -(x2*(-3 + 2*x)*y12*(1 + 2*y)*z12*z)*sdz1
      +(-1 + x)*x2*y12*y*z12*(1 + 2*z)*sdxy1
      -(x2*(-3 + 2*x)*y12*y*z12*z) *sdyz1  
      +(-1 + x)*x2*y12*(1 + 2*y)*z12*z*sdxz1
      +(-1 + x)*x2*y12*y*z12*z*sdxyz1
      +x2*(-3 + 2*x)*y2*(-3 + 2*y)*z12*(1 + 2*z)*cd.node2()
      -((-1 + x)*x2*y2*(-3 + 2*y)*z12*(1 + 2*z))*sdx2
      -(x2*(-3 + 2*x)*(-1 + y)*y2*z12*(1 + 2*z))*sdy2
      +x2*(-3 + 2*x)*y2*(-3 + 2*y)*z12*z*sdz2
      +(-1 + x)*x2*(-1 + y)*y2*z12*(1 + 2*z)*sdxy2
      -(x2*(-3 + 2*x)*(-1 + y)*y2*z12*z)*sdyz2
      -((-1 + x)*x2*y2*(-3 + 2*y)*z12*z)*sdxz2
      +(-1 + x)*x2*(-1 + y)*y2*z12*z*sdxyz2 
      -(x12*(1 + 2*x)*y2*(-3 + 2*y)*z12*(1 + 2*z))*cd.node3()
      -(x12*x*y2*(-3 + 2*y)*z12*(1 + 2*z))*sdx3
      +x12*(1 + 2*x)*(-1 + y)*y2*z12*(1 + 2*z)*sdy3
      -(x12*(1 + 2*x)*y2*(-3 + 2*y)*z12*z)*sdz3
      +x12*x*(-1 + y)*y2*z12*(1 + 2*z)*sdxy3
      +x12*(1 + 2*x)*(-1 + y)*y2*z12*z*sdyz3
      -(x12*x*y2*(-3 + 2*y)*z12*z)*sdxz3
      +x12*x*(-1 + y)*y2*z12*z*sdxyz3
      -(x12*(1 + 2*x)*y12*(1 + 2*y)*z2*(-3 + 2*z))*cd.node4()
      -(x12*x*y12*(1 + 2*y)*z2*(-3 + 2*z))*sdx4
      -(x12*(1 + 2*x)*y12*y*z2*(-3 + 2*z))*sdy4
      +x12*(1 + 2*x)*y12*(1 + 2*y)*(-1 + z)*z2*sdz4
      -(x12*x*y12*y*z2*(-3 + 2*z))*sdxy4
      +x12*(1 + 2*x)*y12*y*(-1 + z)*z2*sdyz4
      +x12*x*y12*(1 + 2*y)*(-1 + z)*z2*sdxz4
      +x12*x*y12*y*(-1 + z)*z2*sdxyz4
      +x2*(-3 + 2*x)*y12*(1 + 2*y)*z2*(-3 + 2*z)*cd.node5()
      -((-1 + x)*x2*y12*(1 + 2*y)*z2*(-3 + 2*z))*sdx5
      +x2*(-3 + 2*x)*y12*y*z2*(-3 + 2*z)*sdy5
      -(x2*(-3 + 2*x)*y12*(1 + 2*y)*(-1 + z)*z2)*sdz5
      -((-1 + x)*x2*y12*y*z2*(-3 + 2*z))*sdxy5
      -(x2*(-3 + 2*x)*y12*y*(-1 + z)*z2)*sdyz5
      +(-1 + x)*x2*y12*(1 + 2*y)*(-1 + z)*z2*sdxz5
      +(-1 + x)*x2*y12*y*(-1 + z)*z2*sdxyz5
      -(x2*(-3 + 2*x)*y2*(-3 + 2*y)*z2*(-3 + 2*z))*cd.node6()
      +(-1 + x)*x2*y2*(-3 + 2*y)*z2*(-3 + 2*z)*sdx6
      +x2*(-3 + 2*x)*(-1 + y)*y2*z2*(-3 + 2*z)*sdy6
      +x2*(-3 + 2*x)*y2*(-3 + 2*y)*(-1 + z)*z2*sdz6
      -((-1 + x)*x2*(-1 + y)*y2*z2*(-3 + 2*z))*sdxy6
      -(x2*(-3 + 2*x)*(-1 + y)*y2*(-1 + z)*z2)*sdyz6
      -((-1 + x)*x2*y2*(-3 + 2*y)*(-1 + z)*z2)*sdxz6
      +(-1 + x)*x2*(-1 + y)*y2*(-1 + z)*z2*sdxyz6
      +x12*(1 + 2*x)*y2*(-3 + 2*y)*z2*(-3 + 2*z)*cd.node7()
      +x12*x*y2*(-3 + 2*y)*z2*(-3 + 2*z)*sdx7
      -(x12*(1 + 2*x)*(-1 + y)*y2*z2*(-3 + 2*z))*sdy7
      -(x12*(1 + 2*x)*y2*(-3 + 2*y)*(-1 + z)*z2)*sdz7
      -(x12*x*(-1 + y)*y2*z2*(-3 + 2*z))*sdxy7
      +x12*(1 + 2*x)*(-1 + y)*y2*(-1 + z)*z2*sdyz7
      -(x12*x*y2*(-3 + 2*y)*(-1 + z)*z2)*sdxz7
      +x12*x*(-1 + y)*y2*(-1 + z)*z2*sdxyz7);
  }
  
  //! First derivative at coord.
  template <class CellData>
  void derivate(const vector<double> &coords, const CellData &cd,
		vector<double> &derivs) const
  {
    const double x=coords[0], y=coords[1], z=coords[2];  
    const double x2=x*x;
    const double y2=y*y;
    const double z2=z*z;
    const double x12=(x-1)*(x-1);
    const double y12=(y-1)*(y-1);
    const double z12=(z-1)*(z-1);

    derivs.size(3);

    const double sdx0=derivs_[cd.node0_index()][0]*scalefactors_[cd.elem][0];
    const double sdx1=derivs_[cd.node1_index()][0]*scalefactors_[cd.elem][0];
    const double sdx2=derivs_[cd.node2_index()][0]*scalefactors_[cd.elem][0];
    const double sdx3=derivs_[cd.node3_index()][0]*scalefactors_[cd.elem][0];
    const double sdx4=derivs_[cd.node4_index()][0]*scalefactors_[cd.elem][0];
    const double sdx5=derivs_[cd.node5_index()][0]*scalefactors_[cd.elem][0];
    const double sdx6=derivs_[cd.node6_index()][0]*scalefactors_[cd.elem][0];
    const double sdx7=derivs_[cd.node7_index()][0]*scalefactors_[cd.elem][0];

    const double sdy0=derivs_[cd.node0_index()][1]*scalefactors_[cd.elem][1];
    const double sdy1=derivs_[cd.node1_index()][1]*scalefactors_[cd.elem][1];
    const double sdy2=derivs_[cd.node2_index()][1]*scalefactors_[cd.elem][1];
    const double sdy3=derivs_[cd.node3_index()][1]*scalefactors_[cd.elem][1];
    const double sdy4=derivs_[cd.node4_index()][1]*scalefactors_[cd.elem][1];
    const double sdy5=derivs_[cd.node5_index()][1]*scalefactors_[cd.elem][1];
    const double sdy6=derivs_[cd.node6_index()][1]*scalefactors_[cd.elem][1];
    const double sdy7=derivs_[cd.node7_index()][1]*scalefactors_[cd.elem][1];

    const double sdz0=derivs_[cd.node0_index()][2]*scalefactors_[cd.elem][2];
    const double sdz1=derivs_[cd.node1_index()][2]*scalefactors_[cd.elem][2];
    const double sdz2=derivs_[cd.node2_index()][2]*scalefactors_[cd.elem][2];
    const double sdz3=derivs_[cd.node3_index()][2]*scalefactors_[cd.elem][2];
    const double sdz4=derivs_[cd.node4_index()][2]*scalefactors_[cd.elem][2];
    const double sdz5=derivs_[cd.node5_index()][2]*scalefactors_[cd.elem][2];
    const double sdz6=derivs_[cd.node6_index()][2]*scalefactors_[cd.elem][2];
    const double sdz7=derivs_[cd.node7_index()][2]*scalefactors_[cd.elem][2];

    const double sdxy0=derivs_[cd.node0_index()][3]*scalefactors_[cd.elem][0]*scalefactors_[cd.elem][1];
    const double sdxy1=derivs_[cd.node1_index()][3]*scalefactors_[cd.elem][0]*scalefactors_[cd.elem][1];
    const double sdxy2=derivs_[cd.node2_index()][3]*scalefactors_[cd.elem][0]*scalefactors_[cd.elem][1];
    const double sdxy3=derivs_[cd.node3_index()][3]*scalefactors_[cd.elem][0]*scalefactors_[cd.elem][1];
    const double sdxy4=derivs_[cd.node4_index()][3]*scalefactors_[cd.elem][0]*scalefactors_[cd.elem][1];
    const double sdxy5=derivs_[cd.node5_index()][3]*scalefactors_[cd.elem][0]*scalefactors_[cd.elem][1];
    const double sdxy6=derivs_[cd.node6_index()][3]*scalefactors_[cd.elem][0]*scalefactors_[cd.elem][1];
    const double sdxy7=derivs_[cd.node7_index()][3]*scalefactors_[cd.elem][0]*scalefactors_[cd.elem][1];

    const double sdyz0=derivs_[cd.node0_index()][4]*scalefactors_[cd.elem][2]*scalefactors_[cd.elem][1];
    const double sdyz1=derivs_[cd.node1_index()][4]*scalefactors_[cd.elem][2]*scalefactors_[cd.elem][1];
    const double sdyz2=derivs_[cd.node2_index()][4]*scalefactors_[cd.elem][2]*scalefactors_[cd.elem][1];
    const double sdyz3=derivs_[cd.node3_index()][4]*scalefactors_[cd.elem][2]*scalefactors_[cd.elem][1];
    const double sdyz4=derivs_[cd.node4_index()][4]*scalefactors_[cd.elem][2]*scalefactors_[cd.elem][1];
    const double sdyz5=derivs_[cd.node5_index()][4]*scalefactors_[cd.elem][2]*scalefactors_[cd.elem][1];
    const double sdyz6=derivs_[cd.node6_index()][4]*scalefactors_[cd.elem][2]*scalefactors_[cd.elem][1];
    const double sdyz7=derivs_[cd.node7_index()][4]*scalefactors_[cd.elem][2]*scalefactors_[cd.elem][1];

    const double sdxz0=derivs_[cd.node0_index()][5]*scalefactors_[cd.elem][2]*scalefactors_[cd.elem][0];
    const double sdxz1=derivs_[cd.node1_index()][5]*scalefactors_[cd.elem][2]*scalefactors_[cd.elem][0];
    const double sdxz2=derivs_[cd.node2_index()][5]*scalefactors_[cd.elem][2]*scalefactors_[cd.elem][0];
    const double sdxz3=derivs_[cd.node3_index()][5]*scalefactors_[cd.elem][2]*scalefactors_[cd.elem][0];
    const double sdxz4=derivs_[cd.node4_index()][5]*scalefactors_[cd.elem][2]*scalefactors_[cd.elem][0];
    const double sdxz5=derivs_[cd.node5_index()][5]*scalefactors_[cd.elem][2]*scalefactors_[cd.elem][0];
    const double sdxz6=derivs_[cd.node6_index()][5]*scalefactors_[cd.elem][2]*scalefactors_[cd.elem][0];
    const double sdxz7=derivs_[cd.node7_index()][5]*scalefactors_[cd.elem][2]*scalefactors_[cd.elem][0];

    const double sdxyz0=derivs_[cd.node0_index()][6]*scalefactors_[cd.elem][2]*scalefactors_[cd.elem][1]*scalefactors_[cd.elem][0];
    const double sdxyz1=derivs_[cd.node1_index()][6]*scalefactors_[cd.elem][2]*scalefactors_[cd.elem][1]*scalefactors_[cd.elem][0];
    const double sdxyz2=derivs_[cd.node2_index()][6]*scalefactors_[cd.elem][2]*scalefactors_[cd.elem][1]*scalefactors_[cd.elem][0];
    const double sdxyz3=derivs_[cd.node3_index()][6]*scalefactors_[cd.elem][2]*scalefactors_[cd.elem][1]*scalefactors_[cd.elem][0];
    const double sdxyz4=derivs_[cd.node4_index()][6]*scalefactors_[cd.elem][2]*scalefactors_[cd.elem][1]*scalefactors_[cd.elem][0];
    const double sdxyz5=derivs_[cd.node5_index()][6]*scalefactors_[cd.elem][2]*scalefactors_[cd.elem][1]*scalefactors_[cd.elem][0];
    const double sdxyz6=derivs_[cd.node6_index()][6]*scalefactors_[cd.elem][2]*scalefactors_[cd.elem][1]*scalefactors_[cd.elem][0];
    const double sdxyz7=derivs_[cd.node7_index()][6]*scalefactors_[cd.elem][2]*scalefactors_[cd.elem][1]*scalefactors_[cd.elem][0];

    derivs[0]=6*(-1 + x)*x*y12*(1 + 2*y)*z12*(1 + 2*z)*cd.node0()
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
      +(1 - 4*x + 3*x2)*(-1 + y)*y2*(-1 + z)*z2*sdxyz7;
      
    derivs[1]=6*x12*(1 + 2*x)*(-1 + y)*y*z12*(1 + 2*z)*cd.node0()
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
      (-1 + x)*x2*(1 - 4*y + 3*y2)*z12*z*sdxyz1
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
      +x12*x*y*(-2 + 3*y)*(-1 + z)*z2*sdxyz7;
      
    derivs[2]=6*x12*(1 + 2*x)*y12*(1 + 2*y)*(-1 + z)*z*cd.node0()
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
      +x12*x*(-1 + y)*y2*z*(-2 + 3*z)*sdxyz7;
  }  

  //! return the parametric coordinates for value within the element.
  //! iterative solution...
  template <class CellData>
  void get_coords(vector<double> &coords, const T& value, 
		  const CellData &cd) const  
      {
	HexLocate< HexTricubicHmtScaleFactors<T> > CL;
	CL.get_coords(this, coords, value, cd);
      };

  //! add derivative values (dx, dy, dz, dxy, dyz, dzx, dxyz) for nodes.
  void add_derivatives(const vector<T> &p) { derivs_.push_back(p); }

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
    const TypeDescription *sub = SCIRun::get_type_description((T*)0);
    TypeDescription::td_vec *subs = scinew TypeDescription::td_vec(1);
    (*subs)[0] = sub;
    td = scinew TypeDescription(HexTricubicHmtScaleFactors<T>::type_name(0), subs, 
				string(__FILE__),
				"SCIRun");
  }
  return td;
}


const int HEXTRICUBICHMTSCALEFACTORS_VERSION = 1;
template <class T>
void
HexTricubicHmtScaleFactors<T>::io(Piostream &stream)
{
  stream.begin_class(type_name(-1), HEXTRICUBICHMTSCALEFACTORS_VERSION);
  Pio(stream, derivs_);
  Pio(stream, scalefactors_);
  stream.end_class();
}

} //namespace SCIRun

#endif // HexTricubicHmtScaleFactors_h
