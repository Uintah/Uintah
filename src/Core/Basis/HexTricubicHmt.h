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

template <class T>
class HexTricubicHmt : public HexApprox
{
public:
  typedef T value_type;

  HexTricubicHmt() {}
  virtual ~HexTricubicHmt() {}
  
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

    return (T)(x12*(1 + 2*x)*y12*(1 + 2*y)*z12*(1 + 2*z)*cd.node0()
      +x12*x*y12*(1 + 2*y)*z12*(1 + 2*z)*derivs_[cd.node0_index()][0]
      +x12*(1 + 2*x)*y12*y*z12*(1 + 2*z)*derivs_[cd.node0_index()][1]
      +x12*(1 + 2*x)*y12*(1 + 2*y)*z12*z*derivs_[cd.node0_index()][2]
      +x12*x*y12*y*z12*(1 + 2*z)*derivs_[cd.node0_index()][3]
      +x12*(1 + 2*x)*y12*y*z12*z*derivs_[cd.node0_index()][4]
      +x12*x*y12*(1 + 2*y)*z12*z*derivs_[cd.node0_index()][5]
      +x12*x*y12*y*z12*z*derivs_[cd.node0_index()][6]
      -(x2*(-3 + 2*x)*y12*(1 + 2*y)*z12*(1 + 2*z))*cd.node1()
      +(-1 + x)*x2*y12*(1 + 2*y)*z12*(1 + 2*z)*derivs_[cd.node1_index()][0]
      -(x2*(-3 + 2*x)*y12*y*z12*(1 + 2*z))*derivs_[cd.node1_index()][1]
      -(x2*(-3 + 2*x)*y12*(1 + 2*y)*z12*z)*derivs_[cd.node1_index()][2]
      +(-1 + x)*x2*y12*y*z12*(1 + 2*z)*derivs_[cd.node1_index()][3]
      -(x2*(-3 + 2*x)*y12*y*z12*z) *derivs_[cd.node1_index()][4]  
      +(-1 + x)*x2*y12*(1 + 2*y)*z12*z*derivs_[cd.node1_index()][5]
      +(-1 + x)*x2*y12*y*z12*z*derivs_[cd.node1_index()][6]
      +x2*(-3 + 2*x)*y2*(-3 + 2*y)*z12*(1 + 2*z)*cd.node2()
      -((-1 + x)*x2*y2*(-3 + 2*y)*z12*(1 + 2*z))*derivs_[cd.node2_index()][0]
      -(x2*(-3 + 2*x)*(-1 + y)*y2*z12*(1 + 2*z))*derivs_[cd.node2_index()][1]
      +x2*(-3 + 2*x)*y2*(-3 + 2*y)*z12*z*derivs_[cd.node2_index()][2]
      +(-1 + x)*x2*(-1 + y)*y2*z12*(1 + 2*z)*derivs_[cd.node2_index()][3]
      -(x2*(-3 + 2*x)*(-1 + y)*y2*z12*z)*derivs_[cd.node2_index()][4]
      -((-1 + x)*x2*y2*(-3 + 2*y)*z12*z)*derivs_[cd.node2_index()][5]
      +(-1 + x)*x2*(-1 + y)*y2*z12*z*derivs_[cd.node2_index()][6] // h
      -(x12*(1 + 2*x)*y2*(-3 + 2*y)*z12*(1 + 2*z))*cd.node3()
      -(x12*x*y2*(-3 + 2*y)*z12*(1 + 2*z))*derivs_[cd.node3_index()][0]
      +x12*(1 + 2*x)*(-1 + y)*y2*z12*(1 + 2*z)*derivs_[cd.node3_index()][1]
      -(x12*(1 + 2*x)*y2*(-3 + 2*y)*z12*z)*derivs_[cd.node3_index()][2]
      +x12*x*(-1 + y)*y2*z12*(1 + 2*z)*derivs_[cd.node3_index()][3]
      +x12*(1 + 2*x)*(-1 + y)*y2*z12*z*derivs_[cd.node3_index()][4]
      -(x12*x*y2*(-3 + 2*y)*z12*z)*derivs_[cd.node3_index()][5]
      +x12*x*(-1 + y)*y2*z12*z*derivs_[cd.node3_index()][6]
      -(x12*(1 + 2*x)*y12*(1 + 2*y)*z2*(-3 + 2*z))*cd.node4()
      -(x12*x*y12*(1 + 2*y)*z2*(-3 + 2*z))*derivs_[cd.node4_index()][0]
      -(x12*(1 + 2*x)*y12*y*z2*(-3 + 2*z))*derivs_[cd.node4_index()][1]
      +x12*(1 + 2*x)*y12*(1 + 2*y)*(-1 + z)*z2*derivs_[cd.node4_index()][2]
      -(x12*x*y12*y*z2*(-3 + 2*z))*derivs_[cd.node4_index()][3]
      +x12*(1 + 2*x)*y12*y*(-1 + z)*z2*derivs_[cd.node4_index()][4]
      +x12*x*y12*(1 + 2*y)*(-1 + z)*z2*derivs_[cd.node4_index()][5]
      +x12*x*y12*y*(-1 + z)*z2*derivs_[cd.node4_index()][6]
      +x2*(-3 + 2*x)*y12*(1 + 2*y)*z2*(-3 + 2*z)*cd.node5()
      -((-1 + x)*x2*y12*(1 + 2*y)*z2*(-3 + 2*z))*derivs_[cd.node5_index()][0]
      +x2*(-3 + 2*x)*y12*y*z2*(-3 + 2*z)*derivs_[cd.node5_index()][1]
      -(x2*(-3 + 2*x)*y12*(1 + 2*y)*(-1 + z)*z2)*derivs_[cd.node5_index()][2]
      -((-1 + x)*x2*y12*y*z2*(-3 + 2*z))*derivs_[cd.node5_index()][3]
      -(x2*(-3 + 2*x)*y12*y*(-1 + z)*z2)*derivs_[cd.node5_index()][4]
      +(-1 + x)*x2*y12*(1 + 2*y)*(-1 + z)*z2*derivs_[cd.node5_index()][5]
      +(-1 + x)*x2*y12*y*(-1 + z)*z2*derivs_[cd.node5_index()][6]
      -(x2*(-3 + 2*x)*y2*(-3 + 2*y)*z2*(-3 + 2*z))*cd.node6()
      +(-1 + x)*x2*y2*(-3 + 2*y)*z2*(-3 + 2*z)*derivs_[cd.node6_index()][0]
      +x2*(-3 + 2*x)*(-1 + y)*y2*z2*(-3 + 2*z)*derivs_[cd.node6_index()][1]
      +x2*(-3 + 2*x)*y2*(-3 + 2*y)*(-1 + z)*z2*derivs_[cd.node6_index()][2]
      -((-1 + x)*x2*(-1 + y)*y2*z2*(-3 + 2*z))*derivs_[cd.node6_index()][3]
      -(x2*(-3 + 2*x)*(-1 + y)*y2*(-1 + z)*z2)*derivs_[cd.node6_index()][4]
      -((-1 + x)*x2*y2*(-3 + 2*y)*(-1 + z)*z2)*derivs_[cd.node6_index()][5]
      +(-1 + x)*x2*(-1 + y)*y2*(-1 + z)*z2*derivs_[cd.node6_index()][6]
      +x12*(1 + 2*x)*y2*(-3 + 2*y)*z2*(-3 + 2*z)*cd.node7()
      +x12*x*y2*(-3 + 2*y)*z2*(-3 + 2*z)*derivs_[cd.node7_index()][0]
      -(x12*(1 + 2*x)*(-1 + y)*y2*z2*(-3 + 2*z))*derivs_[cd.node7_index()][1]
      -(x12*(1 + 2*x)*y2*(-3 + 2*y)*(-1 + z)*z2)*derivs_[cd.node7_index()][2]
      -(x12*x*(-1 + y)*y2*z2*(-3 + 2*z))*derivs_[cd.node7_index()][3]
      +x12*(1 + 2*x)*(-1 + y)*y2*(-1 + z)*z2*derivs_[cd.node7_index()][4]
      -(x12*x*y2*(-3 + 2*y)*(-1 + z)*z2)*derivs_[cd.node7_index()][5]
      +x12*x*(-1 + y)*y2*(-1 + z)*z2*derivs_[cd.node7_index()][6]);
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

    derivs[0]=6*(-1 + x)*x*y12*(1 + 2*y)*z12*(1 + 2*z)*cd.node0()
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
      +(1 - 4*x + 3*x2)*(-1 + y)*y2*(-1 + z)*z2*derivs_[cd.node7_index()][6];
      
    derivs[1]=6*x12*(1 + 2*x)*(-1 + y)*y*z12*(1 + 2*z)*cd.node0()
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
      (-1 + x)*x2*(1 - 4*y + 3*y2)*z12*z*derivs_[cd.node1_index()][6]
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
      +x12*x*y*(-2 + 3*y)*(-1 + z)*z2*derivs_[cd.node7_index()][6];
      
    derivs[2]=6*x12*(1 + 2*x)*y12*(1 + 2*y)*(-1 + z)*z*cd.node0()
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
      +x12*x*(-1 + y)*y2*z*(-2 + 3*z)*derivs_[cd.node7_index()][6];
  }
  

  //! return the parametric coordinates for value within the element.
  template <class CellData>
  void get_coords(vector<double> &coords, const T& value, 
		  const CellData &cd) const  
      {
	HexLocate< HexTricubicHmt<T> > CL;
	CL.get_coords(this, coords, value, cd);
      };
    
  //! add derivative values (dx, dy, dz, dxy, dyz, dzx, dxyz) for nodes.
  void add_derivatives(const vector<T> &p) { derivs_.push_back(p); }

  static  const string type_name(int n = -1);
  virtual void io (Piostream& str);

protected:
  //! Find a reasonably close starting set of parametric coordinates, 
  //! to val.
  template <class CellData>
  void initial_guess(vector<double> &coords, const T &val, 
		     const CellData &cd) const;

  //! next_guess is the next Newton iteration step.
  template <class CellData>
  void next_guess(vector<double> &coords, const T &val, 
		  const CellData &cd) const;

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
    const TypeDescription *sub = SCIRun::get_type_description((T*)0);
    TypeDescription::td_vec *subs = scinew TypeDescription::td_vec(1);
    (*subs)[0] = sub;
    td = scinew TypeDescription(HexTricubicHmt<T>::type_name(0), subs, 
				string(__FILE__),
				"SCIRun");
  }
  return td;
}

const int HEXTRICUBICHMT_VERSION = 1;
template <class T>
void
HexTricubicHmt<T>::io(Piostream &stream)
{
  stream.begin_class(type_name(-1), HEXTRICUBICHMT_VERSION);
  Pio(stream, derivs_);
  stream.end_class();
}

} //namespace SCIRun

#endif // HexTricubicHmt_h
