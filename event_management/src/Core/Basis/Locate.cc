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
//    File   : Locate.cc
//    Author : Martin Cole, Frank B. Sachse
//    Date   : Oct 09 2005

#include <Core/Basis/Locate.h>
#include <Core/Util/NotFinished.h>

namespace SCIRun {

template <>
Point difference(const Point& interp, const Point& value)
{
  return (interp - value).point();
}

template <>
bool
compare_distance(const Point &interp, const Point &val, 
		 double &cur_d, double dist)
{
  Vector v = interp - val;
  cur_d = v.length();   
  return  cur_d < dist;
}

template <>
bool check_zero(const std::vector<Point> &val) 
{
  std::vector<Point>::const_iterator iter = val.begin();
  while(iter != val.end()) {
    const Point &v=*iter++;
    if (fabs(v.x())>1e-7 || fabs(v.y())>1e-7 || fabs(v.z())>1e-7)
      return false;
  }
  return true;
}


// locate for Point 
template <>
double getnextx3(std::vector<double> &x,  
		 const Point& y, const std::vector<Point>& yd)
{
  double J[9], JInv[9];

  J[0]=yd[0].x();
  J[3]=yd[0].y();
  J[6]=yd[0].z();
  J[1]=yd[1].x(); 
  J[4]=yd[1].y();
  J[7]=yd[1].z();
  J[2]=yd[2].x();
  J[5]=yd[2].y();
  J[8]=yd[2].z();

  InverseMatrix3x3(J, JInv);
   
  const double dx= JInv[0]*y.x()+JInv[1]*y.y()+JInv[2]*y.z();
  const double dy= JInv[3]*y.x()+JInv[4]*y.y()+JInv[5]*y.z();
  const double dz= JInv[6]*y.x()+JInv[7]*y.y()+JInv[8]*y.z();
   
  x[0] -= dx;
  x[1] -= dy;
  x[2] -= dz;

  return sqrt(dx*dx+dy*dy+dz*dz);    
}

// locate for Point
template <>
double getnextx2(std::vector<double> &x, 
		 const Point& y, const std::vector<Point>& yd)
{
  const double J00=yd[0].x();
  const double J10=yd[0].y();
  const double J20=yd[0].z();
  const double J01=yd[1].x();
  const double J11=yd[1].y();
  const double J21=yd[1].z();
  const double detJ=-J10*J01+J10*J21-J00*J21-J11*J20+J11*J00+J01*J20;
  const double F1=y.x();
  const double F2=y.y();
  const double F3=y.z();
 
  if (detJ) {
    double dx=(F2*J01-F2*J21+J21*F1-J11*F1+F3*J11-J01*F3)/detJ;
    double dy=-(-J20*F2+J20*F1+J00*F2+F3*J10-F3*J00-F1*J10)/detJ;
    
    x[0]+=dx;
    x[1]+=dy;
    return  sqrt(dx*dx+dy*dy);
  }
  else
    return 0;
}


// locate for Point
template <>
double getnextx1(std::vector<double> &x, 
		 const Point& y, const std::vector<Point>& yd)
{
  const Point &yd0=yd[0]; 

  const double dx=((yd0.x() ? y.x()/yd0.x() : 0.)+(yd0.y() ? y.y()/yd0.y() : 0.)+(yd0.z() ? y.z()/yd0.z() : 0.))/3.;
  x[0] -= dx;

  return sqrt(dx*dx);
}

} //namespace SCIRun

