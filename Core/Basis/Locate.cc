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

using std::cerr;

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

// locate for Point 
template <>
double getnextx3(vector<double> &x, vector<double> &xold, 
		 const Point& y, const vector<Point>& yd)
{
  double J[9], JInv[9];
  J[0]=yd[0].x();
  J[1]=yd[0].y();
  J[2]=yd[0].z();
  J[3]=yd[1].x();
  J[4]=yd[1].y();
  J[5]=yd[1].z();
  J[6]=yd[2].x();
  J[7]=yd[2].y();
  J[8]=yd[2].z();
  InverseMatrix3x3(J, JInv);
   
  x[0] -= JInv[0]*y.x()+JInv[1]*y.y()+JInv[2]*y.z();
  x[1] -= JInv[3]*y.x()+JInv[4]*y.y()+JInv[5]*y.z();
  x[2] -= JInv[6]*y.x()+JInv[7]*y.y()+JInv[8]*y.z();
   
  const double dx=x[0]-xold[0];
  const double dy=x[1]-xold[1];
  const double dz=x[2]-xold[2];
  return sqrt(dx*dx+dy*dy+dz*dz);    
}

// locate for Point
template <>
double getnextx2(vector<double> &x, vector<double> &xold,
		 const Point& y, const vector<Point>& yd)
{
  NOT_FINISHED("Locate.cc: getnextx2");
  return -1;
}


// locate for Point
template <>
double getnextx1(vector<double> &x, vector<double> &xold,
		 const Point& y, const vector<Point>& yd)
{
  Point yd0=yd[0]; 
  //  cerr << x[0] << "\t" << y.x() << "\t" << y.y() << "\t" << y.z() << "\t" << yd0.x() << "\t" << yd0.y() << "\t" << yd0.z() << "\n" ;

  x[0] -= ((yd0.x() ? y.x()/yd0.x() : 0.)+(yd0.y() ? y.y()/yd0.y() : 0.)+(yd0.z() ? y.z()/yd0.z() : 0.))/3.;
  const double dx=x[0]-xold[0];
  return sqrt(dx*dx);
}

} //namespace SCIRun

