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
//    File   : LinearLagrangian.h
//    Author : Martin Cole
//    Date   : Wed Sep  8 16:29:44 2004

#if !defined(LinearLagrangian_h)
#define LinearLagrangian_h

#include <vector>
#include <string>
#include <Core/Geometry/Point.h>
#include <Core/Util/TypeDescription.h>
#include <Core/Datatypes/Datatype.h>
#include <Core/Geometry/Transform.h>
#include <float.h>

namespace SCIRun {

using std::vector;
using std::string;

template <class T>
class LinearLagrangianBase : public Datatype {
public:
  LinearLagrangianBase() {}
  virtual ~LinearLagrangianBase() {}
  
  int order() const { return 1; }

protected:
  double distance(const T&, const T&) const;
};

template <>
double
LinearLagrangianBase<Point>::distance(const Point &p0, const Point &p1) const;

template <class T>
double
LinearLagrangianBase<T>::distance(const T &v0, const T &v1) const
{
  return fabs(v1 - v0);
}


template <class T>
double calc_next_guess(const double xi, const T &val,
		       const T &interp, const T &deriv, const T &deriv2);

//! f(u) =C'(u) dot (C(u) - P)
//! the distance from P to C(u) is minimum when f(u) = 0
template <>
double calc_next_guess<Point>(const double xi, const Point &val,
			      const Point &interp, const Point &deriv, 
			      const Point &deriv2);
  
template <class T>
double 
calc_next_guess(const double xi, const T &val,
		       const T &interp, const T &deriv, const T &deriv2) 
{
  T num = val - interpolate(xi, cd); 
  T den = derivate(xi, cd);
  return num / den + xi;
}

} //namespace SCIRun

#endif // LinearLagrangian_h
