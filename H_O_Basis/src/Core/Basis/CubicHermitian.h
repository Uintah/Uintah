//  The contents of this file are subject to the University of Utah Public
//  License (the "License"); you may not use this file except in compliance
//  with the License.
//  
//  Software distributed under the License is distributed on an "AS IS"
//  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
//  License for the specific language governing rights and limitations under
//  the License.
//  
//  The Original Source Code is SCIRun, released March 12, 2001.
//  
//  The Original Source Code was developed by the University of Utah.
//  Portions created by UNIVERSITY are Copyright (C) 2001, 1994
//  University of Utah. All Rights Reserved.
//  
//    File   : CubicHermitian.h
//    Author : Martin Cole
//    Date   : Mon Mar  8 13:17:22 2004

#if !defined(CubicHermitian_h)
#define CubicHermitian_h

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
class CubicHermitianBase : public Datatype {
public:
  CubicHermitianBase() {}
  virtual ~CubicHermitianBase() {}
  
  int order() const { return 3; }
protected:

  //! Functions for interpolating value in cell.
  inline 
  double psi01(double xi) const {
    return 1.0 - 3 * (xi*xi) + 2 * (xi*xi*xi);
  } 
  inline 
  double psi02(double xi) const {
    return (xi*xi) * (3 - (2*xi));
  }
  inline
  double psi11(double xi) const {
    return xi * ((xi-1) * (xi -1));
  }
  inline
  double psi12(double xi) const {
    return (xi*xi) * (xi - 1);
  }
  
  //! Functions for interpolating derivative in cell
  inline
  double psi_deriv_01(double xi) const {
    return -6. * xi + 6. * (xi*xi);
  }
  inline
  double psi_deriv_02(double xi) const {
    return 6.*xi - 6.*(xi*xi);
  }
  inline 
  double psi_deriv_11(double xi) const {
    return 3. * (xi*xi) - 4.*xi + 1.;
  }
  inline
  double psi_deriv_12(double xi) const {
    return 3. * (xi*xi) - 2.*xi;
  }
  
   //! Functions for interpolating second derivative in cell
  inline
  double psi_deriv2_01(double xi) const {
    return -6. + 12. * xi;
  }
  inline
  double psi_deriv2_02(double xi) const {
    return 6. - 12. * xi;
  }
  inline 
  double psi_deriv2_11(double xi) const {
    return 6. * xi - 4.;
  }
  inline
  double psi_deriv2_12(double xi) const {
    return 6. * xi - 2.;
  }

  double distance(const T&, const T&) const;

};



//! ----------------- Base --------------------

template <>
double
CubicHermitianBase<Point>::distance(const Point &p0, const Point &p1) const;

template <class T>
double
CubicHermitianBase<T>::distance(const T &v0, const T &v1) const
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

#endif // CubicHermitian_h
