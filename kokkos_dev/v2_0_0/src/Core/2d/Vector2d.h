/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/


/*
 *  Vector2d.h: ?
 *
 *  Written by:
 *   Author?
 *   Department of Computer Science
 *   University of Utah
 *   Date?
 *
 *  Copyright (C) 199? SCI Group
 */

#ifndef Geometry_Vector2d_h
#define Geometry_Vector2d_h 1

#include <Core/share/share.h>

#include <Core/Util/Assert.h>
#include <Core/Math/Expon.h>

#include <sgi_stl_warnings_off.h>
#include <string>
#include <iosfwd>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {
using std::string;


class Piostream;
class Point;
class TypeDescription;

class SCICORESHARE Vector2d {
  double x_,y_;
public:
  inline explicit Vector2d(const Point2d&);
  inline Vector2d(double x, double y): x_(x), y_(y)
  { }
  inline Vector2d(const Vector2d&);
  inline Vector2d();
  //inline Vector2d(double init) : x_(init), y_(init) {ASSERT(init==0)}
  inline double length() const;
  inline double length2() const;
  friend SCICORESHARE inline double Dot(const Vector2d&, const Vector2d&);
  friend SCICORESHARE inline double Dot(const Point2d&, const Vector2d&);
  friend SCICORESHARE inline double Dot(const Vector2d&, const Point2d&);
  inline Vector2d& operator=(const Vector2d&);

  /* !!!
  () index from 0
  [] index from 1
  !!! */

  //Note vector(0)=vector.x();vector(1)=vector.x()
  inline double& operator()(int idx) {
    // Ugly, but works
    return (&x_)[idx];
  }

  //Note vector(0)=vector.x();vector(1)=vector.x()
  inline double operator()(int idx) const {
    // Ugly, but works
    return (&x_)[idx];
  }

  //Note vector[1]=vector.x();vector[2]=vector.y()
  inline double& operator[](int idx) {
    // Ugly, but works
    return (&x_)[idx-1];
  }

  //Note vector[1]=vector.x();vector[2]=vector.y()
  inline double operator[](int idx) const {
    // Ugly, but works
    return (&x_)[idx-1];
  }

  // checks if one vector is exactly the same as another
  int operator==(const Vector2d&) const;
  int operator!=(const Vector2d&) const;

  inline Vector2d operator*(const double) const;
  inline Vector2d operator*(const Vector2d&) const;
  inline Vector2d& operator*=(const double);
  inline Vector2d operator/(const double) const;
  inline Vector2d operator/(const Vector2d&) const;
  inline Vector2d& operator/=(const double);
  inline Vector2d operator+(const Vector2d&) const;
  inline Vector2d& operator+=(const Vector2d&);
  inline Vector2d operator-() const;
  inline Vector2d operator-(const Vector2d&) const;
  inline Vector2d& operator-=(const Vector2d&);
  inline double normalize();
  Vector2d normal() const;
/*   friend SCICORESHARE inline Vector2d Cross(const Vector2d&, const Vector2d&); */
  friend SCICORESHARE inline Vector2d Abs(const Vector2d&); 
  inline void x(double);
  inline double x() const;
  inline void y(double);
  inline double y() const;

/*   void rotz90(const int); */
  inline Point2d point() const;
    
  string get_string() const;

  //! support dynamic compilation
  static const string& get_h_file_path();

  friend class Point2d;
    
  //  friend SCICORESHARE inline Vector2d Interpolate(const Vector2d&, const Vector2d&, double);
    
  //void find_orthogonal(Vector2d&, Vector2d&) const;
    
  friend SCICORESHARE void Pio( Piostream&, Vector2d& );

  inline Point2d asPoint() const;
  inline double minComponent() const {
    if(x_<y_)
      return x_;
    else
      return y_;
  }
  inline double maxComponent() const {
    if(x_>y_)
      return x_;
    else
      return y_;
  }
};

SCICORESHARE std::ostream& operator<<(std::ostream& os, const Vector2d& p);
SCICORESHARE std::istream& operator>>(std::istream& os, Vector2d& p);

} // End namespace SCIRun

// This cannot be above due to circular dependencies
#include <Core/Geometry/Point.h>

namespace SCIRun {

inline Vector2d::Vector2d(const Point2d& p)
    : x_(p.x_), y_(p.y_)
{
}

inline Vector2d::Vector2d()
{
}

inline Vector2d::Vector2d(const Vector2d& p)
{
    x_=p.x_;
    y_=p.y_;
}

inline double Vector2d::length2() const
{
    return x_*x_+y_*y_;
}

inline Vector2d& Vector2d::operator=(const Vector2d& v)
{
    x_=v.x_;
    y_=v.y_;
    return *this;
}

inline Vector2d Vector2d::operator*(const double s) const
{
    return Vector2d(x_*s, y_*s);
}

inline Vector2d Vector2d::operator/(const double d) const
{
    return Vector2d(x_/d, y_/d);
}

inline Vector2d Vector2d::operator/(const Vector2d& v2) const
{
    return Vector2d(x_/v2.x_, y_/v2.y_);
}

inline Vector2d Vector2d::operator+(const Vector2d& v2) const
{
    return Vector2d(x_+v2.x_, y_+v2.y_);
}

inline Vector2d Vector2d::operator*(const Vector2d& v2) const
{
    return Vector2d(x_*v2.x_, y_*v2.y_);
}

inline Vector2d Vector2d::operator-(const Vector2d& v2) const
{
    return Vector2d(x_-v2.x_, y_-v2.y_);
}

inline Vector2d& Vector2d::operator+=(const Vector2d& v2)
{
    x_+=v2.x_;
    y_+=v2.y_;
    return *this;
}

inline Vector2d& Vector2d::operator-=(const Vector2d& v2)
{
    x_-=v2.x_;
    y_-=v2.y_;
    return *this;
}

inline Vector2d Vector2d::operator-() const
{
    return Vector2d(-x_,-y_);
}

inline double Vector2d::length() const
{
    return Sqrt(x_*x_+y_*y_);
}

inline Vector2d Abs(const Vector2d& v)
{
    double x=v.x_<0?-v.x_:v.x_;
    double y=v.y_<0?-v.y_:v.y_;
    return Vector2d(x,y);
}

/* inline Vector2d Cross(const Vector2d& v1, const Vector2d& v2) */
/* { */
/*     return Vector2d( */
/* 	v1.y_*v2._z-v1._z*v2.y_, */
/* 	v1._z*v2.x_-v1.x_*v2._z, */
/* 	v1.x_*v2.y_-v1.y_*v2.x_); */
/* } */

/* inline Vector2d Interpolate(const Vector2d& v1, const Vector2d& v2, */
/* 			  double weight) */
/* { */
/*     double weight1=1.0-weight; */
/*     return Vector2d( */
/* 	v2.x_*weight+v1.x_*weight1, */
/* 	v2.y_*weight+v1.y_*weight1, */
/* 	v2._z*weight+v1._z*weight1); */
/* } */

inline Vector2d& Vector2d::operator*=(const double d)
{
    x_*=d;
    y_*=d;
    return *this;
}

inline Vector2d& Vector2d::operator/=(const double d)
{
    x_/=d;
    y_/=d;
    return *this;
}

inline void Vector2d::x(double d)
{
    x_=d;
}

inline double Vector2d::x() const
{
    return x_;
}

inline void Vector2d::y(double d)
{
    y_=d;
}

inline double Vector2d::y() const
{
    return y_;
}



inline Point2d Vector2d::point() const
{
    return Point2d(x_,y_);
}

inline double Dot(const Vector2d& v1, const Vector2d& v2)
{
    return v1.x_*v2.x_+v1.y_*v2.y_;
}

inline double Dot(const Vector2d& v, const Point2d& p)
{
    return v.x_*p.x_+v.y_*p.y_;
}

inline
double Vector2d::normalize()
{
    double l2=x_*x_+y_*y_;
    double l=Sqrt(l2);
    ASSERT(l>0.0);
    x_/=l;
    y_/=l;
    return l;
}

inline Point2d Vector2d::asPoint() const {
    return Point2d(x_,y_);
}


inline SCICORESHARE Vector2d Min(const Vector2d &v1, const Vector2d &v2)
{
  return Vector2d(Min(v1.x(), v2.x()),
		  Min(v1.y(), v2.y()));

}

inline SCICORESHARE Vector2d Max(const Vector2d &v1, const Vector2d &v2)
{
  return Vector2d(Max(v1.x(), v2.x()),
		  Max(v1.y(), v2.y()));
}

const TypeDescription* get_type_description(Vector2d*);

} // End namespace SCIRun


#endif
