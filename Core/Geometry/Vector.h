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
 *  Vector.h: ?
 *
 *  Written by:
 *   Author?
 *   Department of Computer Science
 *   University of Utah
 *   Date?
 *
 *  Copyright (C) 199? SCI Group
 */

#ifndef Geometry_Vector_h
#define Geometry_Vector_h 1

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

class SCICORESHARE Vector {
  double _x,_y,_z;
public:
  inline explicit Vector(const Point&);
  inline Vector(double x, double y, double z): _x(x), _y(y), _z(z)
  { }
  inline Vector(const Vector&);
  inline Vector();
  inline explicit Vector(double init) : _x(init), _y(init), _z(init) {}
  inline double length() const;
  inline double length2() const;
  friend SCICORESHARE inline double Dot(const Vector&, const Vector&);
  friend SCICORESHARE inline double Dot(const Point&, const Vector&);
  friend SCICORESHARE inline double Dot(const Vector&, const Point&);
  inline Vector& operator=(const Vector&);

#ifdef COMMENT_OUT
  /* !!!
  () index from 0
  [] index from 0
  !!! */

  //Note vector(0)=vector.x();vector(1)=vector.y();vector(2)=vector.z()
  inline double& operator()(int idx) {
    // Ugly, but works
    return (&_x)[idx];
  }

  //Note vector(0)=vector.x();vector(1)=vector.y();vector(2)=vector.z()
  inline double operator()(int idx) const {
    // Ugly, but works
    return (&_x)[idx];
  }
#endif

  //Note vector[0]=vector.x();vector[1]=vector.y();vector[2]=vector.z()
  inline double& operator[](int idx) {
    // Ugly, but works
    return (&_x)[idx];
  }

  //Note vector[0]=vector.x();vector[1]=vector.y();vector[2]=vector.z()
  inline double operator[](int idx) const {
    // Ugly, but works
    return (&_x)[idx];
  }

  // checks if one vector is exactly the same as another
  int operator==(const Vector&) const;
  int operator!=(const Vector&) const;

  inline Vector operator*(const double) const;
  inline Vector operator*(const Vector&) const;
  inline Vector& operator*=(const double);
  inline Vector& operator*=(const Vector&);
  inline Vector operator/(const double) const;
  inline Vector operator/(const Vector&) const;
  inline Vector& operator/=(const double);
  inline Vector operator+(const Vector&) const;
  inline Vector& operator+=(const Vector&);
  inline Vector operator-() const;
  inline Vector operator-(const Vector&) const;
  inline Vector& operator-=(const Vector&);
  inline double normalize();
  inline double safe_normalize();
  Vector normal() const;
  friend SCICORESHARE inline Vector Cross(const Vector&, const Vector&);
  friend SCICORESHARE inline Vector Abs(const Vector&);
  inline void x(double);
  inline double x() const;
  inline void y(double);
  inline double y() const;
  inline void z(double);
  inline double z() const;

  inline void u(double);
  inline double u() const;
  inline void v(double);
  inline double v() const;
  inline void w(double);
  inline double w() const;

  void rotz90(const int);
    
  string get_string() const;

  //! support dynamic compilation
  static const string& get_h_file_path();

  friend class Point;
  friend class Transform;
    
  friend SCICORESHARE inline Vector Interpolate(const Vector&, const Vector&, double);
    
  void find_orthogonal(Vector&, Vector&) const;
  bool check_find_orthogonal(Vector&, Vector&) const;

  friend SCICORESHARE void Pio( Piostream&, Vector& );

  inline const Point &point() const;
  inline Point &asPoint() const;
  inline double minComponent() const {
    if(_x<_y){
      if(_x<_z)
	return _x;
      else
	return _z;
    } else {
      if(_y<_z)
	return _y;
      else
	return _z;
    }
  }
  inline double maxComponent() const {
    if(_x>_y){
      if(_x>_z)
	return _x;
      else
	return _z;
    } else {
      if(_y>_z)
	return _y;
      else
	return _z;
    }
  }

  inline void Set(double x, double y, double z)
    { 
      _x = x;
      _y = y;
      _z = z;
    }
      
};

SCICORESHARE std::ostream& operator<<(std::ostream& os, const Vector& p);
SCICORESHARE std::istream& operator>>(std::istream& os, Vector& p);

} // End namespace SCIRun

// This cannot be above due to circular dependencies
#include <Core/Geometry/Point.h>

namespace SCIRun {

inline Vector::Vector(const Point& p)
    : _x(p._x), _y(p._y), _z(p._z)
{
}

inline Vector::Vector()
{
}

inline Vector::Vector(const Vector& p)
{
    _x=p._x;
    _y=p._y;
    _z=p._z;
}

inline double Vector::length2() const
{
    return _x*_x+_y*_y+_z*_z;
}

inline Vector& Vector::operator=(const Vector& v)
{
    _x=v._x;
    _y=v._y;
    _z=v._z;
    return *this;
}

inline Vector Vector::operator*(const double s) const
{
    return Vector(_x*s, _y*s, _z*s);
}

inline Vector& Vector::operator*=(const Vector& v)
{
  _x *= v._x;
  _y *= v._y;
  _z *= v._z;
  return *this;
}

// Allows for double * Vector so that everything doesn't have to be
// Vector * double
inline Vector operator*(const double s, const Vector& v) {
    return v*s;
}

inline Vector Vector::operator/(const double d) const
{
    return Vector(_x/d, _y/d, _z/d);
}

inline Vector Vector::operator/(const Vector& v2) const
{
    return Vector(_x/v2._x, _y/v2._y, _z/v2._z);
}

inline Vector Vector::operator+(const Vector& v2) const
{
    return Vector(_x+v2._x, _y+v2._y, _z+v2._z);
}

inline Vector Vector::operator*(const Vector& v2) const
{
    return Vector(_x*v2._x, _y*v2._y, _z*v2._z);
}

inline Vector Vector::operator-(const Vector& v2) const
{
    return Vector(_x-v2._x, _y-v2._y, _z-v2._z);
}

inline Vector& Vector::operator+=(const Vector& v2)
{
    _x+=v2._x;
    _y+=v2._y;
    _z+=v2._z;
    return *this;
}

inline Vector& Vector::operator-=(const Vector& v2)
{
    _x-=v2._x;
    _y-=v2._y;
    _z-=v2._z;
    return *this;
}

inline Vector Vector::operator-() const
{
    return Vector(-_x,-_y,-_z);
}

inline double Vector::length() const
{
    return Sqrt(_x*_x+_y*_y+_z*_z);
}

inline Vector Abs(const Vector& v)
{
    double x=v._x<0?-v._x:v._x;
    double y=v._y<0?-v._y:v._y;
    double z=v._z<0?-v._z:v._z;
    return Vector(x,y,z);
}

inline Vector Cross(const Vector& v1, const Vector& v2)
{
    return Vector(
	v1._y*v2._z-v1._z*v2._y,
	v1._z*v2._x-v1._x*v2._z,
	v1._x*v2._y-v1._y*v2._x);
}

inline Vector Interpolate(const Vector& v1, const Vector& v2,
			  double weight)
{
    double weight1=1.0-weight;
    return Vector(
	v2._x*weight+v1._x*weight1,
	v2._y*weight+v1._y*weight1,
	v2._z*weight+v1._z*weight1);
}

inline Vector& Vector::operator*=(const double d)
{
    _x*=d;
    _y*=d;
    _z*=d;
    return *this;
}

inline Vector& Vector::operator/=(const double d)
{
    _x/=d;
    _y/=d;
    _z/=d;
    return *this;
}

inline void Vector::x(double d)
{
    _x=d;
}

inline double Vector::x() const
{
    return _x;
}

inline void Vector::y(double d)
{
    _y=d;
}

inline double Vector::y() const
{
    return _y;
}

inline void Vector::z(double d)
{
    _z=d;
}

inline double Vector::z() const
{
    return _z;
}



inline void Vector::u(double d)
{
    _x=d;
}

inline double Vector::u() const
{
    return _x;
}

inline void Vector::v(double d)
{
    _y=d;
}

inline double Vector::v() const
{
    return _y;
}

inline void Vector::w(double d)
{
    _z=d;
}

inline double Vector::w() const
{
    return _z;
}

inline double Dot(const Vector& v1, const Vector& v2)
{
    return v1._x*v2._x+v1._y*v2._y+v1._z*v2._z;
}

inline double Dot(const Vector& v, const Point& p)
{
    return v._x*p._x+v._y*p._y+v._z*p._z;
}

inline
double Vector::normalize()
{
    double l2=_x*_x+_y*_y+_z*_z;
    double l=Sqrt(l2);
    ASSERT(l>0.0);
    _x/=l;
    _y/=l;
    _z/=l;
    return l;
}

inline
double Vector::safe_normalize()
{
  double l=Sqrt(_x*_x + _y*_y + _z*_z + 1.0e-12);
  _x/=l;
  _y/=l;
  _z/=l;
  return l;
}


inline const Point &Vector::point() const {
    return (const Point &)(*this);
}

inline Point &Vector::asPoint() const {
    return (Point &)(*this);
}


inline SCICORESHARE Vector Min(const Vector &v1, const Vector &v2)
{
  return Vector(Min(v1.x(), v2.x()),
		Min(v1.y(), v2.y()),
		Min(v1.z(), v2.z()));
}

inline SCICORESHARE Vector Max(const Vector &v1, const Vector &v2)
{
  return Vector(Max(v1.x(), v2.x()),
		Max(v1.y(), v2.y()),
		Max(v1.z(), v2.z()));
}

const TypeDescription* get_type_description(Vector*);

} // End namespace SCIRun


#endif
