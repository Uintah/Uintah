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
 *  Point.h: ?
 *
 *  Written by:
 *   Author?
 *   Department of Computer Science
 *   University of Utah
 *   Date?
 *
 *  Copyright (C) 199? SCI Group
 */

#ifndef Geometry_Point_h
#define Geometry_Point_h 1

#include <Core/share/share.h>
#include <Core/Math/MinMax.h>
#include <string>

#include <iosfwd>

namespace SCIRun {
using std::string;
    
class TypeDescription;
class RigorousTest;
class Piostream;
class Vector;

class SCICORESHARE Point {
  double _x,_y,_z;
public:
  inline explicit Point(const Vector& v);
  inline Point(double x, double y, double z): _x(x), _y(y), _z(z) {}
  Point(double, double, double, double);
  inline Point(const Point&);
  inline Point();
  int operator==(const Point&) const;
  int operator!=(const Point&) const;
  inline Point& operator=(const Point&);
  inline Vector operator+(const Point&) const;
  inline Vector operator-(const Point&) const;
  inline Point operator+(const Vector&) const;
  inline Point operator-(const Vector&) const;
  inline Point operator*(double) const;
  inline Point& operator*=(const double);
  inline Point& operator+=(const Vector&);
  inline Point& operator-=(const Vector&);
  inline Point& operator/=(const double);
  inline Point operator/(const double) const;
  inline Point operator-() const;
  inline double& operator()(int idx);
  inline double operator()(int idx) const;
  inline void addscaled(const Point& p, const double scale);  // this += p * w;
  inline void x(const double);
  inline double x() const;
  inline void y(const double);
  inline double y() const;
  inline void z(const double);
  inline double z() const;
  inline const Vector &vector() const;
  inline Vector &asVector() const;
    
  string get_string() const;

  //! support dynamic compilation
  static const string& get_h_file_path();
    
  friend SCICORESHARE class Vector;
  friend SCICORESHARE inline double Dot(const Point&, const Point&);
  friend SCICORESHARE inline double Dot(const Vector&, const Point&);
  friend SCICORESHARE inline double Dot(const Point&, const Vector&);
  //    friend inline double Dot(const Point&, const Vector&);
  friend SCICORESHARE inline Point Min(const Point&, const Point&);
  friend SCICORESHARE inline Point Max(const Point&, const Point&);
  friend SCICORESHARE Point Interpolate(const Point&, const Point&, double);
  friend SCICORESHARE Point AffineCombination(const Point&, double,
					      const Point&, double,
					      const Point&, double,
					      const Point&, double);
  friend SCICORESHARE Point AffineCombination(const Point&, double,
					      const Point&, double,
					      const Point&, double);
  friend SCICORESHARE Point AffineCombination(const Point&, double,
					      const Point&, double);
  friend SCICORESHARE void Pio( Piostream&, Point& );



  // is one point within a small interval of another?

  int Overlap( double a, double b, double e );
  int InInterval( Point a, double epsilon );
    
  static void test_rigorous(RigorousTest* __test);


};

SCICORESHARE std::ostream& operator<<(std::ostream& os, const Point& p);
SCICORESHARE std::istream& operator>>(std::istream& os, Point& p);

} // End namespace SCIRun

// This cannot be above due to circular dependencies
#include <Core/Geometry/Vector.h>

namespace SCIRun {

inline Point::Point(const Vector& v)
    : _x(v._x), _y(v._y), _z(v._z)
{
}

inline Point::Point(const Point& p)
{
    _x=p._x;
    _y=p._y;
    _z=p._z;
}

inline Point::Point()
{
}


inline Point& Point::operator=(const Point& p)
{
    _x=p._x;
    _y=p._y;
    _z=p._z;
    return *this;
}

inline Vector Point::operator+(const Point& p) const
{
    return Vector(_x+p._x, _y+p._y, _z+p._z);
}

inline Vector Point::operator-(const Point& p) const
{
    return Vector(_x-p._x, _y-p._y, _z-p._z);
}

inline Point Point::operator+(const Vector& v) const
{
    return Point(_x+v._x, _y+v._y, _z+v._z);
}

inline Point Point::operator-(const Vector& v) const
{
    return Point(_x-v._x, _y-v._y, _z-v._z);
}

inline Point& Point::operator+=(const Vector& v)
{
    _x+=v._x;
    _y+=v._y;
    _z+=v._z;
    return *this;
}

inline Point& Point::operator-=(const Vector& v)
{
    _x-=v._x;
    _y-=v._y;
    _z-=v._z;
    return *this;
}

inline Point& Point::operator*=(const double d)
{
    _x*=d;
    _y*=d;
    _z*=d;
    return *this;
}

inline Point& Point::operator/=(const double d)
{
    _x/=d;
    _y/=d;
    _z/=d;
    return *this;
}

inline Point Point::operator-() const
{
    return Point(-_x, -_y, -_z);
}

inline Point Point::operator*(double d) const
{
    return Point(_x*d, _y*d, _z*d);
}

inline Point Point::operator/(const double d) const
{
    return Point(_x/d,_y/d,_z/d);
}

inline double& Point::operator()(int idx) {
	return (&_x)[idx];
}

inline double Point::operator()(int idx) const {
	return (&_x)[idx];
}

inline void Point::addscaled(const Point& p, const double scale) {
  // this += p * w;
  _x += p._x * scale;
  _y += p._y * scale;
  _z += p._z * scale;
}

inline void Point::x(const double d)
{
    _x=d;
}

inline double Point::x() const
{
    return _x;
}

inline void Point::y(const double d)
{
    _y=d;
}

inline double Point::y() const
{
    return _y;
}

inline void Point::z(const double d)
{
    _z=d;
}

inline double Point::z() const
{
    return _z;
}

inline const Vector &Point::vector() const
{
    return (const Vector &)(*this);
}

inline Vector &Point::asVector() const
{
    return (Vector &)(*this);
}

inline Point Min(const Point& p1, const Point& p2)
{

  double x=Min(p1._x, p2._x);
  double y=Min(p1._y, p2._y);
  double z=Min(p1._z, p2._z);
  return Point(x,y,z);
}

inline Point Max(const Point& p1, const Point& p2)
{

  double x=Max(p1._x, p2._x);
  double y=Max(p1._y, p2._y);
  double z=Max(p1._z, p2._z);
  return Point(x,y,z);
}

inline double Dot(const Point& p, const Vector& v)
{
    return p._x*v._x+p._y*v._y+p._z*v._z;
}

inline double Dot(const Point& p1, const Point& p2)
{
  return p1._x*p2._x+p1._y*p2._y+p1._z*p2._z;
}

const TypeDescription* get_type_description(Point*);

} // End namespace SCIRun



#endif //ifndef Geometry_Point_h
