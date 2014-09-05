/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/



/*
 *  Point2d.h: 
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   July 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#ifndef Point2d_h
#define Point2d_h 

#include <Core/Math/MinMax.h>
#include <sgi_stl_warnings_off.h>
#include <string>
#include <iosfwd>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {
using std::string;
    

class TypeDescription;
class RigorousTest;
class Piostream;
class Vector2d;

class Point2d {
  double x_,y_;
public:
  Point2d(const Vector2d& v);
  inline Point2d(double x, double y): x_(x), y_(y) {}
  Point2d(double, double, double, double);
  inline Point2d(const Point2d&);
  inline Point2d();
  int operator==(const Point2d&) const;
  int operator!=(const Point2d&) const;
  inline Point2d& operator=(const Point2d&);
  inline Vector2d operator-(const Point2d&) const;
  Point2d operator+(const Vector2d&) const;
  Point2d operator-(const Vector2d&) const;
  inline Point2d operator*(double) const;
  inline Point2d& operator*=(const double);
  //inline Point2d& operator+=(const Vector&);
  //inline Point2d& operator-=(const Vector&);
  inline Point2d& operator/=(const double);
  inline Point2d operator/(const double) const;
  inline Point2d operator-() const;
  inline double& operator()(int idx);
  inline double operator()(int idx) const;
  inline void x(const double);
  inline double x() const;
  inline void y(const double);
  inline double y() const;
  //inline Vector vector() const;
  //inline Vector asVector() const;
    
  string get_string() const;

  //! support dynamic compilation
  static const string& get_h_file_path();
    
  friend class Vector2d;
  friend inline double Dot(const Point2d&, const Point2d&);
  friend inline double Dot(const Vector2d&, const Point2d&);
  friend inline double Dot(const Point2d&, const Vector2d&);
  //friend inline double Dot(const Point2d&, const Vector2d&);
  friend inline Point2d Min(const Point2d&, const Point2d&);
  friend inline Point2d Max(const Point2d&, const Point2d&);
  friend Point2d Interpolate(const Point2d&, const Point2d&, double);

  friend void Pio( Piostream&, Point2d& );
};

std::ostream& operator<<(std::ostream& os, const Point2d& p);
std::istream& operator>>(std::istream& os, Point2d& p);

} // End namespace SCIRun

// This cannot be above due to circular dependencies
#include <Core/2d/Vector2d.h>

namespace SCIRun {

inline Point2d::Point2d(const Point2d& p)
{
    x_=p.x_;
    y_=p.y_;
}

inline Point2d::Point2d()
{
}

inline Point2d& Point2d::operator=(const Point2d& p)
{
    x_=p.x_;
    y_=p.y_;
    return *this;
}


Vector2d 
Point2d::operator-(const Point2d& p) const
{
  return Vector2d( x_ - p.x_ ,  y_ - p.y_ ) ;
}

inline Point2d Point2d::operator-() const
{
    return Point2d(-x_, -y_);
}

inline Point2d Point2d::operator*(double d) const
{
    return Point2d(x_*d, y_*d);
}

inline Point2d& Point2d::operator*=(const double d)
{
    x_*=d;y_*=d;
    return *this;
}


inline Point2d& Point2d::operator/=(const double d)
{
    x_/=d;
    y_/=d;
    return *this;
}

inline Point2d Point2d::operator/(const double d) const
{
    return Point2d(x_/d,y_/d);
}

inline void Point2d::x(const double d)
{
    x_=d;
}

inline double& Point2d::operator()(int idx) {
  return (&x_)[idx];
}

inline double Point2d::operator()(int idx) const {
  return (&x_)[idx];
}


inline double Point2d::x() const
{
  return x_;
}

inline void Point2d::y(const double d)
{
  y_=d;
}

inline double Point2d::y() const
{
  return y_;
}


inline Point2d Min(const Point2d& p1, const Point2d& p2)
{

  double x=Min(p1.x_, p2.x_);
  double y=Min(p1.y_, p2.y_);
  return Point2d(x,y);
}

inline Point2d Max(const Point2d& p1, const Point2d& p2)
{

  double x=Max(p1.x_, p2.x_);
  double y=Max(p1.y_, p2.y_);
  return Point2d(x,y);
}


inline double Dot(const Point2d& p1, const Point2d& p2)
{
  return p1.x_*p2.x_+p1.y_*p2.y_;
}

const TypeDescription* get_type_description(Point2d*);

} // End namespace SCIRun


#endif //ifndef Geometry_Point2d_h
