/*
 * The MIT License
 *
 * Copyright (c) 1997-2023 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */


/*
 *  FloatPoint.h: ?
 *
 *  Written by:
 *   Author?
 *   Department of Computer Science
 *   University of Utah
 *   Date?
 *
 */

#ifndef Geometry_FloatPoint_h
#define Geometry_FloatPoint_h 1

#include <Core/Math/MinMax.h>

#include <float.h>

#include   <string>
#include   <iosfwd>

namespace Uintah {
    
class TypeDescription;
class RigorousTest;
class FloatVector;

class FloatPoint {
  float x_,y_,z_;
public:
  inline explicit FloatPoint(const FloatVector& v);
  inline FloatPoint(float x, float y, float z): x_(x), y_(y), z_(z) {}
  FloatPoint(float, float, float, float);
  inline FloatPoint(const FloatPoint&);
  inline FloatPoint();
  int operator==(const FloatPoint&) const;
  int operator!=(const FloatPoint&) const;
  inline FloatPoint& operator=(const FloatPoint&);
  inline FloatVector operator+(const FloatPoint&) const;
  inline FloatVector operator-(const FloatPoint&) const;
  inline FloatPoint operator+(const FloatVector&) const;
  inline FloatPoint operator-(const FloatVector&) const;
  inline FloatPoint operator*(float) const;
  inline FloatPoint& operator*=(const float);
  inline FloatPoint& operator+=(const FloatVector&);
  inline FloatPoint& operator-=(const FloatVector&);
  inline FloatPoint& operator/=(const float);
  inline FloatPoint operator/(const float) const;
  inline FloatPoint operator-() const;
  inline float& operator()(int idx);
  inline float operator()(int idx) const;

  inline void addscaled(const FloatPoint& p, const float scale);  // this += p * w;

  inline float x() const;
  inline float y() const;
  inline float z() const;

  inline void x( const float );
  inline void y( const float );
  inline void z( const float );

  inline FloatVector &asVector() const;
  inline FloatVector toVector() const;

  //! support dynamic compilation
  static const std::string& get_h_file_path();
    
  friend class FloatVector;
  friend inline float Dot(const FloatPoint&, const FloatPoint&);
  friend inline float Dot(const FloatVector&, const FloatPoint&);
  friend inline float Dot(const FloatPoint&, const FloatVector&);
  //    friend inline float Dot(const FloatPoint&, const FloatVector&);
  friend inline FloatPoint Min(const FloatPoint&, const FloatPoint&);
  friend inline FloatPoint Max(const FloatPoint&, const FloatPoint&);

  // is one point within a small interval of another?

  int Overlap( float a, float b, float e );
  int InInterval( FloatPoint a, float epsilon );
    
  static void test_rigorous(RigorousTest* __test);

  friend std::ostream& operator<<(std::ostream& os, const FloatPoint& p);
  friend std::istream& operator>>(std::istream& os, FloatPoint& p);

}; // end class FloatPoint


// Actual declarations of these functions as 'friend' above doesn't
// (depending on the compiler) actually declare them.
FloatPoint Interpolate(const FloatPoint&, const FloatPoint&, float);
FloatPoint AffineCombination(const FloatPoint&, float, const FloatPoint&, float,
                                 const FloatPoint&, float, const FloatPoint&, float);
FloatPoint AffineCombination(const FloatPoint&, float, const FloatPoint&, float, const FloatPoint&, float);
FloatPoint AffineCombination(const FloatPoint&, float, const FloatPoint&, float);

std::ostream& operator<<(std::ostream& os, const FloatPoint& p);
std::istream& operator>>(std::istream& os, FloatPoint& p);

inline 
FloatPoint operator*(float d, const FloatPoint &p) {
  return p*d;
}
inline 
FloatPoint operator+(const FloatVector &v, const FloatPoint &p) {
  return p+v;
}

} // End namespace Uintah

// This cannot be above due to circular dependencies
#include <Core/Geometry/FloatVector.h>

namespace Uintah {

inline FloatPoint::FloatPoint(const FloatVector& v)
    : x_(v.x_), y_(v.y_), z_(v.z_)
{
}

inline FloatPoint::FloatPoint(const FloatPoint& p)
{
    x_=p.x_;
    y_=p.y_;
    z_=p.z_;
}

inline FloatPoint::FloatPoint() : x_(DBL_MAX), y_(DBL_MAX), z_(DBL_MAX)
{
}


inline FloatPoint& FloatPoint::operator=(const FloatPoint& p)
{
    x_=p.x_;
    y_=p.y_;
    z_=p.z_;
    return *this;
}

inline FloatVector FloatPoint::operator+(const FloatPoint& p) const
{
    return FloatVector(x_+p.x_, y_+p.y_, z_+p.z_);
}

inline FloatVector FloatPoint::operator-(const FloatPoint& p) const
{
    return FloatVector(x_-p.x_, y_-p.y_, z_-p.z_);
}

inline FloatPoint FloatPoint::operator+(const FloatVector& v) const
{
    return FloatPoint(x_+v.x_, y_+v.y_, z_+v.z_);
}

inline FloatPoint FloatPoint::operator-(const FloatVector& v) const
{
    return FloatPoint(x_-v.x_, y_-v.y_, z_-v.z_);
}

inline FloatPoint& FloatPoint::operator+=(const FloatVector& v)
{
    x_+=v.x_;
    y_+=v.y_;
    z_+=v.z_;
    return *this;
}

inline FloatPoint& FloatPoint::operator-=(const FloatVector& v)
{
    x_-=v.x_;
    y_-=v.y_;
    z_-=v.z_;
    return *this;
}

inline FloatPoint& FloatPoint::operator*=(const float d)
{
    x_*=d;
    y_*=d;
    z_*=d;
    return *this;
}

inline FloatPoint& FloatPoint::operator/=(const float d)
{
    x_/=d;
    y_/=d;
    z_/=d;
    return *this;
}

inline FloatPoint FloatPoint::operator-() const
{
    return FloatPoint(-x_, -y_, -z_);
}

inline FloatPoint FloatPoint::operator*(float d) const
{
    return FloatPoint(x_*d, y_*d, z_*d);
}

inline FloatPoint FloatPoint::operator/(const float d) const
{
    return FloatPoint(x_/d,y_/d,z_/d);
}

inline float& FloatPoint::operator()(int idx) {
	return (&x_)[idx];
}

inline float FloatPoint::operator()(int idx) const {
	return (&x_)[idx];
}

inline void FloatPoint::addscaled(const FloatPoint& p, const float scale) {
  // this += p * w;
  x_ += p.x_ * scale;
  y_ += p.y_ * scale;
  z_ += p.z_ * scale;
}

inline void FloatPoint::x(const float d)
{
    x_=d;
}

inline float FloatPoint::x() const
{
    return x_;
}

inline void FloatPoint::y(const float d)
{
    y_=d;
}

inline float FloatPoint::y() const
{
    return y_;
}

inline void FloatPoint::z(const float d)
{
    z_=d;
}

inline float FloatPoint::z() const
{
    return z_;
}

inline FloatVector &FloatPoint::asVector() const
{
    return (FloatVector &)(*this);
}

inline FloatVector FloatPoint::toVector() const
{
  return FloatVector(x_,y_,z_);
}

inline FloatPoint Min(const FloatPoint& p1, const FloatPoint& p2)
{

  float x=Min(p1.x_, p2.x_);
  float y=Min(p1.y_, p2.y_);
  float z=Min(p1.z_, p2.z_);
  return FloatPoint(x,y,z);
}

inline FloatPoint Max(const FloatPoint& p1, const FloatPoint& p2)
{

  float x=Max(p1.x_, p2.x_);
  float y=Max(p1.y_, p2.y_);
  float z=Max(p1.z_, p2.z_);
  return FloatPoint(x,y,z);
}

inline float Dot(const FloatPoint& p, const FloatVector& v)
{
    return p.x_*v.x_+p.y_*v.y_+p.z_*v.z_;
}

inline float Dot(const FloatPoint& p1, const FloatPoint& p2)
{
  return p1.x_*p2.x_ + p1.y_*p2.y_ + p1.z_*p2.z_;
}

const TypeDescription* get_type_description(FloatPoint*);

} // End namespace Uintah



#endif //ifndef Geometry_FloatPoint_h
