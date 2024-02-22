/*
 * The MIT License
 *
 * Copyright (c) 1997-2024 The University of Utah
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
 *  FloatVector.h: ?
 *
 *  Written by:
 *   Author?
 *   Department of Computer Science
 *   University of Utah
 *   Date?
 *
 */

#ifndef Geometry_FloatVector
#define Geometry_FloatVector 1


#include <Core/Util/Assert.h>
#include <Core/Math/Expon.h>

#include   <string>
#include   <iosfwd>

namespace Uintah {

class FloatPoint;
class TypeDescription;

class FloatVector {
  float x_,y_,z_;
public:
  inline explicit FloatVector(const FloatPoint&);
  inline FloatVector(float x, float y, float z): x_(x), y_(y), z_(z)
  { }
  inline FloatVector(const FloatVector&);
  inline FloatVector();
  inline explicit FloatVector(float init) : x_(init), y_(init), z_(init) {}
  inline float length() const;
  inline float length2() const;
  friend inline float Dot(const FloatVector&, const FloatVector&);
  friend inline float Dot(const FloatPoint&, const FloatVector&);
  friend inline float Dot(const FloatVector&, const FloatPoint&);
  inline FloatVector& operator=(const FloatVector&);
  inline FloatVector& operator=(const float&);
  inline FloatVector& operator=(const int&);
  
  static FloatVector fromString( const std::string & source ); // Creates a FloatVector from a string that looksl like "[Num, Num, Num]".

#ifdef COMMENT_OUT
  /* !!!
  () index from 0
  [] index from 0
  !!! */

  //Note vector(0)=vector.x();vector(1)=vector.y();vector(2)=vector.z()
  inline float& operator()(int idx) {
    // Ugly, but works
    return (&x_)[idx];
  }

  //Note vector(0)=vector.x();vector(1)=vector.y();vector(2)=vector.z()
  inline float operator()(int idx) const {
    // Ugly, but works
    return (&x_)[idx];
  }
#endif

  //Note vector[0]=vector.x();vector[1]=vector.y();vector[2]=vector.z()
  inline float& operator[](int idx) {
    // Ugly, but works
    return (&x_)[idx];
  }

  //Note vector[0]=vector.x();vector[1]=vector.y();vector[2]=vector.z()
  inline float operator[](int idx) const {
    // Ugly, but works
    return (&x_)[idx];
  }

  // checks if one vector is exactly the same as another
  int operator==(const FloatVector&) const;
  int operator!=(const FloatVector&) const;

  inline FloatVector operator*(const float) const;
  inline FloatVector operator*(const FloatVector&) const;
  inline FloatVector& operator*=(const float);
  inline FloatVector& operator*=(const FloatVector&);
  inline FloatVector operator/(const float) const;
  inline FloatVector operator/(const FloatVector&) const;
  inline FloatVector& operator/=(const float);
  inline FloatVector operator+(const FloatVector&) const;
  inline FloatVector& operator+=(const FloatVector&);
  inline FloatVector operator-() const;
  inline FloatVector operator-(const FloatVector&) const;
  inline FloatVector operator-(const FloatPoint&) const;
  inline FloatVector& operator-=(const FloatVector&);
  inline float normalize();
  inline float safe_normalize();
  FloatVector normal() const;
  friend inline FloatVector Cross(const FloatVector&, const FloatVector&);
  friend inline FloatVector Abs(const FloatVector&);
  inline void x(float);
  inline float x() const;
  inline void y(float);
  inline float y() const;
  inline void z(float);
  inline float z() const;

  inline void u(float);
  inline float u() const;
  inline void v(float);
  inline float v() const;
  inline void w(float);
  inline float w() const;

  void rotz90(const int);
    
  std::string get_string() const;

  //! support dynamic compilation
  static const std::string& get_h_file_path();

  friend class FloatPoint;
    
  friend inline FloatVector Interpolate(const FloatVector&, const FloatVector&, float);
    
  void find_orthogonal(FloatVector&, FloatVector&) const;
  bool check_find_orthogonal(FloatVector&, FloatVector&) const;


  inline const FloatPoint &point() const;
  inline FloatPoint &asPoint() const;
  inline float minComponent() const {
    if(x_<y_){
      if(x_<z_)
	return x_;
      else
	return z_;
    } else {
      if(y_<z_)
	return y_;
      else
	return z_;
    }
  }
  inline float maxComponent() const {
    if(x_>y_){
      if(x_>z_)
	return x_;
      else
	return z_;
    } else {
      if(y_>z_)
	return y_;
      else
	return z_;
    }
  }

  inline void Set(float x, float y, float z)
    { 
      x_ = x;
      y_ = y;
      z_ = z;
    }
      
  friend std::ostream& operator<<(std::ostream& os, const FloatVector& p);
  friend std::istream& operator>>(std::istream& os, FloatVector& p);

}; // end class FloatVector

// Actual declarations of these functions as 'friend' above doesn't
// (depending on the compiler) actually declare them.

std::ostream& operator<<(std::ostream& os, const FloatVector& p);
std::istream& operator>>(std::istream& os, FloatVector& p);
  
} // End namespace Uintah

// This cannot be above due to circular dependencies
#include <Core/Geometry/FloatPoint.h>

namespace Uintah {


inline FloatVector::FloatVector(const FloatPoint& p)
    : x_(p.x_), y_(p.y_), z_(p.z_)
{
}

inline FloatVector::FloatVector()
{
}

inline FloatVector::FloatVector(const FloatVector& p)
{
    x_=p.x_;
    y_=p.y_;
    z_=p.z_;
}

inline float FloatVector::length2() const
{
    return x_*x_+y_*y_+z_*z_;
}

inline FloatVector& FloatVector::operator=(const FloatVector& v)
{
    x_=v.x_;
    y_=v.y_;
    z_=v.z_;
    return *this;
}

// for initializing in dynamic code
// one often want template<class T> T val = 0.0;

inline FloatVector& FloatVector::operator=(const float& d)
{
  x_ = d;
  y_ = d;
  z_ = d;
  return *this;
}

inline FloatVector& FloatVector::operator=(const int& d)
{
  x_ = static_cast<int>(d);
  y_ = static_cast<int>(d);
  z_ = static_cast<int>(d);
  return *this;
}

inline bool operator<(FloatVector v1, FloatVector v2)
{
  return(v1.length()<v2.length());
}

inline bool operator<=(FloatVector v1, FloatVector v2)
{
  return(v1.length()<=v2.length());
}

inline bool operator>(FloatVector v1, FloatVector v2)
{
  return(v1.length()>v2.length());
}

inline bool operator>=(FloatVector v1, FloatVector v2)
{
  return(v1.length()>=v2.length());
}



inline FloatVector FloatVector::operator*(const float s) const
{
    return FloatVector(x_*s, y_*s, z_*s);
}

inline FloatVector& FloatVector::operator*=(const FloatVector& v)
{
  x_ *= v.x_;
  y_ *= v.y_;
  z_ *= v.z_;
  return *this;
}

// Allows for float * FloatVector so that everything doesn't have to be
// FloatVector * float
inline FloatVector operator*(const float s, const FloatVector& v) {
    return v*s;
}

inline FloatVector FloatVector::operator/(const float d) const
{
    return FloatVector(x_/d, y_/d, z_/d);
}

inline FloatVector FloatVector::operator/(const FloatVector& v2) const
{
    return FloatVector(x_/v2.x_, y_/v2.y_, z_/v2.z_);
}

inline FloatVector FloatVector::operator+(const FloatVector& v2) const
{
    return FloatVector(x_+v2.x_, y_+v2.y_, z_+v2.z_);
}

inline FloatVector FloatVector::operator*(const FloatVector& v2) const
{
    return FloatVector(x_*v2.x_, y_*v2.y_, z_*v2.z_);
}

inline FloatVector FloatVector::operator-(const FloatVector& v2) const
{
    return FloatVector(x_-v2.x_, y_-v2.y_, z_-v2.z_);
}

inline FloatVector FloatVector::operator-(const FloatPoint& v2) const
{
    return FloatVector(x_-v2.x_, y_-v2.y_, z_-v2.z_);
}

inline FloatVector& FloatVector::operator+=(const FloatVector& v2)
{
    x_+=v2.x_;
    y_+=v2.y_;
    z_+=v2.z_;
    return *this;
}

inline FloatVector& FloatVector::operator-=(const FloatVector& v2)
{
    x_-=v2.x_;
    y_-=v2.y_;
    z_-=v2.z_;
    return *this;
}

inline FloatVector FloatVector::operator-() const
{
    return FloatVector(-x_,-y_,-z_);
}

inline float FloatVector::length() const
{
    return Sqrt(x_*x_+y_*y_+z_*z_);
}

inline FloatVector Abs(const FloatVector& v)
{
    float x=v.x_<0?-v.x_:v.x_;
    float y=v.y_<0?-v.y_:v.y_;
    float z=v.z_<0?-v.z_:v.z_;
    return FloatVector(x,y,z);
}

inline FloatVector Cross(const FloatVector& v1, const FloatVector& v2)
{
    return FloatVector(
	v1.y_*v2.z_-v1.z_*v2.y_,
	v1.z_*v2.x_-v1.x_*v2.z_,
	v1.x_*v2.y_-v1.y_*v2.x_);
}

inline FloatVector Interpolate(const FloatVector& v1, const FloatVector& v2,
			  float weight)
{
    float weight1=1.0-weight;
    return FloatVector(
	v2.x_*weight+v1.x_*weight1,
	v2.y_*weight+v1.y_*weight1,
	v2.z_*weight+v1.z_*weight1);
}

inline FloatVector& FloatVector::operator*=(const float d)
{
    x_*=d;
    y_*=d;
    z_*=d;
    return *this;
}

inline FloatVector& FloatVector::operator/=(const float d)
{
    x_/=d;
    y_/=d;
    z_/=d;
    return *this;
}

inline void FloatVector::x(float d)
{
    x_=d;
}

inline float FloatVector::x() const
{
    return x_;
}

inline void FloatVector::y(float d)
{
    y_=d;
}

inline float FloatVector::y() const
{
    return y_;
}

inline void FloatVector::z(float d)
{
    z_=d;
}

inline float FloatVector::z() const
{
    return z_;
}



inline void FloatVector::u(float d)
{
    x_=d;
}

inline float FloatVector::u() const
{
    return x_;
}

inline void FloatVector::v(float d)
{
    y_=d;
}

inline float FloatVector::v() const
{
    return y_;
}

inline void FloatVector::w(float d)
{
    z_=d;
}

inline float FloatVector::w() const
{
    return z_;
}

inline float Dot(const FloatVector& v1, const FloatVector& v2)
{
    return v1.x_*v2.x_+v1.y_*v2.y_+v1.z_*v2.z_;
}

inline float Dot(const FloatVector& v, const FloatPoint& p)
{
    return v.x_*p.x_+v.y_*p.y_+v.z_*p.z_;
}

inline
float FloatVector::normalize()
{
    float l2=x_*x_+y_*y_+z_*z_;
    float l=Sqrt(l2);
    ASSERT(l>0.0);
    x_/=l;
    y_/=l;
    z_/=l;
    return l;
}

inline
float FloatVector::safe_normalize()
{
  float l=Sqrt(x_*x_ + y_*y_ + z_*z_ + 1.0e-12);
  x_/=l;
  y_/=l;
  z_/=l;
  return l;
}


inline const FloatPoint &FloatVector::point() const {
    return (const FloatPoint &)(*this);
}

inline FloatPoint &FloatVector::asPoint() const {
    return (FloatPoint &)(*this);
}


inline FloatVector Min(const FloatVector &v1, const FloatVector &v2)
{
  return FloatVector(Min(v1.x(), v2.x()),
		Min(v1.y(), v2.y()),
		Min(v1.z(), v2.z()));
}

inline FloatVector Max(const FloatVector &v1, const FloatVector &v2)
{
  return FloatVector(Max(v1.x(), v2.x()),
		Max(v1.y(), v2.y()),
		Max(v1.z(), v2.z()));
}

const TypeDescription* get_type_description(FloatVector*);

} // End namespace Uintah


#endif
