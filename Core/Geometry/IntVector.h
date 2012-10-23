/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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
 *  IntVector.h:
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   April 2000
 *
 */

#ifndef Geometry_IntVector_h
#define Geometry_IntVector_h

#include <iosfwd>

#include <Core/Math/MinMax.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>

namespace SCIRun {

class Piostream;

class IntVector {
public:
  inline IntVector() {
  }
  inline ~IntVector() {
  }
  inline IntVector(const IntVector& copy) {
    for (int indx = 0; indx < 3; indx ++)
      value_[indx] = copy.value_[indx];
  }
  inline IntVector& operator=(const IntVector& copy) {
    for (int indx = 0; indx < 3; indx ++)
      value_[indx] = copy.value_[indx];
    return *this;	
  }
  inline explicit IntVector(const Point& p)
  {
    value_[0]=static_cast<int>(p.x());
    value_[1]=static_cast<int>(p.y());
    value_[2]=static_cast<int>(p.z());
  }
  inline bool operator==(const IntVector& a) const {
    return value_[0] == a.value_[0] && value_[1] == a.value_[1] && value_[2] == a.value_[2];
  }
  


  /* Do not use these operators,  STL require < operator in different manner
  inline bool operator>=(const IntVector& a) const { 
    return value_[0] >= a.value_[0] && value_[1] >= a.value_[1] && value_[2] >= a.value_[2]; 
  } 

  inline bool operator<=(const IntVector& a) const { 
    return value_[0] <= a.value_[0] && value_[1] <= a.value_[1] && value_[2] <= a.value_[2]; 
  } 

  inline bool operator>(const IntVector& a) const { 
    return value_[0] > a.value_[0] && value_[1] > a.value_[1] && value_[2] > a.value_[2]; 
  }

  inline bool operator<(const IntVector& a) const { 
    return value_[0] < a.value_[0] && value_[1] < a.value_[1] && value_[2] < a.value_[2]; 
  }*/

  // can test for equality - if this < a and a < this, they are equal
     inline bool operator<(const IntVector& a) const {
       if (value_[2] < a.value_[2])
         return true;
       if (value_[2] > a.value_[2])
         return false;
       if (value_[1] < a.value_[1])
         return true;
       if (value_[1] > a.value_[1])
         return false;
       return value_[0] < a.value_[0];
     }

  inline bool operator!=(const IntVector& a) const {
    return value_[0] != a.value_[0] || value_[1] != a.value_[1] || value_[2] != a.value_[2];
  }

  inline IntVector(int x, int y, int z) {
    value_[0] = x;
    value_[1] = y;
    value_[2] = z;
  }

  inline IntVector operator*(const IntVector& v) const {
    return IntVector(value_[0]*v.value_[0], value_[1]*v.value_[1],
		     value_[2]*v.value_[2]);
  }
  inline IntVector operator/(const IntVector& v) const {
    return IntVector(value_[0]/v.value_[0], value_[1]/v.value_[1],
		     value_[2]/v.value_[2]);
  }
  inline IntVector operator+(const IntVector& v) const {
    return IntVector(value_[0]+v.value_[0], value_[1]+v.value_[1], 
		     value_[2]+v.value_[2]);
  }
  inline IntVector operator-() const {
    return IntVector(-value_[0], -value_[1], -value_[2]);
  }
  inline IntVector operator-(const IntVector& v) const {
    return IntVector(value_[0]-v.value_[0], value_[1]-v.value_[1], 
		     value_[2]-v.value_[2]);
  }

  inline IntVector& operator+=(const IntVector& v) {
    value_[0]+=v.value_[0];
    value_[1]+=v.value_[1];
    value_[2]+=v.value_[2];
    return *this;
  }

  inline IntVector& operator-=(const IntVector& v) {
    value_[0]-=v.value_[0];
    value_[1]-=v.value_[1];
    value_[2]-=v.value_[2];
    return *this;
  }

  // IntVector i(0)=i.x()
  //           i(1)=i.y()
  //           i(2)=i.z()
  //   --tan
  inline int operator()(int i) const {
    return value_[i];
  }

  inline int& operator()(int i) {
    return value_[i];
  }

  inline int operator[](int i) const {
    return value_[i];
  }

  inline int& operator[](int i) {
    return value_[i];
  }

  inline int x() const {
    return value_[0];
  }
  inline int y() const {
    return value_[1];
  }
  inline int z() const {
    return value_[2];
  }
  inline void x(int x) {
    value_[0]=x;
  }
  inline void y(int y) {
    value_[1]=y;
  }
  inline void z(int z) {
    value_[2]=z;
  }
  inline int& modifiable_x() {
    return value_[0];
  }
  inline int& modifiable_y() {
    return value_[1];
  }
  inline int& modifiable_z() {
    return value_[2];
  }
  // get the array pointer
  inline int* get_pointer() {
    return value_;
  }
  inline Vector asVector() const {
    return Vector(value_[0], value_[1], value_[2]);
  }
  inline Point asPoint() const {
    return Point(value_[0], value_[1], value_[2]);
  }
  friend inline Vector operator*(const Vector&, const IntVector&);
  friend inline Vector operator*(const IntVector&, const Vector&);
  friend inline IntVector Abs(const IntVector& v);

   //! support dynamic compilation
  static const string& get_h_file_path();

 SCISHARE friend void Pio( Piostream&, IntVector& );

 SCISHARE friend std::ostream& operator<<(std::ostream&, const SCIRun::IntVector&);

private:
  int value_[3];
}; // end class IntVector

inline Vector operator*(const Vector& a, const IntVector& b) {
  return Vector(a.x()*b.x(), a.y()*b.y(), a.z()*b.z());
}
inline Vector operator*(const IntVector& a, const Vector& b) {
  return Vector(a.x()*b.x(), a.y()*b.y(), a.z()*b.z());
}
inline Vector operator/(const Vector& a, const IntVector& b) {
  return Vector(a.x()/b.x(), a.y()/b.y(), a.z()/b.z());
}
inline IntVector Min(const IntVector& a, const IntVector& b) {
  return IntVector(Min(a.x(), b.x()), Min(a.y(), b.y()), Min(a.z(), b.z()));
}
inline IntVector Max(const IntVector& a, const IntVector& b) {
  return IntVector(Max(a.x(), b.x()), Max(a.y(), b.y()), Max(a.z(), b.z()));
}
inline IntVector Abs(const IntVector& v)
{
    int x=v.value_[0]<0?-v.value_[0]:v.value_[0];
    int y=v.value_[1]<0?-v.value_[1]:v.value_[1];
    int z=v.value_[2]<0?-v.value_[2]:v.value_[2];
    return IntVector(x,y,z);
}

/**
* Returns true if the given ranges intersect
*/
inline bool doesIntersect(const IntVector& low1, const IntVector &high1, const IntVector& low2, const IntVector &high2)
{
  return low1.x()<high2.x() && 
         low1.y()<high2.y() && 
         low1.z()<high2.z() &&    // intersect if low1 is less than high2 
         high1.x()>low2.x() && 
         high1.y()>low2.y() && 
         high1.z()>low2.z();     // and high1 is greater than their low2
}

// This will round the Vector v to the nearest integer
inline IntVector roundNearest(const Vector& v)
{
   int x =  (v.x() < 0) ? (int) (v.x() - 0.5) : (int) (v.x() +0.5);
   int y =  (v.y() < 0) ? (int) (v.y() - 0.5) : (int) (v.y() +0.5);
   int z =  (v.z() < 0) ? (int) (v.z() - 0.5) : (int) (v.z() +0.5);
   
   IntVector ret(x,y,z);
   return ret;
}

} // End namespace SCIRun

#endif
