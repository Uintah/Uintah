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
 *  IntVector.h:
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   April 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#ifndef Geometry_IntVector_h
#define Geometry_IntVector_h

#include <sgi_stl_warnings_off.h>
#include <iosfwd>
#include <sgi_stl_warnings_on.h>
#include <Core/Math/MinMax.h>
#include <Core/Geometry/Vector.h>

namespace SCIRun {

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

  inline bool operator==(const IntVector& a) const {
    return value_[0] == a.value_[0] && value_[1] == a.value_[1] && value_[2] == a.value_[2];
  }

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
  friend inline Vector operator*(const Vector&, const IntVector&);
  friend inline Vector operator*(const IntVector&, const Vector&);
private:
  int value_[3];
};

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
    int x=v.x()<0?-v.x():v.x();
    int y=v.y()<0?-v.y():v.y();
    int z=v.z()<0?-v.z():v.z();
    return IntVector(x,y,z);
}

} // End namespace SCIRun

std::ostream& operator<<(std::ostream&, const SCIRun::IntVector&);


#endif
