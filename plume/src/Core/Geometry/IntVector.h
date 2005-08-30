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

  inline bool operator==(const IntVector& a) const {
    return value_[0] == a.value_[0] && value_[1] == a.value_[1] && value_[2] == a.value_[2];
  }

  // commented out by BJW, nobody uses them, and they can't test for STL equality
/*   inline bool operator>=(const IntVector& a) const { */
/*     return value_[0] >= a.value_[0] && value_[1] >= a.value_[1] && value_[2] >= a.value_[2]; */
/*   } */

/*   inline bool operator<=(const IntVector& a) const { */
/*     return value_[0] <= a.value_[0] && value_[1] <= a.value_[1] && value_[2] <= a.value_[2]; */
/*   } */

/*   inline bool operator>(const IntVector& a) const { */
/*     return value_[0] > a.value_[0] && value_[1] > a.value_[1] && value_[2] > a.value_[2]; */
/*   } */

/*   inline bool operator<(const IntVector& a) const { */
/*     return value_[0] < a.value_[0] && value_[1] < a.value_[1] && value_[2] < a.value_[2]; */
/*   } */

  // can test for equality - if this < a and a < this, they are equal
     inline bool operator<(const IntVector& a) const {
       if (value_[0] < a.value_[0])
         return true;
       if (value_[0] > a.value_[0])
         return false;
       if (value_[1] < a.value_[1])
         return true;
       if (value_[1] > a.value_[1])
         return false;
       return value_[2] < a.value_[2];
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
  friend inline IntVector Abs(const IntVector& v);

   //! support dynamic compilation
  static const string& get_h_file_path();

 friend void Pio( Piostream&, IntVector& );

 friend std::ostream& operator<<(std::ostream&, const SCIRun::IntVector&);

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

} // End namespace SCIRun

#endif
