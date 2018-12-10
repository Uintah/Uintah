/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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

#ifndef CORE_GEOMETRY_INTVECTOR_H
#define CORE_GEOMETRY_INTVECTOR_H

#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Math/MinMax.h>

#include <iosfwd>

namespace Uintah {


class IntVector {

public:

  inline  IntVector() {}
  inline ~IntVector() {}

  inline IntVector( const IntVector & copy )
  {
    for (int indx = 0; indx < 3; indx++) {
      m_value[indx] = copy.m_value[indx];
    }
  }

  inline IntVector& operator=( const IntVector & copy )
  {
    for (int indx = 0; indx < 3; indx++) {
      m_value[indx] = copy.m_value[indx];
    }
    return *this;
  }

  inline explicit IntVector( const Point & p )
  {
    m_value[0] = static_cast<int>(p.x());
    m_value[1] = static_cast<int>(p.y());
    m_value[2] = static_cast<int>(p.z());
  }

  // Creates an IntVector from a string that looks like "[1, 2, 3]".
  static IntVector fromString( const std::string & source ); 

  inline bool operator==( const IntVector & a ) const
  {
    return m_value[0] == a.m_value[0] && m_value[1] == a.m_value[1] && m_value[2] == a.m_value[2];
  }
  
  // Used to test for equality - if this < a and a < this, they are equal
  inline bool operator<( const IntVector & a ) const
  {
    if (m_value[2] < a.m_value[2]) {
      return true;
    }
    else if (m_value[2] > a.m_value[2]) {
      return false;
    }
    else if (m_value[1] < a.m_value[1]) {
      return true;
    }
    else if (m_value[1] > a.m_value[1]) {
      return false;
    }
    else {
      return m_value[0] < a.m_value[0];
    }
  }

  inline bool operator!=( const IntVector& a ) const
  {
    return m_value[0] != a.m_value[0] || m_value[1] != a.m_value[1] || m_value[2] != a.m_value[2];
  }

  inline IntVector( int x
                  , int y
                  , int z
                  )
  {
    m_value[0] = x;
    m_value[1] = y;
    m_value[2] = z;
  }

  inline IntVector( int v )
  {
    m_value[0] = v;
    m_value[1] = v;
    m_value[2] = v;
  }

  inline IntVector operator*( const IntVector & v ) const
  {
    return IntVector(m_value[0] * v.m_value[0], m_value[1] * v.m_value[1], m_value[2] * v.m_value[2]);
  }
  
  inline IntVector operator*( const int a ) const
  {
    return IntVector(a * m_value[0], a * m_value[1], a * m_value[2]);
  }

  inline IntVector operator/( const IntVector & v ) const
  {
    return IntVector(m_value[0] / v.m_value[0], m_value[1] / v.m_value[1], m_value[2] / v.m_value[2]);
  }

  inline IntVector operator+( const IntVector & v ) const
  {
    return IntVector(m_value[0] + v.m_value[0], m_value[1] + v.m_value[1], m_value[2] + v.m_value[2]);
  }

  inline IntVector operator-() const
  {
    return IntVector(-m_value[0], -m_value[1], -m_value[2]);
  }

  inline IntVector operator-( const IntVector & v ) const
  {
    return IntVector(m_value[0] - v.m_value[0], m_value[1] - v.m_value[1], m_value[2] - v.m_value[2]);
  }

  inline IntVector& operator+=( const IntVector & v )
  {
    m_value[0] += v.m_value[0];
    m_value[1] += v.m_value[1];
    m_value[2] += v.m_value[2];
    return *this;
  }

  inline IntVector& operator-=( const IntVector & v )
  {
    m_value[0] -= v.m_value[0];
    m_value[1] -= v.m_value[1];
    m_value[2] -= v.m_value[2];
    return *this;
  }

  // IntVector i(0)=i.x()
  //           i(1)=i.y()
  //           i(2)=i.z()
  //   --tan
  inline int  operator()( int i ) const { return m_value[i]; }
  inline int& operator()( int i )       { return m_value[i]; }

  inline int  operator[]( int i ) const { return m_value[i]; }
  inline int& operator[]( int i )       { return m_value[i]; }

  inline int x() const { return m_value[0]; }
  inline int y() const { return m_value[1]; }
  inline int z() const { return m_value[2]; }

  inline void x( int x ) { m_value[0] = x; }
  inline void y( int y ) { m_value[1] = y; }
  inline void z( int z ) { m_value[2] = z; }

  inline int & modifiable_x() { return m_value[0]; }
  inline int & modifiable_y() { return m_value[1]; }
  inline int & modifiable_z() { return m_value[2]; }

  // Get the array pointer:
  inline int * get_pointer() { return m_value; }

  inline Vector asVector() const { return Vector( m_value[0], m_value[1], m_value[2] ); }
  inline Point  asPoint() const {  return Point(  m_value[0], m_value[1], m_value[2] ); }

  friend inline Vector operator*( const Vector &, const IntVector & );
  friend inline Vector operator*( const IntVector &, const Vector & );
  friend inline IntVector Abs( const IntVector & v );

  //! support dynamic compilation
  static const std::string& get_h_file_path();

  friend std::ostream& operator<<( std::ostream &, const Uintah::IntVector & );

private:

  int m_value[3]{0,0,0};

}; // end class IntVector

inline Vector operator*( const Vector    & a
                       , const IntVector & b
                       )
{
  return Vector(a.x() * b.x(), a.y() * b.y(), a.z() * b.z());
}

inline Vector operator*( const IntVector & a
                       , const Vector    & b
                       )
{
  return Vector(a.x() * b.x(), a.y() * b.y(), a.z() * b.z());
}

inline Vector operator/( const Vector    & a
                       , const IntVector & b
                       )
{
  return Vector(a.x() / b.x(), a.y() / b.y(), a.z() / b.z());
}

inline IntVector Min( const IntVector & a
                    , const IntVector & b
                    )
{
  return IntVector(Min(a.x(), b.x()), Min(a.y(), b.y()), Min(a.z(), b.z()));
}

inline IntVector Max( const IntVector & a
                    , const IntVector & b
                    )
{
  return IntVector(Max(a.x(), b.x()), Max(a.y(), b.y()), Max(a.z(), b.z()));
}

inline IntVector Abs(const IntVector& v)
{
  int x = v.m_value[0] < 0 ? -v.m_value[0] : v.m_value[0];
  int y = v.m_value[1] < 0 ? -v.m_value[1] : v.m_value[1];
  int z = v.m_value[2] < 0 ? -v.m_value[2] : v.m_value[2];

  return IntVector(x, y, z);
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
inline IntVector roundNearest( const Vector & v )
{
  int x = (v.x() < 0) ? (int)(v.x() - 0.5) : (int)(v.x() + 0.5);
  int y = (v.y() < 0) ? (int)(v.y() - 0.5) : (int)(v.y() + 0.5);
  int z = (v.z() < 0) ? (int)(v.z() - 0.5) : (int)(v.z() + 0.5);

  IntVector vec(x, y, z);

  return vec;
}

} // End namespace Uintah

#endif // CORE_GEOMETRY_INTVECTOR_H
