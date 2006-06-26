// Quadric.h - Classes for dealing with 3D Quadrics.
//
// Peter-Pike Sloan
// Hacked by Dave McAllister, August 1999.

#ifndef _quadric_h
#define _quadric_h

#include <Rendering/ZTex/Matrix44.h>
#include <Rendering/ZTex/Vector.h>

// 3-Ddimensional quadrics (x,y,z generally).
// The matrix looks like this (w/respect to indices).

/*
+--+--+--+--+
+ 0+ 4+ 5+ 6+
+--+--+--+--+
+ 4+ 1+ 7+ 8+
+--+--+--+--+
+ 5+ 7+ 2+ 9+
+--+--+--+--+
+ 6+ 8+ 9+ 3+
+--+--+--+--+

X^2 -> 0
Y^2 -> 1
Z^2 -> 2
C -> 3
XY -> 4
XZ -> 5
X -> 6
YZ -> 7
Y -> 8
Z -> 9
*/

namespace SemotusVisum {
namespace Rendering {

struct Quadric3
{
  double vals[10];

  // XXX There are no constructors.

  // Creates a quadric from the infinite line containing p0 and p1.
  void CreateLine(const Vector &p0, const Vector &p1);

  // Returns the point p that minimizes pQp.
  // returns true if it found a min, false if non-invertible.
  bool FindMin(Vector& p) const;

  inline void zero()
  {
    for(int i=0;i<10;i++)
      vals[i] = 0.0;
  }

  // Generate the matrix associated with the quadric.
  inline Matrix44 matrix44() const
  {
    double mat[4][4];
    mat[0][0] = vals[0]; mat[0][1] = vals[4]; mat[0][2] = vals[5]; mat[0][3] = vals[6]; 
    mat[1][0] = vals[4]; mat[1][1] = vals[1]; mat[1][2] = vals[7]; mat[1][3] = vals[8]; 
    mat[2][0] = vals[5]; mat[2][1] = vals[7]; mat[2][2] = vals[2]; mat[2][3] = vals[9]; 
    mat[3][0] = vals[6]; mat[3][1] = vals[8]; mat[3][2] = vals[9]; mat[3][3] = vals[3]; 

    return Matrix44((double *)mat);
  }

  // Add 2 Quadric3s.
  inline Quadric3 operator+(const Quadric3& q) const
  {
    Quadric3 rval;
    for(int i=0; i<10; i++)
      rval.vals[i] = vals[i] + q.vals[i];
    return rval;
  }

  inline Quadric3 operator+=(const Quadric3& q)
  {
    for(int i=0; i<10; i++)
      vals[i] += q.vals[i];
    return *this;
  }

  // Compute the outer product of this 4-vector with itself.
  // Used to generate a quadric representing a plane.
  inline void DoSym(const double A, const double B, const double C, const double D)
  {
    vals[0] = A*A;
    vals[1] = B*B;
    vals[2] = C*C;
    vals[3] = D*D;

    vals[4] = A*B;
    vals[5] = A*C;
    vals[6] = A*D;

    vals[7] = B*C;
    vals[8] = B*D;

    vals[9] = C*D;
  };

  inline void DoSym(const Vector &N, const double D)
  {
    DoSym(N.x, N.y, N.z, D);
  }

  // Compute p * Q * p.
  inline double MulPt(const Vector& p) const
  {
    return ((p.x*p.x*vals[0] + p.y*p.y*vals[1] + p.z*p.z*vals[2] + vals[3] +
	     2*p.x*p.y*vals[4] + 2*p.x*p.z*vals[5] + 2*p.y*p.z*vals[7] +
	     2*p.x*vals[6] + 2*p.y*vals[8] + 2*p.z*vals[9]));
  }

  // Compute the 4-vector Q * p.
  inline void MulVec(const Vector& p, double res[4]) const
  {
    // Compute Qp
    res[0] = (p.x*vals[0] + p.y*vals[4] + p.z*vals[5] + vals[6]);
    res[1] = (p.x*vals[4] + p.y*vals[1] + p.z*vals[7] + vals[8]);
    res[2] = (p.x*vals[5] + p.y*vals[7] + p.z*vals[2] + vals[9]);
    res[3] = (p.x*vals[6] + p.y*vals[8] + p.z*vals[9] + vals[3]);
  }

  // Using above equation, squared terms multiplied by v, rest by v/2
  // Compute Q' such that pQ'p = v * pQp
  inline void Scale(double v)
  {
    // Squared stuff.
    for(int i=0;i<10;i++)
      vals[i] *= v;
  }
};


} // namespace Tools
} // namespace Remote

#endif
