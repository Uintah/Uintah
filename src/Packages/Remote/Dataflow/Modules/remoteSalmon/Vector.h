#ifndef _Vector_h
#define _Vector_h

#include <SCICore/Util/Assert.h>

#include <SCICore/Math/MiscMath.h>
#include <SCICore/Math/MinMax.h>
#include <SCICore/Math/Trig.h>
#include <SCICore/Math/Expon.h>

#include <string>
#include <iostream>
using namespace std;

// #define SCI_VEC_DEBUG
#ifdef SCI_VEC_DEBUG
#define ASSERTVEC(x) ASSERTL0(x)
#else
#define ASSERTVEC(x)
#endif

using namespace SCICore::Math;

class dVector
{
public:
  double x, y, z;
private:
#ifdef SCI_VEC_DEBUG
  int init;
#endif
public:
  inline dVector(double dx, double dy, double dz): x(dx), y(dy), z(dz)
#ifdef SCI_VEC_DEBUG
    , init(true)
#endif
  { }
  inline dVector(const dVector&);
  inline dVector();

  inline dVector operator-() const;
  inline dVector operator*(const double) const;
  inline dVector& operator*=(const double);
  inline dVector operator/(const double) const;
  inline dVector& operator/=(const double);
  inline dVector operator+(const dVector&) const;
  inline dVector& operator+=(const dVector&);
  inline dVector operator-(const dVector&) const;
  inline dVector& operator-=(const dVector&);
  inline dVector& operator=(const dVector&);

  inline bool operator<(const dVector&) const; // These were added to appease STL's vectors.
  inline bool operator==(const dVector&) const;
  inline bool operator!=(const dVector&) const;
	
  inline double length() const;
  inline double length2() const;
  inline double normalize();
  inline dVector normal() const;
	
  friend inline double Dot(const dVector&, const dVector&);
  friend inline dVector Cross(const dVector&, const dVector&);
  friend inline dVector Abs(const dVector&); // Component-wise absolute value.
  friend inline dVector CompMult(const dVector&, const dVector&); // Component-wise multiply.
	
				// Find the point p in terms of the
				// u,v,w basis.
  friend inline dVector Solve(const dVector &p, const dVector &u, const dVector &v, const dVector &w);

				// Compute the plane (N and D) given
				// the triangle.
  inline void ComputePlane(const dVector &V0, const dVector &V1,
    const dVector &V2, dVector &N, double &D);

  friend inline dVector LinearInterp(const dVector&, const dVector&, double);
  friend inline dVector CubicInterp(const dVector&, const dVector&, double);
	
  void find_orthogonal(dVector&, dVector&) const;
	
  string print() const;
};

inline ostream& operator<<(ostream& os, const dVector& p);
inline istream& operator>>(istream& is, dVector& p);

//////////////////////////////////////////////////////////////////////
// Implementation of all inline functions.

inline dVector::dVector()
{
#ifdef SCI_VEC_DEBUG
  init=false;
#endif
}

inline dVector::dVector(const dVector& p)
{
  x=p.x;
  y=p.y;
  z=p.z;
#ifdef SCI_VEC_DEBUG
  init=true;
#endif
}

inline double dVector::length() const
{
#ifdef SCI_VEC_DEBUG
  ASSERTVEC(init);
#endif
  return Sqrt(x*x+y*y+z*z);
}

inline double dVector::length2() const
{
  ASSERTVEC(init);
  return x*x+y*y+z*z;
}

inline dVector& dVector::operator=(const dVector& v)
{
  ASSERTVEC(v.init);
  x=v.x;
  y=v.y;
  z=v.z;
#ifdef SCI_VEC_DEBUG
  init=true;
#endif
  return *this;
}

inline dVector dVector::operator*(const double s) const
{
  ASSERTVEC(init);
  return dVector(x*s, y*s, z*s);
}

inline dVector dVector::operator/(const double d) const
{
  ASSERTVEC(init);
  return dVector(x/d, y/d, z/d);
}

inline dVector dVector::operator+(const dVector& v) const
{
  ASSERTVEC(init && v.init);
  return dVector(x+v.x, y+v.y, z+v.z);
}

inline dVector dVector::operator-(const dVector& v) const
{
  ASSERTVEC(init && v.init);
  return dVector(x-v.x, y-v.y, z-v.z);
}

inline dVector& dVector::operator+=(const dVector& v)
{
  ASSERTVEC(init && v.init);
  x+=v.x;
  y+=v.y;
  z+=v.z;
  return *this;
}

inline dVector& dVector::operator-=(const dVector& v)
{
  ASSERTVEC(init && v.init);
  x-=v.x;
  y-=v.y;
  z-=v.z;
  return *this;
}

inline dVector dVector::operator-() const
{
  ASSERTVEC(init);
  return dVector(-x, -y, -z);
}

inline dVector Abs(const dVector& v)
{
  ASSERTVEC(v.init);
  double x=Abs(v.x);
  double y=Abs(v.y);
  double z=Abs(v.z);
  return dVector(x, y, z);
}

inline dVector CompMult(const dVector& v1, const dVector& v2)
{
  ASSERTVEC(v1.init && v2.init);
  return dVector(v1.x*v2.x, v1.y*v2.y, v1.z*v2.z);
}

inline dVector Cross(const dVector& v1, const dVector& v2)
{
  ASSERTVEC(v1.init && v2.init);
  return dVector(
		v1.y*v2.z-v1.z*v2.y,
		v1.z*v2.x-v1.x*v2.z,
		v1.x*v2.y-v1.y*v2.x);
}

inline dVector LinearInterp(const dVector& v1, const dVector& v2, double t)
{
  ASSERTVEC(v1.init && v2.init);
  return dVector(v1.x+(v2.x-v1.x)*t,
    v1.y+(v2.y-v1.y)*t,
    v1.z+(v2.z-v1.z)*t);
}

inline dVector CubicInterp(const dVector& v1, const dVector& v2, double f)
{
  ASSERTVEC(v1.init && v2.init);

  double t = f*f*(3-2*f);
  return dVector(v1.x+(v2.x-v1.x)*t,
    v1.y+(v2.y-v1.y)*t,
    v1.z+(v2.z-v1.z)*t);
}

inline dVector& dVector::operator*=(const double d)
{
  ASSERTVEC(init);
  x*=d;
  y*=d;
  z*=d;
  return *this;
}

inline dVector& dVector::operator/=(const double d)
{
  ASSERTVEC(init);
  double dinv = 1.0 / d;
  x*=dinv;
  y*=dinv;
  z*=dinv;
  return *this;
}

inline bool dVector::operator==(const dVector& v) const
{
  ASSERTVEC(init && v.init);
  return v.x == x && v.y == y && v.z == z;
}

inline bool dVector::operator!=(const dVector& v) const
{
  ASSERTVEC(init && v.init);
  return v.x != x || v.y != y || v.z != z;
}

inline bool dVector::operator<(const dVector& v) const
{
  ASSERTVEC(init && v.init);
  return v.length2() < length2();
}

inline double Dot(const dVector& v1, const dVector& v2)
{
  ASSERTVEC(v1.init && v2.init);
  return v1.x*v2.x+v1.y*v2.y+v1.z*v2.z;
}

inline double dVector::normalize()
{
  ASSERTVEC(init);
  double l2=x*x+y*y+z*z;
  ASSERTVEC(l2 >= 0.0);
  double len=Sqrt(l2);
  ASSERTVEC(len > 0.0);
  double linv = 1./len;
  x*=linv;
  y*=linv;
  z*=linv;
  return len;
}

inline dVector dVector::normal() const
{
  ASSERTVEC(init);
  dVector v(*this);
  v.normalize();
  return v;
}

#define PRDIG 9

inline string dVector::print() const
{
  char xx[40], yy[40], zz[40];
  return string("[") + gcvt(x, PRDIG, xx) +
    string(", ") + gcvt(y, PRDIG, yy) +
    string(", ") + gcvt(z, PRDIG, zz)
#ifdef SCI_VEC_DEBUG
    + (init ? "" : " **uninit!")
#endif
    + "]";
}

inline void dVector::find_orthogonal(dVector& v1, dVector& v2) const
{
  ASSERTVEC(init);
  dVector v0(Cross(*this, dVector(1,0,0)));
  if(v0.length2() == 0){
    v0=Cross(*this, dVector(0,1,0));
  }
  v1=Cross(*this, v0);
  v1.normalize();
  v2=Cross(*this, v1);
  v2.normalize();
}

inline dVector Solve(const dVector &p, const dVector &u, const dVector &v, const dVector &w)
{
  double det = 1/(w.z * u.x * v.y  -  w.z * u.y * v.x  -
    u.z * w.x * v.y  -  u.x * v.z * w.y  +
    v.z * w.x * u.y  +  u.z * v.x * w.y);
  
  return dVector(((v.x * w.y - w.x * v.y) * p.z +
    (v.z * w.x - v.x * w.z) * p.y + (w.z * v.y - v.z * w.y) * p.x) * det,
    - ((u.x * w.y - w.x * u.y) * p.z + (u.z * w.x - u.x * w.z) * p.y +
      (u.y * w.z - u.z * w.y) * p.x) * det,
    ((u.x * v.y - u.y * v.x) * p.z + (v.z * u.y - u.z * v.y) * p.x +
      (u.z * v.x - u.x * v.z) * p.y) * det);
}

// Given three points, find the plane that they lie on.
inline void ComputePlane(const dVector &V0, const dVector &V1,
  const dVector &V2,  dVector &N, double &D)
{
  dVector E0 = V1 - V0;
  E0.normalize();

  dVector E1 = V2 - V0;
  E1.normalize();

  N = Cross(E0, E1);
  N.normalize();

  D = -Dot(V0, N);

  // cerr << D << " " << Dot(V1, N) << " " << Dot(V2, N) << endl;
}

ostream& operator<<(ostream& os, const dVector& v)
{
  os << v.print();
  return os;
}

istream& operator>>(istream& is, dVector& v)
{
  double x, y, z;
  char st;
  is >> st >> x >> st >> y >> st >> z >> st;
  v=dVector(x,y,z);
  return is;
}

#endif
