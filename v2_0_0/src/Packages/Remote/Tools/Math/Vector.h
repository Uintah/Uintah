#ifndef _Vector_h
#define _Vector_h

#include <Packages/Remote/Tools/Util/Assert.h>
#include <Packages/Remote/Tools/Math/MiscMath.h>

#include <string>
#include <iostream>

namespace Remote {
using namespace std;
// #define SCI_VEC_DEBUG
#ifdef SCI_VEC_DEBUG
#define ASSERTVEC(x) ASSERT0(x)
#else
#define ASSERTVEC(x)
#endif

class Vector
{
public:
  double x, y, z;
private:
#ifdef SCI_VEC_DEBUG
  int init;
#endif
public:
  inline Vector(double dx, double dy, double dz): x(dx), y(dy), z(dz)
#ifdef SCI_VEC_DEBUG
    , init(true)
#endif
  { }
  inline Vector(const Vector&);
  inline Vector();

  inline Vector operator-() const;
  inline Vector operator*(const double) const;
  inline Vector& operator*=(const double);
  inline Vector operator/(const double) const;
  inline Vector& operator/=(const double);
  inline Vector operator+(const Vector&) const;
  inline Vector& operator+=(const Vector&);
  inline Vector operator-(const Vector&) const;
  inline Vector& operator-=(const Vector&);
  inline Vector& operator=(const Vector&);

				// These were added to appease STL's
				// vectors.
  inline bool operator<(const Vector&) const; 
  inline bool operator==(const Vector&) const;
  inline bool operator!=(const Vector&) const;
	
  inline double length() const;
  inline double length2() const;
  inline double normalize();
  inline Vector normal() const;
	
  friend inline double Dot(const Vector&, const Vector&);
  friend inline Vector Cross(const Vector&, const Vector&);
				// Core/CCA/Component-wise absolute value.
  friend inline Vector Abs(const Vector&);
				// Core/CCA/Component-wise multiply.
  friend inline Vector CompMult(const Vector&, const Vector&); 
	
				// Find the point p in terms of the
				// u,v,w basis.
  friend inline Vector Solve(const Vector &p, const Vector &u,
    const Vector &v, const Vector &w);
  
				// Compute the plane (N and D) given
				// the triangle.
  inline void ComputePlane(const Vector &V0, const Vector &V1,
    const Vector &V2, Vector &N, double &D);

  friend inline Vector LinearInterp(const Vector&, const Vector&, double);
  friend inline Vector CubicInterp(const Vector&, const Vector&, double);
	
  void find_orthogonal(Vector&, Vector&) const;
	
  string print() const;
};

inline ostream& operator<<(ostream& os, const Vector& p);
inline istream& operator>>(istream& is, Vector& p);

//////////////////////////////////////////////////////////////////////
// Implementation of all inline functions.

inline Vector::Vector()
{
#ifdef SCI_VEC_DEBUG
  init=false;
#endif
}

inline Vector::Vector(const Vector& p)
{
  x=p.x;
  y=p.y;
  z=p.z;
#ifdef SCI_VEC_DEBUG
  init=true;
#endif
}

inline double Vector::length() const
{
#ifdef SCI_VEC_DEBUG
  ASSERTVEC(init);
#endif
  return Sqrt(x*x+y*y+z*z);
}

inline double Vector::length2() const
{
  ASSERTVEC(init);
  return x*x+y*y+z*z;
}

inline Vector& Vector::operator=(const Vector& v)
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

inline Vector Vector::operator*(const double s) const
{
  ASSERTVEC(init);
  return Vector(x*s, y*s, z*s);
}

inline Vector Vector::operator/(const double d) const
{
  ASSERTVEC(init);
  return Vector(x/d, y/d, z/d);
}

inline Vector Vector::operator+(const Vector& v) const
{
  ASSERTVEC(init && v.init);
  return Vector(x+v.x, y+v.y, z+v.z);
}

inline Vector Vector::operator-(const Vector& v) const
{
  ASSERTVEC(init && v.init);
  return Vector(x-v.x, y-v.y, z-v.z);
}

inline Vector& Vector::operator+=(const Vector& v)
{
  ASSERTVEC(init && v.init);
  x+=v.x;
  y+=v.y;
  z+=v.z;
  return *this;
}

inline Vector& Vector::operator-=(const Vector& v)
{
  ASSERTVEC(init && v.init);
  x-=v.x;
  y-=v.y;
  z-=v.z;
  return *this;
}

inline Vector Vector::operator-() const
{
  ASSERTVEC(init);
  return Vector(-x, -y, -z);
}

inline Vector Abs(const Vector& v)
{
  ASSERTVEC(v.init);
  double x=Abs(v.x);
  double y=Abs(v.y);
  double z=Abs(v.z);
  return Vector(x, y, z);
}

inline Vector CompMult(const Vector& v1, const Vector& v2)
{
  ASSERTVEC(v1.init && v2.init);
  return Vector(v1.x*v2.x, v1.y*v2.y, v1.z*v2.z);
}

inline Vector Cross(const Vector& v1, const Vector& v2)
{
  ASSERTVEC(v1.init && v2.init);
  return Vector(
		v1.y*v2.z-v1.z*v2.y,
		v1.z*v2.x-v1.x*v2.z,
		v1.x*v2.y-v1.y*v2.x);
}

inline Vector LinearInterp(const Vector& v1, const Vector& v2, double t)
{
  ASSERTVEC(v1.init && v2.init);
  return Vector(v1.x+(v2.x-v1.x)*t,
    v1.y+(v2.y-v1.y)*t,
    v1.z+(v2.z-v1.z)*t);
}

inline Vector CubicInterp(const Vector& v1, const Vector& v2, double f)
{
  ASSERTVEC(v1.init && v2.init);

  double t = f*f*(3-2*f);
  return Vector(v1.x+(v2.x-v1.x)*t,
    v1.y+(v2.y-v1.y)*t,
    v1.z+(v2.z-v1.z)*t);
}

inline Vector& Vector::operator*=(const double d)
{
  ASSERTVEC(init);
  x*=d;
  y*=d;
  z*=d;
  return *this;
}

inline Vector& Vector::operator/=(const double d)
{
  ASSERTVEC(init);
  double dinv = 1.0 / d;
  x*=dinv;
  y*=dinv;
  z*=dinv;
  return *this;
}

inline bool Vector::operator==(const Vector& v) const
{
  ASSERTVEC(init && v.init);
  return v.x == x && v.y == y && v.z == z;
}

inline bool Vector::operator!=(const Vector& v) const
{
  ASSERTVEC(init && v.init);
  return v.x != x || v.y != y || v.z != z;
}

inline bool Vector::operator<(const Vector& v) const
{
  ASSERTVEC(init && v.init);
  return v.length2() < length2();
}

inline double Dot(const Vector& v1, const Vector& v2)
{
  ASSERTVEC(v1.init && v2.init);
  return v1.x*v2.x+v1.y*v2.y+v1.z*v2.z;
}

inline double Vector::normalize()
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

inline Vector Vector::normal() const
{
  ASSERTVEC(init);
  Vector v(*this);
  v.normalize();
  return v;
}

#define PRDIG 9

inline string Vector::print() const
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

inline void Vector::find_orthogonal(Vector& v1, Vector& v2) const
{
  ASSERTVEC(init);
  Vector v0(Cross(*this, Vector(1,0,0)));
  if(v0.length2() == 0){
    v0=Cross(*this, Vector(0,1,0));
  }
  v1=Cross(*this, v0);
  v1.normalize();
  v2=Cross(*this, v1);
  v2.normalize();
}

inline Vector Solve(const Vector &p, const Vector &u, const Vector &v, const Vector &w)
{
  double det = 1/(w.z * u.x * v.y  -  w.z * u.y * v.x  -
    u.z * w.x * v.y  -  u.x * v.z * w.y  +
    v.z * w.x * u.y  +  u.z * v.x * w.y);
  
  return Vector(((v.x * w.y - w.x * v.y) * p.z +
    (v.z * w.x - v.x * w.z) * p.y + (w.z * v.y - v.z * w.y) * p.x) * det,
    - ((u.x * w.y - w.x * u.y) * p.z + (u.z * w.x - u.x * w.z) * p.y +
      (u.y * w.z - u.z * w.y) * p.x) * det,
    ((u.x * v.y - u.y * v.x) * p.z + (v.z * u.y - u.z * v.y) * p.x +
      (u.z * v.x - u.x * v.z) * p.y) * det);
}

// Given three points, find the plane that they lie on.
inline void ComputePlane(const Vector &V0, const Vector &V1,
  const Vector &V2,  Vector &N, double &D)
{
  Vector E0 = V1 - V0;
  E0.normalize();

  Vector E1 = V2 - V0;
  E1.normalize();

  N = Cross(E0, E1);
  N.normalize();

  D = -Dot(V0, N);

  // cerr << D << " " << Dot(V1, N) << " " << Dot(V2, N) << endl;
}

inline ostream& operator<<(ostream& os, const Vector& v)
{
  os << v.print();
  return os;
}

inline istream& operator>>(istream& is, Vector& v)
{
  double x, y, z;
  char st;
  is >> st >> x >> st >> y >> st >> z >> st;
  v=Vector(x,y,z);
  return is;
}

} // End namespace Remote


#endif
