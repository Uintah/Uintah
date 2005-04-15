//  LatticeGeom.h - A base class for regular geometries with alligned axes
//
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//
//  Copyright (C) 2000 SCI Institute


#ifndef SCI_project_LatticeGeom_h
#define SCI_project_LatticeGeom_h 1

#include <SCICore/Geometry/Vector.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Datatypes/StructuredGeom.h>
#include <SCICore/Datatypes/DiscreteAttrib.h>
#include <SCICore/Containers/LockingHandle.h>
#include <SCICore/Math/MiscMath.h>
#include <SCICore/Geometry/Transform.h>

#include <sstream>
#include <vector>
#include <string>

namespace SCICore {
namespace Datatypes {

using SCICore::Geometry::Vector;
using SCICore::Geometry::Point;
using std::vector;
using std::string;
using SCICore::PersistentSpace::Piostream;
using SCICore::PersistentSpace::PersistentTypeID;
using SCICore::Math::Interpolate;
using SCICore::Geometry::Transform;

  
class LatticeGeom : public StructuredGeom
{
public:

  LatticeGeom();
  LatticeGeom(int x);
  LatticeGeom(int x, int y);
  LatticeGeom(int x, int y, int z);
  LatticeGeom(int x, const Point &a, const Point &b);
  LatticeGeom(int x, int y, const Point &a, const Point &b);
  LatticeGeom(int x, int y, int z, const Point &a, const Point &b);

  ~LatticeGeom();


  virtual void resize(int x);
  virtual void resize(int x, int y);
  virtual void resize(int x, int y, int z);

  // This transform can be directly manipulated.  updateTransform()
  // must be called after to flush the changes to the scaled transform.
  Transform &getUnscaledTransform() { return d_prescale; }
  void updateTransform();  // flushes d_prescale -> d_transform


  inline int getDim() { return d_dim; }
  inline int getSizeX() { return d_nx; }
  inline int getSizeY() { return d_ny; }
  inline int getSizeZ() { return d_nz; }
  virtual void setBoundingBox(BBox &box);

  // Return the transform from attribute space to world space.
  const Transform &getTransform() { return d_transform; }


  // Transform from attribute space to world space.
  inline void ftransform(const Point &p, Point &r);
  virtual void transform(const Point &p, Point &r);
  virtual Point getPoint(int i, int j = 0, int k = 0);
  
  // Transform from world space to attribute space.
  inline void fitransform(const Point &p, Point &r);
  virtual void itransform(const Point &p, Point &r);
  virtual void locate(const Point& p, int &i, int &j, int &k);

  // TODO: This needs to be tested.
  template <class T>
  void interp(DiscreteAttrib<T> *att, const Point &p, T &out);

  template <class T, class A>
  void finterp(A *att, const Point &p, T &out);

  
  inline int clampi(int i);
  inline int clampj(int j);
  inline int clampk(int k);


  // Persistent representation.
  virtual void io(Piostream&);
  static PersistentTypeID type_id;

  virtual string getInfo();

protected:

  virtual bool computeBoundingBox();
  
  // Number of grid lines in each axis.
  int d_dim;
  int d_nx, d_ny, d_nz;

  Transform d_transform;
  Transform d_prescale;
};




void
LatticeGeom::ftransform(const Point &p, Point &r)
{
  r = d_transform.project(p);
}

void
LatticeGeom::fitransform(const Point &p, Point &r)
{
  r = d_transform.unproject(p);
}


#if 1
// FAST, signal error if debugging on, else do nothing.
inline int
LatticeGeom::clampi(int x)
{
  //MASSERT(x >= 0 && x < d_nx, "LATTICE OUT OF BOUNDS");
  if (x >= 0 && x < d_nx) throw "LATTICE OUT OF BOUNDS";
  return x;
}

inline int
LatticeGeom::clampj(int y)
{
  //MASSERT(y >= 0 && y < d_ny, "LATTICE OUT OF BOUNDS");
  if (y >= 0 && y < d_ny) throw "LATTICE OUT OF BOUNDS";
  return y;
}

inline int
LatticeGeom::clampk(int z)
{
  //MASSERT(z >= 0 && z < d_nz, "LATTICE OUT OF BOUNDS");
  if (z >= 0 && z < d_nz) throw "LATTICE OUT OF BOUNDS";
  return z;
}
#elif 0
// Clip to valid region.
inline int
LatticeGeom::clampi(int x)
{
  return Min(Max(0, x), d_nx-1);
}

inline int
LatticeGeom::clampj(int y)
{
  return Min(Max(0, y), d_ny-1);
}

inline int
LatticeGeom::clampk(int z)
{
  return Min(Max(0, z), d_nz-1);
}
#else
// Wrap
inline int
LatticeGeom::clampi(int x)
{
  int m = x % d_nx;
  if (m < 0) m+=d_nx;
  return m;
}

inline int
LatticeGeom::clampj(int y)
{
  int m = y % d_ny;
  if (m < 0) m+=d_ny;
  return m;
}

inline int
LatticeGeom::clampk(int z)
{
  int m = z % d_nz;
  if (m < 0) m+=d_nz;
  return m;
}
#endif



template <class T>
void
LatticeGeom::interp(DiscreteAttrib<T> *att, const Point &p, T &out)
{
  // Transform point into attribute space.
  Point ijk;
  ftransform(p, ijk);

  int i0 = ijk.x();
  int j0 = ijk.y();
  int k0 = ijk.z();

  int i1 = i0+1;
  int j1 = j0+1;
  int k1 = k0+1;

  const double fx = ijk.x() - i0;
  const double fy = ijk.y() - j0;
  const double fz = ijk.z() - k0;

  // Clip all the integer values to [0, n{xyz}) range.
  // Several clip algorithms.  clamp, wrap, defval, other?
  i0 = clampi(i0);
  j0 = clampj(j0);
  k0 = clampk(k0);

  i1 = clampi(i1);
  j1 = clampj(j1);
  k1 = clampk(k1);

  // Might want to transform fx, fy, fz into other filter here.
  // For example, using gaussian weights instead of linear.

#if 1
  // This interpolates using T as intermediate result, which is
  // most likely incorrect for T with low bit values.  This is
  // not particularly numerically stable.
  //
  // For example, 8 bit rgb values should be interpolated by converting
  // to integer values first
  //
  // Makes unnecessary copies of T if (size(T) <= 8).
  T x00 = Interpolate(att->get3(i0, j0, k0),
		      att->get3(i1, j0, k0), fx);
  T x01 = Interpolate(att->get3(i0, j0, k1),
		      att->get3(i1, j0, k1), fx);
  T x10 = Interpolate(att->get3(i0, j1, k0),
		      att->get3(i1, j1, k0), fx);
  T x11 = Interpolate(att->get3(i0, j1, k1),
		      att->get3(i1, j1, k1), fx);
  T y0 = Interpolate(x00, x10, fy);
  T y1 = Interpolate(x01, x11, fy);

  out = Interpolate(y0, y1, fz);
#else
  // Interpolate should be void Interp(dest, s1, s2, fx, omfx);
  // type of dest should be bigger than s1, s2
  //
  // TT is T with more precision.
  //
  // Convert reduces precision from TT to T.
  TT a, b, c, d;
  Interp(a, att->get3(i0, j0, k0), att->get3(i1, j0, k0), fx, omfx);
  Interp(b, att->get3(i0, j1, k0), att->get3(i1, j1, k0), fx, omfx);

  Interp(c, a, b, fy, omfy);

  Interp(a, att->get3(i0, j0, k1), att->get3(i1, j0, k1), fx, omfx);
  Interp(b, att->get3(i0, j1, k1), att->get3(i1, j1, k1), fx, omfx);
  Interp(d, a, b, fy, omfy);

  Interp(a, c, d, fz, 1.0 - fz);
  Convert(out, a);
#endif
}  
  

template <class T, class A>
void
LatticeGeom::finterp(A *att, const Point &p, T &out)
{
  // Transform point into attribute space.
  Point ijk;
  ftransform(p, ijk);

  int i0 = ijk.x();
  int j0 = ijk.y();
  int k0 = ijk.z();

  int i1 = i0+1;
  int j1 = j0+1;
  int k1 = k0+1;

  const double fx = ijk.x() - i0;
  const double fy = ijk.y() - j0;
  const double fz = ijk.z() - k0;

  // Clip all the integer values to [0, n{xyz}) range.
  // Several clip algorithms.  clamp, wrap, defval, other?
  i0 = clampi(i0);
  j0 = clampj(j0);
  k0 = clampk(k0);

  i1 = clampi(i1);
  j1 = clampj(j1);
  k1 = clampk(k1);

  // Might want to transform fx, fy, fz into other filter here.
  // For example, using gaussian weights instead of linear.

#if 1
  // This interpolates using T as intermediate result, which is
  // most likely incorrect for T with low bit values.  This is
  // not particularly numerically stable.
  //
  // For example, 8 bit rgb values should be interpolated by converting
  // to integer values first
  //
  // Makes unnecessary copies of T if (size(T) <= 8).
  T x00 = Interpolate(att->fget3(i0, j0, k0),
		      att->fget3(i1, j0, k0), fx);
  T x01 = Interpolate(att->fget3(i0, j0, k1),
		      att->fget3(i1, j0, k1), fx);
  T x10 = Interpolate(att->fget3(i0, j1, k0),
		      att->fget3(i1, j1, k0), fx);
  T x11 = Interpolate(att->fget3(i0, j1, k1),
		      att->fget3(i1, j1, k1), fx);
  T y0 = Interpolate(x00, x10, fy);
  T y1 = Interpolate(x01, x11, fy);

  out = Interpolate(y0, y1, fz);
#else
  // Interpolate should be void Interp(dest, s1, s2, fx, omfx);
  // type of dest should be bigger than s1, s2
  //
  // TT is T with more precision.
  //
  // Convert reduces precision from TT to T.
  TT a, b, c, d;
  Interp(a, att->fget3(i0, j0, k0), att->fget3(i1, j0, k0), fx, omfx);
  Interp(b, att->fget3(i0, j1, k0), att->fget3(i1, j1, k0), fx, omfx);

  Interp(c, a, b, fy, omfy);

  Interp(a, att->fget3(i0, j0, k1), att->fget3(i1, j0, k1), fx, omfx);
  Interp(b, att->fget3(i0, j1, k1), att->fget3(i1, j1, k1), fx, omfx);
  Interp(d, a, b, fy, omfy);

  Interp(a, c, d, fz, 1.0 - fz);
  Convert(out, a);
#endif
}  

  
} // end Datatypes
} // end SCICore


#endif
