//  LatticeGeom.h - A base class for regular geometries with alligned axes
//
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//
//  Copyright (C) 2000 SCI Institute


#ifndef SCI_project_Lattice3Geom_h
#define SCI_project_Lattice3Geom_h 1

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

  
class Lattice3Geom : public StructuredGeom
{
public:

  Lattice3Geom();
  Lattice3Geom(int x, int y, int z);
  Lattice3Geom(int x, int y, int z, const Point &a, const Point &b);

  ~Lattice3Geom();

  virtual string getInfo();

  inline int getSizeX() {return d_nx;};
  inline int getSizeY() {return d_ny;};
  inline int getSizeZ() {return d_nz;};
		    
  virtual void resize(int x, int y, int z);

  virtual void setBoundingBox(BBox &box);
  const Transform &trans() { return d_trans; }

  //////////
  // Return the point relative to the min in the bounding box
  virtual Point getPoint(int, int, int);

  virtual void transform(const Point &p, Point &r);
  virtual void itransform(const Point &p, Point &r);

  inline void ftransform(const Point &p, Point &r);
  inline void fitransform(const Point &p, Point &r);


  template <class T>
  void interp(DiscreteAttrib<T> *att, const Point &p, T &out);

  template <class T, class A>
  void finterp(A *att, const Point &p, T &out);

  
  /////////
  // Return the indexes of the node defining the cell containing p which
  // is closest to the orgin
  //virtual bool locate(const Point& p, int&, int&, int&);

  // Persistent representation...
  virtual void io(Piostream&);
  static PersistentTypeID type_id;

  inline int clampi(int i);
  inline int clampj(int j);
  inline int clampk(int k);

protected:
  virtual bool computeBoundingBox();
  
  // Number of grid lines in each axis.
  int d_nx, d_ny, d_nz;

  Transform d_trans;
};




void
Lattice3Geom::ftransform(const Point &p, Point &r)
{
  r = d_trans.project(p);
}

void
Lattice3Geom::fitransform(const Point &p, Point &r)
{
  r = d_trans.unproject(p);
}


#if 1
// FAST, signal error if debugging on, else do nothing.
inline int
Lattice3Geom::clampi(int x)
{
  //MASSERT(x >= 0 && x < d_nx, "LATTICE OUT OF BOUNDS");
  if (x >= 0 && x < d_nx) throw "LATTICE OUT OF BOUNDS";
  return x;
}

inline int
Lattice3Geom::clampj(int y)
{
  //MASSERT(y >= 0 && y < d_ny, "LATTICE OUT OF BOUNDS");
  if (y >= 0 && y < d_ny) throw "LATTICE OUT OF BOUNDS";
  return y;
}

inline int
Lattice3Geom::clampk(int z)
{
  //MASSERT(z >= 0 && z < d_nz, "LATTICE OUT OF BOUNDS");
  if (z >= 0 && z < d_nz) throw "LATTICE OUT OF BOUNDS";
  return z;
}
#elif 0
// Clip to valid region.
inline int
Lattice3Geom::clampi(int x)
{
  return Min(Max(0, x), d_nx-1);
}

inline int
Lattice3Geom::clampj(int y)
{
  return Min(Max(0, y), d_ny-1);
}

inline int
Lattice3Geom::clampk(int z)
{
  return Min(Max(0, z), d_nz-1);
}
#else
// Wrap
inline int
Lattice3Geom::clampi(int x)
{
  int m = x % d_nx;
  if (m < 0) m+=d_nx;
  return m;
}

inline int
Lattice3Geom::clampj(int y)
{
  int m = y % d_ny;
  if (m < 0) m+=d_ny;
  return m;
}

inline int
Lattice3Geom::clampk(int z)
{
  int m = z % d_nz;
  if (m < 0) m+=d_nz;
  return m;
}
#endif



template <class T>
void
Lattice3Geom::interp(DiscreteAttrib<T> *att, const Point &p, T &out)
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
Lattice3Geom::finterp(A *att, const Point &p, T &out)
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
