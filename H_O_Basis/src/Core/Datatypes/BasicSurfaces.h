
/*
 *  BasicSurfaces.h: Cylinders and stuff
 *
 *  Written by:
 *   Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   November 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Datatypes_BasicSurfaces_h
#define SCI_Datatypes_BasicSurfaces_h 1

#include <Core/share/share.h>

#include <Core/Datatypes/Surface.h>
//#include <Core/Datatypes/MeshTet.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Containers/Array1.h>

namespace SCIRun {

class SCICORESHARE CylinderSurface : public Surface {
  Point p1;
  Point p2;
  double radius;
  int nu;
  int nv;
  int ndiscu;

  Vector u;
  Vector v;

  Vector axis;
  double rad2;
  double height;

public:
  CylinderSurface(const Point& p1, const Point& p2, double radius,
		  int nu, int nv, int ndiscu);
  CylinderSurface(const CylinderSurface&);
  virtual ~CylinderSurface();
  virtual Surface* clone();

  virtual bool inside(const Point& p);
  virtual void construct_grid();
  virtual void construct_grid(int, int, int, const Point &, double);

  virtual GeomObj* get_geom(const ColorMapHandle&);

  // Persistent representation.
  virtual void io(Piostream&);
  static PersistentTypeID type_id;
};


class SCICORESHARE PointSurface : public Surface {
  Point pos;

public:
  PointSurface(const Point& pos);
  PointSurface(const PointSurface&);
  virtual ~PointSurface();
  virtual Surface* clone();

  virtual bool inside(const Point& p);
  virtual void construct_grid();
  virtual void construct_grid(int, int, int, const Point &, double);
  virtual GeomObj* get_geom(const ColorMapHandle&);

  // Persistent representation.
  virtual void io(Piostream&);
  static PersistentTypeID type_id;
};


class SCICORESHARE SphereSurface : public Surface {
  Point cen;
  Vector pole;
  double radius;
  int nu;
  int nv;

  Vector u;
  Vector v;

  double rad2;

public:
  SphereSurface(const Point& cen, double radius, const Vector& pole,
		int nu, int nv);
  SphereSurface(const SphereSurface&);
  virtual ~SphereSurface();
  virtual Surface* clone();

  virtual bool inside(const Point& p);
  virtual void construct_grid();
  virtual void construct_grid(int, int, int, const Point &, double);
  virtual GeomObj* get_geom(const ColorMapHandle&);

  // Persistent representation.
  virtual void io(Piostream&);
  static PersistentTypeID type_id;
};


class SCICORESHARE PointsSurface : public Surface {
public:
  Array1<Point> pos;
  Array1<double> val;

public:
  PointsSurface();
  PointsSurface(const Array1<Point>& pos, const Array1<double>& val);
  PointsSurface(const PointsSurface&);
  virtual ~PointsSurface();
  virtual Surface* clone();

  virtual bool inside(const Point& p);
  virtual void construct_grid();
  virtual void construct_grid(int, int, int, const Point &, double);
  virtual GeomObj* get_geom(const ColorMapHandle&);

  // Persistent representation.
  virtual void io(Piostream&);
  static PersistentTypeID type_id;
};

} // End namespace SCIRun

#endif /* SCI_Datatypes_BasicSurfaces_h */

