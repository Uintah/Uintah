//  IrregLatticeGeom.h - Cubic lattice with jittered geometry.
//
//  Written by:
//   Michael Callahan
//   Department of Computer Science
//   University of Utah
//   November 2000
//
//  Copyright (C) 2000 SCI Institute


#ifndef SCI_project_IrregLatticeGeom_h
#define SCI_project_IrregLatticeGeom_h 1

#include <SCICore/Datatypes/LatticeGeom.h>
#include <SCICore/Datatypes/AccelAttrib.h>
#include <SCICore/Geometry/Point.h>

namespace SCICore {
namespace Datatypes {


class IrregLatticeGeom : public LatticeGeom
{
public:

  IrregLatticeGeom(int x, int y, int z);
  IrregLatticeGeom(int x, int y, int z, const Point &a, const Point &b);

  void init_test_attrib();

  virtual void transform(const Point &p, Point &r);

protected:
  //virtual bool computeBoundingBox();

private:
  DiscreteAttrib<Vector> *attrib;
};


IrregLatticeGeom::IrregLatticeGeom(int x, int y, int z)
  : LatticeGeom(x, y, z)
{
  init_test_attrib();
}


IrregLatticeGeom::IrregLatticeGeom(int x, int y, int z,
			     const Point &a, const Point &b)
  : LatticeGeom(x, y, z, a, b)
{
  init_test_attrib();
}


class RandFunctor : public AttribFunctor<Vector>
{
public:
  const double scale;

  RandFunctor(double s) : scale(s) {}
  virtual void operator() (Vector &v)
  {
    v.x(drand48() * scale - scale*0.5);
    v.y(drand48() * scale - scale*0.5);
    v.z(drand48() * scale - scale*0.5);
  }
};

void
IrregLatticeGeom::init_test_attrib()
{
  attrib = new AccelAttrib<Vector>(d_nx, d_ny, d_nz);
  RandFunctor f(0.5);
  attrib->iterate(f);
}


void
IrregLatticeGeom::transform(const Point &p, Point &r)
{
  Point q = p + attrib->get3(p.x(), p.y(), p.z());
  LatticeGeom::transform(q, r);
}


}
}

#endif
