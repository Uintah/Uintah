//  LatticeGeom.cc - A base class for regular geometries with alligned axes
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//  Copyright (C) 2000 SCI Institute

#include <Core/Datatypes/LatticeGeom.h>
#include <Core/Math/MinMax.h>


namespace SCIRun {

//////////
// PIO support
string LatticeGeom::typeName(int){
  static string className = "LatticeGeom";
  return className;
}

static Persistent* maker(){
  return new LatticeGeom();
}

PersistentTypeID LatticeGeom::type_id(LatticeGeom::typeName(0), 
				      Geom::typeName(0), 
				      maker);

#define LATTICEGEOM_VERSION 1
void
LatticeGeom::io(Piostream& stream)
{
  stream.begin_class(typeName(0).c_str(), LATTICEGEOM_VERSION);
  Pio(stream, d_dim);
  Pio(stream, d_nx);
  Pio(stream, d_ny);
  Pio(stream, d_nz);
  Pio(stream, d_transform);
  Pio(stream, d_prescale);
  stream.end_class();
}


LatticeGeom::LatticeGeom() :
  d_dim(0), d_nx(1), d_ny(1), d_nz(1)
{
}

LatticeGeom::LatticeGeom(int ix)
{
  d_prescale.load_identity();
  resize(ix);
}

LatticeGeom::LatticeGeom(int ix, int iy)
{
  d_prescale.load_identity();
  resize(ix, iy);
}

LatticeGeom::LatticeGeom(int ix, int iy, int iz)
{
  d_prescale.load_identity();
  resize(ix, iy, iz);
}



LatticeGeom::LatticeGeom(int ix,
			 const Point &a, const Point &b)
{
  d_dim = 1;
  d_nx = Max(ix, 1);
  d_ny = 1;
  d_nz = 1;

  BBox box;
  box.extend(a);
  box.extend(b);
  setBoundingBox(box);
}

LatticeGeom::LatticeGeom(int ix, int iy,
			 const Point &a, const Point &b)
{
  d_dim = 2;
  d_nx = Max(ix, 1);
  d_ny = Max(iy, 1);
  d_nz = 1;

  BBox box;
  box.extend(a);
  box.extend(b);
  setBoundingBox(box);
}

LatticeGeom::LatticeGeom(int ix, int iy, int iz,
			 const Point &a, const Point &b)
{
  d_dim = 3;
  d_nx = Max(ix, 1);
  d_ny = Max(iy, 1);
  d_nz = Max(iz, 1);

  BBox box;
  box.extend(a);
  box.extend(b);
  setBoundingBox(box);
}

LatticeGeom::~LatticeGeom()
{
}

string
LatticeGeom::getInfo()
{
  ostringstream retval;
  retval <<
    "name = " << d_name << endl <<
    "x = " << d_nx << endl <<
    "y = " << d_ny << endl <<
    "z = " << d_nz << endl;
  return retval.str();
}

string
LatticeGeom::getTypeName(int n){
  return typeName(n);
}

bool
LatticeGeom::computeBoundingBox()
{
  const int x = d_nx-1;
  const int y = d_ny-1;
  const int z = d_nz-1;

  d_bbox.reset();
  d_bbox.extend(d_transform.project(Point(0, 0, 0)));
  d_bbox.extend(d_transform.project(Point(0, 0, z)));
  d_bbox.extend(d_transform.project(Point(0, y, 0)));
  d_bbox.extend(d_transform.project(Point(0, y, z)));
  d_bbox.extend(d_transform.project(Point(x, 0, 0)));
  d_bbox.extend(d_transform.project(Point(x, 0, z)));
  d_bbox.extend(d_transform.project(Point(x, y, 0)));
  d_bbox.extend(d_transform.project(Point(x, y, z)));
  return true;
}


void
LatticeGeom::transform(const Point &p, Point &r)
{
  ftransform(p, r);
}

void
LatticeGeom::itransform(const Point &p, Point &r)
{
  fitransform(p, r);
}



Point
LatticeGeom::getPoint(int x, int y, int z)
{
  Point p(x, y, z);
  Point r;
  transform(p, r);
  return r;
}


bool
LatticeGeom::locate(const Point &p, int &i, int &j, int &k)
{
  Point r;
  itransform(p, r);
  i = r.x() + 0.5;
  j = r.y() + 0.5;
  k = r.z() + 0.5;
  if (i < 0 || i >= d_nx ||
      j < 0 || j >= d_ny ||
      k < 0 || k >= d_nz)
  {
    return false;
  }
  return true;
}


void
LatticeGeom::resize(int x)
{
  d_dim = 1;
  d_nx = Max(x, 1);
  d_ny = 1;
  d_nz = 1;
  updateTransform();
}

void
LatticeGeom::resize(int x, int y)
{
  d_dim = 2;
  d_nx = Max(x, 1);
  d_ny = Max(y, 1);
  d_nz = 1;
  updateTransform();
}

void
LatticeGeom::resize(int x, int y, int z)
{
  d_dim = 3;
  d_nx = Max(x, 1);
  d_ny = Max(y, 1);
  d_nz = Max(z, 1);
  updateTransform();
}


void
LatticeGeom::setBoundingBox(BBox &box)
{
  d_prescale.load_identity();

  Vector offset(box.min());
  d_prescale.post_translate(offset);

  Vector diag = box.diagonal();
  Vector sdiag(diag.x(), diag.y(), diag.z());

  d_prescale.post_scale(sdiag);
  d_prescale.compute_imat();

  updateTransform();
}

void
LatticeGeom::updateTransform()
{
  d_transform = d_prescale;

  const double x = 1.0 / Max(d_nx - 1.0, 1.0);
  const double y = 1.0 / Max(d_ny - 1.0, 1.0);
  const double z = 1.0 / Max(d_nz - 1.0, 1.0);
  Vector scale(x, y, z);
  d_transform.post_scale(scale);

  d_transform.compute_imat();

  computeBoundingBox();
}



} // End namespace SCIRun
