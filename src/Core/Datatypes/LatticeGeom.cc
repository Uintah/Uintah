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
  return new LatticeGeom(0, 0, 0);
}

PersistentTypeID LatticeGeom::type_id(LatticeGeom::typeName(0), 
				      Geom::typeName(0), 
				      maker);

#define LATTICEGEOM_VERSION 1
void
LatticeGeom::io(Piostream& stream)
{
  stream.begin_class(typeName(0).c_str(), LATTICEGEOM_VERSION);
  Pio(stream, dim_);
  Pio(stream, nx_);
  Pio(stream, ny_);
  Pio(stream, nz_);
  Pio(stream, transform_);
  Pio(stream, prescale_);
  stream.end_class();
}


LatticeGeom::LatticeGeom(int ix)
{
  prescale_.load_identity();
  resize(ix);
}

LatticeGeom::LatticeGeom(int ix, int iy)
{
  prescale_.load_identity();
  resize(ix, iy);
}

LatticeGeom::LatticeGeom(int ix, int iy, int iz)
{
  prescale_.load_identity();
  resize(ix, iy, iz);
}



LatticeGeom::LatticeGeom(int ix,
			 const Point &a, const Point &b)
{
  dim_ = 1;
  nx_ = Max(ix, 1);
  ny_ = 1;
  nz_ = 1;

  BBox box;
  box.extend(a);
  box.extend(b);
  setBoundingBox(box);
}

LatticeGeom::LatticeGeom(int ix, int iy,
			 const Point &a, const Point &b)
{
  dim_ = 2;
  nx_ = Max(ix, 1);
  ny_ = Max(iy, 1);
  nz_ = 1;

  BBox box;
  box.extend(a);
  box.extend(b);
  setBoundingBox(box);
}

LatticeGeom::LatticeGeom(int ix, int iy, int iz,
			 const Point &a, const Point &b)
{
  dim_ = 3;
  nx_ = Max(ix, 1);
  ny_ = Max(iy, 1);
  nz_ = Max(iz, 1);

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
    "name = " << name_ << endl <<
    "x = " << nx_ << endl <<
    "y = " << ny_ << endl <<
    "z = " << nz_ << endl;
  return retval.str();
}

string
LatticeGeom::getTypeName(int n){
  return typeName(n);
}

bool
LatticeGeom::computeBoundingBox()
{
  const int x = nx_-1;
  const int y = ny_-1;
  const int z = nz_-1;

  bbox_.reset();
  bbox_.extend(transform_.project(Point(0, 0, 0)));
  bbox_.extend(transform_.project(Point(0, 0, z)));
  bbox_.extend(transform_.project(Point(0, y, 0)));
  bbox_.extend(transform_.project(Point(0, y, z)));
  bbox_.extend(transform_.project(Point(x, 0, 0)));
  bbox_.extend(transform_.project(Point(x, 0, z)));
  bbox_.extend(transform_.project(Point(x, y, 0)));
  bbox_.extend(transform_.project(Point(x, y, z)));
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
LatticeGeom::locate(int *loc, const Point &p)
{
  Point r;
  itransform(p, r);
  loc[0] = int(r.x() + 0.5);
  loc[1] = int(r.y() + 0.5);
  loc[2] = int(r.z() + 0.5);

  if (loc[0] < 0 || loc[0] >= nx_ ||
      loc[1] < 0 || loc[1] >= ny_ ||
      loc[2] < 0 || loc[2] >= nz_)
  {
    return false;
  }
  return true;
}


void
LatticeGeom::resize(int x)
{
  dim_ = 1;
  nx_ = Max(x, 1);
  ny_ = 1;
  nz_ = 1;
  updateTransform();
}

void
LatticeGeom::resize(int x, int y)
{
  dim_ = 2;
  nx_ = Max(x, 1);
  ny_ = Max(y, 1);
  nz_ = 1;
  updateTransform();
}

void
LatticeGeom::resize(int x, int y, int z)
{
  dim_ = 3;
  nx_ = Max(x, 1);
  ny_ = Max(y, 1);
  nz_ = Max(z, 1);
  updateTransform();
}


void
LatticeGeom::setBoundingBox(BBox &box)
{
  prescale_.load_identity();

  Vector offset(box.min());
  prescale_.post_translate(offset);

  Vector diag = box.diagonal();
  Vector sdiag(diag.x(), diag.y(), diag.z());

  prescale_.post_scale(sdiag);
  prescale_.compute_imat();

  updateTransform();
}

void
LatticeGeom::updateTransform()
{
  transform_ = prescale_;

  const double x = 1.0 / Max(nx_ - 1.0, 1.0);
  const double y = 1.0 / Max(ny_ - 1.0, 1.0);
  const double z = 1.0 / Max(nz_ - 1.0, 1.0);
  Vector scale(x, y, z);
  transform_.post_scale(scale);

  transform_.compute_imat();

  computeBoundingBox();
}



} // End namespace SCIRun
