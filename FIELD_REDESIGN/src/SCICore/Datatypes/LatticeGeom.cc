//  LatticeGeom.cc - A base class for regular geometries with alligned axes
//
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//
//  Copyright (C) 2000 SCI Institute

#include <SCICore/Datatypes/LatticeGeom.h>
#include <SCICore/Math/MinMax.h>


namespace SCICore {
namespace Datatypes {

using namespace SCICore::Math;

PersistentTypeID LatticeGeom::type_id("LatticeGeom", "Datatype", 0);

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


void
LatticeGeom::locate(const Point &p, int &i, int &j, int &k)
{
  Point r;
  itransform(p, r);
  i = r.x() + 0.5;
  j = r.y() + 0.5;
  k = r.z() + 0.5;
}


#if 0
bool LatticeGeom::locate(const Point& p, int& ix, int& iy, int& iz){
  Vector pn=p-d_bbox.min();
  double mdx=diagonal.x();
  double mdy=diagonal.y();
  double mdz=diagonal.z();
  double x=pn.x()*(d_nx-1)/mdx;
  double y=pn.y()*(d_ny-1)/mdy;
  double z=pn.z()*(d_nz-1)/mdz;
  ix=(int)x;
  iy=(int)y;
  iz=(int)z;
  return true;
}
#endif


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

#if 0  
void
LatticeGeom::setBoundingBox(const Point& imin, const Point& imax)
{
  bbox.reset();
  // extend the bbox to include min and max
  bbox.extend(imin);
  bbox.extend(imax);
  has_bbox = true;
  diagonal = imax - imin;
  compute_deltas();
}

void LatticeGeom::setBoundingBox(const BBox& ibbox){
  bbox = ibbox;
  has_bbox = true;
  compute_deltas();
}

void LatticeGeom::compute_deltas(){
  dx = (bbox.max().x() - bbox.min().x())/(double) d_nx;
  dy = (bbox.max().y() - bbox.min().y())/(double) d_ny;
  dz = (bbox.max().z() - bbox.min().z())/(double) d_nz;
}
#endif
    

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
LatticeGeom::io(Piostream&)
{
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



} // end Datatypes
} // end SCICore
