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


namespace SCICore{
namespace Datatypes{

using namespace SCICore::Math;

PersistentTypeID LatticeGeom::type_id("LatticeGeom", "Datatype", 0);

LatticeGeom::LatticeGeom() :
  d_dim(0), d_nx(0), d_ny(0), d_nz(0)
{
}


LatticeGeom::LatticeGeom(int ix)
{
  d_dim = 1;
  d_nx = Max(1, ix);
  d_ny = 0;
  d_nz = 0;
  computeBoundingBox();
}

LatticeGeom::LatticeGeom(int ix, int iy)
{
  d_dim = 2;
  d_nx = Max(1, ix);
  d_ny = Max(1, iy);
  d_nz = 0;
  computeBoundingBox();
}

LatticeGeom::LatticeGeom(int ix, int iy, int iz)
{
  d_dim = 3;
  d_nx = Max(1, ix);
  d_ny = Max(1, iy);
  d_nz = Max(1, iz);
  computeBoundingBox();
}

LatticeGeom::LatticeGeom(int ix,
			 const Point &a, const Point &b)
{
  d_dim = 1;
  d_nx = Max(1, ix);
  d_ny = 0;
  d_nz = 0;

  BBox box;
  box.extend(a);
  box.extend(b);
  setBoundingBox(box);
}

LatticeGeom::LatticeGeom(int ix, int iy,
			 const Point &a, const Point &b)
{
  d_dim = 2;
  d_nx = Max(1, ix);
  d_ny = Max(1, iy);
  d_nz = 0;

  BBox box;
  box.extend(a);
  box.extend(b);
  setBoundingBox(box);
}

LatticeGeom::LatticeGeom(int ix, int iy, int iz,
			 const Point &a, const Point &b)
{
  d_dim = 3;
  d_nx = Max(1, ix);
  d_ny = Max(1, iy);
  d_nz = Max(1, iz);

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
  d_bbox.reset();
  d_bbox.extend(d_trans.project(Point(0, 0, 0)));
  d_bbox.extend(d_trans.project(Point(d_nx-1, d_ny-1, d_nz-1)));
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
  d_nx = Min(x, 1);
  d_ny = 0;
  d_nz = 0;
  computeBoundingBox();
}

void
LatticeGeom::resize(int x, int y)
{
  d_dim = 2;
  d_nx = Min(x, 1);
  d_ny = Min(y, 1);
  d_nz = 0;
  computeBoundingBox();
}

void
LatticeGeom::resize(int x, int y, int z)
{
  d_dim = 3;
  d_nx = Min(x, 1);
  d_ny = Min(y, 1);
  d_nz = Min(z, 1);
  computeBoundingBox();
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
  Vector offset(box.min());
  d_trans.post_translate(offset);

  Vector diag = box.diagonal();
  Vector sdiag(diag.x() / (d_nx - 1.0),
	       diag.y() / (d_ny - 1.0),
	       diag.z() / (d_nz - 1.0));

  d_trans.post_scale(sdiag);
  d_trans.compute_imat();
  computeBoundingBox();
}

void
LatticeGeom::io(Piostream&)
{
}


} // end Datatypes
} // end SCICore
