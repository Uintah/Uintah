//  Lattice3Geom.cc - A base class for regular geometries with alligned axes
//
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//
//  Copyright (C) 2000 SCI Institute

#include <SCICore/Datatypes/Lattice3Geom.h>
#include <SCICore/Math/MinMax.h>


namespace SCICore{
namespace Datatypes{

using namespace SCICore::Math;

PersistentTypeID Lattice3Geom::type_id("Lattice3Geom", "Datatype", 0);

Lattice3Geom::Lattice3Geom() :
  d_nx(0), d_ny(0), d_nz(0)
{
}

Lattice3Geom::Lattice3Geom(int ix, int iy, int iz)
{
  d_nx = Max(1, ix);
  d_ny = Max(1, iy);
  d_nz = Max(1, iz);
  compute_bbox();
}

Lattice3Geom::Lattice3Geom(int ix, int iy, int iz,
			   const Point &a, const Point &b)
{
  d_nx = Max(1, ix);
  d_ny = Max(1, iy);
  d_nz = Max(1, iz);

  BBox box;
  box.extend(a);
  box.extend(b);
  set_bbox(box);
}

Lattice3Geom::~Lattice3Geom()
{
}

string
Lattice3Geom::get_info()
{
  ostringstream retval;
  retval <<
    "name = " << name << endl <<
    "x = " << d_nx << endl <<
    "y = " << d_ny << endl <<
    "z = " << d_nz << endl;
  
  return retval.str();
}


bool
Lattice3Geom::compute_bbox()
{
  bbox.reset();
  bbox.extend(d_trans.project(Point(0, 0, 0)));
  bbox.extend(d_trans.project(Point(d_nx-1, d_ny-1, d_nz-1)));
  return true;
}


void
Lattice3Geom::transform(const Point &p, Point &r)
{
  ftransform(p, r);
}

void
Lattice3Geom::itransform(const Point &p, Point &r)
{
  fitransform(p, r);
}



Point
Lattice3Geom::get_point(int x, int y, int z)
{
  Point p(x, y, z);
  Point r;
  transform(p, r);
  return r;
}


#if 0
bool Lattice3Geom::locate(const Point& p, int& ix, int& iy, int& iz){
  Vector pn=p-bbox.min();
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
Lattice3Geom::resize(int x, int y, int z)
{
  d_nx = Min(x, 1);
  d_ny = Min(y, 1);
  d_nz = Min(z, 1);
  compute_bbox();
}

#if 0  
void
Lattice3Geom::set_bbox(const Point& imin, const Point& imax)
{
  bbox.reset();
  // extend the bbox to include min and max
  bbox.extend(imin);
  bbox.extend(imax);
  has_bbox = true;
  diagonal = imax - imin;
  compute_deltas();
}

void Lattice3Geom::set_bbox(const BBox& ibbox){
  bbox = ibbox;
  has_bbox = true;
  compute_deltas();
}

void Lattice3Geom::compute_deltas(){
  dx = (bbox.max().x() - bbox.min().x())/(double) d_nx;
  dy = (bbox.max().y() - bbox.min().y())/(double) d_ny;
  dz = (bbox.max().z() - bbox.min().z())/(double) d_nz;
}
#endif
    

void
Lattice3Geom::set_bbox(BBox &box)
{
  Vector offset(box.min());
  d_trans.post_translate(offset);

  Vector diag = box.diagonal();
  Vector sdiag(diag.x() / (d_nx - 1.0),
	       diag.y() / (d_ny - 1.0),
	       diag.z() / (d_nz - 1.0));

  d_trans.post_scale(sdiag);
  d_trans.compute_imat();
  compute_bbox();
}

void
Lattice3Geom::io(Piostream&)
{
}


} // end Datatypes
} // end SCICore
