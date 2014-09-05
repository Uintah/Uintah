/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

/*
 * GeomSphere.cc: Sphere objects
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Core/Geom/GeomSphere.h>
#include <Core/Util/NotFinished.h>
#include <Core/Geom/GeomSave.h>
#include <Core/Geom/GeomTri.h>
#include <Core/Geometry/BBox.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Persistent/PersistentSTL.h>

#include <iostream>
using std::cerr;
using std::ostream;

namespace SCIRun {

Persistent* make_GeomSphere()
{
    return scinew GeomSphere;
}

PersistentTypeID GeomSphere::type_id("GeomSphere", "GeomObj", make_GeomSphere);

GeomSphere::GeomSphere(int nu, int nv)
: GeomObj(), cen(0,0,0), rad(1), nu(nu), nv(nv)
{
    adjust();
}


GeomSphere::GeomSphere(const Point& cen, double rad, int nu, int nv)
: GeomObj(), cen(cen), rad(rad), nu(nu), nv(nv)
{
    adjust();
}


void GeomSphere::move(const Point& _cen, double _rad, int _nu, int _nv)
{
    cen=_cen;
    rad=_rad;
    nu=_nu;
    nv=_nv;
    adjust();
}

void GeomSphere::move(const Point& _cen) {
  cen = _cen;
  adjust();
}

GeomSphere::GeomSphere(const GeomSphere& copy)
: GeomObj(copy), cen(copy.cen), rad(copy.rad), nu(copy.nu), nv(copy.nv)
{
    adjust();
}

GeomSphere::~GeomSphere()
{
}

void GeomSphere::adjust()
{
}

GeomObj* GeomSphere::clone()
{
    return scinew GeomSphere(*this);
}


void GeomSphere::get_bounds(BBox& bb)
{
    bb.extend(cen, rad);
}

#define GEOMSPHERE_VERSION 1

void GeomSphere::io(Piostream& stream)
{

    stream.begin_class("GeomSphere", GEOMSPHERE_VERSION);
    GeomObj::io(stream);
    Pio(stream, cen);
    Pio(stream, rad);
    Pio(stream, nu);
    Pio(stream, nv);
    stream.end_class();
}

void GeomSphere::getnunv(int num_polygons, int &nu, int &nv) {
#define MIN_POLYS 8
#define MAX_POLYS 400
#define MIN_NU 4
#define MAX_NU 20
#define MIN_NV 2
#define MAX_NV 20
  // calculate the spheres nu,nv based on the number of polygons
  float t = (num_polygons - MIN_POLYS)/float(MAX_POLYS - MIN_POLYS);
  nu = int(MIN_NU + t*(MAX_NU - MIN_NU)); 
  nv = int(MIN_NV + t*(MAX_NV - MIN_NV));
}


Persistent* GeomSuperquadric::maker()
{
    return scinew GeomSuperquadric();
}

PersistentTypeID GeomSuperquadric::type_id("GeomSuperquadric",
					   "GeomObj", GeomSuperquadric::maker);


static inline float
spow(double e, double x)
{
  if (e < 0.0)
  {
    return (float)(pow(fabs(e), x) * -1.0);
  }
  else
  {
    return (float)(pow(e, x));
  }
}


GeomSuperquadric::GeomSuperquadric()
{
}


GeomSuperquadric::GeomSuperquadric(int axis, double A, double B,
				   int nu, int nv)
  : axis_(axis),
    A_(A),
    B_(B),
    nu_(nu),
    nv_(nv)
{
  nu_ = Max(2, nu_);
  nv_ = Max(3, nv_);
  // TODO:  ASSERT nu_ * nv_ not a short overflow.

  compute_geometry();
}


void
GeomSuperquadric::compute_geometry()
{
  int ti, pi;

  for (pi = 1; pi < nv_; pi++)
  {
    const double p = (pi / (double)(nv_ + 1)) * M_PI;
    for (ti = 0; ti < nu_; ti++)
    {
      const double t = (ti / (double)nv_) * (2.0 * M_PI);
      const double x = spow(cos(p), B_);
      const double y = spow(cos(t), A_) * spow(sin(p), B_);
      const double z = spow(sin(t), A_) * spow(sin(p), B_);

      const double xxb = pow(x*x, 1/B_);
      const double yya = pow(y*y, 1/A_);
      const double zza = pow(z*z, 1/A_);
      const double R = pow(yya + zza, (A_/B_)-1);
      const float nx = 2*xxb/(B_*x + 1.0e-6);
      const float ny = 2*R*yya/(B_*y + 1.0e-6);
      const float nz = 2*R*zza/(B_*z + 1.0e-6);
      const float nn = 1.0 / sqrt(nx * nx + ny * ny + nz * nz);
      
      switch(axis_) {
      case 0:
	points_.push_back(x);
	points_.push_back(y);
	points_.push_back(z);
	normals_.push_back(nx * nn);
	normals_.push_back(ny * nn);
	normals_.push_back(nz * nn);
	break;
      case 1:
	points_.push_back(z);
	points_.push_back(x);
	points_.push_back(y);
	normals_.push_back(nz * nn);
	normals_.push_back(nx * nn);
	normals_.push_back(ny * nn);
	break;
      case 2:
	points_.push_back(y);
	points_.push_back(z);
	points_.push_back(x);
	normals_.push_back(ny * nn);
	normals_.push_back(nz * nn);
	normals_.push_back(nx * nn);
	break;
      }
    }
  }

  // Add north pole.
  switch(axis_) {
  case 0:
    points_.push_back(1.0);
    points_.push_back(0.0);
    points_.push_back(0.0);
    normals_.push_back(1.0);
    normals_.push_back(0.0);
    normals_.push_back(0.0);
    break;
  case 1:
    points_.push_back(0.0);
    points_.push_back(1.0);
    points_.push_back(0.0);
    normals_.push_back(0.0);
    normals_.push_back(1.0);
    normals_.push_back(0.0);
    break;
  case 2: default:
    points_.push_back(0.0);
    points_.push_back(0.0);
    points_.push_back(1.0);
    normals_.push_back(0.0);
    normals_.push_back(0.0);
    normals_.push_back(1.0);
    break;
  }

  // Add south pole.
  switch(axis_) {
  case 0:
    points_.push_back(-1.0);
    points_.push_back(0.0);
    points_.push_back(0.0);
    normals_.push_back(-1.0);
    normals_.push_back(0.0);
    normals_.push_back(0.0);
    break;
  case 1:
    points_.push_back(0.0);
    points_.push_back(-1.0);
    points_.push_back(0.0);
    normals_.push_back(0.0);
    normals_.push_back(-1.0);
    normals_.push_back(0.0);
    break;
  case 2: default:
    points_.push_back(0.0);
    points_.push_back(0.0);
    points_.push_back(-1.0);
    normals_.push_back(0.0);
    normals_.push_back(0.0);
    normals_.push_back(-1.0);
    break;
  }

  // North pole cap
  tindices_.push_back(points_.size()/3 - 2);
  for (ti = 0; ti < nu_; ti++)
  {
    tindices_.push_back(ti);
  }
  tindices_.push_back(0);
    
  // South pole cap.
  tindices_.push_back(points_.size()/3 - 1);
  for (ti = nu_; ti >= 0; ti--)
  {
    tindices_.push_back((nv_-2) * nu_ + (ti%nu_));
  }

  // Equatorial region
  for (pi = 0; pi < nv_-2; pi++)
  {
    for (ti=0; ti < nu_; ti++)
    {
      qindices_.push_back(pi * nu_ + ti);
      qindices_.push_back((pi+1) * nu_ + ti);
    }
    qindices_.push_back(pi * nu_);
    qindices_.push_back((pi+1) * nu_);
  }
}


GeomSuperquadric::GeomSuperquadric(const GeomSuperquadric& copy)
  : GeomObj(copy)
{
}

GeomSuperquadric::~GeomSuperquadric()
{
}


GeomObj* GeomSuperquadric::clone()
{
    return scinew GeomSuperquadric(*this);
}


void GeomSuperquadric::get_bounds(BBox& bb)
{
    bb.extend(Point(1.0, 1.0, 1.0));
    bb.extend(Point(-1.0, -1.0, -1.0));
}


#define GEOMSUPERQUADRIC_VERSION 1

void GeomSuperquadric::io(Piostream& stream)
{
    stream.begin_class("GeomSuperquadric", GEOMSUPERQUADRIC_VERSION);
    GeomObj::io(stream);
    Pio(stream, axis_);
    Pio(stream, A_);
    Pio(stream, B_);
    Pio(stream, nu_);
    Pio(stream, nv_);
    if (stream.reading()) { compute_geometry(); }
    stream.end_class();
}


Persistent* make_GeomSpheres()
{
  return scinew GeomSpheres;
}


PersistentTypeID GeomSpheres::type_id("GeomSpheres", "GeomObj", make_GeomSpheres);


GeomSpheres::GeomSpheres(double radius, int nu, int nv)
  : GeomObj(),
    nu_(nu),
    nv_(nv),
    global_radius_(radius)
{
}


GeomSpheres::GeomSpheres(const GeomSpheres& copy)
  : GeomObj(copy),
    centers_(copy.centers_),
    radii_(copy.radii_),
    colors_(copy.colors_),
    indices_(copy.indices_),
    nu_(copy.nu_),
    nv_(copy.nv_),
    global_radius_(copy.global_radius_)
{
}


GeomSpheres::~GeomSpheres()
{
}


GeomObj *
GeomSpheres::clone()
{
  return scinew GeomSpheres(*this);
}


void
GeomSpheres::get_bounds(BBox& bb)
{
  const bool ugr = !(radii_.size() == centers_.size());
  for (unsigned int i=0; i < centers_.size(); i++)
  {
    bb.extend(centers_[i], ugr?global_radius_:radii_[i]);
  }
}


static unsigned char
COLOR_FTOB(double v)
{
  const int inter = (int)(v * 255 + 0.5);
  if (inter > 255) return 255;
  if (inter < 0) return 0;
  return (unsigned char)inter;
}


void
GeomSpheres::add(const Point &center)
{
  centers_.push_back(center);
}


void
GeomSpheres::add(const Point &center, const MaterialHandle &mat)
{
  add(center);
  const unsigned char r0 = COLOR_FTOB(mat->diffuse.r());
  const unsigned char g0 = COLOR_FTOB(mat->diffuse.g());
  const unsigned char b0 = COLOR_FTOB(mat->diffuse.b());
  const unsigned char a0 = COLOR_FTOB(mat->transparency);
  colors_.push_back(r0);
  colors_.push_back(g0);
  colors_.push_back(b0);
  colors_.push_back(a0);
}


void
GeomSpheres::add(const Point &center, float index)
{
  add(center);
  indices_.push_back(index);
}


bool
GeomSpheres::add_radius(const Point &c, double r)
{
  if (r < 1.0e-6) { return false; }
  centers_.push_back(c);
  radii_.push_back(r);
  return true;
}

bool
GeomSpheres::add_radius(const Point &c, double r, const MaterialHandle &mat)
{
  if (r < 1.0e-6) { return false; }
  add_radius(c, r);
  const unsigned char r0 = COLOR_FTOB(mat->diffuse.r());
  const unsigned char g0 = COLOR_FTOB(mat->diffuse.g());
  const unsigned char b0 = COLOR_FTOB(mat->diffuse.b());
  const unsigned char a0 = COLOR_FTOB(mat->transparency);
  colors_.push_back(r0);
  colors_.push_back(g0);
  colors_.push_back(b0);
  colors_.push_back(a0);
  return true;
}

bool
GeomSpheres::add_radius(const Point &c, double r, float index)
{
  if (r < 1.0e-6) { return false; }
  add_radius(c, r);
  indices_.push_back(index);
  return true;
}



#define GEOMSPHERES_VERSION 1

void
GeomSpheres::io(Piostream& stream)
{
  stream.begin_class("GeomSpheres", GEOMSPHERES_VERSION);
  GeomObj::io(stream);
  Pio(stream, centers_);
  Pio(stream, radii_);
  Pio(stream, colors_);
  Pio(stream, indices_);
  Pio(stream, nu_);
  Pio(stream, nv_);
  Pio(stream, global_radius_);
  stream.end_class();
}

} // End namespace SCIRun

