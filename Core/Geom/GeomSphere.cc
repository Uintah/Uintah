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


void
GeomSpheres::add_radius(const Point &c, double r)
{
  centers_.push_back(c);
  radii_.push_back(r);
}

void
GeomSpheres::add_radius(const Point &c, double r, const MaterialHandle &mat)
{
  add_radius(c, r);
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
GeomSpheres::add_radius(const Point &c, double r, float index)
{
  add_radius(c, r);
  indices_.push_back(index);
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

