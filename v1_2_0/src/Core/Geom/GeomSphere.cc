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
#include <iostream>
using std::cerr;
using std::ostream;

namespace SCIRun {

Persistent* make_GeomSphere()
{
    return scinew GeomSphere;
}

PersistentTypeID GeomSphere::type_id("GeomSphere", "GeomObj", make_GeomSphere);

GeomSphere::GeomSphere(int nu, int nv, int id)
: GeomObj(id), cen(0,0,0), rad(1), nu(nu), nv(nv)
{
    adjust();
}
GeomSphere::GeomSphere(int nu, int nv, IntVector id)
: GeomObj(id), cen(0,0,0), rad(1), nu(nu), nv(nv)
{
    adjust();
}

GeomSphere::GeomSphere(int nu, int nv, int id_int, IntVector id)
: GeomObj(id_int,id), cen(0,0,0), rad(1), nu(nu), nv(nv)
{
    adjust();
}

GeomSphere::GeomSphere(const Point& cen, double rad, int nu, int nv, int id)
: GeomObj( id ), cen(cen), rad(rad), nu(nu), nv(nv)
{
    adjust();
}

GeomSphere::GeomSphere(const Point& cen, double rad, int nu, int nv, int id_int, IntVector id)
: GeomObj( id_int, id ), cen(cen), rad(rad), nu(nu), nv(nv)
{
    adjust();
}

GeomSphere::GeomSphere(const Point& cen, double rad, int nu, int nv, IntVector id)
: GeomObj( id ), cen(cen), rad(rad), nu(nu), nv(nv)
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

bool
GeomSphere::getId( int& id )
{
  if ( id == 0x1234567)
    return false;
  else {
    id = this->id;
    return true;
  }
}

bool
GeomSphere::getId( IntVector& id )
{
  if ( _id == IntVector(0x1234567,0x1234567,0x1234567) )
    return false;
  else {
    id = this->_id;
    return true;
  }
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

bool GeomSphere::saveobj(ostream& out, const string& format,
			 GeomSave* saveinfo)
{
    cerr << "saveobj Sphere\n";
    if(format == "vrml" || format == "iv"){
	saveinfo->start_tsep(out);
	saveinfo->start_node(out, "Sphere");
	saveinfo->indent(out);
	out << "radius " << rad << "\n";
	saveinfo->end_node(out);
	saveinfo->end_tsep(out);
	return true;
    } else if(format == "rib"){
	saveinfo->start_trn(out);
	saveinfo->indent(out);
	out << "Translate " << cen.x() << " "  << cen.y() << " "  << cen.z() << "\n";
	saveinfo->indent(out);
	out << "Sphere " << rad << " " << -rad << " " << rad << " 360\n";
	saveinfo->end_trn(out);
	return true;
    } else {
	NOT_FINISHED("GeomSphere::saveobj");
	return false;
    }
}

} // End namespace SCIRun

