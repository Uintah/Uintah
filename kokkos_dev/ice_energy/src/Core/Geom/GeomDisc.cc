/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/



/*
 *  GeomDisc.h:  Disc object
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Core/Geom/GeomDisc.h>
#include <Core/Util/NotFinished.h>
#include <Core/Geom/GeomTri.h>
#include <Core/Geometry/BBox.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MiscMath.h>
#include <Core/Math/Trig.h>
#include <iostream>
using std::cerr;
using std::ostream;

namespace SCIRun {

Persistent* make_GeomDisc()
{
    return scinew GeomDisc;
}

PersistentTypeID GeomDisc::type_id("GeomDisc", "GeomObj", make_GeomDisc);

GeomDisc::GeomDisc(int nu, int nv)
: GeomObj(), n(0,0,1), rad(1), nu(nu), nv(nv)
{
    adjust();
}

GeomDisc::GeomDisc(const Point& cen, const Vector& n,
		   double rad, int nu, int nv)
: GeomObj(), cen(cen), n(n), rad(rad), nu(nu), nv(nv)
{
    adjust();
}

void GeomDisc::move(const Point& _cen, const Vector& _n,
		    double _rad, int _nu, int _nv)
{
    cen=_cen;
    n=_n;
    rad=_rad;
    nu=_nu;
    nv=_nv;
    adjust();
}

GeomDisc::GeomDisc(const GeomDisc& copy)
: GeomObj(), v1(copy.v1), v2(copy.v2), cen(copy.cen), n(copy.n),
  rad(copy.rad), nu(copy.nu), nv(copy.nv)
{
    adjust();
}

GeomDisc::~GeomDisc()
{
}

GeomObj* GeomDisc::clone()
{
    return scinew GeomDisc(*this);
}

void GeomDisc::adjust()
{

    if(n.length2() < 1.e-6){
	cerr << "Degenerate normal on Disc!\n";
    } else {
	n.find_orthogonal(v1, v2);
    }
    n.normalize();
    Vector z(0,0,1);
    if(Abs(n.y()) < 1.e-5){
	// Only in x-z plane...
	zrotaxis=Vector(0,-1,0);
    } else {
	zrotaxis=Cross(n, z);
	zrotaxis.normalize();
    }
    double cangle=Dot(z, n);
    zrotangle=-Acos(cangle);
}

void GeomDisc::get_bounds(BBox& bb)
{
    bb.extend_disc(cen, n, rad);
}

#define GEOMDISC_VERSION 1

void GeomDisc::io(Piostream& stream)
{

    stream.begin_class("GeomDisc", GEOMDISC_VERSION);
    GeomObj::io(stream);
    Pio(stream, cen);
    Pio(stream, n);
    Pio(stream, rad);
    Pio(stream, nu);
    Pio(stream, nv);
    stream.end_class();
}

} // End namespace SCIRun


