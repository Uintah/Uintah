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
 *  TriStrip.cc: Triangle Strip object
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Core/Geom/GeomTriStrip.h>
#include <Core/Util/NotFinished.h>
#include <Core/Geom/GeomSave.h>
#include <Core/Geom/GeomTri.h>
#include <Core/Geometry/BBox.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>
using std::cerr;
using std::ostream;

namespace SCIRun {

Persistent* make_GeomTriStrip()
{
    return scinew GeomTriStrip;
}

PersistentTypeID GeomTriStrip::type_id("GeomTriStrip", "GeomObj", make_GeomTriStrip);

Persistent* make_GeomTriStripList()
{
    return scinew GeomTriStripList;
}

PersistentTypeID GeomTriStripList::type_id("GeomTriStripList", "GeomObj",
					   make_GeomTriStripList);

GeomTriStrip::GeomTriStrip()
{
}

GeomTriStrip::GeomTriStrip(const GeomTriStrip& copy)
: GeomVertexPrim(copy)
{
}

GeomTriStrip::~GeomTriStrip() {
}

GeomObj* GeomTriStrip::clone()
{
    return scinew GeomTriStrip(*this);
}

#define GEOMTRISTRIP_VERSION 1

void GeomTriStrip::io(Piostream& stream)
{
    stream.begin_class("GeomTriStrip", GEOMTRISTRIP_VERSION);
    GeomVertexPrim::io(stream);
    stream.end_class();
}

int GeomTriStrip::size(void)
{
    return verts.size();
}

GeomTriStripList::GeomTriStripList()
:n_strips(0)
{

}

GeomTriStripList::~GeomTriStripList()
{
    // everything should be deleted
}

GeomObj* GeomTriStripList::clone()
{
    return new GeomTriStripList(*this);
}

void GeomTriStripList::add(const Point& p)
{
    int last = pts.size();
    pts.grow(3);
    pts[last++] = float(p.x());
    pts[last++] = float(p.y());
    pts[last++] = float(p.z());

}


void GeomTriStripList::add(const Point& p, const Vector& v)
{
    int last = pts.size();
    pts.grow(3);

    pts[last++] = float(p.x());
    pts[last++] = float(p.y());
    pts[last++] = float(p.z());

    last = nrmls.size();
    nrmls.grow(3);
    nrmls[last++] = float(v.x());
    nrmls[last++] = float(v.y());
    nrmls[last++] = float(v.z());
}

void GeomTriStripList::end_strip(void)
{
    int last = pts.size();
    int last_s = strips.size();
    
    strips.grow(1);
    strips[last_s] = last;
}

int GeomTriStripList::size(void)
{
    return nrmls.size()/3;
}

#define GEOMTRISTRIPLIST_VERSION 1

void GeomTriStripList::io(Piostream& stream)
{

    stream.begin_class("GeomTriStripList", GEOMTRISTRIPLIST_VERSION);
    GeomObj::io(stream);
    Pio(stream, n_strips);
    Pio(stream, pts);
    Pio(stream, nrmls);
    Pio(stream, strips);
    stream.end_class();
}

Point GeomTriStripList::get_pm1(void)
{
    int npts = pts.size()-1;
    Point p(pts[npts-2],pts[npts-1],pts[npts]);
    return p;
}

Point GeomTriStripList::get_pm2(void)
{
    int npts = pts.size()-1-3;
    Point p(pts[npts-2],pts[npts-1],pts[npts]);
    return p;
}

void GeomTriStripList::permute(int i0,int i1, int i2)
{
    float *ptr = &pts[pts.size()-1-8];
    float remap[9];

    remap[0] = ptr[i0*3 + 0];
    remap[1] = ptr[i0*3 + 1];
    remap[2] = ptr[i0*3 + 2];

    remap[3] = ptr[i1*3 + 0];
    remap[4] = ptr[i1*3 + 1];
    remap[5] = ptr[i1*3 + 2];

    remap[6] = ptr[i2*3 + 0];
    remap[7] = ptr[i2*3 + 1];
    remap[8] = ptr[i2*3 + 2];

    for(int i=0;i<9;i++) 
	ptr[i] = remap[i];
}

void GeomTriStripList::get_bounds(BBox& box)
{
    for(int i=0;i<pts.size();i+=3)
	box.extend(Point(pts[i],pts[i+1],pts[i+2]));
}

int GeomTriStripList::num_since(void)
{
    int ssize = strips.size();
    int n_on = pts.size()/3;
    if (ssize) {
	int lastp = strips[ssize-1];
	n_on -= lastp/3;
    }

    return n_on;

}

} // End namespace SCIRun

