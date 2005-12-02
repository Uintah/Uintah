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
 *  GeomBox.cc:  A box object
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   Feb. 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#include <Core/Geom/GeomBox.h>
#include <Core/Util/NotFinished.h>
#include <Core/Geometry/BBox.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MinMax.h>
#include <Core/Persistent/PersistentSTL.h>
#include <iostream>
using std::ostream;

namespace SCIRun {

Persistent* make_GeomBox()
{
    return scinew GeomBox(Point(0,0,0), Point(1,1,1), 1);
}

PersistentTypeID GeomBox::type_id("GeomBox", "GeomObj", make_GeomBox);

GeomBox::GeomBox(const Point& p, const Point& q, int op) : GeomObj()
{

  min = Min( p, q );
  max = Max( p, q );

  for (int i=0; i<6; i++ )
    opacity[i] = op;
}

GeomBox::GeomBox(const GeomBox& copy)
: GeomObj(copy)
{
  min = copy.min;
  max = copy.max;
  for (int s=0; s<6; s++)
    opacity[s] = copy.opacity[s];
}

GeomBox::~GeomBox()
{
}

GeomObj* GeomBox::clone()
{
    return scinew GeomBox(*this);
}

void GeomBox::get_bounds(BBox& bb)
{
  bb.extend(min);
  bb.extend(max);
}

#define GEOMBOX_VERSION 1

void GeomBox::io(Piostream& stream)
{

    stream.begin_class("GeomBox", GEOMBOX_VERSION);
    GeomObj::io(stream);
    Pio(stream, min);
    Pio(stream, max);
    
    for ( int j=0; j<6; j++ )
      Pio(stream, opacity[j]);
    stream.end_class();
}

Persistent* make_GeomSimpleBox()
{
    return scinew GeomSimpleBox(Point(0,0,0), Point(1,1,1));
}

PersistentTypeID GeomSimpleBox::type_id("GeomSimpleBox", "GeomObj", make_GeomSimpleBox);


GeomSimpleBox::GeomSimpleBox(const Point& p, const Point& q) : GeomObj()
{
  min = Min( p, q );
  max = Max( p, q );
}


GeomSimpleBox::GeomSimpleBox(const GeomSimpleBox& copy)
  : GeomObj(copy), min(copy.min), max(copy.max)
{
}

GeomSimpleBox::~GeomSimpleBox()
{
}

GeomObj* GeomSimpleBox::clone()
{
    return scinew GeomSimpleBox(*this);
}

void
GeomSimpleBox::get_bounds(BBox& bb)
{
  bb.extend(min);
  bb.extend(max);
}

#define GEOMSIMPLEBOX_VERSION 1

void
GeomSimpleBox::io(Piostream& stream)
{

    stream.begin_class("GeomSimpleBox", GEOMSIMPLEBOX_VERSION);
    GeomObj::io(stream);
    Pio(stream, min);
    Pio(stream, max);
    stream.end_class();
}


Persistent* make_GeomCBox()
{
    return scinew GeomCBox(Point(0,0,0), Point(1,1,1));
}

PersistentTypeID GeomCBox::type_id("GeomCBox", "GeomObj", make_GeomCBox);


GeomCBox::GeomCBox(const Point& p, const Point& q) : GeomSimpleBox(p, q)
{
}


GeomCBox::GeomCBox(const GeomCBox& copy)
  : GeomSimpleBox(copy)
{
}

GeomCBox::~GeomCBox()
{
}

GeomObj* GeomCBox::clone()
{
    return scinew GeomCBox(*this);
}

#define GEOMCBOX_VERSION 1

void
GeomCBox::io(Piostream& stream)
{

    stream.begin_class("GeomCBox", GEOMCBOX_VERSION);
    GeomSimpleBox::io(stream);
    stream.end_class();
}


Persistent* make_GeomBoxes()
{
  return scinew GeomBoxes;
}


PersistentTypeID GeomBoxes::type_id("GeomBoxes", "GeomObj", make_GeomBoxes);


GeomBoxes::GeomBoxes(double edge, int nu, int nv)
  : GeomObj(),
    nu_(nu),
    nv_(nv),
    global_edge_(edge)
{
}


GeomBoxes::GeomBoxes(const GeomBoxes& copy)
  : GeomObj(copy),
    centers_(copy.centers_),
    edges_(copy.edges_),
    colors_(copy.colors_),
    indices_(copy.indices_),
    nu_(copy.nu_),
    nv_(copy.nv_),
    global_edge_(copy.global_edge_)
{
}


GeomBoxes::~GeomBoxes()
{
}


GeomObj *
GeomBoxes::clone()
{
  return scinew GeomBoxes(*this);
}


void
GeomBoxes::get_bounds(BBox& bb)
{
  const bool ugr = !(edges_.size() == centers_.size());
  for (unsigned int i=0; i < centers_.size(); i++)
  {
    bb.extend(centers_[i], ugr?global_edge_:edges_[i]);
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
GeomBoxes::add(const Point &center)
{
  centers_.push_back(center);
}


void
GeomBoxes::add(const Point &center, const MaterialHandle &mat)
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
GeomBoxes::add(const Point &center, float index)
{
  add(center);
  indices_.push_back(index);
}


bool
GeomBoxes::add_edge(const Point &c, double r)
{
  if (r < 1.0e-6) { return false; }
  centers_.push_back(c);
  edges_.push_back(r);
  return true;
}

bool
GeomBoxes::add_edge(const Point &c, double r, const MaterialHandle &mat)
{
  if (r < 1.0e-6) { return false; }
  add_edge(c, r);
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
GeomBoxes::add_edge(const Point &c, double r, float index)
{
  if (r < 1.0e-6) { return false; }
  add_edge(c, r);
  indices_.push_back(index);
  return true;
}



#define GEOMBOXES_VERSION 1

void
GeomBoxes::io(Piostream& stream)
{
  stream.begin_class("GeomBoxes", GEOMBOXES_VERSION);
  GeomObj::io(stream);
  Pio(stream, centers_);
  Pio(stream, edges_);
  Pio(stream, colors_);
  Pio(stream, indices_);
  Pio(stream, nu_);
  Pio(stream, nv_);
  Pio(stream, global_edge_);
  stream.end_class();
}

} // End namespace SCIRun
