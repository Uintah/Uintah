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
 *  GeomQuads.cc: Fast Quads object
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   May 2003
 *
 *  Copyright (C) 2003 SCI Group
 */

#include <Core/Geom/GeomQuads.h>
#include <Core/Util/NotFinished.h>
#include <Core/Geom/GeomSave.h>
#include <Core/Geom/GeomTri.h>
#include <Core/Geometry/BBox.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Persistent/PersistentSTL.h>
#include <iostream>
#include <functional>
#include <algorithm>

using std::cerr;
using std::ostream;
#include <stdio.h>

namespace SCIRun {

static bool
pair_less(const pair<double, unsigned int> &a,
	  const pair<double, unsigned int> &b)
{
  return a.first < b.first;
}


Persistent* make_GeomTranspQuads()
{
    return scinew GeomTranspQuads;
}

PersistentTypeID GeomTranspQuads::type_id("GeomTranspQuads", "GeomFastQuads", make_GeomTranspQuads);


Persistent* make_GeomFastQuads()
{
    return scinew GeomFastQuads;
}

PersistentTypeID GeomFastQuads::type_id("GeomFastQuads", "GeomObj", make_GeomFastQuads);


GeomFastQuads::GeomFastQuads()
  : material_(0)
{
}

GeomFastQuads::GeomFastQuads(const GeomFastQuads& copy)
  : points_(copy.points_),
    colors_(copy.colors_),
    normals_(copy.normals_),
    material_(0)
{
}

GeomFastQuads::~GeomFastQuads()
{
}


GeomObj*
GeomFastQuads::clone()
{
  return scinew GeomFastQuads(*this);
}


int
GeomFastQuads::size()
{
  return points_.size() / 12;
}


void
GeomFastQuads::get_bounds(BBox& bb)
{
  for(unsigned int i=0;i<points_.size();i+=3)
  {
    bb.extend(Point(points_[i+0], points_[i+1], points_[i+2]));
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
GeomFastQuads::add(const Point &p0, const Point &p1,
		   const Point &p2, const Point &p3)
{
  // Assume planar, use first three to compute normal.
  Vector n(Cross(p1-p0, p2-p0));
#ifndef SCI_NORM_OGL
  if(n.length2() > 0)
  {
    n.normalize();
  }
  else
  {
    cerr << "Degenerate triangle in GeomQuads::add(" << p1 << ", " << p2 << ", " << p3 << ")" << endl;
    return;
  }
#endif

  add(p0, n, p1, n, p2, n, p3, n);
}



void
GeomFastQuads::add(const Point &p0, const Vector &n0,
		   const Point &p1, const Vector &n1,
		   const Point &p2, const Vector &n2,
		   const Point &p3, const Vector &n3)
{
  points_.push_back(p0.x());
  points_.push_back(p0.y());
  points_.push_back(p0.z());

  points_.push_back(p1.x());
  points_.push_back(p1.y());
  points_.push_back(p1.z());

  points_.push_back(p2.x());
  points_.push_back(p2.y());
  points_.push_back(p2.z());

  points_.push_back(p3.x());
  points_.push_back(p3.y());
  points_.push_back(p3.z());


  normals_.push_back(n0.x());
  normals_.push_back(n0.y());
  normals_.push_back(n0.z());

  normals_.push_back(n1.x());
  normals_.push_back(n1.y());
  normals_.push_back(n1.z());

  normals_.push_back(n2.x());
  normals_.push_back(n2.y());
  normals_.push_back(n2.z());

  normals_.push_back(n3.x());
  normals_.push_back(n3.y());
  normals_.push_back(n3.z());
}


void
GeomFastQuads::add(const Point &p0, const MaterialHandle &m0,
		   const Point &p1, const MaterialHandle &m1,
		   const Point &p2, const MaterialHandle &m2,
		   const Point &p3, const MaterialHandle &m3)
{
  // Assume planar, use first three to compute normal.
  Vector n(Cross(p1-p0, p2-p0));
#ifndef SCI_NORM_OGL
  if(n.length2() > 0)
  {
    n.normalize();
  }
  else
  {
    cerr << "Degenerate triangle in GeomQuads::add(" << p1 << ", " << p2 << ", " << p3 << ")" << endl;
    return;
  }
#endif

  add(p0, n, m0, p1, n, m1, p2, n, m2, p3, n, m3);
}



void
GeomFastQuads::add(const Point &p0, double i0,
		   const Point &p1, double i1,
		   const Point &p2, double i2,
		   const Point &p3, double i3)
{
  add(p0, p1, p2, p3);
  
  indices_.push_back(i0);
  indices_.push_back(i1);
  indices_.push_back(i2);
  indices_.push_back(i3);
}



void
GeomFastQuads::add(const Point &p0, const Vector &n0,
		   const MaterialHandle &m0,
		   const Point &p1, const Vector &n1,
		   const MaterialHandle &m1,
		   const Point &p2, const Vector &n2,
		   const MaterialHandle &m2,
		   const Point &p3, const Vector &n3,
		   const MaterialHandle &m3)
{
  add(p0, n0, p1, n1, p2, n2, p3, n3);

  colors_.push_back(COLOR_FTOB(m0->diffuse.r()));
  colors_.push_back(COLOR_FTOB(m0->diffuse.g()));
  colors_.push_back(COLOR_FTOB(m0->diffuse.b()));
  colors_.push_back(COLOR_FTOB(m0->transparency * m0->transparency *
			       m0->transparency * m0->transparency));

  colors_.push_back(COLOR_FTOB(m1->diffuse.r()));
  colors_.push_back(COLOR_FTOB(m1->diffuse.g()));
  colors_.push_back(COLOR_FTOB(m1->diffuse.b()));
  colors_.push_back(COLOR_FTOB(m1->transparency * m1->transparency *
			       m1->transparency * m1->transparency));

  colors_.push_back(COLOR_FTOB(m2->diffuse.r()));
  colors_.push_back(COLOR_FTOB(m2->diffuse.g()));
  colors_.push_back(COLOR_FTOB(m2->diffuse.b()));
  colors_.push_back(COLOR_FTOB(m2->transparency * m2->transparency *
			       m2->transparency * m2->transparency));

  colors_.push_back(COLOR_FTOB(m3->diffuse.r()));
  colors_.push_back(COLOR_FTOB(m3->diffuse.g()));
  colors_.push_back(COLOR_FTOB(m3->diffuse.b()));
  colors_.push_back(COLOR_FTOB(m3->transparency * m2->transparency *
			       m3->transparency * m2->transparency));

  material_ = m0;
}


void
GeomFastQuads::add(const Point &p0, const Vector &n0, double i0,
		   const Point &p1, const Vector &n1, double i1,
		   const Point &p2, const Vector &n2, double i2,
		   const Point &p3, const Vector &n3, double i3)
{
  add(p0, n0, p1, n1, p2, n2, p3, n3);

  indices_.push_back(i0);
  indices_.push_back(i1);
  indices_.push_back(i2);
  indices_.push_back(i3);
}



#define GEOMFASTQUADS_VERSION 1

void GeomFastQuads::io(Piostream& stream)
{

    stream.begin_class("GeomFastQuads", GEOMFASTQUADS_VERSION);
    Pio(stream, points_);
    Pio(stream, colors_);
    Pio(stream, indices_);
    Pio(stream, normals_);
    stream.end_class();
}


GeomTranspQuads::GeomTranspQuads()
  : xreverse_(false),
    yreverse_(false),
    zreverse_(false)
{
}

GeomTranspQuads::GeomTranspQuads(const GeomTranspQuads& copy)
  : GeomFastQuads(copy),
    xlist_(copy.xlist_),
    ylist_(copy.ylist_),
    zlist_(copy.zlist_),
    xreverse_(copy.xreverse_),
    yreverse_(copy.yreverse_),
    zreverse_(copy.zreverse_)
{
}

GeomTranspQuads::~GeomTranspQuads()
{
}


GeomObj* GeomTranspQuads::clone()
{
    return scinew GeomTranspQuads(*this);
}

void
GeomTranspQuads::SortPolys()
{
  const unsigned int vsize = points_.size() / 12;
  if (xlist_.size() == vsize*4) return;

  xreverse_ = false;
  yreverse_ = false;
  zreverse_ = false;

  vector<pair<float, unsigned int> > tmp(vsize);
  unsigned int i;

  for (i = 0; i < vsize;i++)
  {
    tmp[i].first = points_[i*12+0] + points_[i*12+3] +
      points_[i*12+6] + points_[i*12+9];
    tmp[i].second = i*4;
  }
  std::sort(tmp.begin(), tmp.end(), pair_less);

  xlist_.resize(vsize*4);
  for (i=0; i < vsize; i++)
  {
    xlist_[i*4+0] = tmp[i].second + 0;
    xlist_[i*4+1] = tmp[i].second + 1;
    xlist_[i*4+2] = tmp[i].second + 2;
    xlist_[i*4+3] = tmp[i].second + 3;
  }

  for (i = 0; i < vsize;i++)
  {
    tmp[i].first = points_[i*12+1] + points_[i*12+4] +
      points_[i*12+7] + points_[i*12+10];
    tmp[i].second = i*4;
  }
  std::sort(tmp.begin(), tmp.end(), pair_less);

  ylist_.resize(vsize*4);
  for (i=0; i < vsize; i++)
  {
    ylist_[i*4+0] = tmp[i].second + 0;
    ylist_[i*4+1] = tmp[i].second + 1;
    ylist_[i*4+2] = tmp[i].second + 2;
    ylist_[i*4+3] = tmp[i].second + 3;
  }

  for (i = 0; i < vsize;i++)
  {
    tmp[i].first = points_[i*12+2] + points_[i*12+5] +
      points_[i*12+8] + points_[i*12+11];
    tmp[i].second = i*4;
  }
  std::sort(tmp.begin(), tmp.end(), pair_less);

  zlist_.resize(vsize*4);
  for (i=0; i < vsize; i++)
  {
    zlist_[i*4+0] = tmp[i].second + 0;
    zlist_[i*4+1] = tmp[i].second + 1;
    zlist_[i*4+2] = tmp[i].second + 2;
    zlist_[i*4+3] = tmp[i].second + 3;
  }
}


#define GEOMTRANSPQUADS_VERSION 1

void GeomTranspQuads::io(Piostream& stream)
{
  stream.begin_class("GeomTranspQuads", GEOMTRANSPQUADS_VERSION);
  GeomFastQuads::io(stream);
  stream.end_class();
}

} // End namespace SCIRun

