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
 *  GeomTriangles.cc: Triangle Strip object
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   June 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#include <Core/Geom/GeomTriangles.h>
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


Persistent* make_GeomTriangles()
{
    return scinew GeomTriangles;
}

PersistentTypeID GeomTriangles::type_id("GeomTriangles", "GeomObj", make_GeomTriangles);

Persistent* make_GeomTranspTriangles()
{
    return scinew GeomTranspTriangles;
}

PersistentTypeID GeomTranspTriangles::type_id("GeomTranspTriangles", "GeomFastTriangles", make_GeomTranspTriangles);


Persistent* make_GeomFastTriangles()
{
    return scinew GeomFastTriangles;
}

PersistentTypeID GeomFastTriangles::type_id("GeomFastTriangles", "GeomObj", make_GeomFastTriangles);

Persistent* make_GeomTrianglesP()
{
    return scinew GeomTrianglesP;
}

PersistentTypeID GeomTrianglesP::type_id("GeomTrianglesP", "GeomObj", make_GeomTrianglesP);

Persistent* make_GeomTrianglesPC()
{
    return scinew GeomTrianglesPC;
}

PersistentTypeID GeomTrianglesPC::type_id("GeomTrianglesPC", "GeomTrianglesP", make_GeomTrianglesPC);

Persistent* make_GeomTrianglesVP()
{
    return scinew GeomTrianglesVP;
}

PersistentTypeID GeomTrianglesVP::type_id("GeomTrianglesVP", "GeomObj", make_GeomTrianglesVP);

Persistent* make_GeomTrianglesVPC()
{
    return scinew GeomTrianglesVPC;
}

PersistentTypeID GeomTrianglesVPC::type_id("GeomTrianglesVPC", "GeomTrianglesVP", make_GeomTrianglesVPC);

Persistent* make_GeomTrianglesPT1d()
{
    return scinew GeomTrianglesPT1d;
}

PersistentTypeID GeomTrianglesPT1d::type_id("GeomTrianglesPT1d", "GeomObj", make_GeomTrianglesPT1d);

Persistent* make_GeomTranspTrianglesP()
{
    return scinew GeomTranspTrianglesP;
}

PersistentTypeID GeomTranspTrianglesP::type_id("GeomTranspTrianglesP", "GeomObj", make_GeomTranspTrianglesP);

GeomTriangles::GeomTriangles()
{
}

GeomTriangles::GeomTriangles(const GeomTriangles& copy)
: GeomVertexPrim(copy)
{
}

GeomTriangles::~GeomTriangles() {
}

void GeomTriangles::add(const Point& p1, const Point& p2, const Point& p3) {
    Vector n(Cross(p2-p1, p3-p1));
#ifndef SCI_NORM_OGL
    if(n.length2() > 0){
	n.normalize();
    } else {
	cerr << "Degenerate triangle in GeomTriangles::add(" << p1 << ", " << p2 << ", " << p3 << ")" << endl;
	return;
    }
#endif
    normals.add(n);
    GeomVertexPrim::add(p1);
    GeomVertexPrim::add(p2);
    GeomVertexPrim::add(p3);
}

int GeomTriangles::size(void)
{
    return verts.size();
}

void GeomTriangles::add(const Point& p1, const Vector& v1,
			const Point& p2, const Vector& v2,
			const Point& p3, const Vector& v3) {
    Vector n(Cross(p2-p1, p3-p1));
#ifndef SCI_NORM_OGL
    if(n.length2() > 0){
	n.normalize();
    } else {
	cerr << "Degenerate triangle in GeomTriangles::add(" << p1 << ", v1, " << p2 << ", v2, " << p3 << ", v3)" << endl;
	return;
    }
#endif
    normals.add(n);
    GeomVertexPrim::add(p1, v1);
    GeomVertexPrim::add(p2, v2);
    GeomVertexPrim::add(p3, v3);
}

void GeomTriangles::add(const Point& p1, const MaterialHandle& m1,
			const Point& p2, const MaterialHandle& m2,
			const Point& p3, const MaterialHandle& m3) {
    Vector n(Cross(p2-p1, p3-p1));
#ifndef SCI_NORM_OGL
    if(n.length2() > 0){
	n.normalize();
    } else {
	cerr << "Degenerate triangle in GeomTriangles::add(" << p1 << ", m1, " << p2 << ", m2, " << p3 << ", m3)" << endl;
	return;
    }
#endif
    normals.add(n);
    GeomVertexPrim::add(p1, m1);
    GeomVertexPrim::add(p2, m2);
    GeomVertexPrim::add(p3, m3);
}

void GeomTriangles::add(const Point& p1, const Color& c1,
			const Point& p2, const Color& c2,
			const Point& p3, const Color& c3) {
    Vector n(Cross(p2-p1, p3-p1));
#ifndef SCI_NORM_OGL
    if(n.length2() > 0){
	n.normalize();
    } else {
	cerr << "Degenerate triangle in GeomTriangles::add(" << p1 << ", c1, " << p2 << ", c2, " << p3 << ", c3)" << endl;
	return;
    }
#endif
    normals.add(n);
    GeomVertexPrim::add(p1, c1);
    GeomVertexPrim::add(p2, c2);
    GeomVertexPrim::add(p3, c3);
}

void GeomTriangles::add(const Point& p1, const Vector& v1, 
			const MaterialHandle& m1, const Point& p2, 
			const Vector& v2, const MaterialHandle& m2,
			const Point& p3, const Vector& v3, 
			const MaterialHandle& m3) {
    Vector n(Cross(p2-p1, p3-p1));
#ifndef SCI_NORM_OGL
    if(n.length2() > 0){
	n.normalize();
    } else {
	cerr << "Degenerate triangle in GeomTriangles::add(" << p1 << ", v1, m1, " << p2 << ", v2, m2, " << p3 << ", v3, m3)" << endl;
	return;
    }
#endif
    normals.add(n);
    GeomVertexPrim::add(p1, v1, m1);
    GeomVertexPrim::add(p2, v2, m2);
    GeomVertexPrim::add(p3, v3, m3);
}

void GeomTriangles::add(GeomVertex* v1, GeomVertex* v2, GeomVertex* v3) {
    Vector n(Cross(v3->p - v1->p, v2->p - v1->p));
#ifndef SCI_NORM_OGL
    if(n.length2() > 0){
	n.normalize();
    } else {
	cerr << "Degenerate triangle in GeomTriangles::add(v1->" << v1->p << ", v2->" << v2->p << ", v3->" << v3->p << ")" << endl;
//	cerr << "Degenerate triangle!!!\n" << endl;
	return;
    }
#endif
    normals.add(n);
    GeomVertexPrim::add(v1);
    GeomVertexPrim::add(v2);
    GeomVertexPrim::add(v3);
}

GeomObj* GeomTriangles::clone()
{
    return scinew GeomTriangles(*this);
}

#define GEOMTRIANGLES_VERSION 1

void GeomTriangles::io(Piostream& stream)
{

    stream.begin_class("GeomTriangles", GEOMTRIANGLES_VERSION);
    GeomVertexPrim::io(stream);
    Pio(stream, normals);
    stream.end_class();
}

GeomFastTriangles::GeomFastTriangles()
  : material_(0)
{
}

GeomFastTriangles::GeomFastTriangles(const GeomFastTriangles& copy)
  : points_(copy.points_),
    colors_(copy.colors_),
    normals_(copy.normals_),
    face_normals_(copy.face_normals_),
    material_(0)
{
}

GeomFastTriangles::~GeomFastTriangles()
{
}


GeomObj*
GeomFastTriangles::clone()
{
  return scinew GeomFastTriangles(*this);
}


int
GeomFastTriangles::size()
{
  return points_.size() / 9;
}


void
GeomFastTriangles::get_bounds(BBox& bb)
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
GeomFastTriangles::add(const Point &p0,
		       const Point &p1,
		       const Point &p2)
{
  Vector n(Cross(p1-p0, p2-p0));
#ifndef SCI_NORM_OGL
  if(n.length2() > 0)
  {
    n.normalize();
  }
  else
  {
    cerr << "Degenerate triangle in GeomTriangles::add(" << p1 << ", " << p2 << ", " << p3 << ")" << endl;
    return;
  }
#endif

  points_.push_back(p0.x());
  points_.push_back(p0.y());
  points_.push_back(p0.z());

  points_.push_back(p1.x());
  points_.push_back(p1.y());
  points_.push_back(p1.z());

  points_.push_back(p2.x());
  points_.push_back(p2.y());
  points_.push_back(p2.z());

  face_normals_.push_back(n.x());
  face_normals_.push_back(n.y());
  face_normals_.push_back(n.z());

  face_normals_.push_back(n.x());
  face_normals_.push_back(n.y());
  face_normals_.push_back(n.z());

  face_normals_.push_back(n.x());
  face_normals_.push_back(n.y());
  face_normals_.push_back(n.z());
}


void
GeomFastTriangles::add(const Point &p0, const Vector &n0,
		       const Point &p1, const Vector &n1,
		       const Point &p2, const Vector &n2)
{
  add(p0, p1, p2);

  normals_.push_back(n0.x());
  normals_.push_back(n0.y());
  normals_.push_back(n0.z());

  normals_.push_back(n1.x());
  normals_.push_back(n1.y());
  normals_.push_back(n1.z());

  normals_.push_back(n2.x());
  normals_.push_back(n2.y());
  normals_.push_back(n2.z());
}


void
GeomFastTriangles::add(const Point &p0, const MaterialHandle &m0,
		       const Point &p1, const MaterialHandle &m1,
		       const Point &p2, const MaterialHandle &m2)
{
  add(p0, p1, p2);

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

  material_ = m0;
}


void
GeomFastTriangles::add(const Point &p0, double i0,
		       const Point &p1, double i1,
		       const Point &p2, double i2)
{
  add(p0, p1, p2);

  indices_.push_back(i0);
  indices_.push_back(i1);
  indices_.push_back(i2);
}


void
GeomFastTriangles::add(const Point &p0, const Vector &n0,
		       const MaterialHandle &m0,
		       const Point &p1, const Vector &n1,
		       const MaterialHandle &m1,
		       const Point &p2, const Vector &n2,
		       const MaterialHandle &m2)
{
  add(p0, m0, p1, m1, p2, m2);

  normals_.push_back(n0.x());
  normals_.push_back(n0.y());
  normals_.push_back(n0.z());

  normals_.push_back(n1.x());
  normals_.push_back(n1.y());
  normals_.push_back(n1.z());

  normals_.push_back(n2.x());
  normals_.push_back(n2.y());
  normals_.push_back(n2.z());
}


void
GeomFastTriangles::add(const Point &p0, const Vector &n0, double i0,
		       const Point &p1, const Vector &n1, double i1,
		       const Point &p2, const Vector &n2, double i2)
{
  add(p0, i0, p1, i1, p2, i2);

  normals_.push_back(n0.x());
  normals_.push_back(n0.y());
  normals_.push_back(n0.z());

  normals_.push_back(n1.x());
  normals_.push_back(n1.y());
  normals_.push_back(n1.z());

  normals_.push_back(n2.x());
  normals_.push_back(n2.y());
  normals_.push_back(n2.z());
}


#define GEOMFASTTRIANGLES_VERSION 1

void GeomFastTriangles::io(Piostream& stream)
{

    stream.begin_class("GeomFastTriangles", GEOMFASTTRIANGLES_VERSION);
    Pio(stream, points_);
    Pio(stream, colors_);
    Pio(stream, indices_);
    Pio(stream, normals_);
    stream.end_class();
}


GeomTranspTriangles::GeomTranspTriangles()
  : xreverse_(false),
    yreverse_(false),
    zreverse_(false)
{
}


GeomTranspTriangles::GeomTranspTriangles(const GeomTranspTriangles& copy)
  : GeomFastTriangles(copy),
    xlist_(copy.xlist_),
    ylist_(copy.ylist_),
    zlist_(copy.zlist_),
    xreverse_(copy.xreverse_),
    yreverse_(copy.yreverse_),
    zreverse_(copy.zreverse_)
{
}

GeomTranspTriangles::~GeomTranspTriangles()
{
}


GeomObj* GeomTranspTriangles::clone()
{
    return scinew GeomTranspTriangles(*this);
}

void
GeomTranspTriangles::SortPolys()
{
  const unsigned int vsize = points_.size() / 9;
  if (xlist_.size() == vsize*3) return;

  xreverse_ = false;
  yreverse_ = false;
  zreverse_ = false;

  vector<pair<float, unsigned int> > tmp(vsize);
  unsigned int i;

  for (i = 0; i < vsize;i++)
  {
    tmp[i].first = points_[i*9+0] + points_[i*9+3] + points_[i*9+6];
    tmp[i].second = i*3;
  }
  std::sort(tmp.begin(), tmp.end(), pair_less);

  xlist_.resize(vsize*3);
  for (i=0; i < vsize; i++)
  {
    xlist_[i*3+0] = tmp[i].second + 0;
    xlist_[i*3+1] = tmp[i].second + 1;
    xlist_[i*3+2] = tmp[i].second + 2;
  }

  for (i = 0; i < vsize;i++)
  {
    tmp[i].first = points_[i*9+1] + points_[i*9+4] + points_[i*9+7];
    tmp[i].second = i*3;
  }
  std::sort(tmp.begin(), tmp.end(), pair_less);

  ylist_.resize(vsize*3);
  for (i=0; i < vsize; i++)
  {
    ylist_[i*3+0] = tmp[i].second + 0;
    ylist_[i*3+1] = tmp[i].second + 1;
    ylist_[i*3+2] = tmp[i].second + 2;
  }

  for (i = 0; i < vsize;i++)
  {
    tmp[i].first = points_[i*9+2] + points_[i*9+5] + points_[i*9+8];
    tmp[i].second = i*3;
  }
  std::sort(tmp.begin(), tmp.end(), pair_less);

  zlist_.resize(vsize*3);
  for (i=0; i < vsize; i++)
  {
    zlist_[i*3+0] = tmp[i].second + 0;
    zlist_[i*3+1] = tmp[i].second + 1;
    zlist_[i*3+2] = tmp[i].second + 2;
  }
}


#define GEOMTRANSPTRIANGLES_VERSION 2

void GeomTranspTriangles::io(Piostream& stream)
{
  stream.begin_class("GeomTranspTriangles", GEOMTRANSPTRIANGLES_VERSION);
  GeomFastTriangles::io(stream);
  stream.end_class();
}


GeomTrianglesPT1d::GeomTrianglesPT1d()
:GeomTrianglesP(),cmap(0)
{
}

GeomTrianglesPT1d::~GeomTrianglesPT1d()
{
}

int GeomTrianglesPT1d::add(const Point& p1,const Point& p2,const Point& p3,
			   const float& f1,const float& f2,const float& f3)
{
  if (GeomTrianglesP::add(p1,p2,p3)) {
    scalars.add(f1);
    scalars.add(f2);
    scalars.add(f3);
    return 1;
  }
  return 0;
}


#define GeomTrianglesPT1d_VERSION 1

void GeomTrianglesPT1d::io(Piostream& stream)
{

    stream.begin_class("GeomTrianglesPT1d", GeomTrianglesPT1d_VERSION);
    GeomTrianglesP::io(stream);
    Pio(stream, scalars); // just save scalar values
    stream.end_class();
}


GeomTranspTrianglesP::GeomTranspTrianglesP()
  : alpha_(0.2), sorted_p_(false)
{
}

GeomTranspTrianglesP::GeomTranspTrianglesP(double aval)
  : alpha_(aval), sorted_p_(false)
{
}

GeomTranspTrianglesP::~GeomTranspTrianglesP()
{
}

int
GeomTranspTrianglesP::vadd(const Point& p1, 
			   const Point& p2,
			   const Point& p3)
{
  if (add(p1,p2,p3)) {
    const unsigned int index = xlist_.size();
    const Vector center = (p1.vector()+p2.vector()+p3.vector())*(1.0/3.0);
    xlist_.push_back(pair<float, unsigned int>(center.x(), index));
    ylist_.push_back(pair<float, unsigned int>(center.y(), index));
    zlist_.push_back(pair<float, unsigned int>(center.z(), index));
    sorted_p_ = false;
    return 1;
  } 
  return 0;
}


void
GeomTranspTrianglesP::SortPolys()
{
  std::sort(xlist_.begin(), xlist_.end(), pair_less);
  std::sort(ylist_.begin(), ylist_.end(), pair_less);
  std::sort(zlist_.begin(), zlist_.end(), pair_less);
  sorted_p_ = true;
}


// grows points, normals and centers...
#if 0
void GeomTranspTrianglesP::MergeStuff(GeomTranspTrianglesP* other)
{
  points.resize(points.size() + other->points.size());
  normals.resize(normals.size() + other->normals.size());

  xc.resize(xc.size() + other->xc.size());
  yc.resize(yc.size() + other->yc.size());
  zc.resize(zc.size() + other->zc.size());
  
  int start = points.size()-other->points.size();
  int i;
  for(i=0;i<other->points.size();i++) {
    points[i+start] = other->points[i];
  }
  start = normals.size() - other->normals.size();
  for(i=0;i<other->normals.size();i++) {
    normals[i+start] = other->normals[i];
  }

  start = xc.size() - other->xc.size();
  for(i=0;i<other->xc.size();i++) {
    xc[start +i] = other->xc[i];
    yc[start +i] = other->yc[i];
    zc[start +i] = other->zc[i];
  }
  other->points.resize(0);
  other->normals.resize(0);
  other->xc.resize(0);
  other->yc.resize(0);
  other->zc.resize(0);
}
#endif

#define GeomTranspTrianglesP_VERSION 1

void GeomTranspTrianglesP::io(Piostream& stream)
{

    stream.begin_class("GeomTranspTrianglesP", GeomTranspTrianglesP_VERSION);
    GeomTrianglesP::io(stream);
    Pio(stream, alpha_); // just save transparency value...
    stream.end_class();
}

GeomTrianglesP::GeomTrianglesP()
:has_color(0)
{
    // don't really need to do anythin...
}

GeomTrianglesP::~GeomTrianglesP()
{

}

void GeomTrianglesP::get_triangles( Array1<float> &v)
{
  int end = v.size();
  v.grow (points.size());
  for (int i=0; i<points.size(); i++)
    v[end+i] = points[i];
}

int GeomTrianglesP::size(void)
{
    return points.size()/9;
}

void GeomTrianglesP::reserve_clear(int n)
{
    points.setsize(n*9);
    normals.setsize(n*3);

    points.remove_all();
    normals.remove_all();
}

int GeomTrianglesP::add(const Point& p1, const Point& p2, const Point& p3)
{
    Vector n(Cross(p2-p1, p3-p1));
#ifndef SCI_NORM_OGL
    if(n.length2() > 0){
        n.normalize();
    }   	
    else {
//	cerr << "degenerate triangle!!!\n" << endl;
	return 0;
    }
#endif

    int idx=normals.size();
    normals.grow(3);
    normals[idx+0]=n.x();
    normals[idx+1]=n.y();
    normals[idx+2]=n.z();


    idx=points.size();
    points.grow(9);
    points[idx+0]=p1.x();
    points[idx+1]=p1.y();
    points[idx+2]=p1.z();
    points[idx+3]=p2.x();
    points[idx+4]=p2.y();
    points[idx+5]=p2.z();
    points[idx+6]=p3.x();
    points[idx+7]=p3.y();
    points[idx+8]=p3.z();
    return 1;
}

// below is just a virtual function...

int GeomTrianglesP::vadd(const Point& p1, const Point& p2, const Point& p3)
{
    Vector n(Cross(p2-p1, p3-p1));
#ifndef SCI_NORM_OGL
    if(n.length2() > 0){
        n.normalize();
    }   	
    else {
//	cerr << "degenerate triangle!!!\n" << endl;
	return 0;
    }
#endif

    int idx=normals.size();
    normals.grow(3);
    normals[idx]=n.x();
    normals[idx+1]=n.y();
    normals[idx+2]=n.z();


    idx=points.size();
    points.grow(9);
    points[idx]=p1.x();
    points[idx+1]=p1.y();
    points[idx+2]=p1.z();
    points[idx+3]=p2.x();
    points[idx+4]=p2.y();
    points[idx+5]=p2.z();
    points[idx+6]=p3.x();
    points[idx+7]=p3.y();
    points[idx+8]=p3.z();
    return 1;
}

GeomObj* GeomTrianglesP::clone()
{
    return new GeomTrianglesP(*this);
}

void GeomTrianglesP::get_bounds(BBox& box)
{
    for(int i=0;i<points.size();i+=3)
	box.extend(Point(points[i],points[i+1],points[i+2]));
}

#define GEOMTRIANGLESP_VERSION 1

void GeomTrianglesP::io(Piostream& stream)
{

    stream.begin_class("GeomTrianglesP", GEOMTRIANGLESP_VERSION);
    GeomObj::io(stream);
    Pio(stream, points);
    Pio(stream, normals);
    stream.end_class();
}

GeomTrianglesPC::GeomTrianglesPC()
{
    // don't really need to do anythin...
}

GeomTrianglesPC::~GeomTrianglesPC()
{

}

int GeomTrianglesPC::add(const Point& p1, const Color& c1,
			const Point& p2, const Color& c2,
			const Point& p3, const Color& c3)
{
    if (GeomTrianglesP::add(p1,p2,p3)) {
	colors.add(c1.r());
	colors.add(c1.g());
	colors.add(c1.b());

	colors.add(c2.r());
	colors.add(c2.g());
	colors.add(c2.b());

	colors.add(c3.r());
	colors.add(c3.g());
	colors.add(c3.b());
	return 1;
    }

    return 0;
}

#define GEOMTRIANGLESPC_VERSION 1

void GeomTrianglesPC::io(Piostream& stream)
{

    stream.begin_class("GeomTrianglesPC", GEOMTRIANGLESPC_VERSION);
    GeomTrianglesP::io(stream);
    Pio(stream, colors);
    stream.end_class();
}

GeomTrianglesVP::GeomTrianglesVP()
{
    // don't really need to do anythin...
}

GeomTrianglesVP::~GeomTrianglesVP()
{

}

int GeomTrianglesVP::size(void)
{
    return points.size()/9;
}

void GeomTrianglesVP::reserve_clear(int n)
{
    int np = points.size()/9;
    int delta = n - np;

    points.remove_all();
    normals.remove_all();

    if (delta > 0) {
	points.grow(delta);
	normals.grow(delta);
    }
	
}

int GeomTrianglesVP::add(const Point& p1, const Vector &v1,
			 const Point& p2, const Vector &v2,	
			 const Point& p3, const Vector &v3)
{
    int idx=normals.size();
    normals.grow(9);
    normals[idx]=v1.x();
    normals[idx+1]=v1.y();
    normals[idx+2]=v1.z();
    normals[idx+3]=v2.x();
    normals[idx+4]=v2.y();
    normals[idx+5]=v2.z();
    normals[idx+6]=v3.x();
    normals[idx+7]=v3.y();
    normals[idx+8]=v3.z();

    idx=points.size();
    points.grow(9);
    points[idx]=p1.x();
    points[idx+1]=p1.y();
    points[idx+2]=p1.z();
    points[idx+3]=p2.x();
    points[idx+4]=p2.y();
    points[idx+5]=p2.z();
    points[idx+6]=p3.x();
    points[idx+7]=p3.y();
    points[idx+8]=p3.z();
    return 1;
}

GeomObj* GeomTrianglesVP::clone()
{
    return new GeomTrianglesVP(*this);
}

void GeomTrianglesVP::get_bounds(BBox& box)
{
    for(int i=0;i<points.size();i+=3)
	box.extend(Point(points[i],points[i+1],points[i+2]));
}

#define GEOMTRIANGLESVP_VERSION 1

void GeomTrianglesVP::io(Piostream& stream)
{

    stream.begin_class("GeomTrianglesVP", GEOMTRIANGLESVP_VERSION);
    GeomObj::io(stream);
    Pio(stream, points);
    Pio(stream, normals);
    stream.end_class();
}

GeomTrianglesVPC::GeomTrianglesVPC()
{
    // don't really need to do anythin...
}

GeomTrianglesVPC::~GeomTrianglesVPC()
{

}

int GeomTrianglesVPC::add(const Point& p1, const Vector &v1, const Color& c1,
			  const Point& p2, const Vector &v2, const Color& c2,
			  const Point& p3, const Vector &v3, const Color& c3)
{
    if (GeomTrianglesVP::add(p1,v1,p2,v2,p3,v3)) {
	colors.add(c1.r());
	colors.add(c1.g());
	colors.add(c1.b());

	colors.add(c2.r());
	colors.add(c2.g());
	colors.add(c2.b());

	colors.add(c3.r());
	colors.add(c3.g());
	colors.add(c3.b());
	return 1;
    }

    return 0;
}

#define GEOMTRIANGLESVPC_VERSION 1

void GeomTrianglesVPC::io(Piostream& stream)
{

    stream.begin_class("GeomTrianglesVPC", GEOMTRIANGLESVPC_VERSION);
    GeomTrianglesVP::io(stream);
    Pio(stream, colors);
    stream.end_class();
}


} // End namespace SCIRun

