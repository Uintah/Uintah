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
 *  GeomLine.cc:  Line object
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifdef _WIN32
#pragma warning(disable:4291) // quiet the visual C++ compiler
#endif

#include <Core/Geom/GeomLine.h>
#include <Core/Util/NotFinished.h>
#include <Core/Containers/TrivialAllocator.h>
#include <Core/Geometry/BBox.h>
#include <Core/Persistent/PersistentSTL.h>
#include <iostream>
#include <algorithm>

#include <stdlib.h>

namespace SCIRun {

using std::cerr;
using std::ostream;


Persistent* make_GeomLine()
{
    return new GeomLine(Point(0,0,0), Point(1,1,1));
}

PersistentTypeID GeomLine::type_id("GeomLine", "GeomObj", make_GeomLine);

static TrivialAllocator Line_alloc(sizeof(GeomLine));

void* GeomLine::operator new(size_t)
{
    return Line_alloc.alloc();
}

void GeomLine::operator delete(void* rp, size_t)
{	
    Line_alloc.free(rp);
}

GeomLine::GeomLine(const Point& p1, const Point& p2) : 
  GeomObj(), 
  p1(p1), 
  p2(p2),
  lineWidth_(1.0)
{
}

GeomLine::GeomLine(const GeomLine& copy) : 
  GeomObj(), 
  p1(copy.p1), 
  p2(copy.p2),
  lineWidth_(1.0)
{
}

GeomLine::~GeomLine()
{
}

GeomObj* GeomLine::clone()
{    return new GeomLine(*this);
}

void GeomLine::get_bounds(BBox& bb)
{
  bb.extend(p1);
  bb.extend(p2);
}

void
GeomLine::setLineWidth(float val) 
{
  lineWidth_ = val;
}

#define GEOMLINE_VERSION 1

void GeomLine::io(Piostream& stream)
{

  stream.begin_class("GeomLine", GEOMLINE_VERSION);
  GeomObj::io(stream);
  Pio(stream, p1);
  Pio(stream, p2);
  stream.end_class();
}

Persistent* make_GeomLines()
{
  return new GeomLines();
}

PersistentTypeID GeomLines::type_id("GeomLines", "GeomObj", make_GeomLines);

GeomLines::GeomLines()
{
}

GeomLines::GeomLines(const GeomLines& copy)
  : pts(copy.pts)
{
}

GeomLines::~GeomLines()
{
}

GeomObj* GeomLines::clone()
{
  return new GeomLines(*this);
}

void GeomLines::get_bounds(BBox& bb)
{
  for(int i=0;i<pts.size();i++)
    bb.extend(pts[i]);
}

#define GEOMLINES_VERSION 1

void GeomLines::io(Piostream& stream)
{

  stream.begin_class("GeomLines", GEOMLINES_VERSION);
  GeomObj::io(stream);
  Pio(stream, pts);
  stream.end_class();
}

void GeomLines::add(const Point& p1, const Point& p2)
{
  pts.add(p1);
  pts.add(p2);
}


Persistent* make_GeomCLines()
{
  return new GeomCLines();
}

PersistentTypeID GeomCLines::type_id("GeomCLines", "GeomObj", make_GeomCLines);

GeomCLines::GeomCLines()
  : line_width_(1.0)
{
}

GeomCLines::GeomCLines(const GeomCLines& copy)
  : line_width_(copy.line_width_),
    points_(copy.points_),
    colors_(copy.colors_)
{
}

GeomCLines::~GeomCLines()
{
}

GeomObj* GeomCLines::clone()
{
  return new GeomCLines(*this);
}

void GeomCLines::get_bounds(BBox& bb)
{
  for(unsigned int i=0;i<points_.size();i+=3)
  {
    bb.extend(Point(points_[i+0], points_[i+1], points_[i+2]));
  }
}

#define GEOMLINES_VERSION 1

void GeomCLines::io(Piostream& stream)
{

  stream.begin_class("GeomCLines", GEOMLINES_VERSION);
  GeomObj::io(stream);
  Pio(stream, line_width_);
  Pio(stream, points_);
  Pio(stream, colors_);
  stream.end_class();
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
GeomCLines::add(const Point& p1, MaterialHandle c1,
		const Point& p2, MaterialHandle c2)
{
  points_.push_back(p1.x());
  points_.push_back(p1.y());
  points_.push_back(p1.z());
  points_.push_back(p2.x());
  points_.push_back(p2.y());
  points_.push_back(p2.z());

  const unsigned char r0 = COLOR_FTOB(c1->diffuse.r());
  const unsigned char g0 = COLOR_FTOB(c1->diffuse.g());
  const unsigned char b0 = COLOR_FTOB(c1->diffuse.b());
  const unsigned char a0 = COLOR_FTOB(c1->transparency);

  colors_.push_back(r0);
  colors_.push_back(g0);
  colors_.push_back(b0);
  colors_.push_back(a0);

  const unsigned char r1 = COLOR_FTOB(c2->diffuse.r());
  const unsigned char g1 = COLOR_FTOB(c2->diffuse.g());
  const unsigned char b1 = COLOR_FTOB(c2->diffuse.b());
  const unsigned char a1 = COLOR_FTOB(c2->transparency);

  colors_.push_back(r1);
  colors_.push_back(g1);
  colors_.push_back(b1);
  colors_.push_back(a1);
}



Persistent* make_GeomTranspLines()
{
  return new GeomTranspLines();
}

PersistentTypeID GeomTranspLines::type_id("GeomTranspLines", "GeomCLines",
					  make_GeomTranspLines);

GeomTranspLines::GeomTranspLines()
  : GeomCLines(),
    xreverse_(false),
    yreverse_(false),
    zreverse_(false)
{
}

GeomTranspLines::GeomTranspLines(const GeomTranspLines& copy)
  : GeomCLines(copy),
    xindices_(copy.xindices_),
    yindices_(copy.yindices_),
    zindices_(copy.zindices_),
    xreverse_(copy.xreverse_),
    yreverse_(copy.yreverse_),
    zreverse_(copy.zreverse_)
{
}

GeomTranspLines::~GeomTranspLines()
{
}

GeomObj* GeomTranspLines::clone()
{
  return new GeomTranspLines(*this);
}

#define GEOMLINES_VERSION 1

void GeomTranspLines::io(Piostream& stream)
{
  stream.begin_class("GeomTranspLines", GEOMLINES_VERSION);
  GeomCLines::io(stream);
  stream.end_class();
}

static bool
pair_less(const pair<float, unsigned int> &a,
	  const pair<float, unsigned int> &b)
{
  return a.first < b.first;
}
 

void
GeomTranspLines::sort()
{
  const unsigned int vsize = points_.size() / 6;
  if (xindices_.size() == vsize*2) return;
  
  xreverse_ = false;
  yreverse_ = false;
  zreverse_ = false;

  vector<pair<float, unsigned int> > tmp(vsize);
  unsigned int i;

  for (i = 0; i < vsize;i++)
  {
    tmp[i].first = points_[i*6+0] + points_[i*6+3];
    tmp[i].second = i*6;
  }
  std::sort(tmp.begin(), tmp.end(), pair_less);

  xindices_.resize(vsize*2);
  for (i=0; i < vsize; i++)
  {
    xindices_[i*2+0] = tmp[i].second / 3;
    xindices_[i*2+1] = tmp[i].second / 3 + 1;
  }

  for (i = 0; i < vsize;i++)
  {
    tmp[i].first = points_[i*6+1] + points_[i*6+4];
    tmp[i].second = i*6;
  }
  std::sort(tmp.begin(), tmp.end(), pair_less);

  yindices_.resize(vsize*2);
  for (i=0; i < vsize; i++)
  {
    yindices_[i*2+0] = tmp[i].second / 3;
    yindices_[i*2+1] = tmp[i].second / 3 + 1;
  }

  for (i = 0; i < vsize;i++)
  {
    tmp[i].first = points_[i*6+2] + points_[i*6+5];
    tmp[i].second = i*6;
  }
  std::sort(tmp.begin(), tmp.end(), pair_less);

  zindices_.resize(vsize*2);
  for (i=0; i < vsize; i++)
  {
    zindices_[i*2+0] = tmp[i].second / 3;
    zindices_[i*2+1] = tmp[i].second / 3 + 1;
  }
}


// for lit streamlines
Persistent* make_TexGeomLines()
{
  return new TexGeomLines();
}

PersistentTypeID TexGeomLines::type_id("TexGeomLines", "GeomObj", make_TexGeomLines);

TexGeomLines::TexGeomLines()
  : tmapid(0),
    tex_per_seg(1),
    mutex("TexGeomLines mutex"),
    alpha(1.0)
{
}

TexGeomLines::TexGeomLines(const TexGeomLines& copy)
  : mutex("TexGeomLines mutex"), pts(copy.pts)
{
}

TexGeomLines::~TexGeomLines()
{
}

GeomObj* TexGeomLines::clone()
{
  return new TexGeomLines(*this);
}

void TexGeomLines::get_bounds(BBox& bb)
{
  for(int i=0;i<pts.size();i++)
    bb.extend(pts[i]);
}

#define TexGeomLines_VERSION 1

void TexGeomLines::io(Piostream& stream)
{

  stream.begin_class("TexGeomLines", TexGeomLines_VERSION);
  GeomObj::io(stream);
  Pio(stream, pts);
  stream.end_class();
}

// this is used by the hedgehog...

void TexGeomLines::add(const Point& p1, const Point& p2,double scale)
{
  pts.add(p1);
  pts.add(p2);
  
  tangents.add((p2-p1)*scale);
} 

void TexGeomLines::add(const Point& p1, const Vector& dir, const Colorub& c) {
  pts.add(p1);
  pts.add(p1+dir);

  Vector v(dir);
  v.normalize();
  tangents.add(v);
  colors.add(c);
}

// this is used by the streamline module...

void TexGeomLines::batch_add(Array1<double>&, Array1<Point>& ps)
{
  tex_per_seg = 0;  // this is not the hedgehog...
  int pstart = pts.size();
  int tstart = tangents.size();

  pts.grow(2*(ps.size()-1));
  tangents.grow(2*(ps.size()-1));  // ignore times for now...

  int i;
  for(i=0;i<ps.size()-1;i++) {// forward differences to get tangents...
    Vector v = ps[i+1]-ps[i];
    v.normalize();

    tangents[tstart++] = v; // vector is set...
    pts[pstart++] = ps[i];
    if (i) { // only store it once...
      tangents[tstart++] = v; // duplicate it otherwise
      pts[pstart++] = ps[i];
    }
  }
  tangents[tstart] = tangents[tstart-1]; // duplicate last guy...
  pts[pstart] = ps[i]; // last point...
}
void TexGeomLines::batch_add(Array1<double>&, Array1<Point>& ps,
			     Array1<Color>& cs)
{
  tex_per_seg = 0;  // this is not the hedgehog...
  int pstart = pts.size();
  int tstart = tangents.size();
  int cstart = colors.size();

  //  cerr << "Adding with colors...\n";

  pts.grow(2*(ps.size()-1));
  tangents.grow(2*(ps.size()-1));
  colors.grow(2*(ps.size()-1));

  int i;
  for(i=0;i<ps.size()-1;i++) {// forward differences to get tangents...
    Vector v = ps[i+1]-ps[i];
    v.normalize();

    tangents[tstart++] = v; // vector is set...
    pts[pstart++] = ps[i];
    colors[cstart++] = Colorub(cs[i]);
    if (i) { // only store it once...
      tangents[tstart++] = v; // duplicate it otherwise
      pts[pstart++] = ps[i];
      colors[cstart++] = Colorub(cs[i]);
    }
  }
  tangents[tstart] = tangents[tstart-1]; // duplicate last guy...
  pts[pstart] = ps[i]; // last point...
  colors[cstart] = Colorub(cs[i]);
}



// this code sorts in three axis...

struct SortHelper {
  static Point* pts_array;
  int                  id; // id for this guy...
};

Point* SortHelper::pts_array=0;

int CompX(const void* e1, const void* e2)
{
  SortHelper *a = (SortHelper*)e1;
  SortHelper *b = (SortHelper*)e2;

  if (SortHelper::pts_array[a->id].x() >
      SortHelper::pts_array[b->id].x())
    return 1;
  if (SortHelper::pts_array[a->id].x() <
      SortHelper::pts_array[b->id].x())
    return -1;

  return 0; // they are equal...
}

int CompY(const void* e1, const void* e2)
{
  SortHelper *a = (SortHelper*)e1;
  SortHelper *b = (SortHelper*)e2;

  if (SortHelper::pts_array[a->id].y() >
      SortHelper::pts_array[b->id].y())
    return 1;
  if (SortHelper::pts_array[a->id].y() <
      SortHelper::pts_array[b->id].y())
    return -1;

  return 0; // they are equal...
}

int CompZ(const void* e1, const void* e2)
{
  SortHelper *a = (SortHelper*)e1;
  SortHelper *b = (SortHelper*)e2;

  if (SortHelper::pts_array[a->id].z() >
      SortHelper::pts_array[b->id].z())
    return 1;
  if (SortHelper::pts_array[a->id].z() <
      SortHelper::pts_array[b->id].z())
    return -1;

  return 0; // they are equal...
}

void TexGeomLines::SortVecs()
{
  SortHelper::pts_array = &pts[0];

  
  Array1<SortHelper> help; // list for help stuff...

  int realsize = pts.size()/2;
  int imul=2;

  sorted.resize(3*realsize); // resize the array...

  help.resize(realsize);

  int i;
  for(i=0;i<realsize;i++) {
    help[i].id = imul*i;  // start it in order...
  }

  cerr << "Doing first Sort!\n";

  qsort(&help[0],help.size(),sizeof(SortHelper),CompX);
  //	int (*) (const void*,const void*)CompX);

  // now dump these ids..

  for(i=0;i<realsize;i++) {
    sorted[i] = help[i].id;
    help[i].id = imul*i;  // reset for next list...
  }
  cerr << "Doing 2nd Sort!\n";
  
  qsort(&help[0],help.size(),sizeof(SortHelper),CompZ);

  int j;
  for(j=0;j<realsize;j++,i++) {
    sorted[i] = help[j].id;
    help[j].id=imul*j;
  }

  cerr << "Doing 3rd Sort!\n";
  qsort(&help[0],help.size(),sizeof(SortHelper),CompY);

  for(j=0;j<realsize;j++,i++) {
    sorted[i] = help[j].id;
  }

  // that should be everything...
}







Persistent* make_GeomCLineStrips()
{
  return new GeomCLineStrips();
}

PersistentTypeID GeomCLineStrips::type_id("GeomCLineStrips", "GeomObj", make_GeomCLineStrips);

GeomCLineStrips::GeomCLineStrips()
  : line_width_(1.0)
{
}

GeomCLineStrips::GeomCLineStrips(const GeomCLineStrips& copy)
  : line_width_(copy.line_width_),
    points_(copy.points_),
    colors_(copy.colors_)
{
}

GeomCLineStrips::~GeomCLineStrips()
{
}

GeomObj* GeomCLineStrips::clone()
{
  return new GeomCLineStrips(*this);
}

void GeomCLineStrips::get_bounds(BBox& bb)
{
  const int n_strips = points_.size();
  for(unsigned int s = 0; s < n_strips; s++) {
    const int n_coords = points_[s].size();
    for (unsigned int i = 0; i < n_coords; i+=3) {
      bb.extend(Point(points_[s][i+0], points_[s][i+1], points_[s][i+2]));
    }
  }
}

#define GEOMCLINESTRIPS_VERSION 1

void GeomCLineStrips::io(Piostream& stream)
{

  stream.begin_class("GeomCLineStrips", GEOMCLINESTRIPS_VERSION);
  GeomObj::io(stream);
  Pio(stream, line_width_);
  Pio(stream, points_);
  Pio(stream, colors_);
  stream.end_class();
}

bool GeomCLineStrips::saveobj(ostream&, const string&, GeomSave*)
{
#if 0
  NOT_FINISHED("GeomCLineStrips::saveobj");
  return false;
#else
  return true;
#endif
}


void
GeomCLineStrips::add(const vector<Point> &p, 
		     const vector<MaterialHandle> &c)
{
  points_.push_back(vector<float>());
  colors_.push_back(vector<unsigned char>());

  const int n_points = p.size();
  const int n_colors = c.size();
  ASSERT(n_colors == n_points);

  for (unsigned int i = 0; i < n_points; i++) 
    add(p[i], c[i]);
}

void
GeomCLineStrips::add(const Point &p, 
		     const MaterialHandle c)
{
  if (points_.empty()) points_.push_back(vector<float>());
  if (colors_.empty()) colors_.push_back(vector<unsigned char>());
  
  points_.back().push_back(p.x());
  points_.back().push_back(p.y());
  points_.back().push_back(p.z());

  colors_.back().push_back(COLOR_FTOB(c->diffuse.r()));
  colors_.back().push_back(COLOR_FTOB(c->diffuse.g()));
  colors_.back().push_back(COLOR_FTOB(c->diffuse.b()));
  colors_.back().push_back(COLOR_FTOB(c->transparency));
}

void
GeomCLineStrips::newline()
{
  points_.push_back(vector<float>());
  colors_.push_back(vector<unsigned char>());
}




} // End namespace SCIRun

