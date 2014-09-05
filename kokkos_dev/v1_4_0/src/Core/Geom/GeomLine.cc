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
#include <iostream>
using std::cerr;
using std::ostream;

#include <stdlib.h>

namespace SCIRun {


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

bool GeomLine::saveobj(ostream&, const string&, GeomSave*)
{
#if 0
    NOT_FINISHED("GeomLine::saveobj");
    return false;
#else
	return true;
#endif
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

bool GeomLines::saveobj(ostream&, const string&, GeomSave*)
{
#if 0
    NOT_FINISHED("GeomLines::saveobj");
    return false;
#else
    return true;
#endif
}

void GeomLines::add(const Point& p1, const Point& p2)
{
  pts.add(p1);
  pts.add(p2);
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

bool TexGeomLines::saveobj(ostream&, const string&, GeomSave*)
{
    NOT_FINISHED("TexGeomLines::saveobj");
    return false;
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

} // End namespace SCIRun

