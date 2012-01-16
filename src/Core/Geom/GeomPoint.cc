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
 * GeomPoint.cc: Points objects
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Core/Geom/GeomPoint.h>
#include <Core/Containers/Sort.h>
#include <Core/Geometry/BBox.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/NotFinished.h>
#include <Core/Persistent/PersistentSTL.h>
#include <cstdio>
#include <iostream>
#include <algorithm>
#include <cstdio>

using std::cerr;
using std::cout;
using std::endl;
using std::ostream;

namespace SCIRun {

Persistent* make_GeomPoints()
{
  return scinew GeomPoints();
}

PersistentTypeID GeomPoints::type_id("GeomPoints", "GeomObj", make_GeomPoints);


Persistent* make_GeomTranspPoints()
{
  return scinew GeomTranspPoints();
}

PersistentTypeID GeomTranspPoints::type_id("GeomTranspPoints", "GeomPoints", make_GeomTranspPoints);

Persistent* make_GeomTimedParticles()
{
  return scinew GeomTimedParticles(0);
}

PersistentTypeID GeomTimedParticles::type_id("GeomTimedParticles", 
					     "GeomObj", 
					     make_GeomTimedParticles);

GeomPoints::GeomPoints(const GeomPoints &copy)
  : points_(copy.points_), pickable(copy.pickable)
{
}

GeomPoints::GeomPoints()
  : pickable(false)
{
}

GeomPoints::~GeomPoints()
{
}

GeomObj* GeomPoints::clone()
{
  return scinew GeomPoints(*this);
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
GeomPoints::add(const Point &p, const MaterialHandle &m)
{
  add(p);
  
  colors_.push_back(COLOR_FTOB(m->diffuse.r()));
  colors_.push_back(COLOR_FTOB(m->diffuse.g()));
  colors_.push_back(COLOR_FTOB(m->diffuse.b()));
  colors_.push_back(COLOR_FTOB(m->transparency));
}



void
GeomPoints::get_bounds(BBox& bb)
{
  for (unsigned int i=0; i<points_.size(); i+=3)
  {
    bb.extend(Point(points_[i], points_[i+1], points_[i+2]));
  }
}

#define GEOMPOINTS_VERSION 3


void
GeomPoints::io(Piostream& stream)
{
  int version=stream.begin_class("GeomPoints", GEOMPOINTS_VERSION);
  GeomObj::io(stream);
  Pio(stream, points_);
  Pio(stream, colors_);
  Pio(stream, indices_);
  if (version == 2)
  {
    int have_normal;
    Vector n;
    Pio(stream, have_normal);
    Pio(stream, n);
  }
  stream.end_class();
}


GeomTranspPoints::GeomTranspPoints()
  : GeomPoints(),
    xreverse_(false),
    yreverse_(false),
    zreverse_(false)
{
}


GeomTranspPoints::GeomTranspPoints(const GeomTranspPoints &copy)
  : GeomPoints(copy),
    xindices_(copy.xindices_),
    yindices_(copy.yindices_),
    zindices_(copy.zindices_),
    xreverse_(copy.xreverse_),
    yreverse_(copy.yreverse_),
    zreverse_(copy.zreverse_)
{
}


GeomTranspPoints::~GeomTranspPoints()
{
}


GeomObj*
GeomTranspPoints::clone()
{
  return scinew GeomTranspPoints(*this);
}


static bool
pair_less(const pair<float, unsigned int> &a,
	  const pair<float, unsigned int> &b)
{
  return a.first < b.first;
}


void
GeomTranspPoints::sort()
{
  const unsigned int vsize = points_.size() / 3;
  if (xindices_.size() == vsize) return;

  xreverse_ = false;
  yreverse_ = false;
  zreverse_ = false;

  vector<pair<float, unsigned int> > tmp(vsize);
  unsigned int i;

  for (i = 0; i < vsize;i++)
  {
    tmp[i].first = points_[i*3+0];
    tmp[i].second = i;
  }
  std::sort(tmp.begin(), tmp.end(), pair_less);

  xindices_.resize(vsize);
  for (i=0; i < vsize; i++)
  {
    xindices_[i] = tmp[i].second;
  }

  for (i = 0; i < vsize;i++)
  {
    tmp[i].first = points_[i*3+1];
    tmp[i].second = i;
  }
  std::sort(tmp.begin(), tmp.end(), pair_less);

  yindices_.resize(vsize);
  for (i=0; i < vsize; i++)
  {
    yindices_[i] = tmp[i].second;
  }

  for (i = 0; i < vsize;i++)
  {
    tmp[i].first = points_[i*3+2];
    tmp[i].second = i;
  }
  std::sort(tmp.begin(), tmp.end(), pair_less);

  zindices_.resize(vsize);
  for (i=0; i < vsize; i++)
  {
    zindices_[i] = tmp[i].second;
  }
}


#define GEOMTRANSPPOINTS_VERSION 1

void
GeomTranspPoints::io(Piostream& stream)
{
  stream.begin_class("GeomTranspPoints", GEOMTRANSPPOINTS_VERSION);
  GeomPoints::io(stream);
  stream.end_class();
}


GeomTimedParticles::GeomTimedParticles(const GeomTimedParticles&)
: cmap(0)
{
  cerr << "No real Copy Constructor...\n";
}



GeomTimedParticles::GeomTimedParticles(char *fname)
:drawMode(0),cmap(0)
{
  if (!fname) {
    cerr << "Woah - file is bogus...\n";
    return;
  }

  FILE *f = fopen(fname,"r");

  if (!f) {
    cerr << fname << " Couldn't open the file\n";
    return;
  }

  char buf[4096];
  int nparticles;
  fscanf(f,"%d", &nparticles);

  particles.setsize(nparticles);
  Array1<InstTimedParticle> ps;

  int interval=0;

  // keep a histogram of the "time" values - used later for resampling

  const int HSIZE=4096;

  TimeHist.setsize(HSIZE);
  for(int index=0;index<HSIZE;index++) {
    TimeHist[index] = 0;
  }

  while(fgets(buf,4096,f)) {
    int dummy;
    sscanf(buf,"%d",&dummy);
//    cout << "buf = " << buf << endl;
    if (buf[0] != '0' && dummy != 0) { // starting a number...
      int cur_part;
      sscanf(buf,"%d",&cur_part);
      if( cur_part == 0 )
	cout << "cur_part == 0 at buf = " << buf << endl;
//      cout << "cur_part = " << cur_part << "from scanf\n";
      cur_part -= 1; // get it into our indexing scheme...
      
      ++interval;
      if (interval > particles.size()*0.05) {
	cerr << " 5%\n";
	interval = 0;
      }

      InstTimedParticle curp;

      int done=0;
      while(!done && fgets(buf,4096,f)) {
	if (4 == sscanf(buf,"%f %f %f %f",&curp.t,&curp.x,&curp.y,&curp.z)) {
	  ps.add(curp);
	  int hval = (int)(curp.t*HSIZE);
	  if (hval >= HSIZE) hval = HSIZE-1;
	  TimeHist[hval]++;
	} else {
	  done = 1;
	}	
      }

      // now this has to be shoved into the right place...

//      cout << "cur_part = " << cur_part << endl;
      particles[cur_part].nslots = ps.size();
      particles[cur_part].pos = scinew InstTimedParticle[ps.size()];

      for(int i=0;i<ps.size();i++) {
	particles[cur_part].pos[i] = ps[i]; // copy it over...
	curp = ps[i];
      }

      // clear it out for the next guy

      ps.remove_all();

    } else { // do nothing - just eat this line...
      
    }
  }
#if 0
  for(int i=0;i<particles.size();i++) {
    cerr << "\nDoing Particle: " << i << endl;
    for(int j=0;j<particles[i].nslots;j++) {
      cerr << "  " << particles[i].pos[j].x << " ";
      cerr << particles[i].pos[j].y << " ";
      cerr << particles[i].pos[j].z << "\n";
    }
  }
#endif

  // now lets resample this 
  // try unifrom at first...

  resamp_times.resize(50);

  resamp_pts.resize(resamp_times.size());

  cerr << "Doing first remap...\n";

  int i;
  for(i=0;i<resamp_times.size();i++) {
    resamp_times[i] = i/(resamp_times.size()-1.0);
    resamp_pts[i].setsize(particles.size()*3);
  }

  cerr << "Starting resampling...\n";

  for(i=0;i<particles.size();i++) {
    float x,y,z;
    for(int j=0;j<resamp_times.size();j++) {
      if (!particles[i].AtTime(resamp_times[j],x,y,z)) {
	x = y = z = -1500000;
      }
      resamp_pts[j][i*3] = x;
      resamp_pts[j][i*3 + 1] = y;
      resamp_pts[j][i*3 + 2] = z;
    }
  }
  cerr << "Resampling done!\n";
}

GeomTimedParticles::~GeomTimedParticles()
{
  for(int i=0;i<particles.size();i++) {
    delete particles[i].pos;
  }
}


GeomObj* GeomTimedParticles::clone()
{
    return scinew GeomTimedParticles(*this);
}

void GeomTimedParticles::get_bounds(BBox& bb)
{
    for (int i=0; i<particles.size(); i++)
      for(int j=0;j<particles[i].nslots; j++)
	bb.extend(Point(particles[i].pos[j].x,
			particles[i].pos[j].y,
			particles[i].pos[j].z));

}

#define GeomTimedParticles_VERSION 2

void GeomTimedParticles::io(Piostream& stream)
{
    stream.begin_class("GeomTimedParticles", GeomTimedParticles_VERSION);
    GeomObj::io(stream);
    stream.end_class();
}

void GeomTimedParticles::draw(DrawInfoOpenGL*, Material*, double)
{
  NOT_FINISHED("GeomTimedParticles::draw");
}

} // End namespace SCIRun


