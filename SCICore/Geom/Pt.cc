//static char *id="@(#) $Id$";

/*
 * Pt.cc: Pts objects
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <SCICore/Geom/Pt.h>
#include <SCICore/Containers/String.h>
#include <SCICore/Containers/Sort.h>
#include <SCICore/Geometry/BBox.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/Util/NotFinished.h>

namespace SCICore {
namespace GeomSpace {

Persistent* make_GeomPts()
{
    return scinew GeomPts(0);
}

PersistentTypeID GeomPts::type_id("GeomPts", "GeomObj", make_GeomPts);


Persistent* make_GeomTimedParticles()
{
    return scinew GeomTimedParticles(0);
}

PersistentTypeID GeomTimedParticles::type_id("GeomTimedParticles", 
					     "GeomObj", 
					     make_GeomTimedParticles);

GeomPts::GeomPts(const GeomPts &copy)
: pts(copy.pts), have_normal(copy.have_normal), n(copy.n), pickable(copy.pickable), cmap(0) {
}

GeomPts::GeomPts(int size)
: pts(0, size*3), have_normal(0), pickable(0), cmap(0)
{
}

GeomPts::GeomPts(int size, const Vector &n)
: pts(0, size*3), n(n), have_normal(1), pickable(0)
{
}

void GeomPts::DoSort()
{
  using SCICore::Containers::SortObjs;

  SortObjs sorter;
  // first have to make arrays and stuff

  Array1<unsigned int> remaped;

  double minx,maxx;
  double miny,maxy;
  double minz,maxz;

  minx = maxx = pts[0];
  miny = maxy = pts[1];
  minz = maxz = pts[2];

  int i;
  for(i=3;i<pts.size();i+=3) {
    if (pts[i + 0] < minx) minx = pts[i + 0];
    if (pts[i + 1] < miny) miny = pts[i + 1];
    if (pts[i + 2] < minz) minz = pts[i + 2];

    if (pts[i + 0] > maxx) maxx = pts[i + 0];
    if (pts[i + 1] > maxy) maxy = pts[i + 1];
    if (pts[i + 2] > maxz) maxz = pts[i + 2];
  }

  cerr << "Getting into sort...\n";

  // now remap into the remaped array...

  remaped.setsize(pts.size()/3);

  const unsigned int BIGNUM= (1<<24);

  float mapx = BIGNUM/(maxx-minx);
  float mapy = BIGNUM/(maxy-miny);
  float mapz = BIGNUM/(maxz-minz);

  for(i=0;i<remaped.size();i++) {
    remaped[i] = (pts[i*3 + 0]-minx)*mapx;
  }
  
  cerr << "Did first remap...\n";

  sortx.setsize(remaped.size());

  sorter.DoRadixSort(remaped,sortx);
  cerr << "Done!\n";

  for(i=0;i<remaped.size();i++) {
    remaped[i] = (pts[i*3 + 1]-miny)*mapy;
  }
  sorty.setsize(remaped.size());
  sorter.DoRadixSort(remaped,sorty);

  for(i=0;i<remaped.size();i++) {
    remaped[i] = (pts[i*3 + 2]-minz)*mapz;
  }
  sortz.setsize(remaped.size());
  sorter.DoRadixSort(remaped,sortz);

  cerr << "Done remaping!\n";

}

GeomPts::~GeomPts()
{
}

GeomObj* GeomPts::clone()
{
    return scinew GeomPts(*this);
}

void GeomPts::get_bounds(BBox& bb)
{
    for (int i=0; i<pts.size(); i+=3)
	bb.extend(Point(pts[i], pts[i+1], pts[i+2]));
}

#define GEOMPTS_VERSION 2

void GeomPts::io(Piostream& stream)
{
    using SCICore::PersistentSpace::Pio;
    using SCICore::Containers::Pio;
    using SCICore::Geometry::Pio;

    int version=stream.begin_class("GeomPts", GEOMPTS_VERSION);
    GeomObj::io(stream);
    Pio(stream, pts);
    if (version > 1) {
	Pio(stream, have_normal);
	Pio(stream, n);
    }
    stream.end_class();
}

bool GeomPts::saveobj(ostream&, const clString&, GeomSave*)
{
    NOT_FINISHED("GeomPts::saveobj");
    return false;
}

GeomTimedParticles::GeomTimedParticles(const GeomTimedParticles&)
: cmap(0)
{
  cerr << "No real Copy Constructor...\n";
}

#include <stdio.h>

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
	  int hval = curp.t*HSIZE;
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

bool GeomTimedParticles::saveobj(ostream&, const clString&, GeomSave*)
{
    NOT_FINISHED("GeomTimedParticles::saveobj");
    return false;
}

void GeomTimedParticles::draw(DrawInfoOpenGL*, Material*, double)
{
  NOT_FINISHED("GeomTimedParticles::draw");
}

} // End namespace Geometry
} // End namespace SCICore

//
// $Log$
// Revision 1.3  1999/08/17 23:50:32  sparker
// Removed all traces of the old Raytracer and X11 renderers.
// Also removed a .o and .d file
//
// Revision 1.2  1999/08/17 06:39:21  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:51  mcq
// Initial commit
//
// Revision 1.2  1999/07/07 21:10:56  dav
// added beginnings of support for g++ compilation
//
// Revision 1.1.1.1  1999/04/24 23:12:19  dav
// Import sources
//
//

