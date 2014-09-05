
/*
 * Pt.h: Pts objects
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   Feb 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#ifndef SCI_Geom_Point_h
#define SCI_Geom_Point_h 1

#include <SCICore/Geom/GeomObj.h>
#include <SCICore/Geom/Color.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Geometry/BBox.h>

namespace SCICore {
namespace GeomSpace {

using SCICore::Geometry::BBox;

class SCICORESHARE GeomPts : public GeomObj {
public:
    Array1<float> pts;
    inline void add(const Point& p) {
	int s=pts.size();
	pts.grow(3);
 	pts[s]=p.x();
	pts[s+1]=p.y();
	pts[s+2]=p.z();
    }
    inline void add(const Point& p, const float &v) {
	int s=pts.size();
	pts.grow(3);
 	pts[s]=p.x();
	pts[s+1]=p.y();
	pts[s+2]=p.z();
	
	scalars.add(v); // use this as well...
    }

    inline void add(const Point& p, const Color &v) {
	int s=pts.size();
	pts.grow(3);
 	pts[s]=p.x();
	pts[s+1]=p.y();
	pts[s+2]=p.z();
	
	int ss=colors.size();
	colors.grow(3);
	colors[ss]=v.r();
	colors[ss+1]=v.g();
	colors[ss+2]=v.b();
    }

    inline void add(const Point& p, const float &sv, const Vector& v) {
	int s=pts.size();
	pts.grow(3);
 	pts[s]=p.x();
	pts[s+1]=p.y();
	pts[s+2]=p.z();
	
	scalars.add(sv); // use this as well...

	normals.add(v.x());
	normals.add(v.y());
	normals.add(v.z());
    }

    int have_normal;
    Vector n;
    int pickable;	// hack so we don't draw non-pickable pts during a pick

    unsigned char* cmap; // color map - if you have scalar values...

    Array1<float>  scalars;  // change to something else???
    Array1<float>  normals;  // ditto?
    Array1<float> colors;

    int list_pos; // posistion in the list...

    Array1<unsigned int> sortx; // sorted arrays...
    Array1<unsigned int> sorty;
    Array1<unsigned int> sortz;

    void DoSort(); // sorts the arrays...

    GeomPts(const GeomPts&);
    GeomPts(int size);
    GeomPts(int size, const Vector &);
    virtual ~GeomPts();
    virtual GeomObj* clone();
    virtual void get_bounds(BBox&);

#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif

    virtual void io(Piostream&);
    static PersistentTypeID type_id;
    virtual bool saveobj(std::ostream&, const clString& format, GeomSave*);
};

// special stuff for particles...

struct InstTimedParticle {
  float x,y,z;  // where the particle is in space
  float t;      // time for this particular particle...
};

struct TimedParticle {
  InstTimedParticle *pos; // array of timed particles
  int nslots;             // number of elements in this array

  inline int AtTime(double t, float &x, float& y, float& z) {
    
    int index=1;

    while((index < nslots) && !((pos[index-1].t <= t) &&
	  (pos[index].t > t))) {
      index++; // find interval...
    }
    
    // see if we are in range...

    if ((pos[index-1].t <= t) &&
	  (pos[index].t > t)) {
      // precompute 1/interval...
      float delta = (t-pos[index-1].t)/(pos[index].t - pos[index-1].t);

      float w1 = 1.0-delta;

      x = pos[index-1].x*w1 + pos[index].x*delta;
      y = pos[index-1].y*w1 + pos[index].y*delta;
      z = pos[index-1].z*w1 + pos[index].z*delta;

      return 1; // found a valid particle...
    } else {
      return 0;
    }
    //    return 0; // nothing was found...
  };

};

// we will defenitly need some form of hiarchy, to do
// spatial partioning with - probably also have a "time" hiearchy
// this should make it easy to parallelize things...

#include <SCICore/Geometry/BBox.h>

class SCICORESHARE GeomTimedParticles : public GeomObj {
  Array1< TimedParticle > particles; // actual particles

  int drawMode;                      // 0 - pts at t0
                                     // 1 - pts at correct time
                                     // 2 - lines with t as texture
                                     // 3 - resampled points 
                                     // 4 - above - interpolate in time
                                     // 5 - all - wi texture - time is%

  BBox dataBox; // bounding box for the particles

  unsigned char* cmap; // color map 

  // this is a "cheap" hack, resample to 50 time steps - uniform in time
  
  Array1<int> TimeHist;

  Array1<float> resamp_times;

  Array1< Array1<float> > resamp_pts;
public:
  GeomTimedParticles(const GeomTimedParticles&);
  GeomTimedParticles(char *file); // initialize with a file

  void SetDrawMode(int dmode) { drawMode = dmode; };

  void BumpMode() { drawMode++; if (drawMode > 5) drawMode = 0; };

  void AddCmap(unsigned char* cm) { cmap = cm; };

  virtual ~GeomTimedParticles();
  virtual GeomObj* clone();
  virtual void get_bounds(BBox&);
  
#ifdef SCI_OPENGL
  virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif

  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  virtual bool saveobj(std::ostream&, const clString& format, GeomSave*);
};

} // End namespace GeomSpace
} // End namespace SCICore

//
// $Log$
// Revision 1.5  1999/10/07 02:07:49  sparker
// use standard iostreams and complex type
//
// Revision 1.4  1999/08/28 17:54:44  sparker
// Integrated new Thread library
//
// Revision 1.3  1999/08/17 23:50:32  sparker
// Removed all traces of the old Raytracer and X11 renderers.
// Also removed a .o and .d file
//
// Revision 1.2  1999/08/17 06:39:22  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:51  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:56:13  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:05:11  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:19  dav
// Import sources
//
//

#endif /* SCI_Geom_Point_h */
