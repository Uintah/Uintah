
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

#include <Geom/Geom.h>
#include <Geometry/Point.h>
#include <Geometry/Vector.h>

#include <Multitask/Task.h>
#include <Multitask/ITC.h>

class GeomPts : public GeomObj {
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
    virtual void get_bounds(BSphere&);

#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif
    virtual void make_prims(Array1<GeomObj*>& free,
			    Array1<GeomObj*>& dontfree);
    virtual void preprocess();
    virtual void intersect(const Ray& ray, Material*,
			   Hit& hit);
    virtual Vector normal(const Point& p, const Hit&);

    virtual void io(Piostream&);
    static PersistentTypeID type_id;
    virtual bool saveobj(ostream&, const clString& format, GeomSave*);
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

#include <Geometry/BBox.h>

class GeomTimedParticles : public GeomObj {
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
  virtual void get_bounds(BSphere&);
  
#ifdef SCI_OPENGL
  virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif
  virtual void make_prims(Array1<GeomObj*>& free,
			  Array1<GeomObj*>& dontfree);
  virtual void preprocess();
  virtual void intersect(const Ray& ray, Material*,
			 Hit& hit);
  virtual Vector normal(const Point& p, const Hit&);

  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  virtual bool saveobj(ostream&, const clString& format, GeomSave*);
};


#endif /* SCI_Geom_Point_h */
