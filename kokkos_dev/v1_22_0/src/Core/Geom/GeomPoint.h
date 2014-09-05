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
 * Pt.h: GeomPoint objects
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

#include <Core/Geom/GeomObj.h>
#include <Core/Datatypes/Color.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/BBox.h>
#include <Core/Geom/Material.h>

namespace SCIRun {


class SCICORESHARE GeomPoints : public GeomObj {
protected:
  vector<float> points_;
  vector<unsigned char> colors_;
  vector<float> indices_;

  bool pickable;  // hack so we don't draw non-pickable pts during a pick

public:

  GeomPoints();
  GeomPoints(const GeomPoints&);
  virtual ~GeomPoints();
  virtual GeomObj* clone();

  virtual void get_bounds(BBox&);

  inline void add(const Point& p) {
    points_.push_back(p.x());
    points_.push_back(p.y());
    points_.push_back(p.z());
  }

  void add(const Point& p, const MaterialHandle &c);
  void add(const Point& p, double index)
  {
    add(p);
    indices_.push_back(index);
  }

#ifdef SCI_OPENGL
  virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif

  virtual void io(Piostream&);
  static PersistentTypeID type_id;
};


class SCICORESHARE GeomTranspPoints : public GeomPoints {
protected:
  vector<unsigned int> xindices_;
  vector<unsigned int> yindices_;
  vector<unsigned int> zindices_;
  bool xreverse_;
  bool yreverse_;
  bool zreverse_;

  void sort();

public:
  GeomTranspPoints();
  GeomTranspPoints(const GeomTranspPoints&);
  virtual ~GeomTranspPoints();
  virtual GeomObj* clone();

#ifdef SCI_OPENGL
  virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif

  virtual void io(Piostream&);
  static PersistentTypeID type_id;
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

#include <Core/Geometry/BBox.h>

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
};

} // End namespace SCIRun


#endif /* SCI_Geom_Point_h */
