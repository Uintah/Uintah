/*
  Has a child whose intersection is computed when the value associated
  with the object has a higher value than the global threshold.

  Author: James Bigler (bigler@cs.utah.edu)
  Date:   July 16, 2002

*/

#ifndef GLYPH_H
#define GLYPH_H 1

#include <Packages/rtrt/Core/Object.h>
#include <Core/Geometry/Point.h>
#include <Packages/rtrt/Core/Array1.h>
#include <stdlib.h>

namespace rtrt {
  class Glyph;
  class GlyphGroup;
  class Grid;
}

namespace SCIRun {
  void Pio(Piostream&, rtrt::Glyph*&);
  void Pio(Piostream&, rtrt::GlyphGroup*&);
}

// There is a global variable called float glyph_threshold;
// This variable must be [0, 1]

namespace rtrt {

class Glyph : public Object {
protected:
  Object *child;
  float value;
public:
  Glyph(Object *obj, const float value);
  virtual ~Glyph();

  Glyph() : Object(0) {} // for Pio.
  
  //! Persistent I/O.
  static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  friend void SCIRun::Pio(SCIRun::Piostream&, Glyph*&);
  
  virtual void intersect(Ray& ray, HitInfo& hit, DepthStats* st,
			 PerProcessorContext* cx);
  virtual void light_intersect(Ray& ray, HitInfo& hit, Color& atten,
			       DepthStats* st, PerProcessorContext* ppc);
  virtual void softshadow_intersect(Light* light, Ray& ray,
				    HitInfo& hit, double dist, Color& atten,
				    DepthStats* st, PerProcessorContext* ppc);
  virtual void multi_light_intersect(Light* light, const Point& orig,
				     const Array1<Vector>& dirs,
				     const Array1<Color>& attens,
				     double dist,
				     DepthStats* st, PerProcessorContext* ppc);

  virtual Vector normal(const Point&, const HitInfo& hit);
  virtual void animate(double t, bool& changed);
  virtual void preprocess(double maxradius, int& pp_offset, int& scratchsize);
  virtual void compute_bounds(BBox&, double offset);
  virtual void print(ostream& out);

  float get_value() const { return value; }
};

  // This class creates num_levels Grids that contain all the glyphs.
  // Each level has all the glyps that would satisfy the glyph intersection.
  // For example, if you had 10 levels, level 0 would contain all the glyphs.
  // Level 1 would contain all the glyps that had values > 0.1.  Level 5
  // would contain all the glyps that had values > 0.5 and so on.
class GlyphGroup : public Object {
protected:
  Array1<Grid*> grids;
  int num_levels;
public:
  // This takes all the glyps and builds the Grids.
  GlyphGroup(Array1<Glyph *> &glyphs, int gridcellsize = 3,
	     int num_levels = 10);
  virtual ~GlyphGroup();

  GlyphGroup() : Object(0) {} // for Pio.

  inline int get_index(const float val) const;
  
  //! Persistent I/O.
  static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  friend void SCIRun::Pio(SCIRun::Piostream&, GlyphGroup*&);
  
  virtual void intersect(Ray& ray, HitInfo& hit, DepthStats* st,
			 PerProcessorContext* cx);
  virtual void light_intersect(Ray& ray, HitInfo& hit, Color& atten,
			       DepthStats* st, PerProcessorContext* ppc);
  virtual void softshadow_intersect(Light* light, Ray& ray,
				    HitInfo& hit, double dist, Color& atten,
				    DepthStats* st, PerProcessorContext* ppc);
  virtual void multi_light_intersect(Light* light, const Point& orig,
				     const Array1<Vector>& dirs,
				     const Array1<Color>& attens,
				     double dist,
				     DepthStats* st, PerProcessorContext* ppc);

  virtual Vector normal(const Point&, const HitInfo& hit);
  virtual void preprocess(double maxradius, int& pp_offset, int& scratchsize);
  virtual void compute_bounds(BBox&, double offset);
  virtual void print(ostream& out);
};
  
} // end namespace rtrt

#endif
