
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/BBox.h>
#include <Packages/rtrt/Core/Stats.h>
#include <Packages/rtrt/Core/Glyph.h>

using namespace rtrt;
using namespace SCIRun;

namespace rtrt {
  float glyph_threshold = 0.9;
}

Persistent* glyph_maker() {
  return new Glyph();
}

// initialize the static member type_id
PersistentTypeID Glyph::type_id("Glyph", "Object", glyph_maker);

const int GLYPH_VERSION = 1;

void 
Glyph::io(SCIRun::Piostream &str)
{
  str.begin_class("Glyph", GLYPH_VERSION);
  Object::io(str);
  Pio(str, value);
  Pio(str, child);
  str.end_class();
}

Glyph::Glyph(Object *obj, const float value):
  Object(0), child(obj), value(value)
{}

Glyph::~Glyph() {}

void Glyph::intersect(Ray& ray, HitInfo& hit, DepthStats* st,
	       PerProcessorContext* ppc) {
  if (value > glyph_threshold)
    child->intersect(ray, hit, st, ppc);
}
  
void Glyph::light_intersect(Ray& ray, HitInfo& hit, Color& atten,
		     DepthStats* st, PerProcessorContext* ppc) {
  if (value > glyph_threshold)
    child->light_intersect(ray, hit, atten, st, ppc);
}

void Glyph::softshadow_intersect(Light* light, Ray& ray,
			  HitInfo& hit, double dist, Color& atten,
			  DepthStats* st, PerProcessorContext* ppc) {
  if (value > glyph_threshold)
    child->softshadow_intersect(light, ray, hit, dist, atten, st, ppc);
}

void Glyph::multi_light_intersect(Light* light, const Point& orig,
			   const Array1<Vector>& dirs,
			   const Array1<Color>& attens,
			   double dist,
			   DepthStats* st, PerProcessorContext* ppc) {
  if (value > glyph_threshold)
    child->multi_light_intersect(light, orig, dirs, attens, dist, st, ppc);
}

Vector Glyph::normal(const Point& p, const HitInfo& hit) {
  return child->normal(p, hit);
}

void Glyph::compute_bounds(BBox& b, double offset) {
  child->compute_bounds(b, offset);
}

void Glyph::print(ostream& out) {
  out << value;
  child->print(out);
}

void Glyph::animate(double t, bool& changed) {
  child->animate(t, changed);
}

void Glyph::preprocess(double maxradius, int& pp_offset, int& scratchsize) {
  child->preprocess(maxradius, pp_offset, scratchsize);
}
