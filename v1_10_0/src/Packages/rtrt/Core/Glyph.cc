
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/BBox.h>
#include <Packages/rtrt/Core/Stats.h>
#include <Packages/rtrt/Core/Glyph.h>
#include <Packages/rtrt/Core/Grid.h>
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/Array1.h>

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
  //child->animate(t, changed);
}

void Glyph::preprocess(double maxradius, int& pp_offset, int& scratchsize) {
  child->preprocess(maxradius, pp_offset, scratchsize);
}


//////////////////////////////////////////////////////////////////////////////
//
// GlyphGroup
//
//

// This is an ugly macro designed to be used like an inline function.
// The reason that this was done, was that Dav de St. Germain told me not
// to put an externed global variable in the header.  Whatever!

int GlyphGroup::get_index(const float val) const {
  int idx=(int)(val*num_levels);
  if (idx>=num_levels) return idx-1;
  else if (idx<0) return 0;
  else return idx;
}

GlyphGroup::GlyphGroup(Array1<Glyph *> &glyphs, int gridcellsize,
		       int num_levels):
  Object(0), grids(num_levels+1), num_levels(num_levels)
{
  // The size of grids should be num_levels + 1, so that when we do indexing
  // we don't have to check the edge condition.
  // We must be able to take values that range [0 to 1].  If we compute the
  // index by (value * num_levels) the range [0, 1) would work ok.  The
  // problem arises for the case value == 1.  The index would then equal
  // num_levels.  Rather than checking the bounds we will make this edge
  // cast point to the last index in the array.  So grids[num_levels] will
  // point to grids[num_levels-1].

  // Allocate memory for all the arrays for the grids.
  Group *groups = new Group[num_levels];
  //  Array1<Group> groups(num_levels);
  // Now we need to assign the glyps to the groups
  for(int i = 0; i < glyphs.size(); i++) {
    // This represents the last group where the glyph will appear.
    int index = get_index(glyphs[i]->get_value());
    // Now we need to start with 0 and go up to the index.
    for(int j = 0; j <= index; j++)
      groups[j].add((Object*)glyphs[i]);
  }

  // Now we need to create the grids.
  for(int g = 0; g < num_levels; g++) {
    grids[g] = new Grid((Object*)&(groups[g]), gridcellsize);
  }
  // Set the last grid pointer to the second to the one.
  grids[num_levels] = grids[num_levels-1];
}

GlyphGroup::~GlyphGroup() {
  // Delete all the Grids.
  // As of this writing Grids didn't actually delete any memroy when the
  // destructor was called.
  for(int i =0; i < num_levels; i++)
    delete grids[i];
}

#define GLYPHGROUP_VERSION 1

void GlyphGroup::io(SCIRun::Piostream &stream) {
  stream.begin_class("GlyphGroup", GLYPHGROUP_VERSION);
  Object::io(stream);
  Pio(stream, num_levels);
  Pio(stream, grids);
  stream.end_class();
}
  
void GlyphGroup::intersect(Ray& ray, HitInfo& hit, DepthStats* st,
			   PerProcessorContext* cx) {
  grids[get_index(glyph_threshold)]->intersect(ray, hit, st, cx);
}

void GlyphGroup::light_intersect(Ray& ray, HitInfo& hit, Color& atten,
				 DepthStats* st, PerProcessorContext* ppc) {
  grids[get_index(glyph_threshold)]->
    light_intersect(ray, hit, atten, st, ppc);
}

void GlyphGroup::softshadow_intersect(Light* light, Ray& ray,
				      HitInfo& hit, double dist, Color& atten,
				      DepthStats* st,
				      PerProcessorContext* ppc) {
  grids[get_index(glyph_threshold)]->
    softshadow_intersect(light, ray, hit, dist, atten, st, ppc);
}

void GlyphGroup::multi_light_intersect(Light* light, const Point& orig,
				       const Array1<Vector>& dirs,
				       const Array1<Color>& attens,
				       double dist, DepthStats* st,
				       PerProcessorContext* ppc) {
  grids[get_index(glyph_threshold)]->
    multi_light_intersect(light, orig, dirs, attens, dist, st, ppc);
}

Vector GlyphGroup::normal(const Point&, const HitInfo&) {
  cerr << "Error: GlyphGroup normal should not be called!\n";
  return Vector(0,0,0);
}

void GlyphGroup::preprocess(double maxradius, int& pp_offset,
			    int& scratchsize) {
  for(int i =0; i < num_levels; i++)
    grids[i]->preprocess(maxradius, pp_offset, scratchsize);
}

void GlyphGroup::compute_bounds(BBox& b, double offset) {
  for(int i =0; i < num_levels; i++)
    grids[i]->compute_bounds(b, offset);
//  cerr << "MY BOUNDING BOX IS: " << b.min() <<" -- "<< b.max() <<"\n";
}

void GlyphGroup::print(ostream& out) {
  out << "GlyphGoup:start\n";
  out << "num_levels = "<<num_levels<<'\n';
  for(int i =0; i < num_levels; i++)
    grids[i]->print(out);
  out << "GlyphGoup:end\n";
}
